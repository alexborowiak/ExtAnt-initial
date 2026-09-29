"""Quantile, smoothing and aggregation helpers for seasonal LESFMIP analysis.

Functions here run on dask workers, so upload this module to the cluster
(``client.upload_file``) after editing it.
"""

import numpy as np
import xarray as xr
from numba import njit
from scipy import sparse
from statsmodels.nonparametric.smoothers_lowess import lowess

from xarray_datatree_utils import reduce_to_dataset, skip_empty


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@skip_empty
def seasonal_mean(ds):
    return ds.resample(time='QS-DEC').mean()


@skip_empty
def space_mean(ds):
    return ds.mean(dim=['lat', 'lon'])


@skip_empty
def rolling_std(da, window=11, dim=['window', 'member']):
    return da.rolling(year=window, center=True).construct('window').std(dim=dim)


@skip_empty
def rolling_quantile(da, rolling_dim, window, quantiles, extra_dims=None):
    """Rolling quantiles via xarray's construct (slow, memory-heavy; see rolling_percentile_xr)."""
    dims = ["window", *(extra_dims or [])]
    return da.rolling({rolling_dim: window}, center=True).construct("window").quantile(quantiles, dim=dims)


# ---------------------------------------------------------------------------
# Rolling percentiles (numba)
# ---------------------------------------------------------------------------

@njit
def rolling_percentile_numba(x, window, quantiles):
    """Centred rolling quantiles pooling a (year, member) array over the window and members."""
    nyear, nmember = x.shape
    nquantile = len(quantiles)
    half = window // 2

    out = np.full((nquantile, nyear), np.nan)

    for t in range(half, nyear - half):
        values = np.empty(window * nmember)
        n = 0

        for y in range(t - half, t + half + 1):
            for m in range(nmember):
                v = x[y, m]

                if not np.isnan(v):
                    values[n] = v
                    n += 1

        if n > 0:
            out[:, t] = np.quantile(values[:n], quantiles)

    return out


@skip_empty
def rolling_percentile_xr(
    da,
    rolling_dim="year",
    window=11,
    quantiles=(0.1, 0.5, 0.9),
    extra_dims=("member",),
):
    """Centred rolling quantiles along ``rolling_dim``, pooling over ``extra_dims``.

    Incomplete windows at either end are NaN.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    core_dims = [rolling_dim, *extra_dims]

    out = xr.apply_ufunc(
        rolling_percentile_numba,
        da,
        input_core_dims=[core_dims],
        output_core_dims=[["quantile", rolling_dim]],
        kwargs={
            "window": window,
            "quantiles": quantiles,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {
                "quantile": len(quantiles),
                rolling_dim: da.sizes[rolling_dim],
            },
            "allow_rechunk": True,
        },
    )

    return out.assign_coords(quantile=quantiles)


# ---------------------------------------------------------------------------
# Quantile ranges (upper minus lower quantile)
# ---------------------------------------------------------------------------

def quantile_range(da, quantiles=(0.05, 0.95), dims=("member", "year")):
    """Return the pooled upper-minus-lower quantile range over ``dims``.

    Dims missing from ``da`` are ignored.
    """
    dims = tuple(d for d in dims if d in da.dims)

    if da.chunks is not None:
        da = da.chunk({d: -1 for d in dims})

    q_da = da.quantile(quantiles, dim=dims, skipna=True)

    return (
        q_da.sel(quantile=quantiles[1], drop=True)
        - q_da.sel(quantile=quantiles[0], drop=True)
    )


def rolling_quantile_range(da, window=11, quantiles=(0.05, 0.95), rolling_dim="year", member_dim="member"):
    """Return the rolling pooled upper-minus-lower quantile range."""
    q_da = rolling_percentile_xr(
        da,
        rolling_dim=rolling_dim,
        window=window,
        quantiles=quantiles,
        extra_dims=(member_dim,),
    )

    return (
        q_da.sel(quantile=quantiles[1], drop=True)
        - q_da.sel(quantile=quantiles[0], drop=True)
    )


def quantile_response(tree, quantiles, years=11, reference="hist-nat", variable="tas"):
    """Experiment quantiles over the final ``years`` minus reference quantiles over its full record.

    Quantiles pool members and years. The tree is laid out as /<model>/<experiment>;
    the result is a DataArray with ``model``, ``experiment`` (reference excluded)
    and ``quantile`` dims.
    """
    def pooled(ds):
        return ds.chunk({"member": -1, "year": -1}).quantile(quantiles, dim=("member", "year"), skipna=True)

    exp_q = reduce_to_dataset(tree.isel(year=slice(-years, None)), pooled)
    ref_q = reduce_to_dataset(tree.match(f"*/{reference}"), pooled).sel(experiment=reference, drop=True)
    return (exp_q - ref_q).drop_sel(experiment=reference)[variable]


# ---------------------------------------------------------------------------
# LOWESS smoothing
# ---------------------------------------------------------------------------

def _lowess_1d(y, x, frac, it, delta):
    """Smooth one series with statsmodels, passing all-NaN input straight through."""
    if not np.isfinite(y).any():
        return np.full_like(y, np.nan, dtype=float)
    return lowess(y, x, frac=frac, it=it, delta=delta, return_sorted=False)


def lowess_xarray(da, core_dims="year", window=81, it=3, delta=0.0):
    """Apply statsmodels LOWESS (robust, slow) along one dimension of a DataArray.

    Parameters
    ----------
    da : DataArray to smooth
    core_dims : str, dimension to smooth along
    window : int, number of points in each local window
    it : int, number of bisquare reweighting passes; 0 disables robustness
    delta : float, interpolation distance in coordinate units; 0 fits every point
    """
    x = np.arange(da.sizes[core_dims])
    frac = np.clip(window / da.sizes[core_dims], 0.0, 1.0)
    dims = da.dims
    if da.chunks is not None:
        da = da.chunk({core_dims: -1})
    return xr.apply_ufunc(
        _lowess_1d,
        da,
        input_core_dims=[[core_dims]],
        output_core_dims=[[core_dims]],
        kwargs={"x": x, "frac": frac, "it": it, "delta": delta},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).transpose(*dims)


def lowess_matrix(n, frac):
    """Sparse weight matrices for local linear regression with tricube weights."""
    r = max(int(np.ceil(frac * n)), 2)
    x = np.arange(n, dtype=float)
    rows, cols, w_vals, wu_vals, wu2_vals = [], [], [], [], []

    for i in range(n):
        lo = min(max(i - r // 2, 0), n - r)
        idx = np.arange(lo, lo + r)
        u = x[idx] - x[i]

        d = np.abs(u).max()
        w = np.ones_like(u) if d == 0 else (1 - np.abs(u / d) ** 3) ** 3

        rows.append(np.full(r, i))
        cols.append(idx)
        w_vals.append(w)
        wu_vals.append(w * u)
        wu2_vals.append(w * u**2)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    W = sparse.csr_matrix((np.concatenate(w_vals), (rows, cols)), shape=(n, n))
    WU = sparse.csr_matrix((np.concatenate(wu_vals), (rows, cols)), shape=(n, n))
    WU2 = sparse.csr_matrix((np.concatenate(wu2_vals), (rows, cols)), shape=(n, n))

    return W, WU, WU2


@skip_empty
def lowess_matrix_xarray(da, core_dims='year', window: int = 81):
    """Non-robust LOWESS as a sparse matrix product (fast); NaNs are dropped from each fit."""
    frac = window / da.sizes[core_dims]
    W, WU, WU2 = lowess_matrix(da.sizes[core_dims], frac=frac)

    def smooth(y):
        shape = y.shape
        y = y.reshape(-1, shape[-1])

        valid = np.isfinite(y)
        y0 = np.where(valid, y, 0.0)
        m = valid.astype(float)

        s0 = (W @ m.T).T
        s1 = (WU @ m.T).T
        s2 = (WU2 @ m.T).T

        wy = (W @ y0.T).T
        wuy = (WU @ y0.T).T

        denom = s0 * s2 - s1**2

        out = np.full_like(y0, np.nan, dtype=float)
        np.divide(
            s2 * wy - s1 * wuy,
            denom,
            out=out,
            where=denom != 0,
        )

        out[~valid] = np.nan

        return out.reshape(shape)

    return xr.apply_ufunc(
        smooth,
        da,
        input_core_dims=[[core_dims]],
        output_core_dims=[[core_dims]],
        dask="parallelized",
        output_dtypes=[float],
    )
