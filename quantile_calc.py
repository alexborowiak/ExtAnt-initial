"""Quantile, smoothing and aggregation helpers for seasonal LESFMIP analysis.

Functions here run on dask workers, so upload this module to the cluster
(``client.upload_file``) after editing it.

The numba kernels are compiled twice: parallel over grid points for data in
memory, and serial for dask blocks. Dask already runs the blocks in parallel,
and numba's default threading layer is not safe to call from several threads
at once (nesting would also oversubscribe the workers' cores).
"""

import numpy as np
import xarray as xr
from numba import njit, prange
from scipy import sparse

from xarray_datatree_utils import reduce_to_dataset, skip_empty
from xarray_stats import nan_quantile


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
def numpy_lerp(low, high, fraction):
    """Linear interpolation exactly as np.quantile does it (its _lerp), so results match numpy bit for bit."""
    difference = high - low
    if fraction >= 0.5:
        return high - difference * (1 - fraction)
    return low + difference * fraction


@njit
def interpolate_sorted(ordered, n, q):
    """Quantile ``q`` of the first ``n`` values of an ascending array, as np.quantile computes it."""
    position = (n - 1) * q
    k = int(np.floor(position))
    if k >= n - 1:
        return ordered[n - 1]
    return numpy_lerp(ordered[k], ordered[k + 1], position - k)


@njit
def _slide(window, n, remove, n_remove, add, n_add, out):
    """One merge pass: ``window[:n]`` minus ``remove`` plus ``add`` (all ascending) into ``out``; returns its length."""
    i = r = a = k = 0
    while i < n:
        value = window[i]
        if r < n_remove and value == remove[r]:
            r += 1
            i += 1
        elif a < n_add and add[a] < value:
            out[k] = add[a]
            a += 1
            k += 1
        else:
            out[k] = value
            i += 1
            k += 1
    while a < n_add:
        out[k] = add[a]
        a += 1
        k += 1
    return k


def _rolling_quantiles(x, window, quantiles, out):
    """Centred rolling quantiles of each point's (year, sample) values, pooled over the window and samples.

    The pooled window is kept sorted: each step removes the year leaving it
    and merges in the year entering it (each year sorted once), instead of
    sorting window x samples values afresh every year. NaN are ignored;
    incomplete windows at either end are NaN.

    Args:
        x (np.ndarray): (point, year, sample).
        out (np.ndarray): (point, quantile, year), filled in place.
    """
    n_points, n_years, n_samples = x.shape
    half = window // 2
    for s in prange(n_points):
        years = np.empty((n_years, n_samples))
        counts = np.zeros(n_years, dtype=np.int64)
        for y in range(n_years):
            for j in range(n_samples):
                value = x[s, y, j]
                if not np.isnan(value):
                    years[y, counts[y]] = value
                    counts[y] += 1
            years[y, :counts[y]] = np.sort(years[y, :counts[y]])

        pool = np.empty(window * n_samples)
        scratch = np.empty(window * n_samples)
        n = 0
        out[s] = np.nan
        for t in range(half, n_years - half):
            if t == half:
                for y in range(window):
                    n = _slide(pool, n, years[0], 0, years[y], counts[y], scratch)
                    pool, scratch = scratch, pool
            else:
                leaving, entering = t - half - 1, t + half
                n = _slide(pool, n, years[leaving], counts[leaving], years[entering], counts[entering], scratch)
                pool, scratch = scratch, pool
            if n > 0:
                for i in range(quantiles.size):
                    out[s, i, t] = interpolate_sorted(pool, n, quantiles[i])


_rolling_quantiles_serial = njit(nogil=True)(_rolling_quantiles)
_rolling_quantiles_parallel = njit(parallel=True)(_rolling_quantiles)


def _rolling_quantiles_block(x, window, quantiles, n_pooled, parallel):
    """``_rolling_quantiles`` on a block whose last axes are (year, *pooled dims)."""
    lead = x.shape[:x.ndim - 1 - n_pooled]
    n_years = x.shape[len(lead)]
    points = np.ascontiguousarray(x.reshape(-1, n_years, int(np.prod(x.shape[len(lead) + 1:]))))
    out = np.empty((points.shape[0], quantiles.size, n_years))
    (_rolling_quantiles_parallel if parallel else _rolling_quantiles_serial)(points, window, quantiles, out)
    return out.reshape(*lead, quantiles.size, n_years)


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
    if da.chunks is not None:
        da = da.chunk({d: -1 for d in core_dims})

    out = xr.apply_ufunc(
        _rolling_quantiles_block,
        da,
        input_core_dims=[core_dims],
        output_core_dims=[["quantile", rolling_dim]],
        kwargs={
            "window": window,
            "quantiles": quantiles,
            "n_pooled": len(extra_dims),
            "parallel": da.chunks is None,
        },
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {
                "quantile": len(quantiles),
                rolling_dim: da.sizes[rolling_dim],
            },
        },
    )

    return out.assign_coords(quantile=quantiles)


# ---------------------------------------------------------------------------
# Quantile ranges (upper minus lower quantile)
# ---------------------------------------------------------------------------

def quantile_range(da, quantiles=(0.05, 0.95), dims=("member", "year")):
    """Return the pooled upper-minus-lower quantile range over ``dims``.

    Dims missing from ``da`` are ignored. NaN are skipped, as ``da.quantile(skipna=True)``
    does, but with the vectorised ``nan_quantile`` rather than numpy's per-series loop.
    """
    dims = tuple(d for d in dims if d in da.dims)
    q_da = nan_quantile(da, list(quantiles), dims)
    return q_da.isel(quantile=1, drop=True) - q_da.isel(quantile=0, drop=True)


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

@njit
def _lowess_pass(x, y, k, robustness, fit, weights):
    """One pass of local linear fits with tricube x weights (times ``robustness``), as statsmodels does it."""
    n = x.size
    left, right = 0, k
    for i in range(n):
        xval = x[i]
        #(c): Slide the k-point neighbourhood until x[i] is at (or just left of) its centre
        while right < n and xval > (x[left] + x[right]) / 2.0:
            left += 1
            right += 1
        radius = max(xval - x[left], x[right - 1] - xval)
        total = 0.0
        nonzero = 0
        for j in range(left, right):
            distance = abs(x[j] - xval) / radius
            distance = 1.0 - distance * distance * distance
            weights[j] = distance * distance * distance * robustness[j]
            total += weights[j]
            nonzero += weights[j] > 1e-12
        if nonzero < 2:
            fit[i] = y[i]
            continue
        mean_x = 0.0
        for j in range(left, right):
            weights[j] /= total
            mean_x += weights[j] * x[j]
        spread = 0.0
        for j in range(left, right):
            spread += weights[j] * (x[j] - mean_x) ** 2
        spread = max(spread, 1e-12)
        value = 0.0
        for j in range(left, right):
            value += weights[j] * (1.0 + (xval - mean_x) * (x[j] - mean_x) / spread) * y[j]
        fit[i] = value


@njit
def _robustness_weights(y, fit, robustness):
    """Bisquare weights of the residuals in units of 6 x their median absolute value, as statsmodels does it."""
    residual = np.abs(y - fit)
    median = np.median(residual)
    for j in range(y.size):
        scaled = (1.0 if residual[j] > 0 else 0.0) if median == 0 else residual[j] / (6.0 * median)
        scaled = min(scaled, 1.0)
        robustness[j] = (1.0 - scaled * scaled) ** 2


def _lowess(y, frac, it, out):
    """statsmodels' LOWESS (delta = 0, x = 0, 1, 2, ...) on every row of ``y``.

    Like statsmodels, NaN are dropped before fitting (the neighbourhoods are
    the k nearest valid points) and are NaN in the output.
    """
    n_points, n_all = y.shape
    for s in prange(n_points):
        x = np.empty(n_all)
        values = np.empty(n_all)
        where = np.empty(n_all, dtype=np.int64)
        n = 0
        for i in range(n_all):
            if not np.isnan(y[s, i]):
                x[n], values[n], where[n] = i, y[s, i], i
                n += 1
        out[s] = np.nan
        if n == 0:
            continue
        x, values = x[:n], values[:n]
        k = min(max(int(frac * n + 1e-10), 2), n)
        robustness = np.ones(n)
        fit = np.empty(n)
        weights = np.empty(n)
        for iteration in range(it + 1):
            _lowess_pass(x, values, k, robustness, fit, weights)
            if iteration < it:
                _robustness_weights(values, fit, robustness)
        for i in range(n):
            out[s, where[i]] = fit[i]


_lowess_serial = njit(nogil=True)(_lowess)
_lowess_parallel = njit(parallel=True)(_lowess)


def _lowess_block(y, frac, it, parallel):
    rows = np.ascontiguousarray(y.reshape(-1, y.shape[-1]), dtype=float)
    out = np.empty_like(rows)
    (_lowess_parallel if parallel else _lowess_serial)(rows, frac, it, out)
    return out.reshape(y.shape)


def lowess_xarray(da, core_dims="year", window=81, it=3):
    """Robust LOWESS along one dimension: statsmodels' algorithm (delta = 0), compiled with numba.

    Gives statsmodels' numbers (to rounding) about 100x faster than calling it
    series by series.

    Parameters
    ----------
    da : DataArray to smooth
    core_dims : str, dimension to smooth along
    window : int, number of points in each local window
    it : int, number of bisquare reweighting passes; 0 disables robustness
    """
    frac = float(np.clip(window / da.sizes[core_dims], 0.0, 1.0))
    dims = da.dims
    if da.chunks is not None:
        da = da.chunk({core_dims: -1})
    return xr.apply_ufunc(
        _lowess_block,
        da,
        input_core_dims=[[core_dims]],
        output_core_dims=[[core_dims]],
        kwargs={"frac": frac, "it": it, "parallel": da.chunks is None},
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
