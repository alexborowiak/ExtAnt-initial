import numpy as np
import xarray as xr
from numba import njit
from .datatree import skip_empty
from scipy import sparse


SAMPLE_DIM='time'

# def gradientS


def linear_fit(da, dim=SAMPLE_DIM, x=None):
    """Least-squares straight line through each series, ignoring missing samples.

    Args:
        da (xr.DataArray): Data to fit.
        dim (str | Sequence[str]): Dimension(s) holding the samples, e.g. "year",
            or ("year", "season") for one line through all four seasons.
        x (xr.DataArray | None): The regressor. Defaults to the "year"
            coordinate (or the coordinate of the first of ``dim``).

    Returns:
        tuple[xr.DataArray, xr.DataArray]: ``(slope, fitted)``, the slope per unit
        of ``x`` with ``dim`` reduced, and the fitted line on ``da``'s shape.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    if x is None:
        x = da[SAMPLE_DIM if SAMPLE_DIM in dims else dims[0]]

    #(c): Mask x wherever y is missing, so both means use the same samples
    x = x.astype(float).broadcast_like(da).where(da.notnull())
    dx = x - x.mean(dims)
    dy = da - da.mean(dims)
    slope = (dx * dy).sum(dims) / (dx**2).sum(dims)
    fit_da = da.mean(dims) + slope * dx
    fit_da = xr.merge([fit_da.to_dataset(name='fit'), slope.to_dataset(name='gradient')])
    return fit_da

# def trend(da, dim=SAMPLE_DIM):
#     """Least-squares trend, per decade."""
#     slope, _ = linear_fit(da, dim)
#     return 10 * slope



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