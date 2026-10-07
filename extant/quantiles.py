"""Rolling quantiles, smoothing and quantile ranges for seasonal LESFMIP analysis.

Functions here run on dask workers, so send the package to the cluster
(``cluster.upload_package(client)``) after editing it.

The numba kernels are compiled twice: parallel over grid points for data in
memory, and serial for dask blocks. Dask already runs the blocks in parallel,
and numba's default threading layer is not safe to call from several threads
at once (nesting would also oversubscribe the workers' cores).
"""

import numpy as np
import xarray as xr
from numba import njit, prange
from scipy import sparse

from .config import WINDOW
from .datatree import skip_empty
from .stats import nan_quantile


def _centred_sums(x, window):
    """Sums over centred ``window``-long windows along the last axis, shortened at the ends."""
    half = window // 2
    n = x.shape[-1]
    cumulative = np.concatenate([np.zeros_like(x[..., :1]), np.cumsum(x, axis=-1)], axis=-1)
    position = np.arange(n)
    return cumulative[..., np.minimum(position + half + 1, n)] - cumulative[..., np.maximum(position - half, 0)]


def _rolling_std_block(x, window):
    """Standard deviation (ddof=0) of every member's values in each centred window; x is (..., member, year)."""
    valid = np.isfinite(x)
    x = np.where(valid, x, 0.0).astype(np.float64)
    count = _centred_sums(valid.sum(axis=-2).astype(np.float64), window)
    mean = _centred_sums(x.sum(axis=-2), window)
    square = _centred_sums((x * x).sum(axis=-2), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean /= count
        variance = np.maximum(square / count - mean * mean, 0.0)
    return np.where(count > 0, np.sqrt(variance), np.nan)


@skip_empty
def rolling_std(da, window=WINDOW, dim="year", member_dim="member"):
    """Standard deviation of every member's values in a centred ``window``-year window, for each year.

    The same numbers as ``da.rolling(year=window, center=True).construct('window').std(['window', 'member'])``
    (ddof=0; NaN skipped; windows shortened at the ends of the record), but
    from running sums of x and x², so no copy of the data per window: the
    construct version holds ``window`` copies of the data in memory at once.

    Args:
        da (xr.DataArray | xr.Dataset): With ``member_dim`` and ``dim``.
        window (int): Window length in years.
        dim, member_dim (str): The rolling and the pooled dimension.

    Returns:
        The same type, without ``member_dim``.
    """
    if da.chunks:
        da = da.chunk({member_dim: -1, dim: -1})
    out = xr.apply_ufunc(
        _rolling_std_block, da,
        input_core_dims=[[member_dim, dim]], output_core_dims=[[dim]],
        kwargs={"window": window}, dask="parallelized", output_dtypes=[np.float64],
    )
    return out.transpose(dim, ...)


def pooled_quantiles(ds, quantiles, dims=("member", "year")):
    """Quantiles of every variable over ``dims`` pooled, with ``stats.nan_quantile``.

    The same numbers as ``ds.quantile(quantiles, dim=dims)``, whose NaN-skipping
    path loops over every series in Python (minutes on a full grid).
    """
    return ds.map(nan_quantile, q=quantiles, dim=[d for d in dims if d in ds.dims])


@skip_empty
def rolling_quantile(da, rolling_dim, window, quantiles, extra_dims=None):
    """Rolling quantiles via xarray's construct, keeping the shortened windows at either end (slow, memory-heavy).

    Notebook 05's plumes use it to reach the ends of the record. The analysis
    uses ``centred_rolling_quantiles``, where those years are NaN.
    """
    dims = ["window", *(extra_dims or [])]
    return da.rolling({rolling_dim: window}, center=True).construct("window").quantile(quantiles, dim=dims)


# ---------------------------------------------------------------------------
# Centred rolling quantiles (numba)
# ---------------------------------------------------------------------------
# Year t's window is the ``window`` years centred on it: years t - half to
# t + half, with half = window // 2 (1950 to 1970 for t = 1960 and a 21-year
# window). Its quantiles pool every member's values in those years. Years
# within half a window of either end of the record have no complete window,
# and are NaN. How the windows are moved along the record, rather than each
# sorted from scratch, is in ``centred_rolling_quantiles``.

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
def _sorted_without_nan(values):
    """The values that are not NaN, sorted."""
    return np.sort(values[~np.isnan(values)])


@njit
def _swap_years(window_values, leaving_year_values, entering_year_values):
    """Move a sorted window on by one year: drop the leaving year's values and merge in the entering year's.

    One pass along the three sorted arrays at once, like the merge step of a
    merge sort. Each step looks at the smallest value not yet handled, which
    is one of:

    - the next leaving value, still in ``window_values``: dropped;
    - the next entering value, if it is smaller than the next window value: written out;
    - otherwise the next window value, which stays in the window: written out.

    Args:
        window_values (np.ndarray): The old window's values, sorted.
        leaving_year_values (np.ndarray): The leaving year's values, sorted; each one is in ``window_values``.
        entering_year_values (np.ndarray): The entering year's values, sorted.

    Returns:
        np.ndarray: The new window's values, sorted.
    """
    out = np.empty(window_values.size - leaving_year_values.size + entering_year_values.size)
    i = 0    # the next value of window_values
    j = 0    # the next value of leaving_year_values
    k = 0    # the next value of entering_year_values
    n = 0    # the next free place in out
    while i < window_values.size:
        if j < leaving_year_values.size and window_values[i] == leaving_year_values[j]:
            #(c): This value is the leaving year's: drop it
            i += 1
            j += 1
        elif k < entering_year_values.size and entering_year_values[k] < window_values[i]:
            #(c): An entering value comes before the next value that stays: write it first
            out[n] = entering_year_values[k]
            k += 1
            n += 1
        else:
            #(c): A value that stays in the window
            out[n] = window_values[i]
            i += 1
            n += 1
    #(c): Entering values larger than every value that stayed go at the end
    out[n:] = entering_year_values[k:]
    return out


@njit
def _centred_rolling_quantiles_at_point(values, window, quantiles, out):
    """Centred rolling quantiles of one point's ``values`` (year, member), written into ``out`` (quantile, year).

    The first complete window is sorted from scratch; every later window is the
    one before it moved on by a year (see ``centred_rolling_quantiles``).
    """
    n_years = values.shape[0]
    half = window // 2
    for t in range(half, n_years - half):
        first_year, last_year = t - half, t + half    # the window centred on year t
        if t == half:
            #(c): The first complete window (years 0 to window - 1): sort all of its values
            window_values = _sorted_without_nan(values[first_year:last_year + 1].ravel())
        else:
            #(c): The window centred on t - 1 moves on a year: its first year (first_year - 1) leaves,
            #(c): and this window's last year enters
            leaving_year_values = _sorted_without_nan(values[first_year - 1])
            entering_year_values = _sorted_without_nan(values[last_year])
            window_values = _swap_years(window_values, leaving_year_values, entering_year_values)
        if window_values.size == 0:
            continue
        for i in range(quantiles.size):
            out[i, t] = interpolate_sorted(window_values, window_values.size, quantiles[i])


def _centred_rolling_quantiles(x, window, quantiles, out):
    """``_centred_rolling_quantiles_at_point`` at every point of ``x`` (point, year, member)."""
    for p in prange(x.shape[0]):
        _centred_rolling_quantiles_at_point(x[p], window, quantiles, out[p])


_centred_rolling_quantiles_serial = njit(nogil=True)(_centred_rolling_quantiles)
_centred_rolling_quantiles_parallel = njit(parallel=True)(_centred_rolling_quantiles)


def _centred_rolling_quantiles_block(x, window, quantiles, parallel):
    """The rolling quantiles of a block whose last two axes are (year, member); returns (..., quantile, year)."""
    *lead, n_years, n_members = x.shape
    points = np.ascontiguousarray(x.reshape(-1, n_years, n_members), dtype=float)
    out = np.full((points.shape[0], quantiles.size, n_years), np.nan)
    kernel = _centred_rolling_quantiles_parallel if parallel else _centred_rolling_quantiles_serial
    kernel(points, window, quantiles, out)
    return out.reshape(*lead, quantiles.size, n_years)


@skip_empty
def centred_rolling_quantiles(da, window=WINDOW, quantiles=(0.05, 0.5, 0.95)):
    """Quantiles of every member's values in the ``window`` years centred on each year.

    Year t pools years t - half to t + half of every member, with
    half = window // 2 (1950 to 1970 for t = 1960 and a 21-year window), NaN
    skipped: the same numbers as ``np.quantile`` of those values. Years within
    ``half`` of either end of the record have no complete window, and are NaN.

    How: neighbouring windows share all but one year. Moving from the window
    centred on 1959 to the one centred on 1960, 1949 leaves (the old window's
    first year) and 1970 enters (the new window's last year); every year in
    between is in both:

        year           1949  1950  ...  1969  1970
        window 1959    [===================]
        window 1960          [===================]
                       leaves                 enters

    So at each grid point the first complete window is sorted once, and then
    kept sorted as it moves along the record: each year, the leaving year's
    values are taken out and the entering year's merged in (``_swap_years``).
    That is one pass over the window's ~1000 values (21 years x ~50 members)
    instead of sorting them again, about 5x faster, with identical results.

    Args:
        da (xr.DataArray | xr.Dataset): With ``member`` and ``year`` dims.
        window (int): Window length in years (odd).
        quantiles (Sequence[float]): Quantile levels.

    Returns:
        The same type, with ``member`` replaced by ``quantile``.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    if da.chunks:
        da = da.chunk({"year": -1, "member": -1})
    out = xr.apply_ufunc(
        _centred_rolling_quantiles_block, da,
        input_core_dims=[["year", "member"]], output_core_dims=[["quantile", "year"]],
        kwargs={"window": window, "quantiles": quantiles, "parallel": not da.chunks},
        dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"quantile": quantiles.size, "year": da.sizes["year"]}},
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
