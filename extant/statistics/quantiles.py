"""Quantiles of every series at once, ignoring NaN: over any dims (``nan_quantile``, ``pooled_quantiles``,
``quantile_range``) and in a window centred on each year (``centred_rolling_quantiles``, numba).

Functions here run on dask workers, so send the package to the cluster
(``jasmin.upload_package(client)``) after editing it.

The numba kernels are compiled twice: parallel over grid points for data in
memory, and serial for dask blocks. Dask already runs the blocks in parallel,
and numba's default threading layer is not safe to call from several threads
at once (nesting would also oversubscribe the workers' cores).
"""

import numpy as np
import xarray as xr
from numba import njit, prange

from ..utils.datatree import skip_empty


# ---------------------------------------------------------------------------
# Quantiles over several dims at once, ignoring NaN
# ---------------------------------------------------------------------------

def _sorted_quantiles(values, q, n_dims=1):
    """Linear-interpolation quantiles over the last ``n_dims`` axes, ignoring NaN, for every series at once."""
    values = values.reshape(*values.shape[:values.ndim - n_dims], -1)
    ordered = np.sort(values, axis=-1)  # NaN sorts to the end
    n = np.sum(~np.isnan(ordered), axis=-1, keepdims=True)
    position = (n - 1) * q
    lower = np.clip(np.floor(position), 0, None).astype(int)
    upper = np.clip(lower + 1, None, np.maximum(n - 1, 0)).astype(int)
    fraction = position - np.floor(position)
    low_values = np.take_along_axis(ordered, lower, axis=-1)
    high_values = np.take_along_axis(ordered, upper, axis=-1)
    #(c): numpy's own interpolation (np.lib._function_base_impl._lerp), so the results match it bit for bit
    difference = high_values - low_values
    interpolated = np.where(fraction >= 0.5, high_values - difference * (1 - fraction),
                            low_values + difference * fraction)
    return np.where(n > 0, interpolated, np.nan)


def nan_quantile(da, q, dim):
    """Quantiles over ``dim`` ignoring NaN: the same numbers as ``da.quantile(q, dim)``, much faster on maps.

    numpy's nanquantile, which xarray uses for float data, loops over every
    series in Python, which takes minutes on a full grid. Sorting once and
    interpolating between order statistics gives identical results (numpy's
    default 'linear' method) in one vectorised step.

    Example: [3.0, NaN, 1.0, 2.0] sorts to [1.0, 2.0, 3.0, NaN], 3 of them
    valid, so q = 0.25 is at position (3 - 1) * 0.25 = 0.5, halfway from the
    1st value to the 2nd: 1.5, as ``np.nanquantile`` gives.

    Args:
        da (xr.DataArray): Data.
        q (float | Sequence[float]): Quantile level(s).
        dim (str | Sequence[str]): Dimension(s) to reduce; several are pooled.

    Returns:
        xr.DataArray: With a ``quantile`` dim (last) if ``q`` is a sequence, without one if it is a scalar.
    """
    levels = np.atleast_1d(np.asarray(q, dtype=float))
    dims = [dim] if isinstance(dim, str) else list(dim)
    if da.chunks:
        da = da.chunk({d: -1 for d in dims})
    result = xr.apply_ufunc(
        _sorted_quantiles, da,
        input_core_dims=[dims], output_core_dims=[["quantile"]],
        kwargs={"q": levels, "n_dims": len(dims)}, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"quantile": levels.size}},
    ).assign_coords(quantile=levels)
    return result.isel(quantile=0, drop=True) if np.ndim(q) == 0 else result


def pooled_quantiles(ds, quantiles, dims=("member", "year")):
    """Quantiles of every variable over ``dims`` pooled, with ``nan_quantile``.

    The same numbers as ``ds.quantile(quantiles, dim=dims)``, whose NaN-skipping
    path loops over every series in Python (minutes on a full grid).
    """
    return ds.map(nan_quantile, q=quantiles, dim=[d for d in dims if d in ds.dims])


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
def centred_rolling_quantiles(da, window, quantiles=(0.05, 0.5, 0.95)):
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
        window (int): Window length in years (odd), e.g. ``config.WINDOW``.
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
