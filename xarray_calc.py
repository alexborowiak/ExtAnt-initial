import xarray as xr


def split_time(da, parts=('year', 'season'), dim='time'):
    """Split a time axis into separate dims for each datetime component.
    Takes a DataArray with a datetime ``dim`` and unstacks it so that each
    entry in ``parts`` becomes an independent dim. This lets a rolling
    window move N units within a single group rather than across all of
    them, e.g. ``parts=('year', 'season')`` on quarterly data gives a
    window of 20 years in one season rather than 5 years of all four.
    Component values come from the coordinate labels, so DJF is indexed by
    its January year under ``QE-DEC`` and by its December year under
    ``QS-DEC``. Combinations absent from the record are padded with NaN.
    Parameters
    ----------
    da : xr.DataArray
        Data with a datetime ``dim``.
    parts : sequence of str
        Names of ``.dt`` accessors to split on, e.g. ``('year', 'season')``
        or ``('year', 'month')``. Must be jointly unique over ``dim``.
        Sets the dim order of the output.
    dim : str, default 'time'
        Name of the time dim to replace.
    Returns
    -------
    xr.DataArray
        Same data with ``dim`` replaced by one dim per entry in ``parts``.
        Each dim is sorted by its own values, so string-valued components
        such as ``season`` come out alphabetically, not chronologically.
    """
    parts = tuple(parts)
    return (
        da.assign_coords({p: getattr(da[dim].dt, p) for p in parts})
        .set_index({dim: parts})
        .unstack(dim)
    )

def dayofyear_climatology(da: xr.DataArray, window_size: int) -> xr.DataArray:
    """Day-of-year climatology smoothed with a centred window.

    Assumes every day-of-year contributes the same number of samples.

    Args:
        da (xr.DataArray): Input array with a `time` coordinate on a 360-day calendar.
        window_size (int): Number of days in the centred window.

    Returns:
        xr.DataArray: Climatology indexed by `dayofyear`, with `time` reduced.
    """
    clim = da.groupby('time.dayofyear').mean('time')
    half = window_size // 2

    # Wrap padding supplies the neighbours rolling would treat as out-of-bounds,
    # so its edge NaNs fall entirely within the padding and are trimmed away.
    padded = clim.pad(dayofyear=(half, half), mode='wrap')
    smoothed = padded.rolling(dayofyear=window_size, center=True).mean()

    # pad fills the dayofyear index with NaN, promoting it to float.
    return (
        smoothed
        .isel(dayofyear=slice(half, half + clim.sizes['dayofyear']))
        .assign_coords(dayofyear=clim.dayofyear)
    )


def dayofyear_quantile(da: xr.DataArray, window_size: int, q: float) -> xr.DataArray:
    """Day-of-year quantile climatology over a centred window.

    Args:
        da (xr.DataArray): Input array with a `time` coordinate on a 360-day calendar.
        window_size (int): Number of days in the centred window.
        q (float): Quantile in [0, 1].

    Returns:
        xr.DataArray: Quantiles indexed by `dayofyear`, with `time` reduced.
    """
    # construct exposes the window as a dimension rather than reducing it, so the
    # quantile sees every raw sample in the window instead of a daily summary.
    windowed = da.rolling(time=window_size, center=True).construct('window')

    window_q = windowed.groupby('time.dayofyear').quantile(q, dim=['time', 'window'])

    return window_q