import xarray as xr

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