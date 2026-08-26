import numpy as np
import xarray as xr
import cftime


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



def join_doy(da, day_dim='dayofyear', year_dim='year', dim='time'):
    """Collapse split year/day dims into a 360-day calendar time axis.
    Takes a DataArray with separate ``year`` and ``dayofyear`` dims on a
    ``360_day`` calendar and stacks them into a single datetime dim of
    ``cftime.Datetime360Day`` objects, with month and day derived from the
    day-of-year by integer division (12 months of 30 days). Assumes the day
    axis is complete and runs 1 to 360; any NaN padding is carried through
    rather than dropped.
    Parameters
    ----------
    da : xr.DataArray
        Data with ``year_dim`` and ``day_dim`` dims.
    day_dim, year_dim : str
        Names of the dims to collapse.
    dim : str, default 'time'
        Name of the datetime dim to create.
    Returns
    -------
    xr.DataArray
        Same data with the two dims replaced by a chronological ``dim``.
    """
    out = da.stack({dim: (year_dim, day_dim)})
    yr = out[year_dim].values
    doy = out[day_dim].values
    dates = [
        cftime.Datetime360Day(y, (d - 1) // 30 + 1, (d - 1) % 30 + 1, 12)
        for y, d in zip(yr, doy)
    ]
    return out.reset_index(dim, drop=True).assign_coords({dim: dates})
    

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

def dayofyear_windowed_quantile(da, q=(0.9, 0.96, 0.99), window=21, sigma=None,
                                pool_dims=('year', 'member', 'window')):
    """
    Computes day-of-year percentiles by pooling samples from a centred circular moving window,
    optionally weighting samples by their distance from the target day of year.

    The time dimension is split into year and dayofyear internally.

    Parameters
    ----------
    da : xr.DataArray
        The input data array, with a time dimension and all of pool_dims except year and window.
    q : sequence of float, default=(0.9, 0.96, 0.99)
        The quantiles to compute.
    window : int, default=21
        The width in days of the centred moving window. Should be odd.
    sigma : float, optional
        The standard deviation in days of a Gaussian weighting across the window, so days
        nearer the target day of year count for more. If None, the window is unweighted.
    pool_dims : sequence of str, default=('year', 'member', 'window')
        The dimensions pooled over when taking the quantiles. Must include 'window'.

    Returns
    -------
    xr.DataArray
        The quantiles, with dimensions quantile and dayofyear, and pool_dims reduced.
    """
    half = window // 2
    pool_chunks = {d: -1 for d in pool_dims if d != 'window'}

    split = split_time(da, ('year', 'dayofyear'), 'time')
    padded = split.chunk({'dayofyear': -1, **pool_chunks}).pad(
        dayofyear=half, mode='wrap', coord_pad_mode='wrap'
    )
    windowed = (
        padded.rolling(dayofyear=window, center=True)
        .construct('window')
        .isel(dayofyear=slice(half, -half))
        .chunk({d: -1 for d in pool_dims})
    )

    if sigma is None:
        return windowed.quantile(q, dim=list(pool_dims)).compute()

    offsets = np.arange(window) - half
    weights = xr.DataArray(np.exp(-0.5 * (offsets / sigma) ** 2), dims='window')
    return windowed.weighted(weights).quantile(q, dim=list(pool_dims)).compute()


def exceedance_flags(da, q_da):
    """Flag days exceeding a day-of-year quantile threshold.
    Selects the threshold matching each timestep's calendar day and
    compares elementwise, returning a boolean on the original time axis
    rather than an aggregate, so the caller chooses how to bin it. Any
    coordinates the threshold selection carries in, such as a scalar
    ``height`` or ``quantile`` and the broadcast ``dayofyear``, are
    dropped, since they are not dims of the result and interfere with
    later stacking. Missing values in ``da`` compare False and so read as
    non-exceedances rather than as gaps; count valid days separately if
    that distinction matters. ``q_da`` must be on the same calendar as
    ``da``, since the match is on day-of-year number alone.
    Parameters
    ----------
    da : xr.DataArray
        Data with a datetime ``time`` dim.
    q_da : xr.DataArray
        Thresholds with a ``dayofyear`` dim covering the calendar.
    Returns
    -------
    xr.DataArray
        Boolean, same shape as ``da``, True where the threshold is
        exceeded.
    """
    q_doy = q_da.sel(dayofyear=da.time.dt.dayofyear)
    exceed = da > q_doy
    exceed = exceed.drop_vars([c for c in exceed.coords if c not in exceed.dims])
    return exceed


def doy_threshold_seasonal_probability(ensemble_ds, q_doy):
    """Compute the seasonal probability of threshold exceedance for one ensemble.
    Counts days exceeding the day-of-year quantile threshold, aggregates
    those counts into DJF/MAM/JJA/SON bins, and divides by the number of
    valid days in each bin to give an exceedance probability rather than a
    raw count. Bins are ``QS-DEC``, so DJF at year Y spans Dec of Y plus
    Jan and Feb of Y+1. Numerator and denominator are resampled with
    identical bin edges, so the two share a time axis exactly and the
    division needs no alignment. The first and last bins of the record are
    partial and their probabilities are estimated from fewer days than the
    rest; mask on ``valid`` if that matters for the downstream statistics.
    Parameters
    ----------
    ensemble_ds : xr.DataArray
        Daily data with a ``time`` dim on a ``360_day`` calendar, matching
        the input to ``exceedance_flags``.
    q_doy : xr.DataArray
        Day-of-year quantile thresholds, passed straight to
        ``exceedance_flags``.
    Returns
    -------
    xr.Dataset
        ``year`` and ``season`` dims, with variables ``exceed`` (days over
        the threshold), ``valid`` (non-NaN days contributing to the bin)
        and ``p`` (the exceedance probability).
    """
    # Exceedance flags on the original time axis.
    exceed_doy = exceedance_flags(ensemble_ds, q_doy)
    # Same binning for both. count() skips NaN, so valid is days with
    # data rather than days in the calendar quarter.
    exceed = exceed_doy.resample(time='QS-DEC').sum()
    valid = ensemble_ds.resample(time='QS-DEC').count()
    # Divide while both are still on the shared time axis, then split
    # once. Splitting first would mean trusting two independently
    # reconstructed (year, season) grids to line up.
    probability = exceed / valid
    exceed = split_time(exceed, ('year', 'season'), 'time')
    valid = split_time(valid, ('year', 'season'), 'time')
    probability = split_time(probability, ('year', 'season'), 'time')
    exceed.name = 'exceed'
    valid.name = 'valid'
    probability.name = 'p'
    merged = xr.merge([exceed, valid, probability])
    return merged

# Calculation below needs full years
# histnat_ensemble_ds = histnat_ensemble_ds.sel(time=slice('1851', '2013'))

def rolling_windows(season_ds, windows=(3, 5, 11, 21), center=True):
    """Pool seasonal exceedance statistics over a range of window lengths.
    Sums ``exceed`` and ``valid`` over centred rolling windows of whole
    years and recomputes ``p`` from the pooled totals, rather than
    averaging the annual probabilities, so years with fewer valid days
    are weighted accordingly. Incomplete windows at the ends come back
    as NaN.
    Parameters
    ----------
    season_ds : xr.Dataset
        Output of ``seasonal_exceedance_probability``.
    windows : sequence of int
        Window lengths in years, stacked along a new ``window`` dim.
    center : bool, default True
        Label each window at its midpoint rather than its right edge.
    Returns
    -------
    xr.Dataset
        Same as ``season_ds`` with an added ``window`` dim.
    """
    to_concat = [season_ds.assign_coords(window=1)]
    for w in windows:
        pooled = season_ds[['exceed', 'valid']].rolling(year=w, center=center).sum()
        pooled['p'] = pooled.exceed / pooled.valid
        pooled = pooled.assign_coords(window=w)
        to_concat.append(pooled)
    out = xr.concat(to_concat, dim='window')
    return out

def subsample_windows(count_ds, window=11, size=10, n_runs=100):
    """Bootstrap the windowed exceedance probability over random member draws.

    Rolls all members once, then sums the drawn subset, exploiting the counts
    being additive. Draws are with replacement. Runs are stacked on a ``sample``
    dim, not ``member``, so the result does not align against a per-member
    counterfactual. Seeded internally, so successive calls give different draws.
    The returned counts are pooled over the drawn members, so they are larger
    than the per-member counts going in and double-count repeated draws.

    Args:
        count_ds (xr.Dataset): Counts with ``exceed`` and ``valid``, on ``year``
            and ``member`` dims. Any other dims broadcast.
        window (int): Window length in years.
        size (int): Members drawn per run.
        n_runs (int): Number of runs.

    Returns:
        xr.Dataset: ``exceed``, ``valid`` and ``p``, with an added ``sample`` dim.
    """
    rng = np.random.default_rng()

    exceed_rolled = count_ds.exceed.rolling(year=window, center=True).sum()
    valid_rolled = count_ds.valid.rolling(year=window, center=True).sum()

    runs = []
    for i in range(n_runs):
        draw = rng.choice(count_ds.sizes['member'], size=size, replace=True)
        exceed = exceed_rolled.isel(member=draw).sum(dim='member')
        valid = valid_rolled.isel(member=draw).sum(dim='member')
        runs.append(
            xr.Dataset({'exceed': exceed, 'valid': valid, 'p': exceed / valid}).assign_coords(sample=i)
        )
    return xr.concat(runs, dim='sample')