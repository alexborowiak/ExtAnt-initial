"""Time axes: months into seasons, time into year and season, and back. For any data with a datetime dim.

    seasonal_mean       monthly -> one value per season, still on time
    split_time          one value per season on time -> year and season
    seasonal_means      monthly -> year and season (the two above, one after the other)
    seasons_in_time     year and season -> one time axis, in decimal years (``split_time`` backwards)
    decimal_year        dates -> decimal years, at the middle of each month
    monthly_time_axis   monthly dates that were shifted -> month starts again
"""

import numpy as np
import xarray as xr

from .datatree import skip_empty


# ---------------------------------------------------------------------------
# Months into seasons, and time into year and season
# ---------------------------------------------------------------------------

@skip_empty
def seasonal_mean(ds, min_months=3, dim="time"):
    """Seasonal (DJF/MAM/JJA/SON) means of complete seasons, in whole years only.

    ``QS-DEC`` bins, so DJF is labelled by the year of its December and a year
    runs from March to the next February. Seasons with fewer than
    ``min_months`` months (the Jan-Feb stub at the start of a record, a season
    cut short at its end) are dropped rather than averaged, and so is any year
    left without all four seasons. Every season then covers the same years, so
    "the final N years" means the same years in each. Which bins to keep is
    read off the time labels alone, so nothing is computed.
    """
    n_months = ds[dim].resample({dim: "QS-DEC"}).count()
    seasonal = ds.resample({dim: "QS-DEC"}).mean().where(n_months >= min_months, drop=True)
    years, n_seasons = np.unique(seasonal[dim].dt.year, return_counts=True)
    return seasonal.isel({dim: np.isin(seasonal[dim].dt.year, years[n_seasons == 4])})


def split_time(da, parts=("year", "season"), dim="time"):
    """Split a time axis into one dim per datetime component, e.g. ``year`` and ``season``.

    A rolling window over ``year`` then moves N years within one season,
    rather than N seasons. Component values come from the time labels, so
    DJF takes the year of its December under ``QS-DEC``. Combinations absent
    from the record are NaN. Each new dim is sorted by its own values, so
    seasons come out alphabetically (DJF, JJA, MAM, SON), not in calendar order.

    Args:
        da (xr.DataArray | xr.Dataset): Data with a datetime ``dim``.
        parts (Sequence[str]): ``.dt`` accessors, jointly unique over ``dim``. Sets the order of the new dims.
        dim (str): The time dim to replace.

    Returns:
        The same data with ``dim`` replaced by one dim per entry in ``parts``.
    """
    parts = tuple(parts)
    return (
        da.assign_coords({p: getattr(da[dim].dt, p) for p in parts})
        .set_index({dim: parts})
        .unstack(dim)
    )


def seasonal_means(monthly, dim="time", min_months=3):
    """Monthly means to seasonal means on ``year`` and ``season``: ``seasonal_mean`` then ``split_time``.

    Refuses a record with a month repeated. ERA5 can stay on its own
    (standard) calendar, because seasons are matched by their (year, season)
    labels, not by date; converting it to the models' 360-day calendar with
    ``align_on='year'`` moves month-start dates into the month before (1 March
    becomes 29 February), which would put the wrong months in each season.

    Args:
        monthly (xr.DataArray | xr.Dataset): Monthly means with a datetime ``dim``.
        dim (str): Name of the time dimension.
        min_months (int): Months a season needs in order to be kept.

    Returns:
        Seasonal means with ``year`` and ``season`` in place of ``dim``.
    """
    month_index = monthly[dim].dt.year * 12 + monthly[dim].dt.month
    if np.unique(month_index).size != month_index.size:
        raise ValueError(
            "some months appear more than once, so seasons would get the wrong months. "
            "convert_calendar('360_day', align_on='year') does this to month-start dates (1 March becomes "
            "29 February); keep ERA5 on its own calendar, or relabel it with monthly_time_axis"
        )
    return split_time(seasonal_mean(monthly, min_months=min_months, dim=dim), ("year", "season"), dim)


# ---------------------------------------------------------------------------
# Back to one time axis
# ---------------------------------------------------------------------------

#(t): Each season's middle month, as a fraction of a year
#(c): DJF is labelled by its December's year, so its middle month (January) is in the next year
SEASON_MIDDLE = xr.DataArray([12.5, 3.5, 6.5, 9.5], coords={'season': ['DJF', 'MAM', 'JJA', 'SON']}) / 12


def seasons_in_time(obj):
    """Put (year, season) on one time axis, in decimal years at each season's middle month.

    year and season stay as coordinates along time, so a seasonal climatology can be
    looked up at every time step with .sel(season=...), like the old day-of-year lookup.

    Args:
        obj (xr.Dataset | xr.DataArray): Data on (year, season, ...).

    Returns:
        The same on (time, ...), time in decimal years, sorted; DJF 1980 is at 1981.04 (January 1981).
    """
    decimal_year_da = obj['year'] + SEASON_MIDDLE.sel(season=obj['season'])
    with_time = obj.assign_coords(time=decimal_year_da)
    stacked = with_time.stack(sample=['year', 'season'])
    stacked = stacked.reset_index('sample')        # year and season become plain coordinates
    stacked = stacked.swap_dims(sample='time')
    return stacked.sortby('time')


def decimal_year(time):
    """Mid-month positions as decimal years, for any calendar: 15 July 1990, and any other day of that July, is 1990.54."""
    return time.dt.year.values + (time.dt.month.values - 0.5) / 12


def monthly_time_axis(monthly, dim="time"):
    """Relabel consecutive monthly means with month-start dates on the standard calendar.

    Repairs a record whose dates were shifted, e.g. by
    ``convert_calendar('360_day', align_on='year')``. Assumes one value per
    month, with no gaps, starting in the month of the first date.
    """
    start = f"{int(monthly[dim].dt.year[0]):04d}-{int(monthly[dim].dt.month[0]):02d}-01"
    return monthly.assign_coords({dim: xr.date_range(start, periods=monthly.sizes[dim], freq="MS")})
