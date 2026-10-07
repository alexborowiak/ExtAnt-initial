"""From monthly means to the seasonal means every analysis starts from.

LESFMIP and ERA5 go through the same steps, so their seasons line up label for
label:

1. convert to analysis units (``config.VARIABLES``), as float32;
2. cut LESFMIP where historical ends (2014), and ERA5 before 1979;
3. average complete seasons (``seasonal_mean``), dropping incomplete ones and
   any year without all four;
4. split ``time`` into ``year`` and ``season`` (``split_time``);
5. ERA5 only: put it on the LESFMIP grid (``match_grid``).

``seasonal_tree`` and ``seasonal_era5`` run the whole chain; notebook 01 saves
their output, and the sample sections of the other notebooks run it on the
sample in memory.
"""

import numpy as np
import xarray as xr

from . import config
from .datatree import skip_empty

#(t): Chunks for the seasonal means: whole time series (rolling windows and member resampling need them whole)
#(c): and a few latitudes per chunk, so each chunk is a band of the map
SEASONAL_CHUNKS = {"member": -1, "year": -1, "season": -1, "lat": 4, "lon": -1}


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def to_analysis_units(ds, variable):
    """Keep only ``variable``, in analysis units, as float32, without the attributes.

    float32 halves the memory and is ample precision. The scalar ``height``
    coordinate differs between models (2 m, 1.5 m), which stops models
    stacking together, so it goes; so do the attributes, whose units would
    now be wrong and whose comments differ between models.
    """
    var = config.VARIABLES[variable]
    out = var.to_analysis_units(ds[[variable]]).astype("float32")
    return out.drop_vars("height", errors="ignore").drop_attrs()


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


def monthly_time_axis(monthly, dim="time"):
    """Relabel consecutive monthly means with month-start dates on the standard calendar.

    Repairs a record whose dates were shifted, e.g. by
    ``convert_calendar('360_day', align_on='year')``. Assumes one value per
    month, with no gaps, starting in the month of the first date.
    """
    start = f"{int(monthly[dim].dt.year[0]):04d}-{int(monthly[dim].dt.month[0]):02d}-01"
    return monthly.assign_coords({dim: xr.date_range(start, periods=monthly.sizes[dim], freq="MS")})


def match_grid(obs, like, tolerance=0.01, lat="lat", lon="lon"):
    """Put ``obs`` on the latitudes and longitudes of ``like``.

    ERA5's grid need not be the LESFMIP grid: it may use 0-360 longitudes where
    LESFMIP uses -180-180, run north to south, cover a different area, or sit
    on different points. So this:

    1. puts ``obs``'s longitudes in ``like``'s convention and sorts both axes;
    2. if every point of ``like`` has a partner in ``obs`` within ``tolerance``
       degrees, takes those values unchanged, relabelled with ``like``'s exact
       coordinates so arithmetic between the two aligns;
    3. otherwise interpolates bilinearly onto ``like``'s points, wrapping round
       in longitude. Points outside ``obs``'s coverage are NaN.

    Interpolation suits data at a similar resolution, like the ERA5 store
    (conservatively regridded to 2.5° in notebook 01, section 2). Regrid much
    finer data conservatively first.

    Args:
        obs (xr.DataArray): Data to put on the grid.
        like (xr.DataArray): Data on the target grid.
        tolerance (float): Largest coordinate difference, in degrees, still treated as the same point.

    Returns:
        xr.DataArray: ``obs`` on ``like``'s ``lat`` and ``lon``.
    """
    target = {name: np.asarray(like[name].values, dtype=float) for name in (lat, lon)}
    #(c): Longitudes into the target's convention: -180-180 if it has any negative longitude, else 0-360.
    #(c): The range starts `tolerance` early, so -180.000001 stays next to -180 rather than wrapping to +180
    start = (-180.0 if target[lon].min() < 0 else 0.0) - tolerance
    wrapped = (obs[lon] - start) % 360 + start
    obs = obs.assign_coords({lon: wrapped.astype(float), lat: obs[lat].astype(float)}).sortby([lat, lon])

    def gap(name):
        return np.abs(obs[name].values[:, None] - target[name][None, :]).min(0).max()

    if gap(lat) <= tolerance and gap(lon) <= tolerance:
        nearest = {name: obs.indexes[name].get_indexer(target[name], method="nearest") for name in (lat, lon)}
        return obs.isel(nearest).assign_coords(target)

    #(c): One extra column at each end, a full turn away, so interpolation wraps round in longitude
    padded = xr.concat([obs.isel({lon: [-1]}).assign_coords({lon: obs[lon][-1:] - 360}), obs,
                        obs.isel({lon: [0]}).assign_coords({lon: obs[lon][:1] + 360})], dim=lon)
    return padded.interp(target)


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------

def seasonal_tree(tree, variable, last_month=config.LAST_MONTH, chunks=SEASONAL_CHUNKS):
    """Monthly LESFMIP (as opened by ``loading.open_lesfmip_monthly``) to seasonal means in analysis units.

    Lazy: nothing is computed until the result is saved, persisted or loaded.

    Args:
        tree (xr.DataTree): /<model>/<experiment>, monthly, on (member, time, lat, lon), raw units.
        variable (str): Key of ``config.VARIABLES``.
        last_month (str): Every experiment is cut here (where historical ends), so "the final
            years" are the same years in all of them. seasonal_mean then keeps whole years only
            (March to February), so with '2014-12' the record ends with DJF 2013.
        chunks (dict): Chunks of the result.

    Returns:
        xr.DataTree: The same nodes on (member, year, season, lat, lon).
    """
    monthly = tree.map_over_datasets(skip_empty(to_analysis_units), variable)
    monthly = monthly.sel(time=slice(None, last_month))
    #(c): Whole time series per chunk, so each season's mean comes from one chunk
    monthly = monthly.chunk({"time": -1, "member": -1, "lat": chunks["lat"], "lon": -1})
    seasons = monthly.map_over_datasets(seasonal_mean)
    seasons = seasons.map_over_datasets(skip_empty(split_time), kwargs=dict(parts=("year", "season"), dim="time"))
    return seasons.chunk(chunks)


def seasonal_era5(monthly, variable, like, start_year=config.ERA5_START_YEAR):
    """Monthly ERA5 (as opened by ``loading.open_era5_monthly``) to seasonal means on the LESFMIP grid.

    Args:
        monthly (xr.DataArray): Monthly means on (time, lat, lon), raw units, on its own calendar.
        variable (str): Key of ``config.VARIABLES``.
        like (xr.DataArray | xr.DataTree): LESFMIP data on the target grid (a tree: its first node with data).
        start_year (int): First year kept.

    Returns:
        xr.DataArray: Named ``variable``, on (year, season, lat, lon).
    """
    if isinstance(like, xr.DataTree):
        like = next(node for node in like.leaves if node.data_vars)[variable]
    era5 = to_analysis_units(monthly.to_dataset(name=variable), variable)[variable]
    era5 = era5.sel(time=slice(str(start_year), None))
    return match_grid(seasonal_means(era5), like).rename(variable)
