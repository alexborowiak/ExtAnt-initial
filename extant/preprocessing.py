"""From monthly means to the seasonal means every analysis starts from.

LESFMIP and ERA5 go through the same steps, so their seasons line up label for
label:

1. convert to analysis units (``config.VARIABLES``), as float32;
2. cut LESFMIP where historical ends (2014), and ERA5 before 1979;
3. average complete seasons (``time_axis.seasonal_mean``), dropping incomplete ones and
   any year without all four;
4. split ``time`` into ``year`` and ``season`` (``time_axis.split_time``);
5. ERA5 only: put it on the LESFMIP grid (``regrid.match_grid``).

``seasonal_tree`` and ``seasonal_era5`` run the whole chain; notebook calc_01 saves
their output, and the sample sections of the other notebooks run it on the
sample in memory. The general steps are in ``utils`` (``time_axis``, ``regrid``); here is only what is this
project's own: its units, its years and its chunks.
"""

import xarray as xr

from . import config
from .utils import regrid, time_axis
from .utils.datatree import skip_empty

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


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------

def seasonal_tree(tree, variable, last_month=config.LAST_MONTH, chunks=SEASONAL_CHUNKS):
    """Monthly LESFMIP (the monthly stores, as notebook calc_01 opens them) to seasonal means in analysis units.

    Lazy: nothing is computed until the result is saved, persisted or loaded.

    Args:
        tree (xr.DataTree): /<model>/<experiment>, monthly, on (member, time, lat, lon), raw units.
        variable (str): Key of ``config.VARIABLES``.
        last_month (str): Every experiment is cut here (where historical ends), so "the final
            years" are the same years in all of them. time_axis.seasonal_mean then keeps whole years only
            (March to February), so with '2014-12' the record ends with DJF 2013.
        chunks (dict): Chunks of the result.

    Returns:
        xr.DataTree: The same nodes on (member, year, season, lat, lon).
    """
    monthly = tree.map_over_datasets(skip_empty(to_analysis_units), variable)
    monthly = monthly.sel(time=slice(None, last_month))
    #(c): Whole time series per chunk, so each season's mean comes from one chunk
    monthly = monthly.chunk({"time": -1, "member": -1, "lat": chunks["lat"], "lon": -1})
    seasons = monthly.map_over_datasets(time_axis.seasonal_mean)
    seasons = seasons.map_over_datasets(skip_empty(time_axis.split_time),
                                        kwargs=dict(parts=("year", "season"), dim="time"))
    return seasons.chunk(chunks)


def seasonal_era5(monthly, variable, like, start_year=config.ERA5_START_YEAR):
    """Monthly ERA5 (as notebook calc_01 opens it) to seasonal means on the LESFMIP grid.

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
    return regrid.match_grid(time_axis.seasonal_means(era5), like).rename(variable)
