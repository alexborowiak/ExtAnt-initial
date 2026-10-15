"""Sea-ice extent and the latitude of the sea-ice edge, for marking the ice edge on the figures.

Bracegirdle et al. (2024, npj Clim. Atmos. Sci. 7, 276, doi:10.1038/s41612-024-00822-y)
mark the ice edge on their zonal-mean figures with an *equivalent latitude*
(Eisenman, 2010; Bracegirdle et al., 2018): the latitude the edge would be at if
the sea ice and the continent it surrounds were a cap round the pole. The area
of the Earth south of latitude phi is

    A(phi) = 2 pi R^2 (1 + sin phi),

so the equivalent latitude is the phi for which A(phi) equals the sea-ice
extent plus the area of the continent:

    phi_eq = arcsin((SIE + continent) / (2 pi R^2) - 1).

The sea-ice extent (SIE) is the area of the cells with at least 15% ice. The
continent is the land south of 60°S, as in the paper (land further north, such
as South Georgia, is not part of the ice edge). With no sea ice the equivalent
latitude is that of the continent alone, about 71°S.

Unlike the paper (which uses each model's native ocean grid and its own cell
areas), everything here is on the common 2.5° grid of the temperature files,
with land the cells where the ocean model gives no concentration. The sea ice
comes from CEDA's copy of the CMIP6 archive, on each model's own grid, and is
put on the 2.5° grid first (section 0), as is ERA5's. Each model keeps its own
edge: nothing is averaged across models.

Sections
--------
0. Onto the 2.5° grid      which cell each point falls in, area-weighted cell means, one model's members
1. Areas on the sphere     cell areas, the area of a polar cap, and its inverse
2. Extent and edge         sea-ice extent, the land mask, the equivalent latitude
3. Every model             the edge in the final years, the mean concentration, the seasonal cycle
4. Against ERA5            how the figures label the extent's mean, variability and trend
5. Where the ice changed   zones of ice kept, lost and gained, and the mean of any field in each
"""

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from . import era5_evaluation as ev
from .config import WINDOW
from .datatree import reduce_to_dataset
from .preprocessing import to_analysis_units

#(t): Mean radius of the Earth (km); areas are in km²
EARTH_RADIUS = 6371.0

#(t): Area of one hemisphere (km²)
HEMISPHERE = 2 * np.pi * EARTH_RADIUS ** 2

#(t): Concentration (%) from which a cell counts as ice-covered: the usual definition of extent
THRESHOLD = 15.0

#(t): Land south of this latitude is the continent the ice edge surrounds
CONTINENT_LAT_MAX = -60.0


# ---------------------------------------------------------------------------
# 0. Onto the 2.5° grid
# ---------------------------------------------------------------------------

#(t): The cell areas of the grid each sea-ice variable is on: the ocean model's (Ofx) and the atmosphere model's (fx)
AREA_VARIABLES = {"siconc": "areacello", "siconca": "areacella"}

#(t): The grid of every LESFMIP store (notebook 01), south of config.LAT_MAX: cell centres every 2.5°, from 87.5°S
#(t): to 40°S and from 180°W to 177.5°E, so each cell reaches 1.25° either side of its centre
GRID_LAT = np.arange(-87.5, config.LAT_MAX + 1.25, 2.5)
GRID_LON = np.arange(-180.0, 180.0, 2.5)


def native_lat_lon(ds):
    """The latitudes and longitudes of a file's points: 2-D (j, i) on an ocean model's grid, 1-D on a regular one."""
    for lat, lon in (("latitude", "longitude"), ("lat", "lon"), ("nav_lat", "nav_lon")):
        if lat in ds.variables and lon in ds.variables:
            return ds[lat], ds[lon]
    raise KeyError(f"no latitude and longitude among {list(ds.variables)}")


def box_index(lat, lon, grid_lat=GRID_LAT, grid_lon=GRID_LON):
    """Which cell of the 2.5° grid each point falls in, numbered row by row: row × len(grid_lon) + column.

    Each cell of the grid reaches half a step either side of its centre,
    closed on its south and west sides, so a point exactly on a cell edge
    (ERA5 has points on 41.25°S, say) goes to the cell north or east of it.
    Longitudes are taken round the globe, so 0-360 and -180-180 both work.
    For example, with cells centred on 87.5°S, 85°S, ... and 180°W, 177.5°W, ...::

        point (-63.0°, 2.0°E):  row    floor((-63.0 + 88.75) / 2.5)          = 10   (the cell centred on 62.5°S)
                                column floor(((2.0 + 181.25) % 360) / 2.5)   = 73   (the cell centred on 2.5°E)
                                box    10 × 144 + 73                         = 1513

    Points outside the grid (north of its last row, or with no position, like
    the missing latitudes over land in some ocean-model files) get -1.

    Args:
        lat, lon (xr.DataArray): The points' latitudes and longitudes (degrees): 2-D (j, i) on an ocean
            model's grid, or 1-D each on a regular grid (they are broadcast against each other).
        grid_lat, grid_lon (np.ndarray): Centres of the 2.5° grid's cells, evenly spaced.

    Returns:
        xr.DataArray: int, named ``box``, on the dims of ``lat`` and ``lon`` together.
    """
    lat, lon = xr.broadcast(lat, lon)
    lat_step, lon_step = grid_lat[1] - grid_lat[0], grid_lon[1] - grid_lon[0]
    row = np.floor((lat - (grid_lat[0] - lat_step / 2)) / lat_step)
    column = np.floor(((lon - (grid_lon[0] - lon_step / 2)) % 360) / lon_step)
    #(c): NaN fails every comparison, and fill values (1e30) are beyond 90°
    inside = (row >= 0) & (row < grid_lat.size) & (np.abs(lat) <= 90)
    return (row * grid_lon.size + column).where(inside, -1).astype(int).rename("box")


def _box_means(values, matrix, n_point_dims, shape):
    """Each cell's area-weighted mean of the points in it: values (..., *points) -> (..., *shape); NaN with none.

    ``matrix`` (points x cells) holds each point's area in the column of its
    cell, so one product sums area x value over each cell's points and another
    sums the area of the points that have a value: their ratio is the mean.
    """
    leading = values.shape[:values.ndim - n_point_dims]
    flat = values.reshape(-1, matrix.shape[0])
    present = np.isfinite(flat)
    sums = np.where(present, flat, 0.0) @ matrix
    areas = present.astype(matrix.dtype) @ matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(areas > 0, sums / areas, np.nan)
    return means.reshape(*leading, *shape).astype(values.dtype)


def to_grid(da, boxes, area, grid_lat=GRID_LAT, grid_lon=GRID_LON):
    """Put a field on the 2.5° grid: each cell the area-weighted mean of the points whose centres fall in it.

    For a cell C of the 2.5° grid, with the points k of the field's own grid
    whose centres are in C, their areas a_k and values c_k,

        c_C = Σ_k a_k c_k / Σ_k a_k,

    leaving out points with no value (land, in an ocean model). A cell with no
    point with a value is missing: on the 2.5° grid, land is where every point
    of the field's own grid in that cell is land. Where the field's points are
    much smaller than the 2.5° cells (an ocean model's 1° grid, ERA5's 0.25°),
    this is the conservative regridding notebook 01 does for ERA5's
    temperature, give or take the points that straddle a cell edge.

    Args:
        da (xr.DataArray): The field on its own grid, e.g. (time, j, i); any other dims are kept.
        boxes (xr.DataArray): ``box_index`` of its points.
        area (xr.DataArray): Each point's area (any units; missing or 0 over land), on the same dims.
        grid_lat, grid_lon (np.ndarray): Centres of the 2.5° grid's cells.

    Returns:
        xr.DataArray: On (..., lat, lon), the same dtype.
    """
    points = list(boxes.dims)
    da, boxes, area = _inside(da, boxes, area)
    shape = (grid_lat.size, grid_lon.size)
    matrix = _matrix(boxes, area, shape)
    if da.chunks:
        da = da.chunk({dim: -1 for dim in points})
    on_grid = xr.apply_ufunc(
        _box_means, da.transpose(..., *points), input_core_dims=[points], output_core_dims=[["lat", "lon"]],
        exclude_dims=set(points), kwargs={"matrix": matrix, "n_point_dims": len(points), "shape": shape},
        dask="parallelized",
        output_dtypes=[da.dtype], dask_gufunc_kwargs={"output_sizes": {"lat": shape[0], "lon": shape[1]}},
    )
    return on_grid.assign_coords(lat=grid_lat, lon=grid_lon)


def _inside(da, boxes, area):
    """``da``, ``boxes`` and ``area`` on the points they share, less the rows and columns with no point in the grid."""
    points = list(boxes.dims)
    da, boxes, area = xr.align(da.reset_coords(drop=True), boxes.reset_coords(drop=True),
                               area.reset_coords(drop=True), join="inner", exclude=set(da.dims) - set(points))
    #(c): Dropped before anything is read, so a global file is read only south of the grid's edge
    inside = boxes >= 0
    keep = {dim: inside.any([other for other in points if other != dim]).values for dim in points}
    return da.isel(keep), boxes.isel(keep), area.isel(keep)


def _matrix(boxes, area, shape):
    """Points x cells, sparse: each point's area in the column of the 2.5° cell it is in (none outside the grid)."""
    import scipy.sparse

    box = boxes.values.ravel()
    weight = np.nan_to_num(area.transpose(*boxes.dims).values.ravel().astype("float64"))
    used = (box >= 0) & (weight > 0)
    return scipy.sparse.csr_array((weight[used], (np.flatnonzero(used), box[used])),
                                  shape=(box.size, shape[0] * shape[1]))


def ocean_area_on_grid(ocean, boxes, area, grid_lat=GRID_LAT, grid_lon=GRID_LON):
    """How much of each 2.5° cell is ocean: the summed areas of the points in it that are ocean.

    A cell on the coast is part land, but its concentration (``to_grid``) is
    the mean over its ocean only. Its whole area would count it as ice from
    shore to shore, and add about 3 million km² of coast to the extent (and
    take it from the continent); its ocean area does not. Each point is in
    exactly one cell, so the ocean areas add up to the field's own, even where
    one cell gets a point more than its neighbour.

    Args:
        ocean (xr.DataArray): True at the field's points that are ocean, e.g. where it has a value.
        boxes (xr.DataArray): ``box_index`` of the points.
        area (xr.DataArray): Each point's area (km²).
        grid_lat, grid_lon (np.ndarray): Centres of the 2.5° grid's cells.

    Returns:
        xr.DataArray: km², on (lat, lon); 0 where a cell is all land.
    """
    ocean, boxes, area = _inside(ocean, boxes, area)
    shape = (grid_lat.size, grid_lon.size)
    matrix = _matrix(boxes, area.where(ocean, 0.0), shape)
    sums = np.ones(matrix.shape[0]) @ matrix
    return xr.DataArray(sums.reshape(shape), dims=("lat", "lon"), coords={"lat": grid_lat, "lon": grid_lon},
                        name="ocean_area", attrs={"units": "km2"})


def open_on_grid(member_files, area_file, variable, ocean_area=None, last_month=config.LAST_MONTH):
    """One model's members for one experiment, from their files on its own grid onto the 2.5° grid (lazily).

    Each member is opened (its record may be split over several files), cut
    at ``last_month``, and the members are stacked along ``member``; then
    ``to_grid`` with the cell areas of the model's grid, and the ocean area of
    each 2.5° cell (``ocean_area_on_grid``), from the points where the first
    member has a value.

    Args:
        member_files (dict): Member label -> its files.
        area_file (Path): The cell areas (m²) of the grid ``variable`` is on: areacello (Ofx) for siconc,
            areacella (fx) for siconca.
        variable (str): 'siconc' (on the ocean grid) or 'siconca' (on the atmosphere grid).
        ocean_area (xr.DataArray | None): For siconca, which is 0 rather than missing over land, so its own
            ocean cannot be told from its land: the ocean area (km²) of each 2.5° cell from elsewhere (ERA5);
            the cells with none are made missing, as an ocean model leaves them.
        last_month (str): The record is cut here.

    Returns:
        xr.Dataset: ``siconc`` (%) on (member, time, lat, lon), whichever ``variable`` it came from, and
        ``ocean_area`` (km²) on (lat, lon).
    """
    area_ds = xr.open_dataset(area_file)
    area_name = next(name for name in area_ds.data_vars if name.startswith("area"))
    #(c): CMIP's cell areas are in m²; km² here, as everywhere in this module
    area = area_ds[area_name] / 1e6
    boxes = box_index(*native_lat_lon(area_ds))

    members = []
    for member, files in member_files.items():
        ds = xr.open_mfdataset(sorted(files), combine="by_coords", data_vars="minimal", coords="minimal",
                               compat="override", chunks={"time": -1})
        members.append(ds[variable].sel(time=slice(None, last_month)).reset_coords(drop=True))
    lengths = {member: da.sizes["time"] for member, da in zip(member_files, members)}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"the members' records differ in length (months): {lengths}")
    stacked = xr.concat(members, dim=pd.Index(list(member_files), name="member", dtype=object), join="override",
                        coords="minimal", compat="override")

    siconc = to_grid(stacked, boxes, area)
    if ocean_area is None:
        ocean_area = ocean_area_on_grid(stacked.isel(member=0, time=0).notnull(), boxes, area)
    else:
        siconc = siconc.where(ocean_area > 0)
    siconc = siconc.astype("float32").assign_attrs(units="%", source_variable=variable)
    return xr.Dataset({"siconc": siconc.chunk({"member": 5, "time": -1, "lat": -1, "lon": -1}),
                       "ocean_area": ocean_area})


# ---------------------------------------------------------------------------
# 1. Areas on the sphere
# ---------------------------------------------------------------------------

def cell_area(lat, lon):
    """Area (km²) of each cell of a regular latitude-longitude grid, exact on the sphere.

    A cell's edges are half-way between its centre and its neighbours'
    (the outer edges half a spacing out, but no further than the poles), and
    its area is R² Δλ |sin φ_north - sin φ_south|.

    Args:
        lat, lon (xr.DataArray): The grid's 1-D coordinates, at least two values each.

    Returns:
        xr.DataArray: On (lat, lon).
    """
    phi = np.asarray(lat, dtype=float)
    edges = np.concatenate([[1.5 * phi[0] - 0.5 * phi[1]], (phi[1:] + phi[:-1]) / 2, [1.5 * phi[-1] - 0.5 * phi[-2]]])
    edges = np.deg2rad(np.clip(edges, -90, 90))
    #(c): Every longitude has the same width on a regular grid; np.gradient gives it at the ends too
    width = np.deg2rad(np.abs(np.gradient(np.asarray(lon, dtype=float))))
    area = EARTH_RADIUS ** 2 * np.abs(np.diff(np.sin(edges)))[:, None] * width[None, :]
    return xr.DataArray(area, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="cell_area",
                        attrs={"units": "km2"})


def cap_area(lat):
    """Area (km²) of the Earth south of ``lat`` (degrees): 2πR²(1 + sin φ)."""
    return HEMISPHERE * (1 + np.sin(np.deg2rad(lat)))


def equivalent_latitude(area):
    """The latitude (degrees) whose polar cap has ``area`` (km²): the inverse of ``cap_area``."""
    return np.rad2deg(np.arcsin(np.clip(area / HEMISPHERE - 1, -1, 1)))


# ---------------------------------------------------------------------------
# 2. Extent and edge
# ---------------------------------------------------------------------------

def land_mask(siconc, months=12):
    """Land: the cells with no concentration in any of the first ``months`` months of the first member.

    The ocean model has no value over land, so the interpolated files leave
    land empty; open water is 0%. One year is enough to tell them apart.
    """
    first = siconc.isel(time=slice(0, months))
    if "member" in first.dims:
        first = first.isel(member=0)
    return first.isnull().all("time")


def continent_area(land, area, lat_max=CONTINENT_LAT_MAX):
    """Area (km²) of the land south of ``lat_max``."""
    return area.where(land & (area["lat"] <= lat_max)).sum(("lat", "lon"))


def continent_from_ocean(ocean_area, lat_max=CONTINENT_LAT_MAX):
    """Area (km²) of the land south of ``lat_max``: each 2.5° cell's area less its ocean (``ocean_area_on_grid``)."""
    area = cell_area(ocean_area["lat"], ocean_area["lon"])
    return (area - ocean_area).where(area["lat"] <= lat_max).sum(("lat", "lon"))


def extent(siconc, area, threshold=THRESHOLD):
    """Sea-ice extent (km²): the area of the cells with at least ``threshold`` % ice."""
    return (area * (siconc >= threshold)).sum(("lat", "lon"))


def edge_example(siconc, land=None, threshold=THRESHOLD, ocean_area=None):
    """One field's extent and equivalent latitude, with the cells behind them, for ``plots.sea_ice.edge_schematic``.

    Args:
        siconc (xr.DataArray): Concentration (%) on (lat, lon), e.g. one member's seasonal mean.
        land (xr.DataArray | None): Land mask; where ``siconc`` is missing if None.
        threshold (float): Concentration (%) counted as ice.
        ocean_area (xr.DataArray | None): Each cell's ocean (km², ``ocean_area_on_grid``): the extent
            counts only that, and the continent is the rest. Whole cells if None.

    Returns:
        xr.Dataset: ``siconc``; the ``ice`` and ``continent`` cells (lat, lon) the
        two areas add up; ``extent``, ``continent_area`` and their ``edge_latitude``.
    """
    area = cell_area(siconc["lat"], siconc["lon"])
    land = siconc.isnull() if land is None else land
    if ocean_area is None:
        sie, continent = extent(siconc, area, threshold), continent_area(land, area)
    else:
        sie, continent = extent(siconc, ocean_area, threshold), continent_from_ocean(ocean_area)
    return xr.Dataset({
        "siconc": siconc,
        "ice": siconc >= threshold,
        "continent": land & (siconc["lat"] <= CONTINENT_LAT_MAX),
        "extent": sie,
        "continent_area": continent,
        "edge_latitude": equivalent_latitude(sie + continent),
    }).assign_attrs(threshold=threshold, continent_lat_max=CONTINENT_LAT_MAX)


# ---------------------------------------------------------------------------
# 3. Every model
# ---------------------------------------------------------------------------

def monthly_extent(ds, variable="siconc", threshold=THRESHOLD):
    """Each month's sea-ice extent of one dataset (a model's members, or ERA5), and the area of its continent.

    With ``ocean_area`` in ``ds`` (``open_on_grid``), the extent counts only the
    ocean of each cell and the continent is the rest; otherwise whole cells,
    land being where the concentration is missing.

    Args:
        ds (xr.Dataset): ``variable``, the concentration (%) on (..., time, lat, lon), and ``ocean_area``
            (km²) on (lat, lon) if there is one.
        variable (str): Name of the concentration.
        threshold (float): Concentration (%) counted as ice.

    Returns:
        xr.Dataset: ``extent`` (km²) on (..., time) and ``continent_area`` (km²).
    """
    siconc = to_analysis_units(ds, variable)[variable]
    if "ocean_area" in ds:
        #(c): Each 2.5° cell's ocean only, so a cell that is half land counts half (ocean_area_on_grid)
        return xr.Dataset({"extent": extent(siconc, ds["ocean_area"], threshold),
                           "continent_area": continent_from_ocean(ds["ocean_area"])})
    area = cell_area(siconc["lat"], siconc["lon"])
    return xr.Dataset({"extent": extent(siconc, area, threshold),
                       "continent_area": continent_area(land_mask(siconc), area)})


def edge_latitude(extents, years=WINDOW):
    """Equivalent latitude of each model's ice edge in each experiment's final ``years``.

    From the extent averaged over members and years (a climatology, as in the
    paper), not the average of each year's latitude.

    Args:
        extents (xr.DataTree): Each member's seasonal ``extent`` and its model's ``continent_area`` (km²),
            /<model>/<experiment>, as notebook 09 makes them.
        years (int): Final years.

    Returns:
        xr.DataArray: Degrees, on (model, experiment, season).
    """
    final = final_mean(extents, years)
    latitude = equivalent_latitude(final["extent"] + final["continent_area"])
    return latitude.transpose("model", "experiment", ...).rename("edge_latitude").assign_attrs(
        units="degrees_north", years=years)


def extent_summary(extents, quantiles=(0.05, 0.95)):
    """Ensemble mean and spread of the extent in every year, for ``plots.sea_ice.extent_timeseries``.

    Returns:
        xr.Dataset: ``mean``, ``low`` and ``high`` (the ``quantiles`` across members) on
        (model, experiment, year, season), in km², and ``continent_area`` on (model, experiment).
    """
    def summarise(ds):
        spread = ds["extent"].quantile(list(quantiles), "member")
        return xr.Dataset({
            "mean": ds["extent"].mean("member"),
            "low": spread.isel(quantile=0, drop=True),
            "high": spread.isel(quantile=1, drop=True),
            "continent_area": ds["continent_area"],
        })

    return reduce_to_dataset(extents, summarise).transpose("model", "experiment", ...).assign_attrs(
        quantiles=list(quantiles))


def concentration_climatology(seasonal, years=WINDOW, variable="siconc"):
    """Ensemble-mean concentration over each experiment's final ``years``, for drawing the 15% ice edge on maps.

    Args:
        seasonal (xr.DataTree): Seasonal concentration (%), as made by ``preprocessing.seasonal_tree``.
        years (int | slice): The final ``years`` years, or the years in a slice (e.g. those compared with ERA5).
        variable (str): Name of the concentration.

    Returns:
        xr.DataArray: % on (model, experiment, season, lat, lon).
    """
    period = seasonal.sel(year=years) if isinstance(years, slice) else seasonal.isel(year=slice(-years, None))
    final = reduce_to_dataset(period, lambda ds: ds.mean(("year", "member")))
    return final[variable].transpose("model", "experiment", ...).assign_attrs(units="%", years=str(years))


def final_mean(extents, years=WINDOW):
    """Each model's ensemble-mean extent over each experiment's final ``years``, and its continent.

    Args:
        extents (xr.DataTree): Each member's seasonal ``extent`` and its model's ``continent_area`` (km²).
        years (int): Final years.

    Returns:
        xr.Dataset: ``extent`` (km²) on (model, experiment, season) and ``continent_area`` (km²).
    """
    final = reduce_to_dataset(extents.isel(year=slice(-years, None)), lambda ds: ds.mean(("member", "year")))
    return final.transpose("model", "experiment", ...).assign_attrs(years=years)


def monthly_cycle(ds, years=None):
    """The mean extent in each calendar month over ``years`` (all if None), in million km²: the seasonal cycle.

    Args:
        ds (xr.Dataset): ``extent`` (km²) on (..., time), from ``monthly_extent``.
        years (slice | None): Calendar years, e.g. ``slice(1979, 2013)``.

    Returns:
        xr.Dataset: ``extent`` (million km²) on (..., month).
    """
    extent = ds["extent"] if years is None else ds["extent"].sel(time=slice(str(years.start), str(years.stop)))
    return (extent.groupby("time.month").mean() / 1e6).to_dataset(name="extent")


# ---------------------------------------------------------------------------
# 4. Against ERA5
# ---------------------------------------------------------------------------

#(t): How the figures label ERA5's tests of the extent (notebook 09, section 4.4), and what each means for the model
#(t): when ERA5 falls below (low) or above (high) its members: the mean of the extent, the year-to-year spread of
#(t): the detrended extent, and the linear trend
ERA5_DIAGNOSTICS = {
    "mean": ev.Diagnostic("Mean extent", "million km²", "model has too much ice", "model has too little ice",
                          treatment="raw"),
    "std": ev.Diagnostic("Year-to-year std", "million km²", "model too variable", "model not variable enough",
                         treatment="detrended"),
    "trend": ev.Diagnostic("Trend", "million km² per decade", "model's trend too positive",
                           "model's trend too negative", treatment="raw"),
}


# ---------------------------------------------------------------------------
# 5. Where the ice changed
# ---------------------------------------------------------------------------

#(t): The zones a cell can be in, comparing an experiment's final years with hist-nat's: open water in both, ice (at
#(t): least THRESHOLD %) in both, ice in hist-nat only (lost), in the experiment only (gained), or the continent
ZONES = ("open water", "ice in both", "ice lost", "ice gained", "continent")


def ice_zones(hist_nat, experiment, threshold=THRESHOLD, lat_max=CONTINENT_LAT_MAX):
    """Each cell's zone, the position of its name in ``ZONES``, from hist-nat's and an experiment's concentration.

    Both are climatologies, e.g. the ensemble means over the final years
    (``concentration_climatology``). Land north of ``lat_max`` (South America,
    the islands) is in no zone, and nothing is where either has no data at all.

    Args:
        hist_nat, experiment (xr.DataArray): Concentration (%) on (..., lat, lon).
        threshold (float): Concentration (%) counted as ice.
        lat_max (float): Land south of this is the continent.

    Returns:
        xr.DataArray: 0 to 4 (NaN in no zone), on the dims of the two together.
    """
    in_hist_nat, in_experiment = hist_nat >= threshold, experiment >= threshold
    ocean = xr.where(in_hist_nat, xr.where(in_experiment, 1, 2), xr.where(in_experiment, 3, 0))
    land = hist_nat.isnull()
    zones = xr.where(land, xr.where(hist_nat["lat"] <= lat_max, 4, np.nan), ocean)
    has_data = hist_nat.notnull().any(("lat", "lon")) & experiment.notnull().any(("lat", "lon"))
    return zones.where(has_data).rename("zone").assign_attrs(zones=list(ZONES), threshold=threshold)


def _zone_masks(zones):
    """One True/False map per zone, on a new ``zone`` dim named as in ``ZONES``."""
    codes = xr.DataArray(np.arange(len(ZONES)), dims="zone", coords={"zone": list(ZONES)})
    return zones == codes


def zone_means(field, zones):
    """The area-weighted mean of ``field`` over each zone's cells, on a new ``zone`` dim.

    Only the cells (and models, experiments, seasons) the two share count: the
    temperature on the sample is a 3x3 block of cells, say.

    Args:
        field (xr.DataArray | xr.Dataset): e.g. the change in the mean temperature, on (..., lat, lon).
        zones (xr.DataArray): ``ice_zones`` output, on dims ``field`` shares.

    Returns:
        xr.DataArray | xr.Dataset: On (..., zone); NaN where a zone has no cell with a value.
    """
    field, zones = xr.align(field, zones, join="inner")
    area = cell_area(field["lat"], field["lon"])
    return field.weighted(area.where(_zone_masks(zones), 0.0)).mean(("lat", "lon"))


def zone_areas(zones):
    """The area (million km²) of each zone, on a new ``zone`` dim."""
    area = cell_area(zones["lat"], zones["lon"])
    return (area.where(_zone_masks(zones)).sum(("lat", "lon")) / 1e6).assign_attrs(units="million km²")
