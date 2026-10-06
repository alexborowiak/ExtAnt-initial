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
areas), everything here is on the common 2.5° grid of the interpolated files,
with land the cells where the ocean model gives no concentration. Each model
keeps its own edge: nothing is averaged across models.

Sections
--------
1. Areas on the sphere     cell areas, the area of a polar cap, and its inverse
2. Extent and edge         sea-ice extent, the land mask, the equivalent latitude
3. Every model             seasonal extent per member, the edge in the final years, the mean concentration
"""

import numpy as np
import xarray as xr

from . import config
from .config import WINDOW
from .datatree import reduce_to_dataset, skip_empty
from .preprocessing import seasonal_means, to_analysis_units

#(t): Mean radius of the Earth (km); areas are in km²
EARTH_RADIUS = 6371.0

#(t): Area of one hemisphere (km²)
HEMISPHERE = 2 * np.pi * EARTH_RADIUS ** 2

#(t): Concentration (%) from which a cell counts as ice-covered: the usual definition of extent
THRESHOLD = 15.0

#(t): Land south of this latitude is the continent the ice edge surrounds
CONTINENT_LAT_MAX = -60.0


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


def extent(siconc, area, threshold=THRESHOLD):
    """Sea-ice extent (km²): the area of the cells with at least ``threshold`` % ice."""
    return (area * (siconc >= threshold)).sum(("lat", "lon"))


def edge_example(siconc, land=None, threshold=THRESHOLD):
    """One field's extent and equivalent latitude, with the cells behind them, for ``plots.sea_ice.edge_schematic``.

    Args:
        siconc (xr.DataArray): Concentration (%) on (lat, lon), e.g. one member's seasonal mean.
        land (xr.DataArray | None): Land mask; where ``siconc`` is missing if None.
        threshold (float): Concentration (%) counted as ice.

    Returns:
        xr.Dataset: ``siconc``; the ``ice`` and ``continent`` cells (lat, lon) the
        two areas add up; ``extent``, ``continent_area`` and their ``edge_latitude``.
    """
    area = cell_area(siconc["lat"], siconc["lon"])
    land = siconc.isnull() if land is None else land
    sie, continent = extent(siconc, area, threshold), continent_area(land, area)
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

def seasonal_extent(monthly, variable="siconc", threshold=THRESHOLD, last_month=config.LAST_MONTH):
    """Each member's sea-ice extent, month by month, averaged into seasons; and each model's continent.

    The extent is worked out every month and then averaged, rather than taken
    from the seasonal-mean concentration, as in the paper. The record is cut
    at ``last_month`` and only complete seasons in whole years are kept, as
    for the other variables (``preprocessing.seasonal_tree``), so "the final
    years" are the same years.

    Args:
        monthly (xr.DataTree): /<model>/<experiment> concentration (%) on (member, time, lat, lon),
            as opened by ``loading.open_lesfmip_monthly``.
        variable (str): Name of the concentration.
        threshold (float): Concentration (%) counted as ice.
        last_month (str): Last month kept.

    Returns:
        xr.DataTree: The same nodes, with ``extent`` (km²) on (member, year, season) and
        ``continent_area`` (km²).
    """
    def node_extent(ds):
        siconc = to_analysis_units(ds, variable)[variable]
        area = cell_area(siconc["lat"], siconc["lon"])
        monthly_extent = extent(siconc, area, threshold).sel(time=slice(None, last_month))
        seasonal = seasonal_means(monthly_extent.rename("extent").to_dataset())
        return seasonal.assign(continent_area=continent_area(land_mask(siconc), area))

    return monthly.map_over_datasets(skip_empty(node_extent))


def edge_latitude(extents, years=WINDOW):
    """Equivalent latitude of each model's ice edge in each experiment's final ``years``.

    From the extent averaged over members and years (a climatology, as in the
    paper), not the average of each year's latitude.

    Args:
        extents (xr.DataTree): Output of ``seasonal_extent``.
        years (int): Final years.

    Returns:
        xr.DataArray: Degrees, on (model, experiment, season).
    """
    final = reduce_to_dataset(extents.isel(year=slice(-years, None)), lambda ds: ds.mean(("member", "year")))
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
        years (int): Final years.
        variable (str): Name of the concentration.

    Returns:
        xr.DataArray: % on (model, experiment, season, lat, lon).
    """
    final = reduce_to_dataset(seasonal.isel(year=slice(-years, None)), lambda ds: ds.mean(("year", "member")))
    return final[variable].transpose("model", "experiment", ...).assign_attrs(units="%", years=years)
