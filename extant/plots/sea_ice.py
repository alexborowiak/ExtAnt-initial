"""Panels of the sea ice: how a field goes onto the 2.5° grid and its edge becomes a latitude, and the edge against
ERA5's.

Draws the output of ``sea_ice``. Each function draws one panel, on an axis from a GridSpec (styled by
``charts.style_chart_ax``) or a map's (``antarctica.map_axes``); the notebook makes the figure and its axes, then
draws each panel and labels them, as everywhere (``charts``). On the maps the edge is the 15% contour of the ensemble-mean concentration, solid for the
experiment and dashed for hist-nat, in ink so that it reads on red and blue alike; ERA5's black too (``charts.ERA5_LINE``), dotted.

Sections
--------
1. Method: from a concentration field to the equivalent latitude, and onto the 2.5° grid
2. Against ERA5: the edge's latitude (``plot_edge_latitudes``), and the verdict of ERA5 among the members
   (``era5_verdict``)

A change in each zone is ``charts.plot_model_dots`` with the zones along the x axis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm

from plotting_modules.constants import FORCING_COLORS
from .. import sea_ice
from . import charts
from .charts import MODEL_MARKERS
from .era5_evaluation import INK, INK_2, MUTED, OUTSIDE_COLOR, where_era5

#(t): Colours of the cells in the method figure: open water, ice, the continent
CELL_COLORS = ("white", "#9ec5f4", "#c9c8c3")


# ---------------------------------------------------------------------------
# 1. Method: from a concentration field to the equivalent latitude, and onto the 2.5° grid
# ---------------------------------------------------------------------------

def plot_cap_area(extent, continent_area, edge_latitude, ax=None):
    """The area south of each latitude, A(φ) = 2πR²(1 + sin φ), read backwards from the ice's and the continent's area.

    The last step of how the ice edge becomes a latitude (notebook 09b, section 2): the curve, and the total area
    taken across to it and down to the latitude whose polar cap holds it.

    Args:
        extent, continent_area (float): The ice's extent and the continent's area (km²).
        edge_latitude (float): The latitude whose cap holds both (degrees, negative).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.

    Example::

        plot_cap_area(9.8e6, 13.9e6, -63.4)
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    million = 1e6
    ice, continent = extent / million, continent_area / million
    total = ice + continent
    #(c): The curve
    phi_arr = np.linspace(-90, -40, 201)
    ax.plot(phi_arr, sea_ice.cap_area(phi_arr) / million, color=INK, lw=1.6)
    #(c): The total, across to the curve and down to its latitude; the continent alone for comparison
    ax.plot([-90, edge_latitude], [total, total], color=OUTSIDE_COLOR, lw=1.4)
    ax.annotate("", xy=(edge_latitude, 0), xytext=(edge_latitude, total),
                arrowprops=dict(arrowstyle="-|>", color=OUTSIDE_COLOR, lw=1.4, shrinkA=0, shrinkB=0))
    ax.axhline(continent, color=MUTED, lw=0.9)
    ax.text(-89, continent, " continent alone", va="bottom", fontsize=8, color=INK_2)
    ax.text(edge_latitude + 0.6, total * 0.45, f"φ = {abs(edge_latitude):.1f}°S", color=INK, fontsize=9)
    ax.text(-89, total, f" {ice:.1f} + {continent:.1f} = {total:.1f}", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(-90, -40)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{abs(value):g}°S"))
    ax.set_xlabel("Latitude φ")
    ax.set_ylabel("Area south of φ (million km²)")
    return ax


#(t): The close-ups of how a field goes onto the 2.5° grid: their region, ((south, north), (west, east)) in degrees,
#(t): and their colours, the concentration in 10% steps with land grey
REGRID_REGION = ((-72.5, -52.5), (0.0, 40.0))
REGRID_LEVELS = np.arange(0, 101, 10)
REGRID_LAND_COLOR = "#c9c8c3"


def _region_points(native, lat, lon, region):
    """The points of a field on its own grid inside ``region``: their longitudes (-180 to 180), latitudes and values."""
    (south, north), (west, east) = region
    #(c): The field and its points' coordinates on the same dims, then as plain arrays of every point
    native, lat, lon = xr.align(native.reset_coords(drop=True), lat.reset_coords(drop=True),
                                lon.reset_coords(drop=True), join="inner")
    lat, lon = xr.broadcast(lat, lon)
    points_lat = lat.values.ravel()
    points_lon = (lon.values.ravel() + 180) % 360 - 180
    values = native.transpose(*lat.dims).values.ravel()
    inside = (points_lat >= south) & (points_lat <= north) & (points_lon >= west) & (points_lon <= east)
    return points_lon[inside], points_lat[inside], values[inside]


def _cell_edges(on_grid):
    """The grid's spacing in latitude and longitude, and its first cell's southern and western edges (degrees)."""
    lat_step = float(on_grid["lat"].values[1] - on_grid["lat"].values[0])
    lon_step = float(on_grid["lon"].values[1] - on_grid["lon"].values[0])
    return (lat_step, lon_step, float(on_grid["lat"].values[0]) - lat_step / 2,
            float(on_grid["lon"].values[0]) - lon_step / 2)


def _draw_cell_lines(ax, on_grid, region):
    """The grid's cells across ``region`` as orange lines, the axis limited to the region, and its axis labels."""
    (south, north), (west, east) = region
    lat_step, lon_step, first_lat_edge, first_lon_edge = _cell_edges(on_grid)
    lat_edges = first_lat_edge + lat_step * np.arange(np.floor((south - first_lat_edge) / lat_step),
                                                      np.ceil((north - first_lat_edge) / lat_step) + 1)
    lon_edges = first_lon_edge + lon_step * np.arange(np.floor((west - first_lon_edge) / lon_step),
                                                      np.ceil((east - first_lon_edge) / lon_step) + 1)
    #(c): The cells' lines take the place of the chart's gridlines
    ax.grid(False)
    for edge in lat_edges:
        ax.axhline(edge, color=OUTSIDE_COLOR, lw=0.7)
    for edge in lon_edges:
        ax.axvline(edge, color=OUTSIDE_COLOR, lw=0.7)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")


def plot_native_points(native, lat, lon, on_grid, region=REGRID_REGION, ax=None):
    """A field on its own grid, close up: one dot per point, coloured by its value, with the 2.5° cells over them.

    The first step of how a field goes onto the 2.5° grid (``sea_ice.to_grid``; notebook 09b, section 1).

    Args:
        native (xr.DataArray): Concentration (%) on the field's own grid, e.g. (j, i); empty over land.
        lat, lon (xr.DataArray): Its points' latitudes and longitudes, on the same dims.
        on_grid (xr.DataArray): ``sea_ice.to_grid`` of it, on (lat, lon): where the 2.5° cells are.
        region (tuple): ((south, north), (west, east)) of the close-up, in degrees.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.collections.PathCollection: The dots, for a colour bar (``REGRID_LEVELS``, land grey).
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    points_lon, points_lat, values = _region_points(native, lat, lon, region)
    cmap = plt.get_cmap("Blues").with_extremes(bad=REGRID_LAND_COLOR)
    dots = ax.scatter(points_lon, points_lat, c=values, cmap=cmap, norm=BoundaryNorm(REGRID_LEVELS, cmap.N), s=22,
                      edgecolor="0.5", linewidth=0.2, plotnonfinite=True)
    _draw_cell_lines(ax, on_grid, region)
    return dots


def plot_grid_cells(native, lat, lon, on_grid, region=REGRID_REGION, ax=None):
    """The 2.5° cells, close up: each coloured by the area-weighted mean of its points, with the points as dots and
    how many of them went into each cell.

    The second step of how a field goes onto the 2.5° grid (``sea_ice.to_grid``; notebook 09b, section 1).

    Args:
        native, lat, lon, on_grid, region: As ``plot_native_points``'.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.collections.QuadMesh: The cells, for a colour bar (``REGRID_LEVELS``, land grey).
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    (south, north), (west, east) = region
    lat_step, lon_step, first_lat_edge, first_lon_edge = _cell_edges(on_grid)
    cmap = plt.get_cmap("Blues").with_extremes(bad=REGRID_LAND_COLOR)
    #(c): The cells of the region, and one more each side so its edges are covered
    cells = on_grid.sel(lat=slice(south - lat_step, north + lat_step), lon=slice(west - lon_step, east + lon_step))
    mesh = ax.pcolormesh(cells["lon"].values, cells["lat"].values, np.ma.masked_invalid(cells.values), cmap=cmap,
                         norm=BoundaryNorm(REGRID_LEVELS, cmap.N), shading="nearest")
    points_lon, points_lat, _ = _region_points(native, lat, lon, region)
    ax.scatter(points_lon, points_lat, s=3, color=INK, alpha=0.5, lw=0)
    #(c): How many points fell in each cell, written in its middle
    rows = np.floor((points_lat - first_lat_edge) / lat_step)
    columns = np.floor((points_lon - first_lon_edge) / lon_step)
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([rows, columns])).groupby(level=[0, 1]).sum()
    for (row, column), count in counts.items():
        y = first_lat_edge + (row + 0.5) * lat_step
        x = first_lon_edge + (column + 0.5) * lon_step
        if south <= y <= north and west <= x <= east:
            ax.text(x, y, str(count), ha="center", va="center", fontsize=6.5, color=INK_2)
    _draw_cell_lines(ax, on_grid, region)
    return mesh


# ---------------------------------------------------------------------------
# 2. Against ERA5: the edge's latitude, and the verdicts of ERA5 among the members
# ---------------------------------------------------------------------------

def plot_edge_latitudes(edges_tree, era5_da, seasons, ax=None):
    """The equivalent latitude of each model's ice edge against ERA5's, season by season, on one panel.

    In each season, each model has a column of dots (its members) and a marker (their mean), the models side by side
    in their order in the tree; ERA5's edge is the black bar across the season.

    Args:
        edges_tree (xr.DataTree): /<model>/historical, ``edge_latitude`` (degrees) on (member, season): each
            member's edge over the years compared.
        era5_da (xr.DataArray): ERA5's, on (season).
        seasons (Sequence[str]): The seasons along the x axis, in order.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis. Model k's colour and marker are the k-th shade of historical's colour
        (``charts.model_shades``) and ``charts.MODEL_MARKERS[k]``: ``charts.model_legend_handles(models,
        experiment='historical')`` for the legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=7.5)
    charts.style_chart_ax(ax)
    leaves = edges_tree.match('*/historical').leaves
    #(c): Each model's offset from the middle of the season
    offsets_arr = np.linspace(-0.3, 0.3, len(leaves)) if len(leaves) > 1 else np.zeros(1)
    shades = charts.model_shades(FORCING_COLORS["historical"], len(leaves))
    for x, season in enumerate(seasons):
        ax.plot([x - 0.42, x + 0.42], [float(era5_da.sel(season=season))] * 2, color=charts.ERA5_LINE["color"],
                lw=2.2, zorder=1)
        for k, node in enumerate(leaves):
            values_arr = node["edge_latitude"].sel(season=season).dropna("member").values
            if not values_arr.size:
                continue
            position = x + offsets_arr[k]
            ax.scatter(np.full(values_arr.size, position), values_arr, s=6, color=shades[k], alpha=0.4, lw=0,
                       zorder=2)
            ax.plot(position, values_arr.mean(), ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=6,
                    color=shades[k], mec="white", mew=0.6, zorder=3)
    ax.set_xticks(range(len(seasons)), seasons)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{abs(value):g}°S"))
    return ax


def era5_verdict(cell_ds, diagnostic):
    """``(ok, message)`` for one statistic of ERA5's extent test (notebook 09a, section 6) at one model and season.

    ``ok`` is True (ERA5 inside the members' 5-95%), False (outside: the message says what that means for the
    model) or None (no data, or too few members for the test to flag anything). For ``era5_evaluation.status_line``.

    Args:
        cell_ds (xr.Dataset): ``era5_evaluation.locate``'s result at one statistic and season, on (member).
        diagnostic: The statistic's entry in ``sea_ice.ERA5_DIAGNOSTICS``.
    """
    if not bool(np.isfinite(cell_ds["era5"])) or not bool(cell_ds["members"].notnull().any()):
        return None, "no data"
    call = int(cell_ds["verdict"])
    if call == 0 and not bool(cell_ds["testable"]):
        return None, "too few members to test"
    if call == 0:
        return True, where_era5(cell_ds["percentile"])
    return False, diagnostic.low if call < 0 else diagnostic.high
