"""Figures of the sea ice: the extent through time, how the edge and the regridding work, the models against ERA5, and the zones.

Draws the output of ``sea_ice``. On the maps (drawn with ``antarctica``) the
edge is the 15% contour of the ensemble-mean concentration, solid for the
experiment and dashed for hist-nat, in ink so that it reads on red and blue
alike; ERA5's in orange (``ERA5_EDGE_COLOR``).

Sections
--------
1. Through time: each experiment's extent, every model
2. Method: from a concentration field to the equivalent latitude, and onto the 2.5° grid
3. Against ERA5: the seasonal cycle, the edge's latitude, and ERA5 among the members
4. Where the ice changed: a field's mean in each zone
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import sea_ice
from .era5_evaluation import ENSEMBLE, EVAL_RC, INK, INK_2, MODEL_COLORS, MODEL_MARKERS, MUTED, OUTSIDE_COLOR, \
    STATUS_FLAG, draw_member_strip, where_era5
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER
from plotting_modules.utils import plot

#(t): ERA5's colour on the maps and in the line plots, apart from the models' ink
ERA5_EDGE_COLOR = "#d95f02"

#(t): Colours of the cells in the method figure: open water, ice, the continent
CELL_COLORS = ("white", "#9ec5f4", "#c9c8c3")


# ---------------------------------------------------------------------------
# 1. Through time: each experiment's extent, every model
# ---------------------------------------------------------------------------

@plot("figure")
def extent_timeseries(summary, seasons=("DJF", "JJA"), experiments=None, title=None):
    """Sea-ice extent through time: ensemble mean (line) and the spread of the members (band); seasons down, models across.

    Args:
        summary (xr.Dataset): ``sea_ice.extent_summary`` output.
        seasons (Sequence[str]): Rows.
        experiments (Sequence[str] | None): Experiments shown; FORCING_REVEAL_ORDER by default.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    present = [str(e) for e in summary["experiment"].values]
    experiments = [e for e in experiments or FORCING_REVEAL_ORDER if e in present]
    models = [str(m) for m in summary["model"].values]
    low, high = (round(100 * q) for q in summary.attrs.get("quantiles", (0.05, 0.95)))
    million = 1e6

    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(len(seasons), len(models), figsize=(3.2 * len(models) + 0.8, 2.6 * len(seasons) + 1),
                                 sharex=True, sharey="row", layout="constrained", squeeze=False)
        for i, season in enumerate(seasons):
            for j, model in enumerate(models):
                ax = axes[i, j]
                for experiment in experiments:
                    cell = summary.sel(model=model, experiment=experiment, season=season)
                    if not bool(cell["mean"].notnull().any()):
                        continue
                    colour = FORCING_COLORS.get(experiment, INK)
                    ax.fill_between(cell["year"], cell["low"] / million, cell["high"] / million, color=colour,
                                    alpha=0.15, lw=0)
                    ax.plot(cell["year"], cell["mean"] / million, color=colour, lw=1.4)
                if i == 0:
                    ax.set_title(model, loc="left")
                if j == 0:
                    ax.set_ylabel(f"{season}\nExtent (million km²)")
                core.style_ax(ax)
        handles = [Line2D([], [], color=FORCING_COLORS.get(e, INK), lw=2, label=e) for e in experiments]
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False,
                   title=f"Ensemble mean; band: {low}–{high}% of members", title_fontsize=8)
        fig.suptitle(title or "Antarctic sea-ice extent (cells with at least 15% ice)", fontsize=11,
                     fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 2. Method: from a concentration field to the equivalent latitude, and onto the 2.5° grid
# ---------------------------------------------------------------------------

def plot_cap_area(extent, continent_area, edge_latitude, ax=None):
    """The area south of each latitude, A(φ) = 2πR²(1 + sin φ), read backwards from the ice's and the continent's area.

    The last step of how the ice edge becomes a latitude (notebook 09b, section 2): the curve, and the total area
    taken across to it and down to the latitude whose polar cap holds it.

    Args:
        extent, continent_area (float): The ice's extent and the continent's area (km²).
        edge_latitude (float): The latitude whose cap holds both (degrees, negative).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure if None.

    Returns:
        matplotlib.axes.Axes: The axis.

    Example::

        plot_cap_area(9.8e6, 13.9e6, -63.4)
    """
    if ax is None:
        fig, ax = plt.subplots()
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
    core.style_ax(ax)
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
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure if None.

    Returns:
        matplotlib.collections.PathCollection: The dots, for a colour bar (``REGRID_LEVELS``, land grey).
    """
    if ax is None:
        fig, ax = plt.subplots()
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
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure if None.

    Returns:
        matplotlib.collections.QuadMesh: The cells, for a colour bar (``REGRID_LEVELS``, land grey).
    """
    if ax is None:
        fig, ax = plt.subplots()
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
# 3. Against ERA5: the seasonal cycle, the edge's latitude, and ERA5 among the members
# ---------------------------------------------------------------------------

MONTH_LETTERS = "JFMAMJJASOND"


def _by_model(data, variable=None):
    """{model: its data}, from a DataTree /<model>/<experiment> holding one experiment (each model's own members).

    ``variable`` picks one variable of each node; without it each node's Dataset.
    """
    return {node.parent.name: node[variable] if variable else node.to_dataset()
            for node in data.leaves if node.data_vars}


@plot("figure")
def seasonal_cycle(cycle, era5, title=None):
    """Each model's seasonal cycle of extent against ERA5's: one panel per model, every member a line.

    Args:
        cycle (xr.DataTree): /<model>/historical, ``extent`` (million km²) on (member, month): each
            member's mean in each calendar month over the same years as ERA5.
        era5 (xr.DataArray): ERA5's, on (month).
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    by_model = _by_model(cycle, "extent")
    months = era5["month"].values
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(by_model), figsize=(2.6 * len(by_model) + 0.6, 3.4), sharey=True,
                                 layout="constrained", squeeze=False)
        for ax, (model, members) in zip(axes[0], by_model.items()):
            for member in members["member"].values:
                ax.plot(months, members.sel(member=member), color=ENSEMBLE["member"], lw=0.6, alpha=0.6)
            if members.sizes["member"]:
                ax.plot(months, members.mean("member"), color=ENSEMBLE["median"], lw=1.8)
            ax.plot(era5["month"], era5, color=ERA5_EDGE_COLOR, lw=1.8, marker="o", ms=3, zorder=5)
            ax.set_title(model, loc="left")
            n_members = members.sizes["member"]
            ax.text(0.03, 0.03, f"{n_members} member{'s' if n_members != 1 else ''}", transform=ax.transAxes,
                    fontsize=7.5, color=INK_2)
            ax.set_xticks(months, [MONTH_LETTERS[m - 1] for m in months])
            core.style_ax(ax)
        axes[0, 0].set_ylabel("Extent (million km²)")
        fig.legend(handles=[Line2D([], [], color=ENSEMBLE["member"], lw=0.8, label="Each member"),
                            Line2D([], [], color=ENSEMBLE["median"], lw=1.8, label="Ensemble mean"),
                            Line2D([], [], color=ERA5_EDGE_COLOR, lw=1.8, marker="o", ms=3, label="ERA5")],
                   loc="outside lower center", ncols=3, frameon=False)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def edge_latitudes(members, era5, title=None):
    """The equivalent latitude of each model's ice edge, every member, against ERA5's, season by season.

    Args:
        members (xr.DataTree): /<model>/historical, ``edge_latitude`` (degrees) on (member, season): each
            member's edge over the years compared.
        era5 (xr.DataArray): ERA5's, on (season).
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    by_model = _by_model(members, "edge_latitude")
    models = list(by_model)
    seasons = [str(s) for s in era5["season"].values]
    offsets = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else np.zeros(1)
    with plt.rc_context(EVAL_RC):
        fig, ax = plt.subplots(figsize=(1.9 * len(seasons) + 1.6, 4.0), layout="constrained")
        for x, season in enumerate(seasons):
            ax.plot([x - 0.42, x + 0.42], [float(era5.sel(season=season))] * 2, color=ERA5_EDGE_COLOR, lw=2.2,
                    zorder=1)
            for k, model in enumerate(models):
                values = by_model[model].sel(season=season).dropna("member").values
                if not values.size:
                    continue
                position = x + offsets[k]
                ax.scatter(np.full(values.size, position), values, s=6, color=MODEL_COLORS[k % len(MODEL_COLORS)],
                           alpha=0.4, lw=0, zorder=2)
                ax.plot(position, values.mean(), ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=6,
                        color=MODEL_COLORS[k % len(MODEL_COLORS)], mec="white", mew=0.6, zorder=3)
        ax.set_xticks(range(len(seasons)), seasons)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}°S"))
        ax.set_ylabel("Equivalent latitude of the ice edge")
        core.style_ax(ax)
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=6,
                          color=MODEL_COLORS[k % len(MODEL_COLORS)], label=model) for k, model in enumerate(models)]
        handles.append(Line2D([], [], color=ERA5_EDGE_COLOR, lw=2.2, label="ERA5"))
        fig.legend(handles=handles, loc="outside right center", frameon=False,
                   title="Ensemble mean\n(dots: each member)", title_fontsize=8)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=np.array([[ax]], dtype=object))


def _verdict(cell, diagnostic):
    """(ok, message) for one statistic of ERA5's extent test (notebook 09, section 4.4) at one model and season."""
    if not bool(np.isfinite(cell["era5"])) or not bool(cell["members"].notnull().any()):
        return None, "no data"
    call = int(cell["verdict"])
    if call == 0 and not bool(cell["testable"]):
        return None, "too few members to test"
    if call == 0:
        return True, where_era5(cell["percentile"])
    return False, diagnostic.low if call < 0 else diagnostic.high


@plot("figure")
def era5_strips(result, statistic, diagnostics=sea_ice.ERA5_DIAGNOSTICS, title=None):
    """One statistic of the extent: where ERA5 falls among each model's members, models down, seasons across.

    Each panel is the members (dots, their 5-95% shaded, their median a line)
    and ERA5 (black). If the model were right, ERA5 would land outside the
    shading only 10% of the time; a tick says it is inside, a cross that it is
    outside (and what that means for the model), a dash that the model has too
    few members for the test to flag anything.

    Args:
        result (xr.DataTree): /<model>/historical, ``era5_evaluation.locate`` results on (statistic, season,
            member), as notebook 09a (section 6) makes them.
        statistic (str): Which statistic.
        diagnostics (dict): ``sea_ice.ERA5_DIAGNOSTICS``.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    from .era5_evaluation import _status_line

    diagnostic = diagnostics[statistic]
    by_model = _by_model(result)
    models = list(by_model)
    seasons = [s for s in ("DJF", "MAM", "JJA", "SON") if s in next(iter(by_model.values()))["season"].values]
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(len(models), len(seasons),
                                 figsize=(2.5 * len(seasons) + 1.2, 1.15 * len(models) + 1.0),
                                 layout="constrained", squeeze=False)
        for i, model in enumerate(models):
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                cell = by_model[model].sel(season=season, statistic=statistic)
                ok, message = _verdict(cell, diagnostic)
                if message == "no data":
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                    continue
                draw_member_strip(ax, cell.dropna("member", how="all"))
                _status_line(ax, 0.01, 0.99, ok, message, fontsize=7)
                ax.grid(axis="x", color="0.9", lw=0.6)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(season, loc="left", pad=13)
                if i == len(models) - 1:
                    ax.set_xlabel(diagnostic.units)
            axes[i, 0].set_ylabel(model, rotation=0, ha="right", va="center", fontsize=9, fontweight="bold")
        fig.legend(handles=[
            Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["inner"], mec="white", label="Each member"),
            Patch(color=ENSEMBLE["range"], label="Members' 5–95%"),
            Line2D([], [], color=ENSEMBLE["median"], lw=1, label="Member median"),
            Line2D([], [], color="#111111", lw=1.8, marker="D", ms=5, mec="white", label="ERA5"),
        ], loc="outside lower center", ncols=4, frameon=False)
        fig.suptitle(title or f"Sea-ice extent, {diagnostic.label.lower()}: ERA5 among each model's members",
                     fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 4. Where the ice changed: a field's mean in each zone
# ---------------------------------------------------------------------------

@plot("figure")
def zone_changes(changes, ylabel, experiments=None, seasons=None, title=None):
    """A change (e.g. of the mean temperature) averaged over each zone of ``sea_ice.ZONES``: experiments down, seasons across.

    Each marker is one model's mean over the zone, the bar the median of the
    models. Comparing "ice lost" with "open water" and "ice in both" says how
    much of the change goes with the ice that went.

    Args:
        changes (xr.DataArray): On (model, experiment, season, zone), from ``sea_ice.zone_means``.
        ylabel (str): What the change is, with its units.
        experiments, seasons (Sequence[str] | None): Rows and columns, in order; all by default.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    experiments = list(experiments if experiments is not None else changes["experiment"].values)
    seasons = list(seasons if seasons is not None else changes["season"].values)
    zones = [str(z) for z in changes["zone"].values]
    models = [str(m) for m in changes["model"].values]
    offsets = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else np.zeros(1)
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(len(experiments), len(seasons),
                                 figsize=(3.0 * len(seasons) + 0.8, 2.1 * len(experiments) + 1.2),
                                 sharex=True, sharey=True, layout="constrained", squeeze=False)
        for i, experiment in enumerate(experiments):
            colour = FORCING_COLORS.get(experiment, INK)
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                ax.axhline(0, color="0.6", lw=0.6, zorder=1)
                for x, zone in enumerate(zones):
                    cell = changes.sel(experiment=experiment, season=season, zone=zone)
                    if not bool(cell.notnull().any()):
                        continue
                    ax.bar(x, float(cell.median("model")), width=0.7, color=colour, alpha=0.3, lw=0, zorder=2)
                    for k, model in enumerate(models):
                        ax.plot(x + offsets[k], float(cell.sel(model=model)), ls="none",
                                marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4, color=colour, mec="white",
                                mew=0.4, zorder=3)
                if i == 0:
                    ax.set_title(season, loc="left")
                if j == 0:
                    ax.set_ylabel(f"{experiment}\n{ylabel}")
                ax.set_xticks(range(len(zones)), [zone.replace(" ", "\n", 1) for zone in zones], fontsize=7.5)
                core.style_ax(ax)
                ax.grid(axis="x", visible=False)
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=5, color="0.35",
                          label=model) for k, model in enumerate(models)]
        handles.append(Patch(color="0.6", alpha=0.5, label="Multi-model median"))
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 7), frameon=False)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)
