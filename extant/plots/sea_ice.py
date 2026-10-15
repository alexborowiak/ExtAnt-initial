"""Figures of the sea ice: the edge on a map, the extent through time, the models against ERA5, and the zones.

Draws the output of ``sea_ice``. The edge is drawn the same way everywhere:
the 15% contour of the ensemble-mean concentration, solid for the experiment
and dashed for hist-nat, in ink so that it reads on red and blue alike; ERA5's
in orange. Notebook 04's map grids draw it through their ``edges``
(``edge_pair``).

Sections
--------
1. On a map: the 15% contour
2. Through time: each experiment's extent, every model
3. Method: from a concentration field to the equivalent latitude, and onto the 2.5° grid
4. Against ERA5: the seasonal cycle, the edge's latitude, and ERA5 among the members
5. Where the ice changed: a field's mean in each zone
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import sea_ice
from .era5_evaluation import ENSEMBLE, EVAL_RC, INK, INK_2, MODEL_COLORS, MODEL_MARKERS, MUTED, OUTSIDE_COLOR, \
    STATUS_FLAG, draw_member_strip, where_era5
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER
from plotting_modules.utils import plot

#(t): Line styles of the ice edge on maps: the experiment's and hist-nat's
EDGE_STYLES = {"experiment": "solid", "hist-nat": "dashed"}

#(t): ERA5's colour on the maps and in the line plots, apart from the models' ink
ERA5_EDGE_COLOR = "#d95f02"

#(t): How each kind of edge is drawn on the map grids (``edges``), by its label: an experiment's and hist-nat's
#(t): (notebook 04), a model's and ERA5's (notebook 09); any other label is drawn as an experiment's
EDGES = {
    "experiment": dict(color=INK, linestyle="solid", linewidth=1.3, label="Sea-ice edge (15%): the experiment"),
    "hist-nat": dict(color=INK, linestyle="dashed", linewidth=1.0, label="Sea-ice edge: hist-nat"),
    "model": dict(color=INK, linestyle="solid", linewidth=1.3, label="Sea-ice edge (15%): the model"),
    "ERA5": dict(color=ERA5_EDGE_COLOR, linestyle="solid", linewidth=1.5, label="Sea-ice edge: ERA5"),
    "edge": dict(color=INK, linestyle="solid", linewidth=1.2, label="Sea-ice edge (15%)"),
}

#(t): Colours of the cells in the method figure: open water, ice, the continent
CELL_COLORS = ("white", "#9ec5f4", "#c9c8c3")


# ---------------------------------------------------------------------------
# 1. On a map: the 15% contour
# ---------------------------------------------------------------------------

@plot("axes")
def draw_ice_edge(ax, siconc, level=sea_ice.THRESHOLD, color=INK, linewidth=1.3, linestyle="solid"):
    """The ``level`` % contour of a concentration field on (lat, lon), on a polar map: the ice edge.

    Returns:
        matplotlib.contour.QuadContourSet
    """
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point

    field = siconc.transpose("lat", "lon")
    values, lon = add_cyclic_point(field.values, coord=field["lon"].values)
    return ax.contour(lon, field["lat"].values, values, levels=[level], colors=[color], linewidths=linewidth,
                      linestyles=linestyle, transform=ccrs.PlateCarree())


def draw_ice_edges(ax, ice, experiment, season):
    """hist-nat's edge (dashed) and ``experiment``'s (solid) on one map; nothing if ``ice`` lacks either.

    Args:
        ice (xr.DataArray | None): One model's ``sea_ice.concentration_climatology``, on (experiment, season, lat, lon).
    """
    if ice is None:
        return
    for name, style in (("hist-nat", EDGE_STYLES["hist-nat"]), (experiment, EDGE_STYLES["experiment"])):
        if name in ice["experiment"].values and season in ice["season"].values:
            draw_ice_edge(ax, ice.sel(experiment=name, season=season), linestyle=style,
                          linewidth=1.1 if name == "hist-nat" else 1.4)
        if name == experiment == "hist-nat":
            break


def ice_edge_handles():
    """Legend entries for ``draw_ice_edges``."""
    return [Line2D([], [], color=INK, lw=1.4, ls=EDGE_STYLES["experiment"], label="Sea-ice edge (15%), the experiment"),
            Line2D([], [], color=INK, lw=1.1, ls=EDGE_STYLES["hist-nat"], label="Sea-ice edge, hist-nat")]


def edge_pair(ice, experiments):
    """hist-nat's concentration and an experiment's (or several), side by side on an ``edge`` dim, for map grids.

    The ``edges`` of ``forced_response.map_grid``, ``row_grid`` (``Row``) and
    ``count_grid`` draw the 15% contour of each: hist-nat's dashed, the
    experiment's solid. An experiment ``ice`` lacks is left empty, so its
    edge is not drawn.

    Args:
        ice (xr.DataArray | None): Concentration with an ``experiment`` dim, e.g.
            ``sea_ice.concentration_climatology``, one model's, or the models' median.
        experiments (str | Sequence[str]): One experiment (the result has no ``experiment`` dim), or several
            (it keeps one, and hist-nat's edge is the same in each).

    Returns:
        xr.DataArray | None: On (edge, ..., lat, lon), ``edge`` being 'hist-nat' and 'experiment'; None if
        ``ice`` is None.
    """
    if ice is None:
        return None
    one = isinstance(experiments, str)
    experiment = ice.reindex(experiment=[experiments] if one else list(experiments))
    if one:
        experiment = experiment.squeeze("experiment", drop=True)
    hist_nat = ice.reindex(experiment=["hist-nat"]).squeeze("experiment", drop=True)
    hist_nat, experiment = xr.broadcast(hist_nat, experiment)
    return xr.concat([hist_nat, experiment], dim=pd.Index(["hist-nat", "experiment"], name="edge"))


def draw_edges(ax, edges, scale=1.0):
    """The 15% contour of each field of ``edges`` (on edge, lat, lon), styled by its label (``EDGES``).

    Args:
        ax: A polar map.
        edges (xr.DataArray): Concentration (%) on (edge, lat, lon).
        scale (float): Line widths are multiplied by this (thinner in print).
    """
    for label in edges["edge"].values:
        field = edges.sel(edge=label)
        if not bool(field.notnull().any()):
            continue
        style = EDGES.get(str(label), EDGES["experiment"])
        draw_ice_edge(ax, field, color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"] * scale)


def edge_handles(labels):
    """Legend entries for the edges labelled ``labels``, as ``draw_edges`` draws them."""
    handles = []
    for label in labels:
        style = EDGES.get(str(label), EDGES["experiment"])
        handles.append(Line2D([], [], color=style["color"], ls=style["linestyle"], lw=style["linewidth"],
                              label=style["label"] if str(label) in EDGES else f"Sea-ice edge: {label}"))
    return handles


# ---------------------------------------------------------------------------
# 2. Through time: each experiment's extent, every model
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
# 3. Method: from a concentration field to the equivalent latitude
# ---------------------------------------------------------------------------

@plot("figure")
def edge_schematic(example, title=None):
    """How the equivalent latitude of the ice edge is found, step by step, on one concentration field.

    a) the concentration, with its 15% contour;
    b) the cells that count: ice (at least 15%) and the continent (land south
       of 60°S), whose areas add up, and the circle of the equivalent latitude,
       which encloses the same area;
    c) the area south of each latitude, A(φ) = 2πR²(1 + sin φ), read backwards
       from that total.

    Args:
        example (xr.Dataset): ``sea_ice.edge_example`` output.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    import cartopy.crs as ccrs
    from plotting_modules import maps  # needs cartopy

    million = 1e6
    sie, continent = float(example["extent"]) / million, float(example["continent_area"]) / million
    edge = float(example["edge_latitude"])
    threshold = example.attrs.get("threshold", sea_ice.THRESHOLD)

    with plt.rc_context(EVAL_RC):
        fig = plt.figure(figsize=(13.5, 5.0), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
        ax_a = fig.add_subplot(grid[0], projection=ccrs.SouthPolarStereo())
        ax_b = fig.add_subplot(grid[1], projection=ccrs.SouthPolarStereo())
        ax_c = fig.add_subplot(grid[2])

        #(t): a) the concentration and its 15% contour
        mappable = maps.draw_polar_contour(ax_a, example["siconc"], levels=np.arange(0, 101, 10), cmap="Blues")
        draw_ice_edge(ax_a, example["siconc"], level=threshold)
        fig.colorbar(mappable, ax=ax_a, orientation="horizontal", shrink=0.75, pad=0.04, label="Concentration (%)")
        ax_a.set_title(f"a) Concentration, and its {threshold:g}% contour", loc="left")

        #(t): b) the cells counted, and the circle enclosing the same area
        cells = (example["ice"].astype(int) + 2 * example["continent"].astype(int)).clip(max=2)
        maps.draw_polar_contour(ax_b, cells, levels=np.arange(-0.5, 3), cmap=ListedColormap(CELL_COLORS),
                                discrete=True)
        ax_b.plot(np.linspace(-180, 180, 361), np.full(361, edge), color=OUTSIDE_COLOR, lw=2,
                  transform=ccrs.PlateCarree())
        ax_b.set_title(f"b) Ice ≥ {threshold:g}% plus the continent", loc="left")
        ax_b.legend(handles=[
            Line2D([], [], ls="none", marker="s", ms=9, color=CELL_COLORS[1], label=f"Ice: {sie:.1f} million km²"),
            Line2D([], [], ls="none", marker="s", ms=9, color=CELL_COLORS[2],
                   label=f"Continent: {continent:.1f} million km²"),
            Line2D([], [], color=OUTSIDE_COLOR, lw=2, label=f"Same area, as a cap: {abs(edge):.1f}°S"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

        #(t): c) the area south of each latitude, read backwards
        phi = np.linspace(-90, -40, 201)
        ax_c.plot(phi, sea_ice.cap_area(phi) / million, color=INK, lw=1.6)
        total = sie + continent
        ax_c.annotate("", xy=(edge, 0), xytext=(edge, total),
                      arrowprops=dict(arrowstyle="-|>", color=OUTSIDE_COLOR, lw=1.4, shrinkA=0, shrinkB=0))
        ax_c.plot([-90, edge], [total, total], color=OUTSIDE_COLOR, lw=1.4)
        ax_c.axhline(continent, color=MUTED, lw=0.9)
        ax_c.text(-89, continent, " continent alone", va="bottom", fontsize=8, color=INK_2)
        ax_c.text(edge + 0.6, total * 0.45, f"φ = {abs(edge):.1f}°S", color=INK, fontsize=9)
        ax_c.text(-89, total, f" {sie:.1f} + {continent:.1f} = {total:.1f}", va="bottom", fontsize=8, color=INK)
        ax_c.set_xlim(-90, -40)
        ax_c.set_ylim(0, None)
        ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}°S"))
        ax_c.set_xlabel("Latitude φ")
        ax_c.set_ylabel("Area south of φ (million km²)")
        ax_c.set_title("c) A(φ) = 2πR²(1 + sin φ), read backwards", loc="left")
        core.style_ax(ax_c)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=np.array([[ax_a, ax_b, ax_c]], dtype=object))



@plot("figure")
def regrid_schematic(native, lat, lon, on_grid, region=((-72.5, -52.5), (0.0, 40.0)), title=None):
    """How a field goes onto the 2.5° grid (``sea_ice.to_grid``), on one field, step by step.

    a) the field on its own grid: each point, coloured by its value, with the
       2.5° cells drawn over them;
    b) each 2.5° cell, coloured by the area-weighted mean of the points in
       it, with the points as dots and their number in the cell;
    c) the whole field on the 2.5° grid, the region of a) and b) outlined.

    Args:
        native (xr.DataArray): Concentration (%) on the field's own grid, e.g. (j, i).
        lat, lon (xr.DataArray): Its points' latitudes and longitudes, on the same dims.
        on_grid (xr.DataArray): ``sea_ice.to_grid`` of it, on (lat, lon).
        region (tuple): ((south, north), (west, east)) of a) and b), in degrees.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    import cartopy.crs as ccrs
    from plotting_modules import maps  # needs cartopy

    (south, north), (west, east) = region
    native, lat, lon = xr.align(native.reset_coords(drop=True), lat.reset_coords(drop=True),
                                lon.reset_coords(drop=True), join="inner")
    lat_step = float(on_grid["lat"].values[1] - on_grid["lat"].values[0])
    lon_step = float(on_grid["lon"].values[1] - on_grid["lon"].values[0])
    lat, lon = xr.broadcast(lat, lon)
    points_lat = lat.values.ravel()
    points_lon = (lon.values.ravel() + 180) % 360 - 180
    values = native.transpose(*lat.dims).values.ravel()
    near = (points_lat >= south) & (points_lat <= north) & (points_lon >= west) & (points_lon <= east)
    first_lat_edge = float(on_grid["lat"].values[0]) - lat_step / 2
    first_lon_edge = float(on_grid["lon"].values[0]) - lon_step / 2
    lat_edges = first_lat_edge + lat_step * np.arange(np.floor((south - first_lat_edge) / lat_step),
                                                      np.ceil((north - first_lat_edge) / lat_step) + 1)
    lon_edges = first_lon_edge + lon_step * np.arange(np.floor((west - first_lon_edge) / lon_step),
                                                      np.ceil((east - first_lon_edge) / lon_step) + 1)
    levels = np.arange(0, 101, 10)
    cmap = plt.get_cmap("Blues").with_extremes(bad="#c9c8c3")
    norm = BoundaryNorm(levels, cmap.N)

    with plt.rc_context(EVAL_RC):
        fig = plt.figure(figsize=(14.0, 5.2), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=(1.25, 1.25, 1))
        ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        ax_c = fig.add_subplot(grid[2], projection=ccrs.SouthPolarStereo())

        #(t): a) the points on the field's own grid, and the 2.5° cells over them
        ax_a.scatter(points_lon[near], points_lat[near], c=values[near], cmap=cmap,
                     norm=norm, s=22, edgecolor="0.5", linewidth=0.2, plotnonfinite=True)
        ax_a.set_title("a) The field on its own grid: one dot per point", loc="left")

        #(t): b) the 2.5° cells, each the mean of its points, with how many points went into each
        cells = on_grid.sel(lat=slice(south - lat_step, north + lat_step), lon=slice(west - lon_step, east + lon_step))
        mesh = ax_b.pcolormesh(cells["lon"].values, cells["lat"].values, np.ma.masked_invalid(cells.values),
                               cmap=cmap, norm=norm, shading="nearest")
        ax_b.scatter(points_lon[near], points_lat[near], s=3, color=INK, alpha=0.5, lw=0)
        rows = np.floor((points_lat[near] - first_lat_edge) / lat_step)
        columns = np.floor((points_lon[near] - first_lon_edge) / lon_step)
        counts = pd.Series(1, index=pd.MultiIndex.from_arrays([rows, columns])).groupby(level=[0, 1]).sum()
        for (row, column), count in counts.items():
            y = first_lat_edge + (row + 0.5) * lat_step
            x = first_lon_edge + (column + 0.5) * lon_step
            if south <= y <= north and west <= x <= east:
                ax_b.text(x, y, str(count), ha="center", va="center", fontsize=6.5, color=INK_2)
        ax_b.set_title("b) Each 2.5° cell: the area-weighted mean of its points", loc="left")

        for ax in (ax_a, ax_b):
            for edge in lat_edges:
                ax.axhline(edge, color=OUTSIDE_COLOR, lw=0.7)
            for edge in lon_edges:
                ax.axvline(edge, color=OUTSIDE_COLOR, lw=0.7)
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
        fig.colorbar(mesh, ax=[ax_a, ax_b], orientation="horizontal", shrink=0.6, pad=0.02,
                     label="Concentration (%); grey: land")

        #(t): c) the whole field on the 2.5° grid
        maps.draw_polar_contour(ax_c, on_grid, levels, "Blues", extend="neither")
        box_lon = np.r_[np.linspace(west, east, 30), np.full(30, east), np.linspace(east, west, 30), np.full(30, west)]
        box_lat = np.r_[np.full(30, south), np.linspace(south, north, 30), np.full(30, north),
                        np.linspace(north, south, 30)]
        ax_c.plot(box_lon, box_lat, color=OUTSIDE_COLOR, lw=1.4, transform=ccrs.PlateCarree())
        ax_c.set_title("c) The whole field on the 2.5° grid", loc="left")
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=np.array([[ax_a, ax_b, ax_c]], dtype=object))


# ---------------------------------------------------------------------------
# 4. Against ERA5: the seasonal cycle, the edge's latitude, and ERA5 among the members
# ---------------------------------------------------------------------------

MONTH_LETTERS = "JFMAMJJASOND"


@plot("figure")
def seasonal_cycle(cycle, era5, title=None):
    """Each model's seasonal cycle of extent against ERA5's: one panel per model, every member a line.

    Args:
        cycle (xr.DataArray): Extent (million km²) on (model, member, month), e.g. the historical
            members' mean over the same years as ERA5.
        era5 (xr.DataArray): ERA5's, on (month).
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    models = [str(m) for m in cycle["model"].values]
    months = cycle["month"].values
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 0.6, 3.4), sharey=True,
                                 layout="constrained", squeeze=False)
        for ax, model in zip(axes[0], models):
            members = cycle.sel(model=model).dropna("member", how="all")
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
        members (xr.DataArray): Degrees on (model, member, season): each member's edge over the years compared.
        era5 (xr.DataArray): ERA5's, on (season).
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    models = [str(m) for m in members["model"].values]
    seasons = [str(s) for s in era5["season"].values]
    offsets = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else np.zeros(1)
    with plt.rc_context(EVAL_RC):
        fig, ax = plt.subplots(figsize=(1.9 * len(seasons) + 1.6, 4.0), layout="constrained")
        for x, season in enumerate(seasons):
            ax.plot([x - 0.42, x + 0.42], [float(era5.sel(season=season))] * 2, color=ERA5_EDGE_COLOR, lw=2.2,
                    zorder=1)
            for k, model in enumerate(models):
                values = members.sel(model=model, season=season).dropna("member").values
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
        result (xr.Dataset): ``era5_evaluation.locate`` results for every model, on (model, statistic, season,
            member), as notebook 09 (section 4.4) makes them.
        statistic (str): Which statistic.
        diagnostics (dict): ``sea_ice.ERA5_DIAGNOSTICS``.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    from .era5_evaluation import _status_line

    diagnostic = diagnostics[statistic]
    models = [str(m) for m in result["model"].values]
    seasons = [str(s) for s in result["season"].values]
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(len(models), len(seasons),
                                 figsize=(2.5 * len(seasons) + 1.2, 1.15 * len(models) + 1.0),
                                 layout="constrained", squeeze=False)
        for i, model in enumerate(models):
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                cell = result.sel(model=model, season=season, statistic=statistic)
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
# 5. Where the ice changed: a field's mean in each zone
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
