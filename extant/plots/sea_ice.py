"""Figures for the sea-ice edge: the edge on a map, the extent through time, and how the equivalent latitude is found.

Draws the output of ``sea_ice``. The edge is drawn the same way everywhere:
the 15% contour of the ensemble-mean concentration, solid for the experiment
and dashed for hist-nat, in ink so that it reads on red and blue alike.

Sections
--------
1. On a map: the 15% contour
2. Through time: each experiment's extent, every model
3. Method: from a concentration field to the equivalent latitude
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from .. import sea_ice
from .era5_evaluation import EVAL_RC, INK, INK_2, MUTED, OUTSIDE_COLOR
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER
from plotting_modules.utils import plot

#(t): Line styles of the ice edge on maps: the experiment's and hist-nat's
EDGE_STYLES = {"experiment": "solid", "hist-nat": "dashed"}

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
