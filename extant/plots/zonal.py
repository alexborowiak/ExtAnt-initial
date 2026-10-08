"""Zonal-mean figures: the forced change at each latitude, model by model, after Bracegirdle et al. (2024).

Draws the output of ``zonal`` (and ``sea_ice.edge_latitude`` for the ice
edge). The layout follows the paper's Figs 2, 6 and 8, but nothing is averaged
across models: each model has its own row (or its own line).

Reading them:
    columns     the mean; the low and the high extremes, both measured from the
                median; the width between them (= high - low)
    lines       experiments (FORCING_COLORS), or models (MODEL_COLORS)
    dots        the zonal-mean change is significant: the mean by the
                member-block test, the width by the hist-nat bootstrap; the
                extremes have no test, so no dots
    grey band   in the width column: the central 95% of the zonal-mean width
                changes hist-nat's own variability gives (one per model)
    dashed      the sea-ice edge as an equivalent latitude, in each
                experiment's final years (hist-nat's too, in green)

Sections
--------
1. The change at each latitude, every model and experiment
2. Where variability is largest: hist-nat's width and the ice edge
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import response_change as rc
from .era5_evaluation import EVAL_RC, INK, MUTED
from .response_change import _experiments
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS
from plotting_modules.utils import plot

#(t): One colour per model, in a fixed order (eight hues; the adjacent pairs are colour-blind safe)
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

#(t): Column titles, and which test's p-values give each column its dots
TITLES = {
    "mean_change": "Mean",
    "low_extreme_change": "Low extremes",
    "high_extreme_change": "High extremes",
    "width_change": "Width (high − low)",
}
PVALUES = {"mean_change": "mean_pvalue", "width_change": "width_pvalue"}

NULL_COLOR = "0.88"
EDGE_LINESTYLE = (0, (4, 2))


def _colors(dim, values):
    """A colour for each experiment (FORCING_COLORS) or model (MODEL_COLORS, by position)."""
    if dim == "experiment":
        return {v: FORCING_COLORS.get(v, INK) for v in values}
    return {v: MODEL_COLORS[k % len(MODEL_COLORS)] for k, v in enumerate(values)}


def _order(da, dim, values=None):
    if dim == "experiment":
        return _experiments(da, values)
    present = [str(v) for v in da[dim].values]
    return [v for v in (values or present) if v in present]


def _limits(arrays, pad=0.08):
    """Symmetric-enough y limits covering every array (and zero)."""
    values = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    values = values[np.isfinite(values)]
    low, high = min(values.min(initial=0), 0), max(values.max(initial=0), 0)
    span = (high - low) or 1.0
    return low - pad * span, high + pad * span


def _latitude_axis(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}°S" if v < 0 else f"{v:g}°"))


def _edge(edges, selection):
    """One equivalent latitude from ``edges``, or None where it has none."""
    if edges is None:
        return None
    try:
        value = float(edges.sel(selection))
    except (KeyError, ValueError):
        return None
    return value if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# 1. The change at each latitude, every model and experiment
# ---------------------------------------------------------------------------

@plot("figure")
def zonal_change_grid(zonal, season, rows="model", lines="experiment", row_values=None, line_values=None,
                      columns=tuple(rc.EXTREMES), edges=None, alpha=0.05, units="°C", title=None, panel_size=(3.0, 1.95),
                      rc_params=None, width=None):
    """The zonal-mean change in the mean, the low and high extremes and the width: one row per model.

    The form of Bracegirdle et al. (2024) Figs 2, 6 and 8, per model. The
    mean column has its own scale; the extremes and the width share one, so
    the width is visibly the high column minus the low.

    Args:
        zonal (xr.Dataset): ``zonal.zonal_summary`` output, on (model, experiment, season, lat).
        season (str): Season shown.
        rows, lines (str): "model" and "experiment", either way round.
        row_values, line_values (Sequence[str] | None): Subsets, in order; all by default.
        columns (Sequence[str]): Variables of ``zonal``, one per column.
        edges (xr.DataArray | None): ``sea_ice.edge_latitude`` output, on (model, experiment, season).
        alpha (float): Significance level of the dots.
        units (str): Units of the variable.
        title (str | None): Figure title; a default if None, none if "" (for a paper figure).
        panel_size (tuple[float, float]): Width and height of each panel, in inches.
        rc_params (dict | None): rcParams to draw with; ``EVAL_RC`` by default (``forced_response.PAPER.rc`` for print).
        width (float | None): The figure's width in inches, instead of ``panel_size[0]`` (e.g. 7.2 for a printed page).

    Returns:
        core.Panels
    """
    data = zonal.sel(season=season)
    row_values = _order(data, rows, row_values)
    line_values = _order(data, lines, line_values)
    colors = _colors(lines, line_values)
    lat = data["lat"].values
    with_band = rows == "model" and "null_lower" in data

    with plt.rc_context(rc_params or EVAL_RC):
        fig, axes = plt.subplots(len(row_values), len(columns),
                                 figsize=(width or panel_size[0] * len(columns) + 1.2,
                                          panel_size[1] * len(row_values) + 1.5),
                                 sharex=True, layout="constrained", squeeze=False)
        for i, row in enumerate(row_values):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
                if column == "width_change" and with_band:
                    ax.fill_between(lat, data["null_lower"].sel(model=row), data["null_upper"].sel(model=row),
                                    color=NULL_COLOR, lw=0, zorder=0)
                for line in line_values:
                    y = data[column].sel({rows: row, lines: line})
                    if not bool(y.notnull().any()):
                        continue
                    ax.plot(lat, y, color=colors[line], lw=1.6, zorder=3)
                    pvalue = PVALUES.get(column)
                    if pvalue in data:
                        significant = (data[pvalue].sel({rows: row, lines: line}) < alpha).values
                        ax.plot(lat[significant], y.values[significant], ls="none", marker="o", ms=4,
                                color=colors[line], mec="white", mew=0.6, zorder=4)

                #(t): The ice edge in each experiment's final years, hist-nat's included
                edge_lines = [*line_values, "hist-nat"] if lines == "experiment" else line_values
                for line in edge_lines:
                    edge = _edge(edges, {rows: row, lines: line, "season": season})
                    if edge is not None:
                        ax.axvline(edge, color=colors.get(line, FORCING_COLORS.get(line, INK)), lw=1.0,
                                   ls=EDGE_LINESTYLE, zorder=2)
                if i == 0:
                    ax.set_title(f"{TITLES.get(column, column)}\n{rc.EXTREMES.get(column, column)}", loc="left")
                if j == 0:
                    ax.set_ylabel(f"{row}\n({units})")
                if i == len(row_values) - 1:
                    ax.set_xlabel("Latitude")
                _latitude_axis(ax)
                core.style_ax(ax)

        #(t): One scale for the mean, one shared by the extremes and the width
        for group in ([c for c in columns if c == "mean_change"], [c for c in columns if c != "mean_change"]):
            if not group:
                continue
            arrays = [data[c].sel({rows: row_values, lines: line_values}) for c in group]
            if with_band and "width_change" in group:
                arrays += [data["null_lower"].sel(model=row_values), data["null_upper"].sel(model=row_values)]
            limits = _limits(arrays)
            for j, column in enumerate(columns):
                if column in group:
                    for ax in axes[:, j]:
                        ax.set_ylim(limits)
        axes[0, 0].set_xlim(lat.min(), lat.max())

        handles = [Line2D([], [], color=colors[v], lw=2, label=v) for v in line_values]
        if any(PVALUES.get(c) in data for c in columns):
            handles.append(Line2D([], [], ls="none", marker="o", ms=5, color=INK, mec="white",
                                  label=f"Significant (p < {alpha:g})"))
        if with_band and "width_change" in columns:
            handles.append(Patch(color=NULL_COLOR, label="hist-nat variability alone (95%)"))
        if edges is not None:
            handles.append(Line2D([], [], color=INK, lw=1.0, ls=EDGE_LINESTYLE, label="Sea-ice edge (equivalent latitude)"))
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 5), frameon=False)
        if title != "":
            fig.suptitle(title or f"Zonal-mean change, {season}: final years against hist-nat", fontsize=11,
                         fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 2. Where variability is largest: hist-nat's width and the ice edge
# ---------------------------------------------------------------------------

@plot("figure")
def zonal_reference_width(zonal, seasons=("DJF", "JJA"), edges=None, units="°C", title=None):
    """hist-nat's zonal-mean width (Q95 − Q05) by latitude, every model, with each model's hist-nat ice edge.

    Where a season varies most from year to year, and whether that is at the
    ice edge, as Bracegirdle et al. (2024) found for winter temperature.

    Args:
        zonal (xr.Dataset): ``zonal.zonal_summary`` output with ``hist_nat_width`` (from the width test).
        seasons (Sequence[str]): One panel each.
        edges (xr.DataArray | None): ``sea_ice.edge_latitude`` output.
        units (str): Units of the variable.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    width = zonal["hist_nat_width"]
    models = [str(m) for m in width["model"].values]
    colors = _colors("model", models)
    lat = width["lat"].values
    top = 1.05 * float(width.sel(season=list(seasons)).max())

    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(seasons), figsize=(4.2 * len(seasons) + 1.6, 3.6), sharey=True,
                                 layout="constrained", squeeze=False)
        for ax, season in zip(axes[0], seasons):
            for model in models:
                ax.plot(lat, width.sel(model=model, season=season), color=colors[model], lw=1.6)
                edge = _edge(edges, {"model": model, "experiment": "hist-nat", "season": season})
                if edge is not None:
                    ax.axvline(edge, color=colors[model], lw=1.0, ls=EDGE_LINESTYLE)
            ax.set_title(season, loc="left")
            ax.set_xlabel("Latitude")
            ax.set_xlim(lat.min(), lat.max())
            ax.set_ylim(0, top)
            _latitude_axis(ax)
            core.style_ax(ax)
        axes[0, 0].set_ylabel(f"hist-nat Q95 − Q05 ({units})")
        handles = [Line2D([], [], color=colors[m], lw=2, label=m) for m in models]
        if edges is not None:
            handles.append(Line2D([], [], color=INK, lw=1.0, ls=EDGE_LINESTYLE, label="Sea-ice edge"))
        fig.legend(handles=handles, loc="outside right center", frameon=False)
        fig.suptitle(title or "Year-to-year variability in hist-nat: zonal-mean width of each season",
                     fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)
