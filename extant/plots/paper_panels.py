"""The paper's candidate panels (notebook 07), each drawn into a figure or into part of one, so they can be assembled.

Every function here draws one panel, or one block of panels, into ``fig``: a
whole matplotlib Figure, or a SubFigure of a bigger one. That is what lets the
same panel go into several layouts (``compose``), as the candidate layouts
L1-L5 need. Given no ``fig``, a function makes its own, a page wide. Each
returns ``core.Panels`` with ``extras["next_tag"]``, the next panel letter, so
that blocks put together letter their panels a), b), c) ... in order.

The panels, and the candidates of the handover notes they draw:

    map_block          any grid of polar maps on one colour scale, hatched where masks are True, with a
                       note in each panel (the number of models) and lines on top (the change in sea ice):
                       A1-A4, B1, B2, C3, E2, E4, F2, and the verification maps
    plumes             A5: each experiment's spread through time in a region, and ERA5
    quantile_curves    B3, D3: the change at every percentile, one line per forcing
    pr_points          C1, C2: probability ratios with their bootstrap ranges, every model, and the
                       ratios the change in the mean alone would give
    scatter_by_season  D2, D4, E3, E5: one change against another, one point per model and forcing
    forcing_bars       E1, F1: a change per forcing, the multi-model median and every model
    ridgelines         R1, R2: whole distributions as overlapping ridges, one per forcing (or model), coloured
                       by their mean
    rank_histograms    verification: ERA5's regional rank histograms against the perfect-model range
    percentile_tables  verification: ERA5's percentile among each experiment's members
    compose            the layouts: blocks stacked in rows, side by side within a row

Colours and patterns are those of ``forced_response``: red and blue for up and
down, each forcing's colour from plotting_modules, and the IPCC AR6 hatching of
the agreement maps (``AGREEMENT``). Nothing here computes a result; the
notebook makes every array a panel is given.
"""

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS
from plotting_modules.utils import plot
from . import forced_response as fr
from .era5_evaluation import INK, MODEL_COLORS, MODEL_MARKERS, PERCENTILE_EDGES, PSEUDO_COLOR, RANK_BAR, \
    percentile_colormap
from .response_change import HATCH_COLOR, round_ticks

# ---------------------------------------------------------------------------
# 1. Scales, hatches and figures
# ---------------------------------------------------------------------------

#(t): The IPCC AR6 advanced approach: no overlay where the change is robust, diagonal lines where fewer than
#(t): two-thirds of the models are significant, crossed lines where they are but fewer than 80% agree on the sign
AGREEMENT = {"no_signal": ("////", "No robust change (fewer than 2/3 of the models significant)"),
             "conflicting": ("xxxx", "Conflicting (2/3 significant, fewer than 80% agree on the sign)")}

#(t): Dots on a single model's map: not significant
NOT_SIGNIFICANT = ("....", "Not significant")

#(t): Colours of what is not a forcing (ERA5 is orange here, as historical is already black)
OTHER_COLORS = {"sum of parts": "#c49a00", "residual": "#c9c8c4", "ERA5": "#d95f02"}


@dataclass
class Hatch:
    """Where a block of maps is hatched: a boolean mask on the dims of the fields, its pattern, and its legend label."""

    mask: xr.DataArray
    pattern: str
    label: str


@dataclass
class Contours:
    """Lines over a block of maps: a field on the dims of the fields, the levels, and a legend label.

    Negative levels are dashed. For example the multi-model change in sea-ice concentration, at ±5 and ±15 %.
    """

    field: xr.DataArray
    levels: tuple
    label: str
    color: str = "#333333"
    linewidth: float = 0.6


def ratio_scale(label="Probability ratio", end=3, step=0.5):
    """A probability ratio on a log scale: the field is log2(PR), the ticks say the ratios (1/8 ... 8).

    Orange is more likely, purple less likely, whichever tail; ratios beyond 2^end take the end colours.
    """
    levels = np.round(np.arange(-end, end + step / 2, step), 10)
    ticks = np.arange(-end, end + 1)
    return fr.Scale(levels, "PuOr_r", label, ticks=ticks,
                    ticklabels=[f"1/{2 ** -t:g}" if t < 0 else f"{2 ** t:g}" for t in ticks])


def percentile_scale(label="ERA5's percentile among the members"):
    """ERA5's percentile in the five classes of the verification figures: beyond 5-95%, 5-25%, central, 75-95%."""
    cmap, norm = percentile_colormap()
    return fr.Scale(np.asarray(PERCENTILE_EDGES), cmap, label, norm=norm, ticks=np.array([0, 5, 25, 75, 95, 100]),
                    extend="neither")


def class_scale(names, colors, label="", wrap=14):
    """A discrete scale of named classes 0, 1, 2 ..., e.g. the diagnoses of the spatial evaluation.

    Each name is broken onto lines of at most ``wrap`` characters, so six of them fit under a page-wide bar.
    """
    return fr.Scale(np.arange(-0.5, len(names)), ListedColormap(list(colors)), label, discrete=True,
                    ticklabels=[textwrap.fill(str(name), wrap) for name in names])


def new_figure(width=fr.PAGE_WIDTH, height=4.0, style=fr.PAPER):
    """An empty figure in ``style``, with constrained layout, for one panel or a layout."""
    with plt.rc_context(style.rc):
        return plt.figure(figsize=(width, height), layout="constrained")


def _target(fig, width, height, style):
    return new_figure(width, height, style) if fig is None else fig


def _done(fig, axes, next_tag, **extras):
    return core.Panels(fig=fig, axes=axes, extras={"next_tag": next_tag, **extras})


def _tag(axes, tag_start, style):
    """Letter the panels from ``tag_start``; return the next letter (unchanged if not lettering)."""
    if tag_start is None:
        return None
    return core.tag_panels(axes, start=tag_start, fontsize=style.tag_size, fontweight="bold")


def _colorbar(fig, mappable, axes, scale, style, shrink=0.8):
    """One horizontal colour bar under ``axes``, ticked as its Scale says."""
    kwargs = dict(ax=list(np.ravel(axes)), location="bottom", shrink=shrink, aspect=45, pad=0.015)
    if scale.discrete:
        edges = np.asarray(scale.levels, dtype=float)
        ticks = (edges[:-1] + edges[1:]) / 2
        bar = fig.colorbar(mappable, boundaries=edges, ticks=ticks, spacing="uniform", **kwargs)
        bar.set_ticks(ticks, labels=scale.ticklabels or [f"{t:g}" for t in ticks])
        bar.ax.tick_params(length=0)
    else:
        n_columns = np.atleast_2d(axes).shape[1]
        ticks = scale.ticks if scale.ticks is not None else round_ticks(scale.levels, min(4 * n_columns, 11))
        bar = fig.colorbar(mappable, ticks=ticks, **kwargs)
        if scale.ticklabels is not None:
            bar.set_ticks(ticks, labels=scale.ticklabels)
    bar.ax.tick_params(labelsize=style.tick_size)
    bar.set_label(scale.label, fontsize=style.cbar_label_size)
    return bar


def _legend(fig, handles, style, ncol=None):
    """A legend under everything in ``fig``."""
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol or min(len(handles), 4), frameon=False,
                   fontsize=style.cbar_label_size, handlelength=2.2, handleheight=1.1)


# ---------------------------------------------------------------------------
# 2. Maps
# ---------------------------------------------------------------------------

def _contour_lines(ax, field, contours, style):
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point

    field = field.transpose("lat", "lon")
    if not bool(field.notnull().any()):
        return
    values, lons = add_cyclic_point(field.values, coord=field["lon"].values)
    ax.contour(lons, field["lat"].values, values, levels=list(contours.levels), colors=contours.color,
               linewidths=contours.linewidth, negative_linestyles="dashed", transform=ccrs.PlateCarree())


@plot("figure")
def map_block(fields, scale, row_dim=None, col_dim="season", rows=None, columns=None, row_labels=None,
              col_labels=None, hatches=(), notes=None, note_format="n = {:g}", contours=None, title=None, fig=None,
              style=fr.PAPER, tag_start=None, legend=True, width=fr.PAGE_WIDTH):
    """A grid of polar maps on one colour scale, with one colour bar under it.

    The maps of every candidate are drawn by this: the multi-model agreement
    maps (A1, B1, C3: the median, hatched by ``AGREEMENT``), every model's
    maps (A3, B2: dotted where not significant), the split of the mean (E2),
    the verification maps, and so on.

    Args:
        fields (xr.DataArray): On (row_dim, col_dim, lat, lon), or (col_dim, lat, lon) if ``row_dim`` is None.
        scale (fr.Scale): The colour scale.
        row_dim, col_dim (str | None): The dims of the rows and the columns.
        rows, columns (Sequence | None): Their values, in order; all of them by default.
        row_labels, col_labels (dict | None): The label of each value; the value itself by default.
        hatches (Sequence[Hatch]): Hatched where each mask is True (a mask may lack dims, which then apply to all).
        notes (xr.DataArray | None): A number for each panel, on (row_dim, col_dim), written in its corner with
            ``note_format``, e.g. the number of models.
        note_format (str): Format of a note.
        contours (Contours | None): Lines over every map.
        title (str | None): A title over the block, on the left (for a block among others in a layout).
        fig (Figure | SubFigure | None): Where to draw; a new figure if None.
        style (fr.Style): Fonts and line widths.
        tag_start (int | None): Letter the panels from this letter (0 is a); no letters if None.
        legend (bool): Explain the hatches and lines under the block.
        width (float): Width of a new figure, in inches.

    Returns:
        core.Panels: ``axes`` (rows, columns); ``extras``: ``cbar``, ``next_tag``.
    """
    import cartopy.crs as ccrs
    from plotting_modules import maps

    row_values = [None] if row_dim is None else list(rows if rows is not None else fields[row_dim].values)
    col_values = list(columns if columns is not None else fields[col_dim].values)
    n_rows, n_cols = len(row_values), len(col_values)
    height = n_rows * width / n_cols + 0.9

    with plt.rc_context(style.rc):
        fig = _target(fig, width, height, style)
        grid = fig.add_gridspec(n_rows, n_cols)
        axes = np.empty((n_rows, n_cols), dtype=object)
        drawn = None
        for i, row in enumerate(row_values):
            for j, column in enumerate(col_values):
                where = {col_dim: column} if row is None else {row_dim: row, col_dim: column}
                ax = fig.add_subplot(grid[i, j], projection=ccrs.SouthPolarStereo())
                axes[i, j] = ax
                field = fields.sel(where)
                if not bool(field.notnull().any()):
                    maps.setup_polar_ax(ax)
                else:
                    artist = maps.draw_polar_contour(ax, field, scale.levels, scale.cmap, scale.norm,
                                                     discrete=scale.discrete, extend=scale.extend)
                    drawn = drawn or artist
                for hatch in hatches:
                    mask = hatch.mask.sel({k: v for k, v in where.items() if k in hatch.mask.dims})
                    if bool(mask.any()):
                        maps.plot_hatch(ax, mask, hatch=hatch.pattern, color=HATCH_COLOR, linewidth=style.hatch_width)
                if contours is not None:
                    _contour_lines(ax, contours.field.sel({k: v for k, v in where.items()
                                                           if k in contours.field.dims}), contours, style)
                if notes is not None:
                    value = notes.sel({k: v for k, v in where.items() if k in notes.dims})
                    ax.text(1.0, 0.0, note_format.format(float(value)), transform=ax.transAxes, ha="right",
                            va="bottom", fontsize=style.tick_size, color="0.3")
                if i == 0:
                    ax.set_title(str((col_labels or {}).get(column, column)), fontsize=style.label_size, pad=3)
                if j == 0 and row is not None:
                    ax.text(-0.07, 0.5, str((row_labels or {}).get(row, row)), transform=ax.transAxes,
                            rotation=90, ha="right", va="center", fontsize=style.label_size)
        bar = _colorbar(fig, drawn, axes, scale, style) if drawn is not None else None
        handles = [Patch(facecolor="white", edgecolor="0.35", hatch=hatch.pattern, label=hatch.label)
                   for hatch in hatches if bool(hatch.mask.any())]
        if contours is not None:
            handles.append(Line2D([], [], color=contours.color, lw=1, label=contours.label))
        if legend:
            _legend(fig, handles, style, ncol=min(len(handles), 2))
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=style.label_size, fontweight="bold")
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag, cbar=bar)


# ---------------------------------------------------------------------------
# 3. Lines and points
# ---------------------------------------------------------------------------

def _forcing_color(name):
    return FORCING_COLORS.get(name, OTHER_COLORS.get(name, INK))


def _forcing_legend(fig, names, style, extra=()):
    handles = [Line2D([], [], color=_forcing_color(name), lw=2, label=name) for name in names]
    _legend(fig, [*handles, *extra], style, ncol=min(len(handles) + len(extra), 6))


@plot("figure")
def plumes(plume_ds, era5=None, experiments=None, bands=("hist-nat", "historical"), columns=None, col_dim="season",
           pairs=((0.05, 0.95),), units="°C", start_year=None, fig=None, style=fr.PAPER, tag_start=None,
           width=fr.PAGE_WIDTH):
    """A5: each experiment's median through time in one region, the spread of some, one panel per season, and ERA5.

    Five overlapping bands cannot be told apart, so only the experiments in
    ``bands`` get theirs; the rest are their median lines.

    Args:
        plume_ds (xr.DataArray): Quantiles on (experiment, quantile, year, col_dim), e.g. every member of
            every model pooled, as anomalies.
        era5 (xr.DataArray | None): ERA5 on (year, col_dim), the same anomalies.
        experiments (Sequence[str] | None): Drawn in this order (the last on top); all by default.
        bands (Sequence[str]): The experiments drawn with their spread.
        columns (Sequence | None): Values of ``col_dim``, in order.
        col_dim (str): The dim of the panels.
        pairs (Sequence[tuple]): (lower, upper) quantile pairs of the bands, widest first.
        units (str): Units of the anomalies.
        start_year (int | None): First year shown.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    from plotting_modules import timeseries

    experiments = list(experiments if experiments is not None else plume_ds["experiment"].values)
    columns = list(columns if columns is not None else plume_ds[col_dim].values)
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 2.2, style)
        axes = np.atleast_2d(fig.subplots(1, len(columns), sharey=True, squeeze=False))
        for ax, column in zip(axes[0], columns):
            for experiment in experiments:
                band = plume_ds.sel({"experiment": experiment, col_dim: column})
                if start_year is not None:
                    band = band.sel(year=slice(start_year, None))
                timeseries.draw_plume(ax, band, pairs if experiment in bands else (), color=_forcing_color(experiment),
                                      alphas=0.2, lw=1.0)
            if era5 is not None:
                series = era5.sel({col_dim: column})
                ax.plot(series["year"], series, color=OTHER_COLORS["ERA5"], lw=0.9, marker="o", ms=1.6, zorder=5)
            ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            ax.set_title(str(column), loc="left", fontsize=style.label_size)
            core.style_ax(ax)
        axes[0, 0].set_ylabel(f"Anomaly ({units})")
        extra = [Line2D([], [], color=OTHER_COLORS["ERA5"], lw=1.2, marker="o", ms=2.5, label="ERA5")]             if era5 is not None else []
        extra.append(Patch(color="0.6", alpha=0.4, label=f"{int(100 * pairs[0][0])}–{int(100 * pairs[0][1])}%"))
        _forcing_legend(fig, experiments, style, extra)
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


@plot("figure")
def quantile_curves(curves, experiments=None, rows=None, row_dim="region", columns=None, col_dim="season",
                    units="°C", fig=None, style=fr.PAPER, tag_start=None, width=fr.PAGE_WIDTH):
    """B3, D3: the change at every percentile, one line per forcing (the multi-model median, the models' range shaded).

    A flat line is a shift of the whole distribution. A line falling to the
    right means the cold tail warms more than the warm tail: the distribution
    narrows, as expected where sea ice is lost.

    Args:
        curves (xr.DataArray): On (model, experiment, quantile, row_dim, col_dim), e.g. notebook 03's
            ``regional_quantile_curve``.
        experiments (Sequence[str] | None): The forcings drawn.
        rows, columns (Sequence | None): Values of ``row_dim`` and ``col_dim``, in order (a ``row_dim`` that
            ``curves`` lacks gives one row).
        units (str): Units of the change.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    experiments = list(experiments if experiments is not None else curves["experiment"].values)
    row_values = [None] if row_dim not in curves.dims else list(rows if rows is not None else curves[row_dim].values)
    columns = list(columns if columns is not None else curves[col_dim].values)
    percentile = 100 * curves["quantile"].values
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 1.7 * len(row_values) + 0.6, style)
        axes = fig.subplots(len(row_values), len(columns), sharex=True, sharey="row", squeeze=False)
        for i, row in enumerate(row_values):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                where = {col_dim: column} if row is None else {row_dim: row, col_dim: column}
                ax.axhline(0, color="0.6", lw=0.6, zorder=0)
                for experiment in experiments:
                    models = curves.sel(where).sel(experiment=experiment)
                    color = _forcing_color(experiment)
                    ax.fill_between(percentile, models.min("model"), models.max("model"), color=color, alpha=0.15,
                                    lw=0)
                    ax.plot(percentile, models.median("model"), color=color, lw=1.4)
                if i == 0:
                    ax.set_title(str(column), loc="left", fontsize=style.label_size)
                if j == 0:
                    prefix = "" if row is None else f"{row}\n"
                    ax.set_ylabel(f"{prefix}Δ ({units})")
                if i == len(row_values) - 1:
                    ax.set_xlabel("Percentile")
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                core.style_ax(ax)
        _forcing_legend(fig, experiments, style, [Patch(color="0.6", alpha=0.3, label="Range of the models")])
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


#(t): The y range of the probability-ratio panels, as log2: a ratio of 0 is drawn at the bottom
RATIO_LIMITS = (-6, 6)


def _log_ratio(values):
    """log2 of probability ratios, clipped to ``RATIO_LIMITS`` so that 0 and infinity are drawn at the ends."""
    return np.clip(np.log2(np.clip(np.asarray(values, dtype=float), 2.0 ** -10, 2.0 ** 10)), *RATIO_LIMITS)


@plot("figure")
def pr_points(ratios, experiments=None, rows=("warm", "cold"), row_dim="tail", columns=None, col_dim="season",
              shift_only=False, fig=None, style=fr.PAPER, tag_start=None, width=fr.PAGE_WIDTH):
    """C1, C2: probability ratios, forcings along x, every model with its bootstrap range, and their median.

    Each model is a marker (its shape) in the forcing's colour with a line
    from its range's lower end to its upper end, spread a little sideways; the
    black bar is the multi-model median. With ``shift_only``, an open marker
    beside each gives the shift-only ratio, the ratio the shift of the mean
    alone would give: the gap between the two is what the change in width
    (and shape) adds. The y
    axis is logarithmic; a ratio of 0 sits on the bottom edge.

    Args:
        ratios (xr.Dataset): ``pr``, ``pr_lower``, ``pr_upper`` (and ``pr_shift_only`` for ``shift_only``) on
            (model, experiment, row_dim, col_dim), e.g. one region and chance of notebook 03's
            ``probability_ratio_regional`` (merged with ``probability_ratio_shift_only_regional``).
        experiments (Sequence[str] | None): Forcings along x.
        rows, columns (Sequence | None): Values of ``row_dim`` and ``col_dim``.
        shift_only (bool): Also draw the shift-only ratios (C2).
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    experiments = list(experiments if experiments is not None else ratios["experiment"].values)
    columns = list(columns if columns is not None else ratios[col_dim].values)
    models = [str(m) for m in ratios["model"].values]
    offsets = np.linspace(-0.28, 0.28, len(models)) if len(models) > 1 else np.zeros(1)
    ticks = np.arange(RATIO_LIMITS[0], RATIO_LIMITS[1] + 1, 2)
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 1.7 * len(rows) + 0.8, style)
        axes = fig.subplots(len(rows), len(columns), sharex=True, sharey=True, squeeze=False)
        for i, row in enumerate(rows):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                ax.axhline(0, color="0.5", lw=0.7, zorder=1)
                for x, experiment in enumerate(experiments):
                    cell = ratios.sel({row_dim: row, col_dim: column, "experiment": experiment})
                    color = _forcing_color(experiment)
                    for k, model in enumerate(models):
                        point = cell.sel(model=model)
                        if not np.isfinite(float(point["pr"])):
                            continue
                        position = x + offsets[k] - (0.06 if shift_only else 0)
                        ax.plot([position] * 2, _log_ratio([point["pr_lower"], point["pr_upper"]]), color=color,
                                lw=0.8, alpha=0.7, zorder=2)
                        ax.plot(position, _log_ratio(point["pr"]), ls="none",
                                marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=3.2, color=color, mec="white",
                                mew=0.3, zorder=3)
                        if shift_only:
                            ax.plot(position + 0.12, _log_ratio(point["pr_shift_only"]), ls="none",
                                    marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=3.2, mfc="white", mec=color,
                                    mew=0.7, zorder=3)
                    ax.plot([x - 0.36, x + 0.36], [float(np.median(_log_ratio(cell["pr"].dropna("model"))))] * 2,
                            color=INK, lw=1.4, zorder=4)
                if i == 0:
                    ax.set_title(str(column), loc="left", fontsize=style.label_size)
                if j == 0:
                    ax.set_ylabel(f"{row}\nProbability ratio")
                ax.set_yticks(ticks, [f"1/{2 ** -t:g}" if t < 0 else f"{2 ** t:g}" for t in ticks])
                ax.set_ylim(RATIO_LIMITS[0] - 0.3, RATIO_LIMITS[1] + 0.3)
                ax.set_xticks(range(len(experiments)), experiments, rotation=40, ha="right")
                ax.set_xlim(-0.6, len(experiments) - 0.4)
                core.style_ax(ax)
                ax.grid(axis="x", visible=False)
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4, color="0.35",
                          label=model) for k, model in enumerate(models)]
        handles.append(Line2D([], [], color=INK, lw=1.4, label="Multi-model median"))
        if shift_only:
            handles.append(Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="0.35",
                                  label="Shift of the mean only"))
        _legend(fig, handles, style, ncol=min(len(handles), 5))
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


@plot("figure")
def scatter_by_season(x, y, xlabel, ylabel, experiments=None, columns=None, col_dim="season", fig=None,
                      style=fr.PAPER, tag_start=None, width=fr.PAGE_WIDTH):
    """D2, D4, E3, E5: one change against another, one point per model and forcing, one panel per season.

    Each point is a model (its marker) and a forcing (its colour); the
    correlation over all the points is in each panel's corner.

    Args:
        x, y (xr.DataArray): On (model, experiment, col_dim), e.g. regional means of the change in sea-ice
            concentration and of the change in width.
        xlabel, ylabel (str): Axis labels.
        experiments (Sequence[str] | None): The forcings drawn.
        columns (Sequence | None): Values of ``col_dim``, in order.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    experiments = list(experiments if experiments is not None else x["experiment"].values)
    columns = list(columns if columns is not None else x[col_dim].values)
    x, y = xr.align(x.sel(experiment=experiments), y.sel(experiment=experiments), join="inner")
    models = [str(m) for m in x["model"].values]
    with plt.rc_context(style.rc):
        fig = _target(fig, width, width / len(columns) + 0.8, style)
        axes = fig.subplots(1, len(columns), sharex=True, sharey=True, squeeze=False)
        for ax, column in zip(axes[0], columns):
            xs, ys = x.sel({col_dim: column}), y.sel({col_dim: column})
            ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            ax.axvline(0, color="0.6", lw=0.6, zorder=0)
            for experiment in experiments:
                for k, model in enumerate(models):
                    ax.plot(float(xs.sel(model=model, experiment=experiment)),
                            float(ys.sel(model=model, experiment=experiment)), ls="none",
                            marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=3.8, color=_forcing_color(experiment),
                            mec="white", mew=0.3)
            valid = np.isfinite(xs.values) & np.isfinite(ys.values)
            if valid.sum() > 2:
                r = np.corrcoef(xs.values[valid], ys.values[valid])[0, 1]
                ax.text(0.97, 0.03, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=style.tick_size)
            ax.set_title(str(column), loc="left", fontsize=style.label_size)
            ax.set_xlabel(xlabel)
            core.style_ax(ax)
        axes[0, 0].set_ylabel(ylabel)
        model_handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4,
                                color="0.35", label=model) for k, model in enumerate(models)]
        _forcing_legend(fig, experiments, style, model_handles)
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


@plot("figure")
def forcing_bars(panels, x_dim="experiment", bars=None, columns=None, col_dim="season", fig=None, style=fr.PAPER,
                 tag_start=None, width=fr.PAGE_WIDTH):
    """E1, F1: a change for each forcing (bars: the multi-model median; markers: each model), one panel per season.

    Args:
        panels (dict): Row label (the y label) -> a DataArray on (model, x_dim, col_dim): one row of panels each.
        x_dim (str): The dim along x, e.g. "experiment", or "term" for additivity (with "sum of parts").
        bars (Sequence | None): Its values, in order.
        columns (Sequence | None): Values of ``col_dim``, in order.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    first = next(iter(panels.values()))
    bars = list(bars if bars is not None else first[x_dim].values)
    columns = list(columns if columns is not None else first[col_dim].values)
    models = [str(m) for m in first["model"].values]
    offsets = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else np.zeros(1)
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 1.6 * len(panels) + 0.9, style)
        axes = fig.subplots(len(panels), len(columns), sharex=True, sharey="row", squeeze=False)
        for i, (label, values) in enumerate(panels.items()):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                ax.axhline(0, color="0.6", lw=0.6, zorder=1)
                for x, bar in enumerate(bars):
                    cell = values.sel({x_dim: bar, col_dim: column})
                    color = _forcing_color(str(bar))
                    ax.bar(x, float(cell.median("model")), width=0.7, color=color, alpha=0.35, lw=0, zorder=2)
                    for k, model in enumerate(models):
                        ax.plot(x + offsets[k], float(cell.sel(model=model)), ls="none",
                                marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=3.2, color=color, mec="white",
                                mew=0.3, zorder=3)
                if i == 0:
                    ax.set_title(str(column), loc="left", fontsize=style.label_size)
                if j == 0:
                    ax.set_ylabel(label)
                ax.set_xticks(range(len(bars)), bars, rotation=40, ha="right")
                ax.set_xlim(-0.6, len(bars) - 0.4)
                core.style_ax(ax)
                ax.grid(axis="x", visible=False)
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4, color="0.35",
                          label=model) for k, model in enumerate(models)]
        handles.append(Patch(color="0.6", alpha=0.5, label="Multi-model median"))
        _legend(fig, handles, style, ncol=min(len(handles), 5))
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


#(t): The ridges' colours, by their mean: blue through purple to red, with no white in the middle, so that a ridge
#(t): with no change still stands out against the page
RIDGE_CMAP = LinearSegmentedColormap.from_list("ridge", ["#3d6fb6", "#7b5c93", "#c0392b"])


@plot("figure")
def ridgelines(densities, means, ridges=None, ridge_dim="experiment", columns=None, col_dim="season", lines=None,
               limit=3.0, labels=None, value_format="{:+.1f}", xlabel="", colorbar_label=None, overlap=1.8, fig=None,
               style=fr.PAPER, tag_start=None, width=fr.PAGE_WIDTH):
    """R1, R2: whole distributions as overlapping ridges, one per forcing (or model), one panel per season.

    Each ridge is a density, drawn up from its own baseline and coloured by
    its mean (blue to red through purple; ``limit`` takes the end colours),
    which is also written at its right end. The ridges below overlap the ones
    above, as in the ridgeline ("joy") plots of whole populations. The dotted
    lines mark levels shared by every ridge, e.g. hist-nat's 5th and 95th
    percentiles, so the share of each ridge beyond them is what the
    probability ratios count.

    Args:
        densities (xr.DataArray): On (ridge_dim, col_dim, x), e.g. ``stats.fast_kde_xr`` of anomalies.
        means (xr.DataArray): Each ridge's mean on (ridge_dim, col_dim): its colour and its number.
        ridges (Sequence | None): Values of ``ridge_dim``, top to bottom; all by default.
        columns (Sequence | None): Values of ``col_dim``, in order.
        lines (xr.DataArray | None): Dotted vertical lines, on (col_dim, one other dim), e.g. (season, quantile).
        limit (float): The mean that takes the end colours.
        labels (dict | None): Each ridge's label, on the left; its value by default.
        value_format (str): Format of the mean written beside each ridge.
        xlabel (str): Label of the x axis.
        colorbar_label (str | None): Draw a colour bar of the means, with this label.
        overlap (float): How many baselines up the highest density in the figure reaches.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    ridges = list(ridges if ridges is not None else densities[ridge_dim].values)
    columns = list(columns if columns is not None else densities[col_dim].values)
    x = densities["x"].values
    #(c): One height scale for every ridge in the figure, so their areas can be compared
    scale = overlap / float(densities.sel({ridge_dim: ridges}).max())
    norm = Normalize(-limit, limit)
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 0.3 * len(ridges) + 1.4, style)
        axes = fig.subplots(1, len(columns), sharex=True, sharey=True, squeeze=False)
        for ax, column in zip(axes[0], columns):
            for k, ridge in enumerate(ridges):
                base = len(ridges) - 1 - k
                density = np.nan_to_num(densities.sel({ridge_dim: ridge, col_dim: column}).values)
                mean = float(means.sel({ridge_dim: ridge, col_dim: column}))
                color = RIDGE_CMAP(norm(mean)) if np.isfinite(mean) else "0.75"
                #(c): The ridges further down are drawn later, so they overlap the ones above
                ax.fill_between(x, base, base + scale * density, color=color, lw=0, zorder=2 * k + 1)
                ax.plot(x, base + scale * density, color="white", lw=0.9, zorder=2 * k + 2)
                if np.isfinite(mean):
                    #(c): A small negative mean rounds to '-0.0'; write it as 0
                    text = value_format.format(mean)
                    text = value_format.format(0.0) if float(text) == 0 else text
                    ax.text(x[-1], base + 0.08, text, ha="right", va="bottom",
                            fontsize=style.tick_size - 0.5, color="0.25", zorder=100)
            if lines is not None:
                for value in np.ravel(lines.sel({col_dim: column}).values):
                    if np.isfinite(value):
                        ax.axvline(value, color="0.45", lw=1.0, ls=(0, (1, 1.6)), zorder=99)
            ax.set_title(str(column), loc="left", fontsize=style.label_size)
            ax.set_xlabel(xlabel)
            ax.set_ylim(-0.15, len(ridges) - 1 + overlap + 0.15)
            ax.set_xlim(x[0], x[-1])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].set_yticks(np.arange(len(ridges))[::-1], [str((labels or {}).get(r, r)) for r in ridges])
        if colorbar_label:
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=RIDGE_CMAP), ax=list(axes[0]), location="bottom",
                               shrink=0.45, aspect=35, pad=0.02, extend="both")
            bar.set_label(colorbar_label, fontsize=style.cbar_label_size)
            bar.ax.tick_params(labelsize=style.tick_size)
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


# ---------------------------------------------------------------------------
# 4. Verification against ERA5
# ---------------------------------------------------------------------------

@plot("figure")
def rank_histograms(ranks, rows=None, row_dim="model", columns=None, col_dim="season", fig=None, style=fr.PAPER,
                    tag_start=None, width=fr.PAGE_WIDTH):
    """ERA5's rank histograms (Suarez-Gutierrez et al., 2021) against the range a perfect model gives.

    Bars are ERA5's rank frequencies on bins of scaled rank (0: below every
    member, 1: above every member); the grey band is the pseudo-observations'
    central range, what the model itself gives when one member stands in for
    ERA5, for this ensemble and record. Flat bars inside the band: ERA5 could
    be a member. A U shape: the model's spread is too small; a dome: too
    large; a slope: its forced response differs. Each panel's corner gives
    ERA5's percentile for the dispersion among the pseudo-observations', ✗ when
    beyond their range.

    Args:
        ranks (xr.Dataset): ``obs_frequency``, ``pseudo_lower``, ``pseudo_upper`` on (row_dim, col_dim, rank),
            and ``percentile``, ``verdict`` on (row_dim, col_dim, statistic), e.g. one region, experiment and
            treatment of notebook 02's ``regional_ranks``.
        rows, columns (Sequence | None): Values of ``row_dim`` and ``col_dim``.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    row_values = list(rows if rows is not None else ranks[row_dim].values)
    columns = list(columns if columns is not None else ranks[col_dim].values)
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 1.1 * len(row_values) + 0.8, style)
        axes = fig.subplots(len(row_values), len(columns), sharex=True, sharey=True, squeeze=False)
        for i, row in enumerate(row_values):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                cell = ranks.sel({row_dim: row, col_dim: column})
                rank = cell["rank"].values
                step = rank[1] - rank[0]
                #(c): Each bin's range drawn across the whole bin: its value repeated at both edges
                edges = np.column_stack([rank - step / 2, rank + step / 2]).ravel()
                ax.fill_between(edges, np.repeat(cell["pseudo_lower"].values, 2),
                                np.repeat(cell["pseudo_upper"].values, 2), color="0.82", lw=0, zorder=1)
                ax.bar(rank, cell["obs_frequency"], width=0.8 * step, color=RANK_BAR, edgecolor="#2a78d6", lw=0.4,
                       zorder=2)
                ax.axhline(100 / rank.size, color=PSEUDO_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=3)
                if "dispersion" in cell["statistic"].values and np.isfinite(float(cell["percentile"].sel(statistic="dispersion"))):
                    dispersion = cell.sel(statistic="dispersion")
                    flag = "✗" if float(dispersion["verdict"]) != 0 else "✓"
                    ax.text(0.97, 0.95, f"{flag} {float(dispersion['percentile']):.0f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=style.tick_size)
                if i == 0:
                    ax.set_title(str(column), loc="left", fontsize=style.label_size)
                if j == 0:
                    ax.set_ylabel(f"{row}\n%", fontsize=style.tick_size)
                if i == len(row_values) - 1:
                    ax.set_xlabel("Rank (scaled)")
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1])
                core.style_ax(ax, grid=False)
        handles = [Patch(color=RANK_BAR, label="ERA5"), Patch(color="0.82", label="Perfect model (5–95%)"),
                   Line2D([], [], color=PSEUDO_COLOR, lw=0.8, ls=(0, (3, 2)), label="Flat")]
        _legend(fig, handles, style)
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


@plot("figure")
def percentile_tables(percentile, verdict, rows=None, row_dim="model", columns=None, col_dim="experiment",
                      panels=None, panel_dim="season", fig=None, style=fr.PAPER, tag_start=None, width=fr.PAGE_WIDTH):
    """ERA5's percentile among each experiment's members as tables of colours, one per season; outlined if flagged.

    ERA5 inside historical (central colours) and outside hist-nat (the
    darkest colours, outlined) is the consistency the attribution needs.

    Args:
        percentile, verdict (xr.DataArray): On (row_dim, col_dim, panel_dim), e.g. one region of notebook 02's
            ``regional_consistency`` (the trend).
        rows, columns, panels (Sequence | None): Values of each dim, in order.
        fig, style, tag_start, width: As for ``map_block``.

    Returns:
        core.Panels
    """
    row_values = list(rows if rows is not None else percentile[row_dim].values)
    col_values = list(columns if columns is not None else percentile[col_dim].values)
    panel_values = list(panels if panels is not None else percentile[panel_dim].values)
    scale = percentile_scale()
    with plt.rc_context(style.rc):
        fig = _target(fig, width, 0.22 * len(row_values) + 1.6, style)
        axes = fig.subplots(1, len(panel_values), sharey=True, squeeze=False)
        image = None
        for ax, panel in zip(axes[0], panel_values):
            table = percentile.sel({panel_dim: panel}).sel({row_dim: row_values, col_dim: col_values})
            flags = verdict.sel({panel_dim: panel}).sel({row_dim: row_values, col_dim: col_values})
            table = table.transpose(row_dim, col_dim)
            image = ax.imshow(table.values, cmap=scale.cmap, norm=scale.norm, aspect="auto")
            for (r, c), flag in np.ndenumerate(flags.transpose(row_dim, col_dim).values):
                if np.isfinite(flag) and flag != 0:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
                value = table.values[r, c]
                if np.isfinite(value):
                    ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=style.tick_size - 0.5)
            ax.set_xticks(range(len(col_values)), col_values, rotation=40, ha="right")
            ax.set_yticks(range(len(row_values)), row_values)
            ax.set_title(str(panel), loc="left", fontsize=style.label_size)
            ax.tick_params(length=0)
        _colorbar(fig, image, axes, scale, style, shrink=0.6)
        next_tag = _tag(axes, tag_start, style)
    return _done(fig, axes, next_tag)


# ---------------------------------------------------------------------------
# 5. Layouts
# ---------------------------------------------------------------------------

def compose(layout, heights, width=fr.PAGE_WIDTH, style=fr.PAPER, tag=True):
    """Put panels together in one figure: rows of blocks, stacked, the blocks of a row side by side.

    Each block is a function of ``fig`` and ``tag_start`` (one of this
    module's panels with its data bound, e.g. by ``functools.partial``), and
    is drawn into its own SubFigure; the panel letters run on from block to
    block.

    Args:
        layout (Sequence[Sequence]): Rows, top to bottom; each a list of blocks, or of (block, width ratio).
        heights (Sequence[float]): Height of each row, in inches.
        width (float): Width of the figure, in inches.
        style (fr.Style): Fonts and line widths.
        tag (bool): Letter the panels.

    Returns:
        core.Panels: ``fig``, and ``extras["blocks"]``: each block's Panels, row by row.
    """
    with plt.rc_context(style.rc):
        fig = new_figure(width, sum(heights), style)
        row_figures = np.atleast_1d(fig.subfigures(len(layout), 1, height_ratios=list(heights)))
        next_tag = 0 if tag else None
        drawn = []
        for row_figure, row in zip(row_figures, layout):
            blocks = [block if isinstance(block, tuple) else (block, 1) for block in row]
            cells = np.atleast_1d(row_figure.subfigures(1, len(blocks), width_ratios=[ratio for _, ratio in blocks]))
            for cell, (block, _) in zip(cells, blocks):
                panels = block(fig=cell, tag_start=next_tag)
                drawn.append(panels)
                next_tag = panels.extras.get("next_tag", next_tag)
    return core.Panels(fig=fig, axes=None, extras={"blocks": drawn})
