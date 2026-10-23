"""The colour scales and styles of notebooks 04-07, and their figures of lines and dots.

Every map in the notebooks is drawn layer by layer with ``antarctica``; its colours come from here, so that colour
and pattern mean the same thing in every figure:

    one quantity      one ``Scale`` (levels, colormap, label), from ``colour_scales``; ``Scale.colours()`` gives
                      ``antarctica.heatmaps_of_antarctica`` its levels, colours, ticks and label in one go
    red <-> blue      an increase <-> a decrease
    dots              not significant (one model), or not robust (fewer than 2/3 of the models significant)
    yellow -> red     |S/N|, how far apart two distributions have moved: below 1, 1 to 2, above 2
    counts            more than half the models changes the hue: reds to browns (increase), blues to
                      purples (decrease), greys to greens (no sign, ``majority_scale``); white is none
    3 x 3 key         the mean and the width together (``joint_scale``, drawn by ``antarctica.bivariate_key``)
    black lines       the sea-ice edge (15% contour), solid for the experiment and dashed for hist-nat

Sections
--------
1. Colour scales
2. Styles: the notebook, the paper and slides; saving
3. Lines and dots: the change at every quantile, and area means
"""

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from .. import config
from .era5_evaluation import EVAL_RC, INK, MODEL_COLORS, MODEL_MARKERS
from plotting_modules.constants import FORCING_COLORS
from plotting_modules import core
from plotting_modules.utils import plot

# ---------------------------------------------------------------------------
# 1. Colour scales
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Scale:
    """A colour scale: contour levels (class edges if ``discrete``), a colormap and a colour-bar label.

    Compared by identity: two Scale objects are never equal, even with the same levels.

    Args:
        levels (array-like): Contour levels, or the class edges of a discrete scale.
        cmap: Colormap or its name.
        label (str): Colour-bar label.
        norm (matplotlib.colors.Normalize | None): Mapping of values to colours, for uneven levels.
        ticks (array-like | None): Colour-bar ticks; chosen from the levels if None.
        discrete (bool): Categorical: drawn cell by cell, one colour-bar block per class.
        ticklabels (Sequence[str] | None): Labels of the classes of a discrete scale.
        key (tuple | None): For a two-way class scale (``bivariate_scale``): (row names, column names,
            row title, column title), drawn as a grid of the classes instead of a colour bar.
        extend (str): Which ends values beyond the levels can fall at ("both", "max", "min"); a colour-bar
            arrow marks each. "max" for a quantity that cannot go below its first level (a spread from 0).
    """

    levels: np.ndarray
    cmap: object = "RdBu_r"
    label: str = ""
    norm: object = None
    ticks: np.ndarray = None
    discrete: bool = False
    ticklabels: tuple = None
    key: tuple = None
    extend: str = "both"

    def colours(self):
        """This scale as the colour settings of ``antarctica.heatmaps_of_antarctica``, passed as ``**scale.colours()``.

        Returns:
            dict: ``levels``, ``cmap``, ``extend`` ('neither' for a discrete scale: every value is one of its classes),
            ``ticks`` (each class's number for a discrete scale), ``ticklabels`` and ``label``. E.g. for
            ``colour_scales('tas').change``: levels -3 to 3 every 0.5, 'RdBu_r', 'both', None, None, 'Change (°C)'.
        """
        ticks = self.ticks
        if self.discrete and ticks is None:
            #(c): A class's number is the middle of its two levels (class 0 between -0.5 and 0.5)
            ticks = (self.levels[:-1] + self.levels[1:]) / 2
        return dict(levels=self.levels, cmap=self.cmap, extend="neither" if self.discrete else self.extend,
                    ticks=ticks, ticklabels=self.ticklabels, label=self.label)


def _ramp(cmap, start, stop, n):
    """``n`` colours from a colormap, between ``start`` and ``stop`` (0 to 1, either way round)."""
    return list(plt.get_cmap(cmap)(np.linspace(start, stop, n)))


def separation_scale(label="|S/N|", step=0.25):
    """|S/N| from 0 to 3: yellows below 1, oranges from 1 to 2, reds from 2 (the darkest above 3).

    |S/N| says how far apart two distributions have moved, in units of their
    year-to-year noise: below 1 they overlap heavily, above 2 they barely do.
    It is not a test, so there is no threshold, only bands.
    """
    levels = _from_zero(3, step)
    per_band = int(round(1 / step))
    #(c): One more colour than bands, for the values above 3 (extend="max")
    colors = ListedColormap([*_ramp("YlOrRd", 0.0, 0.2, per_band), *_ramp("Oranges", 0.4, 0.7, per_band),
                             *_ramp("Reds", 0.6, 0.95, per_band), plt.get_cmap("Reds")(1.0)])
    return Scale(levels, colors, label, norm=BoundaryNorm(levels, colors.N, extend="max"), ticks=_from_zero(3, 0.5),
                 extend="max")


def pvalue_scale(alpha=0.05, label=None):
    """p-values in four bands: below alpha / 5, below alpha (both purple: significant), below 2 alpha (grey), above (white)."""
    levels = np.array([0, alpha / 5, alpha, 2 * alpha, 1])
    colors = ListedColormap(["#3f007d", "#9e9ac8", "#d9d9d9", "white"])
    return Scale(levels, colors, label or f"p-value (significant below {alpha:g})",
                 norm=BoundaryNorm(levels, colors.N), ticks=levels, extend="neither")


def count_scale(n_models, label="Number of models with a significant decrease (blue) or increase (red)"):
    """Discrete and diverging: how many models have a significant decrease (left) or increase (right); white for none.

    Up to half the models are blues and reds; more than half, purples and
    browns, so where most models agree stands out. Notebook 04 maps the
    decreases as negative numbers (the signed counts), so this one colour bar
    serves both rows of counts; the tick labels are the counts themselves.
    """
    half = n_models // 2
    #(c): Darkest at the ends: the most models with a decrease on the left, with an increase on the right
    decreases = [*_ramp("PRGn", 0.0, 0.3, n_models - half), *_ramp("Blues", 0.7, 0.3, half)]
    increases = [*_ramp("Reds", 0.3, 0.7, half), *_ramp("BrBG", 0.3, 0.0, n_models - half)]
    colors = ListedColormap([*decreases, "white", *increases])
    levels = np.arange(-n_models - 0.5, n_models + 1)
    return Scale(levels, colors, label, discrete=True,
                 ticklabels=[str(abs(k)) for k in range(-n_models, n_models + 1)])


def majority_scale(n_models, label="Number of models"):
    """Discrete: how many models, for a count with no sign (e.g. |S/N| above a threshold); white for none.

    Up to half the models are greys, more than half greens, like
    ``count_scale``'s change of hue.
    """
    half = n_models // 2
    colors = ListedColormap(["white", *_ramp("Greys", 0.2, 0.45, half), *_ramp("Greens", 0.55, 0.95, n_models - half)])
    return Scale(np.arange(-0.5, n_models + 1), colors, label, discrete=True,
                 ticklabels=[str(k) for k in range(n_models + 1)])


def class_scale(classes, colors, label=""):
    """Discrete scale for categories numbered 0, 1, 2, ..., e.g. ``rc.CHANGE_CLASSES``.

    Args:
        classes (Sequence[str]): The name of each category, in the order of its number.
        colors (Sequence): One colour per category.
    """
    #(c): Each name broken onto short lines, so four of them fit under one panel
    return Scale(np.arange(-0.5, len(classes)), ListedColormap(list(colors)), label, discrete=True,
                 ticklabels=[textwrap.fill(name, 8) for name in classes])


def bivariate_scale(rows, columns, colors, row_title, column_title):
    """Classes of two things at once, e.g. the mean (rows) and the width (columns), each down, unchanged or up.

    Class ``i * len(columns) + j`` is row ``i`` and column ``j``, so with
    rows and columns both (down, none, up) and signs -1, 0, +1 the class is
    ``3 * (row_sign + 1) + (column_sign + 1)``. Its key is a grid of the
    classes, the rows bottom to top.

    Args:
        rows, columns (Sequence[str]): Names of the row and column categories.
        colors (Sequence): One colour per class, row by row from the bottom.
        row_title, column_title (str): What the rows and the columns are.
    """
    n = len(rows) * len(columns)
    return Scale(np.arange(-0.5, n), ListedColormap(list(colors)), "", discrete=True,
                 key=(list(rows), list(columns), row_title, column_title))


#(t): The colours of ``joint_scale``: the mean sets blue or red, a narrower distribution mixes in yellow and a
#(t): wider one purple; white where neither changed
JOINT_COLORS = ("#7fc8a9", "#4a8fd1", "#4b3f9e",    # cooler: narrower, no change in width, wider
                "#f3e19b", "white", "#cab2d6",      # no change in the mean
                "#f4a261", "#e05a4f", "#9e2a6e")    # warmer


def joint_scale():
    """``bivariate_scale`` of the change in the mean (rows) and in the width (columns): down, none or up."""
    return bivariate_scale(["Lower", "Same", "Higher"], ["Narrow", "Same", "Wide"], JOINT_COLORS, "Mean", "Width")


@dataclass(frozen=True)
class Scales:
    """Every colour scale of notebooks 04, 05 and 06, one per quantity (``colour_scales`` makes them)."""

    temperature: Scale
    change: Scale
    noise: Scale
    spread: Scale
    spread_change: Scale
    tail: Scale
    sn: Scale
    pvalue: Scale
    classes: Scale
    joint: Scale


#(t): The range of each colour scale, per variable, as (end, step): the values (start, stop, step, tick step), the
#(t): largest change, standard deviation, spread and change in spread shown, and the step between colours.
#(c): About a dozen colours per scale: more cannot be told apart in a small printed panel, and every level also
#(c): gets a contour line. Check them on the real data with ``saturation`` (notebook 06)
RANGES = {
    "tas": dict(values=(-60, 10, 5, 15), change=(3, 0.5), noise=(4, 0.5), spread=(12, 1), spread_change=(1, 0.25)),
}


def _symmetric(end, step):
    """Levels from -end to end every step, exactly (no 0.30000000000000004)."""
    n = int(round(end / step))
    return np.round(np.arange(-n, n + 1) * step, 10)


def _from_zero(end, step):
    """Levels from 0 to end every step, exactly."""
    return np.round(np.arange(int(round(end / step)) + 1) * step, 10)


def colour_scales(variable, alpha=0.05):
    """One colour scale per quantity for ``variable``, shared by every figure that shows that quantity.

    Args:
        variable (str): Key of ``config.VARIABLES`` and of ``RANGES``.
        alpha (float): Significance level, for the p-value scale.

    Returns:
        Scales
    """
    if variable not in RANGES:
        raise KeyError(f"no colour ranges for {variable}: add them to forced_response.RANGES")
    var = config.VARIABLES[variable]
    ranges = RANGES[variable]
    start, stop, step, tick = ranges["values"]
    units = var.units
    return Scales(
        temperature=Scale(np.arange(start, stop + step / 2, step), "RdBu_r", var.label,
                          ticks=np.arange(start, stop + step / 2, tick)),
        change=Scale(_symmetric(*ranges["change"]), "RdBu_r", f"Change ({units})"),
        noise=Scale(_from_zero(*ranges["noise"]), "viridis", f"Standard deviation ({units})", extend="max"),
        spread=Scale(_from_zero(*ranges["spread"]), "viridis", f"Spread ({units})", extend="max"),
        spread_change=Scale(_symmetric(*ranges["spread_change"]), "RdBu_r", f"Change in spread ({units})"),
        tail=Scale(_symmetric(*ranges["spread_change"]), "RdBu_r", f"Upper-tail minus lower-tail change ({units})"),
        sn=separation_scale(),
        pvalue=pvalue_scale(alpha),
        classes=class_scale(["neither", "mean only", "width only", "both"], ["white", "#fdb863", "#b2abd2", "#5e3c99"],
                            "What changed"),
        joint=joint_scale(),
    )


# ---------------------------------------------------------------------------
# 2. Styles: the notebook, the paper and slides; saving
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    """Font sizes and line widths of a figure, so one figure function serves the notebook and the paper.

    Args:
        rc (dict): matplotlib rcParams used while drawing.
        label_size (float): Column and row labels.
        title_size (float): Figure title.
        tick_size (float): Colour-bar tick labels.
        cbar_label_size (float): Colour-bar labels.
        tag_size (float): Panel letters.
    """

    rc: dict
    label_size: float = 10
    title_size: float = 12
    tick_size: float = 9
    cbar_label_size: float = 10
    tag_size: float = 9


#(t): On screen, in notebook 04
NOTEBOOK = Style(EVAL_RC)

#(t): In print: 7-8 pt text, for figures 183 mm (7.2 in) wide, a two-column page
PAPER = Style(
    {**EVAL_RC, "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
     "legend.fontsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42},
    label_size=8, title_size=9, tick_size=6.5, cbar_label_size=7.5, tag_size=8,
)


#(t): On a slide: big text, read from the back of a room
SLIDES = Style(
    {**EVAL_RC, "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
     "ytick.labelsize": 11, "legend.fontsize": 11},
    label_size=13, title_size=15, tick_size=11, cbar_label_size=12, tag_size=11,
)


#(t): Print sizes, in inches: the width of a two-column page (183 mm) and of one column (89 mm), and the most a
#(t): figure can be tall with room left for its caption (229 mm)
PAGE_WIDTH = 7.2
COLUMN_WIDTH = 3.5
PAGE_HEIGHT = 9.0


def saturation(da, scale):
    """The share of a field's values beyond either end of its colour scale, which the map cannot tell apart.

    Shown as the end colours (and arrows on the colour bar); a few percent is
    fine, much more means the scale's range (``RANGES``) is too narrow for
    the data. Grid points count equally, wherever they are.

    Args:
        da (xr.DataArray): Any field.
        scale (Scale): The scale it is drawn with.

    Returns:
        dict: ``below`` and ``above``, fractions of the values that are not NaN.
    """
    valid = da.notnull().sum()
    return {"below": float((da < scale.levels[0]).sum() / valid), "above": float((da > scale.levels[-1]).sum() / valid)}


def save(panels, directory, name):
    """Save a figure as a PDF (vector, for the paper) and a 300 dpi PNG; return the two paths.

    Args:
        panels (core.Panels | matplotlib.figure.Figure): The figure.
        directory (pathlib.Path): Folder, made if missing.
        name (str): File name, without extension.
    """
    fig = getattr(panels, "fig", panels)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.pdf", directory / f"{name}.png"]
    #(c): TrueType (Type 42) fonts in the PDF: matplotlib's default Type 3 fonts are refused by many journals
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        for path in paths:
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    return paths


# ---------------------------------------------------------------------------
# 3. Lines and dots: the change at every quantile, and area means
# ---------------------------------------------------------------------------

@plot("figure")
def quantile_curves(curves, title=None, units="°C", col_dim="experiment", row_dim=None, style=NOTEBOOK,
                    panel_size=(3.3, 3.0), width=None):
    """The change at every quantile, one line per model, one panel per experiment (and season).

    A flat line is a pure shift of the distribution; a rising line means the
    upper quantiles move more than the lower ones. Panels in a row share their
    y axis.

    Args:
        curves (xr.DataArray): On (model, col_dim, quantile), or (model, row_dim, col_dim, quantile), e.g.
            notebook 03's ``quantile_curve`` (Antarctic means).
        title (str | None): Figure title.
        units (str): Units of the change.
        col_dim (str): Dim mapped to columns.
        row_dim (str | None): Dim mapped to rows; one row if None.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        panel_size (tuple[float, float]): Width and height of each panel, in inches.
        width (float | None): The figure's width in inches (the legend included), instead of ``panel_size[0]``.

    Returns:
        core.Panels
    """
    models = [str(m) for m in curves["model"].values]
    column_values = list(curves[col_dim].values)
    row_values = [None] if row_dim is None else list(curves[row_dim].values)
    percentile = 100 * curves["quantile"].values

    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(len(row_values), len(column_values), sharex=True, sharey="row", layout="constrained",
                                 squeeze=False,
                                 figsize=(width or panel_size[0] * len(column_values) + 1.6,
                                          panel_size[1] * len(row_values) + 0.6))
        for i, row in enumerate(row_values):
            for j, value in enumerate(column_values):
                ax = axes[i, j]
                selection = {col_dim: value} if row is None else {col_dim: value, row_dim: row}
                ax.axhline(0, color="0.6", lw=0.8, zorder=1)
                for k, model in enumerate(models):
                    ax.plot(percentile, curves.sel(model=model, **selection), color=MODEL_COLORS[k % len(MODEL_COLORS)],
                            marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=0.4 * style.tick_size,
                            lw=0.15 * style.tick_size, zorder=3)
                if i == 0:
                    ax.set_title(str(value), loc="left")
                if i == len(row_values) - 1:
                    ax.set_xlabel("Percentile")
                if j == 0:
                    ax.set_ylabel(f"Change ({units})" if row is None else f"{row}\nChange ({units})")
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                ax.set_xticks([25, 75], minor=True)
                core.style_ax(ax)
        fig.legend(handles=[Line2D([], [], color=MODEL_COLORS[k % len(MODEL_COLORS)], lw=1.6,
                                   marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4, label=model)
                            for k, model in enumerate(models)],
                   loc="outside right center", frameon=False, title="Model", alignment="left")
        if title:
            fig.suptitle(title, fontsize=style.title_size, fontweight="bold", color=INK, x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def forcing_dots(regional, variables, title=None, col_dim="season", columns=None, experiments=None, style=NOTEBOOK,
                 panel_size=(1.6, 1.6), width=None):
    """Area means of each model's change, forcings along the x axis: one row per variable, one column per season.

    Each model is a marker (its shape) in the forcing's colour, spread a
    little sideways; the black bar is the multi-model median. The quickest
    view of how big each forcing's change is and how far the models agree.

    Args:
        regional (xr.Dataset): Area means on (model, experiment, col_dim), e.g. ``rc.regional_mean`` output.
        variables (dict): Variable of ``regional`` -> its y-axis label, top to bottom.
        title (str | None): Figure title.
        col_dim (str): Dim mapped to columns.
        columns (Sequence | None): Its values, in order; all by default.
        experiments (Sequence[str] | None): Forcings along x, in order; all by default.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        panel_size (tuple[float, float]): Width and height of each panel, in inches.
        width (float | None): The figure's width in inches (the legend included), instead of ``panel_size[0]``.

    Returns:
        core.Panels
    """
    columns = list(columns if columns is not None else regional[col_dim].values)
    experiments = list(experiments if experiments is not None else regional["experiment"].values)
    models = [str(m) for m in regional["model"].values]
    offsets = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0.0]

    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(len(variables), len(columns), sharex=True, sharey="row", layout="constrained",
                                 squeeze=False,
                                 figsize=(width or panel_size[0] * len(columns) + 1.4,
                                          panel_size[1] * len(variables) + 0.5))
        for i, (name, label) in enumerate(variables.items()):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                ax.axhline(0, color="0.6", lw=0.7, zorder=1)
                for x, experiment in enumerate(experiments):
                    values = regional[name].sel({col_dim: column, "experiment": experiment})
                    for k, model in enumerate(models):
                        ax.scatter(x + offsets[k], float(values.sel(model=model)), s=18,
                                   marker=MODEL_MARKERS[k % len(MODEL_MARKERS)],
                                   color=FORCING_COLORS.get(experiment, INK), edgecolor="white", linewidth=0.5,
                                   zorder=3)
                    ax.plot([x - 0.32, x + 0.32], [float(values.median("model"))] * 2, color=INK, lw=1.6, zorder=4)
                if i == 0:
                    ax.set_title(str(column), loc="left")
                if j == 0:
                    ax.set_ylabel(label)
                ax.set_xticks(range(len(experiments)), experiments, rotation=40, ha="right")
                ax.set_xlim(-0.6, len(experiments) - 0.4)
                core.style_ax(ax)
                ax.grid(axis="x", visible=False)
        fig.legend(handles=[*[Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=5,
                                     color="0.35", label=model) for k, model in enumerate(models)],
                            Line2D([], [], color=INK, lw=1.6, label="Multi-model median")],
                   loc="outside right center", frameon=False, title="Model", alignment="left")
        if title:
            fig.suptitle(title, fontsize=style.title_size, fontweight="bold", color=INK, x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)
