"""The figures of notebooks 04 and 06. Every section of 04 draws the same four kinds of figure, seasons across:

1. What goes in       ``row_grid``: the quantities a result is made from, then the result,
                      for one model and experiment (one row each)
2. One model          ``row_grid``: the parts and the result for every experiment of one model
3. Every model        ``map_grid``: the result, models down and seasons across, one figure per experiment
4. How many models    ``count_grid``: the multi-model median of the result, then how many models
                      have a significant increase and how many a significant decrease

All four are grids of polar maps drawn by ``block_grid``, as are notebook 06's
paper figures (``PAPER`` style): blocks of rows, each with its colour bars
underneath, after Bracegirdle et al. (2024) Fig. 1. Columns (``field_grid``) or
rows (``row_grid``) that share a ``Scale`` share one colour bar. Colour and
pattern mean the same thing in every figure:

    one quantity      one ``Scale`` (levels, colormap, label), from ``colour_scales``
    red <-> blue      an increase <-> a decrease
    dots              not significant (one model), or not robust (fewer than 2/3 of the models significant)
    yellow -> red     |S/N|, how far apart two distributions have moved: below 1, 1 to 2, above 2
    counts            more than half the models changes the hue: reds to browns (increase), blues to
                      purples (decrease), greys to greens (no sign, ``majority_scale``); white is none
    3 x 3 key         the mean and the width together (``joint_scale``)

Sections
--------
1. Colour scales
2. Styles: the notebook and the paper
3. Map grids
4. Lines and dots: the change at every quantile, and area means
"""

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import config
from .era5_evaluation import EVAL_RC, INK, MODEL_COLORS, MODEL_MARKERS
from plotting_modules.constants import FORCING_COLORS
from .response_change import HATCH_COLOR, HATCH_WIDTH, NOT_SIGNIFICANT_HATCH, round_ticks
from plotting_modules import core
from plotting_modules.utils import plot

# ---------------------------------------------------------------------------
# 1. Colour scales
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Scale:
    """A colour scale: contour levels (class edges if ``discrete``), a colormap and a colour-bar label.

    Compared by identity, so two columns of a ``field_grid`` share a colour bar
    only when they are given the same Scale object.

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
                 norm=BoundaryNorm(levels, colors.N), ticks=levels)


def count_scale(n_models, label="Number of models with a significant decrease (blue) or increase (red)"):
    """Discrete and diverging: how many models have a significant decrease (left) or increase (right); white for none.

    Up to half the models are blues and reds; more than half, purples and
    browns, so where most models agree stands out. ``count_grid`` draws the
    decreases as negative numbers, so this one colour bar serves both of its
    rows; the tick labels are the counts themselves.
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


def _ticks(scale, n_columns, per_column=5, most=13):
    """The scale's own ticks, or round ones from its levels: about ``per_column`` per column under the bar, ``most`` in all."""
    if scale.ticks is not None:
        return scale.ticks
    return round_ticks(scale.levels, min(per_column * n_columns, most))


# ---------------------------------------------------------------------------
# 2. Styles: the notebook and the paper
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
        hatch_width (float): Line width of the dots.
        cbar_height (float): Colour-bar thickness, in inches.
        ticks_per_column (int): About how many colour-bar ticks fit under one panel.
    """

    rc: dict
    label_size: float = 10
    title_size: float = 12
    tick_size: float = 9
    cbar_label_size: float = 10
    tag_size: float = 9
    hatch_width: float = HATCH_WIDTH
    cbar_height: float = 0.15
    ticks_per_column: int = 5


#(t): On screen, in notebook 04
NOTEBOOK = Style(EVAL_RC)

#(t): In print: 7-8 pt text, for figures 183 mm (7.2 in) wide, a two-column page
PAPER = Style(
    {**EVAL_RC, "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
     "legend.fontsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42},
    label_size=8, title_size=9, tick_size=6.5, cbar_label_size=7.5, tag_size=8, hatch_width=0.35, cbar_height=0.1,
    ticks_per_column=3,
)


#(t): On a slide: big text, read from the back of a room
SLIDES = Style(
    {**EVAL_RC, "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
     "ytick.labelsize": 11, "legend.fontsize": 11},
    label_size=13, title_size=15, tick_size=11, cbar_label_size=12, tag_size=11, cbar_height=0.18,
    ticks_per_column=4,
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
# 3. Map grids
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """Rows of a ``block_grid`` that share colour bars, which are drawn under the block.

    Args:
        rows (dict): Row label -> the row's fields, one per column: an xr.Dataset (its variables, in
            order, are the columns) or a list of xr.DataArray (None leaves a panel empty).
        scales (Sequence[Scale]): One per column. Neighbouring columns given the same Scale object
            share one colour bar.
        not_significant (dict | None): Row label -> boolean masks, dotted where True: an xr.Dataset
            with a variable for each column that has a test, or a list with None where there is none.
    """

    rows: dict
    scales: list
    not_significant: dict = None


@dataclass
class Column:
    """One column of a ``field_grid``.

    Args:
        data (xr.DataArray): The field, on (lat, lon), or (row_dim, lat, lon).
        scale (Scale): Its colour scale.
        title (str): Column title ('\\n' for a second line).
        not_significant (xr.DataArray | None): Boolean, the same dims: dotted where True.
    """

    data: xr.DataArray
    scale: Scale
    title: str
    not_significant: xr.DataArray = None


def _fields(row):
    """A row's fields as (names, list of DataArray): a Dataset's variables, or a list as it is."""
    if isinstance(row, xr.Dataset):
        return list(row.data_vars), [row[name] for name in row.data_vars]
    return [None] * len(row), list(row)


def _masks(masks, names):
    """A row's masks, one per column: from a Dataset by variable name, or a list as it is."""
    if masks is None:
        return [None] * len(names)
    if isinstance(masks, xr.Dataset):
        return [masks[name] if name in masks else None for name in names]
    return list(masks)


def _scale_groups(scales):
    """(first, last) column index of each run of neighbouring columns that share one Scale."""
    groups = []
    for j, scale in enumerate(scales):
        if groups and scales[groups[-1][1]] is scale:
            groups[-1][1] = j
        else:
            groups.append([j, j])
    return groups


def _lines(text):
    """How many lines a label takes."""
    return str(text).count("\n") + 1


def _bar_label(label, n_columns, panel_size, style):
    """A colour-bar label broken onto lines no wider than its bar, which spans ``n_columns`` panels."""
    bar_width = 0.8 * (n_columns * panel_size + (n_columns - 1) * core.WSPACE)
    return textwrap.fill(label, width=max(int(bar_width * 72 / (0.6 * style.cbar_label_size)), 8))


def _text_height(size, n_lines=1):
    """Height of ``n_lines`` of text at ``size`` points, in inches (1 pt = 1/72 in, with line spacing)."""
    return n_lines * size * 1.35 / 72


def _draw_key(ax, scale, mappable, style):
    """A grid of a bivariate scale's classes, centred in ``ax``'s slot, each cell as wide as its label needs."""
    rows, columns, row_title, column_title = scale.key
    fig = ax.figure
    box = ax.get_position()
    cell_inches = max(box.height * fig.get_figheight() / len(rows),
                      max(len(name) for name in columns) * 0.7 * style.tick_size / 72 + 0.12)
    width = min(cell_inches * len(columns) / fig.get_figwidth(), box.width)
    ax.set_position([box.x0 + (box.width - width) / 2, box.y0, width, box.height])
    classes = np.arange(len(rows) * len(columns)).reshape(len(rows), len(columns))
    ax.imshow(classes, cmap=mappable.get_cmap(), norm=mappable.norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=style.tick_size)
    ax.set_yticks(range(len(rows)), rows, fontsize=style.tick_size)
    ax.set_xlabel(column_title, fontsize=style.cbar_label_size, labelpad=2)
    ax.set_ylabel(row_title, fontsize=style.cbar_label_size, labelpad=2)
    #(c): Row names on the right, away from the colour bar of the column next to it
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(length=0, pad=2)
    for edge in ("top", "right", "bottom", "left"):
        ax.spines[edge].set_linewidth(0.5)
    #(c): Thin white lines between the cells, so white (no change) still reads as a cell
    ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.6", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    return ax


@plot("figure")
def block_grid(blocks, title=None, column_titles=None, panel_size=2.5, style=NOTEBOOK, tag=False,
               not_significant_label="Not significant", width=None, height=None):
    """Polar maps in blocks of rows, each block with its own colour bars underneath, after Bracegirdle et al. (2024) Fig. 1.

    Every map of notebooks 04 and 06 is drawn by this. A typical figure has
    two blocks: hist-nat and an experiment (the values, on one set of scales),
    then their change (on another). Within a block, neighbouring columns that
    share a Scale share one colour bar; panels are dotted where a mask is True,
    and the dots get a legend at the bottom.

    Args:
        blocks (Sequence[Block]): The blocks, top to bottom. Every row has the same number of columns.
        title (str | None): Figure title, wrapped to the figure's width; none for a paper figure
            (whose caption says it).
        column_titles (Sequence[str] | None): Above the top row; the first row's variable names by default.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels a), b), c), ... row by row.
        not_significant_label (str): Legend entry for the dots.
        width (float | None): The figure's width in inches (``PAGE_WIDTH`` for print): the panels are
            sized to fill it. ``panel_size`` is used if None.
        height (float | None): The most the figure can be tall, in inches (``PAGE_HEIGHT`` for print). If
            the panels would make it taller they are made smaller, and the figure narrower than ``width``.

    Returns:
        core.Panels: ``axes`` is (every row, column); ``extras["cbars"]`` the colour bars.
    """
    import cartopy.crs as ccrs
    from matplotlib.gridspec import GridSpec
    from plotting_modules import maps  # needs cartopy

    first_names, first_fields = _fields(next(iter(blocks[0].rows.values())))
    n_cols = len(first_fields)
    column_titles = list(column_titles) if column_titles is not None else [str(n) for n in first_names]
    labels = [label for block in blocks for label in block.rows]
    dotted = any(block.not_significant for block in blocks)

    #(c): Every length in inches: room on the left for the longest row label
    longest = max((len(line) for label in labels for line in str(label).splitlines()), default=0)
    left = 0.15 + (0.6 * style.label_size / 72 * longest + 0.15 if longest else 0)
    right = 0.15
    if width is not None:
        panel_size = (width - left - right - (n_cols - 1) * core.WSPACE) / n_cols

    #(c): The grid's rows: the panels (with gaps between them), then under each block a gap, its colour bars, and
    #(c): room for their ticks and label. The panels' height is left as None until the rest is known, so that a
    #(c): page height can set it
    #(c): Colour-bar labels are wrapped to the bars at this panel size, before a page height can shrink them
    label_panel_size = panel_size
    heights, kinds = [], []
    for b, block in enumerate(blocks):
        tick_lines = max((_lines(label) for scale in block.scales for label in (scale.ticklabels or ["0"])), default=1)
        label_lines = max(_lines(_bar_label(block.scales[first].label, last - first + 1, label_panel_size, style))
                          for first, last in _scale_groups(block.scales))
        keyed = [scale for scale in block.scales if scale.key]
        for i in range(len(block.rows)):
            if i:
                heights.append(core.HSPACE)
                kinds.append(None)
            heights.append(None)
            kinds.append(("panel", b, i))
        #(c): A grid key is a few cells tall, where a colour bar is a thin strip
        key_height = max((len(scale.key[0]) * 2.2 * _text_height(style.tick_size) for scale in keyed), default=0)
        heights += [0.12, max(style.cbar_height, key_height),
                    _text_height(style.tick_size, tick_lines) + _text_height(style.cbar_label_size, label_lines) + 0.12
                    + (0.15 if b < len(blocks) - 1 else 0)]
        kinds += [None, ("cbar", b), None]
    bottom = _text_height(style.cbar_label_size) + 0.2 if dotted else 0.05
    n_panel_rows = heights.count(None)

    #(c): The title is wrapped to the figure's width, and takes room at the top, which a page height takes from the
    #(c): panels, which narrows the figure: twice round settles it
    unwrapped = title
    for _ in range(2):
        figure_width = left + n_cols * panel_size + (n_cols - 1) * core.WSPACE + right
        if unwrapped:
            title = textwrap.fill(unwrapped, width=max(int(figure_width * 72 / (0.62 * style.title_size)), 30))
        top = 0.1 + _text_height(style.label_size, max(_lines(t) for t in column_titles)) + 0.12 \
            + (_text_height(style.title_size, _lines(title)) + 0.1 if title else 0)
        fixed = top + bottom + sum(h for h in heights if h is not None)
        if height is not None:
            panel_size = min(panel_size, (height - fixed) / n_panel_rows)
    heights = [panel_size if h is None else h for h in heights]
    width = left + n_cols * panel_size + (n_cols - 1) * core.WSPACE + right
    height = fixed + n_panel_rows * panel_size

    with plt.rc_context(style.rc):
        fig = plt.figure(figsize=(width, height))
        grid = GridSpec(len(heights), n_cols, figure=fig, height_ratios=heights, hspace=0,
                        wspace=core.WSPACE / panel_size, left=left / width, right=1 - right / width,
                        top=1 - top / height, bottom=bottom / height)
        axes = np.empty((len(labels), n_cols), dtype=object)
        artists = np.full((len(labels), n_cols), None, dtype=object)
        cbars = []
        row_index = 0
        for b, block in enumerate(blocks):
            block_rows = []
            for i, (label, row) in enumerate(block.rows.items()):
                names, fields = _fields(row)
                masks = _masks((block.not_significant or {}).get(label), names)
                grid_row = kinds.index(("panel", b, i))
                for j, (field, scale, mask) in enumerate(zip(fields, block.scales, masks)):
                    ax = fig.add_subplot(grid[grid_row, j], projection=ccrs.SouthPolarStereo())
                    axes[row_index, j] = ax
                    if field is None or not bool(field.notnull().any()):
                        maps.setup_polar_ax(ax)
                        continue
                    artists[row_index, j] = maps.draw_polar_contour(ax, field, scale.levels, scale.cmap, scale.norm,
                                                                    discrete=scale.discrete, extend=scale.extend)
                    if mask is not None and bool(mask.any()):
                        maps.plot_hatch(ax, mask, hatch=NOT_SIGNIFICANT_HATCH, color=HATCH_COLOR,
                                        linewidth=style.hatch_width)
                block_rows.append(row_index)
                row_index += 1

            #(c): One colour bar per run of columns sharing a Scale, a little narrower than the panels above it
            cbar_row = kinds.index(("cbar", b))
            for first, last in _scale_groups(block.scales):
                scale = block.scales[first]
                drawn = [a for a in artists[block_rows, first:last + 1].ravel() if a is not None]
                if not drawn:
                    continue
                cax = fig.add_subplot(grid[cbar_row, first:last + 1])
                box = cax.get_position()
                if scale.key:
                    cbars.append(_draw_key(cax, scale, drawn[0], style))
                    continue
                inset = 0.06 * box.width
                thickness = style.cbar_height / height
                cax.set_position([box.x0 + inset, box.y1 - thickness, box.width - 2 * inset, thickness])
                cbars.append(core.add_colorbar(
                    fig, drawn[0], cax, levels=scale.levels, label=_bar_label(scale.label, last - first + 1, label_panel_size, style),
                    ticks=None if scale.discrete else _ticks(scale, last - first + 1, style.ticks_per_column),
                    labelsize=style.tick_size, fontsize=style.cbar_label_size,
                    discrete=scale.discrete, ticklabels=scale.ticklabels,
                ))

        core.label_cols(axes[:1], column_titles, fontsize=style.label_size, pad=0.6 * style.label_size)
        core.label_rows(axes, labels, fontsize=style.label_size)
        if tag:
            core.tag_panels(axes, fontsize=style.tag_size)
        if title:
            fig.text(0.5, 1 - 0.1 / height, title, ha="center", va="top", fontsize=style.title_size,
                     fontweight="bold", color=INK)
        if dotted:
            fig.legend(handles=[Patch(facecolor="white", edgecolor="0.4", hatch=NOT_SIGNIFICANT_HATCH,
                                      label=not_significant_label)],
                       loc="lower center", frameon=False, handlelength=2.2, handleheight=1.2,
                       bbox_to_anchor=(0.5, 0.0), fontsize=style.cbar_label_size)
    return core.Panels(fig=fig, axes=axes, artists=artists, extras={"cbars": cbars})


@plot("figure")
def field_grid(columns, title=None, row_dim=None, row_labels=None, panel_size=2.5, style=NOTEBOOK, tag=False,
               not_significant_label="Not significant", width=None, height=None):
    """Polar maps side by side, one column per field; neighbouring columns that share a Scale share a colour bar.

    One ``block_grid`` block, made from columns: its rows are the values of
    ``row_dim`` (or one row).

    Args:
        columns (Sequence[Column]): The columns, left to right. With ``row_dim``, each field
            has that dim, with the same values in every column (the first column's set the rows).
        title (str | None): Figure title.
        row_dim (str | None): Dim mapped to rows; one row if None.
        row_labels (dict | None): Label for each row value; the values themselves by default.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels.
        not_significant_label (str): Legend entry for the dots.
        width (float | None): The figure's width in inches; the panels fill it.
        height (float | None): The most the figure can be tall, in inches.

    Returns:
        core.Panels
    """
    values = [None] if row_dim is None else list(columns[0].data[row_dim].values)

    def pick(da, value):
        return da if da is None or value is None else da.sel({row_dim: value})

    labels = ["" if value is None else str((row_labels or {}).get(value, value)) for value in values]
    dotted = any(column.not_significant is not None for column in columns)
    block = Block(
        rows={label: [pick(column.data, value) for column in columns] for label, value in zip(labels, values)},
        scales=[column.scale for column in columns],
        not_significant={label: [pick(column.not_significant, value) for column in columns]
                         for label, value in zip(labels, values)} if dotted else None,
    )
    return block_grid([block], title=title, column_titles=[column.title for column in columns],
                      panel_size=panel_size, style=style, tag=tag, not_significant_label=not_significant_label,
                      width=width, height=height)


@dataclass
class Row:
    """One row of a ``row_grid``: one quantity, in every column (season).

    Args:
        data (xr.DataArray): The field on (col_dim, lat, lon), or (row_dim, col_dim, lat, lon) for a
            row per value of ``row_grid``'s ``row_dim``.
        scale (Scale): Its colour scale.
        title (str): Row label ('\\n' for a second line).
        not_significant (xr.DataArray | None): Boolean, the same dims: dotted where True.
    """

    data: xr.DataArray
    scale: Scale
    title: str
    not_significant: xr.DataArray = None


@plot("figure")
def row_grid(rows, title=None, row_dim=None, row_labels=None, col_dim="season", col_labels=None, panel_size=2.3,
             style=NOTEBOOK, tag=False, not_significant_label="Not significant", width=None, height=None):
    """Polar maps in rows, one column per season; neighbouring rows that share a Scale share a colour bar.

    ``field_grid`` on its side: there each column is a quantity, here each row
    is, so the seasons (or any ``col_dim``) run across every figure. Rows given
    the same Scale object one after another are one ``block_grid`` block, with
    one colour bar under it, as wide as the figure. For example hist-nat's
    mean, an experiment's mean (one scale, one bar), then the change (another)::

        DJF  MAM  JJA  SON
        [ ]  [ ]  [ ]  [ ]   hist-nat
        [ ]  [ ]  [ ]  [ ]   experiment
        ===== temperature ====
        [ ]  [ ]  [ ]  [ ]   change
        ====== change =======

    Args:
        rows (Sequence[Row]): The rows, top to bottom. With ``row_dim``, each Row is one row per value of
            that dim, labelled with its title and then the value.
        title (str | None): Figure title.
        row_dim (str | None): Dim each Row is split along, e.g. "experiment"; one row per Row if None.
        row_labels (dict | None): Label for each value of ``row_dim``; the values themselves by default.
        col_dim (str): Dim mapped to columns, the same values in every Row (the first Row's set them).
        col_labels (dict | None): Label for each column value; the values by default.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels.
        not_significant_label (str): Legend entry for the dots.
        width (float | None): The figure's width in inches; the panels fill it.
        height (float | None): The most the figure can be tall, in inches.

    Returns:
        core.Panels
    """
    columns = list(rows[0].data[col_dim].values)
    blocks = []
    for row in rows:
        values = [None] if row_dim is None else list(row.data[row_dim].values)
        for value in values:
            selection = {} if value is None else {row_dim: value}
            value_label = None if value is None else str((row_labels or {}).get(value, value))
            label = "\n".join(text for text in (row.title, value_label) if text)
            fields = [row.data.sel({**selection, col_dim: column}) for column in columns]

            #(c): A new block wherever the scale changes; the same scale as the row above joins its block
            if not blocks or blocks[-1].scales[0] is not row.scale:
                blocks.append(Block(rows={}, scales=[row.scale] * len(columns)))
            block = blocks[-1]
            if label in block.rows:
                raise ValueError(f"two rows of one block are labelled {label!r}")
            block.rows[label] = fields
            if row.not_significant is not None:
                block.not_significant = block.not_significant or {}
                block.not_significant[label] = [row.not_significant.sel({**selection, col_dim: column})
                                                for column in columns]

    return block_grid(blocks, title=title, column_titles=[str((col_labels or {}).get(c, c)) for c in columns],
                      panel_size=panel_size, style=style, tag=tag, not_significant_label=not_significant_label,
                      width=width, height=height)


@plot("figure")
def map_grid(da, scale, title=None, row_dim="model", col_dim="experiment", not_significant=None, row_labels=None,
             col_labels=None, panel_size=2.3, style=NOTEBOOK, tag=False, not_significant_label="Not significant",
             width=None, height=None):
    """One quantity over two dims, e.g. every model (rows) and experiment (columns), dotted where not significant.

    A ``field_grid`` with one column per value of ``col_dim``, all on one Scale, so one colour bar.

    Args:
        da (xr.DataArray): On (row_dim, col_dim, lat, lon), or (col_dim, lat, lon) without ``row_dim``.
        scale (Scale): Its colour scale.
        title (str): Figure title.
        row_dim (str | None): Dim mapped to rows; one row if None.
        col_dim (str): Dim mapped to columns.
        not_significant (xr.DataArray | None): Boolean, the same dims: dotted where True.
        row_labels, col_labels (dict | None): Label for each row and column value; the values by default.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels.
        not_significant_label (str): Legend entry for the dots.
        width (float | None): The figure's width in inches; the panels fill it.
        height (float | None): The most the figure can be tall, in inches.

    Returns:
        core.Panels
    """
    columns = [
        Column(da.sel({col_dim: value}), scale, str((col_labels or {}).get(value, value)),
               None if not_significant is None else not_significant.sel({col_dim: value}))
        for value in da[col_dim].values
    ]
    return field_grid(columns, title, row_dim=row_dim, row_labels=row_labels, panel_size=panel_size, style=style,
                      tag=tag, not_significant_label=not_significant_label, width=width, height=height)


@plot("figure")
def count_grid(counts, n_models, title, label, above=(), row_dim="direction", col_dim="experiment", row_labels=None,
               group_dim=None, signed=True, panel_size=2.3, style=NOTEBOOK, tag=False, width=None, height=None):
    """How many models: the response itself first (``above``, e.g. the multi-model median), then a row of counts each.

    With ``signed`` (the default) the counts are of a significant increase
    (the first value of ``row_dim``, reds) and a significant decrease (the
    second, blues), and the decreases are drawn as negative numbers, so that
    one ``count_scale`` serves both rows. Otherwise each row is a count with
    no sign (e.g. models with |S/N| above each threshold) on one
    ``majority_scale``. Either way the colour bar's labels are the counts
    themselves, and its hue changes where more than half the models agree.

    Args:
        counts (xr.DataArray): Model counts on (row_dim, col_dim, lat, lon), e.g.
            ``significance.significant_counts`` output.
        n_models (int): The most models there can be, the end of the colour bar.
        title (str): Figure title.
        label (str): Colour-bar label: what the counts count.
        above (Sequence[Row]): Rows drawn above the counts, each with its own colour bar.
        row_dim, col_dim (str): Dims mapped to the rows of counts and to the columns.
        group_dim (str | None): Dim the rows of counts are repeated for, e.g. "experiment", each row labelled
        with its value then its row label; one set of rows if None.
        row_labels (dict | None): Label for each row value, e.g. {"increase": "Significant increase"}.
        signed (bool): Increase and decrease rows on one diverging scale; False for counts with no sign.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels.
        width (float | None): The figure's width in inches; the panels fill it.
        height (float | None): The most the figure can be tall, in inches.

    Returns:
        core.Panels
    """
    if signed:
        increase, decrease = counts[row_dim].values
        drawn = xr.concat([counts.sel({row_dim: [increase]}), -counts.sel({row_dim: [decrease]})], dim=row_dim)
        scale = count_scale(n_models, label)
    else:
        drawn = counts
        scale = majority_scale(n_models, label)
    groups = [None] if group_dim is None else list(drawn[group_dim].values)
    count_rows = []
    for group in groups:
        selection = {} if group is None else {group_dim: group}
        for value in drawn[row_dim].values:
            row_label = str((row_labels or {}).get(value, value))
            label = row_label if group is None else f"{group}\n{row_label}"
            count_rows.append(Row(drawn.sel({**selection, row_dim: value}), scale, label))
    return row_grid([*above, *count_rows], title, col_dim=col_dim, panel_size=panel_size, style=style, tag=tag,
                    width=width, height=height)


# ---------------------------------------------------------------------------
# 4. Lines and dots: the change at every quantile, and area means
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
