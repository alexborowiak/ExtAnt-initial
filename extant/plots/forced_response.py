"""The figures of notebooks 04 and 06. Every section of 04 draws the same four kinds of figure:

1. What goes in       ``field_grid``: the quantities a result is made from, then the result,
                      for one model, experiment and season
2. One model          ``field_grid``: the parts and the result for every experiment of one model (rows)
3. Every model        ``map_grid``: the result, models down and experiments across
4. How many models    ``count_grid``: how many models have a significant increase (red) or decrease (blue)

All four are grids of polar maps drawn by ``block_grid``, as are notebook 06's
paper figures (``PAPER`` style): blocks of rows, each with its colour bars
underneath, after Bracegirdle et al. (2024) Fig. 1. Columns that share a
``Scale`` share one colour bar. Colour and pattern mean the same thing in every
figure:

    one quantity      one ``Scale`` (levels, colormap, label), from ``colour_scales``
    red <-> blue      an increase <-> a decrease
    dots              not significant (one model), or not robust (fewer than 2/3 of the models significant)
    white             |S/N| below the threshold (``threshold_scale``), or no model (``count_scale``)
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


def threshold_scale(threshold, limit, label, step=1, cmap="RdBu_r"):
    """A diverging scale that is white between -``threshold`` and ``threshold`` (e.g. S/N, white where |S/N| < 2).

    Levels run from -``limit`` to -``threshold`` and from ``threshold`` to
    ``limit`` every ``step``; the one band between ±``threshold`` is white, and
    the bands either side darken away from it.
    """
    levels = np.concatenate([np.arange(-limit, -threshold + step / 2, step), np.arange(threshold, limit + step / 2, step)])
    n_side = (len(levels) - 2) // 2
    base = plt.get_cmap(cmap)
    colors = ListedColormap([*base(np.linspace(0, 0.4, n_side)), "white", *base(np.linspace(0.6, 1, n_side))])
    return Scale(levels, colors, label, norm=BoundaryNorm(levels, colors.N), ticks=levels)


def pvalue_scale(alpha=0.05, label=None):
    """p-values in four bands: below alpha / 5, below alpha (both purple: significant), below 2 alpha (grey), above (white)."""
    levels = np.array([0, alpha / 5, alpha, 2 * alpha, 1])
    colors = ListedColormap(["#3f007d", "#9e9ac8", "#d9d9d9", "white"])
    return Scale(levels, colors, label or f"p-value (significant below {alpha:g})",
                 norm=BoundaryNorm(levels, colors.N), ticks=levels)


def count_scale(n_models, label="Number of models with a significant decrease (blue) or increase (red)"):
    """Discrete and diverging: ``n_models`` with a significant decrease (dark blue), down to none (white), up to an increase.

    ``count_grid`` draws the decreases as negative numbers, so this one colour
    bar serves both of its rows; the tick labels are the counts themselves.
    """
    blues = plt.get_cmap("Blues")(np.linspace(1, 0.35, n_models))
    reds = plt.get_cmap("Reds")(np.linspace(0.35, 1, n_models))
    colors = ListedColormap([*blues, "white", *reds])
    levels = np.arange(-n_models - 0.5, n_models + 1)
    return Scale(levels, colors, label, discrete=True,
                 ticklabels=[str(abs(k)) for k in range(-n_models, n_models + 1)])


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


def colour_scales(variable, alpha=0.05, sn_threshold=2):
    """One colour scale per quantity for ``variable``, shared by every figure that shows that quantity.

    Args:
        variable (str): Key of ``config.VARIABLES`` and of ``RANGES``.
        alpha (float): Significance level, for the p-value scale.
        sn_threshold (float): |S/N| counted as emerged: white below it.

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
        temperature=Scale(np.arange(start, stop + step / 2, step), "viridis", var.label,
                          ticks=np.arange(start, stop + step / 2, tick)),
        change=Scale(_symmetric(*ranges["change"]), "RdBu_r", f"Change ({units})"),
        noise=Scale(_from_zero(*ranges["noise"]), "viridis", f"Standard deviation ({units})", extend="max"),
        spread=Scale(_from_zero(*ranges["spread"]), "viridis", f"Spread ({units})", extend="max"),
        spread_change=Scale(_symmetric(*ranges["spread_change"]), "RdBu_r", f"Change in spread ({units})"),
        tail=Scale(_symmetric(*ranges["spread_change"]), "RdBu_r", f"Upper-tail minus lower-tail change ({units})"),
        sn=threshold_scale(sn_threshold, 6, "S/N"),
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
def count_grid(counts, n_models, title, label, row_dim="direction", col_dim="experiment", row_labels=None,
               panel_size=2.5, style=NOTEBOOK, tag=False, width=None, height=None):
    """How many models have a significant increase (top row, reds) and a significant decrease (bottom row, blues).

    The decreases are drawn as negative numbers, so that one ``count_scale``
    serves both rows; the colour bar's labels are the counts themselves.

    Args:
        counts (xr.DataArray): Model counts on (row_dim, col_dim, lat, lon), the increase first and the
            decrease second, e.g. ``significance.significant_counts`` output.
        n_models (int): The most models there can be, the end of the colour bar.
        title (str): Figure title.
        label (str): Colour-bar label: what blue and red count.
        row_dim, col_dim (str): Dims mapped to rows and columns.
        row_labels (dict | None): Label for each row value, e.g. {"increase": "Significant increase"}.
        panel_size (float): Width and height of each panel, in inches.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        tag (bool): Letter the panels.
        width (float | None): The figure's width in inches; the panels fill it.
        height (float | None): The most the figure can be tall, in inches.

    Returns:
        core.Panels
    """
    increase, decrease = counts[row_dim].values
    signed = xr.concat([counts.sel({row_dim: [increase]}), -counts.sel({row_dim: [decrease]})], dim=row_dim)
    return map_grid(signed, count_scale(n_models, label), title, row_dim=row_dim, col_dim=col_dim,
                    row_labels=row_labels, panel_size=panel_size, style=style, tag=tag, width=width, height=height)


# ---------------------------------------------------------------------------
# 4. Lines and dots: the change at every quantile, and area means
# ---------------------------------------------------------------------------

@plot("figure")
def quantile_curves(curves, title=None, units="°C", col_dim="experiment", style=NOTEBOOK, panel_size=(3.3, 3.0),
                    width=None):
    """The change at every quantile, one line per model, one panel per experiment.

    A flat line is a pure shift of the distribution; a rising line means the
    upper quantiles move more than the lower ones.

    Args:
        curves (xr.DataArray): On (model, col_dim, quantile), e.g. notebook 03's ``quantile_curve``
            (Antarctic means) for one season.
        title (str | None): Figure title.
        units (str): Units of the change.
        col_dim (str): Dim mapped to panels.
        style (Style): ``NOTEBOOK`` or ``PAPER``.
        panel_size (tuple[float, float]): Width and height of each panel, in inches.
        width (float | None): The figure's width in inches (the legend included), instead of ``panel_size[0]``.

    Returns:
        core.Panels
    """
    models = [str(m) for m in curves["model"].values]
    panels_values = list(curves[col_dim].values)
    percentile = 100 * curves["quantile"].values

    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(1, len(panels_values), sharey=True, layout="constrained", squeeze=False,
                                 figsize=(width or panel_size[0] * len(panels_values) + 1.6, panel_size[1] + 0.6))
        for ax, value in zip(axes[0], panels_values):
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)
            for k, model in enumerate(models):
                ax.plot(percentile, curves.sel(model=model, **{col_dim: value}), color=MODEL_COLORS[k % len(MODEL_COLORS)],
                        marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=0.4 * style.tick_size,
                        lw=0.15 * style.tick_size, zorder=3)
            ax.set_title(str(value), loc="left")
            ax.set_xlabel("Percentile")
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_xticks([25, 75], minor=True)
            core.style_ax(ax)
        axes[0, 0].set_ylabel(f"Change ({units})")
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
