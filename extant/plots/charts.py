"""Every plot that isn't a map (lines, dots, bars, histograms): make the axes from a GridSpec, then draw on them.

A figure's panels come from a GridSpec at the call site, ``fig.add_gridspec(n_rows, n_cols).subplots(...)``, in a
figure made with ``layout='constrained'`` (which makes room for the labels, titles, legends and colour bars) and
sized by ``chart_figure_size``. The GridSpec can be cut up as the figure needs: a row for legends
(``grid[1]``), a block of panels next to a map (``grid[0].subgridspec(...)``). What goes on each panel is drawn at
the call site, with this module's ``plot_...(..., ax=ax)`` and ``..._grid(..., axes=axes)``, another module's
one-axis ``plot_...``, or ``ax.plot``. Each ``plot_...`` styles the panel it draws on (``style_chart_ax``). The
text's sizes come from the style the figure is drawn in, ``forced_response.NOTEBOOK``, ``PAPER`` or ``SLIDES``: a
notebook sets ``plt.rcParams.update(fr.NOTEBOOK.rc)`` once in its Setup, and a paper or slide figure is drawn inside
``with plt.rc_context(fr.PAPER.rc):``. The lines that recur (each member, the ensemble mean, ERA5) have their
settings here, passed with ``**``, so they look the same in every figure. For example, each model's seasonal cycle
of sea-ice extent against ERA5's, in two rows of three: the models' members and their mean (a panel per model of
/<model>/historical), then ERA5 on every panel::

    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=2, n_cols=3, n_legends=1), layout='constrained')
    axes = fig.add_gridspec(2, 3).subplots(sharex=True, sharey=True)
    charts.lines_grid(cycle_tree, 'month', axes=axes, variable='extent', **charts.MEMBER_LINE)
    charts.lines_grid(cycle_tree.mean('member'), 'month', axes=axes, variable='extent', **charts.ENSEMBLE_MEAN_LINE)
    for ax in axes.ravel():
        charts.plot_lines(era5_da, 'month', ax=ax, **charts.ERA5_LINE)
    fig.legend(handles=[Line2D([], [], **charts.MEMBER_LINE), Line2D([], [], **charts.ERA5_LINE)],
               loc='outside lower center', ncols=2, frameon=False)
    fig.suptitle('Seasonal cycle of sea-ice extent', x=0.01, ha='left')

A colour bar goes under the panels with ``fig.colorbar(image, ax=axes, location='bottom', shrink=0.6)``. Two legends,
one above the other, each go in a row of the GridSpec (the layout gives each row the room its legend needs)::

    grid = fig.add_gridspec(3, 1, height_ratios=[1, 0.01, 0.01])
    axes = grid[0].subgridspec(1, 4).subplots(sharex=True, sharey=True, squeeze=False)
    legend_axes = [fig.add_subplot(grid[row]) for row in (1, 2)]
    for legend_ax in legend_axes:
        legend_ax.axis('off')
    legend_axes[0].legend(handles=forcing_handles, loc='center', ncols=4, frameon=False)
    legend_axes[1].legend(handles=model_handles, loc='center', ncols=6, frameon=False)

Each panel's title goes on its left (``loc='left'``) and the figure's title at the top left (``x=0.01, ha='left'``);
both are bold, at the style's sizes.

Sections
--------
1. Layout: the figure's size (``chart_figure_size``, or ``panel_width_to_fit`` for a printed page), and the GridSpec's
   room in inches in a figure with maps beside the charts, which isn't constrained (``margins``)
2. One panel's style (``style_chart_ax``), what recurs (each member, the ensemble mean, ERA5, each model, each
   forcing) and the legend entries for it, and the panels' letters (``tag_panels``)
3. Layers along an x axis: ``plot_lines``, ``plot_band``, ``plot_quantile_bands`` (a plume) and
   ``plot_significant_dots`` on one panel, and each one's ``..._grid`` (one panel per leaf of a DataTree, or per value
   of a row dim and a column dim, as the maps' ``antarctica.heatmaps_of_antarctica``); ``grid_panels`` and
   ``grid_axes`` say which data goes on which panel, for every ``..._grid``
4. Models and forcings: ``plot_model_dots`` (each model's value in each category, and their median; and its grid)
   and ``plot_scatter_by_forcing`` (one change against another, a point per model and forcing)

Each layer comes as ``plot_<layer>(data, ax=None, ...)``, which styles one panel and draws on it (a new figure if
``ax`` is None), and ``<layer>_grid(data, row_dim, col_dim, axes=None, variable=None, ...)``, which draws it on every
panel of a grid; nothing draws two layers at once. Rows of different kinds are slices of the same axes (``axes[2]``,
``axes[:, 0]``), each drawn by its own call.
"""

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from plotting_modules.constants import FORCING_COLORS

# ---------------------------------------------------------------------------
# 1. Layout
# ---------------------------------------------------------------------------

#(t): Sizes in inches, from which a figure's size is worked out: each panel's width and height; the gaps between
#(t): columns (room for the y tick labels) and between rows (the x tick labels above and the title below); the room
#(t): left of the panels (the y tick labels and label), above them (the figure's title and the first row's titles),
#(t): under them (the x tick labels and label) and right of them; each legend's row; and each colour bar's row, its
#(t): ticks and label included (as the maps')
PANEL_WIDTH = 2.8
PANEL_HEIGHT = 2.2
COLUMN_GAP = 0.45
ROW_GAP = 0.55
LEFT_MARGIN = 0.75
TOP_MARGIN = 0.75
BOTTOM_MARGIN = 0.5
RIGHT_MARGIN = 0.15
LEGEND_HEIGHT = 0.4
COLORBAR_HEIGHT = 0.7


def chart_figure_size(n_rows, n_cols, n_legends=1, n_colorbars=0, panel_width=PANEL_WIDTH, panel_height=PANEL_HEIGHT,
                      top_margin=TOP_MARGIN, left_margin=LEFT_MARGIN, column_gap=COLUMN_GAP, row_gap=ROW_GAP):
    """The size of a figure of panels with legends (and colour bars) under them, so each panel comes out about
    ``panel_width`` × ``panel_height``.

    Only the size: the figure is made with it and ``layout='constrained'``, and its panels from a GridSpec. The
    layout shares out the room, so the margins and gaps here only decide how big the figure is; give more of one
    where the figure needs more room (long row labels on the left, text right of each panel).

    Args:
        n_rows, n_cols (int): Rows and columns of panels.
        n_legends (int): Legend rows under the panels, one below another.
        n_colorbars (int): Colour bars under the legends, one below another.
        panel_width, panel_height (float): Each panel's size (inches); ``panel_width_to_fit`` for a printed page.
        top_margin (float): The room above the panels (inches), for the title and the first row's titles; less for
            a paper figure, which has no title.
        left_margin (float): The room left of the panels (inches), for the y tick labels and label; more for long
            row labels, a model's name say.
        column_gap, row_gap (float): The gaps between columns and between rows (inches); more for text right of each
            panel, less for cells of a key.

    Returns:
        tuple: (width, height) in inches. E.g. 2 rows, 3 columns and 1 legend: (10.2, 6.6).

    Example::

        fig = plt.figure(figsize=chart_figure_size(n_rows=2, n_cols=3, n_legends=1), layout='constrained')
        axes = fig.add_gridspec(2, 3).subplots(sharex=True, sharey=True)
    """
    width = left_margin + n_cols * panel_width + (n_cols - 1) * column_gap + RIGHT_MARGIN
    height = (top_margin + n_rows * panel_height + (n_rows - 1) * row_gap + BOTTOM_MARGIN
              + n_legends * LEGEND_HEIGHT + n_colorbars * COLORBAR_HEIGHT)
    return width, height


def panel_width_to_fit(n_cols, width):
    """The panel width (inches) at which ``n_cols`` columns of panels fill a figure ``width`` wide.

    Args:
        n_cols (int): Columns of panels.
        width (float): The figure's width (inches), e.g. ``forced_response.PAGE_WIDTH`` (7.2) for a two-column page.

    Returns:
        float: E.g. 4 columns on a 7.2 in page: 1.24.
    """
    return (width - LEFT_MARGIN - RIGHT_MARGIN - (n_cols - 1) * COLUMN_GAP) / n_cols


def margins(fig, left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN):
    """A GridSpec's ``left``, ``right``, ``top`` and ``bottom`` that leave this much room round the panels (inches),
    for a figure that has maps beside its charts.

    Such a figure isn't made with ``layout='constrained'`` (there, a map's colour bar takes its room from the map:
    ``antarctica.colorbar_axis_below``), so its chart panels need room for their labels and legends set by hand. A
    legend then goes in the room below: ``fig.legend(..., loc='lower center')``.

    Args:
        fig (matplotlib.figure.Figure | matplotlib.figure.SubFigure): The figure, or its part the panels fill.
        left, right, top, bottom (float): The room on each side (inches): left for the y tick labels and label, above
            for the titles, below for the x tick labels and label, and a legend's 0.4 in.

    Returns:
        dict: left, right, top and bottom as fractions of ``fig``, for ``fig.add_gridspec(..., **margins(fig))``.

    Example, a row of four panels under maps, with a legend under them::

        maps_fig, series_fig = fig.subfigures(2, 1, height_ratios=[7.0, 2.6])
        grid = series_fig.add_gridspec(1, 4, wspace=0.15, **margins(series_fig, bottom=0.9))
        axes = grid.subplots(sharey=True, squeeze=False)
        series_fig.legend(handles=handles, loc='lower center', ncols=4, frameon=False)
    """
    #(c): Its size from its box, so that one part of a layout (a SubFigure) works too
    width, height = fig.bbox.width / fig.dpi, fig.bbox.height / fig.dpi
    return dict(left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height)


def one_panel(panel_width=PANEL_WIDTH, panel_height=PANEL_HEIGHT):
    """A new figure with one panel, for a ``plot_...`` given no axis.

    Args:
        panel_width, panel_height (float): The panel's size (inches).

    Returns:
        matplotlib.axes.Axes: The panel.
    """
    fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0, panel_width=panel_width,
                                               panel_height=panel_height), layout='constrained')
    return fig.add_gridspec(1, 1).subplots()


# ---------------------------------------------------------------------------
# 2. One panel's style, the lines that recur, and the panels' letters
# ---------------------------------------------------------------------------

def style_chart_ax(ax):
    """The style of every chart: light gridlines behind what is drawn, and no top or right spine.

    Every ``plot_...`` calls it on the panel it draws on, before drawing, so a panel from a plain GridSpec gets the
    style from whatever is drawn on it first; call it yourself on a panel drawn only with ``ax.plot``. Set on the axis
    itself, so it holds whatever is drawn on it afterwards. A panel is styled once: one already styled (its top spine
    hidden) is left as it is, so a layer drawn later doesn't bring back gridlines an earlier one turned off. The text's
    sizes and colours come from the style's rcParams (``forced_response.NOTEBOOK.rc``).

    Args:
        ax (matplotlib.axes.Axes): The axis.
    """
    #(c): Styled already, by an earlier layer
    if not ax.spines['top'].get_visible():
        return
    ax.grid(color='0.85', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


#(t): The lines that recur: each member (thin, light blue) and the ensemble mean (thick, dark blue), the ensemble's
#(t): blues of era5_evaluation (ENSEMBLE['member'], ENSEMBLE['median']); ERA5 (black with dots, as in every figure:
#(t): era5_evaluation.ERA5_COLOR, its ice edge dotted on the maps). Each has its label, so
#(t): ``Line2D([], [], **MEMBER_LINE)`` is its legend entry
MEMBER_LINE = dict(color='#86b6ef', linewidth=0.8, label='Each member')
ENSEMBLE_MEAN_LINE = dict(color='#104281', linewidth=1.8, label='Ensemble mean')
ERA5_LINE = dict(color='#111111', linewidth=1.8, marker='o', markersize=3, zorder=5, label='ERA5')

#(t): Each model's look, wherever models are told apart. The colour is the experiment's (FORCING_COLORS), and model k
#(t): (in the order of the data) is its k-th shade, light to dark (model_shades), with marker MODEL_MARKERS[k], line
#(t): style MODEL_LINESTYLES[k] and width MODEL_LINEWIDTHS[k]: so all of one experiment's models can share a panel and
#(t): still be told apart, and model k looks the same in every experiment but for the hue. The lightest shade has the
#(t): thickest line, to stay clear on white. With more than eight models, the styles repeat
MODEL_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
MODEL_LINESTYLES = ('solid', (0, (6, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), 'solid', (0, (3, 1.5)),
                    (0, (6, 2, 1, 2, 1, 2)), (0, (10, 3)))
MODEL_LINEWIDTHS = (2.6, 1.4, 2.0, 1.2, 1.0, 1.8, 1.0, 2.2)

#(t): Each signal of the attribution (notebooks 02b and 08): its forcing's colour, and for the combinations their nearest
#(t): (ALL is historical's; ANT orange, apart from it; OTH, the forcings other than GHG, a blue-grey)
SIGNAL_COLORS = {'ALL': FORCING_COLORS['historical'], 'ANT': '#d95f02', 'NAT': FORCING_COLORS['hist-nat'],
                 'GHG': FORCING_COLORS['hist-GHG'], 'AER': FORCING_COLORS['hist-aer'],
                 'O3': FORCING_COLORS['hist-totalO3'], 'OTH': '#5d6fb3'}


def model_shades(color, n_models):
    """``n_models`` shades of ``color``, light to dark, far enough apart to tell the models apart.

    The colour's hue and saturation are kept and its lightness spread from 0.75 (still clear on white) to 0.25 (dark,
    but its hue still clear; historical's black gives light to dark grey); a single model keeps the colour itself.

    Args:
        color: Any matplotlib colour, e.g. ``FORCING_COLORS['hist-GHG']``.
        n_models (int): How many shades.

    Returns:
        list[str]: The shades as hex, model k's the k-th.

    Example::

        model_shades(FORCING_COLORS['hist-GHG'], 3)    # pink, red, dark red
    """
    if n_models == 1:
        return [mcolors.to_hex(color)]
    hue, _, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return [mcolors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
            for lightness in np.linspace(0.75, 0.25, n_models)]


def model_lines(experiment, models):
    """Each model's line among the other models of ``experiment``: its shade of the experiment's colour, its line
    style and width, and its name as the label.

    Args:
        experiment (str): A key of FORCING_COLORS, e.g. 'hist-GHG'.
        models (Sequence[str]): Every model drawn, in the order of the data (model k's look is the k-th).

    Returns:
        dict: model -> dict(color, linestyle, linewidth, label): ``**`` it into ``plot_lines`` or ``lines_grid``, and
        into ``Line2D([], [], ...)`` for its legend entry. Add ``marker=MODEL_MARKERS[k]`` for points along the line.

    Example, every hist-GHG model on one panel::

        model_styles = model_lines('hist-GHG', models)
        for model in models:
            plot_lines(zonal_change_da.sel(model=model), 'lat', ax=ax, **model_styles[model])
        fig.legend(handles=[Line2D([], [], **style) for style in model_styles.values()], loc='outside lower center')
    """
    shades = model_shades(FORCING_COLORS[experiment], len(models))
    return {model: dict(color=shades[k], linestyle=MODEL_LINESTYLES[k % len(MODEL_LINESTYLES)],
                        linewidth=MODEL_LINEWIDTHS[k % len(MODEL_LINEWIDTHS)], label=str(model))
            for k, model in enumerate(models)}


def model_legend_handles(models, experiment='historical'):
    """A legend entry for each model: its marker (MODEL_MARKERS) in its shade of ``experiment``'s colour.

    Args:
        models (Sequence[str]): The models, in the order they were drawn (model k has marker k).
        experiment (str): Whose shades (``model_shades``): historical's greys by default, which show each model's
            place from light to dark wherever the panels have several colours (forcings, signals).

    Returns:
        list[Line2D]: For ``legend(handles=...)``.

    Example::

        fig.legend(handles=model_legend_handles(['CanESM5', 'MIROC6']), loc='outside lower center', ncols=2,
                   frameon=False)
    """
    colors = model_shades(FORCING_COLORS[experiment], len(models))
    return [Line2D([], [], linestyle='none', marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], markersize=5,
                   color=colors[k], label=str(model))
            for k, model in enumerate(models)]


def forcing_legend_handles(experiments):
    """A legend entry for each forcing: a line in its colour (FORCING_COLORS).

    Args:
        experiments (Sequence[str]): The experiments, e.g. ['hist-GHG', 'hist-aer', 'hist-totalO3', 'historical'].

    Returns:
        list[Line2D]: For ``legend(handles=...)``.
    """
    return [Line2D([], [], color=FORCING_COLORS[experiment], linewidth=2, label=experiment) for experiment in experiments]


def tag_panels(axes, start=0):
    """Letter the panels a), b), c), ... row by row, just above and left of each panel's top-left corner (after z: aa),
    ab), ...).

    Args:
        axes (np.ndarray): The panels, rows × columns (or any array of axes).
        start (int): The first panel's letter, counting from 0 for a: to carry on from another block's letters.

    Returns:
        int: The next letter's number, for a block that carries on.
    """
    number = start
    for ax in np.ravel(axes):
        #(c): a ... z, then aa, ab, ...
        letters = chr(ord('a') + number) if number < 26 else chr(ord('a') + number // 26 - 1) + chr(ord('a') + number % 26)
        #(c): Just above and left of the panel's top-left corner, a few points clear of it and of the panel's title
        ax.annotate(f'{letters})', (0, 1), xycoords='axes fraction', xytext=(-4, 4), textcoords='offset points',
                    ha='right', va='bottom', fontweight='bold')
        number += 1
    return number


# ---------------------------------------------------------------------------
# 3. Layers along an x axis: lines, bands and dots where significant
# ---------------------------------------------------------------------------

def grid_panels(data, row_dim=None, col_dim=None, variable=None):
    """What goes on each panel of a grid, row by row: one panel per leaf of a DataTree, or per value of ``row_dim``
    (down) and ``col_dim`` (across) of a DataArray.

    Every ``..._grid`` here gets its panels from this, so they all fill a grid the same way. A tree's leaves go in the
    tree's order, each titled with its path in the tree ('CanESM5/historical'): so a tree of six models fills two rows
    of three panels, or one row of six, whichever the axes are. A DataArray's panels go in the order of its rows' and
    columns' values, untitled: label the rows and columns at the call site.

    Args:
        data (xr.DataTree | xr.Dataset | xr.DataArray): A DataTree, each leaf holding ``variable``; or a DataArray,
            or a Dataset holding ``variable``, on (row_dim, col_dim, ...).
        row_dim, col_dim (str | None): A DataArray's dims down the rows and across the columns; None for a single
            row or column, and for a DataTree.
        variable (str | None): The variable taken from each leaf of a DataTree, or from a Dataset.

    Returns:
        tuple: A list of (DataArray, title) for each panel, row by row (the title None for a DataArray's), and the
        (n_rows, n_cols) of a figure that fits them: a tree's leaves in one row.

    Example::

        panels, shape = grid_panels(cycle_tree, variable='extent')
        panels[0]    # (CanESM5's extent on (member, month), 'CanESM5/historical'); shape (1, 6)
    """
    if isinstance(data, xr.DataTree):
        panels = [(node[variable], node.relative_to(data)) for node in data.leaves]
        return panels, (1, len(panels))
    da = data if variable is None else data[variable]
    #(c): The values down the rows and across the columns ([None] for a single row or column)
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    panels = []
    for row_value in row_values:
        for col_value in col_values:
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            panels.append((da.sel(selection), None))
    return panels, (len(row_values), len(col_values))


def grid_axes(axes, shape, n_panels):
    """The axes of a grid: those given, or a new figure's.

    Args:
        axes (np.ndarray | None): The panels, e.g. ``fig.add_gridspec(2, 3).subplots()``, or a slice of them
            (``axes[2]`` for the third row); a new figure of ``shape`` if None.
        shape (tuple): (n_rows, n_cols) of a new figure, as ``grid_panels`` gives it.
        n_panels (int): How many panels will be drawn.

    Returns:
        np.ndarray: The axes, as given or made; ``np.ravel(axes)`` goes through them row by row.
    """
    if axes is None:
        n_rows, n_cols = shape
        fig = plt.figure(figsize=chart_figure_size(n_rows, n_cols, n_legends=0), layout='constrained')
        axes = fig.add_gridspec(n_rows, n_cols).subplots(sharex=True, sharey=True, squeeze=False)
    #(c): Too few axes would leave panels out without a word
    if np.size(axes) < n_panels:
        raise ValueError(f'{n_panels} panels to draw but only {np.size(axes)} axes')
    return axes


def plot_lines(da, x, ax=None, title=None, **style):
    """``da`` against its dim ``x``: one line, or one line per value of its other dim (each member, say), on a panel
    in the chart style (``style_chart_ax``).

    Args:
        da (xr.DataArray): On (x) or (x, one other dim), e.g. a member's extent on (month), or every member's on
            (member, month).
        x (str): The dim along the x axis, e.g. 'year' or 'month'.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        title (str | None): The panel's title, on its left; the title it has is kept if None.
        **style: How the lines are drawn, e.g. ``**MEMBER_LINE`` or ``color='C0', linewidth=1.4``.

    Returns:
        list: The lines (``lines[0].axes`` is the axis).

    Example::

        lines = plot_lines(cycle_tree['CanESM5/historical']['extent'], 'month', title='CanESM5', **MEMBER_LINE)
        plot_lines(era5_cycle_da, 'month', ax=lines[0].axes, **ERA5_LINE)
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    #(c): x first, so that each column of the values (each member) is one line
    lines = ax.plot(da[x].values, da.transpose(x, ...).values, **style)
    if title is not None:
        ax.set_title(title, loc='left')
    return lines


def lines_grid(data, x, row_dim=None, col_dim=None, axes=None, variable=None, **style):
    """``plot_lines`` on every panel of a grid: one panel per leaf of a DataTree, or per value of ``row_dim`` (down)
    and ``col_dim`` (across) of a DataArray (``grid_panels``).

    The style is the same on every panel; for another layer (the members' mean, ERA5), draw again on the same axes.

    Args:
        data (xr.DataTree | xr.Dataset | xr.DataArray): What to draw. A DataTree: each leaf's ``variable``, on (x) or
            (x, one other dim), e.g. /<model>/historical with each model's members on (member, month). A DataArray,
            or a Dataset's ``variable``: on (row_dim, col_dim, x), and one more dim for several lines a panel.
        x (str): The dim along the x axis, e.g. 'month'.
        row_dim, col_dim (str | None): A DataArray's dims down the rows and across the columns, in the order of their
            values; None for a single row or column, and for a DataTree.
        axes (np.ndarray | None): The panels, at least one per leaf or per row and column, e.g.
            ``fig.add_gridspec(2, 3).subplots()``, or a slice of them (``axes[2]`` for the third row). A new figure
            if None.
        variable (str | None): The variable drawn from each leaf of a DataTree, or from a Dataset.
        **style: As ``plot_lines``', the same on every panel, e.g. ``**MEMBER_LINE`` or ``color='C0'``.

    Returns:
        np.ndarray: The panels, as given or made.

    Example, each model's members and their mean, two rows of three models, then ERA5 on every panel::

        fig = plt.figure(figsize=chart_figure_size(n_rows=2, n_cols=3, n_legends=1), layout='constrained')
        axes = fig.add_gridspec(2, 3).subplots(sharex=True, sharey=True)
        lines_grid(cycle_tree, 'month', axes=axes, variable='extent', **MEMBER_LINE)
        lines_grid(cycle_tree.mean('member'), 'month', axes=axes, variable='extent', **ENSEMBLE_MEAN_LINE)
        for ax in axes.ravel():
            plot_lines(era5_cycle_da, 'month', ax=ax, **ERA5_LINE)

    Example, each experiment's ensemble mean, seasons down and models across::

        for experiment in ['hist-GHG', 'historical']:
            lines_grid(mean_da.sel(experiment=experiment), 'year', 'season', 'model', axes=axes,
                       color=FORCING_COLORS[experiment])
    """
    panels, shape = grid_panels(data, row_dim, col_dim, variable)
    axes = grid_axes(axes, shape, len(panels))
    for ax, (panel_da, title) in zip(np.ravel(axes), panels):
        plot_lines(panel_da, x, ax=ax, title=title, **style)
    return axes


def plot_band(low_da, high_da, x, ax=None, title=None, **style):
    """The band between ``low_da`` and ``high_da`` along ``x`` (e.g. the members' 5-95%), on a panel in the chart
    style.

    Args:
        low_da, high_da (xr.DataArray): The band's edges, each on (x).
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        title (str | None): The panel's title, on its left; the title it has is kept if None.
        **style: How the band is drawn, e.g. ``color='C0', alpha=0.15``; it has no edge line unless given a
            ``linewidth``.

    Returns:
        matplotlib.collections.PolyCollection: The band (``band.axes`` is the axis).

    Example, the members' 5-95% of one model's seasonal cycle::

        members_da = cycle_tree['CanESM5/historical']['extent']
        plot_band(members_da.quantile(0.05, 'member'), members_da.quantile(0.95, 'member'), 'month', alpha=0.3)
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    #(c): No edge line unless asked: with the band's own colour it would draw a darker outline
    style = {'linewidth': 0, **style}
    band = ax.fill_between(low_da[x].values, low_da.values, high_da.values, **style)
    if title is not None:
        ax.set_title(title, loc='left')
    return band


def bands_grid(low_data, high_data, x, row_dim=None, col_dim=None, axes=None, variable=None, **style):
    """``plot_band`` on every panel of a grid: one panel per leaf of a DataTree, or per value of ``row_dim`` (down)
    and ``col_dim`` (across) of a DataArray (``grid_panels``).

    Args:
        low_data, high_data (xr.DataTree | xr.Dataset | xr.DataArray): The bands' edges, alike: two DataTrees with
            ``variable`` on (x) in each leaf, or two DataArrays (or Datasets' ``variable``) on (row_dim, col_dim, x).
        x, row_dim, col_dim, axes, variable: As ``lines_grid``'s.
        **style: As ``plot_band``'s, the same on every panel.

    Returns:
        np.ndarray: The panels, as given or made.

    Example, the range of each model's members, a panel per model::

        bands_grid(cycle_tree.min('member'), cycle_tree.max('member'), 'month', axes=axes, variable='extent',
                   color='C0', alpha=0.3)
    """
    low_panels, shape = grid_panels(low_data, row_dim, col_dim, variable)
    high_panels, _ = grid_panels(high_data, row_dim, col_dim, variable)
    axes = grid_axes(axes, shape, len(low_panels))
    for ax, (low_da, title), (high_da, _) in zip(np.ravel(axes), low_panels, high_panels):
        plot_band(low_da, high_da, x, ax=ax, title=title, **style)
    return axes


def plot_quantile_bands(quantiles_da, pairs, x, ax=None, color='C0', alpha=0.35, edge_width=0, title=None):
    """Nested bands between pairs of quantiles (a plume), on a panel in the chart style: each pair a band, the
    narrower ones over the wider, so the middle is the darkest.

    Args:
        quantiles_da (xr.DataArray): On (quantile, x), e.g. the members' quantiles each year.
        pairs (Sequence[tuple[float, float]]): (lower, upper) quantiles, widest first, e.g. [(0.05, 0.95), (0.25, 0.75)];
            each found to the nearest quantile of ``quantiles_da``.
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        color: The bands' colour.
        alpha (float): Each band's opacity; where n bands overlap the colour is 1 - (1 - alpha)^n strong.
        edge_width (float): Width of a thin line along each band's edges; none if 0.
        title (str | None): The panel's title, on its left; the title it has is kept if None.

    Returns:
        list: The bands, widest first (``bands[0].axes`` is the axis).

    Example, the plume and its median::

        plot_quantile_bands(quantiles_da, [(0.05, 0.95), (0.25, 0.75)], 'year', color='C0')
        plot_lines(quantiles_da.sel(quantile=0.5), 'year', color='k')
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    bands = []
    for lower, upper in pairs:
        lower_da = quantiles_da.sel(quantile=lower, method='nearest')
        upper_da = quantiles_da.sel(quantile=upper, method='nearest')
        bands.append(plot_band(lower_da, upper_da, x, ax=ax, color=color, alpha=alpha, zorder=1))
        if edge_width:
            for edge_da in (lower_da, upper_da):
                plot_lines(edge_da, x, ax=ax, color=color, linewidth=edge_width, alpha=0.55, zorder=2)
    if title is not None:
        ax.set_title(title, loc='left')
    return bands


def quantile_bands_grid(quantiles_data, pairs, x, row_dim=None, col_dim=None, axes=None, variable=None, color='C0',
                        alpha=0.35, edge_width=0):
    """``plot_quantile_bands`` on every panel of a grid: one panel per leaf of a DataTree, or per value of
    ``row_dim`` (down) and ``col_dim`` (across) of a DataArray (``grid_panels``).

    Args:
        quantiles_data (xr.DataTree | xr.Dataset | xr.DataArray): A DataTree with ``variable`` on (quantile, x) in
            each leaf, or a DataArray (or a Dataset's ``variable``) on (row_dim, col_dim, quantile, x).
        pairs, x, color, alpha, edge_width: As ``plot_quantile_bands``', the same on every panel.
        row_dim, col_dim, axes, variable: As ``lines_grid``'s.

    Returns:
        np.ndarray: The panels, as given or made.

    Example, each model's members' 5-95% and 25-75%, a panel per model (a DataTree has no ``.quantile``, so it is
    mapped over the nodes, skipping the empty ones)::

        quantiles_tree = cycle_tree.map_over_datasets(
            datatree.skip_empty(lambda ds: ds.quantile([0.05, 0.25, 0.75, 0.95], 'member')))
        quantile_bands_grid(quantiles_tree, [(0.05, 0.95), (0.25, 0.75)], 'month', axes=axes, variable='extent')
    """
    panels, shape = grid_panels(quantiles_data, row_dim, col_dim, variable)
    axes = grid_axes(axes, shape, len(panels))
    for ax, (panel_da, title) in zip(np.ravel(axes), panels):
        plot_quantile_bands(panel_da, pairs, x, ax=ax, color=color, alpha=alpha, edge_width=edge_width, title=title)
    return axes


def plot_significant_dots(da, significant_da, x, ax=None, color='k', title=None):
    """A dot on ``da``'s line wherever ``significant_da`` is True (e.g. where a zonal-mean change is significant), on
    a panel in the chart style.

    Args:
        da (xr.DataArray): The line's values, on (x).
        significant_da (xr.DataArray): True/False on (x), e.g. ``pvalue_da < 0.05``.
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw, e.g. the panel the line is on; a new figure if None.
        color: The dots' colour, the line's.
        title (str | None): The panel's title, on its left; the title it has is kept if None.

    Returns:
        list: The dots (``dots[0].axes`` is the axis).

    Example::

        plot_lines(zonal_ds['mean_change'], 'lat', ax=ax, color='C3')
        plot_significant_dots(zonal_ds['mean_change'], zonal_ds['mean_pvalue'] < 0.05, 'lat', ax=ax, color='C3')
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    where_arr = significant_da.transpose(*da.dims).values.astype(bool)
    dots = ax.plot(da[x].values[where_arr], da.values[where_arr], linestyle='none', marker='o', markersize=4,
                   color=color, markeredgecolor='white', markeredgewidth=0.6, zorder=4)
    if title is not None:
        ax.set_title(title, loc='left')
    return dots


def significant_dots_grid(data, significant_data, x, row_dim=None, col_dim=None, axes=None, variable=None,
                          color='k'):
    """``plot_significant_dots`` on every panel of a grid: one panel per leaf of a DataTree, or per value of
    ``row_dim`` (down) and ``col_dim`` (across) of a DataArray (``grid_panels``).

    Args:
        data, significant_data (xr.DataTree | xr.Dataset | xr.DataArray): The lines' values and where they are
            significant, alike: two DataTrees with ``variable`` on (x) in each leaf (e.g. ``change_tree`` and
            ``pvalue_tree < 0.05``), or two DataArrays (or Datasets' ``variable``) on (row_dim, col_dim, x).
        x, row_dim, col_dim, axes, variable: As ``lines_grid``'s.
        color: As ``plot_significant_dots``', the same on every panel.

    Returns:
        np.ndarray: The panels, as given or made.
    """
    panels, shape = grid_panels(data, row_dim, col_dim, variable)
    significant_panels, _ = grid_panels(significant_data, row_dim, col_dim, variable)
    axes = grid_axes(axes, shape, len(panels))
    for ax, (panel_da, title), (significant_da, _) in zip(np.ravel(axes), panels, significant_panels):
        plot_significant_dots(panel_da, significant_da, x, ax=ax, color=color, title=title)
    return axes


# ---------------------------------------------------------------------------
# 4. Models and forcings
# ---------------------------------------------------------------------------

def plot_model_dots(da, x_dim, ax=None, color='0.35', title=None):
    """Each model's value at each value of ``x_dim``: a marker per model, spread a little sideways, and a bar for the
    multi-model median, on a panel in the chart style.

    The quickest view of how big a change is in each category (a forcing, a zone) and how far the models agree.
    Model k has marker ``MODEL_MARKERS[k]`` and the k-th shade of its category's colour (``model_shades``;
    ``model_legend_handles`` for the legend).

    Args:
        da (xr.DataArray): On (model, x_dim), e.g. each model's area-mean change for each forcing.
        x_dim (str): The categories along the x axis, in the order of their values, labelled with them.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        color: The markers' and bars' colour: one for all, or a dict of one per category (e.g. ``FORCING_COLORS``
            when the categories are forcings; grey for a category it lacks).
        title (str | None): The panel's title, on its left; the title it has is kept if None.

    Returns:
        matplotlib.axes.Axes: The axis.

    Example, one model's change for each forcing, in its colour::

        plot_model_dots(change_da.sel(season='JJA'), 'experiment', color=FORCING_COLORS)
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    x_values = list(da[x_dim].values)
    models = list(da['model'].values)
    #(c): Each model's offset from the middle of its category
    offsets_arr = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else np.zeros(1)
    ax.axhline(0, color='0.6', linewidth=0.6, zorder=1)
    for x, x_value in enumerate(x_values):
        values_da = da.sel({x_dim: x_value})
        if not bool(values_da.notnull().any()):
            continue
        #(c): One colour, or this category's
        category_color = color.get(x_value, '0.35') if isinstance(color, dict) else color
        ax.bar(x, float(values_da.median('model')), width=0.7, color=category_color, alpha=0.3, linewidth=0, zorder=2)
        shades = model_shades(category_color, len(models))
        for k, model in enumerate(models):
            ax.plot(x + offsets_arr[k], float(values_da.sel(model=model)), linestyle='none',
                    marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], markersize=4, color=shades[k],
                    markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    #(c): Each category's name on two lines, broken at its first space, so long names fit
    ax.set_xticks(range(len(x_values)), [str(value).replace(' ', '\n', 1) for value in x_values])
    ax.grid(axis='x', visible=False)
    if title is not None:
        ax.set_title(title, loc='left')
    return ax


def model_dots_grid(data, x_dim, row_dim=None, col_dim=None, axes=None, variable=None, color='0.35'):
    """``plot_model_dots`` on every panel of a grid: one panel per leaf of a DataTree, or per value of ``row_dim``
    (down) and ``col_dim`` (across) of a DataArray (``grid_panels``).

    Args:
        data (xr.DataTree | xr.Dataset | xr.DataArray): A DataTree with ``variable`` on (model, x_dim) in each leaf,
            or a DataArray (or a Dataset's ``variable``) on (row_dim, col_dim, model, x_dim).
        x_dim, color: As ``plot_model_dots``', the same on every panel: for a colour per row (each experiment's), one
            call per row.
        row_dim, col_dim, axes, variable: As ``lines_grid``'s.

    Returns:
        np.ndarray: The panels, as given or made.

    Example, experiments down and seasons across, each row in its experiment's colour::

        for row_axes, experiment in zip(axes, EXPERIMENTS):
            model_dots_grid(zone_means_da.sel(experiment=experiment), 'zone', None, 'season', axes=row_axes,
                            color=FORCING_COLORS[experiment])
    """
    panels, shape = grid_panels(data, row_dim, col_dim, variable)
    axes = grid_axes(axes, shape, len(panels))
    for ax, (panel_da, title) in zip(np.ravel(axes), panels):
        plot_model_dots(panel_da, x_dim, ax=ax, color=color, title=title)
    return axes


def plot_scatter_by_forcing(x_da, y_da, ax=None):
    """One change against another, one point per model and forcing, on a panel in the chart style: the forcing its
    colour, the model its marker and its shade of that colour (``model_shades``).

    The correlation over all the points is at the bottom of the panel, and the zero lines are grey.
    ``forcing_legend_handles`` and ``model_legend_handles`` give the legend.

    Args:
        x_da, y_da (xr.DataArray): On (model, experiment), e.g. one season's change in extent and change in the mean
            temperature over the ice.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.

    Example, one panel per season::

        for ax, season in zip(axes[0], SEASONS):
            plot_scatter_by_forcing(extent_change_da.sel(season=season), mean_change_da.sel(season=season), ax=ax)
    """
    if ax is None:
        ax = one_panel()
    style_chart_ax(ax)
    #(c): Only the models and forcings both have, in the same order
    x_da, y_da = xr.align(x_da, y_da, join='inner')
    y_da = y_da.transpose(*x_da.dims)
    ax.axhline(0, color='0.6', linewidth=0.6, zorder=0)
    ax.axvline(0, color='0.6', linewidth=0.6, zorder=0)
    models = x_da['model'].values
    for experiment in x_da['experiment'].values:
        shades = model_shades(FORCING_COLORS[str(experiment)], len(models))
        for k, model in enumerate(models):
            ax.plot(float(x_da.sel(model=model, experiment=experiment)), float(y_da.sel(model=model, experiment=experiment)),
                    linestyle='none', marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], markersize=5,
                    color=shades[k], markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    #(c): The correlation over every point that has both values, at the bottom in the middle (the corners are left for
    #(c): what each quadrant means, response_change.plot_quadrant_labels)
    valid_arr = np.isfinite(x_da.values) & np.isfinite(y_da.values)
    if valid_arr.sum() > 2:
        r = np.corrcoef(x_da.values[valid_arr], y_da.values[valid_arr])[0, 1]
        ax.text(0.5, 0.02, f'r = {r:.2f}', transform=ax.transAxes, ha='center', va='bottom')
    return ax
