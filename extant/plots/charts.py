"""Every plot that isn't a map (lines, dots, bars, histograms), built like the maps: make the axes, then draw on them.

``chart_figure_size`` gives the figure's size and ``chart_axes`` its panels (each styled by ``style_chart_ax``) and
rows for legends under them. What goes on each panel is drawn at the call site, with ``ax.plot``, xarray's
``.plot``, a module's one-axis ``plot_...(..., ax=ax)`` or anything else, so the same axes take any data, a figure
can have as many rows as it needs, and a block of panels can sit in a bigger figure. The text's sizes come from the
style the figure is drawn in, ``forced_response.NOTEBOOK``, ``PAPER`` or ``SLIDES``: a notebook sets
``plt.rcParams.update(fr.NOTEBOOK.rc)`` once in its Setup, and a paper or slide figure is drawn inside
``with plt.rc_context(fr.PAPER.rc):``. The lines that recur (each member, the ensemble mean, ERA5) have their
settings here, passed with ``**``, so they look the same in every figure. For example, each model's seasonal cycle
of sea-ice extent against ERA5's, in two rows of three::

    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=2, n_cols=3, n_legends=1))
    axes, legend_axes = charts.chart_axes(fig, n_rows=2, n_cols=3, n_legends=1, sharex=True, sharey=True)
    for ax, node in zip(axes.ravel(), cycle_tree.match('*/historical').leaves):
        charts.plot_lines(node['extent'], 'month', ax=ax, **charts.MEMBER_LINE)
        charts.plot_lines(node['extent'].mean('member'), 'month', ax=ax, **charts.ENSEMBLE_MEAN_LINE)
        charts.plot_lines(era5_da, 'month', ax=ax, **charts.ERA5_LINE)
        ax.set_title(node.parent.name, loc='left')
    legend_axes[0].legend(handles=[Line2D([], [], **charts.MEMBER_LINE), Line2D([], [], **charts.ERA5_LINE)],
                          loc='center', ncols=2, frameon=False)
    fig.suptitle('Seasonal cycle of sea-ice extent', x=0.01, ha='left')

Each panel's title goes on its left (``loc='left'``) and the figure's title at the top left (``x=0.01, ha='left'``);
both are bold, at the style's sizes.

A block of panels in a figure of maps: give ``chart_axes`` one cell of the figure's grid (``subplot_spec``), as
``antarctica.map_axes`` takes another.

Sections
--------
1. Layout: the figure's size (``chart_figure_size``, or ``panel_width_to_fit`` for a printed page), then its panels
   and legend axes (``chart_axes``)
2. One panel's style (``style_chart_ax``), what recurs (each member, the ensemble mean, ERA5, each model, each
   forcing) and the legend entries for it, and the panels' letters (``tag_panels``)
3. Layers along an x axis: ``plot_lines``, ``plot_band``, ``plot_quantile_bands`` (a plume) and
   ``plot_significant_dots`` on one panel, and each one's ``..._grid`` (one panel per value of a row dim and a column
   dim, as the maps' ``antarctica.heatmaps_of_antarctica``)
4. Models and forcings: ``plot_model_dots`` (each model's value in each category, and their median; and its grid)
   and ``plot_scatter_by_forcing`` (one change against another, a point per model and forcing)

Each layer comes as ``plot_<layer>(data, ax=None, ...)``, which draws on one panel (a new figure if ``ax`` is None),
and ``<layer>_grid(data, row_dim, col_dim, axes=None, ...)``, which draws it on every panel of a grid; nothing draws
two layers at once. Rows of different kinds are slices of the same axes (``axes[2]``, ``axes[:, 0]``), each drawn by
its own call.
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

#(t): Sizes in inches: each panel's width and height; the gaps between columns (room for the y tick labels) and
#(t): between rows (the x tick labels above and the title below); the room left of the panels (the y tick labels and
#(t): label), above them (the figure's title and the first row's titles), under them (the x tick labels and label) and
#(t): right of them; each legend's row; and each colour bar's row, its ticks and label included (as the maps')
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
    """The size of a figure of panels with legends (and colour bars) under them.

    Only the size, so the figure can be made bigger for something else (maps above the panels, say) before
    ``chart_axes`` fills its part of it.

    Args:
        n_rows, n_cols (int): Rows and columns of panels.
        n_legends (int): Legend rows under the panels, one below another.
        n_colorbars (int): Colour bars under the legends, one below another.
        panel_width, panel_height (float): Each panel's size (inches); ``panel_width_to_fit`` for a printed page.
        top_margin (float): The room above the panels (inches), for the title and the first row's titles; less for
            a paper figure, which has no title (give ``chart_axes`` the same).
        left_margin (float): The room left of the panels (inches), for the y tick labels and label; more for long
            row labels, a model's name say (give ``chart_axes`` the same).
        column_gap, row_gap (float): The gaps between columns and between rows (inches); more for text right of each
            panel, less for cells of a key (give ``chart_axes`` the same).

    Returns:
        tuple: (width, height) in inches. E.g. 2 rows, 3 columns and 1 legend: (10.2, 6.6).
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


def chart_axes(fig, n_rows, n_cols, n_legends=1, n_colorbars=0, sharex=False, sharey=False, subplot_spec=None,
               top_margin=TOP_MARGIN, left_margin=LEFT_MARGIN, column_gap=COLUMN_GAP, row_gap=ROW_GAP):
    """``n_rows`` × ``n_cols`` panels, each styled by ``style_chart_ax``, and legend and colour-bar axes under them.

    The panels fill the figure, or one cell of a larger GridSpec (``subplot_spec``), leaving the rest of the figure
    for other axes. In a figure from ``chart_figure_size`` the panels come out at its panel size, and the gaps
    between them are always ``column_gap`` and ``row_gap`` (inches). A legend's axis spans the panels' width and has
    no frame or ticks: put a legend in it with ``legend_axes[0].legend(handles=..., loc='center', frameon=False)``. A
    colour bar's axis is as the maps' (``antarctica.map_axes``): the middle 60% of its row's width, 0.23 in thick, at
    the top of its row, its ticks and label below it.

    Args:
        fig (matplotlib.figure.Figure | matplotlib.figure.SubFigure): The figure, e.g.
            ``plt.figure(figsize=chart_figure_size(...))``, or one part of a layout (``fig.subfigures(2, 1)[0]``).
        n_rows, n_cols, n_legends, n_colorbars (int): As ``chart_figure_size``'s.
        sharex, sharey (bool | str): Which panels have the same x (y) axis, as in ``plt.subplots``: True for every
            panel, 'row' for the panels of each row, 'col' for those of each column, False for none. Shared axes have
            the same limits and ticks, so ticks set on one panel are set on all that share it. Sharing hides the
            tick labels between the panels, as ``plt.subplots`` does: the x tick labels of all but the bottom row
            (``sharex`` True or 'col'), the y tick labels of all but the first column (``sharey`` True or 'row').
        subplot_spec (matplotlib.gridspec.SubplotSpec | None): The part of the figure to fill, e.g. ``grid[1]`` of
            ``fig.add_gridspec(2, 1)``; the whole figure if None.
        top_margin, left_margin, column_gap, row_gap (float): As given to ``chart_figure_size`` (inches).

    Returns:
        tuple: The panels, an array of ``n_rows`` × ``n_cols``, and a list of the legend axes, top first, then the
        colour bars' axes.

    Example, maps above a row of panels::

        map_width, map_height = antarctica.map_figure_size(n_rows=1, n_cols=4, n_colorbars=1)
        chart_width, chart_height = chart_figure_size(n_rows=1, n_cols=4, n_legends=1)
        fig = plt.figure(figsize=(max(map_width, chart_width), map_height + chart_height))
        grid = fig.add_gridspec(2, 1, height_ratios=[map_height, chart_height], left=0, right=1, top=1, bottom=0)
        map_axes, cbar_axes = antarctica.map_axes(fig, n_rows=1, n_cols=4, n_colorbars=1, subplot_spec=grid[0])
        axes, legend_axes = chart_axes(fig, n_rows=1, n_cols=4, n_legends=1, subplot_spec=grid[1])
    """
    under_height = n_legends * LEGEND_HEIGHT + n_colorbars * COLORBAR_HEIGHT
    if subplot_spec is None:
        #(c): The whole figure, less the room left of the panels and above them. Its size from its box, so that one
        #(c): part of a layout (a SubFigure, from fig.subfigures) works too
        width, height = fig.bbox.width / fig.dpi, fig.bbox.height / fig.dpi
        #(c): Each panel's size: what is left of the figure once the margins, gaps, legends and bars are in, shared out
        panel_width = (width - left_margin - RIGHT_MARGIN - (n_cols - 1) * column_gap) / n_cols
        panel_height = (height - top_margin - BOTTOM_MARGIN - under_height - (n_rows - 1) * row_gap) / n_rows
    else:
        #(c): In part of a larger grid, whose size isn't known here: panels of the usual size, in proportion
        panel_width, panel_height = PANEL_WIDTH, PANEL_HEIGHT
    #(c): The panels' block, the room under it for the x tick labels and label, then one row per legend and colour bar
    block_height = n_rows * panel_height + (n_rows - 1) * row_gap
    height_ratios = [block_height, BOTTOM_MARGIN] + [LEGEND_HEIGHT] * n_legends + [COLORBAR_HEIGHT] * n_colorbars
    if subplot_spec is None:
        outer_grid = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0,
                                      left=left_margin / width, right=1 - RIGHT_MARGIN / width,
                                      top=1 - top_margin / height, bottom=0)
    else:
        outer_grid = subplot_spec.subgridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)
    #(c): The panels, with the gaps between them as a share of a panel's size, as GridSpec takes them
    panel_grid = outer_grid[0].subgridspec(n_rows, n_cols, wspace=column_gap / panel_width,
                                           hspace=row_gap / panel_height)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            #(c): The panel whose axes this one shares: the first panel, the first of its row or of its column. A panel
            #(c): that is itself that first one is still None here, so it shares nothing
            sharex_ax = {False: None, True: axes[0, 0], 'row': axes[row, 0], 'col': axes[0, col]}[sharex]
            sharey_ax = {False: None, True: axes[0, 0], 'row': axes[row, 0], 'col': axes[0, col]}[sharey]
            axes[row, col] = fig.add_subplot(panel_grid[row, col], sharex=sharex_ax, sharey=sharey_ax)
            style_chart_ax(axes[row, col])
    #(c): Axes shared down the columns need their x tick labels only along the bottom row, and those shared along the
    #(c): rows their y tick labels only in the first column
    if sharex in (True, 'col'):
        for ax in axes[:-1].ravel():
            ax.tick_params(axis='x', labelbottom=False)
    if sharey in (True, 'row'):
        for ax in axes[:, 1:].ravel():
            ax.tick_params(axis='y', labelleft=False)

    legend_axes = []
    for legend in range(n_legends):
        legend_ax = fig.add_subplot(outer_grid[2 + legend])
        legend_ax.axis('off')
        legend_axes.append(legend_ax)
    for bar in range(n_colorbars):
        #(c): The bar in the middle 60% of its row, 0.23 in thick at its top; the ticks and label go below it
        bar_grid = outer_grid[2 + n_legends + bar].subgridspec(2, 3, height_ratios=[0.23, COLORBAR_HEIGHT - 0.23],
                                                               width_ratios=[1, 3, 1], hspace=0, wspace=0)
        legend_axes.append(fig.add_subplot(bar_grid[0, 1]))
    return axes, legend_axes


# ---------------------------------------------------------------------------
# 2. One panel's style, the lines that recur, and the panels' letters
# ---------------------------------------------------------------------------

def style_chart_ax(ax):
    """The style of every chart: light gridlines behind what is drawn, and no top or right spine.

    Set on the axis itself, so it holds whatever is drawn on it afterwards, a quick look with xarray's ``.plot``
    included. The text's sizes and colours come from the style's rcParams (``forced_response.NOTEBOOK.rc``).
    ``chart_axes`` calls it on every panel.

    Args:
        ax (matplotlib.axes.Axes): The axis.
    """
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
        legend_axes[0].legend(handles=[Line2D([], [], **style) for style in model_styles.values()], loc='center')
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

        legend_axes[0].legend(handles=model_legend_handles(['CanESM5', 'MIROC6']), loc='center', ncols=2,
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

def plot_lines(da, x, ax=None, **style):
    """``da`` against its dim ``x``: one line, or one line per value of its other dim (each member, say).

    Args:
        da (xr.DataArray): On (x) or (x, one other dim), e.g. a member's extent on (month), or every member's on
            (member, month).
        x (str): The dim along the x axis, e.g. 'year' or 'month'.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        **style: How the lines are drawn, e.g. ``**MEMBER_LINE`` or ``color='C0', linewidth=1.4``.

    Returns:
        list: The lines (``lines[0].axes`` is the axis).

    Example::

        lines = plot_lines(cycle_tree['CanESM5/historical']['extent'], 'month', **MEMBER_LINE)
        plot_lines(era5_cycle_da, 'month', ax=lines[0].axes, **ERA5_LINE)
    """
    if ax is None:
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    #(c): x first, so that each column of the values (each member) is one line
    return ax.plot(da[x].values, da.transpose(x, ...).values, **style)


def lines_grid(da, x, row_dim, col_dim, axes=None, **style):
    """``plot_lines`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        da (xr.DataArray): On (row_dim, col_dim, x), and one more dim for several lines a panel; without row_dim or
            col_dim for a single row or column.
        x (str): The dim along the x axis.
        row_dim (str | None): The dim down the rows, in the order of its values; None for a single row.
        col_dim (str | None): The dim across the columns, in the order of its values; None for a single column.
        axes (np.ndarray | None): The panels, rows × columns: ``chart_axes``', or a slice of them (``axes[2]`` for the
            third row). A new figure if None.
        **style: As ``plot_lines``', the same in every panel.

    Example, each experiment's ensemble mean, seasons down and models across::

        for experiment in ['hist-GHG', 'historical']:
            lines_grid(mean_da.sel(experiment=experiment), 'year', 'season', 'model', axes=axes,
                       color=FORCING_COLORS[experiment])
    """
    #(c): The values down the rows and across the columns ([None] for a single row or column)
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns: a slice of one row (axes[2]) or of one column (axes[:, 0]) comes 1-D
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            #(c): This row's and column's lines
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_lines(da.sel(selection), x, axes[i, j], **style)


def plot_band(low_da, high_da, x, ax=None, **style):
    """The band between ``low_da`` and ``high_da`` along ``x``: e.g. the members' 5-95%.

    Args:
        low_da, high_da (xr.DataArray): The band's edges, each on (x).
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        **style: How the band is drawn, e.g. ``color='C0', alpha=0.15``; it has no edge line unless given a
            ``linewidth``.

    Returns:
        matplotlib.collections.PolyCollection: The band (``band.axes`` is the axis).
    """
    if ax is None:
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    #(c): No edge line unless asked: with the band's own colour it would draw a darker outline
    style = {'linewidth': 0, **style}
    return ax.fill_between(low_da[x].values, low_da.values, high_da.values, **style)


def bands_grid(low_da, high_da, x, row_dim, col_dim, axes=None, **style):
    """``plot_band`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        low_da, high_da (xr.DataArray): The bands' edges, each on (row_dim, col_dim, x); without row_dim or col_dim
            for a single row or column.
        x, row_dim, col_dim, axes: As ``lines_grid``'s.
        **style: As ``plot_band``'s, the same in every panel.
    """
    row_values = [None] if row_dim is None else low_da[row_dim].values
    col_values = [None] if col_dim is None else low_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns, as in lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_band(low_da.sel(selection), high_da.sel(selection), x, axes[i, j], **style)


def plot_quantile_bands(quantiles_da, pairs, x, ax=None, color='C0', alpha=0.35, edge_width=0):
    """Nested bands between pairs of quantiles (a plume): each pair a band, the narrower ones over the wider, so the
    middle is the darkest.

    Args:
        quantiles_da (xr.DataArray): On (quantile, x), e.g. the members' quantiles each year.
        pairs (Sequence[tuple[float, float]]): (lower, upper) quantiles, widest first, e.g. [(0.05, 0.95), (0.25, 0.75)];
            each found to the nearest quantile of ``quantiles_da``.
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        color: The bands' colour.
        alpha (float): Each band's opacity; where n bands overlap the colour is 1 - (1 - alpha)^n strong.
        edge_width (float): Width of a thin line along each band's edges; none if 0.

    Returns:
        list: The bands, widest first (``bands[0].axes`` is the axis).

    Example, the plume and its median::

        plot_quantile_bands(quantiles_da, [(0.05, 0.95), (0.25, 0.75)], 'year', color='C0')
        plot_lines(quantiles_da.sel(quantile=0.5), 'year', color='k')
    """
    if ax is None:
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    bands = []
    for lower, upper in pairs:
        lower_da = quantiles_da.sel(quantile=lower, method='nearest')
        upper_da = quantiles_da.sel(quantile=upper, method='nearest')
        bands.append(plot_band(lower_da, upper_da, x, ax=ax, color=color, alpha=alpha, zorder=1))
        if edge_width:
            for edge_da in (lower_da, upper_da):
                plot_lines(edge_da, x, ax=ax, color=color, linewidth=edge_width, alpha=0.55, zorder=2)
    return bands


def quantile_bands_grid(quantiles_da, pairs, x, row_dim, col_dim, axes=None, color='C0', alpha=0.35, edge_width=0):
    """``plot_quantile_bands`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        quantiles_da (xr.DataArray): On (row_dim, col_dim, quantile, x); without row_dim or col_dim for a single row or
            column.
        pairs, x, color, alpha, edge_width: As ``plot_quantile_bands``', the same in every panel.
        row_dim, col_dim, axes: As ``lines_grid``'s.
    """
    row_values = [None] if row_dim is None else quantiles_da[row_dim].values
    col_values = [None] if col_dim is None else quantiles_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns, as in lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_quantile_bands(quantiles_da.sel(selection), pairs, x, axes[i, j], color=color, alpha=alpha,
                                edge_width=edge_width)


def plot_significant_dots(da, significant_da, x, ax=None, color='k'):
    """A dot on ``da``'s line wherever ``significant_da`` is True: e.g. where a zonal-mean change is significant.

    Args:
        da (xr.DataArray): The line's values, on (x).
        significant_da (xr.DataArray): True/False on (x), e.g. ``pvalue_da < 0.05``.
        x (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw, e.g. the panel the line is on; a new figure if None.
        color: The dots' colour, the line's.

    Returns:
        list: The dots (``dots[0].axes`` is the axis).
    """
    if ax is None:
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    where_arr = significant_da.transpose(*da.dims).values.astype(bool)
    return ax.plot(da[x].values[where_arr], da.values[where_arr], linestyle='none', marker='o', markersize=4,
                   color=color, markeredgecolor='white', markeredgewidth=0.6, zorder=4)


def significant_dots_grid(da, significant_da, x, row_dim, col_dim, axes=None, color='k'):
    """``plot_significant_dots`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        da, significant_da (xr.DataArray): On (row_dim, col_dim, x); without row_dim or col_dim for a single row or
            column.
        x, row_dim, col_dim, axes: As ``lines_grid``'s.
        color: As ``plot_significant_dots``', the same in every panel.
    """
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns, as in lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_significant_dots(da.sel(selection), significant_da.sel(selection), x, axes[i, j], color=color)


# ---------------------------------------------------------------------------
# 4. Models and forcings
# ---------------------------------------------------------------------------

def plot_model_dots(da, x_dim, ax=None, color='0.35'):
    """Each model's value at each value of ``x_dim``: a marker per model, spread a little sideways, and a bar for the
    multi-model median.

    The quickest view of how big a change is in each category (a forcing, a zone) and how far the models agree.
    Model k has marker ``MODEL_MARKERS[k]`` and the k-th shade of its category's colour (``model_shades``;
    ``model_legend_handles`` for the legend).

    Args:
        da (xr.DataArray): On (model, x_dim), e.g. each model's area-mean change for each forcing.
        x_dim (str): The categories along the x axis, in the order of their values, labelled with them.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        color: The markers' and bars' colour: one for all, or a dict of one per category (e.g. ``FORCING_COLORS``
            when the categories are forcings; grey for a category it lacks).

    Returns:
        matplotlib.axes.Axes: The axis.

    Example, one model's change for each forcing, in its colour::

        plot_model_dots(change_da.sel(season='JJA'), 'experiment', color=FORCING_COLORS)
    """
    if ax is None:
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
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
    return ax


def model_dots_grid(da, x_dim, row_dim, col_dim, axes=None, color='0.35'):
    """``plot_model_dots`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        da (xr.DataArray): On (row_dim, col_dim, model, x_dim); without row_dim or col_dim for a single row or column.
        x_dim, color: As ``plot_model_dots``', the same in every panel: for a colour per row (each experiment's), one
            call per row.
        row_dim, col_dim, axes: As ``lines_grid``'s.
    """
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns, as in lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_model_dots(da.sel(selection), x_dim, axes[i, j], color=color)


def plot_scatter_by_forcing(x_da, y_da, ax=None):
    """One change against another, one point per model and forcing: the forcing its colour, the model its marker and
    its shade of that colour (``model_shades``).

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
        fig = plt.figure(figsize=chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
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
