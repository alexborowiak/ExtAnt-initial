"""Maps of Antarctica, drawn layer by layer: a field, its ice edge, dots where it is not significant.

Each layer has two functions: one that draws it on one map axis, and one that loops that over a grid of axes, one
map per value of a row dim (down) and a column dim (across). Nothing draws two layers at once: the call site makes
the axes, then draws each layer on them. Rows of different kinds (the zones, then a change) are slices of the same
axes, each drawn by its own call. For example, one model's change in sea ice in every experiment and season, with
each experiment's ice edge::

    figsize = antarctica.map_figure_size(n_rows=4, n_cols=4, n_colorbar_rows=1)
    fig = plt.figure(figsize=figsize, layout='constrained')
    axes = antarctica.map_axes(fig, n_rows=4, n_cols=4)
    cax = antarctica.colorbar_axis_below(axes)
    antarctica.heatmaps_of_antarctica(change_da, 'experiment', 'season', axes=axes, cax=cax,
                                      levels=np.arange(-55, 56, 10), cmap='RdBu', label='Change (%)')
    antarctica.ice_edges(ice_da, 'experiment', 'season', axes=axes)
    antarctica.label_columns(axes, ['DJF', 'MAM', 'JJA', 'SON'])
    antarctica.label_rows(axes, ['hist-GHG', 'hist-aer', 'hist-totalO3', 'historical'])
    fig.suptitle('One model: change in sea ice')

Sections
--------
1. Layout: the figure's size (``map_figure_size``, or ``panel_size_to_fit`` for a printed page), its map axes
   (``map_axes``), then an axis for each colour bar or key, under the maps it is for (``colorbar_axis_below``,
   ``key_axis_below``)
2. One map's style: the round edge, coastlines and gridlines
3. Fields: ``plot_heatmap_of_antarctica`` (one map), ``heatmaps_of_antarctica`` (a grid, and its colour bar),
   ``bivariate_key`` (a grid of classes instead of a colour bar)
4. The ice edge: ``ice_edge`` (one map), ``ice_edges`` (a grid)
5. Over a field: ``not_significant_dots``, ``hatching`` and ``contour_lines`` (one map), and each ``..._grid``
6. Labels: ``label_rows``, ``label_columns``, ``tag_panels``
"""

import warnings

import cartopy.crs as ccrs
import matplotlib.colorbar
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm

# ---------------------------------------------------------------------------
# 1. Layout
# ---------------------------------------------------------------------------

#(t): Sizes in inches: each map's width and height, the room each row of colour bars takes (its ticks and label
#(t): included), each row with a key (a grid of classes, ``bivariate_key``), a legend's row under the maps, and the
#(t): space above the maps for the title and the column labels, and left of them for the row labels
PANEL_SIZE = 2.3
COLORBAR_HEIGHT = 0.7
KEY_HEIGHT = 1.2
LEGEND_HEIGHT = 0.4
TITLE_HEIGHT = 0.9
ROW_LABEL_WIDTH = 0.5


def map_figure_size(n_rows, n_cols, n_colorbar_rows, n_keys=0, n_legends=0, panel_size=PANEL_SIZE,
                    title_height=TITLE_HEIGHT, colorbar_height=COLORBAR_HEIGHT):
    """The size of a figure of maps with their colour bars (and keys and a legend).

    Only the size, so the figure can be made bigger for something else (a time series under the maps, say) before
    ``map_axes`` fills its part of it. Make the figure with ``layout='constrained'``: matplotlib then fits the maps,
    labels and colour bars into it.

    Args:
        n_rows, n_cols (int): Rows and columns of maps.
        n_colorbar_rows (int): The rows of maps with colour bars under them (``colorbar_axis_below``). Bars side by
            side, under different columns of the same row, count once: 2 for a figure whose top two rows share one
            bar and whose third row has its own.
        n_keys (int): The rows of maps with a key under them (``key_axis_below``), which needs more height than a
            bar; a bar beside a key in the same row is counted with the key, not in ``n_colorbar_rows``.
        n_legends (int): Legends under the maps (``fig.legend(..., loc='outside lower center')``), one row each.
        panel_size (float): Each map's width and height (inches); ``panel_size_to_fit`` for a printed page.
        title_height (float): The room above the maps (inches), for the title and the column labels; less for a
            paper figure, which has no title.
        colorbar_height (float): The room for each row of colour bars (inches), their ticks and labels included;
            more for long class names on several lines.

    Returns:
        tuple: (width, height) in inches. E.g. 3 rows, 4 columns and 2 rows of colour bars: (9.7, 9.2).
    """
    width = ROW_LABEL_WIDTH + n_cols * panel_size
    height = (title_height + n_rows * panel_size + n_colorbar_rows * colorbar_height + n_keys * KEY_HEIGHT
              + n_legends * LEGEND_HEIGHT)
    return width, height


def panel_size_to_fit(n_rows, n_cols, n_colorbar_rows, width, height, n_keys=0, n_legends=0, title_height=TITLE_HEIGHT,
                      colorbar_height=COLORBAR_HEIGHT):
    """The largest map size (inches) at which a figure of maps is no wider than ``width`` and no taller than ``height``.

    For a printed page: the maps fill the page's width, or are made smaller if that would make the figure too tall.

    Args:
        n_rows, n_cols, n_colorbar_rows, n_keys, n_legends, title_height, colorbar_height: As ``map_figure_size``'s.
        width, height (float): The most the figure can be wide and tall (inches), e.g. 7.2 and 9.0 for a two-column
            page (``forced_response.PAGE_WIDTH``, ``PAGE_HEIGHT``).

    Returns:
        float: The maps' width and height. E.g. 3 rows, 4 columns and 2 rows of colour bars on a 7.2 × 9 in page:
        1.675.
    """
    #(c): As wide as the page allows, and as tall as what is left of its height once the bars, keys and legends are in
    across = (width - ROW_LABEL_WIDTH) / n_cols
    down = (height - title_height - n_colorbar_rows * colorbar_height - n_keys * KEY_HEIGHT
            - n_legends * LEGEND_HEIGHT) / n_rows
    return min(across, down)


def map_axes(fig, n_rows, n_cols, subplot_spec=None):
    """``n_rows`` × ``n_cols`` map axes (South Polar Stereographic), filling the figure or part of it.

    Colour bars are added afterwards, each under the maps it is for (``colorbar_axis_below``). Make the figure with
    ``layout='constrained'``, so matplotlib fits the maps, their labels and the colour bars into it.

    Args:
        fig (matplotlib.figure.Figure | matplotlib.figure.SubFigure): The figure, e.g.
            ``plt.figure(figsize=map_figure_size(...), layout='constrained')``, or one part of a layout
            (``fig.subfigures(2, 1)[0]``).
        n_rows, n_cols (int): As ``map_figure_size``'s.
        subplot_spec (matplotlib.gridspec.SubplotSpec | None): The part of the figure to fill, e.g. ``grid[0]`` of
            ``fig.add_gridspec(2, 1)``; the whole figure if None.

    Returns:
        np.ndarray: The map axes, ``n_rows`` × ``n_cols``.

    Example, maps above a time series::

        map_width, map_height = map_figure_size(n_rows=1, n_cols=4, n_colorbar_rows=1)
        fig = plt.figure(figsize=(map_width, map_height + 3), layout='constrained')
        grid = fig.add_gridspec(2, 1, height_ratios=[map_height, 3])
        axes = map_axes(fig, n_rows=1, n_cols=4, subplot_spec=grid[0])
        cax = colorbar_axis_below(axes)
        series_ax = fig.add_subplot(grid[1])
    """
    grid = fig.add_gridspec(n_rows, n_cols) if subplot_spec is None else subplot_spec.subgridspec(n_rows, n_cols)
    return grid.subplots(squeeze=False, subplot_kw=dict(projection=ccrs.SouthPolarStereo()))


def colorbar_axis_below(axes, shrink=0.8):
    """An axis for a colour bar directly under the maps it is for, centred on them.

    In a figure made with ``layout='constrained'``, matplotlib puts the bar under the lowest of ``axes`` and makes
    room for it there, between that row and the next. So a bar for the top two rows sits between the second and
    third rows, and a bar for the last column sits under that column only. The bar is a tenth of a map's width
    thick, however many columns it spans. (In a figure without a layout, one with charts whose margins are set by
    hand, matplotlib makes the room by shrinking the maps instead.)

    Args:
        axes (np.ndarray | matplotlib.axes.Axes): The maps the bar is for: ``axes`` for all of them,
            ``axes[:2, :3]`` for the first three columns of the top two rows, ``axes[2, 3]`` for one map.
        shrink (float): The bar's length, as a fraction of the width of the maps it is under.

    Returns:
        matplotlib.axes.Axes: The bar's axis, for ``heatmaps_of_antarctica(..., cax=...)``.

    Example, two rows of maps sharing one colour scale, a third row of their change on another, and a fourth
    column of the spread on its own scales::

        fig = plt.figure(figsize=map_figure_size(n_rows=3, n_cols=4, n_colorbar_rows=2), layout='constrained')
        axes = map_axes(fig, n_rows=3, n_cols=4)
        temperature_cax = colorbar_axis_below(axes[:2, :3])     # between rows 2 and 3, under columns 1-3
        spread_cax = colorbar_axis_below(axes[:2, 3])           # between rows 2 and 3, under column 4
        change_cax = colorbar_axis_below(axes[2, :3])           # under row 3, columns 1-3
        spread_change_cax = colorbar_axis_below(axes[2, 3])     # under row 3, column 4
    """
    parent_axes = list(np.ravel(axes))
    #(c): The columns of maps the bar is under. Its length over its thickness grows with them, so every bar is a tenth
    #(c): of a map's width thick: a bar under three maps is three times as long as one under one map, and as thick
    columns = {col for ax in parent_axes for col in ax.get_subplotspec().colspan}
    aspect = 10 * len(columns) * shrink
    #(c): panchor=False leaves the maps where they are in their cells; matplotlib places the bar under them
    cax, _ = matplotlib.colorbar.make_axes(parent_axes, location='bottom', shrink=shrink, aspect=aspect, panchor=False)
    return cax


def key_axis_below(axes):
    """An axis for a key (``bivariate_key``) under the maps it is for, centred on them, as ``colorbar_axis_below``.

    The key is as wide as one map, however many columns it is under, and half as tall as it is wide.

    Args:
        axes (np.ndarray | matplotlib.axes.Axes): The maps the key is for, e.g. ``axes[:, 4]`` for the last column.

    Returns:
        matplotlib.axes.Axes: The key's axis, for ``bivariate_key``.
    """
    parent_axes = list(np.ravel(axes))
    columns = {col for ax in parent_axes for col in ax.get_subplotspec().colspan}
    #(c): One map's width, whatever the columns, so its column names fit; up to half the maps' height (fraction), so
    #(c): that a key under one row of maps has its room too
    cax, _ = matplotlib.colorbar.make_axes(parent_axes, location='bottom', shrink=1 / len(columns), aspect=2,
                                           fraction=0.5, panchor=False)
    return cax


# ---------------------------------------------------------------------------
# 2. One map's style
# ---------------------------------------------------------------------------

#(t): What every map shows: south of 40°S
EXTENT = [-180, 180, -90, -40]


def style_antarctic_ax(ax):
    """The style of every map of Antarctica: a round edge, the coastlines and light gridlines.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxes): A South Polar Stereographic axis.
    """
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    #(c): The map's edge: a circle round the pole (the axis runs from 0 to 1 each way)
    angles_arr = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.column_stack([np.cos(angles_arr), np.sin(angles_arr)]) * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    coastlines = ax.coastlines(linewidth=0.6)
    #(c): Light, so they don't compete with the field; drawn in 90 steps each, so the latitude circles come out round
    gridlines = ax.gridlines(crs=ccrs.PlateCarree(), color='0.5', alpha=0.45, linewidth=0.4, linestyle='--')
    gridlines.n_steps = 90
    #(c): Both stay inside the map's circle, so a constrained layout needn't measure them: skipping them makes a big
    #(c): figure of maps several times faster to draw
    coastlines.set_in_layout(False)
    gridlines.set_in_layout(False)


# ---------------------------------------------------------------------------
# 3. Fields
# ---------------------------------------------------------------------------

def plot_heatmap_of_antarctica(da, ax=None, levels=None, cmap='RdBu_r', extend='both'):
    """One field on (lat, lon), cell by cell, on a map of Antarctica.

    Each cell is coloured by the step of ``levels`` its value falls in (a ``BoundaryNorm``), so the colours come in
    steps, as on the colour bar. Empty cells (land, say) are left white.

    Args:
        da (xr.DataArray): The field on (lat, lon), e.g. one model's change in one season.
        ax (cartopy.mpl.geoaxes.GeoAxes | None): Where to draw; a new figure with one map if None.
        levels (array-like | None): The edges of the colour steps; the field's range in 10 steps if None.
        cmap (str | matplotlib.colors.Colormap): The colours.
        extend (str): 'both', 'min', 'max' or 'neither': which ends get a colour of their own for the values beyond
            the levels.

    Returns:
        matplotlib.collections.QuadMesh: What was drawn, for a colour bar (``mesh.axes`` is the axis).

    Example::

        mesh = plot_heatmap_of_antarctica(ice_change_da.sel(model='CanESM5', experiment='hist-GHG', season='JJA'),
                                          levels=np.arange(-55, 56, 10), cmap='RdBu')
        plt.colorbar(mesh, orientation='horizontal')
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
    if levels is None:
        levels = np.linspace(float(da.min()), float(da.max()), 11)
    #(c): One colour for each step between two levels, taken evenly from the colormap, and one for each extended end
    colormap = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, colormap.N, extend=extend)
    #(c): xarray works out each cell's edges from the centres. It also titles the axis ('season = DJF'), which is
    #(c): cleared: label_rows and label_columns say what each map is
    mesh = da.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=colormap, norm=norm,
                              add_colorbar=False)
    ax.set_title('')
    #(c): Inside the map's circle too, so left out of a constrained layout's measuring, as in style_antarctic_ax
    mesh.set_in_layout(False)
    style_antarctic_ax(ax)
    return mesh


def heatmaps_of_antarctica(da, row_dim, col_dim, axes=None, cax=None, levels=None, cmap='RdBu_r', extend='both',
                           label='', ticks=None, ticklabels=None):
    """``plot_heatmap_of_antarctica`` for every value of ``row_dim`` (down) and ``col_dim`` (across), and a colour bar.

    Args:
        da (xr.DataArray): The fields on (row_dim, col_dim, lat, lon), without row_dim or col_dim for a single row
            or column.
        row_dim (str | None): The dim down the rows, in the order of its values; None for a single row.
        col_dim (str | None): The dim across the columns, in the order of its values; None for a single column.
        axes (np.ndarray | None): The map axes, rows × columns: ``map_axes``'s, or a slice of them (``axes[2]`` for
            the third row, ``axes[:, 0]`` for the first column). A new figure if None.
        cax (matplotlib.axes.Axes | None): Where the colour bar goes, e.g. ``colorbar_axis_below(axes)``; no colour
            bar if None (e.g. when another call's bar already covers the same colours).
        levels, cmap, extend: As ``plot_heatmap_of_antarctica``'s; the same on every map, so one colour bar fits
            them all. ``levels`` is the range of all the fields in 10 steps if None.
        label (str): The colour bar's label.
        ticks (array-like | None): The colour bar's ticks; the levels if None.
        ticklabels (Sequence[str] | None): A label for each tick, e.g. the names of the classes of a discrete scale;
            the tick values if None.

    A colour scale of notebooks plot_03 to plot_06 (``forced_response.colour_scales``) passes its levels, colours, ticks and label
    in one go: ``heatmaps_of_antarctica(da, 'model', 'season', axes=axes, cax=cax, **scales.change.colours())``.

    Returns:
        matplotlib.colorbar.Colorbar | None: The colour bar, or None without ``cax``.
    """
    #(c): The values down the rows and across the columns ([None] for a single row or column)
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=map_figure_size(len(row_values), len(col_values), n_colorbar_rows=1),
                         layout='constrained')
        axes = map_axes(fig, len(row_values), len(col_values))
        cax = colorbar_axis_below(axes)
    #(c): The axes as rows × columns: a slice of one row (axes[2]) or of one column (axes[:, 0]) comes 1-D
    axes = np.reshape(axes, (len(row_values), len(col_values)))
    #(c): The same levels on every map, so that one colour bar fits them all
    if levels is None:
        levels = np.linspace(float(da.min()), float(da.max()), 11)

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            #(c): This row's and column's field
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            mesh = plot_heatmap_of_antarctica(da.sel(selection), axes[i, j], levels=levels, cmap=cmap, extend=extend)

    if cax is None:
        return None
    #(c): Every map has the same levels and colours, so the last one's mesh stands for them all
    cbar = plt.colorbar(mesh, cax=cax, orientation='horizontal', label=label, ticks=ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    return cbar


def bivariate_key(cax, cmap, rows, columns, row_title, column_title):
    """The key of a scale of two things at once (the mean and the width, say), drawn as a grid of its classes.

    Used instead of a colour bar, in an axis under the maps it is for (``key_axis_below``; give
    ``map_figure_size`` ``n_keys=1`` for its room). Class
    ``i * len(columns) + j`` is the cell of row ``i`` (from the bottom) and column ``j``, as in
    ``forced_response.bivariate_scale``.

    Args:
        cax (matplotlib.axes.Axes): Where the key goes, e.g. ``key_axis_below(axes)``.
        cmap (matplotlib.colors.ListedColormap): One colour per class, in the order of their numbers.
        rows, columns (Sequence[str]): The names of the rows (bottom first) and of the columns.
        row_title, column_title (str): What the rows and the columns are, e.g. 'Mean' and 'Width'.

    Example, the joint change in the mean and the width (``forced_response.joint_scale``)::

        rows, columns, row_title, column_title = scales.joint.key
        bivariate_key(key_axis_below(axes), scales.joint.cmap, rows, columns, row_title, column_title)
    """
    #(c): The classes' numbers laid out as the grid: row i (from the bottom), column j holds i * len(columns) + j
    classes_arr = np.arange(len(rows) * len(columns)).reshape(len(rows), len(columns))
    cax.imshow(classes_arr, cmap=cmap, vmin=-0.5, vmax=classes_arr.size - 0.5, origin='lower', aspect='auto')
    cax.set_xticks(range(len(columns)), columns)
    cax.set_yticks(range(len(rows)), rows)
    cax.set_xlabel(column_title)
    cax.set_ylabel(row_title)
    #(c): The rows' names on the right
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    #(c): Thin grey lines between the cells, so that a white one (no change) still reads as a cell
    cax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    cax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    cax.grid(which='minor', color='0.6', linewidth=0.4)
    cax.tick_params(which='both', length=0)


# ---------------------------------------------------------------------------
# 4. The ice edge
# ---------------------------------------------------------------------------

#(t): The ice edge: the concentration (%) from which a cell counts as ice
ICE_EDGE_LEVEL = 15.0


def ice_edge(siconc_da, ax=None, color='k', linestyle='solid', linewidth=1.2):
    """The ice edge, the 15% contour of a concentration field on (lat, lon), on a map of Antarctica.

    Args:
        siconc_da (xr.DataArray): Concentration (%) on (lat, lon), empty over land.
        ax (cartopy.mpl.geoaxes.GeoAxes | None): Where to draw, e.g. a map with a field already on it; a new figure
            with one (styled) map if None.
        color, linestyle, linewidth: How the line is drawn: e.g. hist-nat's dashed, ERA5's dotted.

    Returns:
        matplotlib.contour.QuadContourSet | None: The line (``line.axes`` is the axis); None if the field is empty.

    Example, every model's edge in DJF on one map::

        line = ice_edge(ice_da.sel(model='CanESM5', season='DJF'), color='C0')
        for k, model in enumerate(['MIROC6', 'NorESM2-LM'], start=1):
            ice_edge(ice_da.sel(model=model, season='DJF'), ax=line.axes, color=f'C{k}')
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        style_antarctic_ax(ax)
    #(c): Nothing to draw where the field is empty (a model without that experiment)
    if bool(siconc_da.isnull().all()):
        return None
    #(c): The first longitude repeated after the last, so the line joins up across 180°
    field_da = siconc_da.transpose('lat', 'lon')
    values_arr, lon_arr = add_cyclic_point(field_da.values, coord=field_da['lon'].values)
    #(c): A field with no ice at 15% (one model's summer, say) has no line; matplotlib warns about that, harmlessly
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No contour levels were found')
        return ax.contour(lon_arr, field_da['lat'].values, values_arr, levels=[ICE_EDGE_LEVEL], colors=[color],
                          linestyles=linestyle, linewidths=linewidth, transform=ccrs.PlateCarree())


def ice_edges(siconc_da, row_dim, col_dim, axes=None, color='k', linestyle='solid', linewidth=1.2):
    """``ice_edge`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        siconc_da (xr.DataArray): Concentration (%) on (row_dim, col_dim, lat, lon), without row_dim or col_dim for
            a single row or column.
        row_dim (str | None): The dim down the rows; None for a single row.
        col_dim (str | None): The dim across the columns; None for a single column.
        axes (np.ndarray | None): The map axes, rows × columns, e.g. those a heatmap was just drawn on; a new figure
            if None.
        color, linestyle, linewidth: As ``ice_edge``'s, the same on every map.

    The rows follow the order of ``siconc_da[row_dim]``: give it the same order as the layer under it.

    Example, hist-nat's edge (dashed) on every row of a grid of experiments: give hist-nat an experiment dim with
    the grid's experiments, so each row finds it (drop its own experiment label first)::

        hist_nat_every_row_da = ice_da.sel(experiment='hist-nat', drop=True).expand_dims(experiment=EXPERIMENTS)
        ice_edges(hist_nat_every_row_da, 'experiment', 'season', axes=axes, linestyle='dashed')
    """
    row_values = [None] if row_dim is None else siconc_da[row_dim].values
    col_values = [None] if col_dim is None else siconc_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=map_figure_size(len(row_values), len(col_values), n_colorbar_rows=0),
                         layout='constrained')
        axes = map_axes(fig, len(row_values), len(col_values))
        for ax in axes.ravel():
            style_antarctic_ax(ax)
    #(c): The axes as rows × columns, as in heatmaps_of_antarctica
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            ice_edge(siconc_da.sel(selection), axes[i, j], color=color, linestyle=linestyle, linewidth=linewidth)


# ---------------------------------------------------------------------------
# 5. Over a field: dots, hatching and lines
# ---------------------------------------------------------------------------

def not_significant_dots(mask_da, ax=None, size=1.5, color='k'):
    """A small dot in each cell of a map where ``mask_da`` is True: where a change is not significant, say.

    Args:
        mask_da (xr.DataArray): True/False on (lat, lon).
        ax (cartopy.mpl.geoaxes.GeoAxes | None): Where to draw, e.g. a map with the change on it; a new figure with
            one (styled) map if None.
        size (float): The dots' size (points²).
        color: Their colour.

    Returns:
        matplotlib.collections.PathCollection: The dots.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        style_antarctic_ax(ax)
    #(c): Each cell's latitude and longitude, on (lat, lon) like the mask, and the ones where it is True
    lat_da, lon_da = xr.broadcast(mask_da['lat'], mask_da['lon'])
    where_arr = mask_da.transpose('lat', 'lon').values.astype(bool)
    return ax.scatter(lon_da.transpose('lat', 'lon').values[where_arr], lat_da.transpose('lat', 'lon').values[where_arr],
                      s=size, color=color, linewidths=0, transform=ccrs.PlateCarree())


def not_significant_dots_grid(mask_da, row_dim, col_dim, axes=None, size=1.5, color='k'):
    """``not_significant_dots`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        mask_da (xr.DataArray): True/False on (row_dim, col_dim, lat, lon), without row_dim or col_dim for a single
            row or column.
        row_dim (str | None): The dim down the rows; None for a single row.
        col_dim (str | None): The dim across the columns; None for a single column.
        axes (np.ndarray | None): The map axes, rows × columns, e.g. those the change was just drawn on; a new
            figure if None.
        size, color: As ``not_significant_dots``'.
    """
    row_values = [None] if row_dim is None else mask_da[row_dim].values
    col_values = [None] if col_dim is None else mask_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=map_figure_size(len(row_values), len(col_values), n_colorbar_rows=0),
                         layout='constrained')
        axes = map_axes(fig, len(row_values), len(col_values))
        for ax in axes.ravel():
            style_antarctic_ax(ax)
    #(c): The axes as rows × columns, as in heatmaps_of_antarctica
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            not_significant_dots(mask_da.sel(selection), axes[i, j], size=size, color=color)


#(t): Hatching: its lines' colour (black, a little faded, so the field shows through) and width (points)
HATCH_COLOR = (0, 0, 0, 0.6)
HATCH_WIDTH = 0.5


def hatching(mask_da, ax=None, pattern='////', color=HATCH_COLOR, linewidth=HATCH_WIDTH):
    """Hatch the cells of a map where ``mask_da`` is True: e.g. AR6's diagonal lines where no change is robust.

    Args:
        mask_da (xr.DataArray): True/False on (lat, lon).
        ax (cartopy.mpl.geoaxes.GeoAxes | None): Where to draw, e.g. a map with a field on it; a new figure with one
            (styled) map if None.
        pattern (str): matplotlib's hatch pattern: '////' diagonal lines, 'xxxx' crossed lines, '....' dots;
            repeating a character makes it denser.
        color, linewidth: The lines' colour and width (points).

    Returns:
        matplotlib.contour.QuadContourSet | None: The hatching; None where ``mask_da`` is True nowhere.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        style_antarctic_ax(ax)
    if not bool(mask_da.any()):
        return None
    #(c): The mask as 1 (True) and 0, the first longitude repeated after the last so the hatching joins up across 180°;
    #(c): the area between 0.5 and 1.5 (where it is 1) filled with the pattern only
    field_da = mask_da.transpose('lat', 'lon').astype(float)
    values_arr, lon_arr = add_cyclic_point(field_da.values, coord=field_da['lon'].values)
    with plt.rc_context({'hatch.linewidth': linewidth, 'hatch.color': color}):
        return ax.contourf(lon_arr, field_da['lat'].values, values_arr, levels=[0.5, 1.5], colors='none',
                           hatches=[pattern], transform=ccrs.PlateCarree())


def hatching_grid(mask_da, row_dim, col_dim, axes=None, pattern='////', color=HATCH_COLOR, linewidth=HATCH_WIDTH):
    """``hatching`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        mask_da (xr.DataArray): True/False on (row_dim, col_dim, lat, lon), without row_dim or col_dim for a single
            row or column.
        row_dim, col_dim (str | None): As ``heatmaps_of_antarctica``'s.
        axes (np.ndarray | None): The map axes, rows × columns, e.g. those the field was just drawn on.
        pattern, color, linewidth: As ``hatching``'s.
    """
    row_values = [None] if row_dim is None else mask_da[row_dim].values
    col_values = [None] if col_dim is None else mask_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=map_figure_size(len(row_values), len(col_values), n_colorbar_rows=0),
                         layout='constrained')
        axes = map_axes(fig, len(row_values), len(col_values))
        for ax in axes.ravel():
            style_antarctic_ax(ax)
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            hatching(mask_da.sel(selection), axes[i, j], pattern=pattern, color=color, linewidth=linewidth)


def contour_lines(da, levels, ax=None, color='#333333', linewidth=0.6):
    """Lines of a field at some levels on a map, the negative ones dashed: e.g. a change in sea ice at ±5 and ±15%.

    Args:
        da (xr.DataArray): The field on (lat, lon).
        levels (Sequence[float]): Where the lines go.
        ax (cartopy.mpl.geoaxes.GeoAxes | None): Where to draw, e.g. a map with another field on it; a new figure with
            one (styled) map if None.
        color, linewidth: How the lines are drawn.

    Returns:
        matplotlib.contour.QuadContourSet | None: The lines; None if the field is empty.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        style_antarctic_ax(ax)
    if bool(da.isnull().all()):
        return None
    #(c): The first longitude repeated after the last, so the lines join up across 180°
    field_da = da.transpose('lat', 'lon')
    values_arr, lon_arr = add_cyclic_point(field_da.values, coord=field_da['lon'].values)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No contour levels were found')
        return ax.contour(lon_arr, field_da['lat'].values, values_arr, levels=list(levels), colors=color,
                          linewidths=linewidth, negative_linestyles='dashed', transform=ccrs.PlateCarree())


def contour_lines_grid(da, levels, row_dim, col_dim, axes=None, color='#333333', linewidth=0.6):
    """``contour_lines`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        da (xr.DataArray): The field on (row_dim, col_dim, lat, lon), without row_dim or col_dim for a single row or
            column.
        levels, color, linewidth: As ``contour_lines``'.
        row_dim, col_dim (str | None): As ``heatmaps_of_antarctica``'s.
        axes (np.ndarray | None): The map axes, rows × columns.
    """
    row_values = [None] if row_dim is None else da[row_dim].values
    col_values = [None] if col_dim is None else da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=map_figure_size(len(row_values), len(col_values), n_colorbar_rows=0),
                         layout='constrained')
        axes = map_axes(fig, len(row_values), len(col_values))
        for ax in axes.ravel():
            style_antarctic_ax(ax)
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            contour_lines(da.sel(selection), levels, axes[i, j], color=color, linewidth=linewidth)


# ---------------------------------------------------------------------------
# 6. Labels
# ---------------------------------------------------------------------------

def label_rows(axes, labels, fontsize=10):
    """Each row's label, written upwards, left of its first map.

    Args:
        axes (np.ndarray): The map axes, rows × columns.
        labels (Sequence[str]): One per row, top first; '\\n' breaks a label onto two lines.
    """
    for ax, label in zip(np.atleast_2d(axes)[:, 0], labels):
        ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=fontsize)


def label_columns(axes, labels, fontsize=10):
    """Each column's label, above its top map.

    Args:
        axes (np.ndarray): The map axes, rows × columns.
        labels (Sequence[str]): One per column, left first.
    """
    for ax, label in zip(np.atleast_2d(axes)[0], labels):
        ax.set_title(label, fontsize=fontsize)


def tag_panels(axes, start=0, fontsize=9):
    """Letter the maps a), b), c), ... row by row, in each map's top-left corner (after z come aa), ab), ...).

    Args:
        axes (np.ndarray): The map axes, rows × columns.
        start (int): The first map's letter, counting from 0 for a: to carry on from another figure's letters.

    Returns:
        int: The next letter's number, for a figure that carries on.
    """
    number = start
    for ax in np.atleast_2d(axes).ravel():
        #(c): a ... z, then aa, ab, ...
        letters = chr(ord('a') + number) if number < 26 else chr(ord('a') + number // 26 - 1) + chr(ord('a') + number % 26)
        ax.text(0.0, 1.0, f'{letters})', transform=ax.transAxes, ha='left', va='top', fontsize=fontsize,
                fontweight='bold')
        number += 1
    return number
