"""Zonal-mean panels: the forced change at each latitude, after Bracegirdle et al. (2024), layer by layer.

Draws the output of ``zonal`` (and ``sea_ice.edge_latitude`` for the ice edge) on axes from a GridSpec.
The changes are lines along latitude (``charts.lines_grid``), with a dot where the change is significant
(``charts.significant_dots_grid``) and, in the width's column, hist-nat's own band (``charts.bands_grid``, in
``NULL_COLOR``); the ice edge is a dashed vertical line (``ice_edge_lines_grid``). Nothing is averaged across
models: each model has its own row (or its own line). For example, one model's four changes, seasons down::

    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=4, n_cols=4, n_legends=1), layout='constrained')
    axes = fig.add_gridspec(4, 4).subplots(sharex=True, sharey='col')
    charts.lines_grid(model_ds['mean_change'].sel(experiment='hist-GHG'), 'lat', 'season', None, axes=axes[:, 0],
                      color=FORCING_COLORS['hist-GHG'])
    ...
    axes[0, 0].xaxis.set_major_formatter(zonal.latitude_label)

Reading them:
    columns     the mean; the low and the high extremes, both measured from the
                median; the width between them (= high - low)
    lines       experiments (FORCING_COLORS), or models (charts.model_lines)
    dots        the zonal-mean change is significant by the hist-nat bootstrap
                (10 hist-nat members over the same final years), the mean's and
                the width's; the extremes have no test, so no dots
    grey band   in the width column: the central 95% of the zonal-mean width
                changes hist-nat's own variability gives (one per model)
    dashed      the sea-ice edge as an equivalent latitude, in each
                experiment's final years (hist-nat's too, in green)

Sections
--------
1. The latitude axis (``latitude_label``) and the look of the band and the edge
2. The ice edge as a latitude: ``plot_ice_edge_line`` (one panel), ``ice_edge_lines_grid`` (a grid)
"""

import matplotlib.pyplot as plt
import numpy as np

from . import charts

# ---------------------------------------------------------------------------
# 1. The latitude axis, the band and the edge
# ---------------------------------------------------------------------------

#(t): hist-nat's own band of the width change, light grey; the ice edge, dashed
NULL_COLOR = '0.88'
EDGE_LINESTYLE = (0, (4, 2))


def latitude_label(value, position=None):
    """A latitude's tick label, '60°S' rather than -60: give it to ``ax.xaxis.set_major_formatter``.

    Args:
        value (float): The latitude (degrees north).
        position: The tick's position, which matplotlib passes and this ignores.

    Returns:
        str: E.g. '60°S', or '0°'.
    """
    return f'{abs(value):g}°S' if value < 0 else f'{value:g}°'


# ---------------------------------------------------------------------------
# 2. The ice edge as a latitude
# ---------------------------------------------------------------------------

def plot_ice_edge_line(edge_da, ax=None, color='k'):
    """The ice edge's equivalent latitude as a dashed vertical line; nothing where there is none (no ice).

    Args:
        edge_da (xr.DataArray): One equivalent latitude (degrees), e.g. ``sea_ice.edge_latitude`` output at one model,
            experiment and season.
        ax (matplotlib.axes.Axes | None): Where to draw, e.g. a panel of zonal means; a new figure if None.
        color: The line's colour, e.g. the experiment's (``FORCING_COLORS``) or the model's.

    Returns:
        matplotlib.lines.Line2D | None: The line; None where there is no edge.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    edge = float(edge_da)
    if not np.isfinite(edge):
        return None
    return ax.axvline(edge, color=color, linewidth=1.0, linestyle=EDGE_LINESTYLE, zorder=2)


def ice_edge_lines_grid(edge_da, row_dim, col_dim, axes=None, color='k'):
    """``plot_ice_edge_line`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        edge_da (xr.DataArray): Equivalent latitudes on (row_dim, col_dim); without row_dim or col_dim for a single row
            or column.
        row_dim, col_dim (str | None): As ``charts.lines_grid``'s.
        axes (np.ndarray | None): The panels, rows × columns, e.g. those the changes were just drawn on.
        color: As ``plot_ice_edge_line``'s, the same in every panel.

    Example, each season's edge (rows) on all four columns: give the edges a column dim with the four columns, so
    each column finds them::

        ice_edge_lines_grid(edge_da.expand_dims(column=list(rc.EXTREMES)), 'season', 'column', axes=axes)
    """
    row_values = [None] if row_dim is None else edge_da[row_dim].values
    col_values = [None] if col_dim is None else edge_da[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=charts.chart_figure_size(len(row_values), len(col_values), n_legends=0),
                         layout='constrained')
        axes = fig.add_gridspec(len(row_values), len(col_values)).subplots(squeeze=False)
    #(c): The axes as rows × columns, as in charts.lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_ice_edge_line(edge_da.sel(selection), axes[i, j], color=color)
