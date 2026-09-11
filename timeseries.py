"""Anything with time on the x axis: plumes, boxes, differences, reveals, stacks.

draw_* functions take an axes and return artists. stack_grid takes the standard
figure-function arguments and returns core.Panels. See core for the contract.
"""

import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import core
from .core import style_ax
from .utils import plot

def _per_band(value, n):
    """Broadcast a scalar to n bands, or pass a per-band sequence through."""
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value] * n


# --------------------------------------------------------------------------
# Line overlays
# --------------------------------------------------------------------------

@plot("axes")
def draw_series(ax, layers, value=None, x="year", dim=None, hlines=()):
    """Overlay every layer on one axes, selecting `dim=value` where present.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    layers : list of dict
        Each has 'data', optional 'hue', and Line2D style kwargs such as
        'label', 'color', 'linestyle'. Drawn in order. Layers whose data lacks
        `dim` are drawn whole.
    value : object
        Value selected from each layer along `dim`.
    x : str
        Coordinate name for the x axis.
    dim : str or None
        Dimension the panel value is selected along.
    hlines : sequence of (float, str)
        (y, colour) pairs drawn on the panel.

    Returns
    -------
    list of matplotlib.lines.Line2D
    """
    lines = []
    for layer in layers:
        style = {k: v for k, v in layer.items() if k not in ("data", "hue")}
        data = layer["data"]
        if dim is not None and dim in data.dims:
            data = data.sel({dim: value})
        lines += data.plot.line(
            ax=ax, x=x, hue=layer.get("hue"), add_legend=False, **style
        )

    for y, colour in hlines:
        ax.axhline(y, color=colour, lw=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    return lines

def layer_handles(layers):
    """Legend handles matching the styles in a `draw_series` layer list."""
    return [
        Line2D([], [], **{k: v for k, v in layer.items() if k not in ("data", "hue")})
        for layer in layers
    ]


# --------------------------------------------------------------------------
# Quantile plumes and boxes
# --------------------------------------------------------------------------

@plot("axes")
def draw_plume(
    ax,
    da,
    pairs,
    median=0.5,
    color="C0",
    alphas=0.35,
    lw=1.8,
    linestyle="-",
    edge_widths=0.0,
    edge_styles="-",
    edge_alpha=0.55,
    label=None,
    x_dim="year",
    quantile_dim="quantile",
):
    """Nested quantile bands and a median line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    da : xarray.DataArray
        Dims (x_dim, quantile_dim).
    pairs : sequence of (float, float)
        (lower, upper) levels, widest first.
    median : float or None
        Quantile level drawn as a line; None draws no line.
    color : str
        Band, edge and line colour.
    alphas : float or sequence of float
        Band opacity, per pair or broadcast; overlapping bands compound.
    lw, linestyle : float, str
        Median line width and style.
    edge_widths, edge_styles : float or sequence, str or sequence
        Band boundary widths and styles; width 0 draws none.
    edge_alpha : float
        Band boundary opacity.
    label : str or None
        Legend entry for the median line.
    x_dim, quantile_dim : str
        Coordinate names.

    Returns
    -------
    matplotlib.lines.Line2D or None
        The median line, if one was drawn.
    """
    n = len(pairs)
    x = da[x_dim].values

    for (lower, upper), alpha, ew, es in zip(
        pairs,
        _per_band(alphas, n),
        _per_band(edge_widths, n),
        _per_band(edge_styles, n),
    ):
        lo = da.sel({quantile_dim: lower}, method="nearest").values
        hi = da.sel({quantile_dim: upper}, method="nearest").values
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)
        if ew:
            for edge in (lo, hi):
                ax.plot(
                    x,
                    edge,
                    color=color,
                    linewidth=ew,
                    linestyle=es,
                    alpha=edge_alpha,
                    zorder=2,
                )

    if median is None:
        return None

    line, = ax.plot(
        x,
        da.sel({quantile_dim: median}, method="nearest").values,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=3,
        label=label,
    )
    return line

def plume_handles(pairs, alphas, color="0.35"):
    """Grey band patches describing a plume's quantile pairs."""
    return [
        Patch(facecolor=color, alpha=a, label=f"{lo:.0%}–{hi:.0%}")
        for (lo, hi), a in zip(pairs, _per_band(alphas, len(pairs)))
    ]

def plume_legend(ax, pairs, alphas, color="0.35", ncol=2, **kwargs):
    """Legend of labelled medians on `ax`, plus a grey patch per band.

    For a figure-level legend, call `plume_handles` directly and pass the
    result to `fig.legend` rather than building and removing an axes legend.
    """
    handles = ax.get_legend_handles_labels()[0] + plume_handles(pairs, alphas, color)
    return ax.legend(
        handles=handles,
        ncol=ncol,
        loc="upper left",
        framealpha=0.95,
        borderpad=0.6,
        **kwargs,
    )

@plot("axes")
def draw_boxes(
    ax,
    da,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    offset=0.0,
    width=0.8,
    color="teal",
    alpha=0.6,
    x_dim="year",
    quantile_dim="quantile",
):
    """Per-x boxes from pre-computed quantiles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    da : xarray.DataArray
        Dims (x_dim, quantile_dim).
    quantiles : sequence of float
        Five ascending levels, taken in order as whislo, q1, med, q3, whishi.
    offset : float
        Shift of box centres along the x axis.
    width : float
        Box width in x units.
    color : str
        Box, whisker and cap colour.
    alpha : float
        Box, whisker and cap opacity.
    x_dim, quantile_dim : str
        Coordinate names.

    Returns
    -------
    dict
        Artists returned by ax.bxp.
    """
    stats = da.sel({quantile_dim: list(quantiles)}).transpose(x_dim, quantile_dim).values
    return ax.bxp(
        [dict(zip(("whislo", "q1", "med", "q3", "whishi"), row)) for row in stats],
        positions=da[x_dim].values + offset,
        widths=width,
        manage_ticks=False,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor=color, alpha=alpha),
        whiskerprops=dict(color=color, alpha=alpha),
        capprops=dict(color=color, alpha=alpha),
        medianprops=dict(color="black", linewidth=1.5),
    )

@plot("axes")
def draw_difference(ax, da, colors=("crimson", "teal"), alpha=0.25, x_dim="year"):
    """Signed difference series with sign-dependent shading.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    da : xarray.DataArray
        Dim (x_dim,).
    colors : (str, str)
        Fill colours for positive and negative values.
    alpha : float
        Fill opacity.
    x_dim : str
        Coordinate name for the x axis.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Artists returned by ax.plot.
    """
    x = da[x_dim].values
    values = da.values
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.fill_between(
        x, 0, values, where=values > 0, color=colors[0], alpha=alpha, linewidth=0
    )
    ax.fill_between(
        x, 0, values, where=values < 0, color=colors[1], alpha=alpha, linewidth=0
    )
    return ax.plot(x, values, color="0.3", linewidth=1.5)


# --------------------------------------------------------------------------
# Slide buildups
# --------------------------------------------------------------------------

@plot("axes")
def draw_reveal(
    ax,
    series,
    revealed,
    colors,
    fade=True,
    x_dim="year",
    lw=2.2,
    lw_earlier=1.5,
    alpha_earlier=0.35,
    zorder0=3,
):
    """Draw one line per revealed key, emphasising the newest.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    series : dict
        Key -> DataArray with an `x_dim` coordinate.
    revealed : sequence
        Keys to draw, in reveal order; the last is the newest.
    colors : dict
        Key -> colour.
    fade : bool
        Dim everything but the newest line.
    x_dim : str
        Coordinate name for the x axis.
    lw, lw_earlier : float
        Line widths for the newest line and the earlier ones.
    alpha_earlier : float
        Opacity of earlier lines when `fade` is set.
    zorder0 : int
        Base z-order; later reveals stack above earlier ones.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The lines, in reveal order.
    """
    lines = []
    for position, key in enumerate(revealed):
        da = series[key]
        is_newest = position == len(revealed) - 1
        line, = ax.plot(
            da[x_dim],
            da,
            color=colors[key],
            lw=lw if is_newest else lw_earlier,
            alpha=alpha_earlier if fade and not is_newest else 1.0,
            zorder=zorder0 + position,
        )
        lines.append(line)
    return lines


@plot("figure")
def stack_grid(das, dim="season", hlines=(), title=None, x="time",
               fig=None, spec=None, layout=None, **layout_kwargs):
    """One stacked panel per value of a dimension, sharing a y range.

    Parameters
    ----------
    das : dict[str, xarray.DataArray] or xarray.DataArray
        Label -> data with `dim`. A bare DataArray is wrapped using its name.
    dim : str
        Dimension stacked down the page, one panel per value.
    hlines : sequence of (float, str)
        (y, colour) pairs drawn on every panel.
    title : str or None
        Title above the top panel.
    x : str
        X axis label for the bottom panel.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    if not isinstance(das, dict):
        das = {das.name: das}
    values = list(next(iter(das.values()))[dim].values)

    def draw(ax, value, _):
        artists = []
        for label, da in das.items():
            artists += da.sel({dim: value}).plot(ax=ax, label=label, linewidth=1.6)
        for y, colour in hlines:
            ax.axhline(y, color=colour, linestyle="--", linewidth=1.4, alpha=0.8)
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(True, linestyle="--", color="grey", alpha=0.6)
        ax.tick_params(labelsize=12, labelbottom=False)
        ax.annotate(str(value), xy=(0.015, 0.78), xycoords="axes fraction",
                    fontsize=15, fontweight="bold")
        return artists

    layout_kwargs.setdefault("hspace", 0.0)
    layout_kwargs.setdefault("panel_h", 1.4)
    layout_kwargs.setdefault("panel_w", 7.0)
    panels = core.panel_grid(values, [None], draw, fig=fig, spec=spec,
                             layout=layout, sharex=True, **layout_kwargs)

    column = panels.axes[:, 0]
    ylims = [ax.get_ylim() for ax in column]
    for ax in column:
        ax.set_ylim(np.min(ylims), np.max(ylims))

    column[-1].tick_params(labelbottom=True)
    column[-1].set_xlabel(x, fontsize=13)
    if title:
        column[0].set_title(title, loc="left", fontweight="bold", fontsize=15)
    column[0].legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=len(das),
                     frameon=False, fontsize=12)
    return panels


# --------------------------------------------------------------------------
# Stacks: one panel per value of a dimension, sharing a y range
# --------------------------------------------------------------------------

@plot("figure")
def quantile_stack_grid(da, gs, fig, dim="season", hlines=(), **kwargs):
    """One `stack_grid` per quantile, cell i of `gs` holding quantile i.

    A worked example of nesting: each stack is a full figure function given a
    cell of the parent grid rather than a canvas of its own.

    Returns
    -------
    list of core.Panels
    """
    return [
        stack_grid(da.sel(quantile=q), dim=dim, hlines=hlines, fig=fig, spec=gs[i],
                   title=f"{q * 100:g}th Percentile", **kwargs)
        for i, q in enumerate(da["quantile"].values)
    ]
