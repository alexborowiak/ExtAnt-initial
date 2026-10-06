from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def draw_panel(ax, vals):
    members = np.arange(len(vals))
    vals = np.sort(vals)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    pad = 0.05 * (vmax - vmin)

    ax.bar(members, vals, width=0.75, color='#3b6ea5', edgecolor='none', zorder=3)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlim(-0.75, len(members) - 0.25)
    ax.grid(axis='y', color='0.9', lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_season_stack(das, fig, subplot_spec, title, hlines):
    """Plot each season as a stacked panel, overlaying one line per labelled array.

    Args:
        das (dict[str, xr.DataArray]): Label -> data with a 'season' dimension.
        fig (matplotlib.figure.Figure): Figure to draw into.
        subplot_spec (matplotlib.gridspec.SubplotSpec): Parent grid cell to subdivide.
        title (str): Title placed above the top panel.
        hlines (list[tuple[float, str]]): (y, colour) pairs drawn on every panel.

    Returns:
        list[matplotlib.axes.Axes]: Stacked axes, top to bottom.
    """
    if isinstance(das, xr.DataArray):
        das = {das.name: das}
    seasons = next(iter(das.values())).season.values
    gs = GridSpecFromSubplotSpec(len(seasons), 1, subplot_spec=subplot_spec, hspace=0)
    axes = [fig.add_subplot(gs[i]) for i in range(len(seasons))]

    for ax, season in zip(axes, seasons):
        for label, da in das.items():
            da.sel(season=season).plot(ax=ax, label=label, linewidth=1.6)
        for y, colour in hlines:
            ax.axhline(y, color=colour, linestyle='--', linewidth=1.4, alpha=0.8)
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(True, linestyle='--', color='grey', alpha=0.6)
        ax.tick_params(labelsize=12, labelbottom=False)
        ax.annotate(season, xy=(0.015, 0.78), xycoords='axes fraction',
                    fontsize=15, fontweight='bold')

    ylims = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(np.min(ylims), np.max(ylims))

    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel('time', fontsize=13)
    axes[0].set_title(title, loc='left', fontweight='bold', fontsize=15)
    axes[0].legend(loc='lower right', bbox_to_anchor=(1, 1),
                   ncols=len(das), frameon=False, fontsize=12)
    return axes

def plot_quantile_seasons(da, fig, gridspec, hlines):
    """Plot a stack of seasonal panels for each quantile, one stack per grid cell.

    Args:
        da (xr.DataArray): Data with 'quantile' and 'season' dimensions.
        fig (matplotlib.figure.Figure): Figure to draw into.
        gridspec (matplotlib.gridspec.GridSpec): Parent grid; cell i holds quantile i.
        hlines (list[tuple[float, str]]): (y, colour) pairs drawn on every panel.

    Returns:
        list[list[matplotlib.axes.Axes]]: Per-quantile lists of stacked axes.
    """
    quantiles = da['quantile'].values
    return [
        plot_season_stack(da.sel(quantile=q), fig, gridspec[i],
                          f'{q*100:g}th Percentile', hlines)
        for i, q in enumerate(quantiles)
    ]

def plot_panel(layers, ax, value, x, dim=None, hlines=()):
    """Overlay every layer on a single axes, selecting `dim=value` where present.
    Args:
        layers (list[dict]): Each has 'label', 'data', optional 'hue', and Line2D style
            kwargs. Drawn in order. Layers whose data lacks `dim` are drawn whole.
        ax (matplotlib.axes.Axes): Axes to draw into.
        value: Value to select from each layer along `dim`.
        x (str): Coordinate name for the x axis.
        dim (str): Dimension name the panel value is selected along.
        hlines (list[tuple[float, str]]): (y, colour) pairs drawn on the panel.
    """
    for layer in layers:
        style = {k: v for k, v in layer.items() if k not in ("data", "hue")}
        data = layer["data"]
        if dim is not None and dim in data.dims:
            data = data.sel({dim: value})
        data.plot.line(ax=ax, x=x, hue=layer.get("hue"), add_legend=False, **style)
    for y, colour in hlines:
        ax.axhline(y, color=colour, lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_panel_grid(layers, panels, x, dim=None, hlines=(), xlims=None,
                    ylabel="", ncols=1, title="{value}"):
    """Plot one panel per value in a grid, under a legend built from the layers.
    Args:
        layers (list[dict]): Series drawn on every panel, in order. See plot_panel.
        panels (array-like): Panel values, one panel each.
        x (str): Coordinate name for the x axis; also the figure x label.
        dim (str): Dimension name the panel values are selected along.
        hlines (list[tuple[float, str]]): (y, colour) pairs drawn on every panel.
        xlims (tuple[float, float]): Shared x limits; left to matplotlib if None.
        ylabel (str): Figure-level y label.
        ncols (int): Number of grid columns.
        title (str): Panel title template, formatted with `value` (alias `q`).
    Returns:
        tuple[matplotlib.figure.Figure, np.ndarray]: Figure and the used axes.
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 2.8 * nrows),
                             sharex=True, squeeze=False, layout="constrained")
    axes = axes.ravel()
    for ax in axes[n:]:
        fig.delaxes(ax)
    axes = axes[:n]
    for ax, value in zip(axes, panels):
        plot_panel(layers, ax, value, x, dim, hlines)
        ax.set_title(title.format(value=value, q=value))
    if xlims is not None:
        axes[0].set_xlim(xlims)
    for ax in axes[n - ncols:]:
        ax.tick_params(labelbottom=True)
    handles = [
        Line2D([], [], **{k: v for k, v in layer.items() if k not in ("data", "hue")})
        for layer in layers
    ]
    fig.supxlabel(x)
    fig.supylabel(ylabel)
    fig.legend(handles=handles, loc="outside upper center",
               ncols=len(handles), frameon=False)
    return fig, axes

def plot_boxes(da, ax, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), offset=0.0,
               width=0.8, color="teal", alpha=0.6):
    """Draw per-year boxes from pre-computed quantiles.

    Args:
        da (xr.DataArray): Dims ("year", "quantile").
        ax (plt.Axes): Axis to draw on.
        quantiles (tuple[float, ...]): Five ascending levels, taken in order as
            whislo, q1, med, q3, whishi.
        offset (float): Shift of box centres along the year axis.
        width (float): Box width in years.
        color (str): Box, whisker and cap colour.
        alpha (float): Box, whisker and cap opacity.

    Returns:
        dict: Artists returned by ax.bxp.
    """
    stats = da.sel(quantile=list(quantiles)).transpose("year", "quantile").values
    return ax.bxp(
        [dict(zip(("whislo", "q1", "med", "q3", "whishi"), row)) for row in stats],
        positions=da.year.values + offset,
        widths=width,
        manage_ticks=False,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor=color, alpha=alpha),
        whiskerprops=dict(color=color, alpha=alpha),
        capprops=dict(color=color, alpha=alpha),
        medianprops=dict(color="black", linewidth=1.5),
    )


def _per_band(value, n):
    return list(value) if isinstance(value, (list, tuple)) else [value] * n


def plot_plume(ax, da, pairs, median=0.5, color="C0", alphas=0.35, lw=1.8,
               linestyle="-", edge_widths=0.0, edge_styles="-", edge_alpha=0.55,
               label=None, x_dim="year", quantile_dim="quantile"):
    """Draw nested quantile bands and a median line on ax.

    Args:
        ax (plt.Axes): Axis to draw on.
        da (xr.DataArray): Dims (x_dim, quantile_dim).
        pairs (Sequence[tuple[float, float]]): (lower, upper) levels, widest first.
        median (float): Quantile level drawn as a line.
        color (str): Band, edge and line colour.
        alphas (float | Sequence[float]): Band opacity, per pair or broadcast;
            overlapping bands compound.
        lw (float): Median line width.
        linestyle (str): Median line style.
        edge_widths (float | Sequence[float]): Band boundary widths; 0 draws none.
        edge_styles (str | Sequence[str]): Band boundary styles.
        edge_alpha (float): Band boundary opacity.
        label (str | None): Legend entry for the median line.
        x_dim (str): Name of the x coordinate.
        quantile_dim (str): Name of the quantile coordinate.

    Returns:
        plt.Line2D: The median line.
    """
    n = len(pairs)
    x = da[x_dim].values
    for (lower, upper), alpha, ew, es in zip(pairs, _per_band(alphas, n),
                                             _per_band(edge_widths, n),
                                             _per_band(edge_styles, n)):
        lo = da.sel({quantile_dim: lower}, method='nearest').values
        hi = da.sel({quantile_dim: upper}, method='nearest').values
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)
        if ew:
            for edge in (lo, hi):
                ax.plot(x, edge, color=color, linewidth=ew, linestyle=es,
                        alpha=edge_alpha, zorder=2)
    if median:
        line, = ax.plot(x, da.sel({quantile_dim: median}).values, color=color,
                        linewidth=lw, linestyle=linestyle, zorder=3, label=label)
    # ax.margins(x=0)
        return line
    return None

def plume_legend(ax, pairs, alphas, color="0.35", **kwargs):
    """Legend of labelled medians on ax, plus a grey patch per band."""
    handles = ax.get_legend_handles_labels()[0]
    handles += [Patch(facecolor=color, alpha=a, label=f"{lo:.0%}–{hi:.0%}")
                for (lo, hi), a in zip(pairs, _per_band(alphas, len(pairs)))]
    return ax.legend(handles=handles, ncol=2, loc="upper left",
                     framealpha=0.95, borderpad=0.6, **kwargs)



def plot_difference(da, ax, colors=("crimson", "teal"), alpha=0.25):
    """Draw a signed difference series with sign-dependent shading.

    Args:
        da (xr.DataArray): Dim ("year",).
        ax (plt.Axes): Axis to draw on.
        colors (tuple[str, str]): Fill colours for positive and negative values.
        alpha (float): Fill opacity.

    Returns:
        list[plt.Line2D]: Artists returned by ax.plot.
    """
    year = da.year.values
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.fill_between(year, 0, da.values, where=da.values > 0,
                    color=colors[0], alpha=alpha, linewidth=0)
    ax.fill_between(year, 0, da.values, where=da.values < 0,
                    color=colors[1], alpha=alpha, linewidth=0)
    return ax.plot(year, da.values, color="0.3", linewidth=1.5)

def style_axis(ax, scale=1.0):
    """Apply gridlines behind the data, drop top and right spines, and scale any text present.

    Call after labels, title and legend are set: `set_title` and `legend` reapply
    the `rcParams` font sizes and would undo the scaling. Text that is absent
    (empty title, empty axis label, no ticks, no legend) is left alone.

    Args:
        ax (plt.Axes): Axis to format.
        scale (float): Multiplier applied to the current axis label, tick label,
            title and legend entry font sizes.
    """
    ax.grid(axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_title():
        ax.title.set_fontsize(ax.title.get_fontsize() * scale)
    if ax.get_xlabel():
        ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * scale)
    if ax.get_ylabel():
        ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * scale)
    xticklabels = ax.get_xticklabels()
    if xticklabels:
        ax.tick_params(axis="x", labelsize=xticklabels[0].get_fontsize() * scale)
    yticklabels = ax.get_yticklabels()
    if yticklabels:
        ax.tick_params(axis="y", labelsize=yticklabels[0].get_fontsize() * scale)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * scale)
    ax.autoscale(axis='x', tight=True)