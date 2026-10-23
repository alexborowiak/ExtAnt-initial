"""The paper's panels (notebooks 04d, 06 and 07), each drawn on one axis: probability ratios, whole distributions as
ridgelines, and the verification against ERA5; and the colour scales of their maps.

Each ``plot_*(..., ax=None)`` draws one panel, on an axis from ``charts.chart_axes`` (a new figure if ``ax`` is
None); the notebook makes the figure and its axes, draws each panel and labels them, as everywhere (``charts``).
Notebook 07 puts its candidate blocks together into layouts with ``fig.subfigures``, each block drawn into its part
by a ``draw_...(fig, tag_start)`` of the notebook's own. The other panels of the candidates are the shared chart
layers: the time series and percentile curves (``charts.lines_grid``, ``charts.bands_grid``,
``charts.quantile_bands_grid``), the scatter of one change against another (``charts.plot_scatter_by_forcing``) and
the forcings' changes (``charts.model_dots_grid``).

Colours are those of ``forced_response``: red and blue for up and down, and each forcing's colour from
plotting_modules, with ``OTHER_COLORS`` for what is not a forcing. Nothing here computes a result; the notebook
makes every array a panel is given.

Sections
--------
1. Scales: the probability ratio, ERA5's percentile, named classes
2. Probability ratios: every model with its bootstrap range, and the shift-only ratio (``plot_probability_ratios``,
   ``probability_ratios_grid``)
3. Whole distributions as ridgelines (``plot_ridgelines``)
4. Verification against ERA5: regional rank histograms (``plot_regional_rank_histogram``) and ERA5's percentile among
   each experiment (``plot_percentile_table``)
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting_modules.constants import FORCING_COLORS
from . import charts, forced_response as fr
from .charts import MODEL_MARKERS
from .era5_evaluation import INK, PERCENTILE_EDGES, PSEUDO_COLOR, RANK_BAR, percentile_colormap

# ---------------------------------------------------------------------------
# 1. Scales
# ---------------------------------------------------------------------------
#(t): Colours of what is not a forcing (ERA5 black, as in every figure)
OTHER_COLORS = {"sum of parts": "#c49a00", "residual": "#c9c8c4", "ERA5": charts.ERA5_LINE["color"]}


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


def _forcing_color(name):
    return FORCING_COLORS.get(name, OTHER_COLORS.get(name, INK))


# ---------------------------------------------------------------------------
# 2. Probability ratios
# ---------------------------------------------------------------------------

#(t): The y range of the probability-ratio panels, as log2: a ratio of 0 is drawn at the bottom
RATIO_LIMITS = (-6, 6)


def _log_ratio(values):
    """log2 of probability ratios, clipped to ``RATIO_LIMITS`` so that 0 and infinity are drawn at the ends."""
    return np.clip(np.log2(np.clip(np.asarray(values, dtype=float), 2.0 ** -10, 2.0 ** 10)), *RATIO_LIMITS)


def plot_probability_ratios(ratios_ds, ax=None, shift_only=False):
    """Probability ratios, forcings along x: every model with its bootstrap range, and the multi-model median.

    Each model is a marker (its shape) in the forcing's colour with a line from its range's lower end to its upper
    end, spread a little sideways; the bar is the multi-model median, as in ``charts.plot_model_dots``. With
    ``shift_only``, an open marker beside each gives the shift-only ratio, the ratio the shift of the mean alone would
    give: the gap between the two is what the change in width (and shape) adds. The y axis is log2, ticked with the
    ratios (1/8 ... 8); a ratio of 0 sits on the bottom edge.

    Args:
        ratios_ds (xr.Dataset): ``pr``, ``pr_lower``, ``pr_upper`` (and ``pr_shift_only`` for ``shift_only``) on
            (model, experiment), e.g. one region, chance, tail and season of notebook 03's
            ``probability_ratio_regional``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        shift_only (bool): Also draw the shift-only ratios (open markers).

    Returns:
        matplotlib.axes.Axes: The axis. Model k's marker is ``charts.MODEL_MARKERS[k]``, in the k-th shade of its
        forcing's colour (``charts.model_shades``).
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    experiments = list(ratios_ds["experiment"].values)
    models = list(ratios_ds["model"].values)
    #(c): Each model's offset from the middle of its forcing; the shift-only marker a little to its right
    offsets_arr = np.linspace(-0.28, 0.28, len(models)) if len(models) > 1 else np.zeros(1)
    ax.axhline(0, color="0.5", lw=0.7, zorder=1)
    for x, experiment in enumerate(experiments):
        experiment_ds = ratios_ds.sel(experiment=experiment)
        color = _forcing_color(experiment)
        median = float(np.median(_log_ratio(experiment_ds["pr"].dropna("model"))))
        ax.bar(x, median, width=0.7, color=color, alpha=0.3, lw=0, zorder=2)
        shades = charts.model_shades(color, len(models))
        for k, model in enumerate(models):
            model_ds = experiment_ds.sel(model=model)
            if not np.isfinite(float(model_ds["pr"])):
                continue
            position = x + offsets_arr[k] - (0.06 if shift_only else 0)
            ax.plot([position] * 2, _log_ratio([model_ds["pr_lower"], model_ds["pr_upper"]]), color=shades[k], lw=0.8,
                    alpha=0.7, zorder=3)
            ax.plot(position, _log_ratio(model_ds["pr"]), ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)],
                    ms=4, color=shades[k], mec="white", mew=0.3, zorder=4)
            if shift_only:
                ax.plot(position + 0.12, _log_ratio(model_ds["pr_shift_only"]), ls="none",
                        marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=4, mfc="white", mec=shades[k], mew=0.7,
                        zorder=4)
    #(c): log2 ticks labelled with the ratios
    ticks = np.arange(RATIO_LIMITS[0], RATIO_LIMITS[1] + 1, 2)
    ax.set_yticks(ticks, [f"1/{2 ** -t:g}" if t < 0 else f"{2 ** t:g}" for t in ticks])
    ax.set_ylim(RATIO_LIMITS[0] - 0.3, RATIO_LIMITS[1] + 0.3)
    ax.set_xticks(range(len(experiments)), experiments)
    ax.set_xlim(-0.6, len(experiments) - 0.4)
    ax.grid(axis="x", visible=False)
    return ax


def probability_ratios_grid(ratios_ds, row_dim, col_dim, axes=None, shift_only=False):
    """``plot_probability_ratios`` for every value of ``row_dim`` (down) and ``col_dim`` (across).

    Args:
        ratios_ds (xr.Dataset): As ``plot_probability_ratios``', on (row_dim, col_dim, model, experiment) too, e.g.
            (tail, season); without row_dim or col_dim for a single row or column.
        row_dim, col_dim (str | None): As ``charts.lines_grid``'s.
        axes (np.ndarray | None): The panels, rows × columns: ``charts.chart_axes``'. A new figure if None.
        shift_only (bool): As ``plot_probability_ratios``'.
    """
    row_values = [None] if row_dim is None else ratios_ds[row_dim].values
    col_values = [None] if col_dim is None else ratios_ds[col_dim].values
    if axes is None:
        fig = plt.figure(figsize=charts.chart_figure_size(len(row_values), len(col_values), n_legends=0))
        axes, _ = charts.chart_axes(fig, len(row_values), len(col_values), n_legends=0)
    #(c): The axes as rows × columns, as in charts.lines_grid
    axes = np.reshape(axes, (len(row_values), len(col_values)))

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            selection = {dim: value for dim, value in ((row_dim, row_value), (col_dim, col_value)) if dim is not None}
            plot_probability_ratios(ratios_ds.sel(selection), axes[i, j], shift_only=shift_only)


# ---------------------------------------------------------------------------
# 3. Whole distributions as ridgelines
# ---------------------------------------------------------------------------

#(t): The ridges' colours, by their mean: blue through purple to red, with no white in the middle, so that a ridge
#(t): with no change still stands out against the page
RIDGE_CMAP = LinearSegmentedColormap.from_list("ridge", ["#3d6fb6", "#7b5c93", "#c0392b"])


def plot_ridgelines(densities, means, ridge_dim, height_scale, limit=3.0, value_format="{:+.1f}", lines=None, ax=None):
    """Whole distributions as overlapping ridges, one per value of ``ridge_dim`` (a forcing, or a model), top to bottom.

    Each ridge is a density, drawn up from its own baseline and coloured by its mean (``RIDGE_CMAP``, blue to red
    through purple; ``limit`` takes the end colours), which is also written at its right end. The ridges below overlap
    the ones above, as in the ridgeline ("joy") plots of whole populations. The dotted lines mark levels shared by
    every ridge, e.g. hist-nat's 5th and 95th percentiles, so the share of each ridge beyond them is what the
    probability ratios count. Ridge k from the top sits on baseline n - 1 - k: label the y axis with
    ``ax.set_yticks(range(n)[::-1], names)``, and give it room for the top ridge, ``ax.set_ylim(-0.15, n - 1 + top)``
    with ``top`` the most any ridge reaches above its baseline (``height_scale`` × the highest density).

    Args:
        densities (xr.DataArray): On (ridge_dim, x), e.g. ``stats.fast_kde_xr`` of anomalies in one season.
        means (xr.DataArray): Each ridge's mean, on (ridge_dim): its colour and its number.
        ridge_dim (str): The ridges' dim, in the order of its values, top first.
        height_scale (float): The ridges' height per unit of density, the same in every panel so their areas compare:
            e.g. 1.8 / the highest density in the figure, for the highest to reach 1.8 baselines up.
        limit (float): The mean that takes the end colours (the colour bar runs from -limit to limit).
        value_format (str): Format of the mean written beside each ridge.
        lines (xr.DataArray | None): Values of the dotted vertical lines, on any one dim (e.g. quantile).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    ridges = list(densities[ridge_dim].values)
    x = densities["x"].values
    norm = Normalize(-limit, limit)
    for k, ridge in enumerate(ridges):
        base = len(ridges) - 1 - k
        density = np.nan_to_num(densities.sel({ridge_dim: ridge}).values)
        mean = float(means.sel({ridge_dim: ridge}))
        color = RIDGE_CMAP(norm(mean)) if np.isfinite(mean) else "0.75"
        #(c): The ridges further down are drawn later, so they overlap the ones above
        ax.fill_between(x, base, base + height_scale * density, color=color, lw=0, zorder=2 * k + 1)
        ax.plot(x, base + height_scale * density, color="white", lw=0.9, zorder=2 * k + 2)
        if np.isfinite(mean):
            #(c): A small negative mean rounds to '-0.0'; write it as 0
            text = value_format.format(mean)
            text = value_format.format(0.0) if float(text) == 0 else text
            ax.text(x[-1], base + 0.08, text, ha="right", va="bottom", fontsize="small", color="0.25", zorder=100)
    if lines is not None:
        for value in np.ravel(lines.values):
            if np.isfinite(value):
                ax.axvline(value, color="0.45", lw=1.0, ls=(0, (1, 1.6)), zorder=99)
    ax.set_xlim(x[0], x[-1])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    return ax


# ---------------------------------------------------------------------------
# 4. Verification against ERA5
# ---------------------------------------------------------------------------

def plot_regional_rank_histogram(ranks_ds, ax=None):
    """ERA5's rank histogram (Suarez-Gutierrez et al., 2021) for a regional mean, against the range a perfect model gives.

    Bars are ERA5's rank frequencies on bins of scaled rank (0: below every member, 1: above every member); the grey
    band is the pseudo-observations' central range, what the model itself gives when one member stands in for ERA5,
    for this ensemble and record. Flat bars inside the band: ERA5 could be a member. A U shape: the model's spread is
    too small; a dome: too large; a slope: its forced response differs. The corner gives ERA5's percentile for the
    dispersion among the pseudo-observations', ✗ when beyond their range.

    Args:
        ranks_ds (xr.Dataset): ``obs_frequency``, ``pseudo_lower``, ``pseudo_upper`` on (rank), and ``percentile``,
            ``verdict`` on (statistic), e.g. one region, experiment, treatment, model and season of notebook 02's
            ``regional_ranks``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    rank = ranks_ds["rank"].values
    step = rank[1] - rank[0]
    #(c): Each bin's range drawn across the whole bin: its value repeated at both edges
    edges = np.column_stack([rank - step / 2, rank + step / 2]).ravel()
    ax.fill_between(edges, np.repeat(ranks_ds["pseudo_lower"].values, 2), np.repeat(ranks_ds["pseudo_upper"].values, 2),
                    color="0.82", lw=0, zorder=1)
    ax.bar(rank, ranks_ds["obs_frequency"], width=0.8 * step, color=RANK_BAR, edgecolor="#2a78d6", lw=0.4, zorder=2)
    ax.axhline(100 / rank.size, color=PSEUDO_COLOR, lw=0.6, ls=(0, (3, 2)), zorder=3)
    if "dispersion" in ranks_ds["statistic"].values and np.isfinite(float(ranks_ds["percentile"].sel(statistic="dispersion"))):
        dispersion = ranks_ds.sel(statistic="dispersion")
        flag = "✗" if float(dispersion["verdict"]) != 0 else "✓"
        ax.text(0.97, 0.95, f"{flag} {float(dispersion['percentile']):.0f}", transform=ax.transAxes, ha="right",
                va="top", fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.grid(False)
    return ax


def rank_histogram_legend_handles():
    """The legend entries of ``plot_regional_rank_histogram``."""
    return [Patch(color=RANK_BAR, label="ERA5"), Patch(color="0.82", label="Perfect model (5–95%)"),
            Line2D([], [], color=PSEUDO_COLOR, lw=0.8, ls=(0, (3, 2)), label="Flat")]


def plot_percentile_table(percentile, verdict, row_dim="model", col_dim="experiment", ax=None):
    """ERA5's percentile among each experiment's members as a table of colours (``percentile_scale``), each cell its
    value, outlined where flagged.

    ERA5 inside historical (central colours) and outside hist-nat (the darkest colours, outlined) is the consistency
    the attribution needs.

    Args:
        percentile, verdict (xr.DataArray): On (row_dim, col_dim), e.g. one region and season of notebook 02's
            ``regional_consistency`` (the trend).
        row_dim, col_dim (str): The dims down and across, in the order of their values.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.image.AxesImage: The cells, for a colour bar (``percentile_scale()``'s ticks).
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, n_colorbars=1,
                                                          left_margin=1.5))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0, n_colorbars=1, left_margin=1.5)
        ax = axes[0, 0]
    scale = percentile_scale()
    table = percentile.transpose(row_dim, col_dim)
    flags = verdict.transpose(row_dim, col_dim)
    image = ax.imshow(table.values, cmap=scale.cmap, norm=scale.norm, aspect="auto")
    for (r, c), flag in np.ndenumerate(flags.values):
        if np.isfinite(flag) and flag != 0:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
        value = table.values[r, c]
        if np.isfinite(value):
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize="small")
    ax.set_xticks(range(table.sizes[col_dim]), table[col_dim].values, rotation=40, ha="right")
    ax.set_yticks(range(table.sizes[row_dim]), table[row_dim].values)
    ax.tick_params(length=0)
    ax.grid(False)
    return image
