"""Panels for talks: forcings revealed one at a time, labels for line and plume figures, and pictures of how the
forced-response figures are made.

Each ``plot_*(..., ax=None)`` draws one panel, on an axis from ``charts.chart_axes``; the notebook makes the figure and
its axes, draws each panel and labels them, as everywhere (``charts``), in the slides' style
(``with plt.rc_context(fr.SLIDES.rc):``).

Sections
--------
1. The build-up: the forcings' lines revealed one at a time (``plot_buildup``), with a legend that has every forcing
   from the first frame, the ones still to come faded (``plot_buildup_legend``)
2. Labels: names at the end of lines, quantile levels at the edge of a plume
3. How the figures are made: one model's test at a point (``plot_point_test``, which notebook 05 draws for every
   model before counting them), and one cell of the joint key drawn with a sketched distribution
   (``plot_joint_key_cell``)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import charts
from .era5_evaluation import INK, INK_2, MUTED
from plotting_modules.constants import FORCING_COLORS, FORCING_LEGEND_LABELS

# ---------------------------------------------------------------------------
# 1. The build-up
# ---------------------------------------------------------------------------

def plot_buildup(tas_by_forcing, revealed, ylim, fade_earlier=True, baseline=0.0, xlim=(1850, 2014), ax=None):
    """One frame of the build-up: a line per revealed forcing (FORCING_COLORS), the newest on top and thicker, the
    earlier ones faded, and a dashed baseline.

    Every frame of a sequence takes the same ``ylim`` and ``xlim``, so the frames line up when flipped through
    (``utils.shared_ylim`` gives one from every series).

    Args:
        tas_by_forcing (dict[str, xr.DataArray]): Forcing -> temperature series on 'year'.
        revealed (Sequence[str]): The forcings drawn in this frame, in reveal order; the last is the newest.
        ylim, xlim (tuple[float, float]): The axis limits, the same in every frame.
        fade_earlier (bool): Fade all but the newest line.
        baseline (float): The dashed reference line's value.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, panel_width=8.0,
                                                          panel_height=3.6))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    for position, forcing in enumerate(revealed):
        newest = position == len(revealed) - 1
        charts.plot_lines(tas_by_forcing[forcing], "year", ax=ax, color=FORCING_COLORS[forcing],
                          linewidth=2.2 if newest else 1.5, alpha=0.35 if fade_earlier and not newest else 1.0,
                          zorder=3 + position)
    ax.axhline(baseline, color="0.75", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)
    return ax


def plot_buildup_legend(order, revealed, fig):
    """The build-up's legend at the top of the frame, every forcing of ``order`` from the first frame, those not yet
    revealed faded: so it takes the same room in every frame and the axes below never move.

    Args:
        order (Sequence[str]): Every forcing, in reveal order.
        revealed (Sequence[str]): Those drawn in this frame.
        fig (matplotlib.figure.Figure): The frame; give its axes the room above them (``top_margin`` of
            ``charts.chart_axes``, about 1 in for two rows of entries).

    Returns:
        matplotlib.legend.Legend: The legend.
    """
    handles = [Line2D([], [], color=FORCING_COLORS[forcing], lw=2.4, label=FORCING_LEGEND_LABELS[forcing],
                      alpha=1.0 if forcing in revealed else 0.12) for forcing in order]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False)
    for forcing, text in zip(order, legend.get_texts()):
        text.set_alpha(1.0 if forcing in revealed else 0.22)
    return legend


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------

def label_line_end(ax, x, y, text, color, **kwargs):
    """Write ``text`` just past the point (x, y), outside the axes if need be.

    Returns:
        matplotlib.text.Annotation
    """
    return ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                       color=color, va="center", clip_on=False, **kwargs)


def label_band_edges(ax, da, pairs, x, color="0.45", quantile_dim="quantile", **kwargs):
    """Label each edge of a quantile plume with its level, at its last point.

    Args:
        ax (plt.Axes): Axis to annotate.
        da (xr.DataArray): The quantile series the plume was drawn from.
        pairs (array-like): (low, high) quantile pairs, as passed to ``charts.plot_quantile_bands``.
        x (float): x position of the labels.
        color: Text colour.
        quantile_dim (str): Dimension holding the quantiles.
        **kwargs: Passed to ``ax.annotate``.
    """
    for q in np.unique(np.asarray(pairs)):
        label_line_end(ax, x, da.sel({quantile_dim: q}, method="nearest").isel({da.dims[-1]: -1}),
                       f"{q:.0%}", color, **kwargs)


# ---------------------------------------------------------------------------
# 3. How the figures are made, for talks
# ---------------------------------------------------------------------------

def _density(values, x):
    """Gaussian kernel density of the values (NaN skipped) at ``x``, Scott's bandwidth."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    bandwidth = values.std(ddof=1) * values.size ** (-1 / 5)
    z = (x[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z ** 2).sum(1) / (values.size * bandwidth * np.sqrt(2 * np.pi))


#(t): The verdict of a test at a point, by the sign of a significant change (0: not significant): its words and colour
VERDICTS = {1: ("significant increase", "#b2182b"), -1: ("significant decrease", "#2166ac"), 0: ("not significant", MUTED)}


def plot_point_test(hist_nat_da, experiment_da, change, pvalue, experiment, ax=None, alpha=0.05, units="°C"):
    """One model's test at one grid point: hist-nat's values and the experiment's, the change, its p-value, the verdict.

    Each set of values is drawn as its density (members and years pooled), its mean as a dashed line; the change,
    its p-value and the verdict (a significant increase or decrease, or not significant) go in the top-left corner.
    Notebook 05 draws one per model, then counts the verdicts.

    Args:
        hist_nat_da, experiment_da (xr.DataArray): The values at the point, on any dims (e.g. member, year).
        change, pvalue (float): The test's change in the mean, and its p-value.
        experiment (str): The experiment's name, which also picks its colour (FORCING_COLORS).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        alpha (float): The significance level.
        units (str): The values' units.

    Returns:
        matplotlib.axes.Axes: The axis, with the two densities labelled for a legend.

    Example::

        plot_point_test(hist_nat_da, historical_da, change=0.8, pvalue=0.01, experiment='historical')
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, panel_width=3.0,
                                                          panel_height=2.6))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    #(c): Both densities over the range of both sets of values, a tenth more each side
    pooled_arr = np.concatenate([np.ravel(hist_nat_da.values), np.ravel(experiment_da.values)])
    pooled_arr = pooled_arr[np.isfinite(pooled_arr)]
    x_arr = np.linspace(pooled_arr.min() - 0.1 * np.ptp(pooled_arr), pooled_arr.max() + 0.1 * np.ptp(pooled_arr), 300)
    for values_da, color, label in ((hist_nat_da, FORCING_COLORS["hist-nat"], "hist-nat"),
                                    (experiment_da, FORCING_COLORS.get(experiment, INK), experiment)):
        density_arr = _density(values_da.values, x_arr)
        ax.fill_between(x_arr, density_arr, color=color, alpha=0.25, lw=0)
        ax.plot(x_arr, density_arr, color=color, lw=1.6, label=label)
        ax.axvline(float(values_da.mean()), color=color, lw=1.2, ls=(0, (3, 2)))
    #(c): The change, its p-value and the verdict, in the room left above the curves
    verdict, verdict_colour = VERDICTS[int(np.sign(change)) if pvalue < alpha else 0]
    ax.text(0.02, 0.98, f"Δ = {change:+.2f} {units}, p = {pvalue:.2g}", transform=ax.transAxes, va="top",
            fontsize=9, color=INK_2)
    ax.text(0.02, 0.84, verdict, transform=ax.transAxes, va="top", fontsize=9.5, fontweight="bold",
            color=verdict_colour)
    ax.set_yticks([])
    ax.set_xlabel(units, fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
    return ax


def plot_joint_key_cell(shift, width, colour, fill=True, ax=None):
    """One cell of the joint key drawn as what it means: hist-nat's distribution (grey, dashed) and a changed one
    shifted by ``shift`` and ``width`` times as wide, in the cell's colour, on the cell's colour faded.

    No data: a picture to explain the joint maps in a talk (notebook 05), one cell per class of
    ``forced_response.joint_scale``.

    Args:
        shift (float): The changed distribution's mean, hist-nat's being 0 with a standard deviation of 1.
        width (float): Its standard deviation.
        colour: The class's colour.
        fill (bool): Fill the changed distribution (not for the middle class, no change, which would hide hist-nat's).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, panel_width=2.0,
                                                          panel_height=1.6))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    x_arr = np.linspace(-5, 5, 400)
    ax.set_facecolor(colour)
    ax.patch.set_alpha(0.35)
    ax.plot(x_arr, np.exp(-0.5 * x_arr ** 2) / np.sqrt(2 * np.pi), color="0.35", lw=1.4, ls=(0, (3, 2)))
    changed_arr = np.exp(-0.5 * ((x_arr - shift) / width) ** 2) / (width * np.sqrt(2 * np.pi))
    ax.fill_between(x_arr, changed_arr, color=colour, alpha=0.95 if fill else 0.0, lw=0)
    ax.plot(x_arr, changed_arr, color=INK, lw=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 0.7)
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    return ax
