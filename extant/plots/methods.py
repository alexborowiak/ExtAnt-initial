"""Panels that show how each step of the analysis works, on real data at one grid point.

Each ``plot_*(..., ax=None)`` draws one panel; the notebook makes the figure and its axes, draws each panel and titles
them, as everywhere (``charts``). Two kinds:

- schematics of the significance tests, step by step: the hist-nat bootstrap of the width and of the mean, the
  member-block permutation test of the mean change, and the probability ratio with the share of it the shift of the
  mean gives. Their axes come from ``schematic_axes``: a panel (a), an arrow with the step it stands for, nine
  example trials (b), another arrow, and every trial (c). Each is given the test's own pieces at one point, as
  notebook 03 makes them, and rebuilds its example trials from the test's draws, so every number on it is the test's
  own. A bootstrap trial is one sample, so each of its example panels shows that sample alone; a permutation splits
  the members in two, so each of its panels shows both groups;
- quick looks that follow the data through the notebooks: monthly to seasonal means, the year x season split, a
  rolling quantile, the smoothing, the change against hist-nat, the forced response, the two samples of a test, the
  members per model and experiment, and the SAM phases. Their axes come from a GridSpec, each styled by
  the function that draws on it (``charts.style_chart_ax``).

Sections
--------
1. The schematics' layout: ``schematic_axes``, and its arrows (``plot_flow``)
2. The bootstrap of the width and of the mean: panel a (``plot_centred_ensembles``, ``plot_two_ensembles``), the
   example trials (``plot_width_trials``, ``plot_mean_trials``), every trial (``plot_trial_distribution``)
3. The permutation test: the example relabellings (``plot_relabellings``), every relabelling
   (``plot_permutation_distribution``)
4. The probability ratio: the thresholds (``plot_thresholds``), the shift of the mean alone (``plot_shift_only``),
   the bootstrap's ratios (``plot_ratio_trials``)
5. Following the data
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import charts
from plotting_modules.constants import FORCING_COLORS, FORCING_LEGEND_LABELS, SEASON_COLORS, SEASONS

REFERENCE_GREY = "0.55"


def _color(experiment):
    return FORCING_COLORS.get(experiment, "C0")


def _label(experiment):
    return FORCING_LEGEND_LABELS.get(experiment, experiment)


def _clean(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _qrange(values, quantiles):
    low, high = np.quantile(values, quantiles)
    return high - low, low, high


# ---------------------------------------------------------------------------
# 1. The schematics' layout
# ---------------------------------------------------------------------------

def schematic_axes(fig, with_trials=True):
    """The axes of a test's schematic: a, an arrow, b (3 × 3 example trials with a '⋯' row before the last three),
    another arrow, c. Without the trials: a, an arrow, c.

    Every panel but the arrows is styled as a chart (``charts.style_chart_ax``).

    Args:
        fig (matplotlib.figure.Figure): The figure, e.g. ``plt.figure(figsize=(11, 15))`` (with the trials) or
            ``(11, 8.5)`` (without).
        with_trials (bool): Whether there are example trials (the bootstrap and permutation schematics) or not (the
            probability ratio's).

    Returns:
        tuple: Panel a, the arrows' axes (a list: two with the trials, one without), the example trials' axes (an array
        of 3 × 3, empty without), panel c.

    Example::

        fig = plt.figure(figsize=(11, 15))
        ax_a, flow_axes, trial_axes, ax_c = schematic_axes(fig)
        plot_two_ensembles(hist_nat_da, experiment_da, 'hist-GHG', ax=ax_a)
        plot_flow('Draw 10 of the 50 hist-nat members ...', ax=flow_axes[0])
    """
    if not with_trials:
        outer = fig.add_gridspec(3, 1, height_ratios=[2.3, 0.6, 2.3], hspace=0.55)
        ax_a, ax_flow, ax_c = (fig.add_subplot(outer[i]) for i in range(3))
        for ax in (ax_a, ax_c):
            charts.style_chart_ax(ax)
        ax_flow.axis("off")
        return ax_a, [ax_flow], np.empty((0, 3), dtype=object), ax_c

    outer = fig.add_gridspec(5, 1, height_ratios=[2.3, 0.65, 4.1, 0.65, 2.3], hspace=0.55)
    ax_a = fig.add_subplot(outer[0])
    flow_axes = [fig.add_subplot(outer[1]), fig.add_subplot(outer[3])]
    ax_c = fig.add_subplot(outer[4])
    #(c): The trials in rows 1, 2 and 4 of the middle block; row 3 is the '⋯' between the first six and the last three
    grid_b = outer[2].subgridspec(4, 3, height_ratios=[1, 1, 0.28, 1], hspace=0.55, wspace=0.18)
    trial_axes = np.array([[fig.add_subplot(grid_b[row, col]) for col in range(3)] for row in (0, 1, 3)], dtype=object)
    ax_dots = fig.add_subplot(grid_b[2, :])
    ax_dots.axis("off")
    ax_dots.text(0.5, 0.5, "⋯", ha="center", va="center", fontsize=24)
    for ax in (ax_a, ax_c, *trial_axes.ravel()):
        charts.style_chart_ax(ax)
    for ax in flow_axes:
        ax.axis("off")
    return ax_a, flow_axes, trial_axes, ax_c


def plot_flow(text, ax):
    """A downward arrow with the step it stands for, in one of ``schematic_axes``' arrow rows.

    Args:
        text (str): The step, on one or two lines.
        ax (matplotlib.axes.Axes): The arrow's axis.
    """
    ax.annotate("", xy=(0.5, 0.08), xytext=(0.5, 0.38), arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(0.5, 0.72, text, ha="center", va="center", multialignment="center", fontsize=11)


def example_trials(n_trials):
    """The trials shown as examples: the first six and the last three, by their positions among ``n_trials``."""
    return [0, 1, 2, 3, 4, 5, n_trials - 3, n_trials - 2, n_trials - 1]


def _range_bar(ax, low, high, height, color, lw=3):
    ax.plot([low, high], [height, height], color=color, lw=lw, solid_capstyle="butt")


def _trial_axes(ax, position, edges, top):
    ax.text(0.04, 0.94, f"{position + 1:,}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ---------------------------------------------------------------------------
# 2. The bootstrap of the width and of the mean
# ---------------------------------------------------------------------------

def plot_centred_ensembles(hist_nat, final, experiment, observed, quantiles=(0.05, 0.95), units="°C", ax=None):
    """Panel a of the width's bootstrap: the two ensembles over the same final years, each centred on its own mean so
    their widths compare, with their Q05-Q95 ranges as bars and the observed change in width.

    Args:
        hist_nat, final (xr.DataArray): hist-nat's and the experiment's final years at one point, on (member, year).
        experiment (str): Name of the experiment tested.
        observed (float): The experiment's width.
        quantiles (tuple[float, float]): The width's lower and upper quantile.
        units (str): Units of the variable.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the two ensembles.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    nat_values, exp_values = _clean(hist_nat), _clean(final)
    nat_width = _qrange(nat_values, quantiles)[0]
    nat_c, exp_c = nat_values - nat_values.mean(), exp_values - exp_values.mean()
    edges = np.histogram_bin_edges(np.concatenate([nat_c, exp_c]), bins=30)
    ax.hist(nat_c, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=1.4,
            label=f"{_label('hist-nat')}, {hist_nat.sizes['member']} members")
    ax.hist(exp_c, bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=1.4,
            label=f"{_label(experiment)}, {final.sizes['member']} members")
    top = ax.get_ylim()[1]
    _range_bar(ax, *_qrange(nat_c, quantiles)[1:], 0.07 * top, nat_color)
    _range_bar(ax, *_qrange(exp_c, quantiles)[1:], 0.13 * top, exp_color)
    ax.legend(frameon=False, fontsize=10)
    observed = float(observed)
    ax.text(0.02, 0.95, rf"$\Delta_{{obs}}$ = {observed:.2f} − {nat_width:.2f} = {observed - nat_width:+.2f} {units}"
            "\n" "(bars: Q05 to Q95)", transform=ax.transAxes, va="top", fontsize=11)
    return ax


def plot_two_ensembles(hist_nat, final, experiment, units="°C", ax=None):
    """Panel a of the mean's bootstrap and the permutation test: the two ensembles over the same final years, with
    their means (lines) and the difference between them.

    Args:
        hist_nat, final (xr.DataArray): hist-nat's and the experiment's final years at one point, on (member, year).
        experiment (str): Name of the experiment tested.
        units (str): Units of the variable.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the two ensembles.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    nat_values, exp_values = _clean(hist_nat), _clean(final)
    edges = np.histogram_bin_edges(np.concatenate([nat_values, exp_values]), bins=30)
    ax.hist(nat_values, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=1.4,
            label=f"{_label('hist-nat')}, {hist_nat.sizes['member']} members")
    ax.hist(exp_values, bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=1.4,
            label=f"{_label(experiment)}, {final.sizes['member']} members")
    ax.axvline(nat_values.mean(), color=nat_color, lw=2)
    ax.axvline(exp_values.mean(), color=exp_color, lw=2)
    ax.legend(frameon=False, fontsize=10)
    ax.text(0.02, 0.95, rf"$\Delta_{{obs}}$ = {exp_values.mean() - nat_values.mean():+.2f} {units}" "\n"
            "(lines: the means)", transform=ax.transAxes, va="top", fontsize=11)
    return ax


def plot_width_trials(hist_nat, selected, trials, trial_axes, quantiles=(0.05, 0.95)):
    """Panel b of the width's bootstrap: example trials, each its own sample of hist-nat members alone, centred, with
    its Q05-Q95 range and width.

    Each trial is rebuilt from ``selected`` and checked against ``trials``, so the draws must be the ones behind
    ``trials``.

    Args:
        hist_nat (xr.DataArray): hist-nat's final years at one point, on (member, year).
        selected (np.ndarray): (trial, member) True for each trial's hist-nat members.
        trials (xr.DataArray): Each trial's width, on (trial,).
        trial_axes (np.ndarray): The 3 × 3 example axes of ``schematic_axes``.
        quantiles (tuple[float, float]): The width's lower and upper quantile.
    """
    nat_color = _color("hist-nat")
    hist_nat_values = hist_nat.transpose("member", "year").values
    trial_values = np.asarray(trials, dtype=float)
    samples = [(position, _clean(hist_nat_values[selected[position]]))
               for position in example_trials(trial_values.size)]
    samples = [(position, values - values.mean()) for position, values in samples]
    edges = np.histogram_bin_edges(np.concatenate([values for _, values in samples]), bins=16)
    top = max(np.histogram(values, edges, density=True)[0].max() for _, values in samples)
    for ax, (position, values) in zip(trial_axes.ravel(), samples):
        ax.hist(values, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=0.8)
        width, low, high = _qrange(values, quantiles)
        _range_bar(ax, low, high, 0.1 * top, nat_color, lw=2)
        #(c): The same number as the test's own trial (float32 there)
        assert np.isclose(width, trial_values[position], atol=1e-3), "trial does not match the test"
        ax.text(0.96, 0.94, f"W={width:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        _trial_axes(ax, position, edges, top)


def plot_mean_trials(hist_nat, selected, trials, trial_axes):
    """Panel b of the mean's bootstrap: example trials, each its own sample of hist-nat members alone, with its mean.

    Each trial is rebuilt from ``selected`` and checked against ``trials``, so the draws must be the ones behind
    ``trials``.

    Args:
        hist_nat (xr.DataArray): hist-nat's final years at one point, on (member, year).
        selected (np.ndarray): (trial, member) True for each trial's hist-nat members.
        trials (xr.DataArray): Each trial's mean, on (trial,).
        trial_axes (np.ndarray): The 3 × 3 example axes of ``schematic_axes``.
    """
    nat_color = _color("hist-nat")
    hist_nat_values = hist_nat.transpose("member", "year").values
    trial_values = np.asarray(trials, dtype=float)
    samples = [(position, _clean(hist_nat_values[selected[position]]))
               for position in example_trials(trial_values.size)]
    edges = np.histogram_bin_edges(np.concatenate([values for _, values in samples]), bins=16)
    top = max(np.histogram(values, edges, density=True)[0].max() for _, values in samples)
    for ax, (position, values) in zip(trial_axes.ravel(), samples):
        ax.hist(values, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=0.8)
        ax.axvline(values.mean(), color=nat_color, lw=1.5)
        assert np.isclose(values.mean(), trial_values[position], rtol=1e-6, atol=1e-6), "trial does not match the test"
        ax.text(0.96, 0.94, f"mean={values.mean():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        _trial_axes(ax, position, edges, top)


def plot_trial_distribution(trials, observed, hist_nat_value, experiment, n_members, bins=40, ax=None):
    """Panel c of a bootstrap: every trial's value, their central 95%, hist-nat's own value and the experiment's.

    Args:
        trials (xr.DataArray): Each trial's value (width or mean), on (trial,).
        observed (float): The experiment's value.
        hist_nat_value (float): hist-nat's own value, all its members.
        experiment (str): Name of the experiment tested.
        n_members (int): hist-nat's number of members.
        bins (int): Bins of the histogram.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    trials = _clean(trials)
    low, high = np.quantile(trials, [0.025, 0.975])
    ax.axvspan(low, high, color="0.9", zorder=0, label="central 95% of trials")
    ax.hist(trials, bins=bins, density=True, color=nat_color, edgecolor=nat_color, alpha=0.65, lw=0.8,
            label="hist-nat trials")
    ax.axvline(hist_nat_value, color=REFERENCE_GREY, lw=1.5, label=f"hist-nat, all {n_members} members")
    ax.axvline(float(observed), color=exp_color, lw=2.5, ls="--", label=f"{experiment}")
    ax.legend(frameon=False, fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# 3. The permutation test
# ---------------------------------------------------------------------------

def plot_relabellings(experiment_final, hist_nat_final, is_experiment, permutations, experiment, trial_axes):
    """Panel b of the permutation test: example relabellings, whole members shuffled between the two labels (each
    member's years kept together), both groups and their means.

    Each relabelling is rebuilt from ``is_experiment`` and checked against ``permutations``, so the two must come from
    the same draws.

    Args:
        experiment_final, hist_nat_final (xr.DataArray): The final years of each at one point, on (member, year).
        is_experiment (np.ndarray): (trial, pooled member) True for the members each relabelling calls the
            experiment; the experiment's own members come first in the pool.
        permutations (xr.DataArray): Each relabelling's difference in means, on (trial,).
        experiment (str): Name of the experiment.
        trial_axes (np.ndarray): The 3 × 3 example axes of ``schematic_axes``.
    """
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    exp_values = experiment_final.transpose("member", "year").values
    nat_values = hist_nat_final.transpose("member", "year").values
    permutation_values = np.asarray(permutations, dtype=float)
    #(c): The experiment's own members first in the pool, as in the test
    pool = np.concatenate([exp_values, nat_values]).astype(float)
    samples = [(position, pool[is_experiment[position]], pool[~is_experiment[position]])
               for position in example_trials(permutation_values.size)]
    edges = np.histogram_bin_edges(pool[np.isfinite(pool)], bins=16)
    top = max(max(np.histogram(_clean(e), edges, density=True)[0].max(),
                  np.histogram(_clean(r), edges, density=True)[0].max()) for _, e, r in samples)
    for ax, (position, exp_i, nat_i) in zip(trial_axes.ravel(), samples):
        ax.hist(_clean(nat_i), bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=0.8)
        ax.hist(_clean(exp_i), bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=0.8)
        ax.axvline(np.nanmean(nat_i), color=nat_color, lw=1.5)
        ax.axvline(np.nanmean(exp_i), color=exp_color, lw=1.5)
        delta = np.nanmean(exp_i.mean(1)) - np.nanmean(nat_i.mean(1))
        assert np.isclose(delta, permutation_values[position], atol=1e-4), "relabelling does not match the test"
        ax.text(0.96, 0.94, rf"$\Delta$={delta:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        _trial_axes(ax, position, edges, top)


def plot_permutation_distribution(permutations, mean_change, experiment, bins=40, ax=None):
    """Panel c of the permutation test: every relabelling's difference in means, and the observed one.

    Args:
        permutations (xr.DataArray): Each relabelling's difference in means, on (trial,).
        mean_change (float): The observed difference in means.
        experiment (str): Name of the experiment.
        bins (int): Bins of the histogram.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    ax.hist(_clean(permutations), bins=bins, density=True, color=nat_color, edgecolor=nat_color, alpha=0.65, lw=0.8,
            label="Relabellings")
    ax.axvline(float(mean_change), color=exp_color, lw=2.5, ls="--", label="Observed")
    ax.legend(frameon=False, fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# 4. The probability ratio
# ---------------------------------------------------------------------------

#(t): Colours of the two tails in the probability-ratio figures
TAIL_COLORS = {"warm": "#c0392b", "cold": "#2166ac"}
TAIL_SIGNS = {"warm": 1, "cold": -1}


def percentile_words(chance):
    """hist-nat's two percentiles for a chance, as words: e.g. 0.05 -> ("95th", "5th")."""
    return f"{100 * (1 - chance):g}th", f"{100 * chance:g}th"


def _chance(values, level, tail):
    """The share of ``values`` beyond ``level`` on the ``tail`` side, in %."""
    return 100 * np.mean(TAIL_SIGNS[tail] * (values - level) > 0)


def plot_thresholds(experiment_final, hist_nat_final, thresholds, experiment, bins=30, ax=None):
    """Panel a of the probability ratio: the two ensembles' final years, hist-nat's two percentiles (dashed), and how
    often each ensemble goes beyond them, as the ratio.

    Args:
        experiment_final, hist_nat_final (xr.DataArray): The final years at one point, on (member, year).
        thresholds (xr.DataArray): hist-nat's percentiles on (tail,), for one chance.
        experiment (str): Name of the experiment.
        bins (int): Histogram bins.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_values, nat_values = _clean(experiment_final), _clean(hist_nat_final)
    edges = np.histogram_bin_edges(np.concatenate([exp_values, nat_values]), bins=bins)
    for values, name in ((nat_values, "hist-nat"), (exp_values, experiment)):
        ax.hist(values, bins=edges, density=True, color=_color(name), edgecolor=_color(name), alpha=0.28, lw=1.4,
                label=_label(name))
    lines = []
    for tail in ("warm", "cold"):
        value = float(thresholds.sel(tail=tail))
        ax.axvline(value, color=TAIL_COLORS[tail], lw=1.8, ls="--")
        p_nat, p_exp = _chance(nat_values, value, tail), _chance(exp_values, value, tail)
        lines.append(f"{tail}: hist-nat {p_nat:.1f}%, {experiment} {p_exp:.1f}%  →  PR = {p_exp / p_nat:.2f}")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return ax


def plot_shift_only(experiment_final, hist_nat_final, thresholds, shift, experiment, units="°C", bins=30, ax=None):
    """Panel a of the shift-only ratio: hist-nat's seasons, every one moved by the change in the mean, against the
    experiment itself and hist-nat's thresholds; the shift-only ratio next to the probability ratio, whose gap is what
    the width (and shape) adds.

    Args:
        experiment_final, hist_nat_final (xr.DataArray): The final years at one point, on (member, year).
        thresholds (xr.DataArray): hist-nat's percentiles on (tail,), for one chance.
        shift (float): The change in the mean, experiment minus hist-nat.
        experiment (str): Name of the experiment.
        units (str): Units of the variable.
        bins (int): Histogram bins.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    exp_values, nat_values = _clean(experiment_final), _clean(hist_nat_final)
    shift = float(shift)
    moved = nat_values + shift
    edges = np.histogram_bin_edges(np.concatenate([exp_values, moved]), bins=bins)
    ax.hist(moved, bins=edges, density=True, histtype="step", color=nat_color, lw=2,
            label=f"hist-nat + Δ (Δ = {shift:+.2f} {units}): the shift only")
    ax.hist(exp_values, bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=1.4,
            label=f"{experiment}: everything that changed")
    lines = []
    for tail in ("warm", "cold"):
        value = float(thresholds.sel(tail=tail))
        ax.axvline(value, color=TAIL_COLORS[tail], lw=1.8, ls="--")
        p_nat = _chance(nat_values, value, tail)
        lines.append(f"{tail}: shift only {_chance(moved, value, tail) / p_nat:.2f}, "
                     f"probability ratio {_chance(exp_values, value, tail) / p_nat:.2f}")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return ax


def plot_ratio_trials(trials, label, ax=None, **style):
    """Each tail's bootstrap ratios as a histogram of log2, ticked with the ratios; a ratio of 0 (or beyond 64) piles up
    at the end. Call it twice on one axis to compare two ratios (e.g. ``histtype='step'`` for the second).

    Args:
        trials (xr.DataArray): Every bootstrap trial's ratio, on (trial, tail).
        label (str): What the ratio is, for the legend, e.g. 'probability ratio'.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        **style: How the histograms are drawn, e.g. ``alpha=0.45, histtype='stepfilled'``.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    log_edges = np.linspace(-6, 6, 97)
    for tail in ("warm", "cold"):
        ratios = np.clip(np.asarray(trials.sel(tail=tail), dtype=float), 2.0 ** -6, 2.0 ** 6)
        ax.hist(_clean(np.log2(ratios)), bins=log_edges, density=True, color=TAIL_COLORS[tail],
                label=f"{tail}, {label}", **style)
    ax.axvline(0, color="k", lw=1)
    ticks = np.arange(-6, 7, 2)
    ax.set_xticks(ticks, [f"1/{2 ** -t}" if t < 0 else f"{2 ** t}" for t in ticks])
    ax.legend(frameon=False, fontsize=9, ncols=2)
    return ax


# ---------------------------------------------------------------------------
# 5. Following the data
# ---------------------------------------------------------------------------

def _decimal_year(time):
    """Mid-month positions as decimal years, for any calendar."""
    return time.dt.year.values + (time.dt.month.values - 0.5) / 12


#(t): Where each season's months sit within its (year, season) label, as decimal-year offsets
_SEASON_SPAN = {"DJF": (11, 14), "MAM": (2, 5), "JJA": (5, 8), "SON": (8, 11)}


def plot_seasonal_means(monthly, seasonal, years, ax=None):
    """Monthly values (grey) and the seasonal means made from them (a bar across each season's months), for one
    member at one point.

    Args:
        monthly (xr.DataArray): One series on ``time``, in analysis units.
        seasonal (xr.DataArray): The same member's seasonal means on (year, season).
        years (slice): Years shown, e.g. ``slice(1990, 1994)``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend above it.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.8, panel_height=2.6)
    charts.style_chart_ax(ax)
    months = monthly.sel(time=slice(str(years.start), str(years.stop)))
    ax.plot(_decimal_year(months.time), months.values, color="0.6", lw=0.8, marker="o", ms=3, zorder=2)
    for year in range(years.start, years.stop):
        for season in SEASONS:
            first, last = _SEASON_SPAN[season]
            value = float(seasonal.sel(year=year, season=season))
            ax.plot([year + first / 12, year + last / 12], [value, value], color=SEASON_COLORS[season], lw=3,
                    solid_capstyle="butt", zorder=3)
    ax.legend(handles=[Line2D([], [], color="0.6", marker="o", ms=3, lw=0.8, label="Monthly means"),
                       *[Line2D([], [], color=SEASON_COLORS[s], lw=3, label=f"{s} mean") for s in SEASONS]],
              ncols=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    return ax


def plot_seasons_in_time(seasonal, ax=None):
    """Seasonal means in time order: each season's value at its middle month, in the season's colour.

    Args:
        seasonal (xr.DataArray): One member at one point on (year, season).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    for season in SEASONS:
        first, last = _SEASON_SPAN[season]
        ax.scatter(seasonal["year"].values + (first + last) / 24, seasonal.sel(season=season).values, s=5,
                   color=SEASON_COLORS[season])
    return ax


def plot_seasons_split(seasonal, ax=None):
    """The same seasonal means after ``split_time``: one series per season (year × season), in the season's colour.

    Args:
        seasonal (xr.DataArray): One member at one point on (year, season).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the seasons.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    for season in SEASONS:
        ax.plot(seasonal["year"].values, seasonal.sel(season=season).values, color=SEASON_COLORS[season], lw=1,
                label=season)
    ax.legend(frameon=False, fontsize=8, ncols=len(SEASONS))
    return ax


def quantile_shades(n_quantiles):
    """``n_quantiles`` colours of viridis, low to high, one per quantile drawn."""
    return plt.get_cmap("viridis")(np.linspace(0.15, 0.85, n_quantiles))


def plot_window(members, year, window, color="C0", ax=None):
    """Every member through time, with the ``window`` years around ``year`` shaded: what one rolling quantile pools.

    Args:
        members (xr.DataArray): One point and season on (member, year).
        year (int): Window centre.
        window (int): Window length in years.
        color: Colour of the members.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    half = window // 2
    charts.plot_lines(members, "year", ax=ax, color=color, linewidth=0.4, alpha=0.4)
    ax.axvspan(year - half - 0.5, year + half + 0.5, color="0.85", zorder=0)
    return ax


def plot_window_values(members, year, window, quantiles=(0.05, 0.5, 0.95), color="C0", ax=None):
    """The pooled values of one window, as a sideways histogram, with their quantiles (``quantile_shades``).

    Args:
        members (xr.DataArray): One point and season on (member, year).
        year (int): Window centre.
        window (int): Window length in years.
        quantiles (Sequence[float]): Quantiles drawn.
        color: Colour of the histogram.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis; its title can say how many values (``ax.patches`` are the bars).
    """
    if ax is None:
        ax = charts.one_panel(panel_width=1.5)
    charts.style_chart_ax(ax)
    half = window // 2
    pooled = _clean(members.sel(year=slice(year - half, year + half)))
    ax.hist(pooled, bins=20, orientation="horizontal", color=color, alpha=0.5)
    for q, shade in zip(quantiles, quantile_shades(len(quantiles))):
        ax.axhline(np.quantile(pooled, q), color=shade, lw=2)
    return ax


def plot_rolling_quantiles(rolling, year=None, quantiles=(0.05, 0.5, 0.95), ax=None):
    """Rolling quantiles through time (``quantile_shades``), with the window centred on ``year`` marked.

    Args:
        rolling (xr.DataArray): ``quantiles.centred_rolling_quantiles`` output, on (quantile, year).
        year (int | None): The window marked; none if None.
        quantiles (Sequence[float]): Quantiles drawn (must be in ``rolling``).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the quantiles.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    for q, shade in zip(quantiles, quantile_shades(len(quantiles))):
        series = rolling.sel(quantile=q, method="nearest")
        ax.plot(series["year"], series, color=shade, lw=1.5, label=f"Q{q * 100:02.0f}")
        if year is not None:
            ax.plot(year, float(series.sel(year=year)), "o", color=shade, ms=6, mec="white")
    if year is not None:
        ax.axvline(year, color="0.6", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=8, ncols=len(quantiles))
    return ax


def plot_smoothing(rolling, smooth, ax=None):
    """Rolling quantiles before (thin) and after (thick) LOWESS smoothing, at one point.

    Args:
        rolling, smooth (xr.DataArray): On (quantile, year).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the quantiles.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=8.0)
    charts.style_chart_ax(ax)
    shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, rolling.sizes["quantile"]))
    for q, shade in zip(rolling["quantile"].values, shades):
        ax.plot(rolling["year"], rolling.sel(quantile=q), color=shade, lw=0.8, alpha=0.5)
        ax.plot(smooth["year"], smooth.sel(quantile=q), color=shade, lw=2.2, label=f"Q{q * 100:g}")
    ax.legend(frameon=False, fontsize=8, ncols=rolling.sizes["quantile"], loc="upper left")
    return ax


def plot_smoothed_quantiles(smooth, base, experiment, quantiles=(0.05, 0.95), ax=None):
    """The smoothed quantiles of an experiment and of hist-nat (solid and dashed for the two quantiles), with
    hist-nat's whole-record quantiles dotted.

    Args:
        smooth (xr.DataArray): Smoothed rolling quantiles on (experiment, quantile, year), at one point, with the
            experiment and hist-nat.
        base (xr.DataArray): hist-nat's whole-record quantiles on (quantile,), at the same point.
        experiment (str): The experiment.
        quantiles (tuple[float, float]): The pair drawn.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    exp_color, ref_color = _color(experiment), _color("hist-nat")
    styles = dict(zip(quantiles, ("-", "--")))
    for q in quantiles:
        for name, color in (("hist-nat", ref_color), (experiment, exp_color)):
            series = smooth.sel(experiment=name).sel(quantile=q, method="nearest")
            ax.plot(series["year"], series, color=color, lw=1.8, ls=styles[q])
        ax.axhline(float(base.sel(quantile=q, method="nearest")), color="0.4", lw=1, ls=":")
    ax.legend(handles=[Line2D([], [], color=exp_color, lw=2, label=experiment),
                       Line2D([], [], color=ref_color, lw=2, label="hist-nat"),
                       Line2D([], [], color="0.4", lw=1, ls=":", label="hist-nat, whole record"),
                       *[Line2D([], [], color="0.2", ls=styles[q], label=f"Q{q * 100:02.0f}") for q in quantiles]],
              frameon=False, fontsize=8, ncols=2)
    return ax


def plot_quantile_changes(smooth, base, experiment, quantiles=(0.05, 0.95), ax=None):
    """The change in each of two quantiles, an experiment's smoothed quantiles minus hist-nat's whole-record ones, and
    so the change in width, the gap between them (shaded).

    Args:
        smooth, base, experiment, quantiles: As ``plot_smoothed_quantiles``'.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    exp_color = _color(experiment)
    styles = dict(zip(quantiles, ("-", "--")))
    change = smooth.sel(experiment=experiment) - base
    low, high = (change.sel(quantile=q, method="nearest") for q in quantiles)
    ax.plot(low["year"], low, color=exp_color, lw=1.8, ls=styles[quantiles[0]], label=f"ΔQ{quantiles[0] * 100:02.0f}")
    ax.plot(high["year"], high, color=exp_color, lw=1.8, ls=styles[quantiles[1]], label=f"ΔQ{quantiles[1] * 100:02.0f}")
    ax.fill_between(high["year"], low, high, color=exp_color, alpha=0.12, lw=0, label="Δ(width) = gap between the lines")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_two_samples(experiment_values, hist_nat_values, experiment, t=None, p=None, ax=None):
    """The two samples a test compares (e.g. the final years of each ensemble, members and years pooled), with their
    means, and the test's statistic if given.

    Args:
        experiment_values, hist_nat_values (xr.DataArray): The two samples, any shape.
        experiment (str): The experiment's name.
        t, p (float | None): Test statistic and p-value to print.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=6.0)
    charts.style_chart_ax(ax)
    exp, ref = _clean(experiment_values), _clean(hist_nat_values)
    edges = np.histogram_bin_edges(np.concatenate([exp, ref]), bins=30)
    for values, name in ((ref, "hist-nat"), (exp, experiment)):
        ax.hist(values, bins=edges, density=True, color=_color(name), alpha=0.3, label=f"{name} (n = {values.size})")
        ax.axvline(values.mean(), color=_color(name), lw=2)
    if t is not None:
        ax.text(0.02, 0.95, f"t = {t:.2f}, p = {p:.2g}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_member_counts(counts, ax=None):
    """Members per model (rows) and experiment (columns), each cell its number, grey where there are none.

    Args:
        counts (pd.DataFrame): models × experiments.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.image.AxesImage: The cells, for a colour bar (``image.axes`` is the axis).
    """
    if ax is None:
        ax = charts.one_panel(panel_width=0.9 * counts.shape[1] + 0.5, panel_height=0.4 * counts.shape[0] + 0.3)
    charts.style_chart_ax(ax)
    cmap = plt.get_cmap("Blues").with_extremes(bad="0.9")
    values = counts.values.astype(float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto")
    ax.set_xticks(range(counts.shape[1]), counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(counts.shape[0]), counts.index)
    for (row, col), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(col, row, int(value), ha="center", va="center",
                    color="white" if value > np.nanmax(values) * 0.6 else "black")
    ax.grid(False)
    return image


#(t): Colours of the three SAM phases
PHASE_COLORS = {"negative": "#2166ac", "neutral": "0.45", "positive": "#b2182b"}


def plot_sam_index(hist_nat_index, experiment_index, thresholds, experiment, ax=None):
    """The SAM index of both ensembles over the final years, hist-nat's two terciles (dashed), and the share of each
    ensemble's seasons in each phase.

    Args:
        hist_nat_index, experiment_index (xr.DataArray): The index over the final years, on (member, year).
        thresholds (xr.DataArray): hist-nat's terciles, on (quantile,).
        experiment (str): The experiment.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=10.0)
    charts.style_chart_ax(ax)
    lower, upper = (float(value) for value in thresholds.values)
    nat_index, exp_index = _clean(hist_nat_index), _clean(experiment_index)
    edges = np.histogram_bin_edges(np.concatenate([nat_index, exp_index]), bins=30)
    for values, name in ((nat_index, "hist-nat"), (exp_index, experiment)):
        ax.hist(values, bins=edges, density=True, color=_color(name), edgecolor=_color(name), alpha=0.28, lw=1.4,
                label=_label(name))
    for value in (lower, upper):
        ax.axvline(value, color="k", lw=1.2, ls="--")
    shares = []
    for name, values in (("hist-nat", nat_index), (experiment, exp_index)):
        in_phase = [np.mean(values < lower), np.mean((values >= lower) & (values <= upper)), np.mean(values > upper)]
        shares.append(f"{name}: " + ", ".join(f"{phase} {100 * share:.0f}%"
                                              for phase, share in zip(PHASE_COLORS, in_phase)))
    ax.text(0.02, 0.95, "\n".join(shares), transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return ax


def plot_phase_temperatures(hist_nat_values, experiment_values, experiment, units="°C", edges=None, ax=None):
    """Temperature in the seasons of one SAM phase, both ensembles, with their means and the change between them.

    Args:
        hist_nat_values, experiment_values (xr.DataArray): Temperature in one phase's seasons, on (member, year): one
            phase of ``sam.by_phase`` output.
        experiment (str): The experiment.
        units (str): Units of the variable.
        edges (np.ndarray | None): The histogram's bin edges, the same in every phase's panel; 20 bins over these values
            if None.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    nat, exp = _clean(hist_nat_values), _clean(experiment_values)
    if edges is None:
        edges = np.histogram_bin_edges(np.concatenate([nat, exp]), bins=20)
    for values, color in ((nat, _color("hist-nat")), (exp, _color(experiment))):
        if values.size:
            ax.hist(values, bins=edges, density=True, color=color, edgecolor=color, alpha=0.28, lw=1.2)
            ax.axvline(values.mean(), color=color, lw=2)
    change = exp.mean() - nat.mean() if nat.size and exp.size else np.nan
    ax.text(0.03, 0.95, f"Δ = {change:+.2f} {units}\nseasons: {nat.size} and {exp.size}", transform=ax.transAxes,
            va="top", fontsize=9)
    ax.set_yticks([])
    return ax
