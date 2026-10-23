"""Panels for the observed-change analysis; the calculations are in observed_change.

Each ``plot_*(data, ax=None, ...)`` draws one panel, on an axis from ``charts.chart_axes`` (a new figure if ``ax`` is
None); the notebook makes the figure and its axes, draws each panel and labels them, as everywhere (``charts``); each
``..._legend_handles`` gives a panel's legend entries. Colour follows the experiment (FORCING_COLORS) or the signal
(``charts.SIGNAL_COLORS``), and ERA5 is always black.

Sections
--------
1. Consistency: ERA5's trend among each experiment's member trends (``plot_experiment_strips``)
2. Record rates: cumulative record highs and lows against chance and the members (``plot_record_rates``)
3. Scaling factors and attributable trends (``plot_scaling_factors``, ``plot_attributable_trends``)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import charts
from .charts import MODEL_MARKERS, SIGNAL_COLORS
from .era5_evaluation import ERA5_COLOR, INK, INK_2, MUTED, is_testable, where_era5
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER

#(t): Detection classes (observed_change.DETECTION_CLASSES), validated as a categorical palette
DETECTION_COLORS = ("#d9d8d4", "#2a78d6", "#eb6834", "#4a3aa7")


# ---------------------------------------------------------------------------
# 1. Consistency: ERA5's trend among each experiment's member trends
# ---------------------------------------------------------------------------

def plot_experiment_strips(results, statistic="trend", ax=None):
    """Every experiment's member values of one statistic as a row of dots, with ERA5 as a vertical line.

    The experiments whose rows ERA5 (the black line) passes through could have produced the observed trend; ERA5
    outside hist-nat's row means the observed change is unlikely to be natural. Right of each row, ✓ where ERA5 is
    inside the members' 5-95%, ✗ where it is outside, – where too few members can flag anything.

    Args:
        results (dict[str, xr.Dataset]): Experiment -> ``ev.locate`` results on a ``statistic`` dim, one season; the
            rows in the forcings' order (FORCING_REVEAL_ORDER).
        statistic (str): The statistic shown.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    experiments = [e for e in FORCING_REVEAL_ORDER if e in results] + [e for e in results if e not in FORCING_REVEAL_ORDER]
    era5 = float(next(iter(results.values()))["era5"].sel(statistic=statistic))
    for row, experiment in enumerate(experiments):
        cell = results[experiment].sel(statistic=statistic)
        members = cell["members"].values
        members = members[np.isfinite(members)]
        colour = FORCING_COLORS.get(experiment, INK)
        jitter = ((np.arange(members.size) * 0.618034) % 1 - 0.5) * 0.45
        ax.plot([float(cell["lower"]), float(cell["upper"])], [row, row], color=colour, lw=7, alpha=0.22,
                solid_capstyle="butt", zorder=1)
        ax.scatter(members, row + jitter, s=11, color=colour, edgecolor="white", linewidth=0.3, zorder=2)
        position = where_era5(cell["percentile"]).replace("ERA5 ", "").replace(" every member", " all")
        if int(cell["verdict"]) == 0 and not is_testable(cell):
            #(c): Too few members for the test to flag anything: say so rather than show a ✓
            glyph, colour, position = "–", MUTED, f"{position} (N too small)"
        elif int(cell["verdict"]) == 0:
            glyph, colour = "✓", "#0ca30c"
        else:
            glyph, colour = "✗", "#d03b3b"
        ax.annotate(glyph, (1.01, row), xycoords=("axes fraction", "data"), va="center",
                    color=colour, fontsize=10, fontweight="bold", annotation_clip=False)
        ax.annotate(position, (1.01, row), xycoords=("axes fraction", "data"), xytext=(12, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2, annotation_clip=False)
    ax.axvline(era5, color=ERA5_COLOR, lw=2, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_yticks(range(len(experiments)), experiments)
    ax.set_ylim(len(experiments) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    return ax


def experiment_strip_legend_handles():
    """The legend entries of ``plot_experiment_strips``."""
    return [
        Line2D([], [], color=ERA5_COLOR, lw=2, label="ERA5"),
        Line2D([], [], ls="none", marker="o", ms=5, color="0.5", label="Each member"),
        Patch(color="0.5", alpha=0.25, label="Members' 5–95%"),
        Line2D([], [], ls="none", marker="$✓$", ms=8, color="#0ca30c", label="ERA5 consistent"),
        Line2D([], [], ls="none", marker="$✗$", ms=8, color="#d03b3b", label="ERA5 outside the members"),
        Line2D([], [], ls="none", marker="$–$", ms=8, color=MUTED, label="Too few members (N < 20) to test"),
    ]


# ---------------------------------------------------------------------------
# 2. Record rates
# ---------------------------------------------------------------------------

def plot_record_rates(curves, kind, experiments=("hist-nat", "historical"), ax=None):
    """Cumulative record highs or lows: ERA5 against chance and against each experiment's members.

    With no change, records arrive ever more slowly (the dashed curve,
    1 + 1/2 + ... + 1/n per season). Warming keeps record highs coming and
    starves record lows. ERA5 above hist-nat's band means more records than
    natural variability gives.

    Args:
        curves (xr.Dataset): ``oc.record_curves`` output.
        kind (str): 'highs' or 'lows' (a value of its ``kind`` dim).
        experiments (Sequence[str]): Experiments whose member ranges are drawn.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, panel_width=4.5))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    years = curves["year"].values
    for experiment in experiments:
        members = curves["members"].sel(experiment=experiment, kind=kind).dropna("member", how="all")
        colour = FORCING_COLORS.get(experiment, INK)
        low, mid, high = members.quantile([0.05, 0.5, 0.95], dim="member").transpose("quantile", "year").values
        ax.fill_between(years, low, high, color=colour, alpha=0.18, lw=0, step="post")
        ax.step(years, mid, where="post", color=colour, lw=1.4)
    ax.plot(years, curves["expected"], color=INK_2, lw=1.2, ls=(0, (4, 2)))
    ax.step(years, curves["obs"].sel(kind=kind), where="post", color=ERA5_COLOR, lw=2.2)
    final = float(curves["obs"].sel(kind=kind).isel(year=-1))
    ax.text(0.02, 0.97, f"ERA5: {final:.0f} record {kind} · chance: {float(curves['expected'].isel(year=-1)):.1f}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
    return ax


def record_rate_legend_handles(experiments=("hist-nat", "historical")):
    """The legend entries of ``plot_record_rates``: ERA5, chance, and each experiment's members."""
    handles = [Line2D([], [], color=ERA5_COLOR, lw=2.2, label="ERA5"),
               Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 2)), label="No change (1 + 1/2 + … + 1/n)")]
    return handles + [Patch(color=FORCING_COLORS.get(e, INK), alpha=0.35, label=f"{e} members 5–95% (line: median)")
                      for e in experiments]


# ---------------------------------------------------------------------------
# 3. Scaling factors and attributable trends
# ---------------------------------------------------------------------------

def plot_scaling_factors(region_ds, ax=None):
    """Scaling factors β with their 5-95% intervals in one region, every model (its marker) and signal (its colour).

    β = 1 (the dashed line): the model's response has the observed size.
    β = 0 (the grey line): the signal is not in the observations. An interval
    clear of 0 is a detection; one that includes 1 is consistent. The short
    bar is the perfect-model β, a check for bias.

    Args:
        region_ds (xr.Dataset): ``oc.scaling_factors`` output for one region, on (model, signal).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0)
        ax = axes[0, 0]
    models = list(region_ds["model"].values)
    signals = list(region_ds["signal"].values)
    offsets = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else [0.0]
    ax.axhline(0, color=MUTED, lw=0.9, zorder=0)
    ax.axhline(1, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=0)
    for k, model in enumerate(models):
        for i, signal in enumerate(signals):
            x = i + offsets[k]
            row = region_ds.sel(model=model, signal=signal)
            #(c): The model's shade of its signal's colour
            colour = charts.model_shades(SIGNAL_COLORS.get(str(signal), INK), len(models))[k]
            ax.plot([x, x], [float(row["lower"]), float(row["upper"])], color=colour, lw=1.6, zorder=2)
            ax.plot(x, float(row["beta"]), marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=6.5, color=colour,
                    mec="white", mew=0.8, zorder=3)
            ax.plot(x, float(row["perfect_model_beta"]), marker="_", ms=9, mew=1.2, color=INK_2, zorder=3)
    ax.set_xticks(range(len(signals)), signals)
    ax.grid(axis="x", visible=False)
    return ax


def scaling_factor_legend_handles(models):
    """The legend entries of ``plot_scaling_factors``: each model's marker, the range and the perfect-model β."""
    return charts.model_legend_handles(models) + [
        Line2D([], [], color="0.35", lw=1.6, label="5–95% range"),
        Line2D([], [], ls="none", marker="_", ms=9, mew=1.2, color=INK_2, label="Perfect-model β (bias check)"),
    ]


def plot_attributable_trends(region_ds, ax=None):
    """The observed trend (black) and the part each signal explains (β × its fingerprint's trend), every model, in one
    region.

    Args:
        region_ds (xr.Dataset): ``oc.scaling_factors`` output for one region, on (model, signal).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, the models down its y axis.
    """
    if ax is None:
        fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=1, n_legends=0, left_margin=1.5))
        axes, _ = charts.chart_axes(fig, n_rows=1, n_cols=1, n_legends=0, left_margin=1.5)
        ax = axes[0, 0]
    models = list(region_ds["model"].values)
    signals = list(region_ds["signal"].values)
    height = 0.8 / len(signals)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=0)
    for k, model in enumerate(models):
        for i, signal in enumerate(signals):
            y = k + (i - (len(signals) - 1) / 2) * height
            row = region_ds.sel(model=model, signal=signal)
            ax.barh(y, float(row["attributable_trend"]), height=height * 0.85, color=SIGNAL_COLORS.get(str(signal), INK),
                    alpha=0.85, lw=0)
            ax.plot([float(row["attributable_lower"]), float(row["attributable_upper"])], [y, y], color=INK, lw=0.9,
                    zorder=3)
        observed = float(region_ds.sel(model=model)["observed_trend"])
        ax.plot([observed, observed], [k - 0.45, k + 0.45], color=ERA5_COLOR, lw=2.2, zorder=4)
    ax.set_yticks(range(len(models)), models)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    return ax


def attributable_trend_legend_handles(signals):
    """The legend entries of ``plot_attributable_trends``: ERA5's trend, and each signal's attributable trend."""
    return [Line2D([], [], color=ERA5_COLOR, lw=2.2, label="ERA5 observed trend")] + [
        Patch(color=SIGNAL_COLORS.get(str(s), INK), label=f"{s}: attributable (5–95%)") for s in signals]
