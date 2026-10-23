"""Figures for the observed-change analysis; the calculations are in observed_change.

Same conventions and style as plots.era5_evaluation. Colour follows the
experiment (FORCING_COLORS) and ERA5 is always black.

Sections
--------
1. Consistency: ERA5's trend among each experiment's member trends
2. Record rates: cumulative record highs and lows against chance and the members
3. Scaling factors and attributable trends
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import era5_evaluation as ev
from .era5_evaluation import (ERA5_COLOR, EVAL_RC, INK, INK_2, MODEL_MARKERS, MUTED, _legend, _suptitle,
                                   is_testable, order_seasons, where_era5)
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER
from plotting_modules.utils import plot

#(t): Signal colours follow the experiment that defines them
SIGNAL_COLORS = {"ANT": FORCING_COLORS["historical"], "NAT": FORCING_COLORS["hist-nat"],
                 "GHG": FORCING_COLORS["hist-GHG"], "AER": FORCING_COLORS["hist-aer"],
                 "O3": FORCING_COLORS["hist-totalO3"]}

#(t): Detection classes (observed_change.DETECTION_CLASSES), validated as a categorical palette
DETECTION_COLORS = ("#d9d8d4", "#2a78d6", "#eb6834", "#4a3aa7")


def _ordered(names):
    return [e for e in FORCING_REVEAL_ORDER if e in names] + [e for e in names if e not in FORCING_REVEAL_ORDER]


# ---------------------------------------------------------------------------
# 1. Consistency: ERA5's trend among each experiment's member trends
# ---------------------------------------------------------------------------

@plot("axes")
def draw_experiment_strips(ax, results, statistic="trend"):
    """Every experiment's member values of one statistic as a row of dots, with ERA5 as a vertical line.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        results (dict[str, xr.Dataset]): Experiment -> ``ev.locate`` results on a ``statistic`` dim, one season.
        statistic (str): The statistic shown.
    """
    experiments = _ordered(list(results))
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
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


@plot("figure")
def experiment_strips(results, statistic="trend", title=None):
    """ERA5's trend against every experiment's member trends, one panel per season.

    The experiments whose rows ERA5 (the black line) passes through could have
    produced the observed trend; ERA5 outside hist-nat's row means the
    observed change is unlikely to be natural.

    Args:
        results (xr.DataTree | xr.Dataset | dict[str, xr.Dataset]): One model's ``ev.locate`` results on
            (statistic, season): its subtree /<experiment>, a Dataset on an ``experiment`` dim (experiments
            without data are left out), or a dict by experiment.
        statistic (str): The statistic shown.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    if isinstance(results, xr.DataTree):
        results = {node.name: node.to_dataset() for node in results.leaves if node.data_vars}
    if isinstance(results, xr.Dataset):
        results = {str(experiment): results.sel(experiment=experiment) for experiment in results["experiment"].values
                   if bool(results["era5"].sel(experiment=experiment).notnull().any())}
    first = order_seasons(next(iter(results.values())))
    seasons = list(first["season"].values)
    diagnostic = ev.DIAGNOSTICS[statistic]
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(seasons), figsize=(3.6 * len(seasons) + 1, 3.3), layout="constrained",
                                 sharey=True, squeeze=False)
        for ax, season in zip(axes[0], seasons):
            draw_experiment_strips(ax, {e: r.sel(season=season) for e, r in results.items()}, statistic)
            ax.set_title(season, loc="left")
            ax.set_xlabel(f"{diagnostic.label} ({diagnostic.units})" if diagnostic.units else diagnostic.label)
        _legend(fig, [
            Line2D([], [], color=ERA5_COLOR, lw=2, label="ERA5"),
            Line2D([], [], ls="none", marker="o", ms=5, color="0.5", label="Each member"),
            Patch(color="0.5", alpha=0.25, label="Members' 5–95%"),
            Line2D([], [], ls="none", marker="$✓$", ms=8, color="#0ca30c", label="ERA5 consistent"),
            Line2D([], [], ls="none", marker="$✗$", ms=8, color="#d03b3b", label="ERA5 outside the members"),
            Line2D([], [], ls="none", marker="$–$", ms=8, color=MUTED, label="Too few members (N < 20) to test"),
        ])
        _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 2. Record rates
# ---------------------------------------------------------------------------

@plot("figure")
def record_rates(curves, experiments=("hist-nat", "historical"), title=None):
    """Cumulative record highs and lows: ERA5 against chance and against each experiment's members.

    With no change, records arrive ever more slowly (the dashed curve,
    1 + 1/2 + ... + 1/n per season). Warming keeps record highs coming and
    starves record lows. ERA5 above hist-nat's band means more records than
    natural variability gives.

    Args:
        curves (xr.Dataset): ``oc.record_curves`` output.
        experiments (Sequence[str]): Experiments whose member ranges are drawn.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    kinds = list(curves["kind"].values)
    years = curves["year"].values
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 3.8), layout="constrained", squeeze=False)
        for ax, kind in zip(axes[0], kinds):
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
            ax.set_title(f"Record {kind}", loc="left")
            ax.set_xlabel("Year")
            ax.set_ylabel("Cumulative number of records")
            core.style_ax(ax)
        handles = [Line2D([], [], color=ERA5_COLOR, lw=2.2, label="ERA5"),
                   Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 2)), label="No change (1 + 1/2 + … + 1/n)")]
        handles += [Patch(color=FORCING_COLORS.get(e, INK), alpha=0.35, label=f"{e} members 5–95% (line: median)")
                    for e in experiments]
        _legend(fig, handles)
        _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 3. Scaling factors and attributable trends
# ---------------------------------------------------------------------------

def _by_region_and_model(results):
    """Region -> model -> one result, from a Dataset on (region, model, signal) or such a dict already."""
    if isinstance(results, xr.Dataset):
        return {str(region): {str(model): results.sel(region=region, model=model) for model in results["model"].values}
                for region in results["region"].values}
    return results


@plot("figure")
def scaling_factor_plot(results, title=None):
    """Scaling factors β with their 5-95% intervals, one panel per region, every model and signal.

    β = 1 (the upper line): the model's response has the observed size.
    β = 0 (the lower line): the signal is not in the observations. An interval
    clear of 0 is a detection; one that includes 1 is consistent.

    Args:
        results (xr.Dataset | dict[str, dict[str, xr.Dataset]]): ``oc.scaling_factors`` output for every
            region and model, on (region, model, signal), or as a dict region -> model -> output.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    results = _by_region_and_model(results)
    regions = list(results)
    models = list(next(iter(results.values())))
    signals = list(next(iter(next(iter(results.values())).values()))["signal"].values)
    offsets = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else [0.0]
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(regions), figsize=(3.2 + 1.1 * len(signals) * len(regions), 3.9),
                                 layout="constrained", sharey=True, squeeze=False)
        for ax, region in zip(axes[0], regions):
            ax.axhline(0, color=MUTED, lw=0.9, zorder=0)
            ax.axhline(1, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=0)
            for k, model in enumerate(models):
                result = results[region][model]
                for i, signal in enumerate(signals):
                    x = i + offsets[k]
                    row = result.sel(signal=signal)
                    colour = SIGNAL_COLORS.get(signal, INK)
                    ax.plot([x, x], [float(row["lower"]), float(row["upper"])], color=colour, lw=1.6, zorder=2)
                    ax.plot(x, float(row["beta"]), marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=6.5, color=colour,
                            mec="white", mew=0.8, zorder=3)
                    ax.plot(x, float(row["perfect_model_beta"]), marker="_", ms=9, mew=1.2, color=INK_2, zorder=3)
            ax.set_xticks(range(len(signals)), signals)
            ax.set_title(region, loc="left")
            core.style_ax(ax)
        axes[0, 0].set_ylabel("Scaling factor β")
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=7, color="0.35",
                          label=model) for k, model in enumerate(models)]
        handles += [Line2D([], [], color="0.35", lw=1.6, label="5–95% range"),
                    Line2D([], [], ls="none", marker="_", ms=9, mew=1.2, color=INK_2, label="Perfect-model β (bias check)")]
        _legend(fig, handles, ncols=min(len(handles), 5))
        _suptitle(fig, title or "Scaling factors: how much of each model's forced response is in ERA5")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def attributable_trend_plot(results, title=None, units="°C/decade"):
    """The observed trend and the part each signal explains (β x fingerprint trend), per model and region.

    Args:
        results (xr.Dataset | dict[str, dict[str, xr.Dataset]]): ``oc.scaling_factors`` output for every
            region and model, on (region, model, signal), or as a dict region -> model -> output.
        title (str | None): Figure title.
        units (str): Trend units.

    Returns:
        core.Panels
    """
    results = _by_region_and_model(results)
    regions = list(results)
    models = list(next(iter(results.values())))
    signals = list(next(iter(next(iter(results.values())).values()))["signal"].values)
    height = 0.8 / len(signals)
    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(regions), figsize=(4.2 * len(regions), 0.55 * len(models) * len(signals) / 2 + 2),
                                 layout="constrained", sharey=True, squeeze=False)
        for ax, region in zip(axes[0], regions):
            ax.axvline(0, color=MUTED, lw=0.9, zorder=0)
            for k, model in enumerate(models):
                result = results[region][model]
                for i, signal in enumerate(signals):
                    y = k + (i - (len(signals) - 1) / 2) * height
                    row = result.sel(signal=signal)
                    value = float(row["attributable_trend"])
                    ax.barh(y, value, height=height * 0.85, color=SIGNAL_COLORS.get(signal, INK), alpha=0.85, lw=0)
                    ax.plot([float(row["attributable_lower"]), float(row["attributable_upper"])], [y, y],
                            color=INK, lw=0.9, zorder=3)
                observed = float(result["observed_trend"])
                ax.plot([observed, observed], [k - 0.45, k + 0.45], color=ERA5_COLOR, lw=2.2, zorder=4)
            ax.set_yticks(range(len(models)), models)
            ax.set_ylim(len(models) - 0.5, -0.5)
            ax.set_xlabel(f"Trend 1979–2014 ({units})")
            ax.set_title(region, loc="left")
            core.style_ax(ax)
        handles = [Line2D([], [], color=ERA5_COLOR, lw=2.2, label="ERA5 observed trend")]
        handles += [Patch(color=SIGNAL_COLORS.get(s, INK), label=f"{s}: attributable (5–95%)") for s in signals]
        _legend(fig, handles, ncols=min(len(handles), 5))
        _suptitle(fig, title or "How much of the observed trend each signal explains")
    return core.Panels(fig=fig, axes=axes)
