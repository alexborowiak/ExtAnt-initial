"""Small figures that show how each step of the analysis works, on real data at one grid point.

Two kinds:

- schematics of the significance tests, step by step: ``bootstrap_schematic``
  (the hist-nat bootstrap of the change in width) and ``permutation_schematic``
  (the member-block permutation test of the mean change). Both are given the
  test's own pieces at one point, as notebook 03 makes them, and rebuild their
  example trials from its draws, so every number on them is the test's own;
- quick looks that follow the data through the notebooks: monthly to seasonal
  means, the year x season split, a rolling quantile, the smoothing, the
  change against hist-nat, the forced response, and signal and noise.

Every function is a figure function (``@plot("figure")``) returning ``core.Panels``,
which unpacks as ``fig, axes``. The two schematics build their own GridSpec, as
``plots.era5_evaluation.rank_anatomy`` does; the rest use ``core.panel_grid``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_LEGEND_LABELS, SEASON_COLORS, SEASONS
from plotting_modules.utils import plot

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
# Schematics of the tests
# ---------------------------------------------------------------------------

def _schematic_axes(figsize):
    """a, flow, b (3 x 3 small panels with a '...' row before the last three), flow, c."""
    fig = plt.figure(figsize=figsize)
    outer = GridSpec(5, 1, figure=fig, height_ratios=[2.3, 0.65, 4.1, 0.65, 2.3], hspace=0.55)
    ax_a = fig.add_subplot(outer[0])
    ax_flow1 = fig.add_subplot(outer[1])
    ax_flow2 = fig.add_subplot(outer[3])
    ax_c = fig.add_subplot(outer[4])

    grid_b = GridSpecFromSubplotSpec(4, 3, subplot_spec=outer[2], height_ratios=[1, 1, 0.28, 1],
                                     hspace=0.55, wspace=0.18)
    mini = [fig.add_subplot(grid_b[row, col]) for row in (0, 1, 3) for col in range(3)]
    ax_dots = fig.add_subplot(grid_b[2, :])
    ax_dots.axis("off")
    ax_dots.text(0.5, 0.5, "⋯", ha="center", va="center", fontsize=24)
    return fig, ax_a, ax_flow1, mini, ax_flow2, ax_c


def _flow(ax, text):
    """A downward arrow with the step it stands for."""
    ax.axis("off")
    ax.annotate("", xy=(0.5, 0.08), xytext=(0.5, 0.38), arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(0.5, 0.72, text, ha="center", va="center", multialignment="center", fontsize=11)


def _example_trials(n_trials):
    """Positions of the trials shown in panel b: the first six and the last three."""
    return [0, 1, 2, 3, 4, 5, n_trials - 3, n_trials - 2, n_trials - 1]


def _range_bar(ax, low, high, height, color, lw=3):
    ax.plot([low, high], [height, height], color=color, lw=lw, solid_capstyle="butt")


@plot("figure")
def bootstrap_schematic(hist_nat, final, null, selected, window_starts, qrange_change, pvalue, experiment,
                        quantiles=(0.05, 0.95), units="°C", bins=40, figsize=(11, 15)):
    """The hist-nat bootstrap of the change in width (Q95 - Q05), step by step.

    a) the change being tested: the experiment's final years against hist-nat's
       whole record, each with its Q05-Q95 range;
    b) example trials: each pools N different hist-nat members over one random
       window, and its change in width is what natural variability and N
       members alone can give;
    c) every trial's change, against which the observed change is judged.

    Each trial in b is rebuilt from ``selected`` and ``window_starts`` and
    checked against ``null``, so the draws must be the ones behind ``null``.

    Args:
        hist_nat (xr.DataArray): hist-nat's whole record at one point, on (member, year).
        final (xr.DataArray): The experiment's final years at the same point, on (member, year).
        null (xr.DataArray): Each trial's change in width, on (trial,).
        selected (np.ndarray): (trial, member) True for each trial's hist-nat members.
        window_starts (np.ndarray): (trial,) the index along ``year`` where each trial's window starts.
        qrange_change (float): The observed change in width.
        pvalue (float): Its p-value.
        experiment (str): Name of the experiment tested.
        quantiles (tuple[float, float]): The width's lower and upper quantile.
        units (str): Units of the variable.
        bins (int): Bins of the panel c histogram.
        figsize (tuple): Figure size in inches.

    Returns:
        core.Panels: ``axes`` holds a, the nine trial panels, then c.
    """
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    hist_nat_values = hist_nat.transpose("member", "year").values
    year_values = hist_nat["year"].values
    full = _clean(hist_nat_values)
    final_values = _clean(final)
    null_values = np.asarray(null, dtype=float)
    n_members, years, n_trials = int(selected[0].sum()), final.sizes["year"], null_values.size
    change, pvalue = float(qrange_change), float(pvalue)

    fig, ax_a, ax_flow1, mini, ax_flow2, ax_c = _schematic_axes(figsize)

    #(t): a) the experiment's final years against the whole hist-nat record, both centred on 0 to compare widths
    full_c, final_c = full - full.mean(), final_values - final_values.mean()
    edges = np.histogram_bin_edges(np.concatenate([full_c, final_c]), bins=30)
    ax_a.hist(full_c, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=1.4,
              label=f"hist-nat, whole record ({hist_nat_values.shape[0]} members × {hist_nat_values.shape[1]} years)")
    ax_a.hist(final_c, bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=1.4,
              label=f"{experiment}, final {years} years ({final.sizes['member']} members)")
    top = ax_a.get_ylim()[1]
    _range_bar(ax_a, *_qrange(full_c, quantiles)[1:], 0.07 * top, nat_color)
    _range_bar(ax_a, *_qrange(final_c, quantiles)[1:], 0.13 * top, exp_color)
    ax_a.set_title("a  The change in width to test", fontsize=15, pad=10)
    ax_a.set_xlabel(f"Value minus its own mean ({units})", fontsize=12)
    ax_a.set_ylabel("Density", fontsize=12)
    ax_a.legend(frameon=False, fontsize=10)
    ax_a.text(0.02, 0.95, rf"$\Delta_{{obs}}$ = {change:+.2f} {units}" "\n" "(bars: Q05 to Q95)",
              transform=ax_a.transAxes, va="top", fontsize=11)

    _flow(ax_flow1, f"Draw {n_members} different hist-nat members and one random {years}-year window,\n"
                    f"pool their {n_members * years} values (green) and compare their width with the whole record's (grey)")

    #(t): b) example trials, with exactly the draws behind panel c
    samples = []
    for position in _example_trials(n_trials):
        start = int(window_starts[position])
        pooled = _clean(hist_nat_values[selected[position], start:start + years])
        samples.append((position, start, pooled - pooled.mean()))
    edges = np.histogram_bin_edges(np.concatenate([full_c, *[s for *_, s in samples]]), bins=16)
    top = max(np.histogram(s, edges, density=True)[0].max() for *_, s in samples)
    full_range = _qrange(full_c, quantiles)

    for ax, (position, start, values) in zip(mini, samples):
        ax.hist(full_c, bins=edges, density=True, histtype="step", color=REFERENCE_GREY, lw=1.0)
        ax.hist(values, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=0.8)
        width, low, high = _qrange(values, quantiles)
        _range_bar(ax, *full_range[1:], 0.07 * top, REFERENCE_GREY, lw=2)
        _range_bar(ax, low, high, 0.15 * top, nat_color, lw=2)
        delta = width - full_range[0]
        #(c): The same number as the test's own trial (float32 there)
        assert np.isclose(delta, null_values[position], atol=1e-3), "trial does not match the test"
        ax.text(0.04, 0.94, f"{position + 1:,}", transform=ax.transAxes, va="top", fontsize=8)
        ax.text(0.04, 0.80, f"{year_values[start]}–{year_values[start + years - 1]}", transform=ax.transAxes,
                va="top", fontsize=7, color="0.35")
        ax.text(0.96, 0.94, rf"$\Delta$={delta:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
    mini[0].set_title("b  Example trials", fontsize=15, loc="left", pad=10)

    _flow(ax_flow2, f"Repeat {n_trials:,} times, one $\\Delta$ per trial: the changes natural variability\n"
                    r"and sampling alone produce. Then where does the observed $\Delta$ fall among them?")

    #(t): c) every trial's change, and the observed one
    null_values = _clean(null_values)
    low, high = np.quantile(null_values, [0.025, 0.975])
    ax_c.axvspan(low, high, color="0.9", zorder=0, label="central 95% of trials")
    ax_c.hist(null_values, bins=bins, density=True, color=nat_color, edgecolor=nat_color, alpha=0.65, lw=0.8,
              label="hist-nat trials")
    ax_c.axvline(change, color=exp_color, lw=2.5, ls="--", label=f"{experiment} (observed)")
    ax_c.set_title(f"c  Bootstrap distribution\n$p$ = {pvalue:.3f}", fontsize=15, pad=8)
    ax_c.set_xlabel(f"Change in Q95 − Q05 ({units})", fontsize=12)
    ax_c.set_ylabel("Density", fontsize=12)
    ax_c.legend(frameon=False, fontsize=10)

    for ax in (ax_a, ax_c):
        ax.tick_params(labelsize=11)
    return core.Panels(fig=fig, axes=np.array([ax_a, *mini, ax_c], dtype=object))


@plot("figure")
def permutation_schematic(experiment_final, hist_nat_final, permutations, is_experiment, mean_change, pvalue,
                          experiment, units="°C", bins=40, figsize=(11, 15)):
    """The member-block permutation test of the mean change, step by step.

    a) the two ensembles' final years and the difference in their means;
    b) example relabellings: whole members shuffled between the two labels,
       keeping each member's years together;
    c) every relabelling's difference in means, against which the observed
       difference is judged.

    Each relabelling in b is rebuilt from ``is_experiment`` and checked against
    ``permutations``, so the two must come from the same draws.

    Args:
        experiment_final, hist_nat_final (xr.DataArray): The final years of each at one point, on (member, year).
        permutations (xr.DataArray): Each relabelling's difference in means, on (trial,).
        is_experiment (np.ndarray): (trial, pooled member) True for the members each relabelling calls
            the experiment; the experiment's own members come first in the pool.
        mean_change (float): The observed difference in means.
        pvalue (float): Its p-value.
        experiment (str): Name of the experiment.
        units (str): Units of the variable.
        bins (int): Bins of the panel c histogram.
        figsize (tuple): Figure size in inches.

    Returns:
        core.Panels: ``axes`` holds a, the nine relabelling panels, then c.
    """
    exp_color, nat_color = _color(experiment), _color("hist-nat")
    exp_values = experiment_final.transpose("member", "year").values
    nat_values = hist_nat_final.transpose("member", "year").values
    n_exp, n_nat = exp_values.shape[0], nat_values.shape[0]
    years = exp_values.shape[1]
    permutation_values = np.asarray(permutations, dtype=float)
    n_trials = permutation_values.size
    change, pvalue = float(mean_change), float(pvalue)

    fig, ax_a, ax_flow1, mini, ax_flow2, ax_c = _schematic_axes(figsize)

    #(t): a) the two ensembles' final years
    exp_all, nat_all = _clean(exp_values), _clean(nat_values)
    edges = np.histogram_bin_edges(np.concatenate([exp_all, nat_all]), bins=30)
    ax_a.hist(nat_all, bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=1.4,
              label=f"{_label('hist-nat')}, {n_nat} members")
    ax_a.hist(exp_all, bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=1.4,
              label=f"{_label(experiment)}, {n_exp} members")
    ax_a.axvline(nat_all.mean(), color=nat_color, lw=2)
    ax_a.axvline(exp_all.mean(), color=exp_color, lw=2)
    ax_a.set_title(f"a  The final {years} years of each ensemble", fontsize=15, pad=10)
    ax_a.set_xlabel(units, fontsize=12)
    ax_a.set_ylabel("Density", fontsize=12)
    ax_a.legend(frameon=False, fontsize=10)
    ax_a.text(0.02, 0.95, rf"$\Delta_{{obs}}$ = {change:+.2f} {units}" "\n" "(lines: the means)",
              transform=ax_a.transAxes, va="top", fontsize=11)

    _flow(ax_flow1, f"Pool all {n_exp + n_nat} members, each keeping its {years} years together,\n"
                    f"then randomly relabel {n_exp} of them {experiment} and {n_nat} hist-nat")

    #(t): b) example relabellings, the experiment's own members first in the pool
    pool = np.concatenate([exp_values, nat_values]).astype(float)
    samples = [(position, pool[is_experiment[position]], pool[~is_experiment[position]])
               for position in _example_trials(n_trials)]
    edges = np.histogram_bin_edges(pool[np.isfinite(pool)], bins=16)
    top = max(max(np.histogram(_clean(e), edges, density=True)[0].max(),
                  np.histogram(_clean(r), edges, density=True)[0].max()) for _, e, r in samples)
    for ax, (position, exp_i, nat_i) in zip(mini, samples):
        ax.hist(_clean(nat_i), bins=edges, density=True, color=nat_color, edgecolor=nat_color, alpha=0.28, lw=0.8)
        ax.hist(_clean(exp_i), bins=edges, density=True, color=exp_color, edgecolor=exp_color, alpha=0.28, lw=0.8)
        ax.axvline(np.nanmean(nat_i), color=nat_color, lw=1.5)
        ax.axvline(np.nanmean(exp_i), color=exp_color, lw=1.5)
        delta = np.nanmean(exp_i.mean(1)) - np.nanmean(nat_i.mean(1))
        assert np.isclose(delta, permutation_values[position], atol=1e-4), "relabelling does not match the test"
        ax.text(0.04, 0.94, f"{position + 1:,}", transform=ax.transAxes, va="top", fontsize=8)
        ax.text(0.96, 0.94, rf"$\Delta$={delta:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
    mini[0].set_title("b  Example relabellings", fontsize=15, loc="left", pad=10)

    _flow(ax_flow2, f"Repeat {n_trials:,} times and calculate $\\Delta$ for each relabelling,\n"
                    r"then compare the observed $\Delta$ with the permutation distribution")

    #(t): c) every relabelling's difference, and the observed one
    permutation_values = _clean(permutation_values)
    ax_c.hist(permutation_values, bins=bins, density=True, color=nat_color, edgecolor=nat_color, alpha=0.65, lw=0.8,
              label="Relabellings")
    ax_c.axvline(change, color=exp_color, lw=2.5, ls="--", label="Observed")
    ax_c.set_title(f"c  Permutation distribution\n$p$ = {pvalue:.3f}", fontsize=15, pad=8)
    ax_c.set_xlabel(f"Difference in means ({units})", fontsize=12)
    ax_c.set_ylabel("Density", fontsize=12)
    ax_c.legend(frameon=False, fontsize=10)

    for ax in (ax_a, ax_c):
        ax.tick_params(labelsize=11)
    return core.Panels(fig=fig, axes=np.array([ax_a, *mini, ax_c], dtype=object))


# ---------------------------------------------------------------------------
# Following the data
# ---------------------------------------------------------------------------

def _decimal_year(time):
    """Mid-month positions as decimal years, for any calendar."""
    return time.dt.year.values + (time.dt.month.values - 0.5) / 12


#(t): Where each season's months sit within its (year, season) label, as decimal-year offsets
_SEASON_SPAN = {"DJF": (11, 14), "MAM": (2, 5), "JJA": (5, 8), "SON": (8, 11)}


@plot("figure")
def seasonal_mean_demo(monthly, seasonal, years, units="°C"):
    """Monthly values and the seasonal means made from them, for one member at one point.

    Args:
        monthly (xr.DataArray): One series on ``time``, in analysis units.
        seasonal (xr.DataArray): The same member's seasonal means on (year, season).
        years (slice): Years shown, e.g. ``slice(1990, 1994)``.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 1, panel_w=8.8, panel_h=2.6, left=0.8, bottom=0.65, top=0.6)
    ax = panels.axes[0, 0]
    months = monthly.sel(time=slice(str(years.start), str(years.stop)))
    ax.plot(_decimal_year(months.time), months.values, color="0.6", lw=0.8, marker="o", ms=3, zorder=2,
            label="Monthly means")
    for year in range(years.start, years.stop):
        for season in SEASONS:
            first, last = _SEASON_SPAN[season]
            value = float(seasonal.sel(year=year, season=season))
            ax.plot([year + first / 12, year + last / 12], [value, value], color=SEASON_COLORS[season], lw=3,
                    solid_capstyle="butt", zorder=3)
    ax.legend(handles=[Line2D([], [], color="0.6", marker="o", ms=3, lw=0.8, label="Monthly means"),
                       *[Line2D([], [], color=SEASON_COLORS[s], lw=3, label=f"{s} mean") for s in SEASONS]],
              ncols=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9)
    ax.set_xlabel("Year (DJF is labelled by its December)")
    ax.set_ylabel(units)
    core.style_ax(ax)
    return panels


@plot("figure")
def year_season_demo(seasonal, units="°C"):
    """The same seasonal means as one long series, and split into one series per season.

    Args:
        seasonal (xr.DataArray): One member at one point on (year, season).
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 2, panel_w=4.6, panel_h=2.4, wspace=0.2, right=0.8, left=0.8, bottom=0.65, top=0.45, sharey=True)
    ax_time, ax_split = panels.axes[0]
    ax_split.tick_params(labelleft=False)
    for season in SEASONS:
        first, last = _SEASON_SPAN[season]
        x = seasonal["year"].values + (first + last) / 24
        values = seasonal.sel(season=season).values
        ax_time.scatter(x, values, s=5, color=SEASON_COLORS[season])
        ax_split.plot(seasonal["year"].values, values, color=SEASON_COLORS[season], lw=1, label=season)
    ax_time.set_title("Seasonal means in time order (time)", loc="left")
    ax_split.set_title("After split_time: one series per season (year × season)", loc="left")
    ax_split.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax_time.set_ylabel(units)
    for ax in (ax_time, ax_split):
        ax.set_xlabel("Year")
        core.style_ax(ax)
    return panels


@plot("figure")
def rolling_quantile_demo(members, rolling, year, window, quantiles=(0.05, 0.5, 0.95), color="C0", units="°C"):
    """How one point of a rolling quantile is made: every member's years in one window, pooled.

    a) the members, with the window around ``year`` shaded; b) the pooled
    values in that window and their quantiles; c) the rolling quantiles
    through time, with ``year`` marked.

    Args:
        members (xr.DataArray): One point and season on (member, year).
        rolling (xr.DataArray): ``quantiles.centred_rolling_quantiles`` of ``members``, on (quantile, year).
        year (int): Window centre shown.
        window (int): Window length in years.
        quantiles (Sequence[float]): Quantiles shown (must be in ``rolling``).
        color: Colour of the members.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 3, col_widths=(4.2, 1.5, 4.2), panel_h=2.8, wspace=0.35, left=0.8, bottom=0.65, top=0.45, sharey=True)
    ax_members, ax_hist, ax_rolling = panels.axes[0]
    half = window // 2
    shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(quantiles)))

    years = members["year"].values
    ax_members.plot(years, members.transpose("year", "member").values, color=color, lw=0.4, alpha=0.4)
    ax_members.axvspan(year - half - 0.5, year + half + 0.5, color="0.85", zorder=0)
    ax_members.set_title(f"a) Every member; the {window} years around {year}", loc="left")
    ax_members.set_xlabel("Year")
    ax_members.set_ylabel(units)

    pooled = _clean(members.sel(year=slice(year - half, year + half)))
    ax_hist.hist(pooled, bins=20, orientation="horizontal", color=color, alpha=0.5)
    for q, shade in zip(quantiles, shades):
        ax_hist.axhline(np.quantile(pooled, q), color=shade, lw=2)
    ax_hist.set_title(f"b) {pooled.size} values", loc="left")
    ax_hist.set_xlabel("Count")
    ax_hist.tick_params(labelleft=False)

    for q, shade in zip(quantiles, shades):
        series = rolling.sel(quantile=q, method="nearest")
        ax_rolling.plot(series["year"], series, color=shade, lw=1.5, label=f"Q{q * 100:02.0f}")
        ax_rolling.plot(year, float(series.sel(year=year)), "o", color=shade, ms=6, mec="white")
    ax_rolling.axvline(year, color="0.6", lw=0.8, ls="--")
    ax_rolling.set_title("c) Rolling quantiles: one window per year", loc="left")
    ax_rolling.set_xlabel("Year (window centre)")
    ax_rolling.legend(frameon=False, fontsize=8, ncols=len(quantiles))
    ax_rolling.tick_params(labelleft=False)
    for ax in (ax_members, ax_hist, ax_rolling):
        core.style_ax(ax)
    return panels


@plot("figure")
def smoothing_demo(rolling, smooth, units="°C"):
    """Rolling quantiles before (thin) and after (thick) LOWESS smoothing, at one point.

    Args:
        rolling, smooth (xr.DataArray): On (quantile, year).
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 1, panel_w=8.0, panel_h=2.8, left=0.8, bottom=0.65, top=0.45)
    ax = panels.axes[0, 0]
    shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, rolling.sizes["quantile"]))
    for q, shade in zip(rolling["quantile"].values, shades):
        ax.plot(rolling["year"], rolling.sel(quantile=q), color=shade, lw=0.8, alpha=0.5)
        ax.plot(smooth["year"], smooth.sel(quantile=q), color=shade, lw=2.2, label=f"Q{q * 100:g}")
    ax.legend(frameon=False, fontsize=8, ncols=rolling.sizes["quantile"], loc="upper left")
    ax.set_title("Rolling quantiles (thin) and LOWESS-smoothed (thick)", loc="left")
    ax.set_xlabel("Year (window centre)")
    ax.set_ylabel(units)
    core.style_ax(ax)
    return panels


@plot("figure")
def quantile_change_demo(smooth, base, experiment, quantiles=(0.05, 0.95), units="°C"):
    """The change in each quantile: an experiment's smoothed quantiles minus hist-nat's whole-record ones.

    a) the smoothed quantiles of the experiment and of hist-nat, with hist-nat's
    whole-record quantiles as dashed lines; b) the change in each, and so the
    change in width, ΔQ_high − ΔQ_low.

    Args:
        smooth (xr.DataArray): Smoothed rolling quantiles on (experiment, quantile, year), at one point.
        base (xr.DataArray): hist-nat's whole-record quantiles on (quantile,), at the same point.
        experiment (str): The experiment.
        quantiles (tuple[float, float]): The pair whose width is shown.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 2, panel_w=5.0, panel_h=3.0, wspace=0.75, left=0.8, bottom=0.65, top=0.45)
    ax_q, ax_change = panels.axes[0]
    exp_color, ref_color = _color(experiment), _color("hist-nat")
    styles = dict(zip(quantiles, ("-", "--")))
    for q in quantiles:
        for name, color in (("hist-nat", ref_color), (experiment, exp_color)):
            series = smooth.sel(experiment=name).sel(quantile=q, method="nearest")
            ax_q.plot(series["year"], series, color=color, lw=1.8, ls=styles[q])
        ax_q.axhline(float(base.sel(quantile=q, method="nearest")), color="0.4", lw=1, ls=":")
    ax_q.legend(handles=[Line2D([], [], color=exp_color, lw=2, label=experiment),
                         Line2D([], [], color=ref_color, lw=2, label="hist-nat"),
                         Line2D([], [], color="0.4", lw=1, ls=":", label="hist-nat, whole record"),
                         *[Line2D([], [], color="0.2", ls=styles[q], label=f"Q{q * 100:02.0f}") for q in quantiles]],
                frameon=False, fontsize=8, ncols=2)
    ax_q.set_title("a) Smoothed quantiles", loc="left")
    ax_q.set_ylabel(units)

    change = smooth.sel(experiment=experiment) - base
    low, high = (change.sel(quantile=q, method="nearest") for q in quantiles)
    ax_change.plot(low["year"], low, color=exp_color, lw=1.8, ls=styles[quantiles[0]],
                   label=f"ΔQ{quantiles[0] * 100:02.0f}")
    ax_change.plot(high["year"], high, color=exp_color, lw=1.8, ls=styles[quantiles[1]],
                   label=f"ΔQ{quantiles[1] * 100:02.0f}")
    ax_change.fill_between(high["year"], low, high, color=exp_color, alpha=0.12, lw=0,
                           label="Δ(width) = gap between the lines")
    ax_change.axhline(0, color="0.5", lw=0.8)
    ax_change.set_title(f"b) {experiment} minus hist-nat's whole record", loc="left")
    ax_change.legend(frameon=False, fontsize=8)
    for ax in (ax_q, ax_change):
        ax.set_xlabel("Year (window centre)")
        core.style_ax(ax)
    return panels


@plot("figure")
def forced_response_demo(members, forced, quantiles=(0.05, 0.95), color="C0", units="°C"):
    """Members, their forced response, and what is left when it is removed: the internal variability.

    Args:
        members (xr.DataArray): One point and season on (member, year).
        forced (xr.DataArray): ``response_change.forced_response(members)``, on (year,).
        quantiles (tuple[float, float]): Range drawn on the internal variability.
        color: Colour of the members.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 2, panel_w=5.0, panel_h=2.8, wspace=0.75, left=0.8, bottom=0.65, top=0.45)
    ax_members, ax_internal = panels.axes[0]
    years = members["year"].values
    ax_members.plot(years, members.transpose("year", "member").values, color=color, lw=0.4, alpha=0.4)
    ax_members.plot(forced["year"], forced, color="k", lw=2.2)
    ax_members.set_title("a) Members and their forced response (black)", loc="left")
    ax_members.set_ylabel(units)

    internal = (members - forced).transpose("year", "member")
    ax_internal.plot(years, internal.values, color=color, lw=0.4, alpha=0.4)
    for k, q in enumerate(quantiles):
        ax_internal.axhline(float(internal.quantile(q)), color="k", lw=1.2, ls="--",
                            label=f"Q{quantiles[0] * 100:02.0f} and Q{quantiles[1] * 100:02.0f}" if k == 0 else None)
    ax_internal.axhline(0, color="0.5", lw=0.8)
    ax_internal.set_title("b) Members minus the forced response", loc="left")
    ax_internal.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (ax_members, ax_internal):
        ax.set_xlabel("Year")
        core.style_ax(ax)
    return panels


@plot("figure")
def two_sample_demo(experiment_values, hist_nat_values, experiment, t=None, p=None,
                    units="°C", title=None):
    """The two samples a test compares (e.g. the final years of each ensemble, members and years pooled).

    Args:
        experiment_values, hist_nat_values (xr.DataArray): The two samples, any shape.
        experiment (str): The experiment's name.
        t, p (float | None): Test statistic and p-value to print.
        units (str): Units of the variable.
        title (str | None): Panel title.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 1, panel_w=6.0, panel_h=2.8, left=0.8, bottom=0.65, top=0.45)
    ax = panels.axes[0, 0]
    exp, ref = _clean(experiment_values), _clean(hist_nat_values)
    edges = np.histogram_bin_edges(np.concatenate([exp, ref]), bins=30)
    for values, name in ((ref, "hist-nat"), (exp, experiment)):
        ax.hist(values, bins=edges, density=True, color=_color(name), alpha=0.3, label=f"{name} (n = {values.size})")
        ax.axvline(values.mean(), color=_color(name), lw=2)
    if t is not None:
        ax.text(0.02, 0.95, f"t = {t:.2f}, p = {p:.2g}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, loc="left")
    ax.set_xlabel(units)
    ax.set_ylabel("Density")
    core.style_ax(ax)
    return panels


@plot("figure")
def signal_to_noise_demo(ensemble_mean, forced, noise, sn, experiment, threshold=2,
                         units="°C"):
    """Signal, noise and their ratio at one point.

    a) each ensemble's mean (thin) and its LOWESS-smoothed forced response
    (thick); the gap between the thick lines is the signal. b) the noise: the
    pooled rolling standard deviation of both ensembles about their own
    forced response. c) the ratio, with the emergence threshold.

    Args:
        ensemble_mean (xr.DataArray): Ensemble means on (experiment, year).
        forced (xr.DataArray): Their smoothed forced response, the same layout.
        noise (xr.DataArray): Pooled noise on (year,).
        sn (xr.DataArray): S/N on (year,).
        experiment (str): The experiment.
        threshold (float): |S/N| counted as emerged.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 3, panel_w=4.3, panel_h=2.8, wspace=0.75, left=0.8, bottom=0.65, top=0.45, sharex=True)
    ax_signal, ax_noise, ax_ratio = axes = panels.axes[0]
    for name in ("hist-nat", experiment):
        ax_signal.plot(ensemble_mean["year"], ensemble_mean.sel(experiment=name), color=_color(name), lw=0.6,
                       alpha=0.5)
        ax_signal.plot(forced["year"], forced.sel(experiment=name), color=_color(name), lw=2.2, label=name)
    ax_signal.set_title("a) Ensemble means and forced response", loc="left")
    ax_signal.set_ylabel(units)
    ax_signal.legend(frameon=False, fontsize=8)

    ax_noise.plot(noise["year"], noise, color="0.25", lw=1.8)
    ax_noise.set_title("b) Noise: pooled rolling std", loc="left")
    ax_noise.set_ylabel(units)

    ax_ratio.axhspan(-threshold, threshold, color="0.92", zorder=0, label=f"|S/N| < {threshold}")
    ax_ratio.plot(sn["year"], sn, color=_color(experiment), lw=1.8)
    ax_ratio.axhline(0, color="0.5", lw=0.8)
    ax_ratio.set_title(f"c) S/N: ({experiment} − hist-nat) / noise", loc="left")
    ax_ratio.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Year")
        core.style_ax(ax)
    return panels


@plot("figure")
def members_by_experiment(point, experiments, variable="tas", season="DJF", units="°C"):
    """Every member of each experiment (thin) and their mean (thick) at one point, one row per experiment.

    Args:
        point (xr.DataTree): One model's /<experiment> nodes at one point, on (member, year, season).
        experiments (Sequence[str]): Rows, top to bottom.
        variable (str): Variable name.
        season (str): Season shown.
        units (str): Units of the variable.

    Returns:
        core.Panels
    """
    experiments = [e for e in experiments if e in point.children]
    panels = core.panel_grid(len(experiments), 1, panel_w=8.5, panel_h=1.0, hspace=0.15, left=1.4, bottom=0.65,
                             top=0.3, has_title=True, sharex=True, sharey=True)
    axes = panels.axes[:, 0]
    for row, (ax, experiment) in enumerate(zip(axes, experiments)):
        da = point[experiment][variable].sel(season=season)
        ax.plot(da["year"], da.transpose("year", "member").values, color=_color(experiment), lw=0.4, alpha=0.35)
        ax.plot(da["year"], da.mean("member"), color=_color(experiment), lw=2)
        ax.set_ylabel(f"{experiment}\n({da.sizes['member']})", rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(labelbottom=row == len(experiments) - 1)
        core.style_ax(ax)
    axes[-1].set_xlabel("Year")
    core.add_suptitle(panels.fig, panels.layout,
                      f"{season}, every member (thin) and the ensemble mean (thick), {units}",
                      fontsize=10, fontweight="normal")
    return panels


@plot("figure")
def member_count_heatmap(counts, title="Ensemble members"):
    """Members per model (rows) and experiment (columns); grey where there is none.

    Args:
        counts (pd.DataFrame): models x experiments.
        title (str): Figure title.

    Returns:
        core.Panels
    """
    panels = core.panel_grid(1, 1, panel_w=0.9 * counts.shape[1] + 0.5, panel_h=0.4 * counts.shape[0] + 0.3,
                             left=1.8, right=1.0, bottom=1.2, top=0.45)
    ax = panels.axes[0, 0]
    cmap = plt.get_cmap("Blues").with_extremes(bad="0.9")
    values = counts.values.astype(float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto")
    ax.set_xticks(range(counts.shape[1]), counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(counts.shape[0]), counts.index)
    for (row, col), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(col, row, int(value), ha="center", va="center",
                    color="white" if value > np.nanmax(values) * 0.6 else "black")
    panels.fig.colorbar(image, ax=ax, label="Members", fraction=0.08, pad=0.04)
    ax.set_title(title)
    return panels
