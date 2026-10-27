"""Panels of where and how the forced change affects the mean and the variability.

Draws the output of response_change. Each function styles one panel (``charts.style_chart_ax``) and draws on it;
the notebook makes the figure and its axes (a GridSpec), then draws each panel and labels them, as everywhere
(``charts``). Each
experiment has its colour, FORCING_COLORS (hist-nat green, hist-GHG red, ...). The maps of these results are drawn
layer by layer with ``antarctica``, the area means with ``charts.plot_model_dots`` and
``charts.plot_scatter_by_forcing``, in notebook plot_03.

Sections
--------
1. Scales: colour-bar ticks, the number of models, the extremes' titles
2. Local: how the distribution shifts and widens at one grid point (``plot_point_distributions``,
   ``plot_quantile_shifts``)
3. Summary: what each corner of a scatter of two changes means (``plot_quadrant_labels``)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

from .. import era5_evaluation as ev
from . import charts
from .era5_evaluation import INK, INK_2
from plotting_modules.constants import FORCING_COLORS

# ---------------------------------------------------------------------------
# 1. Scales
# ---------------------------------------------------------------------------

def round_ticks(levels, most):
    """Every k-th level, at most ``most`` of them, for a colour bar.

    k divides the number of steps, so the end levels are ticked, and on a
    symmetric scale it divides each half as well, so 0 is ticked too. Of the
    k that fit, the smallest giving round ticks (every 1, 2, 2.5 or 5 x 10^n)
    is taken, e.g. 0, 2, 4 rather than 0, 1.5, 3; failing that, the smallest.
    """
    levels = np.asarray(levels, dtype=float)
    n_steps = len(levels) - 1
    symmetric = n_steps % 2 == 0 and np.isclose(levels[n_steps // 2], 0)
    fitting = [step for step in range(1, n_steps + 1)
               if n_steps % step == 0 and (not symmetric or (n_steps // 2) % step == 0) and n_steps // step + 1 <= most]
    if not fitting:
        return levels[[0, -1]]

    def round_spacing(step):
        spacing = abs(levels[step] - levels[0])
        mantissa = spacing / 10 ** np.floor(np.log10(spacing))
        return any(np.isclose(mantissa, m) for m in (1, 2, 2.5, 5, 10))

    return levels[::next((step for step in fitting if round_spacing(step)), fitting[0])]


def count_colormap(n_models, base_cmap="Purples"):
    """Discrete colormap for a number of models: white for none, one step per model."""
    base = plt.get_cmap(base_cmap)
    cmap = ListedColormap(["white", *[base(x) for x in np.linspace(0.3, 1, n_models)]])
    return cmap, BoundaryNorm(np.arange(-0.5, n_models + 1.5), cmap.N)


#(t): The titles of the extremes' maps and zonal means (notebook plot_03c), by their names in ``rc.EXTREMES``
EXTREME_TITLES = {
    "mean_change": "Mean",
    "low_extreme_change": "Low extremes",
    "high_extreme_change": "High extremes",
    "width_change": "Width",
}


# ---------------------------------------------------------------------------
# 2. Local: how the distribution shifts and widens at one grid point
# ---------------------------------------------------------------------------

def plot_point_distributions(samples, shift_ds, experiment, test_ds=None, units="°C", ax=None):
    """At one grid point: the distribution of the final years in hist-nat and in ``experiment`` (pooled members and
    years), with arrows from each hist-nat quantile (Q05, Q50, Q95) to the experiment's.

    Equal arrows are a pure shift. A box in the corner gives the width in each, and the test results if given.

    Args:
        samples (dict[str, xr.DataArray]): ``rc.final_years`` output, including hist-nat and ``experiment``.
        shift_ds (xr.Dataset): ``rc.quantile_shift`` of ``experiment`` against hist-nat.
        experiment (str): The experiment drawn against hist-nat.
        test_ds (xr.Dataset | None): ``rc.change_summary`` at this model, experiment, season and point; adds the
            test results to the box.
        units (str): Units of the variable.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the two distributions.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=6.0)
    charts.style_chart_ax(ax)
    #(c): The two distributions, sharing one kernel bandwidth
    pooled = {name: samples[name].stack(sample=("member", "year")).dropna("sample") for name in ("hist-nat", experiment)}
    values = np.concatenate([p.values for p in pooled.values()])
    bandwidth = min(float(p.std()) * p.size ** (-1 / 5) for p in pooled.values())
    x = np.linspace(values.min() - 4 * bandwidth, values.max() + 4 * bandwidth, 600)
    grid = xr.DataArray(x, dims="x")
    peak = 0
    for name, sample in pooled.items():
        density = ev.kde(sample, grid, "sample", bandwidth).values
        peak = max(peak, density.max())
        colour = FORCING_COLORS.get(name, INK)
        ax.fill_between(x, density, color=colour, alpha=0.18, lw=0)
        ax.plot(x, density, color=colour, lw=2, label=f"{name} ({sample.size} values)")
        for level in (0.05, 0.5, 0.95):
            q = float(sample.quantile(level))
            ax.plot([q, q], [0, np.interp(q, x, density)], color=colour, lw=1.1, ls=(0, (3, 2)))

    #(c): Three bands, bottom to top: the curves, the quantile arrows, then the legend and test results
    for k, level in enumerate((0.05, 0.5, 0.95)):
        start = float(shift_ds["reference_quantiles"].sel(quantile=level))
        end = float(shift_ds["experiment_quantiles"].sel(quantile=level))
        height = peak * (1.1 + 0.12 * k)
        ax.annotate("", xy=(end, height), xytext=(start, height),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
        ax.text(max(start, end) + 0.02 * np.ptp(x), height, f"ΔQ{round(level * 100):02d} = {end - start:+.2f} {units}",
                va="center", fontsize=8, color=INK)
    ax.set_ylim(0, peak * 1.9)
    ax.legend(loc="upper left", frameon=False)

    width = {name: float(p.quantile(0.95) - p.quantile(0.05)) for name, p in pooled.items()}
    lines = [f"Width Q95 − Q05: {width['hist-nat']:.2f} → {width[experiment]:.2f} {units}"]
    if test_ds is not None:
        lines += [
            f"Mean: {float(test_ds['mean_change']):+.2f} {units}, mean test p = {float(test_ds['mean_pvalue']):.2g}, "
            f"S/N = {float(test_ds['signal_to_noise']):.1f}",
            f"Width: {float(test_ds['width_change']):+.2f} {units}, bootstrap p = {float(test_ds['width_pvalue']):.2g}",
        ]
    ax.text(0.99, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=INK_2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    return ax


def plot_quantile_shifts(shifts, experiment=None, ax=None):
    """At one grid point: the change in every quantile against hist-nat, for every experiment, with its bootstrap
    5-95% range.

    A flat curve is a pure shift. A rising curve means the warm tail warms faster than the cold tail (the
    distribution widens upwards); a falling curve, the cold tail warms faster.

    Args:
        shifts (dict[str, xr.Dataset]): Experiment -> ``rc.quantile_shift`` against hist-nat.
        experiment (str | None): The experiment drawn thicker, e.g. the one of ``plot_point_distributions``' panel.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the experiments.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    ax.axhline(0, color="0.6", lw=0.8)
    for name, shift_ds in shifts.items():
        colour = FORCING_COLORS.get(name, INK)
        percent = 100 * shift_ds["quantile"].values
        ax.fill_between(percent, shift_ds["shift_lower"], shift_ds["shift_upper"], color=colour, alpha=0.18, lw=0)
        ax.plot(percent, shift_ds["shift"], color=colour, lw=2.2 if name == experiment else 1.4, marker="o", ms=3,
                label=name)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# 3. Summary: what each corner of a scatter of two changes means
# ---------------------------------------------------------------------------

#(t): Each change a scatter can show: its axis label, and what its low and high ends mean
SCATTER_LABELS = {
    "mean_change": ("Mean response (°C)", "cooler", "warmer"),
    "width_change": ("Change in width Q95 − Q05 (°C)", "narrower", "wider"),
    "upper_tail_change": ("Warm-tail change Δ(Q95 − Q50) (°C)", "warm tail shorter", "warm tail longer"),
    "lower_tail_change": ("Cold-tail change Δ(Q50 − Q05) (°C)", "cold tail shorter", "cold tail longer"),
    "tail_asymmetry": ("Tail asymmetry, warm − cold tail change (°C)", "cold tail stretches more",
                       "warm tail stretches more"),
}


def plot_quadrant_labels(x, y, ax=None):
    """In each corner of a scatter of two changes, what a point there means: e.g. 'warmer / wider' at the top right.

    Args:
        x, y (str): The changes on the x and y axes, keys of ``SCATTER_LABELS``, e.g. 'mean_change' and
            'width_change'.
        ax (matplotlib.axes.Axes | None): Where to write, e.g. a panel of ``charts.plot_scatter_by_forcing``; a new
            figure if None.

    Returns:
        matplotlib.axes.Axes: The axis. Its axis labels are ``SCATTER_LABELS[x][0]`` and ``SCATTER_LABELS[y][0]``.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    _, x_low, x_high = SCATTER_LABELS[x]
    _, y_low, y_high = SCATTER_LABELS[y]
    corners = {(0.98, 0.98, "right", "top"): f"{x_high}\n{y_high}", (0.02, 0.98, "left", "top"): f"{x_low}\n{y_high}",
               (0.02, 0.02, "left", "bottom"): f"{x_low}\n{y_low}",
               (0.98, 0.02, "right", "bottom"): f"{x_high}\n{y_low}"}
    for (corner_x, corner_y, ha, va), text in corners.items():
        ax.text(corner_x, corner_y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7, color="0.55",
                multialignment=ha)
    return ax
