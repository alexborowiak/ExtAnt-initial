"""Figures for where and how the forced change affects the mean and the variability.

Draws the output of response_change. Follows the plotting_modules conventions
(``@plot``, ``core.Panels``) and shares its style with era5_evaluation_plots.

Colour and pattern mean the same thing in every figure:
    red <-> blue      warmer <-> cooler mean (maps)
    ////              the distribution significantly wider (Q95 - Q05)
    \\\\              the distribution significantly narrower
    white             no robust change in the mean
    experiment        FORCING_COLORS (hist-nat green, hist-GHG red, ...)

Sections
--------
1. Maps: the mean response, with the width change hatched on top
2. Local: how the distribution shifts and widens at one grid point
3. Summary: mean change against width change, every model and experiment
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import era5_evaluation as ev
import response_change as rc
from era5_evaluation_plots import EVAL_RC, INK, INK_2, MODEL_MARKERS, order_seasons
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, FORCING_REVEAL_ORDER
from plotting_modules.utils import plot

WIDER_HATCH = "////"
NARROWER_HATCH = "\\\\\\\\"
HATCH_COLOR = (0, 0, 0, 0.6)
HATCH_WIDTH = 0.6


def _experiments(da, experiments=None):
    """Experiments in the notebook's reveal order, keeping only those present."""
    present = [str(e) for e in da["experiment"].values]
    order = experiments or [e for e in FORCING_REVEAL_ORDER if e in present] + \
        [e for e in present if e not in FORCING_REVEAL_ORDER]
    return [e for e in order if e in present]


def symmetric_levels(da, n_steps=12, quantile=0.98):
    """Levels symmetric about zero, on a round step (1, 2, 2.5 or 5 x 10^k), covering the ``quantile`` of |da|."""
    limit = float(np.nanquantile(np.abs(da.values), quantile)) or 1.0
    raw_step = 2 * limit / n_steps
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(np.log(s / raw_step)))
    half = int(np.ceil(limit / step - 1e-9))
    return np.arange(-half, half + 1) * step


def count_colormap(n_models, base_cmap="Purples"):
    """Discrete colormap for a number of models: white for none, one step per model."""
    base = plt.get_cmap(base_cmap)
    cmap = ListedColormap(["white", *[base(x) for x in np.linspace(0.3, 1, n_models)]])
    return cmap, BoundaryNorm(np.arange(-0.5, n_models + 1.5), cmap.N)


# ---------------------------------------------------------------------------
# 1. Maps: the mean response, with the width change hatched on top
# ---------------------------------------------------------------------------

@plot("figure")
def joint_change_maps(summary, model, experiments=None, mean_test="robust", levels=None, title=None):
    """Where the mean and the width have changed: one model, experiments down, seasons across.

    Colour is the mean response, shown only where the mean change passes
    ``mean_test`` (white elsewhere). Hatching marks a significant change in the
    width of the distribution: //// wider, \\\\ narrower. Coloured and hatched
    therefore means both the mean and the variability have changed.

    Args:
        summary (xr.Dataset): ``rc.change_summary`` output.
        model (str): Model to show.
        experiments (Sequence[str] | None): Rows; the notebook's forcing order by default.
        mean_test (str): Key of ``rc.MEAN_TESTS``.
        levels (array-like | None): Colour levels; symmetric about zero by default.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    from plotting_modules import maps  # needs cartopy

    data = order_seasons(summary.sel(model=model))
    experiments = _experiments(data, experiments)
    data = data.sel(experiment=experiments)
    shown = data["mean_change"].where(data[f"mean_{mean_test}"])
    levels = symmetric_levels(data["mean_change"]) if levels is None else levels

    with plt.rc_context(EVAL_RC):
        panels = maps.polar_grid(
            shown, row_dim="experiment", col_dim="season", levels=levels, cmap="RdBu_r",
            title=title or f"{model}: mean response and width change (final years vs hist-nat)",
            cbar_label=f"Mean response (°C), where {rc.MEAN_TESTS[mean_test]}",
            tag=False, left=1.4, bottom=0.55,
        )
        for i, experiment in enumerate(experiments):
            for j, season in enumerate(data["season"].values):
                cell = data.sel(experiment=experiment, season=season)
                changed = cell["width_significant"]
                maps.plot_hatch(panels.axes[i, j], changed & (cell["width_change"] > 0), hatch=WIDER_HATCH,
                                color=HATCH_COLOR, linewidth=HATCH_WIDTH)
                maps.plot_hatch(panels.axes[i, j], changed & (cell["width_change"] < 0), hatch=NARROWER_HATCH,
                                color=HATCH_COLOR, linewidth=HATCH_WIDTH)
        panels.fig.legend(
            handles=[
                Patch(facecolor="white", edgecolor="0.25", hatch=WIDER_HATCH, label="Significantly wider (Q95 − Q05)"),
                Patch(facecolor="white", edgecolor="0.25", hatch=NARROWER_HATCH, label="Significantly narrower"),
                Patch(facecolor="white", edgecolor="0.6", label="White: no robust mean change"),
            ],
            loc="lower center", ncol=3, frameon=False, handlelength=2.2, handleheight=1.2,
            bbox_to_anchor=(0.5, 0.0),
        )
    return panels


@plot("figure")
def joint_change_count_maps(counts, season, n_models, experiments=None, title=None):
    """How many models show a robust mean change together with a significant widening, or a narrowing.

    Args:
        counts (xr.DataArray): ``rc.joint_change_counts`` output.
        season (str): Season to show.
        n_models (int): Number of models, for the colour scale.
        experiments (Sequence[str] | None): Rows; the notebook's forcing order by default.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    from plotting_modules import maps  # needs cartopy

    experiments = _experiments(counts, experiments)
    cmap, norm = count_colormap(n_models)
    with plt.rc_context(EVAL_RC):
        panels = maps.polar_grid(
            counts.sel(season=season, experiment=experiments), row_dim="experiment", col_dim="change",
            levels=np.arange(-0.5, n_models + 1.5), cmap=cmap, norm=norm,
            title=title or f"Models where the mean and the width both changed, {season}",
            cbar_label="Number of models", tag=False, left=1.4, discrete=True,
        )
    return panels


# ---------------------------------------------------------------------------
# 2. Local: how the distribution shifts and widens at one grid point
# ---------------------------------------------------------------------------

@plot("figure")
def shift_and_widen(samples, shifts, experiment, reference=rc.REFERENCE, test=None, units="°C", title=None):
    """How the distribution changes at one grid point: shift of the middle, change of the width and tails.

    a) The distribution of the final years in ``reference`` and in
       ``experiment`` (pooled members and years), with arrows from each
       reference quantile (Q05, Q50, Q95) to the experiment's.
    b) The change in every quantile, for every experiment in ``shifts``, with
       its bootstrap 5-95% range. A flat curve is a pure shift. A rising curve
       means the warm tail warms faster than the cold tail (the distribution
       widens upwards); a falling curve, the cold tail warms faster.

    Args:
        samples (dict[str, xr.DataArray]): ``rc.final_years`` output, including ``reference``.
        shifts (dict[str, xr.Dataset]): Experiment -> ``rc.quantile_shift`` against ``reference``.
        experiment (str): The experiment shown in panel a.
        reference (str): The baseline experiment.
        test (xr.Dataset | None): ``rc.change_summary`` at this model, experiment, season and
            point; adds the test results to panel a.
        units (str): Units of the variable.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    with plt.rc_context(EVAL_RC):
        fig, (ax_dist, ax_shift) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained",
                                                gridspec_kw={"width_ratios": (1.35, 1)})

        #(t): a) the two distributions, sharing one kernel bandwidth
        pooled = {name: samples[name].stack(sample=("member", "year")).dropna("sample") for name in (reference, experiment)}
        values = np.concatenate([p.values for p in pooled.values()])
        bandwidth = min(float(p.std()) * p.size ** (-1 / 5) for p in pooled.values())
        x = np.linspace(values.min() - 4 * bandwidth, values.max() + 4 * bandwidth, 600)
        grid = xr.DataArray(x, dims="x")
        peak = 0
        for name, sample in pooled.items():
            density = ev.kde(sample, grid, "sample", bandwidth).values
            peak = max(peak, density.max())
            colour = FORCING_COLORS.get(name, INK)
            ax_dist.fill_between(x, density, color=colour, alpha=0.18, lw=0)
            ax_dist.plot(x, density, color=colour, lw=2, label=f"{name} ({sample.size} values)")
            for level in (0.05, 0.5, 0.95):
                q = float(sample.quantile(level))
                ax_dist.plot([q, q], [0, np.interp(q, x, density)], color=colour, lw=1.1, ls=(0, (3, 2)))

        #(c): Three bands, bottom to top: the curves, the quantile arrows, then the legend and test results
        shift = shifts[experiment]
        for k, level in enumerate((0.05, 0.5, 0.95)):
            start = float(shift["reference_quantiles"].sel(quantile=level))
            end = float(shift["experiment_quantiles"].sel(quantile=level))
            height = peak * (1.1 + 0.12 * k)
            ax_dist.annotate("", xy=(end, height), xytext=(start, height),
                             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
            ax_dist.text(max(start, end) + 0.02 * np.ptp(x), height, f"ΔQ{round(level * 100):02d} = {end - start:+.2f} {units}",
                         va="center", fontsize=8, color=INK)
        ax_dist.set_ylim(0, peak * 1.9)
        ax_dist.set_xlabel(f"Seasonal-mean tas ({units})\n ")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title(f"a) {reference} vs {experiment}: final years, members pooled", loc="left")
        ax_dist.legend(loc="upper left", frameon=False)

        width = {name: float(p.quantile(0.95) - p.quantile(0.05)) for name, p in pooled.items()}
        lines = [f"Width Q95 − Q05: {width[reference]:.2f} → {width[experiment]:.2f} {units}"]
        if test is not None:
            lines += [
                f"Mean: {float(test['mean_change']):+.2f} {units}, mean test p = {float(test['mean_pvalue']):.2g}, "
                f"S/N = {float(test['signal_to_noise']):.1f}",
                f"Width: {float(test['width_change']):+.2f} {units}, bootstrap p = {float(test['width_pvalue']):.2g}",
            ]
        ax_dist.text(0.99, 0.98, "\n".join(lines), transform=ax_dist.transAxes, ha="right", va="top", fontsize=8,
                     color=INK_2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

        #(t): b) the change in every quantile, for every experiment
        ax_shift.axhline(0, color="0.6", lw=0.8)
        for name, result in shifts.items():
            colour = FORCING_COLORS.get(name, INK)
            percent = 100 * result["quantile"].values
            ax_shift.fill_between(percent, result["shift_lower"], result["shift_upper"], color=colour, alpha=0.18, lw=0)
            ax_shift.plot(percent, result["shift"], color=colour, lw=2.2 if name == experiment else 1.4,
                          marker="o", ms=3, label=name)
        ax_shift.set_xlim(0, 100)
        ax_shift.set_xlabel("Quantile (%)\nflat: a pure shift  ·  rising: warm tail warms faster  ·  "
                            "falling: cold tail warms faster")
        ax_shift.set_ylabel(f"Change vs {reference} ({units})")
        ax_shift.set_title("b) Change in every quantile (bootstrap 5–95%)", loc="left")
        ax_shift.legend(loc="upper left", frameon=False, fontsize=8)

        for ax in (ax_dist, ax_shift):
            core.style_ax(ax)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=np.array([[ax_dist, ax_shift]], dtype=object))


# ---------------------------------------------------------------------------
# 3. Summary: mean change against width change, every model and experiment
# ---------------------------------------------------------------------------

#(t): Axis labels and corner labels for the variables the summary scatter can show
_SCATTER = {
    "mean_change": ("Mean response (°C)", "cooler", "warmer"),
    "width_change": ("Change in width Q95 − Q05 (°C)", "narrower", "wider"),
    "upper_tail_change": ("Warm-tail change Δ(Q95 − Q50) (°C)", "warm tail shorter", "warm tail longer"),
    "lower_tail_change": ("Cold-tail change Δ(Q50 − Q05) (°C)", "cold tail shorter", "cold tail longer"),
    "tail_asymmetry": ("Tail asymmetry, warm − cold tail change (°C)", "cold tail stretches more",
                       "warm tail stretches more"),
}


@plot("figure")
def response_scatter(regional, x="mean_change", y="width_change", experiments=None, diagonal=False, title=None):
    """Regional-mean changes against each other, per season: e.g. the mean against the width, or one tail against the other.

    Colour is the experiment (FORCING_COLORS) and the marker the model, so each
    point is one model's response to one forcing. With ``x="lower_tail_change"``,
    ``y="upper_tail_change"`` and ``diagonal=True``, points above the 1:1 line
    are where the warm tail stretched more than the cold tail.

    Args:
        regional (xr.Dataset): ``rc.regional_mean`` output, on (model, experiment, season).
        x, y (str): Variables for the axes; keys of ``_SCATTER``.
        experiments (Sequence[str] | None): Experiments shown; the notebook's forcing order by default.
        diagonal (bool): Draw the 1:1 line.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    regional = order_seasons(regional)
    experiments = _experiments(regional, experiments)
    models = list(regional["model"].values)
    seasons = list(regional["season"].values)
    x_label, x_low, x_high = _SCATTER[x]
    y_label, y_low, y_high = _SCATTER[y]

    with plt.rc_context(EVAL_RC):
        fig, axes = plt.subplots(1, len(seasons), figsize=(3.5 * len(seasons) + 1.6, 3.8), layout="constrained",
                                 sharex=True, sharey=True, squeeze=False)
        for ax, season in zip(axes[0], seasons):
            ax.axhline(0, color="0.7", lw=0.8, zorder=0)
            ax.axvline(0, color="0.7", lw=0.8, zorder=0)
            for experiment in experiments:
                for k, model in enumerate(models):
                    point = regional.sel(model=model, experiment=experiment, season=season)
                    ax.scatter(float(point[x]), float(point[y]), s=48, marker=MODEL_MARKERS[k % len(MODEL_MARKERS)],
                               color=FORCING_COLORS.get(experiment, INK), edgecolor="white", linewidth=0.9, zorder=3)
            ax.set_title(season, loc="left")
            if diagonal:
                ax.axline((0, 0), slope=1, color="0.55", lw=0.9, ls=(0, (4, 2)), zorder=0)
            for (tx, ty, ha, va), text in zip(((0.98, 0.98, "right", "top"), (0.02, 0.98, "left", "top"),
                                               (0.02, 0.02, "left", "bottom"), (0.98, 0.02, "right", "bottom")),
                                              (f"{x_high}\n{y_high}", f"{x_low}\n{y_high}",
                                               f"{x_low}\n{y_low}", f"{x_high}\n{y_low}")):
                ax.text(tx, ty, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7, color="0.55",
                        multialignment=ha)
            core.style_ax(ax)
        fig.supxlabel(x_label, fontsize=9)
        axes[0, 0].set_ylabel(y_label)

        experiment_handles = [Line2D([], [], ls="none", marker="o", ms=7, color=FORCING_COLORS.get(e, INK), label=e)
                              for e in experiments]
        model_handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=7, color="0.35",
                                label=m) for k, m in enumerate(models)]
        fig.legend(handles=experiment_handles, loc="outside right upper", frameon=False, title="Experiment",
                   alignment="left")
        fig.legend(handles=model_handles, loc="outside right lower", frameon=False, title="Model", alignment="left")
        lat_max = regional.attrs.get("lat_max")
        region = f" south of {abs(lat_max):g}°S" if lat_max else ""
        x_name = x_label.split(" (")[0]
        default = f"Regional mean{region}: {y_label.split(' (')[0]} against {x_name[0].lower()}{x_name[1:]}"
        fig.suptitle(title or default, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return core.Panels(fig=fig, axes=axes)
