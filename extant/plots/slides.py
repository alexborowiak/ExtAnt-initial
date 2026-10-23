"""Figures for talks: labels for line and plume figures, and pictures of how the forced-response figures are made.

Sections
--------
1. Labels: names at the end of lines, quantile levels at the edge of a plume
2. How the figures are made: one model's test at a point (``plot_point_test``, which notebook 05 draws for
   every model before counting them), and the joint key drawn with sketched distributions
   (``joint_key_explainer``)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .era5_evaluation import INK, INK_2, MUTED
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS
from plotting_modules.utils import plot


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
        pairs (array-like): (low, high) quantile pairs, as passed to ``timeseries.draw_plume``.
        x (float): x position of the labels.
        color: Text colour.
        quantile_dim (str): Dimension holding the quantiles.
        **kwargs: Passed to ``ax.annotate``.
    """
    for q in np.unique(np.asarray(pairs)):
        label_line_end(ax, x, da.sel({quantile_dim: q}, method="nearest").isel({da.dims[-1]: -1}),
                       f"{q:.0%}", color, **kwargs)


# ---------------------------------------------------------------------------
# How the figures are made, for talks
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
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure if None.
        alpha (float): The significance level.
        units (str): The values' units.

    Returns:
        matplotlib.axes.Axes: The axis.

    Example::

        plot_point_test(hist_nat_da, historical_da, change=0.8, pvalue=0.01, experiment='historical')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.6))
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
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
    return ax


@plot("figure")
def joint_key_explainer(scale, title="Reading the colours: how the distribution changed"):
    """The 3 x 3 key of ``forced_response.joint_scale`` drawn with sketched distributions: what each colour means.

    Each cell has hist-nat's distribution (grey outline) and a changed one in
    the cell's colour: shifted down, not or up (rows, bottom to top), and
    narrower, the same or wider (columns). No data: a picture to explain the
    joint maps in a talk.

    Args:
        scale (forced_response.Scale): ``joint_scale()``.
        title (str): Figure title.

    Returns:
        core.Panels
    """
    from .forced_response import SLIDES

    rows, columns, row_title, column_title = scale.key
    shifts, widths = (-1.2, 0.0, 1.2), (0.65, 1.0, 1.5)
    x = np.linspace(-5, 5, 400)

    def gaussian(mean, sd):
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    with plt.rc_context(SLIDES.rc):
        fig, axes = plt.subplots(len(rows), len(columns), figsize=(9, 7.5), sharex=True, sharey=True,
                                 gridspec_kw=dict(hspace=0.08, wspace=0.08, left=0.2, right=0.97, top=0.86, bottom=0.12))
        for i in range(len(rows)):
            for j in range(len(columns)):
                #(c): Row 0 of the key is its bottom row
                ax = axes[len(rows) - 1 - i, j]
                cls = i * len(columns) + j
                colour = scale.cmap(cls)
                ax.set_facecolor(colour)
                ax.patch.set_alpha(0.35)
                ax.plot(x, gaussian(0, 1), color="0.35", lw=1.4, ls=(0, (3, 2)))
                changed = gaussian(shifts[i], widths[j])
                ax.fill_between(x, changed, color=colour, alpha=0.95 if cls != 4 else 0.0, lw=0)
                ax.plot(x, changed, color=INK, lw=1.4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(0, 0.7)
                if i == 0:
                    ax.set_xlabel(columns[j], fontsize=13)
                if j == 0:
                    ax.set_ylabel(rows[i], fontsize=13, rotation=0, ha="right", va="center")
        fig.text(0.585, 0.03, f"{column_title} of the distribution", ha="center", fontsize=14, color=INK_2)
        fig.text(0.03, 0.49, row_title, ha="center", va="center", rotation=90, fontsize=14, color=INK_2)
        fig.legend(handles=[Line2D([], [], color="0.35", lw=1.4, ls=(0, (3, 2)), label="hist-nat"),
                            Line2D([], [], color=INK, lw=1.4, label="the experiment")],
                   loc="upper right", bbox_to_anchor=(0.97, 0.95), ncols=2, frameon=False, fontsize=12)
        fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.2, ha="left", y=0.97)
    return core.Panels(fig=fig, axes=axes)
