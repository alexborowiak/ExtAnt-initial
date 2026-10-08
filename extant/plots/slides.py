"""Figures for talks: labels for line and plume figures, and pictures of how the forced-response figures are made.

Sections
--------
1. Labels: names at the end of lines, quantile levels at the edge of a plume
2. How the figures are made: a count of models from its tests (``count_schematic``), the joint key drawn
   with sketched distributions (``joint_key_explainer``), and maps revealed row by row (``reveal_frames``)
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


@plot("figure")
def count_schematic(hist_nat, experiment_values, changes, pvalues, counts, point, experiment, alpha=0.05, units="°C",
                    title=None):
    """How a "how many models" map is made: one test per model at one grid point, counted, then mapped.

    Left, one panel per model: hist-nat's and the experiment's final years
    at the point (members and years pooled), their means, the change and its
    p-value, and the verdict. Right: the count map of the experiment, with
    the point starred: the number there is the number of red (or blue)
    verdicts on the left.

    Args:
        hist_nat, experiment_values (dict[str, xr.DataArray]): Model -> its values at the point.
        changes, pvalues (xr.DataArray): The change and its p-value at the point, on ``model``.
        counts (xr.DataArray): ``significance.significant_counts`` output for the experiment, on
            (direction, lat, lon).
        point (dict): ``{"lat": ..., "lon": ...}``.
        experiment (str): Its name, and its colour (FORCING_COLORS).
        alpha (float): Significance level.
        units (str): Units of the values.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    import cartopy.crs as ccrs
    from matplotlib.gridspec import GridSpec
    from plotting_modules import maps  # needs cartopy

    from .forced_response import SLIDES, count_scale

    models = list(hist_nat)
    n_cols = int(np.ceil(len(models) / 2)) if len(models) > 2 else len(models)
    n_rows = int(np.ceil(len(models) / n_cols))
    colour = FORCING_COLORS.get(experiment, INK)
    verdicts = {1: ("significant increase", "#b2182b"), -1: ("significant decrease", "#2166ac"),
                0: ("not significant", MUTED)}
    n_models = len(models)
    scale = count_scale(n_models)
    signed = counts.sel(direction="increase") - counts.sel(direction="decrease")

    with plt.rc_context(SLIDES.rc):
        fig = plt.figure(figsize=(3.0 * n_cols + 6.0, 2.6 * n_rows + 1.2))
        grid = GridSpec(n_rows, n_cols + 2, figure=fig, width_ratios=[3.0] * n_cols + [1.4, 4.2], wspace=0.35,
                        hspace=0.55, left=0.04, right=0.97, top=0.82 if title else 0.9, bottom=0.12)
        pooled = np.concatenate([np.ravel(v.values) for v in [*hist_nat.values(), *experiment_values.values()]])
        pooled = pooled[np.isfinite(pooled)]
        x = np.linspace(pooled.min() - 0.1 * np.ptp(pooled), pooled.max() + 0.1 * np.ptp(pooled), 300)
        for k, model in enumerate(models):
            ax = fig.add_subplot(grid[k // n_cols, k % n_cols])
            for values, color, label in ((hist_nat[model], FORCING_COLORS["hist-nat"], "hist-nat"),
                                         (experiment_values[model], colour, experiment)):
                density = _density(values.values, x)
                ax.fill_between(x, density, color=color, alpha=0.25, lw=0)
                ax.plot(x, density, color=color, lw=1.6, label=label)
                ax.axvline(float(values.mean()), color=color, lw=1.2, ls=(0, (3, 2)))
            change, pvalue = float(changes.sel(model=model)), float(pvalues.sel(model=model))
            sign = int(np.sign(change)) if pvalue < alpha else 0
            verdict, verdict_colour = verdicts[sign]
            #(c): The model above the panel; the change, p-value and verdict in room left above the curves
            ax.set_title(model, loc="left", fontsize=11)
            ax.text(0.02, 0.98, f"Δ = {change:+.2f} {units}, p = {pvalue:.2g}", transform=ax.transAxes, va="top",
                    fontsize=9, color=INK_2)
            ax.text(0.02, 0.84, verdict, transform=ax.transAxes, va="top", fontsize=9.5, fontweight="bold",
                    color=verdict_colour)
            ax.set_yticks([])
            ax.set_xlabel(units, fontsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_ylim(0, 1.5 * ax.get_ylim()[1])
            if k == 0:
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.18), ncols=2, frameon=False, fontsize=9.5)

        #(c): The step between: an arrow and what it does
        ax_arrow = fig.add_subplot(grid[:, n_cols])
        ax_arrow.axis("off")
        ax_arrow.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                          arrowprops=dict(arrowstyle="-|>", lw=2.5, color=INK_2, mutation_scale=25))
        n_up = int((np.sign(changes) * (pvalues < alpha) > 0).sum())
        n_down = int((np.sign(changes) * (pvalues < alpha) < 0).sum())
        ax_arrow.text(0.5, 0.58, "count the\nverdicts", ha="center", va="bottom", fontsize=11, color=INK_2,
                      transform=ax_arrow.transAxes)
        ax_arrow.text(0.5, 0.42, f"{n_up} up\n{n_down} down\nof {n_models}", ha="center", va="top", fontsize=12,
                      fontweight="bold", color=INK, transform=ax_arrow.transAxes)

        #(c): The count map, with the point starred
        ax_map = fig.add_subplot(grid[:, n_cols + 1], projection=ccrs.SouthPolarStereo())
        mesh = maps.draw_polar_contour(ax_map, signed, scale.levels, scale.cmap, discrete=True)
        ax_map.plot(point["lon"], point["lat"], marker="*", ms=18, color="gold", mec=INK, mew=1.2,
                    transform=ccrs.PlateCarree(), zorder=5)
        ax_map.set_title(f"{experiment}: models with a\nsignificant change", fontsize=11)
        cbar = fig.colorbar(mesh, ax=ax_map, orientation="horizontal", fraction=0.06, pad=0.06,
                            boundaries=scale.levels, ticks=(scale.levels[:-1] + scale.levels[1:]) / 2)
        cbar.set_ticklabels(scale.ticklabels)
        cbar.ax.tick_params(length=0, labelsize=9)
        cbar.set_label("decrease (blue) or increase (red)", fontsize=9.5)
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, x=0.04, ha="left")
    return core.Panels(fig=fig, axes=np.array([ax_map]))


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


def reveal_frames(da, scale, directory, row_dim="experiment", col_dim="season", not_significant=None, title=None,
                  panel_size=2.4, not_significant_label="Not significant"):
    """Frames for a talk: a grid of maps with its rows revealed one at a time; returns the saved paths.

    Every frame is drawn on the same canvas (the rows not yet revealed are
    empty circles, and the colour bar and legend are there from the start),
    so flipping through them only adds maps. Saved with
    ``plotting_modules.utils.save_frame`` as ``<step>_<row>.png``.

    Args:
        da (xr.DataArray): On (row_dim, col_dim, lat, lon).
        scale (forced_response.Scale): Its colour scale.
        directory (pathlib.Path): Folder for the frames.
        row_dim, col_dim (str): Dims mapped to rows (revealed in order) and columns.
        not_significant (xr.DataArray | None): Boolean, the same dims: dotted where True.
        title (str | None): Title of every frame.
        panel_size (float): Width and height of each panel, in inches.
        not_significant_label (str): Legend entry for the dots.
    """
    from plotting_modules.utils import save_frame

    from .forced_response import SLIDES, Block, block_grid

    rows, columns = list(da[row_dim].values), list(da[col_dim].values)

    def shown(source, step):
        return {str(row): [source.sel({row_dim: row, col_dim: column}) if k < step else None for column in columns]
                for k, row in enumerate(rows)}

    paths = []
    for step in range(1, len(rows) + 1):
        block = Block(shown(da, step), [scale] * len(columns),
                      None if not_significant is None else shown(not_significant, step))
        panels = block_grid([block], title=title, column_titles=[str(c) for c in columns], panel_size=panel_size,
                            style=SLIDES, not_significant_label=not_significant_label)
        paths.append(save_frame(panels.fig, directory, step, str(rows[step - 1])))
    return paths
