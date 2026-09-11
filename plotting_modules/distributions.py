"""Distributions: KDEs, histograms and per-member bars.

The grid functions here take a dict of datasets keyed by experiment, which is
why they build their own panel callbacks rather than taking a DataArray.
"""

import numpy as np

from . import core
from .utils import period_label, plot

@plot("axes")
def draw_members(ax, values, width=0.75, color="#3b6ea5"):
    """Sorted per-member bars on one axes."""
    values = np.sort(np.asarray(values))
    members = np.arange(len(values))
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    pad = 0.05 * (vmax - vmin)

    bars = ax.bar(members, values, width=width, color=color, edgecolor="none", zorder=3)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlim(-0.75, len(members) - 0.25)
    ax.tick_params(labelsize=8)
    style_ax(ax)
    return bars


# --------------------------------------------------------------------------
# Grids: experiments down the page, periods across it
# --------------------------------------------------------------------------




def hist_bins(point, periods, experiments, variable="tas", n_bins=20):
    """Common bin edges spanning every experiment and period."""
    vals = []
    for exp in experiments:
        for period in periods:
            vals.extend(point.sel(year=period)[exp][variable].values.ravel())
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


# --------------------------------------------------------------------------
# Grids: experiments down the page, periods across it
# --------------------------------------------------------------------------

PANEL_KWARGS = dict(panel_w=1.8, panel_h=1.6, wspace=0.08, hspace=0.10,
                    left=0.75, bottom=0.55)


def _grid(rows, cols, draw, xlabel, col_fmt=str, **kwargs):
    """Shared body of every distribution grid: panels, hide inner, label, return."""
    for key, value in PANEL_KWARGS.items():
        kwargs.setdefault(key, value)
    panels = core.panel_grid(rows, cols, draw, sharex=True, sharey=True, **kwargs)
    core.hide_inner_labels(panels.axes)
    core.label_cols(panels.axes, cols, col_fmt, pad=6, fontweight="normal")
    core.label_rows(panels.axes, rows, rotate=True, fontweight="normal")
    for ax in panels.axes[-1, :]:
        ax.set_xlabel(xlabel)
    return panels


@plot("figure")
def kde_grid(kde, periods, colors, variable="tas", **kwargs):
    """One KDE per (experiment, period), experiments down, periods across.

    Parameters
    ----------
    kde : dict[str, xarray.Dataset]
        Experiment -> dataset holding `variable` with an 'x' coordinate.
    periods : sequence
        Year selections, each a scalar or a slice.
    colors : dict[str, str]
        Experiment -> colour, keyed the same way as `kde`.
    variable : str
        Variable to plot.
    **kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    def draw(ax, exp, period):
        da = kde[exp][variable].sel(year=period)
        if "year" in da.dims:
            da = da.mean("year")
        line, = ax.plot(da.x, da, color=colors[exp])
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        core.style_ax(ax)
        return line

    return _grid(list(kde), periods, draw, variable, col_fmt=period_label, **kwargs)


@plot("figure")
def kde_overlay(kde, periods, colors, variable="tas", **kwargs):
    """One panel per period, all experiments overlaid.

    Same arguments as `kde_grid`. Kept separate rather than folded in as a mode
    flag: the two share no layout, no legend handling and no axis labelling.

    Returns
    -------
    core.Panels
    """
    experiments = list(kde)

    def draw(ax, _, period):
        lines = []
        for exp in experiments:
            da = kde[exp][variable].sel(year=period)
            if "year" in da.dims:
                da = da.mean("year")
            lines += ax.plot(da.x, da, color=colors[exp], label=exp)
        ax.set_title("")
        ax.set_ylabel("")
        core.style_ax(ax)
        return lines

    panels = _grid([None], periods, draw, variable, col_fmt=period_label, **kwargs)
    panels.axes[0, 0].set_ylabel("Density")
    handles, labels = panels.axes[0, 0].get_legend_handles_labels()
    panels.extras["legend"] = panels.fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=len(experiments), frameon=False,
    )
    return panels


@plot("figure")
def hist_grid(point, periods, colors, variable="tas", n_bins=20, **kwargs):
    """One histogram per (experiment, period), on shared bins.

    `colors` is keyed by experiment, matching `kde_grid`.

    Returns
    -------
    core.Panels
    """
    experiments = list(point)
    bins = hist_bins(point, periods, experiments, variable, n_bins)

    def draw(ax, exp, period):
        out = point.sel(year=period)[exp][variable].plot.hist(
            ax=ax, bins=bins, color=colors[exp], alpha=0.5
        )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(bins[0], bins[-1])
        core.style_ax(ax)
        return out

    panels = _grid(experiments, periods, draw, variable, col_fmt=period_label, **kwargs)
    panels.extras["bins"] = bins
    return panels
