"""Distributions: KDEs, histograms and per-member bars.

The grid functions here take datasets keyed by experiment and draw directly
into their axes.
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
    core.style_ax(ax)
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


def _new_grid(n_rows, n_cols, **kwargs):
    """Create the shared distribution layout."""
    for key, value in PANEL_KWARGS.items():
        kwargs.setdefault(key, value)
    return core.panel_grid(n_rows, n_cols, sharex=True, sharey=True, **kwargs)


def _label_grid(panels, row_keys, col_keys, xlabel, col_fmt=str):
    """Apply the shared distribution labels after drawing."""
    core.hide_inner_labels(panels.axes)
    core.label_cols(panels.axes, col_keys, col_fmt, pad=6, fontweight="normal")
    core.label_rows(panels.axes, row_keys, rotate=True, fontweight="normal")
    for ax in panels.axes[-1, :]:
        ax.set_xlabel(xlabel)
    return panels


@plot("figure")
def kde_grid(kde, periods, colors, variable="tas", fig=None, spec=None, layout=None,
             ax=None, axes=None, **layout_kwargs):
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
    ax, axes : matplotlib.axes.Axes or array-like, optional
        Caller-owned panel axes. Use `ax` for a one-panel result and `axes`
        for the complete grid; their figure is inferred when omitted.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    experiments = list(kde)
    panels = _new_grid(
        len(experiments), len(periods), fig=fig, spec=spec, layout=layout,
        ax=ax, axes=axes, **layout_kwargs,
    )
    for row, exp in enumerate(experiments):
        for col, period in enumerate(periods):
            da = kde[exp][variable].sel(year=period)
            if "year" in da.dims:
                da = da.mean("year")
            panel_ax = panels.axes[row, col]
            line, = panel_ax.plot(da.x, da, color=colors[exp])
            panels.artists[row, col] = line
            panel_ax.set_title("")
            panel_ax.set_xlabel("")
            panel_ax.set_ylabel("")
            core.style_ax(panel_ax)
    return _label_grid(panels, experiments, periods, variable, period_label)


@plot("figure")
def kde_overlay(kde, periods, colors, variable="tas", fig=None, spec=None, layout=None,
                ax=None, axes=None, **layout_kwargs):
    """One panel per period, all experiments overlaid.

    Same arguments as `kde_grid`. Kept separate rather than folded in as a mode
    flag: the two share no layout, no legend handling and no axis labelling.

    `ax`, `axes`, `fig`, `spec`, `layout` and `layout_kwargs` follow the
    `kde_grid` contract.

    Returns
    -------
    core.Panels
    """
    experiments = list(kde)

    panels = _new_grid(
        1, len(periods), fig=fig, spec=spec, layout=layout,
        ax=ax, axes=axes, **layout_kwargs,
    )
    for col, period in enumerate(periods):
        panel_ax = panels.axes[0, col]
        lines = []
        for exp in experiments:
            da = kde[exp][variable].sel(year=period)
            if "year" in da.dims:
                da = da.mean("year")
            lines += panel_ax.plot(da.x, da, color=colors[exp], label=exp)
        panels.artists[0, col] = lines
        panel_ax.set_title("")
        panel_ax.set_ylabel("")
        core.style_ax(panel_ax)
    _label_grid(panels, [None], periods, variable, period_label)
    panels.axes[0, 0].set_ylabel("Density")
    handles, labels = panels.axes[0, 0].get_legend_handles_labels()
    panels.extras["legend"] = panels.fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=len(experiments), frameon=False,
    )
    return panels


@plot("figure")
def hist_grid(point, periods, colors, variable="tas", n_bins=20, fig=None,
              spec=None, layout=None, ax=None, axes=None, **layout_kwargs):
    """One histogram per (experiment, period), on shared bins.

    `colors` is keyed by experiment, matching `kde_grid`. `ax`, `axes`, `fig`,
    `spec`, `layout` and `layout_kwargs` follow the `kde_grid` contract.

    Returns
    -------
    core.Panels
    """
    experiments = list(point)
    bins = hist_bins(point, periods, experiments, variable, n_bins)

    panels = _new_grid(
        len(experiments), len(periods), fig=fig, spec=spec, layout=layout,
        ax=ax, axes=axes, **layout_kwargs,
    )
    for row, exp in enumerate(experiments):
        for col, period in enumerate(periods):
            panel_ax = panels.axes[row, col]
            panels.artists[row, col] = point.sel(year=period)[exp][variable].plot.hist(
                ax=panel_ax, bins=bins, color=colors[exp], alpha=0.5
            )
            panel_ax.set_title("")
            panel_ax.set_xlabel("")
            panel_ax.set_ylabel("")
            panel_ax.set_xlim(bins[0], bins[-1])
            core.style_ax(panel_ax)
    _label_grid(panels, experiments, periods, variable, period_label)
    panels.extras["bins"] = bins
    return panels
