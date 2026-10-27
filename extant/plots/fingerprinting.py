"""Panels of the optimal fingerprinting (notebook fingerprinting_02); the calculations are in ``extant.fingerprinting``.

Each ``plot_*(..., ax=None)`` styles one panel (``charts.style_chart_ax``) and draws on it (a new figure if ``ax``
is None); the notebook makes the figure and its axes (a GridSpec), draws each panel and labels them, as everywhere
(``charts``).

    plot_entry_table         a vector as a table of regions x seasons: the observations, a fingerprint, the fit
    plot_scaling_factors     β and its 90% range for one signal set, OLS and TLS, with lines at 0 and 1
    plot_attributable_trends each signal's attributable trend (β x) in one region, each season, next to the observed
    plot_model_betas         β with every model's own fingerprints, and leaving each model out
    plot_correlations        the correlation of the fingerprints, or of the β's across the Monte Carlo trials
    plot_residual_null       the residual test: the observed statistic against its null distribution

Colours: each signal's (``charts.SIGNAL_COLORS``); black is ERA5.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import charts
from .charts import MODEL_MARKERS, SIGNAL_COLORS

#(t): Markers of the two fits
METHOD_MARKERS = {"ols": "o", "tls": "D"}


def _color(signal):
    return SIGNAL_COLORS.get(str(signal), "0.3")


def plot_entry_table(table, scale, row_dim="region", col_dim="season", ax=None):
    """A vector drawn back as a table of regions (rows) x seasons (columns), each cell its value, on a colour scale.

    Args:
        table (xr.DataArray): On (row_dim, col_dim), e.g. ``fingerprinting.unflatten`` of y.
        scale (fr.Scale): The colour scale; the same for every table side by side, so one colour bar fits them all.
        row_dim, col_dim (str): The dims down and across.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.image.AxesImage: The cells, for a colour bar (``image.axes`` is the axis).
    """
    if ax is None:
        ax = charts.one_panel(panel_width=1.9)
    charts.style_chart_ax(ax)
    rows, columns = list(table[row_dim].values), list(table[col_dim].values)
    values = table.transpose(row_dim, col_dim).values
    image = ax.imshow(values, cmap=scale.cmap, vmin=scale.levels[0], vmax=scale.levels[-1], aspect="auto")
    for (r, c), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    return image


def plot_scaling_factors(set_ds, ax=None):
    """β and its 90% range for one signal set, every signal (rows, in its colour) and fit (marker), with lines at 0 and
    1 (dashed).

    Clear of 0: the signal is detected. Across 1: the model's response has the observed size. Above 1: the models
    underestimate it; below, overestimate.

    Args:
        set_ds (xr.Dataset): ``beta``, ``lower``, ``upper`` on (fit, signal) for one set; NaN for a signal not in it.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, the set's signals down its y axis.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=5.0)
    charts.style_chart_ax(ax)
    methods = list(set_ds["fit"].values)
    signals = [s for s in set_ds["signal"].values if bool(set_ds["beta"].sel(signal=s).notnull().any())]
    for y, signal in enumerate(signals):
        for k, method in enumerate(methods):
            cell = set_ds.sel(signal=signal, fit=method)
            offset = (k - (len(methods) - 1) / 2) * 0.25
            ax.plot([float(cell["lower"]), float(cell["upper"])], [y + offset] * 2, color=_color(signal), lw=1.6)
            ax.plot(float(cell["beta"]), y + offset, marker=METHOD_MARKERS.get(str(method), "o"), ms=5,
                    color=_color(signal), mec="white", mew=0.6)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.axvline(1, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(range(len(signals)), signals)
    ax.set_ylim(len(signals) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    return ax


def plot_attributable_trends(observed, attributable, col_dim="season", ax=None):
    """Each signal's attributable trend, β times its fingerprint, with its 90% range (bars), beside the observed trend
    (black diamond), each season, in one region.

    Args:
        observed (xr.DataArray): ERA5's trends in the region, on (col_dim).
        attributable (xr.Dataset): ``trend``, ``lower``, ``upper`` on (signal, col_dim) in the region.
        col_dim (str): The dim along the x axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis. Each signal's bars are labelled with it, ERA5's diamonds 'ERA5', so
        ``ax.get_legend_handles_labels()`` gives the legend.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=6.0)
    charts.style_chart_ax(ax)
    columns = list(observed[col_dim].values)
    signals = list(attributable["signal"].values)
    width = 0.8 / (len(signals) + 1)
    for k, signal in enumerate(signals):
        cell = attributable.sel(signal=signal)
        x = np.arange(len(columns)) - 0.4 + (k + 0.5) * width
        values = cell["trend"].sel({col_dim: columns}).values
        low = values - cell["lower"].sel({col_dim: columns}).values
        high = cell["upper"].sel({col_dim: columns}).values - values
        ax.bar(x, values, width=width * 0.9, color=_color(signal), alpha=0.8, label=str(signal))
        ax.errorbar(x, values, yerr=[low, high], fmt="none", ecolor="0.25", lw=0.8)
    x = np.arange(len(columns)) - 0.4 + (len(signals) + 0.5) * width
    ax.plot(x, observed.sel({col_dim: columns}).values, ls="none", marker="D", ms=5, color="k", label="ERA5")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(range(len(columns)), columns)
    ax.grid(axis="x", visible=False)
    return ax


def plot_model_betas(beta, each_model, left_out, ax=None):
    """β of the multi-model fingerprints (bar: the 90% range) beside β with each model's own (markers) and leaving each
    out (crosses).

    Args:
        beta (xr.Dataset): ``beta``, ``lower``, ``upper`` on (signal).
        each_model (xr.DataArray): β on (model, signal), each model's own fingerprints.
        left_out (xr.DataArray): β on (model, signal), the multi-model fingerprints without that model.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, the signals down its y axis. Model k's marker is ``charts.MODEL_MARKERS[k]``,
        in the k-th shade of its signal's colour (``charts.model_shades``).
    """
    if ax is None:
        ax = charts.one_panel(panel_width=5.5)
    charts.style_chart_ax(ax)
    signals = list(beta["signal"].values)
    models = [str(m) for m in each_model["model"].values]
    for y, signal in enumerate(signals):
        cell = beta.sel(signal=signal)
        ax.fill_betweenx([y - 0.3, y + 0.3], float(cell["lower"]), float(cell["upper"]), color=_color(signal),
                         alpha=0.25, lw=0)
        ax.plot([float(cell["beta"])] * 2, [y - 0.3, y + 0.3], color=_color(signal), lw=2)
        shades = charts.model_shades(_color(signal), len(models))
        for k, model in enumerate(models):
            ax.plot(float(each_model.sel(model=model, signal=signal)), y - 0.12, ls="none",
                    marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], color=shades[k], ms=5, mec="white", mew=0.4)
            ax.plot(float(left_out.sel(model=model, signal=signal)), y + 0.15, ls="none", marker="x",
                    color="0.25", ms=4)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.axvline(1, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(range(len(signals)), signals)
    ax.set_ylim(len(signals) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    return ax


def plot_correlations(matrix, ax=None):
    """A correlation matrix, each cell its value: above about 0.8 the two signals cannot be told apart, so report their
    combined effect.

    Args:
        matrix (xr.DataArray): A square DataArray of correlations, its two dims labelled alike.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.image.AxesImage: The cells, for a colour bar (``image.axes`` is the axis).
    """
    if ax is None:
        ax = charts.one_panel(panel_width=3.0, panel_height=3.0)
    charts.style_chart_ax(ax)
    names = [str(v) for v in matrix[matrix.dims[0]].values]
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for (r, c), value in np.ndenumerate(matrix.values):
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.6 else "k")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0)
    ax.grid(False)
    return image


def plot_residual_null(test, ax=None):
    """The residual consistency test of one signal set: the null distribution (grey), and the observed statistic.

    Args:
        test (dict): ``fingerprinting.residual_test`` output.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    #(c): TLS's null has a long tail (near-degenerate refits); the axis stops past the observed value and the null's
    #(c): 99th percentile, so the bulk stays visible
    end = 1.15 * max(test["statistic"], np.percentile(test["null"], 99))
    ax.hist(np.clip(test["null"], None, end), bins=np.linspace(0, end, 41), color="0.75")
    ax.axvline(test["statistic"], color="k", lw=1.8)
    ax.set_xlim(0, end)
    return ax
