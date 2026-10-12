"""Figures of the optimal fingerprinting (notebook 08); the calculations are in ``extant.fingerprinting``.

    entry_tables          vectors as tables of regions x seasons: the observations, each fingerprint, the fit
    scaling_factor_plot   β and its 90% range for every signal set, OLS and TLS, with lines at 0 and 1
    attributable_trends   each signal's attributable trend (β x) per region and season, next to the observed
    model_betas           β with every model's own fingerprints, and leaving each model out
    correlation_maps      the fingerprints' correlation, and the β's across the Monte Carlo trials
    residual_null         the residual test: the observed statistic against its null distribution

Colours: each signal's forcing colour (``SIGNAL_COLORS``); black is ERA5.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS
from plotting_modules.utils import plot
from . import forced_response as fr

#(t): The colour of each signal: its forcing's, and for the combinations their nearest
SIGNAL_COLORS = {"ALL": FORCING_COLORS["historical"], "ANT": "#d95f02", "NAT": FORCING_COLORS["hist-nat"],
                 "GHG": FORCING_COLORS["hist-GHG"], "AER": FORCING_COLORS["hist-aer"],
                 "O3": FORCING_COLORS["hist-totalO3"], "OTH": "#5d6fb3"}

#(t): Markers of the two fits
METHOD_MARKERS = {"ols": "o", "tls": "D"}


def _color(signal):
    return SIGNAL_COLORS.get(str(signal), "0.3")


@plot("figure")
def entry_tables(tables, scale, row_dim="region", col_dim="season", style=fr.NOTEBOOK, title=None):
    """Vectors drawn back as tables of regions (rows) x seasons (columns), side by side, on one colour scale.

    Args:
        tables (dict): Panel title -> a DataArray on (row_dim, col_dim), e.g. ``fingerprinting.unflatten`` of y.
        scale (fr.Scale): The colour scale.
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    first = next(iter(tables.values()))
    rows, columns = list(first[row_dim].values), list(first[col_dim].values)
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(1, len(tables), figsize=(1.0 + 1.9 * len(tables), 0.45 * len(rows) + 1.6),
                                 sharey=True, squeeze=False, layout="constrained")
        image = None
        for ax, (label, table) in zip(axes[0], tables.items()):
            values = table.sel({row_dim: rows, col_dim: columns}).transpose(row_dim, col_dim).values
            image = ax.imshow(values, cmap=scale.cmap, vmin=scale.levels[0], vmax=scale.levels[-1], aspect="auto")
            for (r, c), value in np.ndenumerate(values):
                if np.isfinite(value):
                    ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=style.tick_size - 1)
            ax.set_xticks(range(len(columns)), columns)
            ax.set_yticks(range(len(rows)), rows)
            ax.set_title(label, loc="left", fontsize=style.label_size)
            ax.tick_params(length=0)
        bar = fig.colorbar(image, ax=list(axes[0]), location="bottom", shrink=0.6, aspect=40)
        bar.set_label(scale.label)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def scaling_factor_plot(betas, style=fr.NOTEBOOK, title=None):
    """β and its 90% range for every signal set (rows of signals) and fit (marker), with lines at 0 and 1.

    Clear of 0: the signal is detected. Across 1: the model's response has
    the observed size. Above 1: the models underestimate it; below, overestimate.

    Args:
        betas (xr.Dataset): ``beta``, ``lower``, ``upper`` on (set, fit, signal), NaN for a signal not in a set.
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    sets = list(betas["set"].values)
    methods = list(betas["fit"].values)
    with plt.rc_context(style.rc):
        heights = [int(betas["beta"].sel(set=name).notnull().any("fit").sum()) for name in sets]
        fig, axes = plt.subplots(len(sets), 1, figsize=(6.0, 0.42 * sum(heights) + 0.6 * len(sets) + 0.8),
                                 sharex=True, squeeze=False, layout="constrained",
                                 gridspec_kw={"height_ratios": [max(h, 1) for h in heights]})
        for ax, name in zip(axes[:, 0], sets):
            signals = [s for s in betas["signal"].values if bool(betas["beta"].sel(set=name, signal=s).notnull().any())]
            for y, signal in enumerate(signals):
                for k, method in enumerate(methods):
                    cell = betas.sel(set=name, signal=signal, fit=method)
                    offset = (k - (len(methods) - 1) / 2) * 0.25
                    ax.plot([float(cell["lower"]), float(cell["upper"])], [y + offset] * 2, color=_color(signal), lw=1.6)
                    ax.plot(float(cell["beta"]), y + offset, marker=METHOD_MARKERS.get(str(method), "o"), ms=5,
                            color=_color(signal), mec="white", mew=0.6)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.axvline(1, color="0.4", lw=0.8, ls="--")
            ax.set_yticks(range(len(signals)), signals)
            ax.set_ylim(-0.6, len(signals) - 0.4)
            ax.invert_yaxis()
            ax.set_title(str(name), loc="left", fontsize=style.label_size)
            core.style_ax(ax)
        axes[-1, 0].set_xlabel("Scaling factor β (90% range)")
        fig.legend(handles=[Line2D([], [], ls="none", marker=METHOD_MARKERS.get(str(m), "o"), color="0.3", label=str(m).upper())
                            for m in methods], loc="outside lower center", ncol=len(methods), frameon=False)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def attributable_trends(observed, attributable, units="°C", row_dim="region", col_dim="season", style=fr.NOTEBOOK,
                        title=None):
    """Each signal's attributable trend, β times its fingerprint, with its 90% range, beside the observed trend.

    Args:
        observed (xr.DataArray): ERA5's trends on (row_dim, col_dim).
        attributable (xr.Dataset): ``trend``, ``lower``, ``upper`` on (signal, row_dim, col_dim).
        units (str): Units of the trends (per decade).
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    rows, columns = list(observed[row_dim].values), list(observed[col_dim].values)
    signals = list(attributable["signal"].values)
    width = 0.8 / (len(signals) + 1)
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(len(rows), 1, figsize=(7.0, 1.8 * len(rows) + 0.9), sharex=True, squeeze=False,
                                 layout="constrained")
        for ax, row in zip(axes[:, 0], rows):
            for k, signal in enumerate(signals):
                cell = attributable.sel({"signal": signal, row_dim: row})
                x = np.arange(len(columns)) - 0.4 + (k + 0.5) * width
                values = cell["trend"].sel({col_dim: columns}).values
                low = values - cell["lower"].sel({col_dim: columns}).values
                high = cell["upper"].sel({col_dim: columns}).values - values
                ax.bar(x, values, width=width * 0.9, color=_color(signal), alpha=0.8, label=str(signal))
                ax.errorbar(x, values, yerr=[low, high], fmt="none", ecolor="0.25", lw=0.8)
            x = np.arange(len(columns)) - 0.4 + (len(signals) + 0.5) * width
            ax.plot(x, observed.sel({row_dim: row, col_dim: columns}).values, ls="none", marker="D", ms=5, color="k",
                    label="ERA5")
            ax.axhline(0, color="0.5", lw=0.7)
            ax.set_xticks(range(len(columns)), columns)
            ax.set_ylabel(f"{row}\n{units} per decade")
            core.style_ax(ax)
            ax.grid(axis="x", visible=False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def model_betas(beta, each_model, left_out, style=fr.NOTEBOOK, title=None):
    """β of the multi-model fingerprints (bar: the 90% range) beside β with each model's own (markers) and leaving each out (crosses).

    Args:
        beta (xr.Dataset): ``beta``, ``lower``, ``upper`` on (signal).
        each_model (xr.DataArray): β on (model, signal), each model's own fingerprints.
        left_out (xr.DataArray): β on (model, signal), the multi-model fingerprints without that model.
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    from .era5_evaluation import MODEL_MARKERS

    signals = list(beta["signal"].values)
    models = [str(m) for m in each_model["model"].values]
    with plt.rc_context(style.rc):
        fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(signals) + 1.4), layout="constrained")
        for y, signal in enumerate(signals):
            cell = beta.sel(signal=signal)
            ax.fill_betweenx([y - 0.3, y + 0.3], float(cell["lower"]), float(cell["upper"]), color=_color(signal),
                             alpha=0.25, lw=0)
            ax.plot([float(cell["beta"])] * 2, [y - 0.3, y + 0.3], color=_color(signal), lw=2)
            for k, model in enumerate(models):
                ax.plot(float(each_model.sel(model=model, signal=signal)), y - 0.12, ls="none",
                        marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], color=_color(signal), ms=5, mec="white", mew=0.4)
                ax.plot(float(left_out.sel(model=model, signal=signal)), y + 0.15, ls="none", marker="x",
                        color="0.25", ms=4)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.axvline(1, color="0.4", lw=0.8, ls="--")
        ax.set_yticks(range(len(signals)), signals)
        ax.invert_yaxis()
        ax.set_xlabel("Scaling factor β")
        core.style_ax(ax)
        handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], color="0.35", label=model)
                   for k, model in enumerate(models)]
        handles.append(Line2D([], [], ls="none", marker="x", color="0.25", label="leaving that model out"))
        fig.legend(handles=handles, loc="outside right center", frameon=False, title="Own fingerprints")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=np.array([[ax]]))


@plot("figure")
def correlation_maps(matrices, style=fr.NOTEBOOK, title=None):
    """Correlation matrices side by side, e.g. of the fingerprints and of the β's across the Monte Carlo trials.

    Above about 0.8 the two signals cannot be told apart: report their combined effect.

    Args:
        matrices (dict): Panel title -> a square DataArray of correlations, its two dims labelled alike.
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(1, len(matrices), figsize=(3.2 * len(matrices) + 0.8, 3.3), squeeze=False,
                                 layout="constrained")
        image = None
        for ax, (label, matrix) in zip(axes[0], matrices.items()):
            names = [str(v) for v in matrix[matrix.dims[0]].values]
            image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
            for (r, c), value in np.ndenumerate(matrix.values):
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=style.tick_size,
                        color="white" if abs(value) > 0.6 else "k")
            ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
            ax.set_yticks(range(len(names)), names)
            ax.set_title(label, loc="left", fontsize=style.label_size)
            ax.tick_params(length=0)
        fig.colorbar(image, ax=list(axes[0]), shrink=0.8, label="Correlation")
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def residual_null(tests, style=fr.NOTEBOOK, title=None):
    """The residual consistency test of each signal set: the null distribution, and the observed statistic.

    Args:
        tests (dict): Panel title -> ``fingerprinting.residual_test`` output.
        style (fr.Style): Fonts.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    with plt.rc_context(style.rc):
        fig, axes = plt.subplots(1, len(tests), figsize=(3.0 * len(tests) + 0.6, 2.6), squeeze=False,
                                 layout="constrained")
        for ax, (label, test) in zip(axes[0], tests.items()):
            #(c): TLS's null has a long tail (near-degenerate refits); the axis stops past the observed value and the
            #(c): null's 99th percentile, so the bulk stays visible
            end = 1.15 * max(test["statistic"], np.percentile(test["null"], 99))
            ax.hist(np.clip(test["null"], None, end), bins=np.linspace(0, end, 41), color="0.75")
            ax.axvline(test["statistic"], color="k", lw=1.8)
            ax.set_xlim(0, end)
            ax.set_title(f"{label}\np = {test['pvalue']:.3f}", loc="left", fontsize=style.label_size)
            ax.set_xlabel("rᵀ C₂⁻¹ r")
            core.style_ax(ax)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return core.Panels(fig=fig, axes=axes)
