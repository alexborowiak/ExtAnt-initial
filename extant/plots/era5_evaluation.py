"""Figures for the ERA5 evaluation; the calculations live in era5_evaluation.

Follows the plotting_modules conventions: ``draw_*`` functions draw onto an
axes you pass in (``@plot("axes")``); the others build a whole figure and
return ``core.Panels`` (``@plot("figure")``). Nothing here computes a test
statistic. Every function draws the output of an era5_evaluation function, so
the numbers on a figure are exactly the numbers the tests used. Needs
plotting_modules on the path (the notebooks add ../../extant-functions); it could
move into plotting_modules with only its imports changed.

Colour means the same thing in every figure:
    black           ERA5
    blue            the model ensemble (darker = more central, or pooled)
    orange          ERA5 outside the ensemble range
    grey, dashed    what a perfect model gives (the pseudo-observation range)
    green           hist-nat, when shown for context
    blue <-> red    ERA5 below <-> above the members (scorecard and maps)
    ✓ / ✗           consistent / flagged at the test's significance level

Sections
--------
1. Style and shared helpers
2. Distributions: KDE and Q-Q panels
3. Time series: the ensemble plume, with ERA5 inside or outside it
4. Moment test (and the other statistics): ERA5 among the members' values
5. Rank histograms (Suarez-Gutierrez et al., 2021)
6. Multi-model summaries: scorecard, signal-noise diagram, maps
7. Method demonstration on synthetic ensembles
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .. import era5_evaluation as ev
from plotting_modules import core
from plotting_modules.constants import FORCING_COLORS, SEASONS
from plotting_modules.utils import plot


# ---------------------------------------------------------------------------
# 1. Style and shared helpers
# ---------------------------------------------------------------------------

INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#a3a29d"        # reference lines
ERA5_COLOR = "#111111"
OUTSIDE_COLOR = "#eb6834"
CONTEXT_COLOR = FORCING_COLORS["hist-nat"]
PSEUDO_COLOR = "#3d3d3a"
STATUS_OK = "#0ca30c"
STATUS_FLAG = "#d03b3b"

#(t): One blue ramp for the ensemble, light (wide ranges) to dark (central values)
ENSEMBLE = {
    "range": "#cde2fb",    # full member range
    "outer": "#9ec5f4",    # 5-95%
    "member": "#86b6ef",   # individual members
    "inner": "#5598e7",    # central 75%, 12.5-87.5%
    "pooled": "#1c5cab",   # pooled ensemble
    "median": "#104281",   # ensemble median
}

#(t): Scorecard and map classes for ERA5's percentile among the members
PERCENTILE_BOUNDS = (0, 5, 25, 75, 95, 100)
PERCENTILE_COLORS = ("#1c5cab", "#9ec5f4", "#f0efec", "#f7aba4", "#a6272a")

#(t): Model identity in scatter plots: colour and marker, so identity never rests on colour alone
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

EVAL_RC = {
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#8a8985",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.fontsize": 8.5,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.solid_capstyle": "round",
}


def _styled(func):
    """Draw a figure function's artists under EVAL_RC."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(EVAL_RC):
            return func(*args, **kwargs)
    return wrapper


def order_seasons(obj, dim="season"):
    """Put a season dimension in calendar order (DJF, MAM, JJA, SON) rather than alphabetical."""
    if dim not in obj.dims:
        return obj
    return obj.sel({dim: [s for s in SEASONS if s in obj[dim].values]})


#(c): Class edges with the ends pushed just outside 0-100. Percentiles of exactly 0 or 100 (ERA5
#(c): beyond every member) are common, and contourf leaves holes where a flat region sits on a level.
PERCENTILE_EDGES = (-0.5, *PERCENTILE_BOUNDS[1:-1], 100.5)


def percentile_colormap():
    """Five-class diverging colormap for ERA5's percentile: significant low, low, central, high, significant high."""
    cmap = ListedColormap(PERCENTILE_COLORS).with_extremes(bad="white")
    return cmap, BoundaryNorm(PERCENTILE_EDGES, cmap.N)


def _ordinal(value):
    n = int(round(float(value)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _panel_label(ax, text, y=0.97):
    """Bold label in a panel's top-left corner, as in plotting_modules.timeseries.stack_grid."""
    ax.text(0.02, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2), zorder=20)


def _status_line(ax, x, y, ok, message, ha="left", fontsize=7.5):
    """A ✓ (consistent), ✗ (flagged) or – (untestable: ok is None) glyph, next to a message in ink."""
    glyph, colour = ("–", MUTED) if ok is None else ("✓", STATUS_OK) if ok else ("✗", STATUS_FLAG)
    step = fontsize * 1.25
    glyph_offset, text_offset = (0, step) if ha == "left" else (0, -step)
    kwargs = dict(xycoords="axes fraction", textcoords="offset points", va="top", zorder=20)
    artists = [
        ax.annotate(glyph, (x, y), xytext=(glyph_offset, 0), ha=ha, color=colour, fontsize=fontsize + 1.5,
                    fontweight="bold", **kwargs),
        ax.annotate(message, (x, y), xytext=(text_offset, 0), ha=ha, color=INK if not ok else INK_2,
                    fontsize=fontsize, **kwargs),
    ]
    #(c): Keep the text out of constrained layout, so a long verdict never shrinks the panel
    for artist in artists:
        artist.set_in_layout(False)


def where_era5(percentile, rank_test=False):
    """Plain-language position of ERA5's statistic, from its percentile.

    A moment or statistic test compares ERA5 with the members' values; a rank-histogram test
    compares ERA5's histogram with the perfect-model (pseudo-observation) cases.
    """
    value = float(percentile)
    if rank_test:
        if value <= 0 or value >= 100:
            return "beyond every perfect-model case"
        return f"{_ordinal(value)} pct of perfect-model cases"
    if value <= 0:
        return "ERA5 below every member"
    if value >= 100:
        return "ERA5 above every member"
    return f"ERA5 at {_ordinal(value)} pct"


def is_testable(row):
    """Whether a result could have been flagged at all (enough members); True for results without the flag."""
    return "testable" not in row or bool(row["testable"] == 1)


def verdict(result, statistic, short=False):
    """``(ok, message)`` summarising one statistic of a test result at a single point.

    ``ok`` is True (consistent), False (flagged) or None (too few members for
    the test to flag anything, so "not flagged" means nothing). ``short`` drops
    the percentile, for compact panels: "Rank dispersion: consistent", or just
    the reading (e.g. "spread too small (U shape)") if flagged.
    """
    row = result.sel(statistic=statistic)
    key = str(statistic).split(" (")[0]
    diagnostic = ev.DIAGNOSTICS[key]
    call = int(row["verdict"])
    if call == 0 and not is_testable(row):
        return None, "too few members to test" if short else f"{diagnostic.label}: too few members to test"
    reading = None if call == 0 else diagnostic.low if call < 0 else diagnostic.high
    if short:
        return call == 0, f"{diagnostic.label}: consistent" if call == 0 else reading
    where = where_era5(row["percentile"], rank_test=result.attrs.get("test") == "rank histogram")
    if call == 0:
        return True, f"{diagnostic.label}: {where}"
    return False, f"{diagnostic.label}: {reading} ({where})"


#(t): The rank-histogram verdicts that mean something for each treatment. Raw values are
#(c): dominated by any mean-state bias, which shows up in the mean rank, not the dispersion.
RANK_VERDICTS = {
    "raw": ("mean_rank", "outside"),
    "anomaly": ("dispersion", "outside", "rank_trend"),
    "detrended": ("dispersion", "outside"),
}


def _rank_verdicts(result, statistics=None):
    if statistics is not None:
        return statistics
    return RANK_VERDICTS[str(result["treatment"].values[0])]


def _legend(fig, handles, ncols=None, **kwargs):
    """Figure legend under the panels, where it cannot collide with the title."""
    return fig.legend(handles=handles, loc="outside lower center", ncols=ncols or len(handles),
                      frameon=False, handlelength=1.8, columnspacing=1.6, **kwargs)


def _suptitle(fig, title):
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")


def _era5_handle(marker="o"):
    return Line2D([], [], color=ERA5_COLOR, lw=1.6, marker=marker, ms=3, label="ERA5")


# ---------------------------------------------------------------------------
# 2. Distributions: KDE and Q-Q panels
# ---------------------------------------------------------------------------

@plot("axes")
def draw_kde(ax, summary, members=True):
    """Every member's KDE (thin), the pooled ensemble KDE and ERA5's KDE, with a rug of ERA5's values.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        summary (xr.Dataset): One treatment of ``ev.distribution_summary``,
            reduced to dims ``x``, ``member`` and ``year``.
        members (bool): Draw the individual member curves.
    """
    x = summary["x"].values
    member_kde = summary["member_kde"].transpose("member", "x").values
    if members:
        ax.plot(x, member_kde.T, color=ENSEMBLE["member"], lw=0.5, alpha=0.4, zorder=1)
    ax.plot(x, summary["ensemble_kde"], color=ENSEMBLE["pooled"], lw=2.0, zorder=3)
    ax.plot(x, summary["obs_kde"], color=ERA5_COLOR, lw=2.0, zorder=4)

    #(c): Zoom to where any curve carries weight, and leave a strip below zero for the rug
    curves = np.vstack([member_kde, summary["obs_kde"].values])
    peak = np.nanmax(curves)
    visible = x[np.nanmax(curves, axis=0) > 2e-3 * peak]
    ax.set_xlim(visible.min(), visible.max())
    ax.set_ylim(-0.1 * peak, 1.12 * peak)
    values = summary["obs"].values
    values = values[np.isfinite(values)]
    ax.plot(values, np.full(values.shape, -0.05 * peak), "|", color=ERA5_COLOR, ms=6, mew=0.8, alpha=0.7, zorder=5)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)


@plot("axes")
def draw_qq(ax, summary):
    """Quantile-quantile plot: members' quantiles (5-95% envelope and median) against ERA5's.

    On the 1:1 line the distributions match. A parallel offset is a mean bias;
    a line steeper than 1:1 means the model is more variable than ERA5, and
    shallower means less variable.
    """
    obs_q = summary["obs_quantiles"].values
    member_q = summary["member_quantiles"].transpose("member", "quantile").values
    low, mid, high = np.nanquantile(member_q, [0.05, 0.5, 0.95], axis=0)

    ax.fill_between(obs_q, low, high, color=ENSEMBLE["outer"], lw=0, zorder=1)
    ax.plot(obs_q, mid, color=ENSEMBLE["pooled"], lw=1.8, marker="o", ms=3, zorder=3)
    limits = np.array([np.nanmin([obs_q.min(), low.min()]), np.nanmax([obs_q.max(), high.max()])])
    limits += np.array([-1, 1]) * 0.05 * np.ptp(limits)
    ax.plot(limits, limits, color=INK_2, lw=0.9, zorder=2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)


DISTRIBUTION_COLUMNS = (("kde", "raw"), ("kde", "anomaly"), ("qq", "anomaly"))

#(t): The moment-test statistic reported on each KDE panel, by treatment
_KDE_STATISTIC = {"raw": "mean", "anomaly": "std", "detrended": "std"}


@plot("figure")
@_styled
def distribution_grid(summaries, season=None, columns=DISTRIBUTION_COLUMNS, moments=None, units="°C",
                      title=None):
    """KDE and Q-Q panels of ERA5 against the ensemble: do raw values fit? do anomalies?

    Rows are seasons for one model, or models for one season:

        distribution_grid(result["distributions"], moments=result["moments"])
        distribution_grid({m: r["distributions"] for m, r in results.items()}, season="DJF",
                          moments={m: r["moments"] for m, r in results.items()})

    Args:
        summaries (dict): ``{treatment: Dataset}`` (one model), or
            ``{model: {treatment: Dataset}}`` (several), from ``ev.distribution_summary``.
        season (str | None): The season shown when rows are models.
        columns (Sequence[tuple[str, str]]): ``(kind, treatment)`` per column, kind "kde" or "qq".
        moments (xr.Dataset | dict | None): ``ev.moment_test`` output, nested like
            ``summaries``; adds the mean (raw) or std (anomalies) verdict to each KDE panel.
        units (str): Units of the variable.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    one_model = isinstance(next(iter(summaries.values())), xr.Dataset)
    if one_model:
        seasons = [s for s in SEASONS if s in next(iter(summaries.values()))["season"].values]
        rows = [(s, summaries, {"season": s}, moments) for s in seasons]
    else:
        rows = [(model, summary, {"season": season}, (moments or {}).get(model))
                for model, summary in summaries.items()]

    fig, axes = plt.subplots(len(rows), len(columns), figsize=(3.2 * len(columns), 1.95 * len(rows)),
                             layout="constrained", squeeze=False)
    for i, (row_label, summary, sel, moment) in enumerate(rows):
        for j, (kind, treatment) in enumerate(columns):
            ax = axes[i, j]
            data = summary[treatment].sel(sel)
            if kind == "kde":
                draw_kde(ax, data)
                if moment is not None:
                    ok, message = verdict(moment.sel(sel), _KDE_STATISTIC[treatment])
                    _status_line(ax, 0.98, 0.97, ok, message, ha="right", fontsize=7)
            else:
                draw_qq(ax, data)
            core.style_ax(ax)
            ax.tick_params(axis="y", labelleft=kind == "qq")
            if i == 0:
                ax.set_title(f"{ev.TREATMENTS[treatment]}: {'KDE' if kind == 'kde' else 'Q–Q'}", loc="left")
            if i == len(rows) - 1:
                ax.set_xlabel(f"ERA5 quantile ({units})" if kind == "qq" else f"{ev.TREATMENTS[treatment]} ({units})")
            if kind == "qq" and j == len(columns) - 1:
                ax.set_ylabel(f"Model quantile ({units})")
        #(c): Row label in the margin, leaving the top of each panel for its verdict
        axes[i, 0].set_ylabel(str(row_label), fontsize=10, fontweight="bold")

    _legend(fig, [
        Line2D([], [], color=ENSEMBLE["member"], lw=1, label="Each member"),
        Line2D([], [], color=ENSEMBLE["pooled"], lw=2, label="Pooled ensemble"),
        Line2D([], [], color=ERA5_COLOR, lw=2, label="ERA5"),
        Line2D([], [], color=ERA5_COLOR, lw=0, marker="|", ms=7, label="ERA5 values"),
        Patch(color=ENSEMBLE["outer"], label="Members' 5–95% (Q–Q)"),
        Line2D([], [], color=INK_2, lw=0.9, label="1:1"),
    ])
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 3. Time series: the ensemble plume, with ERA5 inside or outside it
# ---------------------------------------------------------------------------

@plot("axes")
def draw_plume_panel(ax, plume, context=None, x_dim="year", note=True):
    """Ensemble plume through time with ERA5 on top, marking where ERA5 leaves the ensemble range.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        plume (xr.Dataset): ``ev.ensemble_plume`` output reduced to ``x_dim``.
        context (xr.Dataset | None): A second plume (e.g. hist-nat), drawn as outlines.
        x_dim (str): Dimension along the x axis.
        note (bool): Write how often ERA5 is outside the range, against chance.
    """
    x = plume[x_dim].values
    quantiles = plume["quantiles"]
    ax.fill_between(x, plume["minimum"], plume["maximum"], color=ENSEMBLE["range"], lw=0, zorder=1)
    ax.fill_between(x, quantiles.sel(quantile=0.05), quantiles.sel(quantile=0.95),
                    color=ENSEMBLE["outer"], lw=0, zorder=2)
    ax.fill_between(x, quantiles.sel(quantile=0.125), quantiles.sel(quantile=0.875),
                    color=ENSEMBLE["inner"], lw=0, zorder=3)
    ax.plot(x, quantiles.sel(quantile=0.5), color=ENSEMBLE["median"], lw=1.3, zorder=4)

    if context is not None:
        context_q = context["quantiles"]
        for level in (0.05, 0.95):
            ax.plot(x, context_q.sel(quantile=level), color=CONTEXT_COLOR, lw=1.0, zorder=5)
        ax.plot(x, context_q.sel(quantile=0.5), color=CONTEXT_COLOR, lw=1.2, ls=(0, (4, 2)), zorder=5)

    obs = plume["obs"].values
    ax.plot(x, obs, color=ERA5_COLOR, lw=1.2, marker="o", ms=2.4, zorder=6)
    for flag, marker in (("above", "^"), ("below", "v")):
        hit = plume[flag].values == 1
        ax.scatter(x[hit], obs[hit], marker=marker, s=42, color=OUTSIDE_COLOR, edgecolor="white",
                   linewidth=0.9, zorder=7)

    if note:
        n = int(plume["below"].count())
        outside = int((plume["below"] + plume["above"]).sum())
        expected = float(plume["expected_outside"])
        central = 100 * float(plume["central"].mean()) if "central" in plume else np.nan
        ax.text(0.99, 0.03, f"outside {outside}/{n} ({100 * outside / max(n, 1):.0f}%) · chance {expected:.0f}%"
                            f" · central 75%: {central:.0f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK_2, zorder=20,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))


def _dim_label(dim, value):
    return ev.TREATMENTS.get(str(value), str(value)) if dim == "treatment" else str(value)


@plot("figure")
@_styled
def plume_grid(plumes, row_dim="season", col_dim="treatment", context=None, units="°C", sharey=False,
               title=None):
    """A grid of ensemble plumes with ERA5, e.g. seasons x (raw, anomaly), or models x seasons.

    Args:
        plumes (xr.Dataset): ``ev.ensemble_plume`` output with ``row_dim`` and ``col_dim``
            (concatenate models along a ``model`` dim for a multi-model grid).
        row_dim, col_dim (str): Dimensions mapped to rows and columns.
        context (xr.Dataset | None): Plumes of a second experiment (e.g. hist-nat), drawn as
            outlines. Panels it has no data for (e.g. a treatment it lacks) are drawn without it.
        units (str): Units of the variable.
        sharey (bool | str): Passed to ``plt.subplots``; share when all panels use one scale.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    plumes = order_seasons(plumes)
    rows, cols = plumes[row_dim].values, plumes[col_dim].values
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.3 * len(cols), 1.75 * len(rows) + 0.6),
                             layout="constrained", squeeze=False, sharex=True, sharey=sharey)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            sel = {row_dim: row, col_dim: col}
            panel_context = None
            if context is not None:
                try:
                    panel_context = context.sel({d: v for d, v in sel.items() if d in context.dims})
                except KeyError:
                    panel_context = None
            draw_plume_panel(ax, plumes.sel(sel), panel_context)
            core.style_ax(ax)
            if i == 0:
                ax.set_title(_dim_label(col_dim, col), loc="left")
            if j == 0:
                _panel_label(ax, _dim_label(row_dim, row))
            if j == 0 or not sharey:
                is_anomaly = "anomaly" in (str(row), str(col))
                ax.set_ylabel(f"{'Anomaly' if is_anomaly else 'tas'} ({units})")
        axes[-1, 0].set_xlabel("Year")

    handles = [
        _era5_handle(),
        (Line2D([], [], ls="none", marker="^", ms=7, color=OUTSIDE_COLOR, mec="white"),
         Line2D([], [], ls="none", marker="v", ms=7, color=OUTSIDE_COLOR, mec="white")),
        Line2D([], [], color=ENSEMBLE["median"], lw=1.3),
        Patch(color=ENSEMBLE["inner"]),
        Patch(color=ENSEMBLE["outer"]),
        Patch(color=ENSEMBLE["range"]),
    ]
    labels = ["ERA5", "ERA5 above / below every member", "Ensemble median", "Central 75% (12.5–87.5%)", "5–95%",
              "Full range"]
    if context is not None:
        handles += [Line2D([], [], color=CONTEXT_COLOR, lw=1.2, ls=(0, (4, 2))), Line2D([], [], color=CONTEXT_COLOR, lw=1)]
        labels += ["hist-nat median", "hist-nat 5–95%"]
    fig.legend(handles, labels, loc="outside lower center", ncols=min(len(handles), 4 if len(cols) < 3 else 8),
               frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)}, columnspacing=1.4)
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 4. Moment test (and the other statistics): ERA5 among the members' values
# ---------------------------------------------------------------------------

@plot("axes")
def draw_member_strip(ax, result):
    """One statistic: each member as a dot, their central range shaded, ERA5 as a black line.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        result (xr.Dataset): ``ev.locate`` output (one statistic of a moment or statistic test) reduced to ``member``.
    """
    members = result["members"].values
    members = np.sort(members[np.isfinite(members)])
    era5 = float(result["era5"])
    #(c): Low-discrepancy vertical jitter, so the strip looks the same every time it is drawn
    jitter = ((np.arange(members.size) * 0.618034) % 1 - 0.5) * 0.62

    #(c): Everything sits below y = 0.55; the strip above is left clear for the verdict text
    ax.fill_between([float(result["lower"]), float(result["upper"])], -0.5, 0.55, color=ENSEMBLE["range"], lw=0,
                    zorder=0)
    ax.scatter(members, jitter, s=13, color=ENSEMBLE["inner"], edgecolor="white", linewidth=0.4, zorder=2)
    ax.plot([float(result["median"])] * 2, [-0.5, 0.55], color=ENSEMBLE["median"], lw=1.0, zorder=1)
    ax.plot([era5, era5], [-0.5, 0.55], color=ERA5_COLOR, lw=1.8, zorder=3)
    ax.plot(era5, 0.44, marker="D", color=ERA5_COLOR, ms=6, mec="white", mew=0.8, zorder=4)

    span = np.ptp(np.append(members, era5)) or 1.0
    ax.set_xlim(min(members.min(), era5) - 0.1 * span, max(members.max(), era5) + 0.1 * span)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)


@plot("figure")
@_styled
def statistic_grid(result, row_dim="season", statistics=None, title=None):
    """A moment or statistic test at one point: rows are seasons, columns are statistics.

    Each panel shows where ERA5 (black) falls among the members (dots). If the
    model were perfect, ERA5 would land outside the shaded range (the members'
    5-95%) only 10% of the time.

    Args:
        result (xr.Dataset): ``ev.moment_test`` or ``ev.statistic_test`` output at one point, with ``row_dim``.
        row_dim (str): Dimension mapped to rows.
        statistics (Sequence[str] | None): Subset and order of the columns.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    result = order_seasons(result)
    statistics = list(statistics or result["statistic"].values)
    rows = result[row_dim].values
    fig, axes = plt.subplots(len(rows), len(statistics), figsize=(2.35 * len(statistics), 1.3 * len(rows) + 0.7),
                             layout="constrained", squeeze=False)
    for i, row in enumerate(rows):
        for j, statistic in enumerate(statistics):
            ax = axes[i, j]
            cell = result.sel({row_dim: row, "statistic": statistic})
            draw_member_strip(ax, cell)
            #(c): Short text so it fits the panel: where ERA5 sits if consistent, the reading if flagged
            ok, reading = verdict(result.sel({row_dim: row}), statistic, short=True)
            _status_line(ax, 0.01, 0.99, ok, where_era5(cell["percentile"]) if ok else reading, fontsize=7)
            ax.grid(axis="x", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            if i == 0:
                diagnostic = ev.DIAGNOSTICS[statistic]
                ax.set_title(diagnostic.label, loc="left", pad=13)
                ax.annotate(f"of {ev.TREATMENTS[diagnostic.treatment].lower()}", (0, 1), xycoords="axes fraction",
                            xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_2, va="bottom")
            if i == len(rows) - 1:
                units = ev.DIAGNOSTICS[statistic].units
                ax.set_xlabel(units if units else "(unitless)")
        axes[i, 0].set_ylabel(str(row), rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")

    _legend(fig, [
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["inner"], mec="white", label="Each member"),
        Patch(color=ENSEMBLE["range"], label="Members' 5–95%"),
        Line2D([], [], color=ENSEMBLE["median"], lw=1, label="Member median"),
        Line2D([], [], color=ERA5_COLOR, lw=1.8, marker="D", ms=5, mec="white", label="ERA5"),
    ])
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 5. Rank histograms (Suarez-Gutierrez et al., 2021)
# ---------------------------------------------------------------------------

RANK_BAR = "#b7d3f6"
RANK_LINE = "#2a78d6"


@plot("axes")
def draw_rank_histogram(ax, result, smooth=5, statistics=None):
    """ERA5's rank histogram against a perfect model's, in the style of Suarez-Gutierrez et al. (2021).

    Bars are ERA5's rank frequencies and the solid line their running mean.
    The dashed lines bound the running means of the pseudo-observations: the
    range a perfect model's histogram falls in, given this ensemble and record.
    The lowest and highest ranks (ERA5 outside the ensemble) are drawn
    separately: crosses for ERA5 and dash-dot bars for the perfect-model range.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        result (xr.Dataset): ``ev.rank_histogram_test`` output at one point (with histograms).
        smooth (int): Running-mean window, in ranks.
        statistics (Sequence[str] | None): Keys of ``ev.RANKS`` to print as verdicts;
            by default the ones that mean something for the result's treatment (``RANK_VERDICTS``).
    """
    statistics = _rank_verdicts(result, statistics)
    alpha = result.attrs["alpha"]
    frequency = result["obs_frequency"]
    pseudo = result["pseudo_frequency"]
    ranks = frequency["rank"].values
    n_bins = ranks.size
    levels = [alpha / 2, 1 - alpha / 2]

    ax.bar(ranks[1:-1], frequency.values[1:-1], width=0.8, color=RANK_BAR, lw=0, zorder=1)
    ax.axhline(100 / n_bins, color=MUTED, lw=0.9, zorder=0)

    smooth_obs = ev.running_mean(frequency, smooth)
    smooth_band = ev.running_mean(pseudo, smooth).quantile(levels, dim="member")
    ax.plot(smooth_obs["rank"], smooth_obs, color=RANK_LINE, lw=2.4, zorder=4)
    for level in levels:
        band = smooth_band.sel(quantile=level)
        ax.plot(band["rank"], band, color=PSEUDO_COLOR, lw=1.3, ls=(0, (4, 2.5)), zorder=3)

    for rank in (ranks[0], ranks[-1]):
        low, high = pseudo.sel(rank=rank).quantile(levels, dim="member").values
        ax.plot([rank, rank], [low, high], color=PSEUDO_COLOR, lw=1.3, ls="-.", zorder=3)
        ax.plot([rank, rank], [low, high], ls="none", marker="x", ms=6, mew=1.6, color=PSEUDO_COLOR, zorder=3)
        ax.plot(rank, float(frequency.sel(rank=rank)), marker="X", ms=12, color=RANK_LINE, mec="white",
                mew=0.9, zorder=5)

    top = max(float(frequency.max()), float(pseudo.sel(rank=[ranks[0], ranks[-1]]).quantile(levels[1], "member").max()),
              float(smooth_band.max()))
    ax.set_ylim(0, top * (1.12 + 0.13 * len(statistics)))
    ax.set_xlim(ranks[0] - 1.5, ranks[-1] + 1.5)
    ax.set_xlabel(f"Rank of ERA5 among {n_bins - 1} members")
    ax.set_ylabel("Frequency (%)")
    for k, statistic in enumerate(statistics):
        ok, message = verdict(result, statistic)
        _status_line(ax, 0.02, 0.98 - 0.105 * k, ok, message)


def _rank_legend_handles():
    return [
        Patch(color=RANK_BAR, label="ERA5 rank frequency"),
        Line2D([], [], color=RANK_LINE, lw=2.4, label="Running mean"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, ls=(0, (4, 2.5)), label="Perfect-model 5–95%"),
        Line2D([], [], ls="none", marker="X", ms=10, color=RANK_LINE, mec="white", label="ERA5 outside ensemble"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, ls="-.", marker="x", ms=5, label="Perfect-model range, outside"),
        Line2D([], [], color=MUTED, lw=0.9, label="Flat (infinite record)"),
    ]


@plot("figure")
@_styled
def rank_histogram_row(results, smooth=5, title=None):
    """Suarez-Gutierrez et al. style rank histograms side by side, e.g. raw / anomalies / detrended.

    Args:
        results (dict[str, xr.Dataset]): Panel title -> ``ev.rank_histogram_test`` output at one point.
        smooth (int): Running-mean window, in ranks.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.9), layout="constrained",
                             squeeze=False)
    for ax, (label, result) in zip(axes[0], results.items()):
        draw_rank_histogram(ax, result, smooth=smooth)
        core.style_ax(ax)
        ax.set_title(label, loc="left")
    _legend(fig, _rank_legend_handles(), ncols=3)
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


def _anatomy_years(result):
    """Three instructive years: ERA5's lowest, most central and highest rank."""
    rank = result["obs_rank"].dropna("year")
    years = [int(rank.idxmin()), int(np.abs(rank - 0.5).idxmin()), int(rank.idxmax())]
    return sorted(dict.fromkeys(years))


@plot("figure")
@_styled
def rank_anatomy(ensemble, obs, result, years=None, units="°C", title=None):
    """How the rank-histogram test works, step by step, for one season at one point.

    a) ERA5 and the members through time. b) In three years, count the
    members below ERA5: that count is ERA5's rank. c) ERA5's rank through the
    record; a drift means the signal differs. d) The ranks from every year,
    counted into a histogram and compared with what a perfect model gives.

    Args:
        ensemble (xr.DataArray): Treated members on (member, year), as passed to the test.
        obs (xr.DataArray): Treated ERA5 on (year,).
        result (xr.Dataset): ``ev.rank_histogram_test`` output for the same season and point.
        years (Sequence[int] | None): Years to dissect in panel b; three instructive ones if None.
        units (str): Units of the variable.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    years = years or _anatomy_years(result)
    n_members = ensemble.sizes["member"]
    fig = plt.figure(figsize=(12.5, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.9, 1), height_ratios=(1, 1))
    ax_series = fig.add_subplot(grid[0, 0])
    ax_strips = fig.add_subplot(grid[0, 1], sharey=ax_series)
    ax_rank = fig.add_subplot(grid[1, 0])
    ax_hist = fig.add_subplot(grid[1, 1])

    #(t): a) the members and ERA5
    x = ensemble["year"].values
    ax_series.plot(x, ensemble.transpose("year", "member").values, color=ENSEMBLE["member"], lw=0.6, alpha=0.55,
                   zorder=1)
    ax_series.plot(x, obs.values, color=ERA5_COLOR, lw=1.6, marker="o", ms=2.8, zorder=3)
    for year in years:
        ax_series.axvspan(year - 0.45, year + 0.45, color="#e7e6e2", zorder=0)
    ax_series.set_ylabel(f"{ev.TREATMENTS[str(result['treatment'].values[0])]} ({units})")
    ax_series.set_title("a) ERA5 (black) and every member (blue)", loc="left")

    #(t): b) ranking ERA5 in each highlighted year
    for k, year in enumerate(years):
        values = np.sort(ensemble.sel(year=year).values)
        value = float(obs.sel(year=year))
        below = values < value
        offsets = ((np.arange(values.size) * 0.618034) % 1 - 0.5) * 0.5
        ax_strips.scatter(k + offsets[below], values[below], s=16, color=ENSEMBLE["pooled"], edgecolor="white",
                          linewidth=0.4, zorder=2)
        ax_strips.scatter(k + offsets[~below], values[~below], s=16, color=ENSEMBLE["outer"], edgecolor="white",
                          linewidth=0.4, zorder=2)
        ax_strips.plot([k - 0.34, k + 0.34], [value, value], color=ERA5_COLOR, lw=1.8, zorder=3)
        ax_strips.plot(k, value, marker="D", color=ERA5_COLOR, ms=7, mec="white", zorder=4)
        ax_strips.annotate(f"rank {below.sum()} of {n_members}", (k, 1.0), xycoords=("data", "axes fraction"),
                           ha="center", va="top", fontsize=8, color=INK_2)
    ax_strips.set_xticks(range(len(years)), [str(y) for y in years])
    ax_strips.set_xlim(-0.6, len(years) - 0.4)
    ax_strips.tick_params(axis="y", labelleft=False)
    ax_strips.set_title("b) Rank = members below ERA5", loc="left")
    ax_strips.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["pooled"], label="below ERA5"),
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["outer"], label="above ERA5"),
    ], loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.2)

    #(t): c) ERA5's rank through time, and its trend
    rank = result["obs_rank"]
    _, fitted = ev.linear_fit(rank, "year")
    ax_rank.axhspan(-0.04, 1 / n_members, color="#fbe3d9", lw=0, zorder=0)
    ax_rank.axhspan(1 - 1 / n_members, 1.04, color="#fbe3d9", lw=0, zorder=0)
    ax_rank.plot(rank["year"], rank, color=ERA5_COLOR, lw=0.8, marker="o", ms=4, zorder=3)
    ax_rank.plot(rank["year"], fitted, color=RANK_LINE, lw=2.0, zorder=2)
    for year in years:
        ax_rank.axvspan(year - 0.45, year + 0.45, color="#e7e6e2", zorder=0)
    ax_rank.set_ylim(-0.04, 1.04)
    ax_rank.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0\nbelow all", "0.25", "0.5", "0.75", "1\nabove all"])
    ax_rank.set_ylabel("ERA5 rank / N")
    ax_rank.set_xlabel("Year")
    ax_rank.set_title("c) ERA5's rank through time (blue: trend; shaded: outside the ensemble)", loc="left")
    ok, message = verdict(result, "rank_trend")
    _status_line(ax_rank, 0.01, 0.97, ok, message)

    #(t): d) the ranks from every year, counted into a histogram
    #(c): ~35 ranks spread over N bins is sparse, so group them into 10 bins of rank / N
    draw_grouped_rank_histogram(ax_hist, result, n_bins=10, statistics=("dispersion", "outside"))
    ax_hist.set_ylabel("Frequency (%)")
    ax_hist.set_xlabel("Rank of ERA5 within the ensemble")
    ax_hist.set_title("d) All the ranks: histogram vs a perfect model", loc="left")

    for ax in (ax_series, ax_strips, ax_rank, ax_hist):
        core.style_ax(ax)
    _legend(fig, [
        Line2D([], [], color=ENSEMBLE["member"], lw=1, label="Each member"),
        _era5_handle(),
        Line2D([], [], color=RANK_LINE, lw=2, label="Trend of ERA5's rank"),
        Patch(color="#fbe3d9", label="ERA5 outside the ensemble"),
        Patch(color=RANK_BAR, label="ERA5 rank frequency"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, marker="_", ms=7, label="Perfect-model 5–95%"),
    ])
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=np.array([[ax_series, ax_strips], [ax_rank, ax_hist]], dtype=object))


@plot("axes")
def draw_grouped_rank_histogram(ax, result, n_bins=10, statistics=None):
    """A compact rank histogram on ``n_bins`` bins of rank / N, so ensembles of any size share an axis.

    Bars are ERA5; the grey bars span the perfect-model 5-95% range of each bin.
    ``statistics`` are the verdicts printed (by default ``RANK_VERDICTS`` for the treatment, less the trend).
    """
    if statistics is None:
        statistics = [s for s in _rank_verdicts(result) if s != "rank_trend"]
    alpha = result.attrs["alpha"]
    frequency = ev.regroup_ranks(result["obs_frequency"], n_bins)
    band = ev.regroup_ranks(result["pseudo_frequency"], n_bins).quantile([alpha / 2, 1 - alpha / 2], "member")
    centres = frequency["rank"].values

    ax.bar(centres, frequency, width=0.8 / n_bins, color=RANK_BAR, lw=0, zorder=1)
    low, high = band.values
    ax.vlines(centres, low, high, color=PSEUDO_COLOR, lw=1.3, zorder=3)
    ax.plot(np.repeat(centres, 2), np.column_stack([low, high]).ravel(), ls="none", marker="_", ms=7,
            mew=1.3, color=PSEUDO_COLOR, zorder=3)
    ax.axhline(100 / n_bins, color=MUTED, lw=0.9, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["lowest", "middle", "highest"])
    ax.set_ylim(0, max(float(frequency.max()), float(high.max())) * (1.1 + 0.16 * len(statistics)))
    for k, statistic in enumerate(statistics):
        ok, message = verdict(result, statistic, short=True)
        _status_line(ax, 0.02, 0.98 - 0.13 * k, ok, message, fontsize=7)


@plot("figure")
@_styled
def rank_histogram_grid(results, n_bins=10, title=None):
    """Compact rank histograms, rows x columns, e.g. models x (raw, anomaly, detrended).

    Args:
        results (dict[str, dict[str, xr.Dataset]]): Row label -> column label ->
            ``ev.rank_histogram_test`` output at one point (with histograms).
        n_bins (int): Bins of rank / N.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    row_labels = list(results)
    col_labels = list(next(iter(results.values())))
    fig, axes = plt.subplots(len(row_labels), len(col_labels), figsize=(3.3 * len(col_labels), 1.75 * len(row_labels) + 0.8),
                             layout="constrained", squeeze=False, sharex=True)
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            ax = axes[i, j]
            result = results[row][col]
            draw_grouped_rank_histogram(ax, result, n_bins)
            core.style_ax(ax)
            if i == 0:
                ax.set_title(col, loc="left")
            if j == 0:
                ax.set_ylabel(f"{row}\n(N = {result.attrs['n_members']})", fontsize=8.5, fontweight="bold")
        axes[-1, 0].set_xlabel("Rank of ERA5 within the ensemble")
    _legend(fig, [
        Patch(color=RANK_BAR, label="ERA5 rank frequency (%)"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, marker="_", ms=7, label="Perfect-model 5–95%"),
        Line2D([], [], color=MUTED, lw=0.9, label="Flat"),
    ])
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 6. Multi-model summaries: scorecard, signal-noise diagram, maps
# ---------------------------------------------------------------------------

def _wrap(text, width=16):
    """Break a column label onto lines: before a parenthesis if it has one, else at ``width`` characters."""
    if " (" in str(text):
        head, tail = str(text).split(" (", 1)
        return f"{head}\n({tail}"
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join(lines + [line])


@plot("figure")
@_styled
def scorecard(table, statistics=None, title=None):
    """Every model, statistic and season at one point: where ERA5 falls among the members.

    Colour is ERA5's percentile among the members (blue: ERA5 below most
    members; red: above). The darkest classes are beyond the members' 5-95%
    range, and outlined cells are significant at the test's alpha.

    Args:
        table (xr.Dataset): ``ev.summary_table`` output at one point.
        statistics (Sequence[str] | None): Subset and order of the column groups.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    table = order_seasons(table)
    statistics = list(statistics or table["statistic"].values)
    table = table.sel(statistic=statistics)
    models, seasons = list(table["model"].values), list(table["season"].values)
    n_groups, n_seasons = len(statistics), len(seasons)

    percentile = table["percentile"].transpose("model", "statistic", "season").values.reshape(len(models), -1)
    flagged = (table["verdict"].transpose("model", "statistic", "season").values != 0).reshape(len(models), -1)
    untestable = np.zeros_like(flagged)
    if "testable" in table:
        untestable = (table["testable"].transpose("model", "statistic", "season").values == 0).reshape(len(models), -1)
    cmap, norm = percentile_colormap()

    fig, ax = plt.subplots(figsize=(0.44 * percentile.shape[1] + 2.6, 0.46 * len(models) + 2.2), layout="constrained")
    image = ax.imshow(np.ma.masked_invalid(percentile), cmap=cmap, norm=norm, aspect="auto")
    for (i, j), value in np.ndenumerate(percentile):
        if not np.isfinite(value):
            continue
        dark = value < PERCENTILE_BOUNDS[1] or value > PERCENTILE_BOUNDS[-2]
        colour = "white" if dark else INK
        if untestable[i, j]:
            colour = "#e4e3df" if dark else MUTED
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.5, color=colour,
                fontweight="bold" if flagged[i, j] else "normal", fontstyle="italic" if untestable[i, j] else "normal")
        if flagged[i, j]:
            ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=INK, lw=1.3, zorder=3))
    for group in range(1, n_groups):
        ax.axvline(group * n_seasons - 0.5, color="white", lw=5)

    ax.set_xticks(range(percentile.shape[1]), seasons * n_groups, fontsize=7)
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    top = ax.secondary_xaxis("top")
    top.set_xticks([g * n_seasons + (n_seasons - 1) / 2 for g in range(n_groups)],
                   [_wrap(label) for label in table["label"].values], fontsize=8, fontweight="bold")
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)

    bar = fig.colorbar(image, ax=ax, ticks=PERCENTILE_BOUNDS, shrink=0.85, pad=0.01, aspect=25)
    bar.set_label("ERA5 percentile among the members")
    bar.ax.tick_params(labelsize=7.5)
    alpha = table.attrs.get("alpha", ev.ALPHA)
    note = (f"Outlined: significant at p < {alpha:g} (ERA5 outside the members' "
            f"{100 * alpha / 2:g}–{100 - 100 * alpha / 2:g}% range)")
    if untestable.any():
        note += f".  Faint italic: too few members (N < {int(np.ceil(2 / alpha))}) for the test to flag anything"
    ax.set_xlabel(note, color=INK_2, fontsize=8)
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=np.array([[ax]], dtype=object))


@plot("figure")
@_styled
def signal_noise_diagram(table, x="trend", y="std", title=None):
    """Signal against noise for every model: ERA5's percentile for the trend (x) and the variability (y).

    Inside the grey square, ERA5 is consistent with the model on both. Left of
    it the model's trend is too large, right of it too small; below it the
    model is too variable, above it not variable enough.

    Args:
        table (xr.Dataset): ``ev.summary_table`` output at one point.
        x, y (str): Statistics on each axis.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    table = order_seasons(table)
    seasons = list(table["season"].values)
    models = list(table["model"].values)
    alpha = table.attrs.get("alpha", ev.ALPHA)
    low, high = 100 * alpha / 2, 100 - 100 * alpha / 2

    fig, axes = plt.subplots(1, len(seasons), figsize=(3.4 * len(seasons), 3.9), layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    for ax, season in zip(axes[0], seasons):
        ax.add_patch(Rectangle((low, low), high - low, high - low, color="#f0efec", lw=0, zorder=0))
        ax.axhline(50, color="#dcdbd7", lw=0.8, zorder=0)
        ax.axvline(50, color="#dcdbd7", lw=0.8, zorder=0)
        for k, model in enumerate(models):
            point = table["percentile"].sel(model=model, season=season)
            ax.scatter(float(point.sel(statistic=x)), float(point.sel(statistic=y)), s=58,
                       marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], color=MODEL_COLORS[k % len(MODEL_COLORS)],
                       edgecolor="white", linewidth=1.1, zorder=3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(season, loc="left")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)

    def reading(text):
        return text.removeprefix("model ")

    x_diag, y_diag = ev.DIAGNOSTICS[x], ev.DIAGNOSTICS[y]
    fig.supxlabel(f"ERA5 percentile of the {x_diag.label.lower()}\n"
                  f"← {reading(x_diag.low)}   ·   {reading(x_diag.high)} →", fontsize=9)
    fig.supylabel(f"ERA5 percentile of the {y_diag.label.lower()}\n"
                  f"← {reading(y_diag.low)}   ·   {reading(y_diag.high)} →", fontsize=9)
    handles = [Line2D([], [], ls="none", marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], ms=7,
                      color=MODEL_COLORS[k % len(MODEL_COLORS)], mec="white", label=model)
               for k, model in enumerate(models)]
    handles.append(Patch(color="#f0efec", label=f"Consistent on both\n({low:g}–{high:g}%)"))
    fig.legend(handles=handles, loc="outside right upper", frameon=False, title="Model", alignment="left")
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


@plot("figure")
def percentile_maps(table, statistic, row_dim="model", col_dim="season", title=None, **grid_kwargs):
    """Maps of ERA5's percentile among the members for one statistic (e.g. models x seasons).

    Where ``table`` carries ``ev.field_test`` output, each panel also says
    whether the model is acceptable over the whole map: ✓ if ERA5 is flagged
    over no more area than a perfect model's pseudo-observations would be, ✗
    if over more (p < alpha). Models with too few members to flag anything
    are left blank and labelled.

    Args:
        table (xr.Dataset): Test output on a lat/lon grid with ``row_dim`` and ``col_dim``,
            optionally merged with ``ev.field_test`` output.
        statistic (str): Which statistic to map.
        row_dim, col_dim (str): Dimensions mapped to rows and columns.
        title (str | None): Figure title.
        **grid_kwargs: Passed to ``maps.polar_grid``.

    Returns:
        core.Panels
    """
    from plotting_modules import maps  # needs cartopy

    cmap, norm = percentile_colormap()
    table = order_seasons(table.sel(statistic=statistic))
    field = table["percentile"].drop_vars(["statistic", "label", "units", "treatment"], errors="ignore")
    #(c): Where too few members can flag nothing, a dark class would mean nothing: leave it blank
    if "testable" in table:
        field = field.where(table["testable"] == 1)
    label = str(table["label"].values)
    cbar_label = "ERA5 percentile among the members (blue: ERA5 below; red: above; darkest: outside 5–95%)"
    if "field_verdict" in table:
        alpha = table.attrs.get("alpha", ev.ALPHA)
        cbar_label += (f"\nField test, top left: % of the map where ERA5 is flagged, and the most a perfect model "
                       f"gives ({100 - 100 * alpha:g}% of the time). ✗: more than that (p < {alpha:g})")
    #(c): Room for model names as row labels
    grid_kwargs.setdefault("left", 2.0)
    with plt.rc_context(EVAL_RC):
        panels = maps.polar_grid(
            field, row_dim=row_dim, col_dim=col_dim, levels=np.array(PERCENTILE_EDGES), cmap=cmap, norm=norm,
            title=title or f"ERA5 percentile of the {label.lower()} among each model's members",
            cbar_label=cbar_label,
            tag=False, **grid_kwargs,
        )
        panels.extras["cbar"].set_ticks(PERCENTILE_BOUNDS)

        rows, cols = field[row_dim].values, field[col_dim].values
        axes = np.asarray(panels.axes, dtype=object).reshape(len(rows), len(cols))
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                cell = table.sel({row_dim: row, col_dim: col})
                if bool(field.sel({row_dim: row, col_dim: col}).isnull().all()) and "testable" in table:
                    axes[i, j].text(0.5, 0.5, "too few members\nto test (N < 20)", transform=axes[i, j].transAxes,
                                    ha="center", va="center", fontsize=8, color=INK_2, zorder=20,
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3))
                elif "field_verdict" in cell and bool(cell["field_verdict"].notnull()):
                    _status_line(axes[i, j], 0.0, 1.0, int(cell["field_verdict"]) == 0,
                                 f"{float(cell['flagged_area']):.1f}% (perfect ≤ {float(cell['perfect_area']):.1f}%)",
                                 fontsize=7)
    return panels


# ---------------------------------------------------------------------------
# 7. Method demonstration on synthetic ensembles
# ---------------------------------------------------------------------------

DEMO_STATISTICS = ("mean", "trend", "std", "width")


@plot("figure")
@_styled
def scenario_grid(demo, season="DJF", descriptions=None, n_bins=10, title=None):
    """The tests on toy ensembles where the right answer is known, one scenario per row.

    Columns: the raw time series with the ensemble plume; the rank histogram
    of anomalies and of detrended anomalies; and ERA5's percentile for the
    mean, trend, std and width and the two rank dispersions.

    Args:
        demo (dict[str, dict]): Scenario name -> ``ev.evaluate`` output.
        season (str): Season shown in the time series and the percentile column.
        descriptions (dict[str, str] | None): One line per scenario saying what is wrong;
            ``ev.SCENARIO_DESCRIPTIONS`` by default.
        n_bins (int): Bins in the compact rank histograms.
        title (str | None): Figure title.

    Returns:
        core.Panels
    """
    descriptions = ev.SCENARIO_DESCRIPTIONS if descriptions is None else descriptions
    names = list(demo)
    fig, axes = plt.subplots(len(names), 4, figsize=(15.5, 2.3 * len(names) + 0.8), layout="constrained",
                             squeeze=False, gridspec_kw={"width_ratios": (1.6, 1, 1, 1.25)})
    #(t): The ladder: (label, where to find the result) for each statistic shown
    ladder = [(ev.DIAGNOSTICS[s].label, "moments" if s in ev.MOMENTS else "statistics", s)
              for s in DEMO_STATISTICS] + [
        ("Rank dispersion (anomalies)", "anomaly", "dispersion"),
        ("Rank dispersion (detrended)", "detrended", "dispersion"),
        ("Rank trend (anomalies)", "anomaly", "rank_trend"),
    ]
    for i, name in enumerate(names):
        result = demo[name]
        ax_series, ax_anomaly, ax_detrended, ax_ladder = axes[i]

        draw_plume_panel(ax_series, result["plumes"].sel(treatment="raw", season=season))
        _panel_label(ax_series, name)
        if name in descriptions:
            ax_series.text(0.02, 0.8, descriptions[name], transform=ax_series.transAxes, fontsize=7.5,
                           color=INK_2, va="top", zorder=20,
                           bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        draw_grouped_rank_histogram(ax_anomaly, result["ranks"]["anomaly"], n_bins, statistics=("dispersion",))
        draw_grouped_rank_histogram(ax_detrended, result["ranks"]["detrended"], n_bins, statistics=("dispersion",))

        #(t): Ladder of ERA5 percentiles: dots in the grey band are consistent
        alpha = result["moments"].attrs["alpha"]
        ax_ladder.axvspan(100 * alpha / 2, 100 - 100 * alpha / 2, color="#f0efec", lw=0, zorder=0)
        ax_ladder.axvline(50, color="#dcdbd7", lw=0.8, zorder=0)
        for k, (_, source, statistic) in enumerate(ladder):
            if source in ("moments", "statistics"):
                row = result[source].sel(statistic=statistic, season=season)
            else:
                row = result["ranks"][source].sel(statistic=statistic)
            flagged = int(row["verdict"]) != 0
            ax_ladder.plot(float(row["percentile"]), k, marker="X" if flagged else "o", ms=8 if flagged else 6.5,
                           color=STATUS_FLAG if flagged else STATUS_OK, mec="white", mew=0.8, zorder=3)
        ax_ladder.set_yticks(range(len(ladder)), [label for label, _, _ in ladder], fontsize=7.5)
        ax_ladder.set_ylim(len(ladder) - 0.5, -0.5)
        ax_ladder.set_xlim(-4, 104)
        ax_ladder.set_xticks([0, 25, 50, 75, 100])

        for ax in axes[i]:
            core.style_ax(ax)
        if i == 0:
            ax_series.set_title(f"Raw values, {season}: ensemble plume and ERA5", loc="left")
            ax_anomaly.set_title("Rank histogram: anomalies", loc="left")
            ax_detrended.set_title("Rank histogram: detrended", loc="left")
            ax_ladder.set_title("ERA5 percentile among members", loc="left")

    axes[-1, 0].set_xlabel("Year")
    for ax in axes[-1, 1:3]:
        ax.set_xlabel("Rank of ERA5 within the ensemble")
    axes[-1, 3].set_xlabel("Percentile")
    _legend(fig, [
        _era5_handle(),
        Patch(color=ENSEMBLE["outer"], label="Ensemble 5–95%"),
        Line2D([], [], ls="none", marker="v", ms=7, color=OUTSIDE_COLOR, mec="white", label="ERA5 outside ensemble"),
        Patch(color=RANK_BAR, label="ERA5 rank frequency"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, marker="_", ms=7, label="Perfect-model 5–95%"),
        Line2D([], [], ls="none", marker="o", ms=6, color=STATUS_OK, label="✓ consistent"),
        Line2D([], [], ls="none", marker="X", ms=7, color=STATUS_FLAG, label="✗ flagged"),
    ])
    _suptitle(fig, title)
    return core.Panels(fig=fig, axes=axes)


# ---------------------------------------------------------------------------
# 8. Spatial evaluation (Suarez-Gutierrez et al., 2021, section 2.2.2)
# ---------------------------------------------------------------------------

#(t): The paper's map: ERA5 above every member (red) or below (blue) in >= 10% (light) or >= 20% (dark) of years
EDGE_CLASSES = ("neither", "above ≥ 10%", "above ≥ 20%", "below ≥ 10%", "below ≥ 20%", "both ≥ 10%")
EDGE_COLORS = ("#ffffff", "#f7aba4", "#a6272a", "#9ec5f4", "#1c5cab", "#8a5fbf")

#(t): One colour per ev.DIAGNOSES class (white: adequate), and per ev.PERIOD_DIAGNOSES class
DIAGNOSIS_COLORS = ("#ffffff", "#d03b3b", "#2a78d6", "#eda100", "#8a8985", "#1baf7a")
PERIOD_COLORS = ("#ffffff", "#eda100", "#d03b3b")

#(t): Hatching where ERA5 sits in the members' central 75% too often: > 80% and > 90% of years
CENTRAL_HATCHES = ((ev.CENTRAL_THRESHOLD, "////"), (90, "xxxx"))


def edge_class(table):
    """The paper's colour class at each point: index into ``EDGE_CLASSES``; NaN where untested."""
    above, below = table["above"], table["below"]
    edge, strong = ev.EDGE_THRESHOLD, 2 * ev.EDGE_THRESHOLD
    out = xr.zeros_like(above)
    out = xr.where(above >= edge, 1, out)
    out = xr.where(above >= strong, 2, out)
    out = xr.where(below >= edge, 3, out)
    out = xr.where(below >= strong, 4, out)
    out = xr.where((above >= edge) & (below >= edge), 5, out)
    return out.where(table["adequate"].notnull())


def _class_maps(field, table, labels, colors, row_dim, col_dim, title, cbar_label, hatch=False, **grid_kwargs):
    """Categorical polar maps, one panel per (row, col), with the adequate area in each panel's corner."""
    from plotting_modules import maps  # needs cartopy

    cmap = ListedColormap(colors).with_extremes(bad="white")
    levels = np.arange(-0.5, len(labels))
    grid_kwargs.setdefault("left", 2.0)
    panels = maps.polar_grid(field, row_dim=row_dim, col_dim=col_dim, levels=levels, cmap=cmap,
                             norm=BoundaryNorm(levels, cmap.N), discrete=True, ticklabels=list(labels),
                             title=title, cbar_label=cbar_label, tag=False, **grid_kwargs)
    rows = field[row_dim].values if row_dim in field.dims else [None]
    cols = field[col_dim].values if col_dim in field.dims else [None]
    axes = np.asarray(panels.axes, dtype=object).reshape(len(rows), len(cols))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cell = table.sel({d: v for d, v in ((row_dim, row), (col_dim, col)) if v is not None})
            ax = axes[i, j]
            if bool(cell["adequate"].isnull().all()):
                ax.text(0.5, 0.5, f"not tested\n(N < {ev.MIN_MEMBERS} or < {ev.MIN_YEARS} years)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK_2, zorder=20,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3))
                continue
            if hatch:
                for threshold, pattern in CENTRAL_HATCHES:
                    maps.plot_hatch(ax, cell["central"] > threshold, hatch=pattern, color=(0, 0, 0, 0.45),
                                    linewidth=0.4)
            ax.text(0.0, 1.0, f"adequate {float(ev.adequate_area(cell)):.0f}%", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=INK_2, zorder=20)
    return panels


@plot("figure")
@_styled
def spatial_maps(table, row_dim="model", col_dim="season", title=None, **grid_kwargs):
    """Fig. 5 of Suarez-Gutierrez et al. (2021): where ERA5 is beyond the members, or too often in their middle.

    Colour: ERA5 above every member (red) or below every member (blue) in at
    least 10% (light) or 20% (dark) of the years; purple, both. Hatching: ERA5
    inside the members' central 75% in more than 80% (////) or 90% (xxxx) of
    the years, i.e. the model's spread is too wide. White without hatching is
    adequate; each panel's corner gives the adequate share of the area.

    Args:
        table (xr.Dataset): ``ev.spatial_evaluation`` output with ``row_dim`` and ``col_dim``.
        row_dim, col_dim (str): Dimensions mapped to rows and columns.
        title (str | None): Figure title.
        **grid_kwargs: Passed to ``maps.polar_grid``.

    Returns:
        core.Panels
    """
    table = order_seasons(table)
    return _class_maps(
        edge_class(table), table, EDGE_CLASSES, EDGE_COLORS, row_dim, col_dim,
        title or "Where ERA5 is beyond the ensemble, or too often in its middle",
        "ERA5 above (red) / below (blue) every member in ≥ 10% or ≥ 20% of years.  "
        "Hatched: ERA5 in the members' central 75% in > 80% (////) or > 90% (xxxx) of years",
        hatch=True, **grid_kwargs,
    )


@plot("figure")
@_styled
def diagnosis_maps(table, row_dim="model", col_dim="season", variable="diagnosis", title=None, **grid_kwargs):
    """Step 7 (or step 8, with ``variable="period_diagnosis"``): what is wrong at each grid point.

    Args:
        table (xr.Dataset): ``ev.spatial_evaluation`` output (with ``period_diagnosis`` for step 8).
        row_dim, col_dim (str): Dimensions mapped to rows and columns.
        variable (str): "diagnosis" (``ev.DIAGNOSES``) or "period_diagnosis" (``ev.PERIOD_DIAGNOSES``).
        title (str | None): Figure title.
        **grid_kwargs: Passed to ``maps.polar_grid``.

    Returns:
        core.Panels
    """
    table = order_seasons(table)
    if variable == "period_diagnosis":
        labels, colors = ev.PERIOD_DIAGNOSES.values(), PERIOD_COLORS
        title = title or "Each problem early vs late: the same (variability) or changing (forced response)?"
    else:
        labels, colors = ev.DIAGNOSES.values(), DIAGNOSIS_COLORS
        title = title or "What is wrong at each grid point"
    return _class_maps(table[variable], table, tuple(labels), colors, row_dim, col_dim, title, None,
                       **grid_kwargs)


@plot("figure")
@_styled
def adequate_count_maps(counts, row_dim=None, col_dim="season", title=None, **grid_kwargs):
    """Fig. 8 of Suarez-Gutierrez et al. (2021): how many models are adequate at each grid point.

    Args:
        counts (xr.Dataset): ``ev.adequate_count`` output.
        row_dim, col_dim (str | None): Dimensions mapped to rows and columns.
        title (str | None): Figure title.
        **grid_kwargs: Passed to ``maps.polar_grid``.

    Returns:
        core.Panels
    """
    from plotting_modules import maps  # needs cartopy

    counts = order_seasons(counts)
    n_models = int(counts["n_tested"].max())
    cmap = ListedColormap(["#ffffff", *plt.get_cmap("Blues")(np.linspace(0.3, 1, max(n_models, 1)))])
    levels = np.arange(-0.5, n_models + 1)
    return maps.polar_grid(
        counts["n_adequate"].where(counts["n_tested"] > 0), row_dim=row_dim, col_dim=col_dim, levels=levels,
        cmap=cmap, norm=BoundaryNorm(levels, cmap.N), discrete=True,
        ticklabels=[str(n) for n in range(n_models + 1)],
        title=title or "Number of models that adequately capture ERA5",
        cbar_label=f"Models with no problem (of the {n_models} tested)", tag=False, **grid_kwargs,
    )
