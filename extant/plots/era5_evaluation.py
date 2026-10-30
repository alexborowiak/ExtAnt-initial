"""Panels for the ERA5 evaluation; the calculations live in era5_evaluation.

Each ``plot_*(data, ax=None, ...)`` styles one panel (``charts.style_chart_ax``) and draws on it (a new figure if
``ax`` is None); the notebook makes the figure and its axes (a GridSpec), draws each panel, and labels them, as
everywhere (``charts``).
``member_strips_grid`` draws the member strips of a grid of seasons and statistics; each ``..._legend_handles()``
gives a panel's legend entries. Nothing here computes a test statistic. Every function draws the output of an
era5_evaluation function, so the numbers on a figure are exactly the numbers the tests used.

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
6. Multi-model summaries: scorecard and signal-noise diagram
7. Method demonstration on synthetic ensembles
8. Spatial evaluation: the classes and colours of its maps (drawn in notebook plot_01c)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .. import era5_evaluation as ev
from . import charts
from plotting_modules.constants import FORCING_COLORS, SEASONS


# ---------------------------------------------------------------------------
# 1. Style and shared helpers
# ---------------------------------------------------------------------------

INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#a3a29d"        # reference lines
ERA5_COLOR = charts.ERA5_LINE["color"]
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

#(t): Model identity: the marker, kept with every chart's settings (charts.MODEL_MARKERS; colours: charts.model_shades)
MODEL_MARKERS = charts.MODEL_MARKERS

EVAL_RC = {
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
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
    #(c): The gaps between panels in a figure with layout='constrained', as a share of the figure, so panels whose tick
    #(c): labels are hidden (shared axes) don't touch
    "figure.constrained_layout.wspace": 0.06,
    "figure.constrained_layout.hspace": 0.08,
}


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


def status_line(ax, x, y, ok, message, ha="left", fontsize=7.5):
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


# ---------------------------------------------------------------------------
# 2. Distributions: KDE and Q-Q panels
# ---------------------------------------------------------------------------

def plot_kde(summary, ax=None, members=True):
    """Every member's KDE (thin), the pooled ensemble KDE and ERA5's KDE, with a rug of ERA5's values.

    Args:
        summary (xr.Dataset): One treatment of ``ev.distribution_summary``, reduced to dims ``x``, ``member`` and
            ``year``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        members (bool): Draw the individual member curves.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
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
    return ax


def plot_qq(summary, ax=None):
    """Quantile-quantile plot: members' quantiles (5-95% envelope and median) against ERA5's.

    On the 1:1 line the distributions match. A parallel offset is a mean bias;
    a line steeper than 1:1 means the model is more variable than ERA5, and
    shallower means less variable.

    Args:
        summary (xr.Dataset): As ``plot_kde``'s.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
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
    return ax


#(t): The moment-test statistic whose verdict goes on a KDE panel, by treatment: the mean on raw values, the standard
#(t): deviation on anomalies
KDE_STATISTIC = {"raw": "mean", "anomaly": "std", "detrended": "std"}


def distribution_legend_handles():
    """The legend entries of the KDE and Q-Q panels: each member, the pooled ensemble, ERA5, its values, the members'
    5-95% and the 1:1 line."""
    return [
        Line2D([], [], color=ENSEMBLE["member"], lw=1, label="Each member"),
        Line2D([], [], color=ENSEMBLE["pooled"], lw=2, label="Pooled ensemble"),
        Line2D([], [], color=ERA5_COLOR, lw=2, label="ERA5"),
        Line2D([], [], color=ERA5_COLOR, lw=0, marker="|", ms=7, label="ERA5 values"),
        Patch(color=ENSEMBLE["outer"], label="Members' 5–95% (Q–Q)"),
        Line2D([], [], color=INK_2, lw=0.9, label="1:1"),
    ]


# ---------------------------------------------------------------------------
# 3. Time series: the ensemble plume, with ERA5 inside or outside it
# ---------------------------------------------------------------------------

def plot_plume(plume, ax=None, context=None, x_dim="year", note=True):
    """Ensemble plume through time with ERA5 on top, marking where ERA5 leaves the ensemble range.

    Args:
        plume (xr.Dataset): ``ev.ensemble_plume`` output reduced to ``x_dim``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        context (xr.Dataset | None): A second plume (e.g. hist-nat), drawn as outlines.
        x_dim (str): Dimension along the x axis.
        note (bool): Write how often ERA5 is outside the range, against chance.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=4.3)
    charts.style_chart_ax(ax)
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
    return ax


def plume_legend_handles(context=False):
    """The legend entries of the plume panels: ERA5, where it leaves the ensemble, the median and the three ranges, and
    hist-nat's if drawn (``context``)."""
    handles = [
        Line2D([], [], color=ERA5_COLOR, lw=1.6, marker="o", ms=3, label="ERA5"),
        Line2D([], [], ls="none", marker="^", ms=7, color=OUTSIDE_COLOR, mec="white", label="ERA5 above every member"),
        Line2D([], [], ls="none", marker="v", ms=7, color=OUTSIDE_COLOR, mec="white", label="ERA5 below every member"),
        Line2D([], [], color=ENSEMBLE["median"], lw=1.3, label="Ensemble median"),
        Patch(color=ENSEMBLE["inner"], label="Central 75% (12.5–87.5%)"),
        Patch(color=ENSEMBLE["outer"], label="5–95%"),
        Patch(color=ENSEMBLE["range"], label="Full range"),
    ]
    if context:
        handles += [Line2D([], [], color=CONTEXT_COLOR, lw=1.2, ls=(0, (4, 2)), label="hist-nat median"),
                    Line2D([], [], color=CONTEXT_COLOR, lw=1, label="hist-nat 5–95%")]
    return handles


# ---------------------------------------------------------------------------
# 4. Moment test (and the other statistics): ERA5 among the members' values
# ---------------------------------------------------------------------------

def plot_member_strip(result, ax=None):
    """One statistic: each member as a dot, their central range shaded, ERA5 as a black line.

    Args:
        result (xr.Dataset): ``ev.locate`` output (one statistic of a moment or statistic test) reduced to ``member``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel(panel_height=1.0)
    charts.style_chart_ax(ax)
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
    return ax


def member_strips_grid(result, row_dim, axes=None, statistics=None):
    """``plot_member_strip`` for every value of ``row_dim`` (down) and every statistic (across), each with its verdict.

    The verdict, in each panel's top line: where ERA5 sits if consistent, what is wrong if flagged (``verdict``,
    short). If the model were perfect, ERA5 would land outside the shaded range (the members' 5-95%) only 10% of the
    time.

    Args:
        result (xr.Dataset): ``ev.locate`` results along a ``statistic`` dim (e.g. notebook plot_01b's moment test) at one
            point, with ``row_dim`` (e.g. 'season').
        row_dim (str): The dim down the rows, in the order of its values.
        axes (np.ndarray | None): The panels, rows × columns, e.g. ``fig.add_gridspec(4, 3).subplots()``. A new
            figure if None.
        statistics (Sequence[str] | None): The statistics across, in order; all of ``result``'s by default.
    """
    statistics = list(statistics if statistics is not None else result["statistic"].values)
    row_values = result[row_dim].values
    if axes is None:
        fig = plt.figure(figsize=charts.chart_figure_size(len(row_values), len(statistics), n_legends=0,
                                                          panel_width=2.0, panel_height=0.9), layout='constrained')
        axes = fig.add_gridspec(len(row_values), len(statistics)).subplots(squeeze=False)
    axes = np.reshape(axes, (len(row_values), len(statistics)))

    for i, row_value in enumerate(row_values):
        for j, statistic in enumerate(statistics):
            cell = result.sel({row_dim: row_value, "statistic": statistic})
            plot_member_strip(cell, axes[i, j])
            #(c): Short text so it fits the panel: where ERA5 sits if consistent, the reading if flagged
            ok, reading = verdict(result.sel({row_dim: row_value}), statistic, short=True)
            status_line(axes[i, j], 0.01, 0.99, ok, where_era5(cell["percentile"]) if ok else reading, fontsize=7)


def member_strip_legend_handles():
    """The legend entries of the member strips: each member, their 5-95%, their median and ERA5."""
    return [
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["inner"], mec="white", label="Each member"),
        Patch(color=ENSEMBLE["range"], label="Members' 5–95%"),
        Line2D([], [], color=ENSEMBLE["median"], lw=1, label="Member median"),
        Line2D([], [], color=ERA5_COLOR, lw=1.8, marker="D", ms=5, mec="white", label="ERA5"),
    ]


def statistic_title(result, statistic):
    """The title of a statistic's column: its label, and the treatment it was computed on.

    Args:
        result (xr.Dataset): ``ev.locate`` results along a ``statistic`` dim; a ``treatment`` coordinate along it says
            each statistic's treatment, else ``ev.DIAGNOSTICS``' does.
        statistic (str): One of them.

    Returns:
        str: E.g. 'Std. deviation\\nof anomalies'.
    """
    diagnostic = ev.DIAGNOSTICS[statistic]
    cell = result.sel(statistic=statistic)
    treatment = str(cell["treatment"].values) if "treatment" in cell.coords else diagnostic.treatment
    return f"{diagnostic.label}\nof {ev.TREATMENTS[treatment].lower()}"


# ---------------------------------------------------------------------------
# 5. Rank histograms (Suarez-Gutierrez et al., 2021)
# ---------------------------------------------------------------------------

RANK_BAR = "#b7d3f6"
RANK_LINE = "#2a78d6"


def plot_rank_histogram(result, ax=None, smooth=5, statistics=None):
    """ERA5's rank histogram against a perfect model's, in the style of Suarez-Gutierrez et al. (2021).

    Bars are ERA5's rank frequencies and the solid line their running mean.
    The dashed lines bound the running means of the pseudo-observations: the
    range a perfect model's histogram falls in, given this ensemble and record.
    The lowest and highest ranks (ERA5 outside the ensemble) are drawn
    separately: crosses for ERA5 and dash-dot bars for the perfect-model range.

    Args:
        result (xr.Dataset): ``ev.rank_histogram_test`` output at one point (with histograms).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        smooth (int): Running-mean window, in ranks.
        statistics (Sequence[str] | None): Keys of ``ev.RANKS`` to print as verdicts;
            by default the ones that mean something for the result's treatment (``RANK_VERDICTS``).

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=4.0, panel_height=3.0)
    charts.style_chart_ax(ax)
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
        status_line(ax, 0.02, 0.98 - 0.105 * k, ok, message)
    return ax


def rank_histogram_legend_handles():
    """The legend entries of ``plot_rank_histogram``."""
    return [
        Patch(color=RANK_BAR, label="ERA5 rank frequency"),
        Line2D([], [], color=RANK_LINE, lw=2.4, label="Running mean"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, ls=(0, (4, 2.5)), label="Perfect-model 5–95%"),
        Line2D([], [], ls="none", marker="X", ms=10, color=RANK_LINE, mec="white", label="ERA5 outside ensemble"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, ls="-.", marker="x", ms=5, label="Perfect-model range, outside"),
        Line2D([], [], color=MUTED, lw=0.9, label="Flat (infinite record)"),
    ]


def plot_grouped_rank_histogram(result, ax=None, n_bins=10, statistics=None):
    """A compact rank histogram on ``n_bins`` bins of rank / N, so ensembles of any size share an axis.

    Bars are ERA5; the grey bars span the perfect-model 5-95% range of each bin.

    Args:
        result (xr.Dataset): ``ev.rank_histogram_test`` output at one point (with histograms).
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.
        n_bins (int): Bins of rank / N.
        statistics (Sequence[str] | None): The verdicts printed; by default ``RANK_VERDICTS`` for the treatment, less
            the trend.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
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
        status_line(ax, 0.02, 0.98 - 0.13 * k, ok, message, fontsize=7)
    return ax


def grouped_rank_legend_handles():
    """The legend entries of ``plot_grouped_rank_histogram``."""
    return [
        Patch(color=RANK_BAR, label="ERA5 rank frequency (%)"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, marker="_", ms=7, label="Perfect-model 5–95%"),
        Line2D([], [], color=MUTED, lw=0.9, label="Flat"),
    ]


def anatomy_years(result):
    """Three instructive years for ``plot_rank_counting``: ERA5's lowest, most central and highest rank.

    Args:
        result (xr.Dataset): ``ev.rank_histogram_test`` output with ``obs_rank`` on (year).

    Returns:
        list[int]: The years, in order, without repeats.
    """
    rank = result["obs_rank"].dropna("year")
    years = [int(rank.idxmin()), int(np.abs(rank - 0.5).idxmin()), int(rank.idxmax())]
    return sorted(dict.fromkeys(years))


def plot_rank_counting(ensemble, obs, years, ax=None):
    """How ERA5's rank is counted: in each of ``years``, the members (dots) below ERA5 (dark) and above it (light).

    The second step of how the rank-histogram test works (notebook plot_01b); its first is the members and ERA5 through
    time with ``years`` highlighted, and its last the histogram (``plot_grouped_rank_histogram``).

    Args:
        ensemble (xr.DataArray): Treated members on (member, year), as passed to the test.
        obs (xr.DataArray): Treated ERA5 on (year,).
        years (Sequence[int]): The years shown, e.g. ``anatomy_years(result)``.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with a legend of the dots.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    n_members = ensemble.sizes["member"]
    for k, year in enumerate(years):
        values = np.sort(ensemble.sel(year=year).values)
        value = float(obs.sel(year=year))
        below = values < value
        offsets = ((np.arange(values.size) * 0.618034) % 1 - 0.5) * 0.5
        ax.scatter(k + offsets[below], values[below], s=16, color=ENSEMBLE["pooled"], edgecolor="white",
                   linewidth=0.4, zorder=2)
        ax.scatter(k + offsets[~below], values[~below], s=16, color=ENSEMBLE["outer"], edgecolor="white",
                   linewidth=0.4, zorder=2)
        ax.plot([k - 0.34, k + 0.34], [value, value], color=ERA5_COLOR, lw=1.8, zorder=3)
        ax.plot(k, value, marker="D", color=ERA5_COLOR, ms=7, mec="white", zorder=4)
        ax.annotate(f"rank {below.sum()} of {n_members}", (k, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["pooled"], label="below ERA5"),
        Line2D([], [], ls="none", marker="o", ms=5, color=ENSEMBLE["outer"], label="above ERA5"),
    ], loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.2)
    return ax


def plot_rank_through_time(result, years=(), ax=None):
    """ERA5's rank / N through the record, with its trend; a drift means the model's forced response differs.

    The third step of how the rank-histogram test works (notebook plot_01b). Shaded: outside the ensemble.

    Args:
        result (xr.Dataset): ``ev.rank_histogram_test`` output with ``obs_rank`` on (year).
        years (Sequence[int]): Years highlighted, as in the step before.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis, with the rank trend's verdict.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=5.0)
    charts.style_chart_ax(ax)
    rank = result["obs_rank"]
    n_members = result.attrs["n_members"]
    _, fitted = ev.linear_fit(rank, "year")
    ax.axhspan(-0.04, 1 / n_members, color="#fbe3d9", lw=0, zorder=0)
    ax.axhspan(1 - 1 / n_members, 1.04, color="#fbe3d9", lw=0, zorder=0)
    ax.plot(rank["year"], rank, color=ERA5_COLOR, lw=0.8, marker="o", ms=4, zorder=3)
    ax.plot(rank["year"], fitted, color=RANK_LINE, lw=2.0, zorder=2)
    for year in years:
        ax.axvspan(year - 0.45, year + 0.45, color="#e7e6e2", zorder=0)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0\nbelow all", "0.25", "0.5", "0.75", "1\nabove all"])
    ok, message = verdict(result, "rank_trend")
    status_line(ax, 0.01, 0.97, ok, message)
    return ax


# ---------------------------------------------------------------------------
# 6. Multi-model summaries: scorecard and signal-noise diagram
# ---------------------------------------------------------------------------

def statistic_label(table, statistic):
    """The label of one statistic: the table's ``label`` coordinate if it has one, else ``ev.DIAGNOSTICS``'.

    A name with its treatment in brackets, e.g. "dispersion (anomaly)", keeps the brackets; a name
    ``ev.DIAGNOSTICS`` does not know (an experiment, say) is its own label.
    """
    if "label" in table.coords:
        return str(table["label"].sel(statistic=statistic).values)
    name, _, treatment = str(statistic).partition(" (")
    if name not in ev.DIAGNOSTICS:
        return str(statistic)
    label = ev.DIAGNOSTICS[name].label
    return f"{label} ({treatment}" if treatment else label


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


def plot_scorecard(table, statistics=None, ax=None):
    """Every model, statistic and season at one point: where ERA5 falls among the members, as a table of cells.

    Colour is ERA5's percentile among the members (blue: ERA5 below most
    members; red: above). The darkest classes are beyond the members' 5-95%
    range, and outlined cells are significant at the test's alpha; faint italic
    numbers are where too few members could flag anything. The x label says so.
    Its colour bar: ``fig.colorbar(image, cax=..., ticks=PERCENTILE_BOUNDS)``.

    Args:
        table (xr.Dataset): ``percentile``, ``verdict`` (and ``testable``) on (model, statistic, season)
            at one point, e.g. notebook calc_03b's ``eval_table_ds``.
        statistics (Sequence[str] | None): Subset and order of the column groups.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.image.AxesImage: The cells, for a colour bar (``image.axes`` is the axis).
    """
    table = order_seasons(table)
    statistics = list(statistics or table["statistic"].values)
    table = table.sel(statistic=statistics)
    models, seasons = list(table["model"].values), list(table["season"].values)
    n_groups, n_seasons = len(statistics), len(seasons)
    if ax is None:
        ax = charts.one_panel(panel_width=0.44 * n_groups * n_seasons, panel_height=0.46 * len(models))
    charts.style_chart_ax(ax)

    #(c): One row per model, one column per statistic and season
    percentile = table["percentile"].transpose("model", "statistic", "season").values.reshape(len(models), -1)
    flagged = (table["verdict"].transpose("model", "statistic", "season").values != 0).reshape(len(models), -1)
    untestable = np.zeros_like(flagged)
    if "testable" in table:
        untestable = (table["testable"].transpose("model", "statistic", "season").values == 0).reshape(len(models), -1)
    cmap, norm = percentile_colormap()

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
    #(c): A white gap between the statistics
    for group in range(1, n_groups):
        ax.axvline(group * n_seasons - 0.5, color="white", lw=5)

    #(c): Seasons along the bottom, the statistics' names along the top, no grid or frame
    ax.grid(False)
    ax.set_xticks(range(percentile.shape[1]), seasons * n_groups, fontsize=7)
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    top = ax.secondary_xaxis("top")
    top.set_xticks([g * n_seasons + (n_seasons - 1) / 2 for g in range(n_groups)],
                   [_wrap(statistic_label(table, s)) for s in statistics], fontsize=8, fontweight="bold")
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)

    alpha = table.attrs.get("alpha", ev.ALPHA)
    note = (f"Outlined: significant at p < {alpha:g} (ERA5 outside the members' "
            f"{100 * alpha / 2:g}–{100 - 100 * alpha / 2:g}% range)")
    if untestable.any():
        note += f".  Faint italic: too few members (N < {int(np.ceil(2 / alpha))}) for the test to flag anything"
    ax.set_xlabel(note, color=INK_2, fontsize=8)
    return image


def plot_signal_noise(table, season, x="trend", y="std", ax=None):
    """Signal against noise for every model in one season: ERA5's percentile for the trend (x) and the variability (y).

    Inside the grey square, ERA5 is consistent with the model on both. Left of
    it the model's trend is too large, right of it too small; below it the
    model is too variable, above it not variable enough. ``signal_noise_label``
    gives each axis's label.

    Args:
        table (xr.Dataset): ``percentile``, ``verdict`` (and ``testable``) on (model, statistic, season)
            at one point, e.g. notebook calc_03b's ``eval_table_ds``.
        season (str): The season drawn.
        x, y (str): Statistics on each axis.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis. Model k is the k-th shade of historical's colour (``charts.model_shades``)
        and ``MODEL_MARKERS[k]`` (``charts.model_legend_handles(models, experiment='historical')``).
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    alpha = table.attrs.get("alpha", ev.ALPHA)
    low, high = 100 * alpha / 2, 100 - 100 * alpha / 2
    ax.add_patch(Rectangle((low, low), high - low, high - low, color="#f0efec", lw=0, zorder=0))
    ax.axhline(50, color="#dcdbd7", lw=0.8, zorder=0)
    ax.axvline(50, color="#dcdbd7", lw=0.8, zorder=0)
    models = table["model"].values
    shades = charts.model_shades(FORCING_COLORS["historical"], len(models))
    for k, model in enumerate(models):
        point = table["percentile"].sel(model=model, season=season)
        ax.scatter(float(point.sel(statistic=x)), float(point.sel(statistic=y)), s=58,
                   marker=MODEL_MARKERS[k % len(MODEL_MARKERS)], color=shades[k],
                   edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_aspect("equal")
    return ax


def signal_noise_label(statistic):
    """The label of a signal-noise axis: ERA5's percentile of the statistic, and what each end means.

    Args:
        statistic (str): A key of ``ev.DIAGNOSTICS``, e.g. 'trend'.

    Returns:
        str: E.g. 'ERA5 percentile of the trend\\n← trend too large   ·   trend too small →'.
    """
    diagnostic = ev.DIAGNOSTICS[statistic]
    return (f"ERA5 percentile of the {diagnostic.label.lower()}\n"
            f"← {diagnostic.low.removeprefix('model ')}   ·   {diagnostic.high.removeprefix('model ')} →")


# ---------------------------------------------------------------------------
# 7. Method demonstration on synthetic ensembles
# ---------------------------------------------------------------------------

def plot_percentile_ladder(tests, anomaly_ranks, detrended_ranks, alpha=ev.ALPHA, ax=None):
    """ERA5's percentile for each statistic of a test and the rank tests, one row each: green inside the members'
    central range (the grey band), a red cross where flagged.

    The last column of the toy-ensemble demonstration (notebook plot_01b): one scenario, one season.

    Args:
        tests (xr.Dataset): ``ev.locate`` results on (statistic) for one scenario and season.
        anomaly_ranks, detrended_ranks (xr.Dataset): ``ev.rank_histogram_test`` of the anomalies and of the detrended
            anomalies for the same scenario: their dispersions and the rank trend are the last rows.
        alpha (float): The tests' two-sided significance level, for the grey band.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel()
    charts.style_chart_ax(ax)
    #(c): The ladder: (label, its result, statistic), top to bottom
    ladder = [(ev.DIAGNOSTICS[s].label, tests, s) for s in tests["statistic"].values] + [
        ("Rank dispersion (anomalies)", anomaly_ranks, "dispersion"),
        ("Rank dispersion (detrended)", detrended_ranks, "dispersion"),
        ("Rank trend (anomalies)", anomaly_ranks, "rank_trend"),
    ]
    ax.axvspan(100 * alpha / 2, 100 - 100 * alpha / 2, color="#f0efec", lw=0, zorder=0)
    ax.axvline(50, color="#dcdbd7", lw=0.8, zorder=0)
    for k, (_, result, statistic) in enumerate(ladder):
        row = result.sel(statistic=statistic)
        flagged = int(row["verdict"]) != 0
        ax.plot(float(row["percentile"]), k, marker="X" if flagged else "o", ms=8 if flagged else 6.5,
                color=STATUS_FLAG if flagged else STATUS_OK, mec="white", mew=0.8, zorder=3)
    ax.set_yticks(range(len(ladder)), [label for label, _, _ in ladder], fontsize=7.5)
    ax.set_ylim(len(ladder) - 0.5, -0.5)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    return ax


def scenario_legend_handles():
    """The legend entries of the toy-ensemble demonstration: the plume's, the rank histograms' and the ladder's."""
    return [
        Line2D([], [], color=ERA5_COLOR, lw=1.6, marker="o", ms=3, label="ERA5"),
        Patch(color=ENSEMBLE["outer"], label="Ensemble 5–95%"),
        Line2D([], [], ls="none", marker="v", ms=7, color=OUTSIDE_COLOR, mec="white", label="ERA5 outside ensemble"),
        Patch(color=RANK_BAR, label="ERA5 rank frequency"),
        Line2D([], [], color=PSEUDO_COLOR, lw=1.3, marker="_", ms=7, label="Perfect-model 5–95%"),
        Line2D([], [], ls="none", marker="o", ms=6, color=STATUS_OK, label="✓ consistent"),
        Line2D([], [], ls="none", marker="X", ms=7, color=STATUS_FLAG, label="✗ flagged"),
    ]


# ---------------------------------------------------------------------------
# 8. Spatial evaluation (Suarez-Gutierrez et al., 2021, section 2.2.2)
# ---------------------------------------------------------------------------

#(t): The paper's map: ERA5 above every member (red) or below (blue) in >= 10% (light) or >= 20% (dark) of years
EDGE_CLASSES = ("neither", "above ≥ 10%", "above ≥ 20%", "below ≥ 10%", "below ≥ 20%", "both ≥ 10%")
EDGE_COLORS = ("#ffffff", "#f7aba4", "#a6272a", "#9ec5f4", "#1c5cab", "#8a5fbf")

#(t): One colour per ev.DIAGNOSES class (white: adequate), and per ev.PERIOD_DIAGNOSES class
DIAGNOSIS_COLORS = ("#ffffff", "#d03b3b", "#2a78d6", "#eda100", "#8a8985", "#1baf7a")
PERIOD_COLORS = ("#ffffff", "#eda100", "#d03b3b")

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


