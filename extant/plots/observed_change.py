"""Panels for the observed-change analysis; the calculations are in observed_change.

Each ``plot_*(data, ax=None, ...)`` styles one panel (``charts.style_chart_ax``) and draws on it (a new figure if
``ax`` is None); the notebook makes the figure and its axes (a GridSpec), draws each panel and labels them, as
everywhere (``charts``); each
``..._legend_handles`` gives a panel's legend entries. Colour follows the experiment (FORCING_COLORS),
and ERA5 is always black.

The record rates: cumulative record highs and lows against chance and the members (``plot_record_rates``), and the
detection classes' colours (``DETECTION_COLORS``). The fingerprinting's panels are in ``plots.fingerprinting``.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import charts
from .constants import ERA5_COLOR, FORCING_COLORS, INK, INK_2

#(t): Detection classes (observed_change.DETECTION_CLASSES), validated as a categorical palette
DETECTION_COLORS = ("#d9d8d4", "#2a78d6", "#eb6834", "#4a3aa7")


# ---------------------------------------------------------------------------
# Record rates
# ---------------------------------------------------------------------------

def plot_record_rates(curves, kind, experiments=("hist-nat", "historical"), ax=None):
    """Cumulative record highs or lows: ERA5 against chance and against each experiment's members.

    With no change, records arrive ever more slowly (the dashed curve,
    1 + 1/2 + ... + 1/n per season). Warming keeps record highs coming and
    starves record lows. ERA5 above hist-nat's band means more records than
    natural variability gives.

    Args:
        curves (xr.Dataset): ``oc.record_curves`` output.
        kind (str): 'highs' or 'lows' (a value of its ``kind`` dim).
        experiments (Sequence[str]): Experiments whose member ranges are drawn.
        ax (matplotlib.axes.Axes | None): Where to draw; a new figure with one panel if None.

    Returns:
        matplotlib.axes.Axes: The axis.
    """
    if ax is None:
        ax = charts.one_panel(panel_width=4.5)
    charts.style_chart_ax(ax)
    years = curves["year"].values
    for experiment in experiments:
        members = curves["members"].sel(experiment=experiment, kind=kind).dropna("member", how="all")
        colour = FORCING_COLORS.get(experiment, INK)
        low, mid, high = members.quantile([0.05, 0.5, 0.95], dim="member").transpose("quantile", "year").values
        ax.fill_between(years, low, high, color=colour, alpha=0.18, lw=0, step="post")
        ax.step(years, mid, where="post", color=colour, lw=1.4)
    ax.plot(years, curves["expected"], color=INK_2, lw=1.2, ls=(0, (4, 2)))
    ax.step(years, curves["obs"].sel(kind=kind), where="post", color=ERA5_COLOR, lw=2.2)
    final = float(curves["obs"].sel(kind=kind).isel(year=-1))
    ax.text(0.02, 0.97, f"ERA5: {final:.0f} record {kind} · chance: {float(curves['expected'].isel(year=-1)):.1f}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
    return ax


def record_rate_legend_handles(experiments=("hist-nat", "historical")):
    """The legend entries of ``plot_record_rates``: ERA5, chance, and each experiment's members."""
    handles = [Line2D([], [], color=ERA5_COLOR, lw=2.2, label="ERA5"),
               Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 2)), label="No change (1 + 1/2 + … + 1/n)")]
    return handles + [Patch(color=FORCING_COLORS.get(e, INK), alpha=0.35, label=f"{e} members 5–95% (line: median)")
                      for e in experiments]
