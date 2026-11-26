"""The colour scales of the maps and tables.

Each scale is a plain dict of ``antarctica.heatmaps_of_antarctica``'s colour arguments (``levels``, ``cmap``,
``extend``, ``ticks``, ``ticklabels``, ``label``), passed with ``**``::

    antarctica.heatmaps_of_antarctica(change_da, 'model', 'season', axes=axes, cax=cax, **colour_scales.CHANGE)

One scale per quantity, so the same colour means the same value in every figure. To change a range everywhere, change
it here; for one figure only, copy the dict with a new value: ``{**colour_scales.CHANGE, 'levels': np.arange(-1, 1.1,
0.25)}``. A scale that depends on something (the number of models, the names of classes) is a function that returns
the same kind of dict.

What the colours mean, in every figure:

    red <-> blue      an increase <-> a decrease (of sea ice, the other way round: less ice is red, as warming is)
    grey -> orange    S/N, how far apart two distributions have moved: grey between -1 and 1, orange 1 to 2, red above 2;
       -> red         blues the same distances below 0
    counts            more than half the models changes the hue: reds to browns (increase), blues to purples
                      (decrease), greys to greens (no sign, ``majority_scale``); white is none
    3 x 3 key         the mean and the width together (``JOINT``, its key drawn by ``antarctica.bivariate_key``)

A scale made of bands of different colormaps (S/N, the counts) is built with ``colormap_creators.banded_colormap``:
each ``Band(lower, upper, cmap, start, stop)`` takes its colours from its own colormap between ``start`` and ``stop``.

Sections
--------
1. Scales that depend on something: S/N bands, p-values, model counts, named classes, probability ratios, ERA5's
   percentile
2. One scale per quantity: ``TEMPERATURE``, ``CHANGE``, ``FORCED_WARMING``, ``NOISE``, ``SPREAD``, ``SPREAD_CHANGE``,
   ``VARIANCE_CHANGE``, ``LINEAR_SIGNAL``, ``LINEAR_NOISE``, ``TAIL``, ``SN``, ``PVALUE``, ``CLASSES``, ``JOINT`` (and
   ``JOINT_KEY``), and the sea ice's ``CONCENTRATION``, ``CONCENTRATION_CHANGE`` and ``ICE_CORRELATION``
3. Checking a scale's range (``saturation``)
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .. import config
from ..response_change import CHANGE_CLASSES
from .colormap_creators import Band, banded_colormap

# ---------------------------------------------------------------------------
# 1. Scales that depend on something
# ---------------------------------------------------------------------------

def sn_scale(label="S/N", step=0.25, grey_until=1, orange_until=2, red_until=3):
    """S/N from -red_until to red_until: greys between ±grey_until (not emerged), then oranges up to orange_until and
    reds up to red_until for an increase, light and dark blues the same distances below 0 for a decrease. Purple below
    -red_until, and the darkest red above red_until.

    S/N says how far apart two distributions have moved, in units of their year-to-year noise, and its sign says
    which way: between -1 and 1 they overlap heavily, beyond ±2 they barely do. It is not a test, so there is no
    threshold, only bands.

    The bounds must be multiples of ``step``. Example: ``sn_scale(grey_until=0.5, orange_until=1.5)`` has greys from
    -0.5 to 0.5, oranges from 0.5 to 1.5 and reds from 1.5 to 3 (light blues from -1.5 to -0.5, dark blues from -3
    to -1.5).
    """
    #(c): The greys mirror each other: palest at 0, a little darker towards ±grey_until
    levels, inside_cmap = banded_colormap([
        Band(-red_until, -orange_until, "Blues", 0.95, 0.7),
        Band(-orange_until, -grey_until, "Blues", 0.55, 0.3),
        Band(-grey_until, 0, "Greys", 0.35, 0.1),
        Band(0, grey_until, "Greys", 0.1, 0.35),
        Band(grey_until, orange_until, "Oranges", 0.3, 0.55),
        Band(orange_until, red_until, "Reds", 0.7, 0.95),
    ], levels=np.arange(-red_until, red_until + step / 2, step))
    #(c): The two colours beyond the ends: purple below -red_until, the darkest red above red_until
    cmap = ListedColormap([plt.get_cmap("Purples")(0.85), *inside_cmap.colors, plt.get_cmap("Reds")(1.0)])
    return dict(levels=levels, cmap=cmap, extend="both",
                ticks=[-red_until, -orange_until, -grey_until, 0, grey_until, orange_until, red_until], label=label)


def pvalue_scale(alpha=0.05, label=None):
    """p-values in four bands: below alpha / 5, below alpha (both purple: significant), below 2 alpha (grey), above
    (white)."""
    levels = np.array([0, alpha / 5, alpha, 2 * alpha, 1])
    return dict(levels=levels, cmap=ListedColormap(["#3f007d", "#9e9ac8", "#d9d9d9", "white"]), extend="neither",
                ticks=levels, label=label or f"p-value (significant below {alpha:g})")


def count_scale(n_models, label="Number of models with a significant decrease (blue) or increase (red)"):
    """How many models have a significant decrease (left) or increase (right); white for none. One class per count.

    Up to half the models are blues and reds; more than half, purples and browns, so where most models agree stands
    out. Notebooks plot_03a to plot_03d map the decreases as negative numbers (the signed counts), so this one colour bar serves
    both rows of counts; the tick labels are the counts themselves.

    Example: ``count_scale(5)`` has ``half = 5 // 2 = 2`` and classes -5 to 5: purples for 5 to 3 decreases (the band
    from -5.5 to -2.5), blues for 2 and 1, white for none, reds for 1 and 2 increases, browns for 3 to 5.
    """
    half = n_models // 2
    #(c): One class per count, from -n_models to n_models; darkest at the ends
    levels, cmap = banded_colormap([
        Band(-n_models - 0.5, -half - 0.5, "PRGn", 0.0, 0.3),
        Band(-half - 0.5, -0.5, "Blues", 0.7, 0.3),
        Band(-0.5, 0.5, "Greys", 0, 0),
        Band(0.5, half + 0.5, "Reds", 0.3, 0.7),
        Band(half + 0.5, n_models + 0.5, "BrBG", 0.3, 0.0),
    ], levels=np.arange(-n_models - 0.5, n_models + 1))
    #(c): Beyond 8 models every other count is labelled (the even ones), so a bar under one column stays readable
    step = 2 if n_models > 8 else 1
    return dict(levels=levels, cmap=cmap, extend="neither", ticks=np.arange(-n_models, n_models + 1),
                ticklabels=[str(abs(k)) if k % step == 0 else '' for k in range(-n_models, n_models + 1)], label=label)


def majority_scale(n_models, label="Number of models"):
    """How many models, for a count with no sign (e.g. |S/N| above a threshold); white for none. One class per count.

    Up to half the models are greys, more than half greens, like ``count_scale``'s change of hue.
    """
    half = n_models // 2
    levels, cmap = banded_colormap([
        Band(-0.5, 0.5, "Greys", 0, 0),
        Band(0.5, half + 0.5, "Greys", 0.2, 0.45),
        Band(half + 0.5, n_models + 0.5, "Greens", 0.55, 0.95),
    ], levels=np.arange(-0.5, n_models + 1))
    #(c): Beyond 8 models every other count is labelled, as in count_scale
    step = 2 if n_models > 8 else 1
    return dict(levels=levels, cmap=cmap, extend="neither", ticks=np.arange(n_models + 1),
                ticklabels=[str(k) if k % step == 0 else '' for k in range(n_models + 1)], label=label)


def consistent_scale(n_models, label="Models with ERA5 inside their members' 5-95% range"):
    """How many models are consistent with ERA5, from brown (none) to green (all). One class per count."""
    return dict(levels=np.arange(-0.5, n_models + 1.5),
                cmap=ListedColormap(plt.get_cmap("BrBG")(np.linspace(0, 1, n_models + 1 + 2)[1:-1])), extend="neither",
                ticks=range(n_models + 1), label=label)


def adequate_scale(n_models):
    """How many models have no problem in the spatial evaluation, of the ``n_models`` tested: white for none, a darker blue
    for each model more. One class per count."""
    return dict(levels=np.arange(-0.5, n_models + 1),
                cmap=ListedColormap(["#ffffff", *plt.get_cmap("Blues")(np.linspace(0.3, 1, max(n_models, 1)))]),
                extend="neither", ticks=range(n_models + 1), label=f"Models with no problem (of the {n_models} tested)")


def class_scale(names, colors, label="", wrap=14):
    """Named classes numbered 0, 1, 2, ..., one colour each, e.g. ``rc.CHANGE_CLASSES`` or the diagnoses of the
    spatial evaluation.

    Each name is broken onto lines of at most ``wrap`` characters, so that several fit under one colour bar: 14 for
    six names under a page-wide bar, 8 for four or five under one panel.
    """
    return dict(levels=np.arange(-0.5, len(names)), cmap=ListedColormap(list(colors)), extend="neither",
                ticks=np.arange(len(names)),
                ticklabels=[textwrap.fill(str(name), wrap, break_long_words=False) for name in names], label=label)


def ratio_scale(label="Probability ratio", end=3, step=0.5):
    """A probability ratio on a log scale: the field is log2(PR), the ticks say the ratios (1/8 ... 8).

    Orange is more likely, purple less likely, whichever tail; ratios beyond 2^end take the end colours.
    """
    ticks = np.arange(-end, end + 1)
    return dict(levels=np.round(np.arange(-end, end + step / 2, step), 10), cmap="PuOr_r", extend="both", ticks=ticks,
                ticklabels=[f"1/{2 ** -t:g}" if t < 0 else f"{2 ** t:g}" for t in ticks], label=label)


#(t): Scorecard and map classes for ERA5's percentile among the members
PERCENTILE_BOUNDS = (0, 5, 25, 75, 95, 100)
PERCENTILE_COLORS = ("#1c5cab", "#9ec5f4", "#f0efec", "#f7aba4", "#a6272a")

#(c): Class edges with the ends pushed just outside 0-100. Percentiles of exactly 0 or 100 (ERA5
#(c): beyond every member) are common, and contourf leaves holes where a flat region sits on a level.
PERCENTILE_EDGES = (-0.5, *PERCENTILE_BOUNDS[1:-1], 100.5)


def percentile_colormap():
    """Five-class diverging colormap for ERA5's percentile: significant low, low, central, high, significant high."""
    cmap = ListedColormap(PERCENTILE_COLORS).with_extremes(bad="white")
    return cmap, BoundaryNorm(PERCENTILE_EDGES, cmap.N)


def percentile_scale(label="ERA5's percentile among the members"):
    """ERA5's percentile in the five classes of the verification figures: beyond 5-95%, 5-25%, central, 75-95%."""
    cmap, _ = percentile_colormap()
    return dict(levels=np.asarray(PERCENTILE_EDGES), cmap=cmap, extend="neither",
                ticks=np.array(PERCENTILE_BOUNDS), label=label)


# ---------------------------------------------------------------------------
# 2. One scale per quantity
# ---------------------------------------------------------------------------
#(c): About a dozen colours per scale: more cannot be told apart in a small printed panel. Check them on the real data
#(c): with ``saturation`` (notebook plot_05)

#(t): Temperature itself
TEMPERATURE = dict(levels=np.arange(-60, 11, 5), cmap="RdBu_r", extend="both", ticks=np.arange(-60, 11, 15),
                   label="Near-surface air temperature (°C)")

#(t): A change in the mean or a quantile
CHANGE = dict(levels=np.arange(-3, 3.1, 0.5), cmap="RdBu_r", extend="both", label="Anomaly (°C)")

#(t): The forced warming over a period: blues below 0, oranges from 0 to 1, reds from 1 to 2 (four steps each)
_warming_levels, _warming_cmap = banded_colormap([Band(-0.5, 0, "Blues_r", 0.5, 1, n_steps=4),
                                                  Band(0, 1, "Oranges", 0.0, 0.3, n_steps=4),
                                                  Band(1, 2, "Reds", 0.4, 1.0, n_steps=4)], extend="both")
FORCED_WARMING = dict(levels=_warming_levels, cmap=_warming_cmap, extend="both", label="Forced warming (°C)")

#(t): The year-to-year noise
NOISE = dict(levels=np.arange(0, 4.1, 0.5), cmap="viridis", extend="max", label="Standard deviation (°C)")

#(t): The width Q95 - Q05, and its change
SPREAD = dict(levels=np.arange(0, 13, 1), cmap="viridis", extend="max", label="Spread (°C)")
SPREAD_CHANGE = dict(levels=np.arange(-1, 1.1, 0.25), cmap="RdBu_r", extend="both", label="Change in spread (°C)")

#(t): ERA5's linear S/N since 1979 and its two parts (``emergence.linear_sn``): the trend's rise and the variability about
#(t): it
LINEAR_SIGNAL = dict(levels=np.arange(-2, 2.2, 0.2), cmap="RdBu_r", extend="both",
                     label="Rise of the linear trend over 1979–2013 (°C)")
LINEAR_NOISE = dict(levels=np.arange(0, 2.2, 0.2), cmap="YlOrBr", extend="max",
                    label="Standard deviation with the trend removed (°C)")

#(t): The change in variance
VARIANCE_CHANGE = dict(levels=np.round(np.arange(-2, 2.01, 0.4), 2), cmap="RdBu_r", extend="both",
                       label="Change in variance (°C²)")

#(t): The warm tail's change minus the cold tail's
TAIL = dict(levels=np.arange(-1, 1.1, 0.25), cmap="RdBu_r", extend="both",
            label="Upper-tail minus lower-tail change (°C)")

#(t): S/N in bands, and p-values
SN = sn_scale()
PVALUE = pvalue_scale(config.ALPHA)

#(t): Which of the mean and the width changed
CLASSES = class_scale(CHANGE_CLASSES.values(), ["white", "#fdb863", "#b2abd2", "#5e3c99"], "What changed", wrap=8)

#(t): The change in the mean (rows) and in the width (columns) together, each down, unchanged or up: class
#(t): 3 * (mean sign + 1) + (width sign + 1). The mean sets blue or red, a narrower distribution mixes in yellow and a
#(t): wider one purple; white where neither changed. Its key is a grid of the classes (``antarctica.bivariate_key``)
JOINT = dict(levels=np.arange(-0.5, 9), cmap=ListedColormap(["#7fc8a9", "#4a8fd1", "#4b3f9e",    # cooler
                                                             "#f3e19b", "white", "#cab2d6",      # same mean
                                                             "#f4a261", "#e05a4f", "#9e2a6e"]),  # warmer
             extend="neither", ticks=np.arange(9), label="")
#(t): The key's rows (bottom first), columns, and what they are: bivariate_key(cax, JOINT['cmap'], *JOINT_KEY)
JOINT_KEY = (["Lower", "Same", "Higher"], ["Narrow", "Same", "Wide"], "Mean", "Width")

#(t): The sea-ice concentration (%), light to dark blue; a difference in it, blue where there is more ice and red where
#(t): there is less, so that ice lost reads like warming; and its correlation with the temperature
CONCENTRATION = dict(levels=np.arange(0, 101, 10), cmap="Blues", extend="neither", label="Sea-ice concentration (%)")
CONCENTRATION_CHANGE = dict(levels=np.arange(-55, 56, 10), cmap="RdBu", extend="both", ticks=np.arange(-50, 51, 10),
                            label="Difference in sea-ice concentration (%)")
ICE_CORRELATION = dict(levels=np.round(np.arange(-1, 1.01, 0.2), 1), cmap="PuOr_r", extend="neither",
                       label="Correlation of sea ice and temperature")


# ---------------------------------------------------------------------------
# 3. Checking a scale's range
# ---------------------------------------------------------------------------

def saturation(da, scale):
    """The share of a field's values beyond either end of its colour scale, which the map cannot tell apart.

    Shown as the end colours (and arrows on the colour bar); a few percent is fine, much more means the scale's
    levels (here) are too narrow for the data. Grid points count equally, wherever they are.

    Args:
        da (xr.DataArray): Any field.
        scale (dict): The scale it is drawn with, e.g. ``CHANGE``.

    Returns:
        dict: ``below`` and ``above``, fractions of the values that are not NaN.
    """
    valid = da.notnull().sum()
    return {"below": float((da < scale["levels"][0]).sum() / valid),
            "above": float((da > scale["levels"][-1]).sum() / valid)}
