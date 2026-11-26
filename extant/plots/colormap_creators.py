"""Making colormaps: bands of several colormaps in one bar (``banded_colormap``), a count of models
(``count_colormap``) and a diverging one with set turning points (``make_diverging_colormap``).
"""

from typing import NamedTuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


class Band(NamedTuple):
    """One band of a colour bar: values from ``lower`` to ``upper``, coloured from ``cmap`` between ``start`` and
    ``stop`` (0 to 1, either way round), split into ``n_steps`` if given.

    Example: ``Band(0, 1, "YlOrRd", 0.0, 0.2, n_steps=4)``, or ``Band(1, 3, "Reds")`` for all of Reds and the steps
    from the levels.
    """

    lower: float
    upper: float
    cmap: str
    start: float = 0
    stop: float = 1
    n_steps: int | None = None

def banded_colormap(bands, levels=None, extend="neither"):
    """Levels and colours for a colour bar made of bands, each band with colours from its own colormap.

    Give either ``levels`` for the whole bar, or ``n_steps`` in every band (the band split evenly), not both. A
    band's steps take evenly spaced colours from its start to its stop. An extended end gets one colour more from
    its band's colormap.

    Example: ``banded_colormap([Band(0, 1, "YlOrRd", 0, 0.2, n_steps=2), Band(1, 2, "Reds", n_steps=2)],
    extend="max")`` gives levels 0, 0.5, 1, 1.5, 2 and five colours: YlOrRd at 0 and 0.2, then Reds at 0, 0.5 and 1
    (the last for above 2).

    Args:
        bands (Sequence[Band]): From the lowest band to the highest.
        levels (array-like | None): The edges of the colour steps over all the bands; None to use the ``n_steps``.
        extend (str): 'neither', 'min', 'max' or 'both': which ends get a colour of their own.

    Returns:
        tuple: The levels (np.ndarray) and a ListedColormap with one colour per step plus one per extended end, for
        ``BoundaryNorm(levels, cmap.N, extend=extend)``.
    """
    has_steps = [band.n_steps is not None for band in bands]
    if levels is not None and any(has_steps):
        raise ValueError("Give levels or every band's number of steps, not both")
    if levels is None and not all(has_steps):
        raise ValueError("Without levels, every band needs its number of steps")
    if levels is None:
        #(c): Each band split evenly, its upper edge being the next band's lower edge
        levels = np.append(np.concatenate([np.linspace(band.lower, band.upper, band.n_steps + 1)[:-1]
                                           for band in bands]), bands[-1].upper)
    levels = np.asarray(levels, dtype=float)
    #(c): A step belongs to the band its middle falls in
    middles = (levels[:-1] + levels[1:]) / 2
    colours = []
    for i, band in enumerate(bands):
        n = int(np.sum((middles >= band.lower) & (middles < band.upper)))
        below = i == 0 and extend in ("min", "both")
        above = i == len(bands) - 1 and extend in ("max", "both")
        colours += list(plt.get_cmap(band.cmap)(np.linspace(band.start, band.stop, n + below + above)))
    if len(colours) != len(middles) + (extend in ("min", "both")) + (extend in ("max", "both")):
        raise ValueError("Some steps fall outside every band")
    return levels, ListedColormap(colours)


def count_colormap(n_models, base_cmap="Purples", add_white=True):
    """Discrete colormap for a number of models: white for none, one step per model."""
    base = plt.get_cmap(base_cmap)
    length = n_models if add_white else (n_models+1)
    colors = [base(x) for x in np.linspace(0, 1, length)]
    if add_white: colors = ['white', *colors]
    cmap = ListedColormap(colors)
    return cmap, BoundaryNorm(np.arange(-0.5, n_models + 1.5), cmap.N)


def make_diverging_colormap(bounds, purple_start=-100, red_start=100, white_centre=True):
    bounds = np.asarray(bounds)
    zero_idx = np.where(bounds == 0)[0][0]

    n_neg = zero_idx
    n_pos = len(bounds) - zero_idx - 1

    neg_pos = abs(purple_start) / abs(bounds[0])
    pos_pos = red_start / bounds[-1]

    neg_cmap = mcolors.LinearSegmentedColormap.from_list(
        "negative",
        [(0, "#54278f"), (1 - neg_pos, "#2171b5"), (1, "#c6dbef")])

    pos_cmap = mcolors.LinearSegmentedColormap.from_list(
        "positive",
        [(0, "#fff7bc"), (pos_pos, "#fe9929"), (1, "#7f0000")])

    neg_colors = neg_cmap(np.linspace(0, 1, n_neg))
    pos_colors = pos_cmap(np.linspace(0, 1, n_pos))

    if white_centre:
        neg_colors[-1] = mcolors.to_rgba("white")
        pos_colors[0] = mcolors.to_rgba("white")

    colors = np.vstack([neg_colors, pos_colors])

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm
