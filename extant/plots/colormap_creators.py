import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple
from matplotlib.colors import ListedColormap


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
