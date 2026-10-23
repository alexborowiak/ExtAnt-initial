"""Small shared helpers: a diverging colormap, period labels, and the frames of a slide build-up (one y range for
every frame, each frame saved at the figure's own size).
"""

from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


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

    
# --------------------------------------------------------------------------
# Small shared helpers, used by more than one drawing module
# --------------------------------------------------------------------------


def period_label(period):
    """Human label for a year selection, which may be a slice or a scalar."""
    if isinstance(period, slice):
        return f"{period.start}–{period.stop}"
    return str(period)


def shared_ylim(arrays, pad=0.05, include=()):
    """Y limits spanning every array, so buildup frames line up exactly.

    Compute this once from all the series and reuse it for every frame. Letting
    each frame autoscale is what makes a buildup jump around on screen. It can
    also change the y tick label widths, which under constrained layout moves
    the axes as well, though that depends on the numbers.

    Parameters
    ----------
    arrays : sequence of array_like
        All the series that will ever appear, revealed or not.
    pad : float
        Margin above and below, as a fraction of the data range.
    include : array_like
        Extra values that must stay inside the limits, e.g. a baseline drawn as
        a horizontal line rather than a series.

    Returns
    -------
    (float, float)
    """
    parts = [np.asarray(a).ravel() for a in arrays]
    if len(np.asarray(include).ravel()):
        parts.append(np.asarray(include, dtype=float).ravel())

    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("no finite values to take limits from")

    lo, hi = values.min(), values.max()
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def save_frame(fig, directory, index, name, close=True, facecolor="white", **kwargs):
    """Write one buildup frame at the figure's own size; return its path.

    Do not pass bbox_inches='tight': trimming to the drawn content changes the
    canvas between frames, which is the jitter the fixed geometry exists to
    prevent.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:02d}_{name}.png"
    fig.savefig(path, facecolor=facecolor, **kwargs)
    if close:
        plt.close(fig)
    return path




