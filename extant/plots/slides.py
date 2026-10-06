"""Labels for talk figures: names at the end of lines, quantile levels at the edge of a plume."""

import numpy as np


def label_line_end(ax, x, y, text, color, **kwargs):
    """Write ``text`` just past the point (x, y), outside the axes if need be.

    Returns:
        matplotlib.text.Annotation
    """
    return ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                       color=color, va="center", clip_on=False, **kwargs)


def label_band_edges(ax, da, pairs, x, color="0.45", quantile_dim="quantile", **kwargs):
    """Label each edge of a quantile plume with its level, at its last point.

    Args:
        ax (plt.Axes): Axis to annotate.
        da (xr.DataArray): The quantile series the plume was drawn from.
        pairs (array-like): (low, high) quantile pairs, as passed to ``timeseries.draw_plume``.
        x (float): x position of the labels.
        color: Text colour.
        quantile_dim (str): Dimension holding the quantiles.
        **kwargs: Passed to ``ax.annotate``.
    """
    for q in np.unique(np.asarray(pairs)):
        label_line_end(ax, x, da.sel({quantile_dim: q}, method="nearest").isel({da.dims[-1]: -1}),
                       f"{q:.0%}", color, **kwargs)
