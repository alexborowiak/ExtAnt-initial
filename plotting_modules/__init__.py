"""Shared vocabulary and small helpers for the figures.

    constants      season and forcing vocabulary: names, colours, orderings
    formatting     labels of a point or a selection
    utils          a diverging colormap, period labels, the frames of a slide build-up

The figures themselves are drawn with ``extant.plots``: ``charts`` (every plot that isn't a map) and
``antarctica`` (maps), with the one-axis panels of its other modules.
"""

from . import constants, formatting, utils
from .utils import save_frame, shared_ylim

__all__ = ["constants", "formatting", "utils", "save_frame", "shared_ylim"]
