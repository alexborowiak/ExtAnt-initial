"""Reusable plotting library, organised by what a function draws.

    core           figure structure: GridLayout, panel_grid, labels, colorbars
    maps           anything on a projection (needs cartopy)
    timeseries     plumes, boxes, differences, reveals, stacks
    distributions  KDEs, histograms, per-member bars
    constants      season and forcing vocabulary
    utils          list_plots, the @plot decorator, shared helpers
    figures        the specific paper and talk figures

`maps` and `figures` are not imported here, so importing the package does not
require cartopy. Import them explicitly when you need them:

    >>> from plotting_modules import maps, figures
    >>> from plotting_modules.utils import print_plots
    >>> print_plots()
"""

from . import constants, core, distributions, timeseries, utils
from .core import (
    GridLayout,
    NestedLayout,
    Panels,
    add_colorbar,
    add_suptitle,
    hide_inner_labels,
    label_cols,
    label_rows,
    make_axes,
    open_layout,
    panel_grid,
    reveal_legend,
    style_ax,
    tag_panels,
)
from .utils import list_plots, plot, plot_index, print_plots, save_frame, shared_ylim

__all__ = [
    "constants", "core", "distributions", "figures", "maps", "timeseries", "utils",
    "GridLayout", "NestedLayout", "Panels", "open_layout", "make_axes", "panel_grid",
    "label_cols", "label_rows", "tag_panels", "hide_inner_labels", "style_ax",
    "add_colorbar", "add_suptitle", "reveal_legend",
    "list_plots", "plot", "plot_index", "print_plots", "save_frame", "shared_ylim",
]
