"""The text sizes and line widths of a figure in the notebook, the paper and on a slide; print sizes; saving.

A notebook sets ``plt.rcParams.update(styles.NOTEBOOK.rc)`` once in its Setup. A paper or slide figure is drawn inside
``with plt.rc_context(styles.PAPER.rc):`` and labels its maps with the style's sizes (``styles.PAPER.label_size``),
then is saved with ``styles.save``.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from .constants import INK, INK_2

#(t): What every style starts from
BASE_RC = {
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


@dataclass(frozen=True)
class Style:
    """Font sizes and line widths of a figure: its rcParams, and the sizes of the maps' labels, so a figure looks the same
    in the notebook, the paper or a talk.

    Args:
        rc (dict): matplotlib rcParams used while drawing.
        label_size (float): Column and row labels.
        title_size (float): Figure title.
        tick_size (float): Colour-bar tick labels.
        cbar_label_size (float): Colour-bar labels.
        tag_size (float): Panel letters.
    """

    rc: dict
    label_size: float = 10
    title_size: float = 12
    tick_size: float = 9
    cbar_label_size: float = 10
    tag_size: float = 9


#(t): On screen, in the notebooks
NOTEBOOK = Style(BASE_RC)

#(t): In print: 7-8 pt text, for figures 183 mm (7.2 in) wide, a two-column page
PAPER = Style(
    {**BASE_RC, "font.size": 7, "axes.titlesize": 8, "figure.titlesize": 9, "axes.labelsize": 7, "xtick.labelsize": 7,
     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42},
    label_size=8, title_size=9, tick_size=6.5, cbar_label_size=7.5, tag_size=8,
)

#(t): On a slide: big text, read from the back of a room
SLIDES = Style(
    {**BASE_RC, "font.size": 16, "axes.titlesize": 18, "figure.titlesize": 15, "axes.labelsize": 16,
     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 11},
    label_size=14, title_size=16, tick_size=11, cbar_label_size=12, tag_size=11,
)


#(t): Print sizes, in inches: the width of a two-column page (183 mm) and of one column (89 mm), and the most a
#(t): figure can be tall with room left for its caption (229 mm)
PAGE_WIDTH = 7.2
COLUMN_WIDTH = 3.5
PAGE_HEIGHT = 9.0

#(t): The room above a paper figure's panels (inches): its column labels only, as it has no title
PAPER_TITLE_HEIGHT = 0.35


def save(fig, directory, name):
    """Save a figure as a PDF (vector, for the paper) and a 300 dpi PNG; return the two paths.

    Args:
        fig (matplotlib.figure.Figure): The figure.
        directory (pathlib.Path): Folder, made if missing.
        name (str): File name, without extension.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.pdf", directory / f"{name}.png"]
    #(c): TrueType (Type 42) fonts in the PDF: matplotlib's default Type 3 fonts are refused by many journals
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        for path in paths:
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    return paths
