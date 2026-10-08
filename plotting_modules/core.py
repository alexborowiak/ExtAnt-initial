"""Shared panel-grid and layout helpers.

``panel_grid`` creates an N-by-M array of axes. Plot directly into those axes
with ordinary loops; it does not choose data or draw anything.

``GridLayout`` owns a whole figure and uses inches for panel and colorbar
sizes. ``NestedLayout`` fills one cell of a parent GridSpec and uses relative
sizes. Automatic colorbars always occupy a reserved GridSpec row; a caller who
supplies axes owns the colorbar axes too.
"""

import string
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Layout constants, all in INCHES. Nothing here is a figure fraction, which is
# what keeps panels identical whether there is 1 row or 10. Override any of
# them per-figure via GridLayout's keyword arguments.
# --------------------------------------------------------------------------
PANEL_W = 3.0            # width of one panel
PANEL_H = 3.0            # height of one panel
WSPACE = 0.10            # gap between columns
HSPACE = 0.28            # gap between rows
MARGIN_LEFT = 0.15
MARGIN_RIGHT = 0.15
MARGIN_TOP = 0.40        # room for column titles
MARGIN_BOTTOM = 0.15
ROW_LABEL_SPACE = 0.85   # added to the left margin when row labels are drawn
TITLE_SPACE = 0.45       # extra top room when a suptitle is passed
CBAR_GAP = 0.45          # gap between the bottom row and the bars
CBAR_HEIGHT = 0.16       # bar thickness
CBAR_TICK_SPACE = 0.45   # room below the bars for tick labels
CBAR_LABEL_SPACE = 0.24  # extra room below that for the colorbar axis label


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


class GridLayout:
    """A panel grid that owns a whole figure, measured in inches.

    Parameters
    ----------
    n_rows, n_cols : int
        Shape of the panel grid.
    panel_w, panel_h : float
        Default panel size in inches. Square for maps, usually wider than tall
        for time series.
    row_heights, col_widths : sequence of float or None
        Per-row heights and per-column widths in inches, for grids whose panels
        are not all the same size. Default to panel_h and panel_w repeated.
    wspace, hspace : float
        Gaps between columns and rows, in inches.
    left, right, top, bottom : float or None
        Margins in inches. `left` defaults to MARGIN_LEFT, or that plus
        ROW_LABEL_SPACE when `row_labels` is set.
    row_labels, has_title, has_cbar, has_cbar_label : bool
        Whether to reserve margin for each. The same numbers reserve the space
        and place the artist, so they cannot disagree.
    cbar_height, cbar_gap : float
        Bar thickness and the gap above it, in inches.
    """

    owns_figure = True

    def __init__(
        self,
        n_rows,
        n_cols,
        panel_w=PANEL_W,
        panel_h=PANEL_H,
        row_heights=None,
        col_widths=None,
        wspace=WSPACE,
        hspace=HSPACE,
        left=None,
        right=MARGIN_RIGHT,
        top=MARGIN_TOP,
        bottom=MARGIN_BOTTOM,
        row_labels=False,
        has_title=False,
        has_cbar=False,
        has_cbar_label=False,
        cbar_height=CBAR_HEIGHT,
        cbar_gap=CBAR_GAP,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_heights = list(row_heights) if row_heights else [panel_h] * n_rows
        self.col_widths = list(col_widths) if col_widths else [panel_w] * n_cols
        if len(self.row_heights) != n_rows or len(self.col_widths) != n_cols:
            raise ValueError("row_heights/col_widths must match the grid shape")

        self.wspace = wspace
        self.hspace = hspace
        self.has_cbar = has_cbar
        if has_cbar and cbar_height <= 0:
            raise ValueError("cbar_height must be positive")
        if has_cbar and cbar_gap < 0:
            raise ValueError("cbar_gap cannot be negative")
        self.cbar_height = cbar_height
        self.cbar_gap = cbar_gap

        if left is None:
            left = MARGIN_LEFT + (ROW_LABEL_SPACE if row_labels else 0.0)
        self.left_in = left
        self.right_in = right
        self.top_in = top + (TITLE_SPACE if has_title else 0.0)

        if has_cbar:
            self.cbar_y0 = (
                bottom + CBAR_TICK_SPACE + (CBAR_LABEL_SPACE if has_cbar_label else 0.0)
            )
            self.bottom_in = self.cbar_y0 + self.cbar_height + self.cbar_gap
        else:
            self.cbar_y0 = None
            self.bottom_in = bottom

        self.grid_w = sum(self.col_widths) + (n_cols - 1) * wspace
        self.grid_h = sum(self.row_heights) + (n_rows - 1) * hspace
        self.fig_w = self.left_in + self.grid_w + self.right_in
        self.fig_h = self.top_in + self.grid_h + self.bottom_in
        self._cbar_gs = None

    def fx(self, inches):
        """Convert a horizontal length in inches to a figure fraction."""
        return inches / self.fig_w

    def fy(self, inches):
        """Convert a vertical length in inches to a figure fraction."""
        return inches / self.fig_h

    def make_figure(self, **kwargs):
        """Return a figure sized so each panel is exactly its requested size."""
        return plt.figure(figsize=(self.fig_w, self.fig_h), **kwargs)

    def _panel_gridspec(self, fig, subplot_spec=None):
        """Build the GridSpec containing the actual panels."""
        kwargs = dict(
            width_ratios=self.col_widths,
            height_ratios=self.row_heights,
            # Matplotlib measures these against the mean panel size.
            wspace=self.wspace / np.mean(self.col_widths),
            hspace=self.hspace / np.mean(self.row_heights),
        )
        if subplot_spec is not None:
            return GridSpecFromSubplotSpec(
                self.n_rows, self.n_cols, subplot_spec=subplot_spec, **kwargs
            )
        return GridSpec(
            self.n_rows,
            self.n_cols,
            figure=fig,
            left=self.fx(self.left_in),
            right=self.fx(self.left_in + self.grid_w),
            bottom=self.fy(self.bottom_in),
            top=self.fy(self.bottom_in + self.grid_h),
            **kwargs,
        )

    def make_gridspec(self, fig):
        """Return panel slots and reserve automatic colorbars in GridSpec."""
        self._cbar_gs = None
        if not self.has_cbar:
            return self._panel_gridspec(fig)

        # The outer grid gives the panel block and colorbar row independent
        # heights and their own gap. GridSpec expresses the gap as a fraction
        # of its mean row height, hence the conversion from inches here.
        cbar_hspace = 2 * self.cbar_gap / (self.grid_h + self.cbar_height)
        outer = GridSpec(
            2,
            1,
            figure=fig,
            left=self.fx(self.left_in),
            right=self.fx(self.left_in + self.grid_w),
            bottom=self.fy(self.cbar_y0),
            top=self.fy(self.bottom_in + self.grid_h),
            height_ratios=(self.grid_h, self.cbar_height),
            hspace=cbar_hspace,
        )
        self._cbar_gs = GridSpecFromSubplotSpec(
            1,
            self.n_cols,
            subplot_spec=outer[1, 0],
            width_ratios=self.col_widths,
            wspace=self.wspace / np.mean(self.col_widths),
        )
        return self._panel_gridspec(fig, outer[0, 0])

    def colorbar_spec(self, first_col=0, last_col=None):
        """GridSpec slot for a colorbar spanning the selected panel columns."""
        if not self.has_cbar:
            raise ValueError("layout was built with has_cbar=False")
        last_col = self.n_cols - 1 if last_col is None else last_col
        if self._cbar_gs is None:
            raise RuntimeError("call make_gridspec before requesting a colorbar slot")
        return self._cbar_gs[0, first_col : last_col + 1]

    def title_y(self, fig=None, pad=0.18):
        """Figure-fraction y for a top-aligned title."""
        return self.fy(self.fig_h - pad)


class NestedLayout:
    """A panel grid occupying one cell of a parent grid.

    Same interface as GridLayout, so a figure function does not care which it
    was given. Placement is proportional rather than inch-exact, because the
    canvas belongs to the parent figure.

    Parameters
    ----------
    n_rows, n_cols : int
        Shape of the panel grid.
    spec : matplotlib.gridspec.SubplotSpec
        The parent cell to subdivide.
    row_heights, col_widths : sequence of float or None
        Relative sizes, used as GridSpec ratios.
    wspace, hspace : float
        Gaps, as fractions of the mean panel size.
    has_cbar : bool
        Reserve a thin extra row at the bottom for colorbars.
    cbar_frac : float
        Height of that row relative to one panel.
    """

    owns_figure = False

    def __init__(
        self,
        n_rows,
        n_cols,
        spec,
        row_heights=None,
        col_widths=None,
        wspace=0.05,
        hspace=0.12,
        has_cbar=False,
        cbar_frac=0.08,
        **ignored,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.spec = spec
        self.row_heights = list(row_heights) if row_heights else [1.0] * n_rows
        self.col_widths = list(col_widths) if col_widths else [1.0] * n_cols
        self.wspace = wspace
        self.hspace = hspace
        self.has_cbar = has_cbar
        self.cbar_frac = cbar_frac
        self._gs = None

    def make_figure(self, **kwargs):
        """The parent already owns a figure; nested layouts never make one."""
        raise TypeError("a NestedLayout draws into an existing figure")

    def make_gridspec(self, fig):
        """Subdivide the parent cell, reserving a colorbar row if asked."""
        heights = self.row_heights + ([self.cbar_frac] if self.has_cbar else [])
        self._gs = GridSpecFromSubplotSpec(
            len(heights),
            self.n_cols,
            subplot_spec=self.spec,
            width_ratios=self.col_widths,
            height_ratios=heights,
            wspace=self.wspace,
            hspace=self.hspace,
        )
        return self._gs

    def colorbar_spec(self, first_col=0, last_col=None):
        """GridSpec slot in the reserved bottom row for a colorbar."""
        if not self.has_cbar:
            raise ValueError("layout was built with has_cbar=False")
        if self._gs is None:
            raise RuntimeError("call make_gridspec before requesting a colorbar slot")
        last_col = self.n_cols - 1 if last_col is None else last_col
        return self._gs[-1, first_col : last_col + 1]

    def title_y(self, fig, pad=0.01):
        """Figure-fraction y just above the parent cell."""
        return self.spec.get_position(fig).y1 + pad


def open_layout(n_rows, n_cols, fig=None, spec=None, layout=None, **layout_kwargs):
    """Resolve the three ways of calling a figure function into one triple.

    This is the first line of every figure function in the package, and the
    only place the three calling conventions are distinguished.

    Parameters
    ----------
    n_rows, n_cols : int
        Shape of the panel grid.
    fig : matplotlib.figure.Figure or None
        Existing figure. Required with `spec` only if the spec's figure is not
        yet attached.
    spec : matplotlib.gridspec.SubplotSpec or None
        Parent cell to draw into, which selects a NestedLayout.
    layout : GridLayout or NestedLayout or None
        A layout you built yourself, which wins over both of the above.
    **layout_kwargs
        Passed to whichever layout class is constructed. For a new
        `GridLayout`, `cbar_height` and `cbar_gap` are measured in inches;
        `NestedLayout` instead uses relative `cbar_frac`.

    Returns
    -------
    (fig, gs, layout)
    """
    if layout is None:
        if spec is not None:
            layout = NestedLayout(n_rows, n_cols, spec, **layout_kwargs)
        else:
            layout = GridLayout(n_rows, n_cols, **layout_kwargs)

    if fig is None:
        if layout.owns_figure:
            fig = layout.make_figure()
        elif spec is not None:
            fig = spec.get_gridspec().figure
        if fig is None:
            raise ValueError("nested layouts need a fig, or a spec attached to one")

    return fig, layout.make_gridspec(fig), layout


# --------------------------------------------------------------------------
# Panels: what every figure function returns
# --------------------------------------------------------------------------


@dataclass
class Panels:
    """The result of a figure function.

    Unpacks as ``fig, axes = result`` for the common case; `artists` and
    `layout` stay available as attributes for colorbars and further nesting.
    """

    fig: object
    axes: object
    layout: object = None
    artists: object = None
    extras: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.fig, self.axes))

    @property
    def flat(self):
        """The axes as a flat list, row-major."""
        return list(np.atleast_2d(self.axes).ravel())

    def colorbar_ax(self, first_col=0, last_col=None):
        """Add axes in the reserved GridSpec colorbar row.

        Reserve the row with ``panel_grid(..., colorbar=True)``. For a figure
        with caller-owned axes, create a colorbar axes in its GridSpec instead.
        """
        if self.layout is None:
            raise ValueError("caller-owned axes need a caller-owned colorbar axes")
        return self.fig.add_subplot(self.layout.colorbar_spec(first_col, last_col))


# --------------------------------------------------------------------------
# Axes construction and the panel loop
# --------------------------------------------------------------------------


def _projection_grid(projection, n_rows, n_cols):
    """Normalise `projection` into an (n_rows, n_cols) object array.

    A scalar applies to every panel. A sequence of length n_rows applies one
    per row, which is how a figure gets maps on top and ordinary axes below.
    A nested sequence of shape (n_rows, n_cols) applies one per panel.
    """
    grid = np.empty((n_rows, n_cols), dtype=object)
    if not isinstance(projection, (list, tuple, np.ndarray)):
        grid[:] = projection
        return grid

    arr = np.empty(len(projection), dtype=object)
    arr[:] = list(projection)
    if arr.ndim == 1 and len(arr) == n_rows and not isinstance(
        projection[0], (list, tuple)
    ):
        for r in range(n_rows):
            grid[r, :] = arr[r]
        return grid

    flat = [p for row in projection for p in row]
    if len(flat) != n_rows * n_cols:
        raise ValueError(
            f"projection must be a scalar, {n_rows} per-row values, or "
            f"{n_rows}x{n_cols} per-panel values"
        )
    grid[:] = np.array(flat, dtype=object).reshape(n_rows, n_cols)
    return grid


def _share_key(mode, row, col):
    """Group key for sharing: None, one group, per row, or per column."""
    if not mode:
        return None
    if mode is True:
        return ("all",)
    if mode == "row":
        return ("row", row)
    if mode == "col":
        return ("col", col)
    raise ValueError("sharex/sharey must be True, False, 'row' or 'col'")


def make_axes(
    fig,
    gs,
    n_rows,
    n_cols,
    projection=None,
    row_offset=0,
    col_offset=0,
    sharex=False,
    sharey=False,
):
    """2D object array of axes covering gs[row_offset:, col_offset:].

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the axes to.
    gs : matplotlib.gridspec.GridSpec or GridSpecFromSubplotSpec
        Grid the axes are placed on.
    n_rows, n_cols : int
        Shape of the block to create.
    projection : object or sequence or None
        One projection for the whole block, one per row, or one per panel. Per
        row is what a mixed figure needs: maps across the top, ordinary axes
        below.
    row_offset, col_offset : int
        Where the block starts in `gs`, so one gridspec can hold several blocks
        drawn from different data.
    sharex, sharey : bool or {'row', 'col'}
        Share limits across the whole block, within each row, or within each
        column. Use 'row' rather than True when the rows hold different kinds
        of panel: a map and a histogram have no axis worth sharing.

    Returns
    -------
    numpy.ndarray
        Object array of axes, shape (n_rows, n_cols).
    """
    axes = np.empty((n_rows, n_cols), dtype=object)
    projections = _projection_grid(projection, n_rows, n_cols)
    x_refs, y_refs = {}, {}

    for r in range(n_rows):
        for c in range(n_cols):
            kwargs = {}
            if projections[r, c] is not None:
                kwargs["projection"] = projections[r, c]

            xk, yk = _share_key(sharex, r, c), _share_key(sharey, r, c)
            if xk in x_refs:
                kwargs["sharex"] = x_refs[xk]
            if yk in y_refs:
                kwargs["sharey"] = y_refs[yk]

            ax = fig.add_subplot(gs[row_offset + r, col_offset + c], **kwargs)
            axes[r, c] = ax
            if xk is not None:
                x_refs.setdefault(xk, ax)
            if yk is not None:
                y_refs.setdefault(yk, ax)

    return axes


def _panel_axes(axes, n_rows, n_cols):
    """Return caller-owned axes as a row-major grid of the requested shape."""
    axes = np.asarray(axes, dtype=object)
    if axes.size != n_rows * n_cols:
        raise ValueError(
            f"received {axes.size} axes for a {(n_rows, n_cols)} panel grid"
        )
    return axes.reshape(n_rows, n_cols)


def _share_axes(axes, sharex, sharey):
    """Apply the same sharing groups to caller-owned axes as new axes."""
    x_refs, y_refs = {}, {}
    for (row, col), ax in np.ndenumerate(axes):
        xk, yk = _share_key(sharex, row, col), _share_key(sharey, row, col)
        if xk is not None:
            if xk in x_refs:
                ref = x_refs[xk]
                if not ax.get_shared_x_axes().joined(ax, ref):
                    ax.sharex(ref)
            else:
                x_refs[xk] = ax
        if yk is not None:
            if yk in y_refs:
                ref = y_refs[yk]
                if not ax.get_shared_y_axes().joined(ax, ref):
                    ax.sharey(ref)
            else:
                y_refs[yk] = ax


def panel_grid(
    n_rows,
    n_cols,
    *,
    colorbar=False,
    axes=None,
    ax=None,
    fig=None,
    spec=None,
    layout=None,
    projection=None,
    sharex=False,
    sharey=False,
    gs=None,
    **layout_kwargs,
):
    """Create or accept a grid of axes; return ``Panels(fig, axes, layout)``.

    ``axes`` is always a 2-D array with shape ``(n_rows, n_cols)``. This
    function does not plot or set titles::

        fig, axes = panel_grid(3, 2)
        for row in range(3):
            for col in range(2):
                axes[row, col].plot(data[row, col])

    To reserve a GridSpec row for a horizontal colorbar::

        panels = panel_grid(1, 1, colorbar=True, cbar_height=.2)
        image = panels.axes[0, 0].imshow(data)
        panels.fig.colorbar(
            image, cax=panels.colorbar_ax(), orientation="horizontal"
        )

    For an existing figure, pass its axes directly::

        fig, axes = panel_grid(3, 2, axes=my_axes)

    Parameters
    ----------
    n_rows, n_cols : int
        Number of panel rows and columns. Both must be positive.
    ax, axes : matplotlib.axes.Axes or array-like, optional
        Existing target axes. ``ax`` is the one-panel shortcut; ``axes`` must
        have ``n_rows * n_cols`` axes in row-major order. Their figure is
        inferred when ``fig`` is omitted.
    colorbar : bool
        Reserve a GridSpec row for colorbars when creating the panel axes.
        Use ``panels.colorbar_ax()`` to place an axes in that row.
    fig, gs, spec, layout, **layout_kwargs
        Layout inputs used only when this function creates the axes. ``gs``
        needs its owning ``fig``; otherwise ``open_layout`` resolves a whole
        figure, a nested spec, or a supplied layout.
    projection, sharex, sharey
        `projection` is used when this call builds axes. Sharing is applied
        whether axes are created here or supplied by the caller.

    Returns
    -------
    Panels
    """
    if any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
        for n in (n_rows, n_cols)
    ):
        raise ValueError("n_rows and n_cols must be positive integers")

    if ax is not None:
        if axes is not None:
            raise ValueError("pass either ax or axes, not both")
        axes = ax

    if colorbar:
        if axes is not None or gs is not None:
            raise ValueError(
                "colorbar=True needs a grid created here; for caller-owned "
                "axes, create cax in your GridSpec"
            )
        if layout is not None and not layout.has_cbar:
            raise ValueError("the supplied layout must reserve a colorbar row")
        layout_kwargs["has_cbar"] = True

    if axes is None:
        if gs is None:
            fig, gs, layout = open_layout(
                n_rows, n_cols, fig=fig, spec=spec, layout=layout, **layout_kwargs
            )
        elif fig is None:
            raise ValueError("pass a fig alongside gs")
        axes = make_axes(
            fig, gs, n_rows, n_cols, projection=projection, sharex=sharex, sharey=sharey
        )
    else:
        axes = _panel_axes(axes, n_rows, n_cols)
        axes_fig = axes.flat[0].figure
        if any(candidate.figure is not axes_fig for candidate in axes.flat):
            raise ValueError("all supplied axes must belong to the same figure")
        if fig is None:
            fig = axes_fig
        elif fig is not axes_fig:
            raise ValueError("fig does not own the supplied axes")
        _share_axes(axes, sharex, sharey)

    return Panels(
        fig=fig, axes=axes, layout=layout,
        artists=np.full((n_rows, n_cols), None, dtype=object),
    )


# --------------------------------------------------------------------------
# Decoration: applied to axes that already exist
# --------------------------------------------------------------------------


def label_cols(axes, values, fmt=str, pad=14, fontsize=11, fontweight="bold"):
    """Title the top row of a panel grid. None values are skipped."""
    for ax, val in zip(np.atleast_2d(axes)[0], values):
        if val is not None:
            ax.set_title(fmt(val), pad=pad, fontsize=fontsize, fontweight=fontweight)


def label_rows(
    axes, values, fmt=str, x=-0.05, fontsize=11, fontweight="bold", rotate=False
):
    """Label the first column of a panel grid. None values are skipped.

    `rotate` uses the y axis label instead of horizontal text outside the axes,
    which is cheaper on horizontal space but competes with a real y label.
    """
    for ax, val in zip(np.atleast_2d(axes)[:, 0], values):
        if val is None:
            continue
        if rotate:
            ax.set_ylabel(fmt(val), fontsize=fontsize, fontweight=fontweight)
        else:
            ax.annotate(
                fmt(val),
                xy=(x, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=fontsize,
                fontweight=fontweight,
            )


def tag_panels(axes, start=0, x=0.02, y=0.98, fontsize=10, **kwargs):
    """Letter panels a) b) c) in row-major order, then aa) ab) ... after z); return the next index.

    Chain blocks in a mixed figure with
    ``n = tag_panels(top); tag_panels(bottom, start=n)``.
    """
    letters = string.ascii_lowercase
    i = start
    for ax in np.atleast_2d(axes).ravel():
        tag = f"{letters[i]})" if i < len(letters) else f"{letters[i // len(letters) - 1]}{letters[i % len(letters)]})"
        ax.text(x, y, tag, transform=ax.transAxes, fontsize=fontsize, va="top", **kwargs)
        i += 1
    return i


def hide_inner_labels(axes, x=True, y=True):
    """Strip tick labels from all but the bottom row and left column.

    Only safe when the panels genuinely share limits: pass sharex/sharey to
    `panel_grid`, or set the limits yourself. Hiding ticks on axes with
    independent scales makes panels look comparable when they are not.
    """
    axes = np.atleast_2d(axes)
    if x:
        for ax in axes[:-1, :].ravel():
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
    if y:
        for ax in axes[:, 1:].ravel():
            ax.tick_params(axis="y", labelleft=False, left=False)


def style_ax(ax, scale=1.0, tight_x=False, grid=True):
    """Gridlines behind the data, no top/right spines, optional text scaling.

    Call after labels, title and legend are set: `set_title` and `legend`
    reapply the rcParams font sizes and would undo the scaling. Text that is
    absent (empty title, empty axis label, no ticks, no legend) is left alone.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    scale : float
        Multiplier applied to the current axis label, tick label, title and
        legend entry font sizes.
    tight_x : bool
        Autoscale x tightly. Off by default because it silently overrides an
        x limit set by the caller.
    grid : bool
        Draw gridlines. Off for slide figures carrying their own reference
        lines.
    """
    if grid:
        ax.grid(axis="both", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    if ax.get_title():
        ax.title.set_fontsize(ax.title.get_fontsize() * scale)
    if ax.get_xlabel():
        ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * scale)
    if ax.get_ylabel():
        ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * scale)

    xticklabels = ax.get_xticklabels()
    if xticklabels:
        ax.tick_params(axis="x", labelsize=xticklabels[0].get_fontsize() * scale)
    yticklabels = ax.get_yticklabels()
    if yticklabels:
        ax.tick_params(axis="y", labelsize=yticklabels[0].get_fontsize() * scale)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * scale)

    if tight_x:
        ax.autoscale(axis="x", tight=True)


def add_colorbar(
    fig,
    mappable,
    cax,
    levels=None,
    label=None,
    ticks=None,
    tick_step=2,
    labelsize=9,
    fontsize=10,
    discrete=False,
    ticklabels=None,
):
    """Horizontal colorbar in `cax`, ticked at every `tick_step` level.

    With ``discrete=True``, `levels` are class edges: every class gets an
    equal-width block with its tick at the centre, labelled by `ticklabels`
    (default: the centre values, as integers where they are whole), and the
    bar has no extension triangles.
    """
    if cax is None or cax.figure is not fig:
        raise ValueError("cax must belong to fig")
    kwargs = {}
    if discrete:
        if levels is None:
            raise ValueError("a discrete colorbar needs the class edges as levels")
        edges = np.asarray(levels, dtype=float)
        ticks = (edges[:-1] + edges[1:]) / 2 if ticks is None else ticks
        if ticklabels is None:
            ticklabels = [f"{t:g}" for t in ticks]
        kwargs = dict(boundaries=edges, extend="neither", spacing="uniform")
    elif ticks is None and levels is not None:
        ticks = np.asarray(levels)[::tick_step]
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, **kwargs)
    if ticklabels is not None:
        cb.set_ticks(ticks, labels=ticklabels)
        cb.ax.tick_params(length=0)
    cb.ax.tick_params(labelsize=labelsize)
    if label:
        cb.set_label(label, fontsize=fontsize, labelpad=4)
    return cb


def add_suptitle(fig, layout, title, fontsize=13, fontweight="bold"):
    """Title positioned by the layout, whether it owns the figure or a cell."""
    return fig.text(
        0.5 if layout.owns_figure else layout.spec.get_position(fig).x0 + 0.001,
        layout.title_y(fig),
        title,
        ha="center" if layout.owns_figure else "left",
        va="bottom" if not layout.owns_figure else "top",
        fontsize=fontsize,
        fontweight=fontweight,
    )


def reveal_legend(
    fig,
    order,
    colors,
    labels,
    revealed,
    ncols=3,
    dim_handle=0.12,
    dim_text=0.22,
    loc="outside upper center",
    lw=2.4,
    **kwargs,
):
    """Full legend with unrevealed entries dimmed rather than absent.

    Every frame carries every entry, so the legend occupies the same space
    throughout and the axes below it never move. Building the legend up entry
    by entry would resize it and shift the plot on every frame.

    The 'outside' locations require the figure to use constrained layout.
    """
    handles = [
        Line2D(
            [],
            [],
            color=colors[key],
            lw=lw,
            label=labels[key],
            alpha=1.0 if key in revealed else dim_handle,
        )
        for key in order
    ]
    legend = fig.legend(handles=handles, loc=loc, ncols=ncols, frameon=False, **kwargs)
    for key, text in zip(order, legend.get_texts()):
        text.set_alpha(1.0 if key in revealed else dim_text)
    return legend
