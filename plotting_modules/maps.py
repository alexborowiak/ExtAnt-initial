"""Anything on a projection. The only module that needs cartopy.

Layers:
  Primitive   setup_polar_ax, draw_polar_contour — draw on one axes.
  Selection   select, panel_values, panel, prepare — turn a DataArray plus
              sel/row/col into the coordinates core.panel_grid iterates over.
  Figure      polar_grid — DataArray in, figure out.

Mixed figures skip polar_grid and pass draw_polar to core.panel_grid directly.
"""

from functools import partial

import numpy as np
import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from .core import (
    GridLayout,
    add_colorbar,
    add_suptitle,
    label_cols,
    label_rows,
    panel_grid,
    tag_panels,
)
from .utils import plot

DEFAULT_EXTENT = (-180, 180, -90, -50)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def setup_polar_ax(ax, extent=DEFAULT_EXTENT):
    """Circular boundary, coastlines and gridlines on a polar axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes to configure, expected to use a polar stereographic projection.
    extent : sequence of float
        Extent as (lon_min, lon_max, lat_min, lat_max) in PlateCarree.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_path = mpath.Path(
        np.column_stack([np.cos(theta), np.sin(theta)]) * 0.5 + 0.5
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.set_boundary(circle_path, transform=ax.transAxes)
    ax.coastlines(linewidth=0.7)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.n_steps = 90
    return ax

@plot("axes")
def draw_polar_contour(
    ax, da, levels, cmap="RdBu_r", lat_name="lat", lon_name="lon", extent=DEFAULT_EXTENT
):
    """Filled and line contours of a 2-D field on a polar axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Target axes.
    da : xarray.DataArray
        Field with latitude and longitude dimensions.
    levels : array_like
        Contour levels.
    cmap : str
        Colormap name.
    lat_name, lon_name : str
        Names of the latitude and longitude coordinates.
    extent : sequence of float
        Passed to `setup_polar_ax`.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The filled contour set, for use as a colorbar mappable.
    """
    da = da.transpose(lat_name, lon_name)
    data_cyclic, lons_cyclic = add_cyclic_point(da.values, coord=da[lon_name].values)
    lats = da[lat_name].values

    cf = ax.contourf(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    ax.contour(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=0.5,
    )

    setup_polar_ax(ax, extent)
    return cf


# --------------------------------------------------------------------------
# Turning a DataArray into panel coordinates
# --------------------------------------------------------------------------

def select(da, sel):
    """Apply a sel dict: a sequence subsets and orders, a scalar collapses."""
    for dim, value in sel.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            da = da.sel({dim: list(value)})
        else:
            da = da.sel({dim: value})
    return da

def panel_values(da, dim):
    """Values of `dim`, or [None] when there is no dimension to map."""
    if dim is None or dim not in da.dims:
        return [None]
    return list(np.atleast_1d(da[dim].values))

def panel(da, dim, value):
    """One panel's slice; a None value means nothing to select."""
    return da if value is None else da.sel({dim: value})

def prepare(da, sel=None, row_dim=None, col_dim=None, lat_name="lat", lon_name="lon"):
    """Apply sel, drop degenerate dims, return (da, row_vals, col_vals).

    Raises if any dimension is left neither collapsed by `sel` nor mapped to
    rows, columns, latitude or longitude — silently plotting the first element
    of an unmapped dimension is the failure mode this guards against.
    """
    da = select(da, sel) if sel else da
    squeezable = [
        d for d in da.dims if da.sizes[d] == 1 and d not in (lat_name, lon_name)
    ]
    da = da.squeeze(squeezable)

    unmapped = [d for d in da.dims if d not in (lat_name, lon_name, row_dim, col_dim)]
    if unmapped:
        raise ValueError(
            f"unmapped dimensions {unmapped}; pass them in sel, row_dim or col_dim"
        )

    return da, panel_values(da, row_dim), panel_values(da, col_dim)

def draw_polar(
    ax,
    row_val,
    col_val,
    da=None,
    row_dim=None,
    col_dim=None,
    levels=None,
    cmap="RdBu_r",
    lat_name="lat",
    lon_name="lon",
):
    """A `panel_grid` callback that contours one slice of `da`.

    Bind the trailing arguments with functools.partial and hand the result to
    `panel_grid`; see `polar_grid` for the usual case.
    """
    da_panel = panel(panel(da, row_dim, row_val), col_dim, col_val)
    return draw_polar_contour(ax, da_panel, levels, cmap, lat_name, lon_name)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

@plot("figure")
def polar_grid(da, row_dim=None, col_dim=None, sel=None,
               levels=np.linspace(-3, 3, 13), cmap="RdBu_r",
               lat_name="lat", lon_name="lon", label_fmt=None,
               title=None, cbar_label=None, projection=None, tag=True,
               fig=None, spec=None, layout=None, **layout_kwargs):
    """Grid of polar panels with two dimensions mapped to rows and columns.

    Every dimension other than latitude, longitude, row_dim and col_dim must be
    collapsed by sel.

    Parameters
    ----------
    da : xarray.DataArray
        Field with latitude and longitude dimensions.
    row_dim, col_dim : str or None
        Dimensions mapped to rows and columns; None or a scalar coordinate
        gives one row or column.
    sel : dict or None
        Coordinate selections applied first, e.g. {'experiment': 'historical'};
        a sequence value subsets and orders a dimension, a scalar collapses it.
    levels : array_like
        Contour levels, shared by every panel.
    cmap : str
        Colormap name.
    lat_name, lon_name : str
        Names of the latitude and longitude coordinates.
    label_fmt : dict or None
        Mapping of dimension name to a callable formatting its values.
    title : str or None
        Figure title. Ignored for a nested layout, which does not own the top
        of the canvas.
    cbar_label : str or None
        Label under the colorbar, e.g. units.
    projection : cartopy.crs.Projection or None
        Defaults to SouthPolarStereo.
    tag : bool
        Letter the panels a) b) c).
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    da, row_vals, col_vals = prepare(da, sel, row_dim, col_dim, lat_name, lon_name)
    fmt = label_fmt or {}

    layout_kwargs.setdefault("row_labels", any(v is not None for v in row_vals))
    layout_kwargs.setdefault("has_title", title is not None)
    layout_kwargs.setdefault("has_cbar", True)
    layout_kwargs.setdefault("has_cbar_label", cbar_label is not None)

    panels = panel_grid(
        row_vals,
        col_vals,
        partial(draw_polar, da=da, row_dim=row_dim, col_dim=col_dim,
                levels=levels, cmap=cmap, lat_name=lat_name, lon_name=lon_name),
        fig=fig, spec=spec, layout=layout,
        projection=projection or ccrs.SouthPolarStereo(),
        **layout_kwargs,
    )

    label_cols(panels.axes, col_vals, fmt.get(col_dim, str))
    label_rows(panels.axes, row_vals, fmt.get(row_dim, str))
    if tag:
        tag_panels(panels.axes)

    panels.extras["cbar"] = add_colorbar(
        panels.fig,
        panels.artists[0, 0],
        panels.layout.cbar_ax(panels.fig, 0, len(col_vals) - 1),
        levels=levels,
        label=cbar_label,
    )
    if title and panels.layout.owns_figure:
        add_suptitle(panels.fig, panels.layout, title)

    return panels
