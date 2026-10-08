"""Anything on a projection. The only module that needs cartopy.

Use ``draw_polar_contour`` for one axes and ``polar_grid`` for a grid of maps.

For mixed figures, pass caller-owned ``axes`` to ``polar_grid``.
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from .core import (
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

    #(c): Light enough not to compete with the contour lines in a small, printed panel
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.4,
        color="gray",
        alpha=0.45,
        linestyle="--",
    )
    gl.n_steps = 90
    return ax

@plot("axes")
def draw_polar_contour(
    ax,
    da,
    levels,
    cmap="RdBu_r",
    norm=None,
    lat_name="lat",
    lon_name="lon",
    extent=DEFAULT_EXTENT,
    discrete=False,
    extend="both",
):
    """Filled and line contours of a 2-D field on polar axes.

    With ``discrete=True`` the field is drawn cell by cell with pcolormesh
    instead. Use it for categorical fields (classes, counts): contourf
    interpolates between cells, so a class-0 cell next to a class-2 cell grows
    a false class-1 band between them.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Target axes.
    da : xarray.DataArray
        Field with latitude and longitude dimensions.
    levels : array_like
        Contour levels.
    cmap : str or Colormap, default="RdBu_r"
        Colormap for filled contours.
    norm : matplotlib.colors.Normalize, optional
        Normalisation used to map values to colours.
    lat_name, lon_name : str
        Latitude and longitude coordinate names.
    extent : sequence of float
        Passed to `setup_polar_ax`.
    discrete : bool
        Draw cells with pcolormesh rather than contours; see above.
    extend : {"both", "min", "max", "neither"}
        Which ends of the colour scale values beyond the levels are drawn at,
        with an arrow on the colorbar. "max" for a field that cannot go below
        its first level, e.g. a spread starting at 0.

    Returns
    -------
    matplotlib.contour.QuadContourSet or matplotlib.collections.QuadMesh
        Colorbar mappable.
    """
    da = da.transpose(lat_name, lon_name)
    data_cyclic, lons_cyclic = add_cyclic_point(da.values, coord=da[lon_name].values)
    lats = da[lat_name].values

    if discrete:
        if norm is None:
            norm = BoundaryNorm(levels, plt.get_cmap(cmap).N)
        mesh = ax.pcolormesh(
            lons_cyclic,
            lats,
            np.ma.masked_invalid(data_cyclic),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
        setup_polar_ax(ax, extent)
        return mesh

    cf = ax.contourf(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=extend,
    )

    ax.contour(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        colors="k",
        linewidths=0.25,
        alpha=0.35,
    )

    setup_polar_ax(ax, extent)
    return cf

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

@plot("figure")
def polar_grid(da, row_dim=None, col_dim=None, sel=None,
               levels=np.linspace(-3, 3, 13), cmap="RdBu_r", norm=None,
               lat_name="lat", lon_name="lon", label_fmt=None,
               title=None, cbar_label=None, projection=None, tag=True,
               discrete=False, ticklabels=None,
               fig=None, spec=None, layout=None, ax=None, axes=None, cax=None,
               cbar_height=None, cbar_gap=None, **layout_kwargs):
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
        Figure title. Added only when this function owns the figure; caller-
        owned axes and nested layouts leave the figure title to their caller.
    cbar_label : str or None
        Label under the colorbar, e.g. units.
    projection : cartopy.crs.Projection or None
        Defaults to SouthPolarStereo when this function creates the axes.
    tag : bool
        Letter the panels a) b) c).
    discrete : bool
        Categorical field: draw cells rather than contours, and give the
        colorbar one block per class, a tick at each block's centre and no
        extension triangles. `levels` are then the class edges, e.g.
        ``np.arange(-0.5, n + 0.5)`` for classes 0..n-1.
    ticklabels : sequence of str or None
        Labels for the class ticks when `discrete`; the class centres otherwise.
    ax :cartopy.mpl.geoaxes.GeoAxes or None
        Target for a one-panel grid. Its figure is inferred when `fig` is
        omitted.
    axes : array-like of cartopy.mpl.geoaxes.GeoAxes or None
        Targets for a multi-panel grid, in row-major order. They must already
        use the desired Cartopy projection.
    cax : matplotlib.axes.Axes or None
        Caller-owned colorbar axes. A colorbar is created automatically in a
        reserved GridSpec row only when this function creates its panel axes.
        For a composed figure, create this with ``fig.add_subplot(grid[...])``.
    cbar_height, cbar_gap : float or None
        Thickness and gap above an automatic colorbar, in inches. These apply
        when this function creates a `GridLayout`; nested layouts use
        `cbar_frac`, and a caller-owned `cax` controls its own geometry.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels

    Examples
    --------
    Put signal, noise and S/N into rows of one caller-owned GridSpec::

        import matplotlib.pyplot as plt
        import xarray as xr
        import cartopy.crs as ccrs
        from plotting_modules.core import panel_grid

        fields = xr.concat(
            (signal, noise, sn),
            dim=xr.IndexVariable("kind", ("signal", "noise", "sn")),
        )
        fig = plt.figure()
        grid = fig.add_gridspec(2, 1, height_ratios=[3, .08])
        axes = panel_grid(
            3, fields.sizes["period"], fig=fig, spec=grid[0],
            projection=ccrs.SouthPolarStereo(),
        ).axes
        cax = fig.add_subplot(grid[1])
        polar_grid(
            fields, row_dim="kind", col_dim="period", axes=axes, cax=cax,
            tag=False,
        )

    The figure owns every position in this example; ``polar_grid`` only draws
    into the supplied axes.
    """
    for dim, value in (sel or {}).items():
        if isinstance(value, (list, tuple, np.ndarray)):
            value = list(value)
        da = da.sel({dim: value})
    da = da.squeeze([
        dim for dim in da.dims
        if da.sizes[dim] == 1 and dim not in (lat_name, lon_name)
    ])
    unmapped = [
        dim for dim in da.dims
        if dim not in (lat_name, lon_name, row_dim, col_dim)
    ]
    if unmapped:
        raise ValueError(
            f"unmapped dimensions {unmapped}; pass them in sel, row_dim or col_dim"
        )
    row_values = list(da[row_dim].values) if row_dim in da.dims else [None]
    col_values = list(da[col_dim].values) if col_dim in da.dims else [None]
    fmt = label_fmt or {}
    caller_axes = ax is not None or axes is not None

    layout_kwargs.setdefault("row_labels", any(value is not None for value in row_values))
    layout_kwargs.setdefault("has_title", title is not None)
    layout_kwargs.setdefault("has_cbar_label", cbar_label is not None and cax is None)
    if cbar_height is not None or cbar_gap is not None:
        if caller_axes or cax is not None:
            raise ValueError("cbar_height/cbar_gap do not resize caller-owned cax")
        if spec is not None or layout is not None:
            raise ValueError(
                "cbar_height/cbar_gap need a new GridLayout; use cbar_frac for a "
                "nested layout or configure a supplied layout directly"
            )
        if cbar_height is not None:
            layout_kwargs["cbar_height"] = cbar_height
        if cbar_gap is not None:
            layout_kwargs["cbar_gap"] = cbar_gap

    panels = panel_grid(
        len(row_values), len(col_values),
        fig=fig, spec=spec, layout=layout, ax=ax, axes=axes,
        projection=projection or ccrs.SouthPolarStereo(),
        colorbar=cax is None and not caller_axes,
        **layout_kwargs,
    )
    for row, row_value in enumerate(row_values):
        for col, col_value in enumerate(col_values):
            field = da
            if row_value is not None:
                field = field.sel({row_dim: row_value})
            if col_value is not None:
                field = field.sel({col_dim: col_value})
            panels.artists[row, col] = draw_polar_contour(
                panels.axes[row, col], field, levels, cmap, norm, lat_name, lon_name,
                discrete=discrete,
            )

    label_cols(panels.axes, col_values, fmt.get(col_dim, str))
    label_rows(panels.axes, row_values, fmt.get(row_dim, str))
    if tag:
        tag_panels(panels.axes)

    if cax is None and not caller_axes:
        cax = panels.colorbar_ax()
    if cax is not None:
        panels.extras["cbar"] = add_colorbar(
            panels.fig,
            panels.artists[0, 0],
            cax,
            levels=levels,
            label=cbar_label,
            discrete=discrete,
            ticklabels=ticklabels,
        )
    if title and not caller_axes and panels.layout.owns_figure:
        add_suptitle(panels.fig, panels.layout, title)

    return panels


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


def stipple_mask(p, threshold=0.01):
    """Mask grid cells that fail a significance threshold.

    Args:
        p (xr.DataArray): p-values.
        threshold (float): significance level.

    Returns:
        xr.DataArray: boolean, True where p is not below threshold.
    """
    return p >= threshold


def plot_stipple(ax, mask, every=None, marker='.', size=1.0, color='k'):
    """Scatter stipple markers on a map axis where mask is True.

    Args:
        ax (GeoAxes): axis to draw on.
        mask (xr.DataArray): 2-D boolean mask with lat and lon dims.
        every (int): plot every nth grid cell in lat and lon, or None for all.
        marker (str): matplotlib marker.
        size (float): marker size in points squared.
        color (str): marker colour.

    Returns:
        PathCollection: the scatter artist.
    """
    if every is not None:
        mask = mask.isel(lat=slice(None, None, every), lon=slice(None, None, every))
    points = mask.transpose('lat', 'lon').values
    lon2d, lat2d = np.meshgrid(mask.lon.values, mask.lat.values)
    return ax.scatter(
        lon2d[points],
        lat2d[points],
        s=size,
        marker=marker,
        color=color,
        linewidths=0,
        transform=ccrs.PlateCarree(),
    )


def plot_hatch(ax, mask, hatch='...', color='k', linewidth=0.3):
    """Hatch regions of a map axis where mask is True.

    Args:
        ax (GeoAxes): axis to draw on.
        mask (xr.DataArray): 2-D boolean mask with lat and lon dims.
        hatch (str): matplotlib hatch pattern; repeat to increase density.
        color (str): hatch line colour.
        linewidth (float): hatch line width in points.

    Returns:
        QuadContourSet: the contour artist.
    """
    field = mask.transpose('lat', 'lon').values.astype(float)
    field, lon = add_cyclic_point(field, coord=mask.lon.values)
    with plt.rc_context({'hatch.linewidth': linewidth, 'hatch.color': color}):
        return ax.contourf(
            lon,
            mask.lat.values,
            field,
            levels=[0.5, 1.5],
            colors='none',
            hatches=[hatch],
            transform=ccrs.PlateCarree(),
        )
def map_panels(panels, da, draw, row_dim=None, col_dim=None, sel=None, **kwargs):
    """Apply a drawing function to every panel of a polar_grid figure."""
    if sel:
        da = da.sel(sel)

    artists = np.empty(panels.axes.shape, dtype=object)

    for i in range(panels.axes.shape[0]):
        for j in range(panels.axes.shape[1]):
            indexer = {}
            if row_dim is not None:
                indexer[row_dim] = i
            if col_dim is not None:
                indexer[col_dim] = j

            field = da.isel(indexer) if indexer else da
            artists[i, j] = draw(panels.axes[i, j], field, **kwargs)

    return artists
