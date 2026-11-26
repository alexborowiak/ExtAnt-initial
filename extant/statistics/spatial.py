"""Area weights, area means and regional means on the regular latitude-longitude grid.

A grid cell's area is proportional to the cosine of its latitude, so every mean or share of the map weights each cell
by it. Example: cells at 70°S and 80°S have weights cos 70° = 0.34 and cos 80° = 0.17, so the one at 70°S counts
twice as much.
"""

import numpy as np
import xarray as xr


def area_weights(lat):
    """Each grid cell's share of the area, up to a constant: cos(latitude). ``lat`` in degrees (a DataArray)."""
    return np.cos(np.deg2rad(lat))


def area_mean(da, lat="lat", lon="lon"):
    """Area-weighted mean over ``lat`` and ``lon`` of a DataArray or Dataset; booleans give the share of the area."""
    if isinstance(da, xr.DataArray) and da.dtype == bool:
        da = da.astype(float)
    return da.weighted(area_weights(da[lat])).mean((lat, lon))


def regional_mean(obj, selection):
    """Area-weighted mean over one region, e.g. ``{"lat": slice(-90, -65)}`` for everything south of 65°S.

    Works on DataArrays, Datasets and DataTrees alike.
    """
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(lambda ds: regional_mean(ds, selection) if ds.data_vars else ds)
    return area_mean(obj.sel(selection))


def regional_means(obj, regions):
    """``regional_mean`` of every region, along a new ``region`` dim; NaN for a region with no grid points.

    Works on DataArrays, Datasets and DataTrees alike.

    Args:
        obj: Data on (lat, lon, ...).
        regions (dict): Region name -> lat/lon selection, e.g. ``{"Southern Ocean": {"lat": slice(-65, -50)}}``.
    """
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(lambda ds: regional_means(ds, regions) if ds.data_vars else ds)
    return xr.concat([regional_mean(obj, selection) for selection in regions.values()],
                     dim=xr.Variable("region", list(regions)))
