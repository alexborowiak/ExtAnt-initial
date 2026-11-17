"""Area weights and area means on the regular latitude-longitude grid.

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
