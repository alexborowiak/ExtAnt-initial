"""The sea ice's shared numbers, and the close-ups of how a field goes onto the 2.5° grid.

The regridding, the extent, the edge and the zones are worked out in notebook calc_04, with checks beside each step.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from extant import sea_ice  # noqa: E402


def test_cap_area():
    assert sea_ice.cap_area(-90.0) == 0
    np.testing.assert_allclose(sea_ice.cap_area(0.0), 2 * np.pi * sea_ice.EARTH_RADIUS ** 2)
    #(c): Antarctica alone (about 14 million km²) as a cap reaches about 71°S, as the paper says
    assert sea_ice.cap_area(-72.0) < 14.0e6 < sea_ice.cap_area(-70.0)


def test_regridding_close_ups():
    """A field's own points, then the 2.5° cells with how many points went into each."""
    from extant.plots import sea_ice as sip

    lat = xr.DataArray(np.arange(-80.0, -49.9, 1.0), dims="lat", coords={"lat": np.arange(-80.0, -49.9, 1.0)})
    lon = xr.DataArray(np.arange(0.0, 60.0, 1.0), dims="lon", coords={"lon": np.arange(0.0, 60.0, 1.0)})
    native = xr.where(lat < -61.25, 80.0, 0.0).broadcast_like(lon).transpose("lat", "lon").where(lat > -75)
    grid_lat, grid_lon = np.arange(-87.5, -39.9, 2.5), np.arange(-180.0, 180.0, 2.5)
    on_grid = native.interp(lat=grid_lat, lon=grid_lon, method="nearest")
    lat2d, lon2d = xr.broadcast(lat, lon)
    dots = sip.plot_native_points(native, lat2d, lon2d, on_grid)
    assert dots.get_offsets().shape[0] > 0
    sip.plot_grid_cells(native, lat2d, lon2d, on_grid)
    plt.close("all")
