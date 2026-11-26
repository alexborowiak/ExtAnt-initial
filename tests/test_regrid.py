"""One grid onto another."""

import numpy as np
import xarray as xr

from extant.utils import regrid

RNG = np.random.default_rng(42)


def test_match_grid_snaps_matching_points():
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-87.5, -85.0, -82.5], "lon": [-180.0, -177.5, -175.0, -172.5]})
    obs = like.copy(data=RNG.standard_normal((3, 4))).assign_coords(lat=like.lat + 1e-6, lon=like.lon - 1e-6)
    snapped = regrid.match_grid(obs, like)
    assert (snapped.lat.values == like.lat.values).all()
    np.testing.assert_array_equal(snapped.values, obs.values)


def test_match_grid_converts_longitudes_and_latitude_order():
    """ERA5 on 0-360 longitudes, north to south, and larger than the target: the same values, on the target's labels."""
    lat, lon = np.arange(-40.0, -91.0, -2.5), np.arange(0.0, 360.0, 2.5)
    obs = xr.DataArray(np.random.default_rng(0).standard_normal((lat.size, lon.size)), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-85.0, -82.5, -80.0], "lon": [-180.0, -177.5, 117.5, 177.5]})
    matched = regrid.match_grid(obs, like)
    np.testing.assert_array_equal(matched.lon.values, like.lon.values)
    expected = obs.sel(lat=like.lat, lon=[180.0, 182.5, 117.5, 177.5]).values
    np.testing.assert_array_equal(matched.values, expected)


def test_match_grid_interpolates_other_points():
    """Points between the source points are bilinearly interpolated, wrapping round in longitude."""
    lat, lon = np.arange(-90.0, -39.0, 2.0), np.arange(0.0, 360.0, 2.0)
    #(c): A field linear in latitude and in the longitude's cosine is interpolated almost exactly at 2° spacing
    field = (lat[:, None] + 10 * np.cos(np.deg2rad(lon))[None, :])
    obs = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    like = xr.DataArray(np.zeros((2, 4)), dims=("lat", "lon"),
                        coords={"lat": [-81.25, -61.25], "lon": [-179.0, -1.0, 90.0, 179.0]})
    matched = regrid.match_grid(obs, like)
    expected = like.lat + 10 * np.cos(np.deg2rad(like.lon))
    np.testing.assert_allclose(matched.transpose("lat", "lon"), expected.transpose("lat", "lon"), atol=0.01)
