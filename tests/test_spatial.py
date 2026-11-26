"""Area means and regional means."""

import numpy as np
import xarray as xr

from extant.statistics import spatial


def test_area_mean_of_a_dataset_matches_its_dataarrays():
    da = xr.DataArray(np.random.default_rng(11).standard_normal((2, 2, 3)), dims=("season", "lat", "lon"),
                      coords={"season": ["DJF", "JJA"], "lat": [-80.0, -70.0], "lon": [0.0, 120.0, 240.0]})
    ds = xr.Dataset({"a": da, "b": 2 * da})
    xr.testing.assert_allclose(spatial.area_mean(ds)["b"], spatial.area_mean(2 * da))


def test_regional_mean_is_cos_lat_weighted():
    lat = np.array([-80.0, -70.0, -60.0])
    da = xr.DataArray(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": [0.0, 180.0]})
    weights = np.cos(np.deg2rad([-80.0, -70.0]))
    expected = (1 * weights[0] + 2 * weights[1]) / weights.sum()
    np.testing.assert_allclose(spatial.regional_mean(da, {"lat": slice(-90, -65)}), expected)


def test_regional_means_of_a_tree():
    """One value per region along a region dim, the same numbers as one region at a time; the root stays empty."""
    da = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("lat", "lon"), coords={"lat": [-80.0, -70.0, -60.0], "lon": [0.0, 180.0]})
    regions = {"south": {"lat": slice(-90, -65)}, "north": {"lat": slice(-65, -50)}}
    tree = spatial.regional_means(xr.DataTree.from_dict({"A/historical": da.to_dataset(name="tas")}), regions)
    tas = tree["A/historical"].tas
    assert list(tas["region"].values) == list(regions)
    np.testing.assert_allclose(tas.sel(region="south"), spatial.regional_mean(da, regions["south"]))
    np.testing.assert_allclose(tas.sel(region="north"), 4.5)
