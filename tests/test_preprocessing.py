"""The pre-processing (monthly to seasonal means, ERA5 onto the model grid), on synthetic data and on the sample."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extant import era5_evaluation as ev
from extant import loading
from extant import preprocessing as pp
from extant import significance as sig
from extant.config import WINDOW

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Monthly to seasonal means, and ERA5 onto the model grid
# ---------------------------------------------------------------------------

def test_seasonal_means_matches_lesfmip_pipeline():
    """ERA5 (datetime64, month starts) gets the same (year, season) labels and values as the
    LESFMIP pipeline applied to the same numbers on a cftime mid-month axis."""
    n = 12 * 6
    values = RNG.standard_normal(n)
    era5 = xr.DataArray(values, dims="time", coords={"time": pd.date_range("1979-01-01", periods=n, freq="MS")})
    model_time = [cftime.DatetimeNoLeap(1979 + i // 12, i % 12 + 1, 16, 12) for i in range(n)]
    model = xr.DataArray(values, dims="time", coords={"time": model_time})

    ours = pp.seasonal_means(era5)
    theirs = pp.split_time(pp.seasonal_mean(model), ("year", "season"), "time")

    # The Jan-Feb stub (DJF 1978) is dropped as incomplete, and so is 1984, which has no complete DJF.
    assert list(ours.year.values) == [1979, 1980, 1981, 1982, 1983] and not ours.isnull().any()
    xr.testing.assert_allclose(ours, theirs)
    # DJF 1979 = Dec 1979 + Jan 1980 + Feb 1980, labelled by December's year.
    np.testing.assert_allclose(ours.sel(year=1979, season="DJF"), values[[11, 12, 13]].mean())


def test_seasonal_mean_keeps_whole_years_on_a_360_day_calendar():
    """A model record cut at the end of 2014 ends with DJF 2013, with every season covering the same years."""
    time = [cftime.Datetime360Day(y, m, 16) for y in range(2000, 2015) for m in range(1, 13)]
    da = xr.DataArray(np.arange(len(time), dtype=float), dims="time", coords={"time": time})
    seasons = pp.split_time(pp.seasonal_mean(da), ("year", "season"), "time")
    assert list(seasons.year.values) == list(range(2000, 2014)) and not seasons.isnull().any()


def test_seasonal_means_refuses_shifted_months():
    """convert_calendar('360_day', align_on='year') moves month starts into the previous month: 1 March reads 29 Feb."""
    era5 = xr.DataArray(np.arange(48.0), dims="time", coords={"time": pd.date_range("1979-01-01", periods=48, freq="MS")})
    shifted = era5.convert_calendar("360_day", align_on="year")
    assert list(shifted.time.dt.month.values[:4]) == [1, 2, 2, 3]
    with pytest.raises(ValueError, match="more than once"):
        pp.seasonal_means(shifted)
    repaired = pp.monthly_time_axis(shifted)
    xr.testing.assert_equal(pp.seasonal_means(repaired), pp.seasonal_means(era5))


def test_match_grid_snaps_matching_points():
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-87.5, -85.0, -82.5], "lon": [-180.0, -177.5, -175.0, -172.5]})
    obs = like.copy(data=RNG.standard_normal((3, 4))).assign_coords(lat=like.lat + 1e-6, lon=like.lon - 1e-6)
    snapped = pp.match_grid(obs, like)
    assert (snapped.lat.values == like.lat.values).all()
    np.testing.assert_array_equal(snapped.values, obs.values)


def test_match_grid_converts_longitudes_and_latitude_order():
    """ERA5 on 0-360 longitudes, north to south, and larger than the target: the same values, on the target's labels."""
    lat, lon = np.arange(-40.0, -91.0, -2.5), np.arange(0.0, 360.0, 2.5)
    obs = xr.DataArray(np.random.default_rng(0).standard_normal((lat.size, lon.size)), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-85.0, -82.5, -80.0], "lon": [-180.0, -177.5, 117.5, 177.5]})
    matched = pp.match_grid(obs, like)
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
    matched = pp.match_grid(obs, like)
    expected = like.lat + 10 * np.cos(np.deg2rad(like.lon))
    np.testing.assert_allclose(matched.transpose("lat", "lon"), expected.transpose("lat", "lon"), atol=0.01)


# ---------------------------------------------------------------------------
# The whole chain, on the sample (tests/data): the real files' calendars, units and grids
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def season_tree():
    """lesfmip_season_tree as the notebooks build it: °C, cut at 2014, seasonal means on (year, season)."""
    tree = loading.open_lesfmip_sample(experiments=["hist-nat", "hist-GHG", "historical"])
    return pp.seasonal_tree(tree, "tas").compute()


@pytest.fixture(scope="module")
def era5_seasonal(season_tree):
    return pp.seasonal_era5(loading.open_era5_sample(), "tas", like=season_tree)


def test_era5_sample_is_like_the_monthly_store():
    era5 = loading.open_era5_sample()
    assert 200 < float(era5.mean()) < 280                                   # K
    assert list(era5.time.dt.month.values[:12]) == list(range(1, 13))       # one value per month
    assert str(era5.time.values[0])[:10] == "1979-01-01"


def test_sample_tree_layout():
    tree = loading.open_lesfmip_sample()
    assert set(tree.children) == {"CanESM5", "HadGEM3-GC31-LL"}
    assert {"hist-nat", "historical"} <= set(tree["CanESM5"].children)
    assert float(tree["CanESM5/historical"].tas.mean()) > 200   # K


def test_every_experiment_covers_the_same_whole_years(season_tree):
    """Both calendars (noleap, 360_day) and both record lengths (2014, 2020) end with DJF 2013."""
    for node in season_tree.leaves:
        assert int(node.year.min()) == 1850 and int(node.year.max()) == 2013, node.path
        assert not node.tas.isnull().any(), node.path
        assert node.tas.isel(year=slice(-WINDOW, None)).year.values[0] == 1993
        assert -80 < float(node.tas.mean()) < 0, node.path   # °C
        assert node.tas.dtype == np.float32 and "height" not in node.coords


def test_era5_sample_lines_up_with_the_models(season_tree, era5_seasonal):
    assert sorted(era5_seasonal.season.values) == ["DJF", "JJA", "MAM", "SON"]
    assert int(era5_seasonal.year.min()) == 1979 and era5_seasonal.name == "tas"
    ensemble, obs = ev.align(season_tree["CanESM5/historical"].tas, era5_seasonal, years=slice(1979, 2014))
    assert int(obs.year.min()) == 1979 and int(obs.year.max()) == 2013
    assert int(obs.count()) == obs.size
    assert abs(float(ensemble.mean() - obs.mean())) < 10


def test_hist_nat_bootstrap_on_the_sample(season_tree):
    branch = season_tree["HadGEM3-GC31-LL"]
    result = sig.qrange_significance({"historical": branch["historical"].tas, "hist-GHG": branch["hist-GHG"].tas},
                                     branch["hist-nat"].tas, n_trials=200)
    assert set(result.dims) == {"experiment", "season", "lat", "lon"}
    assert result.qrange_pvalue.notnull().all()
    assert (result.null_lower < result.null_upper).all()
