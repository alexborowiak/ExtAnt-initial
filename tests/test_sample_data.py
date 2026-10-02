"""The notebook's pre-processing on the sample data (tests/data): the real files' calendars, units and grids."""

import numpy as np
import pytest
import xarray as xr

import era5_evaluation as ev
import open_xarray
import significance as sig
from quantile_calc import WINDOW, seasonal_mean
from xarray_calc import split_time

pytest.importorskip("netCDF4")


@pytest.fixture(scope="module")
def season_tree():
    """lesfmip_season_tree as the notebook builds it: °C, cut at 2014, seasonal means on (year, season)."""
    tree = open_xarray.open_lesfmip_sample(experiments=["hist-nat", "hist-GHG", "historical"])
    tree = tree.map_over_datasets(
        lambda ds: (ds - 273.15).astype("float32").drop_vars("height", errors="ignore").drop_attrs())
    tree = tree.sel(time=slice(None, "2014-12")).map_over_datasets(seasonal_mean)
    return tree.map_over_datasets(lambda ds: split_time(ds, ("year", "season"), "time") if ds.data_vars else ds)


@pytest.fixture(scope="module")
def era5_seasonal(season_tree):
    era5 = open_xarray.open_era5_sample()
    return ev.match_grid(ev.seasonal_means(era5), season_tree["HadGEM3-GC31-LL/historical"].tas)


def test_era5_sample_is_as_the_notebook_leaves_it():
    era5 = open_xarray.open_era5_sample()
    assert -80 < float(era5.mean()) < 0                                     # °C
    assert list(era5.time.dt.month.values[:12]) == list(range(1, 13))       # one value per month
    assert str(era5.time.values[0])[:10] == "1979-01-01"


def test_sample_tree_layout():
    tree = open_xarray.open_lesfmip_sample()
    assert set(tree.children) == {"CanESM5", "HadGEM3-GC31-LL"}
    assert {"hist-nat", "historical"} <= set(tree["CanESM5"].children)
    assert float(tree["CanESM5/historical"].tas.mean()) > 200   # K


def test_every_experiment_covers_the_same_whole_years(season_tree):
    """Both calendars (noleap, 360_day) and both record lengths (2014, 2020) end with DJF 2013."""
    for node in season_tree.leaves:
        assert int(node.year.min()) == 1850 and int(node.year.max()) == 2013, node.path
        assert not node.tas.isnull().any(), node.path
        assert node.tas.isel(year=slice(-WINDOW, None)).year.values[0] == 1993


def test_era5_sample_lines_up_with_the_models(season_tree, era5_seasonal):
    assert sorted(era5_seasonal.season.values) == ["DJF", "JJA", "MAM", "SON"]
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
