"""The pre-processing (monthly to seasonal means, ERA5 onto the model grid) on the sample."""

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extant.utils import datatree
from extant import obs_evaluation as ev
from extant import preprocessing as pp
from extant.statistics import quantiles as qc
from extant.statistics import significance as sig
from extant.config import WINDOW

#(t): The sample in this repository: <model>_<experiment>_sample.nc (monthly tas in K) and era5_sample.nc
SAMPLE_DIR = Path(__file__).resolve().parent / "data"


def open_sample_tree(experiments=None):
    """The sample's model files as a /<model>/<experiment> DataTree, loaded, as the notebooks open them."""
    files = sorted(SAMPLE_DIR.glob("*_*_sample.nc"))
    names = [file.stem.removesuffix("_sample").rsplit("_", 1) for file in files]
    return xr.DataTree.from_dict({f"{model}/{experiment}": xr.open_dataset(file).load()
                                  for file, (model, experiment) in zip(files, names)
                                  if experiments is None or experiment in experiments})


# ---------------------------------------------------------------------------
# The whole chain, on the sample (tests/data): the real files' calendars, units and grids
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def season_tree():
    """lesfmip_season_tree as the notebooks build it: °C, cut at 2014, seasonal means on (year, season)."""
    tree = open_sample_tree(experiments=["hist-nat", "hist-GHG", "historical"])
    return pp.seasonal_tree(tree, "tas").compute()


@pytest.fixture(scope="module")
def era5_seasonal(season_tree):
    return pp.seasonal_era5(xr.open_dataarray(SAMPLE_DIR / "era5_sample.nc").load(), "tas", like=season_tree)


def test_era5_sample_is_like_the_monthly_store():
    era5 = xr.open_dataarray(SAMPLE_DIR / "era5_sample.nc").load()
    assert 200 < float(era5.mean()) < 280                                   # K
    assert list(era5.time.dt.month.values[:12]) == list(range(1, 13))       # one value per month
    assert str(era5.time.values[0])[:10] == "1979-01-01"


def test_sample_tree_layout():
    tree = open_sample_tree()
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
    """The bootstrap mapped over every model's hist-nat, as notebooks calc_02a and calc_02b run it."""
    trials = datatree.reduce_to_dataset(season_tree.match("*/hist-nat"), partial(sig.bootstrap_qrange, n_members=10, quantiles=(0.05, 0.95), n_trials=200, seed=0))
    assert set(trials.dims) == {"model", "experiment", "trial", "season", "lat", "lon"}
    assert trials.tas.notnull().all()
    hist_nat = season_tree["HadGEM3-GC31-LL/hist-nat"].tas
    null = trials.tas.sel(model="HadGEM3-GC31-LL", experiment="hist-nat") - qc.quantile_range(hist_nat)
    change = qc.quantile_range(season_tree["HadGEM3-GC31-LL/historical"].tas.isel(year=slice(-21, None))) - qc.quantile_range(hist_nat)
    assert sig.pvalue_two_sided(null, change).notnull().all()
