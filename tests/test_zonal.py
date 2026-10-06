"""Checks of zonal on a synthetic tree: one experiment shifted and widened, one no different from hist-nat."""

import numpy as np
import pytest
import xarray as xr

from extant import response_change as rc
from extant import significance as sig
from extant import zonal

RNG = np.random.default_rng(5)
LAT = np.array([-80.0, -70.0, -60.0])
LON = np.arange(0.0, 360.0, 45.0)
SEASONS = ["DJF", "JJA"]
YEARS = np.arange(1990, 2015)
YEARS_TESTED = 11


def _members(n, shift=0.0, scale=1.0):
    values = shift + scale * RNG.standard_normal((n, YEARS.size, len(SEASONS), LAT.size, LON.size))
    return xr.DataArray(values, dims=("member", "year", "season", "lat", "lon"),
                        coords={"member": np.arange(n), "year": YEARS, "season": SEASONS, "lat": LAT, "lon": LON})


@pytest.fixture(scope="module")
def tree():
    nodes = {}
    for model, n in (("A", 20), ("B", 12)):
        nodes[f"{model}/hist-nat"] = xr.Dataset({"tas": _members(n)})
        nodes[f"{model}/historical"] = xr.Dataset({"tas": _members(n, shift=2.0, scale=1.5)})
        nodes[f"{model}/hist-GHG"] = xr.Dataset({"tas": _members(n)})
    return xr.DataTree.from_dict(nodes)


@pytest.fixture(scope="module")
def width(tree):
    return zonal.width_test(tree, years=YEARS_TESTED, n_trials=200)


def test_width_test_is_the_zonal_mean_of_the_map_test(tree, width):
    branch = tree["A"]
    maps = sig.qrange_significance({e: branch[e].tas for e in ("historical", "hist-GHG")}, branch["hist-nat"].tas,
                                   years=YEARS_TESTED, n_trials=200)
    np.testing.assert_allclose(width.width_change.sel(model="A", experiment=maps.experiment.values),
                               maps.qrange_change.mean("lon").transpose(*width.width_change.sel(model="A").dims))
    np.testing.assert_allclose(width.hist_nat_width.sel(model="A"), maps.hist_nat_qrange.isel(experiment=0).mean("lon"))
    assert width.width_change.dims == ("model", "experiment", "season", "lat")
    assert width.null_lower.dims == ("model", "season", "lat")


def test_width_test_finds_the_widening_and_not_the_rest(width):
    assert (width.width_pvalue.sel(experiment="historical") < 0.05).all()
    assert float((width.width_pvalue.sel(experiment="hist-GHG") < 0.05).mean()) <= 0.25
    #(c): Averaging over longitude narrows the null: the zonal mean of 8 points varies less than one point
    assert (width.null_lower < 0).all() and (width.null_upper > 0).all()


def test_mean_test(tree):
    tested = zonal.mean_test(tree, years=YEARS_TESTED, n_permutations=500)
    expected = zonal.zonal_mean(rc.mean_response(tree, years=YEARS_TESTED))
    np.testing.assert_allclose(tested.mean_change.transpose(*expected.dims), expected.sel(experiment=tested.experiment))
    assert (tested.pvalue.sel(experiment="historical") < 0.05).all()
    assert float((tested.pvalue.sel(experiment="hist-GHG") < 0.05).mean()) <= 0.25


def test_zonal_summary(tree, width):
    mean = rc.mean_response(tree, years=YEARS_TESTED)
    pvalue = xr.full_like(mean, 0.01)
    qrange = xr.Dataset({"qrange_change": xr.full_like(mean, 1.0), "qrange_pvalue": pvalue})
    tails = rc.tail_changes(tree, years=YEARS_TESTED)
    summary = rc.change_summary(mean, pvalue, xr.full_like(mean, 3.0), qrange, tails)

    out = zonal.zonal_summary(summary, mean_tested=xr.Dataset({"pvalue": pvalue.mean("lon")}), width_tested=width)
    assert list(rc.EXTREMES) == [name for name in out.data_vars][:4]
    np.testing.assert_allclose(out.low_extreme_change, -tails.lower_tail_change.mean("lon").transpose(*out.low_extreme_change.dims))
    np.testing.assert_allclose(out.high_extreme_change - out.low_extreme_change,
                               tails.width_change.mean("lon").transpose(*out.low_extreme_change.dims))
    assert {"mean_pvalue", "width_pvalue", "hist_nat_width", "null_lower", "null_upper"} <= set(out.data_vars)
