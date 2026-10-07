"""Checks of zonal on a synthetic tree: one experiment shifted and widened, one no different from hist-nat."""

from functools import partial

import numpy as np
import pytest
import xarray as xr

from extant import datatree
from extant import quantiles as qc
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
    """The zonal width test as notebook 03 runs it: zonal means of the map change and of the bootstrap trials."""
    hist_nat_tree = tree.match("*/hist-nat")
    hist_nat_qrange = datatree.reduce_to_dataset(hist_nat_tree, qc.quantile_range).tas.sel(experiment="hist-nat", drop=True)
    final_qrange = datatree.reduce_to_dataset(tree.isel(year=slice(-YEARS_TESTED, None)), qc.quantile_range).tas
    change = zonal.zonal_mean((final_qrange - hist_nat_qrange).drop_sel(experiment="hist-nat"))
    trials = datatree.reduce_to_dataset(hist_nat_tree, partial(sig.bootstrap_qrange, years=YEARS_TESTED, n_trials=200))
    null = zonal.zonal_mean(trials.tas.sel(experiment="hist-nat", drop=True) - hist_nat_qrange)
    bounds = null.quantile([0.025, 0.975], dim="trial")
    return xr.Dataset({
        "width_change": change,
        "width_pvalue": sig.pvalue_two_sided(null, change),
        "hist_nat_width": zonal.zonal_mean(hist_nat_qrange),
        "null_lower": bounds.isel(quantile=0, drop=True),
        "null_upper": bounds.isel(quantile=1, drop=True),
    })


def test_zonal_mean_of_the_bootstrap_finds_the_widening_and_not_the_rest(width):
    assert (width.width_pvalue.sel(experiment="historical") < 0.05).all()
    assert float((width.width_pvalue.sel(experiment="hist-GHG") < 0.05).mean()) <= 0.25
    #(c): Averaging over longitude narrows the null: the zonal mean of 8 points varies less than one point
    assert (width.null_lower < 0).all() and (width.null_upper > 0).all()
    assert set(width.null_lower.dims) == {"model", "season", "lat"}


def test_member_block_test_of_the_zonal_means(tree):
    member_means = zonal.zonal_mean(tree.isel(year=slice(-YEARS_TESTED, None)).mean("year"))
    pvalue_tree = xr.map_over_datasets(partial(rc.member_block_pvalue, n_permutations=500),
                                       member_means, datatree.hist_nat_like(member_means))
    pvalue = datatree.tree_to_dataset(pvalue_tree, dims=["model", "experiment"]).tas
    assert (pvalue.sel(experiment="historical") < 0.05).all()
    assert float((pvalue.sel(experiment="hist-GHG") < 0.05).mean()) <= 0.25


def test_zonal_summary(tree, width):
    final = datatree.reduce_to_dataset(tree.isel(year=slice(-YEARS_TESTED, None)), lambda ds: ds.mean(["year", "member"]))
    mean = (final - final.sel(experiment="hist-nat")).drop_sel(experiment="hist-nat").tas
    pvalue = xr.full_like(mean, 0.01)
    qrange = xr.Dataset({"qrange_change": xr.full_like(mean, 1.0), "qrange_pvalue": pvalue})
    tails = xr.Dataset({name: xr.full_like(mean, value) for name, value in
                        (("upper_tail_change", 0.6), ("lower_tail_change", 0.4), ("width_change", 1.0),
                         ("tail_asymmetry", 0.2))})
    summary = rc.change_summary(mean, pvalue, xr.full_like(mean, 3.0), qrange, tails)

    out = zonal.zonal_summary(summary, mean_tested=xr.Dataset({"pvalue": pvalue.mean("lon")}), width_tested=width)
    assert list(rc.EXTREMES) == [name for name in out.data_vars][:4]
    np.testing.assert_allclose(out.low_extreme_change, -0.4)
    np.testing.assert_allclose(out.high_extreme_change - out.low_extreme_change, 1.0)
    assert {"mean_pvalue", "width_pvalue", "hist_nat_width", "null_lower", "null_upper"} <= set(out.data_vars)
