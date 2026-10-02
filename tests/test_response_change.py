"""Checks of response_change on small synthetic trees with a known mean shift and widening."""

import numpy as np
import pytest
import scipy.stats
import xarray as xr

import response_change as rc
import significance as sig

RNG = np.random.default_rng(3)
LAT = np.array([-80.0, -70.0, -50.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])
SEASONS = ["DJF", "JJA", "MAM", "SON"]
YEARS = np.arange(1990, 2015)
SHIFT, WIDEN = 2.0, 1.5  # historical: +2 degrees and 1.5x the spread over the whole record


def _member_data(n, shift=0.0, scale=1.0, years=YEARS):
    values = shift + scale * RNG.standard_normal((n, years.size, len(SEASONS), LAT.size, LON.size))
    return xr.DataArray(values, dims=("member", "year", "season", "lat", "lon"),
                        coords={"member": np.arange(n), "year": years, "season": SEASONS, "lat": LAT, "lon": LON})


@pytest.fixture(scope="module")
def tree():
    nodes = {}
    for model, n in (("A", 30), ("B", 20)):
        nodes[f"{model}/hist-nat"] = xr.Dataset({"tas": _member_data(n)}).assign_coords(height=2.0)
        nodes[f"{model}/historical"] = xr.Dataset({"tas": _member_data(n, SHIFT, WIDEN)}).assign_coords(height=2.0)
    return xr.DataTree.from_dict(nodes)


@pytest.fixture(scope="module")
def summary(tree):
    """change_summary with p-values and S/N built so the expected classes are known.

    Mean: significant everywhere, emerged only at lat -80. Width: significant only at lon 0.
    """
    mean = rc.mean_response(tree, years=11)
    pvalue = xr.full_like(mean, 0.001)
    sn = xr.where(mean.lat == -80.0, 3.0, 1.0) + 0 * mean
    width_p = xr.where(mean.lon == 0.0, 0.01, 0.5) + 0 * mean
    qrange = xr.Dataset({"qrange_change": xr.full_like(mean, 1.0), "qrange_pvalue": width_p})
    return rc.change_summary(mean, pvalue, sn.assign_coords(year=2009), qrange, rc.tail_changes(tree, years=11))


# ---------------------------------------------------------------------------
# The mean, the tails and the summary
# ---------------------------------------------------------------------------

def test_mean_response_equals_direct_calculation(tree):
    mean = rc.mean_response(tree, years=11)
    direct = (tree["A/historical"].tas.isel(year=slice(-11, None)).mean(("year", "member"))
              - tree["A/hist-nat"].tas.isel(year=slice(-11, None)).mean(("year", "member")))
    np.testing.assert_allclose(mean.sel(model="A", experiment="historical").transpose(*direct.dims), direct)
    assert list(mean.experiment.values) == ["historical"] and "height" not in mean.coords
    np.testing.assert_allclose(float(mean.mean()), SHIFT, atol=0.1)


def test_tail_changes_decompose_the_width(tree):
    tails = rc.tail_changes(tree, years=11)
    np.testing.assert_allclose(tails.width_change, tails.upper_tail_change + tails.lower_tail_change)
    np.testing.assert_allclose(tails.tail_asymmetry, tails.upper_tail_change - tails.lower_tail_change)
    # A symmetric 1.5x widening of N(0, 1): each tail stretches by 0.5 x 1.645.
    np.testing.assert_allclose(float(tails.upper_tail_change.mean()), 0.5 * 1.645, atol=0.1)
    np.testing.assert_allclose(float(tails.lower_tail_change.mean()), 0.5 * 1.645, atol=0.1)


def test_tail_width_change_is_the_bootstrap_tests_change(tree):
    """Both measure from hist-nat's full record, so the tails add up to the width change the bootstrap tests."""
    tails = rc.tail_changes(tree, years=11)
    width = sig.qrange_significance({"historical": tree["A/historical"].tas}, tree["A/hist-nat"].tas,
                                    years=11, n_trials=20).qrange_change.sel(experiment="historical")
    np.testing.assert_allclose(tails.width_change.sel(model="A", experiment="historical").transpose(*width.dims), width)


def test_change_summary_flags(summary):
    assert set(summary.dims) == {"model", "experiment", "season", "lat", "lon"}
    assert bool(summary.mean_significant.all())
    np.testing.assert_array_equal(summary.mean_robust.any(["model", "experiment", "season", "lon"]).values,
                                  [True, False, False])
    np.testing.assert_array_equal(summary.width_significant.any(["model", "experiment", "season", "lat"]).values,
                                  [True, False, False, False])
    assert {"upper_tail_change", "lower_tail_change", "tail_asymmetry"} <= set(summary.data_vars)


def test_change_class_and_counts(summary):
    classes = rc.change_class(summary).isel(model=0, experiment=0, season=0)
    assert int(classes.sel(lat=-80, lon=0)) == 3
    assert int(classes.sel(lat=-80, lon=90)) == 1
    assert int(classes.sel(lat=-70, lon=0)) == 2
    assert int(classes.sel(lat=-70, lon=90)) == 0
    counts = rc.joint_change_counts(summary)
    assert int(counts.sel(change="mean and wider", lat=-80, lon=0).isel(experiment=0, season=0)) == 2
    assert int(counts.sel(change="mean and narrower").sum()) == 0


def test_strongest_joint_change_and_regional_fractions(summary):
    assert rc.strongest_joint_change(summary, "A", "historical", "DJF") == {"lat": -80.0, "lon": 0.0}
    regional = rc.regional_mean(summary, lat_max=-60)
    fractions = sum(regional[f"fraction_{name.replace(' ', '_')}"] for name in rc.CHANGE_CLASSES.values())
    np.testing.assert_allclose(fractions, 1.0)


# ---------------------------------------------------------------------------
# The member-block test and internal variability
# ---------------------------------------------------------------------------

def test_member_block_test_matches_scipy_and_detects_the_shift(tree):
    result = rc.member_block_test(tree, n_permutations=2000, seed=1)
    np.testing.assert_allclose(result.mean_change.transpose(*rc.mean_response(tree).dims), rc.mean_response(tree))
    assert float(result.pvalue.max()) < 0.01  # a 2-degree shift is unmistakable

    small = xr.DataTree.from_dict({"A/hist-nat": xr.Dataset({"tas": _member_data(8)}),
                                   "A/hist-GHG": xr.Dataset({"tas": _member_data(8, 0.4)})})
    point = dict(season="DJF", lat=-80.0, lon=0.0)
    ours = float(rc.member_block_test(small, years=11, n_permutations=20000, seed=2).pvalue.sel(point).squeeze())
    x, y = (small[f"A/{e}"].tas.isel(year=slice(-11, None)).mean("year").sel(point).values for e in ("hist-GHG", "hist-nat"))
    reference = scipy.stats.permutation_test((x, y), lambda a, b: a.mean() - b.mean(), n_resamples=20000,
                                             random_state=3).pvalue
    assert abs(ours - reference) < 0.02


def test_remove_forced_response_leaves_internal_variability(tree):
    internal = rc.remove_forced_response(tree, window=21)
    anomalies = internal["A/historical"].tas
    # The forced shift is gone and the spread is the historical noise (1.5), not shrunk by removing it.
    np.testing.assert_allclose(float(anomalies.mean()), 0, atol=0.05)
    np.testing.assert_allclose(float(anomalies.std()), WIDEN, rtol=0.05)


# ---------------------------------------------------------------------------
# Additivity
# ---------------------------------------------------------------------------

def test_additivity_recovers_the_residual():
    years = np.arange(1850, 1920)
    responses = {"hist-nat": 0.1, "hist-GHG": 1.0, "hist-aer": -0.4, "hist-totalO3": 0.2}
    nodes = {}
    #(c): A step after the baseline, so the final years carry the full response
    step = xr.DataArray((years > 1905).astype(float), dims="year", coords={"year": years})
    for experiment, response in {**responses, "historical": sum(responses.values()) + 0.3}.items():
        nodes[f"A/{experiment}"] = xr.Dataset({"tas": (_member_data(20, years=years) * 0.1 + response * step)
                                               .transpose("member", "year", ...)})
    changes = rc.own_baseline_change(xr.DataTree.from_dict(nodes), years=11, baseline=slice(1850, 1900))
    result = rc.additivity(changes)
    assert list(result.term.values) == [*rc.ADDITIVE_PARTS, "sum of parts", "historical", "residual"]
    np.testing.assert_allclose(result.response.sel(term="residual").mean(), 0.3, atol=0.02)
    np.testing.assert_allclose(result.response.sel(term="sum of parts").mean(), sum(responses.values()), atol=0.02)
    # With a part missing, the sum is NaN rather than silently partial.
    partial = rc.additivity(changes.where(changes.experiment != "hist-aer"))
    assert bool(partial.response.sel(term="sum of parts").isnull().all())


# ---------------------------------------------------------------------------
# Local distributions
# ---------------------------------------------------------------------------

def test_quantile_shift_recovers_shift_and_widening(tree):
    samples = rc.final_years(tree, "A", ["hist-nat", "historical"], {"lat": -80.0, "lon": 0.0}, "DJF", years=11)
    assert samples["historical"].dims == ("member", "year") and samples["historical"].sizes["year"] == 11
    result = rc.quantile_shift(samples["historical"], samples["hist-nat"], n_boot=300)
    assert set(result.dims) == {"quantile"}
    # A shift of SHIFT plus a widening of WIDEN: the change in quantile q is SHIFT + (WIDEN - 1) * z_q.
    z = xr.DataArray(scipy.stats.norm.ppf(result["quantile"].values), dims="quantile",
                     coords={"quantile": result["quantile"]})
    assert float(np.abs(result["shift"] - (SHIFT + (WIDEN - 1) * z)).max()) < 0.6
    assert bool(((result["shift_lower"] <= result["shift"]) & (result["shift"] <= result["shift_upper"])).all())
    assert float(result["shift"].sel(quantile=0.95) - result["shift"].sel(quantile=0.05)) > 0.5
