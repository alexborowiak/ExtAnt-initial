"""Checks of response_change on small synthetic trees with a known mean shift and widening."""

from functools import partial

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from extant import datatree
from extant import quantiles as qc
from extant import response_change as rc
from extant import significance as sig

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


def _mean_response(tree, years=11):
    """Notebook 03's mean response: final-years ensemble means, each minus hist-nat's."""
    final = datatree.reduce_to_dataset(tree.isel(year=slice(-years, None)), lambda ds: ds.mean(["year", "member"]))
    return (final - final.sel(experiment="hist-nat")).drop_sel(experiment="hist-nat").tas


def _tails(tree, years=11):
    """Notebook 03's tails: the change in Q05, Q50 and Q95, final years against hist-nat's whole record."""
    quantiles = partial(qc.pooled_quantiles, quantiles=[0.05, 0.5, 0.95])
    final = datatree.reduce_to_dataset(tree.isel(year=slice(-years, None)), quantiles).tas.drop_sel(experiment="hist-nat")
    hist_nat = datatree.reduce_to_dataset(tree.match("*/hist-nat"), quantiles).tas.sel(experiment="hist-nat", drop=True)
    q05, q50, q95 = ((final - hist_nat).sel(quantile=q, drop=True) for q in (0.05, 0.5, 0.95))
    upper, lower = q95 - q50, q50 - q05
    return xr.Dataset({"upper_tail_change": upper, "lower_tail_change": lower, "width_change": upper + lower,
                       "tail_asymmetry": upper - lower, "q05_change": q05, "q50_change": q50, "q95_change": q95})


@pytest.fixture(scope="module")
def summary(tree):
    """change_summary with p-values and S/N built so the expected classes are known.

    Mean: significant everywhere, emerged only at lat -80. Width: significant only at lon 0.
    """
    mean = _mean_response(tree)
    pvalue = xr.full_like(mean, 0.001)
    sn = xr.where(mean.lat == -80.0, 3.0, 1.0) + 0 * mean
    width_p = xr.where(mean.lon == 0.0, 0.01, 0.5) + 0 * mean
    qrange = xr.Dataset({"qrange_change": xr.full_like(mean, 1.0), "qrange_pvalue": width_p})
    return rc.change_summary(mean, pvalue, sn.assign_coords(year=2009), qrange, _tails(tree))


# ---------------------------------------------------------------------------
# The mean, the tails and the summary
# ---------------------------------------------------------------------------

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
    #(c): Every season searched: the season comes with the point
    point = rc.strongest_joint_change(summary, "A", "historical")
    assert list(point) == ["season", "lat", "lon"]
    assert rc.strongest_joint_change(summary, "A", "historical", point["season"]) == {"lat": point["lat"],
                                                                                      "lon": point["lon"]}
    regional = rc.regional_mean(summary, lat_max=-60)
    fractions = sum(regional[f"fraction_{name.replace(' ', '_')}"] for name in rc.CHANGE_CLASSES.values())
    np.testing.assert_allclose(fractions, 1.0)


# ---------------------------------------------------------------------------
# The member-block test and internal variability
# ---------------------------------------------------------------------------

def test_member_block_pvalue_matches_scipy_and_detects_the_shift(tree):
    member_means = tree.isel(year=slice(-11, None)).mean("year")
    pvalue = xr.map_over_datasets(partial(rc.member_block_pvalue, n_permutations=2000, seed=1),
                                  member_means, datatree.hist_nat_like(member_means))
    assert float(pvalue["A/historical"].tas.max()) < 0.01  # a 2-degree shift is unmistakable

    point = dict(season="DJF", lat=-80.0, lon=0.0)
    x, y = (_member_data(8, shift).isel(year=slice(-11, None)).mean("year").sel(point) for shift in (0.4, 0.0))
    ours = float(rc.member_block_pvalue(x, y, n_permutations=20000, seed=2))
    reference = scipy.stats.permutation_test((x.values, y.values), lambda a, b: a.mean() - b.mean(),
                                             n_resamples=20000, random_state=3).pvalue
    assert abs(ours - reference) < 0.02


def test_members_minus_the_forced_response_leave_internal_variability(tree):
    internal = tree - tree.map_over_datasets(partial(rc.forced_response, window=21))
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
    tree = xr.DataTree.from_dict(nodes)
    final = datatree.reduce_to_dataset(tree.isel(year=slice(-11, None)), lambda ds: ds.mean(["year", "member"]))
    baseline = datatree.reduce_to_dataset(tree.sel(year=slice(1850, 1900)), lambda ds: ds.mean(["year", "member"]))
    changes = (final - baseline).tas
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
    with_season = rc.final_years(tree, "A", ["historical"], {"season": "DJF", "lat": -80.0, "lon": 0.0}, years=11)
    xr.testing.assert_identical(with_season["historical"], samples["historical"])
    assert samples["historical"].dims == ("member", "year") and samples["historical"].sizes["year"] == 11
    result = rc.quantile_shift(samples["historical"], samples["hist-nat"], n_boot=300)
    assert set(result.dims) == {"quantile"}
    # A shift of SHIFT plus a widening of WIDEN: the change in quantile q is SHIFT + (WIDEN - 1) * z_q.
    z = xr.DataArray(scipy.stats.norm.ppf(result["quantile"].values), dims="quantile",
                     coords={"quantile": result["quantile"]})
    assert float(np.abs(result["shift"] - (SHIFT + (WIDEN - 1) * z)).max()) < 0.6
    assert bool(((result["shift_lower"] <= result["shift"]) & (result["shift"] <= result["shift_upper"])).all())
    assert float(result["shift"].sel(quantile=0.95) - result["shift"].sel(quantile=0.05)) > 0.5


def test_the_step_by_step_permutations_are_the_member_block_tests(tree):
    """Notebook 03's demonstration (weights from default_rng(0), times the member means) is member_block_pvalue's test."""
    point = dict(season="DJF", lat=-80.0, lon=0.0)
    experiment, hist_nat = (tree[f"A/{e}"].tas.sel(point).isel(year=slice(-11, None)).mean("year")
                            for e in ("historical", "hist-nat"))
    weights = rc.permutation_weights(500, experiment.sizes["member"], hist_nat.sizes["member"], np.random.default_rng(0))
    permutations = xr.DataArray(weights @ np.concatenate([experiment.values, hist_nat.values]), dims="trial")
    change = experiment.mean("member") - hist_nat.mean("member")
    by_hand = float(sig.pvalue_two_sided(permutations, change))
    assert by_hand == pytest.approx(float(rc.member_block_pvalue(experiment, hist_nat, n_permutations=500)))
    # A 2-degree shift with 30 members each is far outside every relabelling
    assert by_hand == pytest.approx(2 / 501)

    # With a smaller shift the p-value is not at its floor, and still the same
    shifted = experiment - 1.9
    permutations = xr.DataArray(weights @ np.concatenate([shifted.values, hist_nat.values]), dims="trial")
    by_hand = float(sig.pvalue_two_sided(permutations, shifted.mean("member") - hist_nat.mean("member")))
    assert by_hand > 2 / 501
    assert by_hand == pytest.approx(float(rc.member_block_pvalue(shifted, hist_nat, n_permutations=500)))


def test_width_parts_add_up_to_the_width():
    quantiles = xr.DataArray([[-3.0, 0.5, 2.0], [-1.0, 0.0, 4.0]], dims=("lat", "quantile"),
                             coords={"lat": [-80.0, -70.0], "quantile": [0.05, 0.5, 0.95]})
    parts = rc.width_parts(quantiles)
    assert list(parts["part"].values) == list(rc.WIDTH_PARTS)
    np.testing.assert_allclose(parts.sel(part="lower tail"), [3.5, 1.0])
    np.testing.assert_allclose(parts.sel(part="upper tail"), [1.5, 4.0])
    xr.testing.assert_allclose(parts.sel(part="lower tail", drop=True) + parts.sel(part="upper tail", drop=True),
                               parts.sel(part="width", drop=True))


def test_joint_class_orders_the_nine_classes_as_the_key():
    from extant import significance as sig

    change = xr.DataArray([-1.0, 2.0, 3.0, np.nan], dims="lat")
    significant = xr.DataArray([True, True, False, True], dims="lat")
    signs = sig.significant_sign(change, significant)
    np.testing.assert_array_equal(signs, [-1, 1, 0, np.nan])
    #(c): Lower mean and narrower is 0, no change in either 4, higher and wider 8
    np.testing.assert_array_equal(rc.joint_class(signs, signs), [0, 8, 4, np.nan])
