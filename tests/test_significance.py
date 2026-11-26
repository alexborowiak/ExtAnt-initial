"""The hist-nat bootstrap gives exactly the numbers of the per-trial xarray code, and detects a widening and a shift; the
permutation test and the bootstrap of members against scipy and against theory."""

from functools import partial

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from extant.statistics import significance as sig
from extant.statistics.quantiles import quantile_range
from extant.utils import datatree

RNG = np.random.default_rng(11)


def _ensemble(n_members, n_years=11, shift=0.0, scale=1.0):
    coords = {"member": np.arange(n_members), "year": np.arange(2004 - n_years + 1, 2005), "season": ["DJF", "JJA"],
              "lat": [-80.0, -70.0], "lon": [0.0, 120.0, 240.0]}
    values = shift + scale * RNG.standard_normal(tuple(len(v) for v in coords.values()))
    values[RNG.random(values.shape) < 0.03] = np.nan    # scattered missing values
    values[:, :, 1, 1, 2] = np.nan                     # an all-NaN point
    return xr.DataArray(values, dims=tuple(coords), coords=coords)


def _slow_quantile_range(da, quantiles=(0.05, 0.95), dims=("member", "year")):
    """The original quantile_range: xarray's quantile (numpy's nanquantile, series by series)."""
    q = da.quantile(quantiles, dim=[d for d in dims if d in da.dims], skipna=True)
    return q.sel(quantile=quantiles[1], drop=True) - q.sel(quantile=quantiles[0], drop=True)


def _slow_hist_nat_samples(hist_nat, n_sample, quantiles, n_trials, seed):
    """The bootstrap the slow way: gather each trial's members and take xarray quantiles over all their years."""
    rng = np.random.default_rng(seed)
    members = np.argpartition(rng.random((n_trials, hist_nat.sizes["member"])), n_sample - 1, axis=1)[:, :n_sample]
    pool = hist_nat.isel(member=xr.DataArray(members, dims=("trial", "sample_member")))
    return _slow_quantile_range(pool, quantiles, ("sample_member", "year")).astype("float32").assign_coords(
        trial=np.arange(n_trials))


@pytest.mark.parametrize("dask", [False, True])
def test_bootstrap_qrange_equals_the_per_trial_code(dask):
    hist_nat = _ensemble(14, n_years=21)
    slow = _slow_hist_nat_samples(hist_nat, 8, (0.05, 0.95), 170, seed=4)
    fast = sig.bootstrap_qrange(hist_nat.chunk({"lat": 1}) if dask else hist_nat, n_members=8, quantiles=(0.05, 0.95), n_trials=170, seed=4)
    np.testing.assert_array_equal(fast.transpose(*slow.dims).values, slow.values)


def test_bootstrap_qrange_on_a_dataset_is_the_same():
    """Mapped over a DataTree, the function is given Datasets."""
    hist_nat = _ensemble(14, n_years=21)
    from_array = sig.bootstrap_qrange(hist_nat, n_members=8, quantiles=(0.05, 0.95), n_trials=50, seed=0)
    from_dataset = sig.bootstrap_qrange(hist_nat.to_dataset(name="tas"), n_members=8, quantiles=(0.05, 0.95), n_trials=50, seed=0)
    np.testing.assert_array_equal(from_dataset.tas.values, from_array.values)


def test_the_bootstrap_test_detects_a_wider_experiment():
    """The width test as notebook calc_02a runs it: the experiment's width against 10 hist-nat members over the same years."""
    hist_nat = _ensemble(30, n_years=21)
    trials = sig.bootstrap_qrange(hist_nat, n_members=10, quantiles=(0.05, 0.95), n_trials=300, seed=0)
    for experiment, expected in ((_ensemble(30, n_years=21, scale=2.0), "wide"), (_ensemble(30, n_years=21), "same")):
        pvalue = sig.pvalue_two_sided(trials, quantile_range(experiment))
        assert float(pvalue.min()) >= 1 / 301 and float(pvalue.max()) <= 1
        significant = float((pvalue < 0.05).mean())
        assert significant > 0.9 if expected == "wide" else significant < 0.3
        # The all-NaN point has no width and no p-value
        assert pvalue.isel(season=1, lat=1, lon=2).isnull().all()


def test_pvalue_is_nan_where_the_observed_value_is():
    samples = xr.DataArray(RNG.standard_normal((100, 3)), dims=("trial", "point"))
    observed = xr.DataArray([0.0, np.nan, 5.0], dims="point")
    pvalue = sig.pvalue_two_sided(samples, observed)
    assert np.isnan(float(pvalue[1])) and float(pvalue[2]) == pytest.approx(2 / 101)


def test_hist_nat_bootstrap_needs_enough_members():
    with pytest.raises(ValueError):
        sig.bootstrap_qrange(_ensemble(8, n_years=21), n_members=10, quantiles=(0.05, 0.95), n_trials=10, seed=0)


def test_bootstrap_draws_are_the_trials_of_bootstrap_qrange():
    """Each trial rebuilt from bootstrap_draws (same seed) gives bootstrap_qrange's number, as the schematic checks."""
    hist_nat = _ensemble(30, n_years=21).isel(season=0, lat=0, lon=0)
    selected = sig.bootstrap_draws(30, n_members=10, n_trials=300, seed=4)
    trials = sig.bootstrap_qrange(hist_nat, n_members=10, quantiles=(0.05, 0.95), n_trials=300, seed=4)
    values = hist_nat.values
    for trial in (0, 1, 150, 299):
        width = np.diff(np.nanquantile(values[selected[trial]], [0.05, 0.95]))[0]
        np.testing.assert_allclose(width, float(trials.sel(trial=trial)), atol=1e-5)
    assert int(selected.sum(1).min()) == int(selected.sum(1).max()) == 10


@pytest.mark.parametrize("dask", [False, True])
def test_bootstrap_mean_is_that_of_each_trial(dask):
    """Each trial rebuilt from bootstrap_draws (same seed) gives the same mean, NaN skipped."""
    hist_nat = _ensemble(30, n_years=21, shift=250.0)
    selected = sig.bootstrap_draws(30, n_members=10, n_trials=200, seed=4)
    data = hist_nat.chunk({"lat": 1}) if dask else hist_nat
    means = sig.bootstrap_mean(data, n_members=10, n_trials=200, seed=4)
    values = hist_nat.transpose("member", "year", ...).values
    for trial in (0, 1, 100, 199):
        pooled = values[selected[trial]].reshape(-1, *values.shape[2:])
        np.testing.assert_allclose(means.sel(trial=trial).transpose("season", "lat", "lon"),
                                   np.nanmean(pooled, axis=0), rtol=1e-12)


def test_bootstrap_mean_on_a_dataset_is_the_same():
    hist_nat = _ensemble(14, n_years=21)
    from_array = sig.bootstrap_mean(hist_nat, n_members=8, n_trials=50, seed=0)
    from_dataset = sig.bootstrap_mean(hist_nat.to_dataset(name="tas"), n_members=8, n_trials=50, seed=0)
    np.testing.assert_array_equal(from_dataset.tas.values, from_array.values)


def test_the_mean_test_finds_a_shift():
    """The mean test as notebook calc_02b runs it: the experiment's mean against 10 hist-nat members over the same years."""
    hist_nat = _ensemble(30, n_years=21)
    trials = sig.bootstrap_mean(hist_nat, n_members=10, n_trials=300, seed=0)
    for experiment, expected in ((_ensemble(30, n_years=21, shift=0.5), "shifted"), (_ensemble(30, n_years=21), "same")):
        pvalue = sig.pvalue_two_sided(trials, experiment.mean(["member", "year"]))
        significant = float((pvalue < 0.05).mean())
        assert significant > 0.9 if expected == "shifted" else significant < 0.3


def test_significant_counts_count_each_sign_and_skip_missing_models():
    coords = {"model": ["A", "B", "C"], "lat": [-80.0, -70.0]}
    change = xr.DataArray([[1.0, -2.0], [3.0, np.nan], [-1.0, -0.5]], dims=("model", "lat"), coords=coords)
    significant = xr.DataArray([[True, True], [True, True], [False, True]], dims=("model", "lat"), coords=coords)
    counts = sig.significant_counts(change, significant)
    assert list(counts["direction"].values) == ["increase", "decrease"]
    #(c): Model C's -1.0 is not significant; model B's NaN counts in neither direction
    np.testing.assert_array_equal(counts.sel(direction="increase"), [2, 0])
    np.testing.assert_array_equal(counts.sel(direction="decrease"), [0, 2])


def test_robust_sign_needs_the_threshold_of_models_with_data():
    coords = {"model": ["A", "B", "C"], "lat": [-80.0, -70.0, -60.0]}
    change = xr.DataArray([[1.0, -1.0, np.nan], [2.0, -2.0, np.nan], [-1.0, 1.0, np.nan]], dims=("model", "lat"),
                          coords=coords)
    significant = xr.DataArray([[True, True, False], [True, False, False], [False, True, False]],
                               dims=("model", "lat"), coords=coords)
    #(c): Two of three models significantly up passes 0.66; one of three down does not; no data gives NaN
    np.testing.assert_array_equal(sig.robust_sign(change, significant, threshold=0.66), [1.0, 0.0, np.nan])


def test_welch_counts_only_the_values_that_are_there():
    """With most values missing (as in one SAM phase), the t-test gives scipy's answer on the values that are left."""
    rng = np.random.default_rng(0)
    a = xr.DataArray(rng.normal(0.3, 1, (10, 21)), dims=("member", "year"))
    b = xr.DataArray(rng.normal(0, 1, (10, 21)), dims=("member", "year"))
    kept = xr.DataArray(rng.random((10, 21)) < 1 / 3, dims=("member", "year"))
    result = sig.ttest_welch(a.where(kept), b.where(kept))
    expected = scipy.stats.ttest_ind(a.values[kept.values], b.values[kept.values], equal_var=False)
    np.testing.assert_allclose([float(result["t"]), float(result["p"])], [expected.statistic, expected.pvalue])


# ---------------------------------------------------------------------------
# The permutation test of a difference in means
# ---------------------------------------------------------------------------

def _member_means(n, shift=0.0):
    """Each member's mean over 11 years of N(shift, 1), at 3 points."""
    return xr.DataArray(shift + RNG.standard_normal((n, 11, 3)).mean(axis=1), dims=("member", "lat"),
                        coords={"member": np.arange(n), "lat": [-80.0, -70.0, -50.0]})


def test_member_block_pvalue_matches_scipy_and_detects_the_shift():
    tree = xr.DataTree.from_dict({"A/hist-nat": xr.Dataset({"tas": _member_means(30)}),
                                  "A/historical": xr.Dataset({"tas": _member_means(30, 2.0)})})
    pvalue = xr.map_over_datasets(partial(sig.member_block_pvalue, n_permutations=2000, seed=1), tree, datatree.hist_nat_like(tree))
    assert float(pvalue["A/historical"].tas.max()) < 0.01  # a 2-degree shift is unmistakable

    x, y = _member_means(8, 0.4).isel(lat=0), _member_means(8).isel(lat=0)
    ours = float(sig.member_block_pvalue(x, y, n_permutations=20000, seed=2))
    reference = scipy.stats.permutation_test((x.values, y.values), lambda a, b: a.mean() - b.mean(),
                                             n_resamples=20000, random_state=3).pvalue
    assert abs(ours - reference) < 0.02


def test_the_step_by_step_permutations_are_the_member_block_tests():
    """Notebook calc_02b's demonstration (weights from default_rng(0), times the member means) is member_block_pvalue's test."""
    experiment, hist_nat = _member_means(30, 2.0).isel(lat=0), _member_means(30).isel(lat=0)
    weights = sig.permutation_weights(500, experiment.sizes["member"], hist_nat.sizes["member"], np.random.default_rng(0))
    permutations = xr.DataArray(weights @ np.concatenate([experiment.values, hist_nat.values]), dims="trial")
    change = experiment.mean("member") - hist_nat.mean("member")
    by_hand = float(sig.pvalue_two_sided(permutations, change))
    assert by_hand == pytest.approx(float(sig.member_block_pvalue(experiment, hist_nat, n_permutations=500)))
    # A 2-degree shift with 30 members each is far outside every relabelling
    assert by_hand == pytest.approx(2 / 501)

    # With a smaller shift the p-value is not at its floor, and still the same
    shifted = experiment - 1.95
    permutations = xr.DataArray(weights @ np.concatenate([shifted.values, hist_nat.values]), dims="trial")
    by_hand = float(sig.pvalue_two_sided(permutations, shifted.mean("member") - hist_nat.mean("member")))
    assert by_hand > 2 / 501
    assert by_hand == pytest.approx(float(sig.member_block_pvalue(shifted, hist_nat, n_permutations=500)))


# ---------------------------------------------------------------------------
# Members drawn with replacement
# ---------------------------------------------------------------------------

def test_bootstrap_member_mean():
    rng = np.random.default_rng(2)
    weights = sig.resample_weights(7, 100, rng)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    members = xr.DataArray(rng.normal(size=(30, 4)), dims=("member", "x"))
    trials = sig.bootstrap_member_mean(members, n_trials=4000, seed=0)
    assert trials.dims == ("trial", "x")
    #(c): The trials centre on the ensemble mean, spread by its standard error
    np.testing.assert_allclose(trials.mean("trial"), members.mean("member"), atol=0.02)
    np.testing.assert_allclose(trials.std("trial"), members.std("member") / np.sqrt(30), rtol=0.1)
    #(c): The same seed and number of members draw the same members
    again = sig.bootstrap_member_mean(members * 2, n_trials=4000, seed=0)
    xr.testing.assert_allclose(again, 2 * trials)


def test_quantile_shift_recovers_shift_and_widening():
    shift, widen = 2.0, 1.5
    hist_nat = xr.DataArray(RNG.standard_normal((30, 11)), dims=("member", "year"))
    historical = xr.DataArray(shift + widen * RNG.standard_normal((30, 11)), dims=("member", "year"))
    result = sig.quantile_shift(historical, hist_nat, n_boot=300)
    assert set(result.dims) == {"quantile"}
    # A shift plus a widening: the change in quantile q is shift + (widen - 1) * z_q.
    z = xr.DataArray(scipy.stats.norm.ppf(result["quantile"].values), dims="quantile",
                     coords={"quantile": result["quantile"]})
    assert float(np.abs(result["shift"] - (shift + (widen - 1) * z)).max()) < 0.6
    assert bool(((result["shift_lower"] <= result["shift"]) & (result["shift"] <= result["shift_upper"])).all())
    assert float(result["shift"].sel(quantile=0.95) - result["shift"].sel(quantile=0.05)) > 0.5
