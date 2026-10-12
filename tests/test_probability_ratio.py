"""Probability ratios on normal ensembles, where the answer is known."""

import numpy as np
import scipy.stats
import xarray as xr

from extant import probability_ratio as pr


def _ensemble(rng, n_members=200, n_years=50, mean=0.0, std=1.0):
    values = rng.normal(mean, std, size=(n_members, n_years))
    return xr.DataArray(values, dims=("member", "year"),
                        coords={"member": [f"r{i}" for i in range(n_members)], "year": np.arange(n_years)})


def test_threshold_levels():
    levels = pr.threshold_levels((20, 10))
    assert levels.sel(tail="warm", return_period=20) == 0.95
    assert levels.sel(tail="cold", return_period=20) == 0.05
    assert levels.sel(tail="warm", return_period=10) == 0.9
    assert levels.sel(tail="cold", return_period=10) == 0.1


def test_thresholds_of_a_normal_ensemble():
    rng = np.random.default_rng(0)
    counterfactual = _ensemble(rng)
    levels = pr.threshold_levels((20,))
    thresholds = pr.thresholds(counterfactual, levels)
    assert set(thresholds.dims) == {"tail", "return_period"}
    np.testing.assert_allclose(float(thresholds.sel(tail="warm", return_period=20)), 1.645, atol=0.06)
    np.testing.assert_allclose(float(thresholds.sel(tail="cold", return_period=20)), -1.645, atol=0.06)
    #(c): A Dataset gives the same numbers
    from_dataset = pr.thresholds(counterfactual.to_dataset(name="tas"), levels)["tas"]
    xr.testing.assert_allclose(from_dataset, thresholds)


def test_shift_and_widening():
    rng = np.random.default_rng(1)
    counterfactual = _ensemble(rng)
    levels = pr.threshold_levels((20,))
    thresholds = pr.thresholds(counterfactual, levels).squeeze("return_period")
    p_counterfactual = pr.member_exceedance(counterfactual, thresholds).mean("member")
    np.testing.assert_allclose(p_counterfactual.values, 0.05, atol=0.003)

    #(c): Shifted by one standard deviation: P(Z > 1.645 - 1) / 0.05 for warm seasons
    shifted = _ensemble(rng, mean=1.0)
    ratio = pr.member_exceedance(shifted, thresholds).mean("member") / p_counterfactual
    expected_warm = scipy.stats.norm.sf(1.645 - 1) / 0.05
    np.testing.assert_allclose(float(ratio.sel(tail="warm")), expected_warm, rtol=0.1)
    assert float(ratio.sel(tail="cold")) < 0.2

    #(c): Mean only: hist-nat moved by the shift, i.e. against the threshold moved down by it, gives the same
    mean_only = pr.member_exceedance(counterfactual, thresholds - 1.0).mean("member") / p_counterfactual
    np.testing.assert_allclose(float(mean_only.sel(tail="warm")), expected_warm, rtol=0.1)

    #(c): Wider, with the same mean: both tails more likely, which the mean-only ratio (1) misses
    wider = _ensemble(rng, std=1.5)
    ratio = pr.member_exceedance(wider, thresholds).mean("member") / p_counterfactual
    assert float(ratio.sel(tail="warm")) > 2 and float(ratio.sel(tail="cold")) > 2


def test_member_exceedance_keeps_missing_data_missing():
    values = xr.DataArray([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan]], dims=("member", "year"))
    threshold = xr.DataArray([2.0, 2.0], dims="tail", coords={"tail": ["warm", "cold"]})
    shares = pr.member_exceedance(values, threshold)
    np.testing.assert_allclose(shares.sel(tail="warm").values, [0.5, np.nan])
    np.testing.assert_allclose(shares.sel(tail="cold").values, [0.5, np.nan])


def test_bootstrap_member_mean():
    rng = np.random.default_rng(2)
    weights = pr.resample_weights(7, 100, rng)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    members = xr.DataArray(rng.normal(size=(30, 4)), dims=("member", "x"))
    trials = pr.bootstrap_member_mean(members, n_trials=4000, seed=0)
    assert trials.dims == ("trial", "x")
    #(c): The trials centre on the ensemble mean, spread by its standard error
    np.testing.assert_allclose(trials.mean("trial"), members.mean("member"), atol=0.02)
    np.testing.assert_allclose(trials.std("trial"), members.std("member") / np.sqrt(30), rtol=0.1)
    #(c): The same seed and number of members draw the same members
    again = pr.bootstrap_member_mean(members * 2, n_trials=4000, seed=0)
    xr.testing.assert_allclose(again, 2 * trials)
