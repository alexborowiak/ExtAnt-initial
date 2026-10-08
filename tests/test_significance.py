"""The hist-nat bootstrap gives exactly the numbers of the original per-trial xarray code, and detects a widening."""

import numpy as np
import pytest
import xarray as xr

from extant import significance as sig
from extant.quantiles import quantile_range

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


def _slow_hist_nat_samples(hist_nat, n_sample, years, quantiles, n_trials, batch_size, seed):
    """The original bootstrap: gather each batch of trials and take xarray quantiles."""
    rng = np.random.default_rng(seed)
    n_members, n_years = hist_nat.sizes["member"], hist_nat.sizes["year"]
    half = years // 2
    centres = np.arange(half, n_years - half)
    sampled = np.tile(centres, n_trials // centres.size)
    if n_trials % centres.size:
        sampled = np.concatenate([sampled, rng.choice(centres, n_trials % centres.size, replace=False)])
    rng.shuffle(sampled)
    offsets = np.arange(-half, half + 1)
    batches = []
    for start in range(0, n_trials, batch_size):
        n = min(batch_size, n_trials - start)
        members = np.argpartition(rng.random((n, n_members)), n_sample - 1, axis=1)[:, :n_sample]
        year_index = sampled[start:start + n][:, None] + offsets[None, :]
        pool = hist_nat.isel(member=xr.DataArray(members, dims=("trial", "sample_member")),
                             year=xr.DataArray(year_index, dims=("trial", "sample_year")))
        batches.append(_slow_quantile_range(pool, quantiles, ("sample_member", "sample_year")).astype("float32"))
    return xr.concat(batches, dim="trial").assign_coords(trial=np.arange(n_trials))


@pytest.mark.parametrize("dask", [False, True])
def test_bootstrap_qrange_equals_the_per_trial_code(dask):
    hist_nat = _ensemble(14, n_years=40)
    slow = _slow_hist_nat_samples(hist_nat, 8, 11, (0.05, 0.95), 170, 50, seed=4)
    fast = sig.bootstrap_qrange(hist_nat.chunk({"lat": 1}) if dask else hist_nat, 8, years=11, n_trials=170, seed=4)
    np.testing.assert_array_equal(fast.transpose(*slow.dims).values, slow.values)


def test_bootstrap_qrange_on_a_dataset_is_the_same():
    """Mapped over a DataTree, the function is given Datasets."""
    hist_nat = _ensemble(14, n_years=40)
    from_array = sig.bootstrap_qrange(hist_nat, 8, years=11, n_trials=50)
    from_dataset = sig.bootstrap_qrange(hist_nat.to_dataset(name="tas"), 8, years=11, n_trials=50)
    np.testing.assert_array_equal(from_dataset.tas.values, from_array.values)


def test_the_bootstrap_test_detects_a_wider_experiment():
    """The width test as notebook 03 runs it: the change, the null from hist-nat alone, and the p-value."""
    hist_nat = _ensemble(30, n_years=60)
    hist_nat_qrange = quantile_range(hist_nat)
    null = sig.bootstrap_qrange(hist_nat, 10, years=21, n_trials=300) - hist_nat_qrange
    for experiment, expected in ((_ensemble(30, n_years=21, scale=2.0), "wide"), (_ensemble(30, n_years=21), "same")):
        change = quantile_range(experiment) - hist_nat_qrange
        pvalue = sig.pvalue_two_sided(null, change)
        assert float(pvalue.min()) >= 1 / 301 and float(pvalue.max()) <= 1
        significant = float((pvalue < 0.05).mean())
        assert significant > 0.9 if expected == "wide" else significant < 0.3
        # The all-NaN point has no change and no p-value
        assert pvalue.isel(season=1, lat=1, lon=2).isnull().all()


def test_pvalue_is_nan_where_the_observed_value_is():
    samples = xr.DataArray(RNG.standard_normal((100, 3)), dims=("trial", "point"))
    observed = xr.DataArray([0.0, np.nan, 5.0], dims="point")
    pvalue = sig.pvalue_two_sided(samples, observed)
    assert np.isnan(float(pvalue[1])) and float(pvalue[2]) == pytest.approx(2 / 101)


def test_hist_nat_bootstrap_needs_enough_members():
    with pytest.raises(ValueError):
        sig.bootstrap_qrange(_ensemble(8, n_years=40), 10, years=21)


def test_bootstrap_draws_are_the_trials_of_bootstrap_qrange():
    """Each trial rebuilt from bootstrap_draws (same seed) gives bootstrap_qrange's number, as the schematic checks."""
    hist_nat = _ensemble(30, n_years=60).isel(season=0, lat=0, lon=0)
    selected, window_starts = sig.bootstrap_draws(30, 60, years=21, n_trials=300, seed=4)
    trials = sig.bootstrap_qrange(hist_nat, years=21, n_trials=300, seed=4)
    values = hist_nat.values
    for trial in (0, 1, 150, 299):
        start = window_starts[trial]
        pooled = values[selected[trial], start:start + 21]
        width = np.diff(np.nanquantile(pooled, [0.05, 0.95]))[0]
        np.testing.assert_allclose(width, float(trials.sel(trial=trial)), atol=1e-5)
    assert int(selected.sum(1).min()) == sig.N_BOOTSTRAP_MEMBERS


def test_significant_counts_count_each_sign_and_skip_missing_models():
    coords = {"model": ["A", "B", "C"], "lat": [-80.0, -70.0]}
    change = xr.DataArray([[1.0, -2.0], [3.0, np.nan], [-1.0, -0.5]], dims=("model", "lat"), coords=coords)
    significant = xr.DataArray([[True, True], [True, True], [False, True]], dims=("model", "lat"), coords=coords)
    counts = sig.significant_counts(change, significant)
    assert list(counts["direction"].values) == ["increase", "decrease"]
    #(c): Model C's -1.0 is not significant; model B's NaN counts in neither direction
    np.testing.assert_array_equal(counts.sel(direction="increase"), [2, 0])
    np.testing.assert_array_equal(counts.sel(direction="decrease"), [0, 2])


def test_area_mean_of_a_dataset_matches_its_dataarrays():
    da = _ensemble(3).isel(member=0, year=0)
    ds = xr.Dataset({"a": da, "b": 2 * da})
    xr.testing.assert_allclose(sig.area_mean(ds)["b"], sig.area_mean(2 * da))


def test_robust_sign_needs_the_threshold_of_models_with_data():
    coords = {"model": ["A", "B", "C"], "lat": [-80.0, -70.0, -60.0]}
    change = xr.DataArray([[1.0, -1.0, np.nan], [2.0, -2.0, np.nan], [-1.0, 1.0, np.nan]], dims=("model", "lat"),
                          coords=coords)
    significant = xr.DataArray([[True, True, False], [True, False, False], [False, True, False]],
                               dims=("model", "lat"), coords=coords)
    #(c): Two of three models significantly up passes 0.66; one of three down does not; no data gives NaN
    np.testing.assert_array_equal(sig.robust_sign(change, significant), [1.0, 0.0, np.nan])
