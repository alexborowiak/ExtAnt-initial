"""The numba Q-range tests give exactly the numbers of the original per-trial xarray code."""

import numpy as np
import pytest
import xarray as xr

import significance as sig

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


def _slow_hist_nat_samples(hist_nat, reference, n_sample, years, quantiles, n_trials, batch_size, seed):
    """The original sample_hist_nat_qrange_changes: gather each batch of trials and take xarray quantiles."""
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
        qrange = _slow_quantile_range(pool, quantiles, ("sample_member", "sample_year"))
        batches.append((qrange - reference).astype("float32"))
    return xr.concat(batches, dim="trial").assign_coords(trial=np.arange(n_trials))


@pytest.mark.parametrize("dask", [False, True])
def test_permutation_samples_equal_the_per_trial_code(dask):
    exp, ref = _ensemble(9, shift=1.0, scale=1.5), _ensemble(12)
    slow = sig.permutation_samples(exp, ref, statistic=_slow_quantile_range, n_trials=150, batch_size=40, seed=3)
    if dask:
        exp, ref = exp.chunk({"lat": 1}), ref.chunk({"lat": 1})
    fast = sig.qrange_permutation_samples(exp, ref, n_trials=150, seed=3).compute()
    np.testing.assert_array_equal(fast.transpose(*slow.dims).values, slow.values)
    # statistic=None takes the same fast path
    np.testing.assert_array_equal(sig.permutation_samples(exp, ref, n_trials=150, seed=3).values, fast.values)


@pytest.mark.parametrize("dask", [False, True])
def test_hist_nat_sampling_equals_the_per_trial_code(dask):
    hist_nat = _ensemble(14, n_years=40)
    reference = _slow_quantile_range(hist_nat)
    slow = _slow_hist_nat_samples(hist_nat, reference, 8, 11, (0.05, 0.95), 170, 50, seed=4)
    fast = sig.sample_hist_nat_qrange_changes(hist_nat.chunk({"lat": 1}) if dask else hist_nat, reference, 8,
                                              years=11, n_trials=170, seed=4)
    np.testing.assert_array_equal(fast.transpose(*slow.dims).values, slow.values)


def test_qrange_significance_detects_a_wider_experiment():
    hist_nat = _ensemble(30, n_years=60)
    experiments = {"wide": _ensemble(30, scale=2.0), "same": _ensemble(30)}
    result = sig.qrange_significance(experiments, hist_nat, n_trials=300, alpha=0.05)
    assert set(result.experiment.values) == set(experiments)
    pvalue = result.qrange_permutation_pvalue
    assert float(pvalue.min()) >= 1 / 301 and float(pvalue.max()) <= 1
    assert float(result.qrange_permutation_significant.sel(experiment="wide").mean()) > 0.9
    assert float(result.qrange_permutation_significant.sel(experiment="same").mean()) < 0.3
    assert bool(result.qrange_outside_hist_nat_range.sel(experiment="wide").mean() > 0.9)
