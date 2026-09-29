"""Resampling, permutation and t-test significance for experiment vs reference ensembles.

All resampling works on whole members, so each member's time series stays intact.
"""

from functools import partial

import numpy as np
import scipy.special
import xarray as xr

from quantile_calc import quantile_range
from xarray_datatree_utils import skip_empty


# ---------------------------------------------------------------------------
# p-values and bounds from a sample distribution
# ---------------------------------------------------------------------------

def pvalue_two_sided(samples, observed, dim="trial", method="tails"):
    """Two-sided resampling p-value, with the +1 correction.

    Args:
        samples (xr.DataArray): Null distribution along ``dim``.
        observed (xr.DataArray): Observed statistic.
        dim (str): Sample dimension.
        method (str): "tails" doubles the smaller one-sided tail (clipped to 1);
            "abs" counts samples with |sample| >= |observed|, assuming a null centred on 0.

    Returns:
        xr.DataArray: p-values with ``dim`` reduced.
    """
    n = samples.sizes[dim]

    if method == "abs":
        return ((np.abs(samples) >= np.abs(observed)).sum(dim) + 1) / (n + 1)

    upper = ((samples >= observed).sum(dim) + 1) / (n + 1)
    lower = ((samples <= observed).sum(dim) + 1) / (n + 1)
    return (2 * np.minimum(upper, lower)).clip(max=1)


def outside_bounds(samples, observed, alpha=0.05, dim="trial"):
    """Return the central (1 - alpha) bounds of ``samples`` and whether ``observed`` lies outside."""
    bounds = samples.quantile([alpha / 2, 1 - alpha / 2], dim=dim)
    lower = bounds.sel(quantile=alpha / 2, drop=True)
    upper = bounds.sel(quantile=1 - alpha / 2, drop=True)
    return lower, upper, (observed < lower) | (observed > upper)


# ---------------------------------------------------------------------------
# Member resampling
# ---------------------------------------------------------------------------

def _batches(n_trials, batch_size):
    for start in range(0, n_trials, batch_size):
        yield start, min(batch_size, n_trials - start)


def resample_members(da, n_trials, rng, n_samples=None, replace=True, member_dim="member"):
    """Draw members for each trial, returning ``da`` with a new ``trial`` dim.

    Args:
        n_samples (int | None): Members per trial; defaults to all members.
        replace (bool): True for a bootstrap; False for distinct members per trial.
    """
    n_members = da.sizes[member_dim]
    n_samples = n_members if n_samples is None else n_samples

    if replace:
        idx = rng.integers(0, n_members, size=(n_trials, n_samples))
    else:
        idx = np.argsort(rng.random((n_trials, n_members)), axis=1)[:, :n_samples]

    return da.isel({member_dim: xr.DataArray(idx, dims=("trial", member_dim))})


def pool_members(exp, reference):
    """Concatenate two ensembles along a ``pool_member`` dim, experiment members first."""
    pooled = xr.concat(
        [exp.rename(member="pool_member"), reference.rename(member="pool_member")],
        dim="pool_member",
    )
    return pooled.assign_coords(pool_member=np.arange(pooled.sizes["pool_member"]))


def permutation_batches(exp, reference, statistic, n_trials, batch_size, rng):
    """Yield computed ``statistic(exp*) - statistic(reference*)`` for random member partitions.

    Members are pooled and randomly split into groups of the original ensemble
    sizes. ``statistic`` receives a DataArray with a ``member`` dim.
    """
    n_exp = exp.sizes["member"]
    pooled = pool_members(exp, reference)
    n_total = pooled.sizes["pool_member"]

    for _, n in _batches(n_trials, batch_size):
        order = np.argsort(rng.random((n, n_total)), axis=1)
        idx_exp = xr.DataArray(order[:, :n_exp], dims=("trial", "member"))
        idx_ref = xr.DataArray(order[:, n_exp:], dims=("trial", "member"))

        yield (
            statistic(pooled.isel(pool_member=idx_exp))
            - statistic(pooled.isel(pool_member=idx_ref))
        ).compute()


def permutation_samples(exp, reference, statistic=None, n_trials=10_000, batch_size=100, seed=0):
    """Permutation null distribution of the experiment-minus-reference statistic.

    Args:
        statistic (callable | None): Maps a DataArray with a ``member`` dim to the
            test statistic. Defaults to the pooled Q95-Q05 range.

    Returns:
        xr.DataArray: float32 samples along a ``trial`` dim.
    """
    statistic = statistic or quantile_range
    rng = np.random.default_rng(seed)
    batches = [
        b.astype("float32")
        for b in permutation_batches(exp, reference, statistic, n_trials, batch_size, rng)
    ]
    return xr.concat(batches, dim="trial").assign_coords(trial=np.arange(n_trials))


def sample_hist_nat_qrange_changes(
    hist_nat_da,
    hist_nat_reference_qrange_da,
    n_members_to_sample,
    years=11,
    quantiles=(0.05, 0.95),
    n_trials=10_000,
    batch_size=100,
    seed=0,
):
    """Sample N distinct hist-nat members over random ``years``-year windows.

    Returns each trial's Q-range minus ``hist_nat_reference_qrange_da``: the
    changes obtainable from natural variability and sample size alone.
    Windows are used approximately equally across trials.
    """
    rng = np.random.default_rng(seed)

    n_hist_nat_members = hist_nat_da.sizes["member"]
    n_hist_nat_years = hist_nat_da.sizes["year"]

    if n_members_to_sample > n_hist_nat_members:
        raise ValueError("n_members_to_sample cannot exceed available hist-nat members.")

    if years % 2 == 0:
        raise ValueError("years must be odd.")

    half_window = years // 2

    # All centre years that have a complete window.
    valid_centre_indices = np.arange(half_window, n_hist_nat_years - half_window)
    n_valid_windows = len(valid_centre_indices)

    n_full_repeats = n_trials // n_valid_windows
    n_extra_trials = n_trials % n_valid_windows

    sampled_centre_indices = np.tile(valid_centre_indices, n_full_repeats)

    if n_extra_trials:
        sampled_centre_indices = np.concatenate([
            sampled_centre_indices,
            rng.choice(valid_centre_indices, n_extra_trials, replace=False),
        ])

    rng.shuffle(sampled_centre_indices)

    year_offsets = np.arange(-half_window, half_window + 1)

    qrange_change_batches = []

    for start, n in _batches(n_trials, batch_size):
        # N distinct members per trial.
        sampled_member_indices = np.argpartition(
            rng.random((n, n_hist_nat_members)),
            n_members_to_sample - 1,
            axis=1,
        )[:, :n_members_to_sample]

        # The window of years for each trial.
        sampled_year_indices = (
            sampled_centre_indices[start:start + n][:, None]
            + year_offsets[None, :]
        )

        # trial x sample_member x sample_year x season x lat x lon
        sampled_hist_nat_da = hist_nat_da.isel(
            member=xr.DataArray(sampled_member_indices, dims=("trial", "sample_member")),
            year=xr.DataArray(sampled_year_indices, dims=("trial", "sample_year")),
        )

        sampled_qrange_da = quantile_range(
            sampled_hist_nat_da,
            quantiles=quantiles,
            dims=("sample_member", "sample_year"),
        )

        qrange_change_batches.append(
            (sampled_qrange_da - hist_nat_reference_qrange_da)
            .compute()
            .astype("float32")
        )

    return xr.concat(qrange_change_batches, dim="trial").assign_coords(trial=np.arange(n_trials))


def qrange_significance(
    experiments,
    hist_nat,
    n_members_to_sample=None,
    years=11,
    quantiles=(0.05, 0.95),
    n_trials=10_000,
    batch_size=100,
    alpha=0.05,
    seed=0,
):
    """Both Q-range significance tests for every experiment of one model.

    1. Sample size: is the experiment's final-``years`` Q-range change (vs the
       full hist-nat record) outside what ``n_members_to_sample`` hist-nat members
       over a ``years`` window can produce? The hist-nat sampling depends only on
       hist-nat, so it runs once and is shared by every experiment.
    2. Permutation: does the experiment differ from hist-nat over the same final
       ``years``? Whole members are permuted between the two.

    Args:
        experiments (dict[str, xr.DataArray]): Experiment name -> data with
            ``member`` and ``year`` dims.
        hist_nat (xr.DataArray): The model's hist-nat, full record.
        n_members_to_sample (int | None): Defaults to the smallest ensemble
            among ``experiments`` and ``hist_nat``.

    Returns:
        xr.Dataset: One variable per statistic, with an ``experiment`` dim.
    """
    if n_members_to_sample is None:
        n_members_to_sample = min(da.sizes["member"] for da in [hist_nat, *experiments.values()])

    qrange = partial(quantile_range, quantiles=quantiles)
    hist_nat_period = hist_nat.isel(year=slice(-years, None))
    hist_nat_qrange = qrange(hist_nat).compute()
    hist_nat_period_qrange = qrange(hist_nat_period).compute()

    hist_nat_samples = sample_hist_nat_qrange_changes(
        hist_nat, hist_nat_qrange, n_members_to_sample,
        years=years, quantiles=quantiles, n_trials=n_trials, batch_size=batch_size, seed=seed,
    )

    results = []
    for exp_da in experiments.values():
        exp_da = exp_da.isel(year=slice(-years, None))
        exp_qrange = qrange(exp_da).compute()

        # 1. Experiment vs the long-term hist-nat reference.
        change = exp_qrange - hist_nat_qrange
        lower, upper, outside = outside_bounds(hist_nat_samples, change, alpha=alpha)

        # 2. Experiment vs hist-nat over the same years.
        period_difference = exp_qrange - hist_nat_period_qrange
        permutations = permutation_samples(
            exp_da, hist_nat_period, statistic=qrange,
            n_trials=n_trials, batch_size=batch_size, seed=seed,
        )
        pvalue = pvalue_two_sided(permutations, period_difference, method="tails")

        results.append(xr.Dataset({
            "hist_nat_qrange": hist_nat_qrange,
            "hist_nat_period_qrange": hist_nat_period_qrange,
            "hist_exp_qrange": exp_qrange,
            # Experiment minus long-term hist-nat; sampling uncertainty of N hist-nat members.
            "qrange_change": change,
            "hist_nat_qrange_change_lower_95": lower,
            "hist_nat_qrange_change_upper_95": upper,
            "qrange_outside_hist_nat_range": outside,
            # Experiment versus hist-nat over the same years.
            "qrange_period_difference": period_difference,
            "qrange_permutation_pvalue": pvalue,
            "qrange_permutation_significant": pvalue < alpha,
        }))

    return xr.concat(results, dim="experiment").assign_coords(experiment=list(experiments))


# ---------------------------------------------------------------------------
# Multi-model aggregation
# ---------------------------------------------------------------------------

def model_agreement(change, significant=None, dim="model", sign_threshold=0.8, significant_threshold=0.66):
    """Collapse a per-model field into a multi-model summary (IPCC AR6 "advanced approach").

    Args:
        change (xr.DataArray): Per-model change, with ``dim``.
        significant (xr.DataArray | None): Per-model boolean significance.
        sign_threshold (float): Fraction of models that must share the sign of the median.
        significant_threshold (float): Fraction of models that must be significant.

    Returns:
        xr.Dataset: ``median``, ``n_models``, ``sign_agreement`` and, with
        ``significant``, ``significant_fraction`` plus the three mutually exclusive
        categories ``robust`` (significant and agreeing), ``conflicting``
        (significant but disagreeing) and ``no_signal`` (too few significant).
    """
    valid = change.notnull()
    n_models = valid.sum(dim)
    median = change.median(dim)
    sign_agreement = (np.sign(change) == np.sign(median)).where(valid).sum(dim) / n_models

    out = {"median": median, "n_models": n_models, "sign_agreement": sign_agreement}

    if significant is not None:
        significant_fraction = (significant.astype(bool) & valid).sum(dim) / n_models
        is_significant = significant_fraction >= significant_threshold
        agrees = sign_agreement >= sign_threshold
        out.update(
            significant_fraction=significant_fraction,
            robust=is_significant & agrees,
            conflicting=is_significant & ~agrees,
            no_signal=~is_significant & (n_models > 0),
        )

    return xr.Dataset(out)


def area_mean(da, lat="lat", lon="lon"):
    """Cos(latitude)-weighted mean over ``lat`` and ``lon``; booleans give the area fraction."""
    if da.dtype == bool:
        da = da.astype(float)
    return da.weighted(np.cos(np.deg2rad(da[lat]))).mean((lat, lon))


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------

@skip_empty
def ttest_welch(da1, da2, dims=("year", "member"), variable="tas"):
    """Welch's t-test across ``dims``, returning ``t`` and the two-sided ``p``."""
    n1 = np.prod([da1.sizes[d] for d in dims])
    n2 = np.prod([da2.sizes[d] for d in dims])

    m1, m2 = da1.mean(dims), da2.mean(dims)
    v1, v2 = da1.var(dims, ddof=1), da2.var(dims, ddof=1)

    se2 = v1 / n1 + v2 / n2
    t = (m1 - m2) / np.sqrt(se2)

    df = se2**2 / (
        (v1 / n1) ** 2 / (n1 - 1)
        + (v2 / n2) ** 2 / (n2 - 1)
    )

    p = 2 * scipy.special.stdtr(df, -np.abs(t))

    return xr.Dataset({
        "t": t[variable],
        "p": p[variable],
    })
