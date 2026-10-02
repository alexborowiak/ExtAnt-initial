"""Resampling, permutation and t-test significance for experiment vs reference ensembles.

All resampling works on whole members, so each member's time series stays intact.

The Q-range tests run in numba kernels, compiled (like quantile_calc's)
parallel over grid points for data in memory and serial for dask blocks.
"""

from functools import partial

import numpy as np
import scipy.special
import xarray as xr
from numba import njit, prange

from quantile_calc import numpy_lerp, quantile_range
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
            test statistic. None is the pooled Q95-Q05 range, run by the fast
            ``qrange_permutation_samples`` (same draws, same numbers).
        batch_size (int): Trials per batch for a custom ``statistic``.

    Returns:
        xr.DataArray: float32 samples along a ``trial`` dim.
    """
    if statistic is None:
        return qrange_permutation_samples(exp, reference, n_trials=n_trials, seed=seed).compute()
    rng = np.random.default_rng(seed)
    batches = [
        b.astype("float32")
        for b in permutation_batches(exp, reference, statistic, n_trials, batch_size, rng)
    ]
    return xr.concat(batches, dim="trial").assign_coords(trial=np.arange(n_trials))


# ---------------------------------------------------------------------------
# Q-range kernels (numba)
# ---------------------------------------------------------------------------
# A trial's Q-range pools some members' values. Instead of gathering and sorting
# those values for every trial, each grid point's pool is sorted once, and each
# trial reads its quantiles off it by walking in from the nearer end and counting
# only its own members' values: a few dozen steps for Q05 and Q95 instead of a sort.

@njit
def _sorted_pool(block, values, members, counts):
    """Sort one point's (member, sample) values ascending, dropping NaN; returns how many are valid.

    ``members`` gets the member of each sorted value and ``counts`` each member's number of valid values.
    """
    n_members, n_samples = block.shape
    n = 0
    for m in range(n_members):
        counts[m] = 0
        for j in range(n_samples):
            value = block[m, j]
            if not np.isnan(value):
                values[n] = value
                members[n] = m
                counts[m] += 1
                n += 1
    order = np.argsort(values[:n])
    values[:n] = values[:n][order]
    members[:n] = members[:n][order]
    return n


@njit
def _group_quantile(values, members, n_valid, in_group, n_group, q, from_top):
    """Quantile ``q`` of the group's values, read off the pool (``values``, sorted ascending).

    Gives np.quantile's number for those values alone. Walks from the top for
    high quantiles and from the bottom for low ones, counting only the group.
    """
    position = (n_group - 1) * q
    k = int(np.floor(position))
    low = high = np.nan
    count = -1
    if from_top:
        #(c): Counting down from the largest, value k is number n_group - 1 - k and value k + 1 the one before it
        for i in range(n_valid - 1, -1, -1):
            if in_group[members[i]]:
                count += 1
                if count == n_group - 2 - k:
                    high = values[i]
                elif count == n_group - 1 - k:
                    low = values[i]
                    break
    else:
        for i in range(n_valid):
            if in_group[members[i]]:
                count += 1
                if count == k:
                    low = values[i]
                elif count == k + 1:
                    high = values[i]
                    break
    if k >= n_group - 1:
        return low
    return numpy_lerp(low, high, position - k)


@njit
def _group_qrange(values, members, n_valid, in_group, n_group, q_low, q_high):
    if n_group == 0:
        return np.nan
    return (_group_quantile(values, members, n_valid, in_group, n_group, q_high, True)
            - _group_quantile(values, members, n_valid, in_group, n_group, q_low, False))


def _permutation_qrange(pooled, in_exp, q_low, q_high, out):
    """``out[s, t]``: trial t's experiment-group Q-range minus its reference-group Q-range, at point s.

    Args:
        pooled (np.ndarray): (point, member, sample), experiment and reference members together.
        in_exp (np.ndarray): (trial, member) True for the members labelled experiment in that trial.
    """
    n_points, n_members, n_samples = pooled.shape
    for s in prange(n_points):
        values = np.empty(n_members * n_samples)
        members = np.empty(n_members * n_samples, dtype=np.int64)
        counts = np.empty(n_members, dtype=np.int64)
        in_ref = np.empty(n_members, dtype=np.bool_)
        n_valid = _sorted_pool(pooled[s], values, members, counts)
        for t in range(in_exp.shape[0]):
            n_exp = 0
            for m in range(n_members):
                in_ref[m] = not in_exp[t, m]
                if in_exp[t, m]:
                    n_exp += counts[m]
            out[s, t] = (_group_qrange(values, members, n_valid, in_exp[t], n_exp, q_low, q_high)
                         - _group_qrange(values, members, n_valid, in_ref, n_valid - n_exp, q_low, q_high))


def _window_qrange(data, reference, selected, starts, length, trials, first, q_low, q_high, out):
    """``out[s, t]``: the Q-range of trial t's selected members over its window of years, minus ``reference[s]``.

    Trials are grouped by window, so each point's window is sorted once for all
    the trials that use it: trials ``trials[first[w]:first[w + 1]]`` use the
    years ``starts[w]:starts[w] + length``.

    Args:
        data (np.ndarray): (point, member, year).
        reference (np.ndarray): (point,).
        selected (np.ndarray): (trial, member) True for the members each trial samples.
    """
    n_points, n_members, _ = data.shape
    for s in prange(n_points):
        values = np.empty(n_members * length)
        members = np.empty(n_members * length, dtype=np.int64)
        counts = np.empty(n_members, dtype=np.int64)
        for w in range(starts.size):
            n_valid = _sorted_pool(data[s, :, starts[w]:starts[w] + length], values, members, counts)
            for i in range(first[w], first[w + 1]):
                t = trials[i]
                n_group = 0
                for m in range(n_members):
                    if selected[t, m]:
                        n_group += counts[m]
                out[s, t] = _group_qrange(values, members, n_valid, selected[t], n_group, q_low, q_high) - reference[s]


_permutation_qrange_serial = njit(nogil=True)(_permutation_qrange)
_permutation_qrange_parallel = njit(parallel=True)(_permutation_qrange)
_window_qrange_serial = njit(nogil=True)(_window_qrange)
_window_qrange_parallel = njit(parallel=True)(_window_qrange)


def _permutation_block(pooled, in_exp, q_low, q_high, parallel):
    points = np.ascontiguousarray(pooled.reshape(-1, *pooled.shape[-2:]))
    out = np.empty((points.shape[0], in_exp.shape[0]), dtype=np.float32)
    (_permutation_qrange_parallel if parallel else _permutation_qrange_serial)(points, in_exp, q_low, q_high, out)
    return out.reshape(*pooled.shape[:-2], in_exp.shape[0])


def _window_block(data, reference, selected, starts, length, trials, first, q_low, q_high, parallel):
    points = np.ascontiguousarray(data.reshape(-1, *data.shape[-2:]))
    out = np.empty((points.shape[0], selected.shape[0]), dtype=np.float32)
    kernel = _window_qrange_parallel if parallel else _window_qrange_serial
    reference = np.ascontiguousarray(np.broadcast_to(reference, data.shape[:-2]), dtype=float).reshape(-1)
    kernel(points, reference, selected, starts, length, trials, first, q_low, q_high, out)
    return out.reshape(*data.shape[:-2], selected.shape[0])


def _per_trial(block, inputs, core_dims, n_trials, **kwargs):
    """Run a kernel block function over every point of ``inputs``, adding a ``trial`` dim (first)."""
    first = inputs[0]
    if first.chunks is not None:
        inputs = [inputs[0].chunk({d: -1 for d in core_dims[0]}), *inputs[1:]]
    out = xr.apply_ufunc(
        block, *inputs,
        input_core_dims=core_dims, output_core_dims=[["trial"]],
        kwargs={**kwargs, "parallel": first.chunks is None},
        dask="parallelized", output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"trial": n_trials}},
    )
    return out.transpose("trial", ...).assign_coords(trial=np.arange(n_trials))


def qrange_permutation_samples(exp, reference, quantiles=(0.05, 0.95), n_trials=10_000, seed=0):
    """Permutation null of the experiment-minus-reference pooled Q-range, in one numba pass.

    Whole members are randomly relabelled between the two ensembles, keeping
    their sizes, exactly as ``permutation_samples`` does (the same random
    draws for the same seed, so the same numbers). Lazy for dask input.

    Returns:
        xr.DataArray: float32 samples along a ``trial`` dim (first).
    """
    rng = np.random.default_rng(seed)
    pooled = pool_members(exp, reference)
    n_exp, n_total = exp.sizes["member"], pooled.sizes["pool_member"]
    in_exp = np.zeros((n_trials, n_total), dtype=np.bool_)
    np.put_along_axis(in_exp, np.argsort(rng.random((n_trials, n_total)), axis=1)[:, :n_exp], True, axis=1)
    return _per_trial(_permutation_block, [pooled], [["pool_member", "year"]], n_trials,
                      in_exp=in_exp, q_low=quantiles[0], q_high=quantiles[1])


def sample_hist_nat_qrange_changes(
    hist_nat_da,
    hist_nat_reference_qrange_da,
    n_members_to_sample,
    years=11,
    quantiles=(0.05, 0.95),
    n_trials=10_000,
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

    # N distinct members per trial (the N smallest of one uniform draw per member).
    sampled_member_indices = np.argpartition(
        rng.random((n_trials, n_hist_nat_members)), n_members_to_sample - 1, axis=1,
    )[:, :n_members_to_sample]
    selected = np.zeros((n_trials, n_hist_nat_members), dtype=np.bool_)
    np.put_along_axis(selected, sampled_member_indices, True, axis=1)

    # Trials grouped by window, so each window is sorted once.
    window_starts = sampled_centre_indices - half_window
    trials = np.argsort(window_starts, kind="stable")
    starts, counts = np.unique(window_starts, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)])

    return _per_trial(
        _window_block, [hist_nat_da, hist_nat_reference_qrange_da], [["member", "year"], []], n_trials,
        selected=selected, starts=starts, length=years, trials=trials, first=first,
        q_low=quantiles[0], q_high=quantiles[1],
    ).compute()


def qrange_significance(
    experiments,
    hist_nat,
    n_members_to_sample=None,
    years=11,
    quantiles=(0.05, 0.95),
    n_trials=10_000,
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
        years=years, quantiles=quantiles, n_trials=n_trials, seed=seed,
    )

    results = []
    for exp_da in experiments.values():
        exp_da = exp_da.isel(year=slice(-years, None))
        exp_qrange = qrange(exp_da).compute()

        # 1. Experiment vs the long-term hist-nat reference.
        change = exp_qrange - hist_nat_qrange
        lower, upper, outside = outside_bounds(hist_nat_samples, change, alpha=alpha)

        # 2. Experiment vs hist-nat over the same years.
        #(c): Reduced to a p-value where the samples are made (on the workers, for dask input)
        period_difference = exp_qrange - hist_nat_period_qrange
        permutations = qrange_permutation_samples(exp_da, hist_nat_period, quantiles=quantiles,
                                                  n_trials=n_trials, seed=seed)
        pvalue = pvalue_two_sided(permutations, period_difference, method="tails").compute()

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
