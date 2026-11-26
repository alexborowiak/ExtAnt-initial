"""Significance: p-values, and every test that resamples members. Each works on any ensembles, on (member, year, ...).

All resampling works on whole members, so each member's time series stays intact.

    weighted_member_sums                     each trial's weighted sum over the members, the step every resampling shares
    pvalue_two_sided                         a p-value from any set of samples (bootstrap trials, permutations)
    bootstrap_member_mean                    the ensemble mean of members drawn with replacement (``resample_weights``,
                                             ``resample_members``)
    quantile_shift                           the change in every quantile, with its bootstrap range
    bootstrap_qrange, bootstrap_mean         the width and the mean of 10 hist-nat members drawn at random, trial by trial
                                             (``bootstrap_draws`` says which)
    member_block_pvalue                      the permutation test of a difference in means (``permutation_weights``)
    model_agreement, significant_counts,     how many models agree
    significant_sign, robust_sign
    ttest_welch                              Welch's t-test

The width test and the mean test (notebooks calc_02a and calc_02b, sections *Quantiles, Width and Tails* and *Mean Change and Signal to Noise*) compare an
experiment's final years with hist-nat's over the same years. Their null is a
bootstrap of hist-nat alone: each trial is 10 different hist-nat members, all
the years given pooled (``bootstrap_draws``). ``bootstrap_qrange`` gives each
trial's width (numba kernels, parallel over grid points for data in memory and
serial for dask blocks), ``bootstrap_mean`` its mean, from the same draws.
"""

import numpy as np
import scipy.special
import xarray as xr
from numba import njit, prange

from ..utils.datatree import skip_empty
from .quantiles import numpy_lerp


# ---------------------------------------------------------------------------
# Weighted sums over the members, for resampling
# ---------------------------------------------------------------------------

def _weighted_sums(values, weights):
    return values @ weights.T


def weighted_member_sums(values, weights):
    """Each trial's weighted sum over the members: ``values`` (member, ...) and ``weights`` (trial, member).

    One matrix product per dask block, so the result keeps the blocks of
    ``values`` with the trials added. ``xr.dot`` gives the same numbers, but
    dask's einsum splits every block into hundreds of small ones (about 2,500
    tasks for one full-size node instead of 5), which a whole tree of them
    turns into more tasks than the scheduler can hold.

    Example: values [8, 10, 9, 12, 14] (one per member) and the weights
    [[0, 1, 0, 0, 1], [0.25, 0, 0.5, 0.25, 0]] give [10 + 14, 0.25 * 8 + 0.5 * 9 + 0.25 * 12] = [24, 9.5]:
    the first row adds up the two members a trial drew, the second averages four
    draws with replacement (member 2 drawn twice).

    Args:
        values (xr.DataArray | xr.Dataset): On (member, ...).
        weights (np.ndarray): (trial, member): row k is how much each member counts in trial k.

    Returns:
        The type of ``values``, with ``member`` replaced by ``trial`` (first, numbered from 0).
    """
    if values.chunks:
        values = values.chunk({"member": -1})
    out = xr.apply_ufunc(
        _weighted_sums, values,
        input_core_dims=[["member"]], output_core_dims=[["trial"]],
        kwargs={"weights": weights}, dask="parallelized", output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"trial": weights.shape[0]}},
    )
    return out.transpose("trial", ...).assign_coords(trial=np.arange(weights.shape[0]))


# ---------------------------------------------------------------------------
# The p-value of a value against a set of samples
# ---------------------------------------------------------------------------

def pvalue_two_sided(samples, observed, dim="trial", method="tails"):
    """Two-sided resampling p-value, with the +1 correction; NaN where ``observed`` is NaN.

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
        pvalue = ((np.abs(samples) >= np.abs(observed)).sum(dim) + 1) / (n + 1)
    else:
        upper = ((samples >= observed).sum(dim) + 1) / (n + 1)
        lower = ((samples <= observed).sum(dim) + 1) / (n + 1)
        pvalue = (2 * np.minimum(upper, lower)).clip(max=1)
    #(c): NaN compares False with everything, which would otherwise give the smallest p-value there is
    return pvalue.where(observed.notnull())


# ---------------------------------------------------------------------------
# Members drawn with replacement
# ---------------------------------------------------------------------------

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


def resample_weights(n_members, n_trials, rng):
    """How often each member is drawn in each trial of a bootstrap with replacement, as a share of the draws.

    Trial k draws ``n_members`` members at random with replacement; row k
    holds how many times each member was drawn, over ``n_members``. A row
    times the members' values is then the mean of the members drawn, so every
    trial is one matrix product.

    Example: 4 members, and a trial draws members [2, 0, 2, 3]: member 2 twice,
    0 and 3 once, 1 never, so its row is [1, 0, 2, 1] / 4 = [0.25, 0, 0.5, 0.25].
    Notebook calc_02c (section *Probability Ratios*) checks trial 0.

    Returns:
        np.ndarray: (n_trials, n_members); every row sums to 1.
    """
    #(c): Each trial's members drawn, with replacement: (trial, draw)
    draws = rng.integers(0, n_members, size=(n_trials, n_members))
    #(c): Each draw compared with every member, (trial, draw, member), and the matches added up over the draws
    counts = (draws[:, :, None] == np.arange(n_members)).sum(axis=1)
    return counts / n_members


@skip_empty
def bootstrap_member_mean(members, n_trials=1000, seed=0):
    """The ensemble mean of members drawn with replacement, ``n_trials`` times.

    Each trial draws as many members as there are, with replacement, and
    averages them (``resample_weights``). Calls with the same seed and the same
    number of members draw the same members, so two things made from one
    ensemble (hist-nat's own chance and its shift-only chance, against the
    moved thresholds) are resampled together, and their ratio keeps its pairing.

    Args:
        members (xr.Dataset | xr.DataArray): On (member, ...), e.g. ``probability_ratio.member_exceedance`` output.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` replaced by ``trial`` (first, numbered from 0).
    """
    weights = resample_weights(members.sizes["member"], n_trials, np.random.default_rng(seed))
    return weighted_member_sums(members, weights)


#(t): Quantile levels for the quantile-shift curve
SHIFT_QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)


def quantile_shift(experiment, reference, quantiles=SHIFT_QUANTILES, n_boot=1000, seed=0):
    """Change in each quantile, experiment minus reference, pooling members and years, with a bootstrap range.

    The bootstrap resamples whole members (with replacement) in both
    ensembles, as in ``resample_members``, so each member's years
    stay together.

    Args:
        experiment, reference (xr.DataArray): Samples on (member, year), e.g. from ``response_change.final_years``.
        quantiles (Sequence[float]): Quantile levels.
        n_boot (int): Bootstrap resamples.
        seed (int): Random seed.

    Returns:
        xr.Dataset:
            experiment_quantiles, reference_quantiles   (quantile)
            shift                                       (quantile) experiment minus reference
            shift_lower, shift_upper                    (quantile) bootstrap 5-95% range
    """
    pool = ("member", "year")
    rng = np.random.default_rng(seed)
    exp_q = experiment.quantile(quantiles, dim=pool)
    ref_q = reference.quantile(quantiles, dim=pool)
    boot = (resample_members(experiment, n_boot, rng).quantile(quantiles, dim=pool)
            - resample_members(reference, n_boot, rng).quantile(quantiles, dim=pool))
    #(c): Move the quantile levels aside so xarray's (fast) quantile over trials can add its own dim
    band = (boot.rename(quantile="level").quantile([0.05, 0.95], dim="trial")
            .rename(quantile="bootstrap", level="quantile"))
    return xr.Dataset({
        "experiment_quantiles": exp_q,
        "reference_quantiles": ref_q,
        "shift": exp_q - ref_q,
        "shift_lower": band.sel(bootstrap=0.05, drop=True),
        "shift_upper": band.sel(bootstrap=0.95, drop=True),
    })


# ---------------------------------------------------------------------------
# The hist-nat bootstrap of the width and of the mean
# ---------------------------------------------------------------------------
# A trial's Q-range pools some members' values. Instead of gathering and sorting
# those values for every trial, each grid point's pool is sorted once, and each
# trial reads its quantiles off it by walking in from the nearer end and counting
# only its own members' values: a few dozen steps for Q05 and Q95 instead of a sort.
#
# Example: a trial of 10 members x 21 years has 210 values, so np.quantile puts
# Q05 at position 209 * 0.05 = 10.45, between the trial's 11th and 12th smallest
# values. Walking up the sorted pool and counting only the trial's values, the
# walk stops at the 12th. Q95, at 198.55, is the 11th and 12th largest, so that
# walk starts from the top. Notebook calc_02a (section *Quantiles, Width and Tails*) goes through it on one trial.

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
    #(c): np.quantile's position among the group's n values: Q05 of 210 values is at 10.45, between values 10 and 11
    position = (n_group - 1) * q
    k = int(np.floor(position))
    low = high = np.nan
    #(c): How many of the group's values the walk has passed, less one: 0 at the first of them
    count = -1
    if from_top:
        #(c): Counting down from the largest, value k is number n_group - 1 - k and value k + 1 the one before it
        #(c): (Q95 of 210 is at 198.55: values 198 and 199 from the bottom are numbers 11 and 10 from the top)
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


def _trial_qrange(data, selected, q_low, q_high, out):
    """``out[s, t]``: the Q-range of trial t's members, all their years pooled, at point s.

    Each point's values are sorted once, for every trial.

    Args:
        data (np.ndarray): (point, member, year).
        selected (np.ndarray): (trial, member) True for the members each trial samples.
    """
    n_points, n_members, n_years = data.shape
    for s in prange(n_points):
        values = np.empty(n_members * n_years)
        members = np.empty(n_members * n_years, dtype=np.int64)
        counts = np.empty(n_members, dtype=np.int64)
        n_valid = _sorted_pool(data[s], values, members, counts)
        for t in range(selected.shape[0]):
            n_group = 0
            for m in range(n_members):
                if selected[t, m]:
                    n_group += counts[m]
            out[s, t] = _group_qrange(values, members, n_valid, selected[t], n_group, q_low, q_high)


_trial_qrange_serial = njit(nogil=True)(_trial_qrange)
_trial_qrange_parallel = njit(parallel=True)(_trial_qrange)


def _qrange_block(data, selected, q_low, q_high, parallel):
    points = np.ascontiguousarray(data.reshape(-1, *data.shape[-2:]))
    out = np.empty((points.shape[0], selected.shape[0]), dtype=np.float32)
    kernel = _trial_qrange_parallel if parallel else _trial_qrange_serial
    kernel(points, selected, q_low, q_high, out)
    return out.reshape(*data.shape[:-2], selected.shape[0])


def bootstrap_draws(n_hist_nat_members, n_members, n_trials, seed):
    """Which hist-nat members each bootstrap trial uses: ``n_members`` different ones, at random.

    Args:
        n_hist_nat_members (int): Members in the hist-nat ensemble.
        n_members (int): Members per trial (``config.N_BOOTSTRAP_MEMBERS``).
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        np.ndarray: ``selected``, (trial, member) True for each trial's members.

    Example: 5 members and 2 per trial. A trial's random numbers
    [0.7, 0.1, 0.5, 0.9, 0.3] are smallest at members 1 and 4, so its row is
    [False, True, False, False, True]: every pair as likely as any other, and
    no member twice. Notebook calc_02a (section *Quantiles, Width and Tails*) shows trial 0's members.
    """
    if n_members > n_hist_nat_members:
        raise ValueError("n_members cannot exceed the hist-nat members available.")

    rng = np.random.default_rng(seed)
    #(c): The n_members smallest of one uniform draw per member: n_members different members, every set as likely
    sampled_member_indices = np.argpartition(
        rng.random((n_trials, n_hist_nat_members)), n_members - 1, axis=1,
    )[:, :n_members]
    selected = np.zeros((n_trials, n_hist_nat_members), dtype=np.bool_)
    np.put_along_axis(selected, sampled_member_indices, True, axis=1)
    return selected


@skip_empty
def bootstrap_qrange(hist_nat, n_members, quantiles, n_trials, seed):
    """The Q-range (e.g. Q95 - Q05) of each hist-nat bootstrap trial.

    Trial k pools every year given of ``n_members`` different hist-nat members,
    those ``bootstrap_draws(..., seed=seed)`` picks: given hist-nat's final 21
    years, each trial is 10 of its members over those years. NaN are skipped,
    so a trial of the seasons of one SAM phase pools only those. Every grid
    point uses the same draws.

    Args:
        hist_nat (xr.DataArray | xr.Dataset): hist-nat over the years compared, on (member, year, ...).
        n_members (int): Members per trial.
        quantiles (tuple[float, float]): The lower and upper quantile.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` and ``year`` replaced by ``trial`` (first, numbered from 0).
    """
    selected = bootstrap_draws(hist_nat.sizes["member"], n_members, n_trials, seed)

    if hist_nat.chunks:
        hist_nat = hist_nat.chunk({"member": -1, "year": -1})
    out = xr.apply_ufunc(
        _qrange_block,
        hist_nat,
        input_core_dims=[["member", "year"]],
        output_core_dims=[["trial"]],
        kwargs=dict(selected=selected, q_low=quantiles[0], q_high=quantiles[1], parallel=not hist_nat.chunks),
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"trial": n_trials}},
    )
    return out.transpose("trial", ...).assign_coords(trial=np.arange(n_trials))


@skip_empty
def bootstrap_mean(hist_nat, n_members, n_trials, seed):
    """The mean of each hist-nat bootstrap trial's values, pooled; NaN are skipped.

    The trials of ``bootstrap_qrange`` with the same arguments: trial k pools
    every year given of the ``n_members`` hist-nat members
    ``bootstrap_draws(..., seed=seed)`` picks. A trial's mean is the sum of its
    members' values over how many there are, so each member's sum and count
    over the years are added up for every trial at once. Every grid point uses the same draws.

    Example: a trial of members 1 and 4, whose member sums are 10 and 14 over
    21 years each. Its row of the draws, [0, 1, 0, 0, 1], times the sums
    [8, 10, 9, 12, 14] is 24, times the counts [21, 21, 21, 21, 21] is 42, so
    its mean is 24 / 42 = 0.57; one matrix product gives every row at once. Notebook
    calc_02b (section *Mean Change and Signal to Noise*) goes through it on trial 0.

    Args:
        hist_nat (xr.DataArray | xr.Dataset): hist-nat over the years compared, on (member, year, ...).
        n_members (int): Members per trial.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` and ``year`` replaced by ``trial`` (first, numbered from 0).
    """
    selected = bootstrap_draws(hist_nat.sizes["member"], n_members, n_trials, seed).astype(np.float64)
    #(c): sum skips NaN; count counts what it added, so a member with no values in a trial adds nothing
    total = weighted_member_sums(hist_nat.sum("year", dtype=np.float64), selected)
    count = weighted_member_sums(hist_nat.count("year"), selected)
    return total / count


# ---------------------------------------------------------------------------
# The permutation test of a difference in means, whole members relabelled
# ---------------------------------------------------------------------------

def permutation_weights(n_permutations, n_exp, n_nat, rng):
    """Random relabellings of ``n_exp + n_nat`` members, each as a row of weights.

    A row has +1/n_exp on the members relabelled as the experiment and
    -1/n_nat on the rest, so ``weights @ member_means`` is each relabelling's
    difference in means. The experiment's members come first in the pool.

    Example: 3 experiment members and 2 hist-nat ones, pooled. The random
    numbers [0.6, 0.2, 0.9, 0.1, 0.4] sort to the order [3, 1, 4, 0, 2], so
    pooled members 3, 1 and 4 are relabelled as the experiment:
    [-1/2, 1/3, -1/2, 1/3, 1/3]. Notebook calc_02b (section *Mean Change and Signal to Noise*) checks the first row.

    Returns:
        np.ndarray: (n_permutations, n_exp + n_nat).
    """
    #(c): One random number per pooled member; their order shuffles the members
    order = np.argsort(rng.random((n_permutations, n_exp + n_nat)), axis=1)
    weights = np.full((n_permutations, n_exp + n_nat), -1 / n_nat)
    #(c): The first n_exp members of each shuffle are relabelled as the experiment
    np.put_along_axis(weights, order[:, :n_exp], 1 / n_exp, axis=1)
    return weights


def _member_block_pvalue(experiment, hist_nat, n_permutations, batch_size, seed):
    """Two-sided permutation p-values of the difference in means, for arrays on (..., member)."""
    n_exp, n_nat = experiment.shape[-1], hist_nat.shape[-1]
    observed = experiment.mean(-1) - hist_nat.mean(-1)
    pooled = np.nan_to_num(np.concatenate([experiment, hist_nat], axis=-1))
    rng = np.random.default_rng(seed)
    at_least = np.zeros(observed.shape)
    at_most = np.zeros(observed.shape)
    for start in range(0, n_permutations, batch_size):
        weights = permutation_weights(min(batch_size, n_permutations - start), n_exp, n_nat, rng)
        differences = pooled @ weights.T
        at_least += (differences >= observed[..., None]).sum(-1)
        at_most += (differences <= observed[..., None]).sum(-1)
    pvalue = np.minimum(1, 2 * np.minimum(at_least + 1, at_most + 1) / (n_permutations + 1))
    return np.where(np.isfinite(observed), pvalue, np.nan)


@skip_empty
def member_block_pvalue(experiment, hist_nat, n_permutations=5000, batch_size=500, seed=0):
    """Permutation p-value of the difference in means, experiment minus hist-nat, permuting whole members.

    Give it each member's mean over the years tested (e.g. the final
    ``WINDOW`` years): permuting member means is permuting whole members, each
    keeping its years together. A pooled t-test treats every member-year as
    independent, but a member's years are serially correlated; members are
    independent realisations, so this test respects that correlation.

    Each permutation relabels the pooled members, as many as the experiment
    has as the experiment and the rest as hist-nat (``permutation_weights``,
    the experiment's members first), and takes the difference in means. The
    p-value is two-sided, with the +1 correction, as in
    ``pvalue_two_sided``. The permutations are drawn in batches of
    ``batch_size``, one matrix product each, from ``np.random.default_rng(seed)``.

    Args:
        experiment, hist_nat (xr.DataArray | xr.Dataset): Member means on (member, ...); the numbers of
            members may differ.
        n_permutations (int): Relabellings.
        batch_size (int): Relabellings per matrix product.
        seed (int): Random seed.

    Returns:
        The same type, without ``member``; NaN where the difference in means is NaN.
    """
    if experiment.chunks:
        experiment = experiment.chunk({"member": -1})
    if hist_nat.chunks:
        hist_nat = hist_nat.chunk({"member": -1})
    return xr.apply_ufunc(
        _member_block_pvalue, experiment, hist_nat,
        input_core_dims=[["member"], ["member"]], exclude_dims={"member"},
        kwargs=dict(n_permutations=n_permutations, batch_size=batch_size, seed=seed),
        dask="parallelized", output_dtypes=[float],
    )


# ---------------------------------------------------------------------------
# How many models agree
# ---------------------------------------------------------------------------

def model_agreement(change, significant=None, dim="model", *, sign_threshold, significant_threshold):
    """Collapse a per-model field into a multi-model summary (IPCC AR6 "advanced approach").

    Args:
        change (xr.DataArray): Per-model change, with ``dim``.
        significant (xr.DataArray | None): Per-model boolean significance.
        sign_threshold (float): Fraction of models that must share the sign of the median (``config.SIGN_THRESHOLD``).
        significant_threshold (float): Fraction of models that must be significant
            (``config.SIGNIFICANT_THRESHOLD``).

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


def significant_counts(change, significant, dim="model"):
    """How many models have a significant increase, and how many a significant decrease.

    A model with no data at a point (NaN change) counts in neither.

    Args:
        change (xr.DataArray): Per-model change, with ``dim``.
        significant (xr.DataArray): Per-model boolean significance, the same dims.
        dim (str): The dim counted over.

    Returns:
        xr.DataArray: Counts with a ``direction`` dim of ["increase", "decrease"], and ``dim`` reduced.
    """
    increase = (significant & (change > 0)).sum(dim)
    decrease = (significant & (change < 0)).sum(dim)
    return xr.concat([increase, decrease], dim=xr.Variable("direction", ["increase", "decrease"]))


def significant_sign(change, significant):
    """+1 where a change is significant and positive, -1 significant and negative, 0 not significant; NaN without data."""
    return np.sign(change).where(significant, 0).where(change.notnull())


def robust_sign(change, significant, threshold, dim="model"):
    """Where the models agree: +1 where at least ``threshold`` of them have a significant increase, -1 a decrease.

    0 elsewhere, and NaN where no model has data. The fraction is of the
    models with data at each point, so with six models and a threshold of
    0.66 four must agree; with two, both. Both signs cannot pass a threshold
    above one half.

    Args:
        change (xr.DataArray): Per-model change, with ``dim``.
        significant (xr.DataArray): Per-model boolean significance, the same dims.
        threshold (float): Fraction of the models that must have a significant change of one sign
            (``config.SIGNIFICANT_THRESHOLD``, after the IPCC AR6 advanced approach).
        dim (str): The dim counted over.

    Returns:
        xr.DataArray: -1, 0 or +1 (float, for the NaN), ``dim`` reduced.
    """
    counts = significant_counts(change, significant, dim)
    n_models = change.notnull().sum(dim)
    increase = counts.sel(direction="increase", drop=True) >= threshold * n_models
    decrease = counts.sel(direction="decrease", drop=True) >= threshold * n_models
    return (increase.astype(float) - decrease.astype(float)).where(n_models > 0)


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------

@skip_empty
def ttest_welch(da1, da2, dims=("year", "member"), variable=None):
    """Welch's t-test across ``dims``, returning ``t`` and the two-sided ``p``.

    Takes two DataArrays, or two Datasets (as ``xr.map_over_datasets`` gives them) with ``variable`` naming the one
    tested. Missing values are left out: n is the number of values there are, as for the means and variances.

    Example: 10 members × 21 years is 210 values, but a SAM phase keeps about a third of the seasons (the rest NaN), so
    n is about 70, not 210. Counting 210 would make the standard error too small and p far too small.
    """
    if isinstance(da1, xr.Dataset):
        da1, da2 = da1[variable], da2[variable]

    #(c): The values that are there, not the size of the array
    n1 = da1.count(dims)
    n2 = da2.count(dims)

    m1, m2 = da1.mean(dims), da2.mean(dims)
    v1, v2 = da1.var(dims, ddof=1), da2.var(dims, ddof=1)

    se2 = v1 / n1 + v2 / n2
    t = (m1 - m2) / np.sqrt(se2)

    df = se2**2 / (
        (v1 / n1) ** 2 / (n1 - 1)
        + (v2 / n2) ** 2 / (n2 - 1)
    )

    p = 2 * scipy.special.stdtr(df, -np.abs(t))

    return xr.Dataset({"t": t, "p": p})
