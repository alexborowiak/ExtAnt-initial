"""Resampling and t-test significance for experiment vs reference ensembles.

All resampling works on whole members, so each member's time series stays intact.

The Q-range test is a bootstrap of hist-nat alone (``qrange_significance``):
its Q-range kernels run in numba, compiled (like quantile_calc's) parallel
over grid points for data in memory and serial for dask blocks.
"""

import numpy as np
import scipy.special
import xarray as xr
from numba import njit, prange

from quantile_calc import WINDOW, numpy_lerp, quantile_range
from xarray_datatree_utils import skip_empty
from xarray_stats import nan_quantile


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
    bounds = nan_quantile(samples, [alpha / 2, 1 - alpha / 2], dim)
    lower = bounds.sel(quantile=alpha / 2, drop=True)
    upper = bounds.sel(quantile=1 - alpha / 2, drop=True)
    return lower, upper, (observed < lower) | (observed > upper)


# ---------------------------------------------------------------------------
# Member resampling
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


_window_qrange_serial = njit(nogil=True)(_window_qrange)
_window_qrange_parallel = njit(parallel=True)(_window_qrange)


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


#(t): Members in every hist-nat bootstrap draw: the smallest LESFMIP ensemble, so one null serves every experiment
N_BOOTSTRAP_MEMBERS = 10


def sample_hist_nat_qrange_changes(
    hist_nat_da,
    hist_nat_reference_qrange_da,
    n_members_to_sample=N_BOOTSTRAP_MEMBERS,
    years=WINDOW,
    quantiles=(0.05, 0.95),
    n_trials=10_000,
    seed=0,
):
    """The hist-nat bootstrap behind ``qrange_significance``: N different members over a random ``years``-year window.

    Returns each trial's Q-range minus ``hist_nat_reference_qrange_da``: the
    changes obtainable from natural variability and sample size alone.
    Windows come from anywhere in the record, each used about equally often.
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
    n_members=N_BOOTSTRAP_MEMBERS,
    years=WINDOW,
    quantiles=(0.05, 0.95),
    n_trials=1000,
    alpha=0.05,
    seed=0,
):
    """Is each experiment's change in Q-range more than hist-nat's natural variability can produce?

    The change is the experiment's Q-range over its final ``years`` (members
    and years pooled) minus hist-nat's over its full record. Its null is a
    bootstrap of hist-nat alone: each trial pools ``n_members`` different
    hist-nat members over a random ``years``-year window, and subtracts the
    same full-record Q-range. So the trials are the changes that natural
    variability and sampling produce on their own.

    The null depends only on hist-nat, so it is drawn once per model and
    shared by every experiment. With a fixed number of members (the smallest
    LESFMIP ensemble), it is the same test for every experiment, whatever its
    ensemble size. A larger ensemble pins its Q-range down more precisely than
    10 members can, so for it the test is conservative.

    Args:
        experiments (dict[str, xr.DataArray]): Experiment name -> data with
            ``member`` and ``year`` dims.
        hist_nat (xr.DataArray): The model's hist-nat, full record.
        n_members (int): hist-nat members pooled in each trial.
        years (int): Length of the final period and of each trial's window (odd).
        quantiles (tuple[float, float]): The Q-range's lower and upper quantiles.
        n_trials (int): Bootstrap trials.
        alpha (float): Two-sided significance level.
        seed (int): Random seed.

    Returns:
        xr.Dataset, with an ``experiment`` dim:
            hist_nat_qrange      hist-nat's Q-range over its full record
            experiment_qrange    the experiment's over its final ``years``
            qrange_change        experiment_qrange - hist_nat_qrange
            null_lower           the bootstrap changes' alpha/2 quantile
            null_upper           and their 1 - alpha/2 quantile
            qrange_pvalue        two-sided bootstrap p-value of qrange_change
            qrange_significant   qrange_pvalue < alpha
    """
    hist_nat_qrange = quantile_range(hist_nat, quantiles).compute()
    null = sample_hist_nat_qrange_changes(
        hist_nat, hist_nat_qrange, n_members,
        years=years, quantiles=quantiles, n_trials=n_trials, seed=seed,
    )
    null_range = nan_quantile(null, [alpha / 2, 1 - alpha / 2], "trial")

    results = []
    for exp_da in experiments.values():
        experiment_qrange = quantile_range(exp_da.isel(year=slice(-years, None)), quantiles).compute()
        change = experiment_qrange - hist_nat_qrange
        pvalue = pvalue_two_sided(null, change, method="tails").where(change.notnull())
        results.append(xr.Dataset({
            "hist_nat_qrange": hist_nat_qrange,
            "experiment_qrange": experiment_qrange,
            "qrange_change": change,
            "null_lower": null_range.isel(quantile=0, drop=True),
            "null_upper": null_range.isel(quantile=1, drop=True),
            "qrange_pvalue": pvalue,
            "qrange_significant": pvalue < alpha,
        }))

    out = xr.concat(results, dim="experiment").assign_coords(experiment=list(experiments))
    return out.assign_attrs(n_members=n_members, years=years, n_trials=n_trials, alpha=alpha,
                            quantiles=list(quantiles))


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
