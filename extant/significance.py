"""Resampling and t-test significance for experiment vs hist-nat ensembles.

All resampling works on whole members, so each member's time series stays intact.

The width test's null is a bootstrap of hist-nat alone (``bootstrap_qrange``):
its Q-range kernels run in numba, compiled (like those in ``quantiles``) parallel
over grid points for data in memory and serial for dask blocks.
"""

import numpy as np
import scipy.special
import xarray as xr
from numba import njit, prange

from .config import WINDOW
from .quantiles import numpy_lerp
from .datatree import skip_empty
from .stats import nan_quantile


# ---------------------------------------------------------------------------
# p-values and bounds from a sample distribution
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


def _window_qrange(data, selected, starts, length, trials, first, q_low, q_high, out):
    """``out[s, t]``: the Q-range of trial t's selected members over its window of years.

    Trials are grouped by window, so each point's window is sorted once for all
    the trials that use it: trials ``trials[first[w]:first[w + 1]]`` use the
    years ``starts[w]:starts[w] + length``.

    Args:
        data (np.ndarray): (point, member, year).
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
                out[s, t] = _group_qrange(values, members, n_valid, selected[t], n_group, q_low, q_high)


_window_qrange_serial = njit(nogil=True)(_window_qrange)
_window_qrange_parallel = njit(parallel=True)(_window_qrange)


def _window_block(data, selected, starts, length, trials, first, q_low, q_high, parallel):
    points = np.ascontiguousarray(data.reshape(-1, *data.shape[-2:]))
    out = np.empty((points.shape[0], selected.shape[0]), dtype=np.float32)
    kernel = _window_qrange_parallel if parallel else _window_qrange_serial
    kernel(points, selected, starts, length, trials, first, q_low, q_high, out)
    return out.reshape(*data.shape[:-2], selected.shape[0])


#(t): Members in every hist-nat bootstrap draw: the smallest LESFMIP ensemble, so one null serves every experiment
N_BOOTSTRAP_MEMBERS = 10


def bootstrap_draws(n_hist_nat_members, n_hist_nat_years, n_members=N_BOOTSTRAP_MEMBERS, years=WINDOW,
                    n_trials=10_000, seed=0):
    """Which hist-nat members and which window of years each bootstrap trial uses.

    Each trial takes ``n_members`` different members and one ``years``-long
    window. Every complete window is used about equally often: the windows
    are cycled through in a random order, and the remainder drawn at random.

    Args:
        n_hist_nat_members, n_hist_nat_years (int): Size of the hist-nat ensemble.
        n_members (int): Members per trial.
        years (int): Window length (odd).
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``selected``, (trial, member) True for each
        trial's members, and ``window_starts``, (trial,) the first year index of each trial's window.
    """
    if n_members > n_hist_nat_members:
        raise ValueError("n_members cannot exceed the hist-nat members available.")
    if years % 2 == 0:
        raise ValueError("years must be odd.")

    rng = np.random.default_rng(seed)
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
        rng.random((n_trials, n_hist_nat_members)), n_members - 1, axis=1,
    )[:, :n_members]
    selected = np.zeros((n_trials, n_hist_nat_members), dtype=np.bool_)
    np.put_along_axis(selected, sampled_member_indices, True, axis=1)

    return selected, sampled_centre_indices - half_window


@skip_empty
def bootstrap_qrange(hist_nat, n_members=N_BOOTSTRAP_MEMBERS, years=WINDOW, quantiles=(0.05, 0.95), n_trials=1000,
                     seed=0):
    """The Q-range (e.g. Q95 - Q05) of each hist-nat bootstrap trial.

    Trial k pools the values of ``n_members`` different hist-nat members over
    one ``years``-year window, the draws of ``bootstrap_draws(..., seed=seed)``:
    the members ``selected[k]`` and the years from ``window_starts[k]``. NaN
    are skipped. Every grid point uses the same draws.

    Args:
        hist_nat (xr.DataArray | xr.Dataset): hist-nat's whole record, on (member, year, ...).
        n_members (int): Members per trial.
        years (int): Window length (odd).
        quantiles (tuple[float, float]): The lower and upper quantile.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` and ``year`` replaced by ``trial`` (first, numbered from 0).
    """
    selected, window_starts = bootstrap_draws(hist_nat.sizes["member"], hist_nat.sizes["year"], n_members, years,
                                              n_trials, seed)

    #(c): Trials grouped by window, so each window is sorted once
    trials = np.argsort(window_starts, kind="stable")
    starts, counts = np.unique(window_starts, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)])

    if hist_nat.chunks:
        hist_nat = hist_nat.chunk({"member": -1, "year": -1})
    out = xr.apply_ufunc(
        _window_block,
        hist_nat,
        input_core_dims=[["member", "year"]],
        output_core_dims=[["trial"]],
        kwargs=dict(selected=selected, starts=starts, length=years, trials=trials, first=first,
                    q_low=quantiles[0], q_high=quantiles[1], parallel=not hist_nat.chunks),
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"trial": n_trials}},
    )
    return out.transpose("trial", ...).assign_coords(trial=np.arange(n_trials))


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


def robust_sign(change, significant, threshold=0.66, dim="model"):
    """Where the models agree: +1 where at least ``threshold`` of them have a significant increase, -1 a decrease.

    0 elsewhere, and NaN where no model has data. The fraction is of the
    models with data at each point, so with six models and the default
    threshold four must agree; with two, both. Both signs cannot pass a
    threshold above one half.

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


def area_mean(da, lat="lat", lon="lon"):
    """Cos(latitude)-weighted mean over ``lat`` and ``lon`` of a DataArray or Dataset; booleans give the area fraction."""
    if isinstance(da, xr.DataArray) and da.dtype == bool:
        da = da.astype(float)
    return da.weighted(np.cos(np.deg2rad(da[lat]))).mean((lat, lon))


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------

@skip_empty
def ttest_welch(da1, da2, dims=("year", "member"), variable="tas"):
    """Welch's t-test across ``dims``, returning ``t`` and the two-sided ``p``.

    Takes two DataArrays, or two Datasets (as ``xr.map_over_datasets`` gives them) whose ``variable`` is tested.
    """
    if isinstance(da1, xr.Dataset):
        da1, da2 = da1[variable], da2[variable]

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

    return xr.Dataset({"t": t, "p": p})
