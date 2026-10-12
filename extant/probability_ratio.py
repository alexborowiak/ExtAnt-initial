"""Probability ratios: how much more (or less) often each forcing gives an extreme warm or cold season.

A season is extreme when it is beyond hist-nat's 1-in-N level over the same
final years: warmer than hist-nat's 1 - 1/N quantile (a warm season) or colder
than its 1/N quantile (a cold one), hist-nat's members and years pooled. The
probability ratio is

    PR = P_exp(beyond the level) / P_nat(beyond the level),

1 when the forcing changes nothing, 2 when it doubles the chance, 1/2 when it
halves it. Each probability is the share of an ensemble's member-years beyond
the level, averaged member by member (``member_exceedance``).

Two versions of the experiment's chance separate the shift of the mean from
the rest of the change:

    mean only        hist-nat's own values moved by the change in the mean (experiment minus
                     hist-nat): the chance the experiment would have if its distribution had only
                     shifted. Moving the values up by the change is the same as moving the
                     threshold down by it, which is how the notebook does it.
    mean and width   the experiment's own values: everything that changed.

The gap between the two is what the change in width (and shape) adds; it
matters most for the cold tail, which a narrowing distribution pulls in.

Uncertainty comes from a bootstrap of whole members, with replacement, within
each ensemble (``bootstrap_member_mean``). The thresholds stay fixed: each
trial asks how often the resampled members go beyond the same levels, so the
spread is how well the two ensembles pin down the chance of one defined event.

Sections
--------
1. The levels and the thresholds
2. How often each member goes beyond them
3. The bootstrap
"""

import numpy as np
import xarray as xr

from .datatree import skip_empty
from .stats import nan_quantile

# ---------------------------------------------------------------------------
# 1. The levels and the thresholds
# ---------------------------------------------------------------------------

#(t): Which side of its threshold a season has to fall: above it for a warm season (+1), below it for a cold one (-1)
TAIL_SIGNS = xr.DataArray([1, -1], dims="tail", coords={"tail": ["warm", "cold"]})

#(t): Return periods in years: 1-in-20 seasons for the paper, 1-in-10 to check the sensitivity. Rarer levels are
#(c): avoided on purpose: with hist-nat's chance that small, the ratio blows up on a handful of seasons
RETURN_PERIODS = (20, 10)


def threshold_levels(return_periods=RETURN_PERIODS):
    """The quantile each threshold sits at: 1 - 1/N for a warm season, 1/N for a cold one.

    Args:
        return_periods (Sequence[int]): N, in years.

    Returns:
        xr.DataArray: Quantile levels on (tail, return_period), e.g. 0.95 (warm) and 0.05 (cold) for N = 20.
    """
    periods = xr.DataArray(list(return_periods), dims="return_period",
                           coords={"return_period": list(return_periods)})
    #(c): The warm sign (+1) gives 0.5 + (0.5 - 1/N) = 1 - 1/N, the cold sign (-1) gives 1/N
    levels = 0.5 + TAIL_SIGNS * (0.5 - 1 / periods)
    return np.round(levels, 10).rename("level")


@skip_empty
def thresholds(counterfactual, levels, dims=("member", "year")):
    """hist-nat's value at each level, its members and years pooled.

    Args:
        counterfactual (xr.Dataset | xr.DataArray): hist-nat over the years compared, on (member, year, ...).
        levels (xr.DataArray): ``threshold_levels`` output, on (tail, return_period).
        dims (Sequence[str]): The dims pooled (those present).

    Returns:
        The same type, on (tail, return_period, ...).
    """
    pooled = [dim for dim in dims if dim in counterfactual.dims]
    unique_levels = list(np.unique(levels.values))
    if isinstance(counterfactual, xr.Dataset):
        quantiles = counterfactual.map(nan_quantile, q=unique_levels, dim=pooled)
    else:
        quantiles = nan_quantile(counterfactual, unique_levels, pooled)
    #(c): Picking the levels out with a DataArray swaps the quantile dim for (tail, return_period)
    return quantiles.sel(quantile=levels).drop_vars("quantile")


# ---------------------------------------------------------------------------
# 2. How often each member goes beyond them
# ---------------------------------------------------------------------------

@skip_empty
def member_exceedance(values, threshold, dim="year"):
    """The share of each member's years beyond the threshold: above it for the warm tail, below it for the cold.

    Args:
        values (xr.Dataset | xr.DataArray): Seasonal means on (member, year, ...).
        threshold (the same type): On (tail, return_period, ...), from ``thresholds`` (or moved by a change).
        dim (str): The years.

    Returns:
        The same type, on (member, tail, return_period, ...): 0 to 1, NaN where a member or the threshold
        has no data.
    """
    beyond = (TAIL_SIGNS * (values - threshold)) > 0
    #(c): A comparison with NaN is False, which would count as "not beyond"; keep missing data missing instead
    return beyond.where(values.notnull() & threshold.notnull()).mean(dim)


# ---------------------------------------------------------------------------
# 3. The bootstrap
# ---------------------------------------------------------------------------

def resample_weights(n_members, n_trials, rng):
    """How often each member is drawn in each trial of a bootstrap with replacement, as a share of the draws.

    Trial k draws ``n_members`` members at random with replacement; row k
    holds how many times each member was drawn, over ``n_members``. A row
    times the members' values is then the mean of the members drawn, so every
    trial is one matrix product.

    Returns:
        np.ndarray: (n_trials, n_members); every row sums to 1.
    """
    draws = rng.integers(0, n_members, size=(n_trials, n_members))
    counts = (draws[:, :, None] == np.arange(n_members)).sum(axis=1)
    return counts / n_members


@skip_empty
def bootstrap_member_mean(members, n_trials=1000, seed=0):
    """The ensemble mean of members drawn with replacement, ``n_trials`` times.

    Each trial draws as many members as there are, with replacement, and
    averages them (``resample_weights``). Calls with the same seed and the same
    number of members draw the same members, so two things made from one
    ensemble (hist-nat's own chance and its chance moved by the change in the
    mean) are resampled together, and their ratio keeps its pairing.

    Args:
        members (xr.Dataset | xr.DataArray): On (member, ...), e.g. ``member_exceedance`` output.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` replaced by ``trial`` (first, numbered from 0).
    """
    weights = resample_weights(members.sizes["member"], n_trials, np.random.default_rng(seed))
    weights_da = xr.DataArray(weights, dims=("trial", "member"), coords={"trial": np.arange(n_trials)})
    if isinstance(members, xr.Dataset):
        return members.map(lambda da: xr.dot(weights_da, da, dim="member"))
    return xr.dot(weights_da, members, dim="member")
