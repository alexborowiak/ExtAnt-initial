"""Probability ratios: how much more (or less) often each forcing gives an extreme warm or cold season.

A season is extreme when it is beyond one of hist-nat's percentiles over the
same final years: warmer than its 95th percentile (a warm season) or colder
than its 5th (a cold one), hist-nat's members and years pooled. hist-nat has a
5% chance of either; ``CHANCES`` sets that chance. The probability ratio is

    PR = P_exp(beyond the percentile) / P_nat(beyond the percentile),

1 when the forcing changes nothing, 2 when it doubles the chance, 1/2 when it
halves it. Each probability is the share of an ensemble's member-years beyond
the percentile, averaged member by member (``member_exceedance``).

The shift-only ratio (notebook 03, section 11) asks how much of the ratio the
shift of the mean alone gives: hist-nat's own seasons, every one moved by the
experiment's change in the mean, against the same percentiles. Moving the
seasons up by the change is the same as moving the percentile down by it,
which is how the notebook does it. The gap between the shift-only ratio and
the ratio is what the change in width (and shape) adds; it matters most for
the cold tail, which a narrowing distribution pulls in.

Uncertainty comes from a bootstrap of whole members, with replacement, within
each ensemble (``bootstrap_member_mean``). The thresholds stay fixed: each
trial asks how often the resampled members go beyond the same levels, so the
spread is how well the two ensembles pin down the chance of one defined event.

By SAM phase (notebook 03, sections 10 and 11), the events and thresholds are
the same, and the chance is of a season beyond them among the seasons of one
phase. Members then have different numbers of seasons, so the chance is pooled:
every member's count of seasons beyond the threshold (``member_beyond``) over
their count of seasons. Bootstrapping both counts with the same seed draws the
same members for both, so each trial's chance is pooled too.

Sections
--------
1. The levels and the thresholds
2. How often each member goes beyond them
3. The bootstrap
"""

import numpy as np
import xarray as xr

from .datatree import skip_empty
from .stats import nan_quantile, weighted_member_sums

# ---------------------------------------------------------------------------
# 1. The levels and the thresholds
# ---------------------------------------------------------------------------

#(t): Which side of its threshold a season has to fall: above it for a warm season (+1), below it for a cold one (-1)
TAIL_SIGNS = xr.DataArray([1, -1], dims="tail", coords={"tail": ["warm", "cold"]})

#(t): hist-nat's chance of each event: 5% (warmer than its 95th percentile, colder than its 5th) for the paper, 10%
#(t): (its 90th and 10th) to check the sensitivity
#(c): Rarer events are avoided on purpose: with hist-nat's chance that small, the ratio blows up on a handful of seasons
CHANCES = (0.05, 0.10)


def threshold_levels(chances=CHANCES):
    """The quantile each threshold sits at: 1 - chance for a warm season, the chance for a cold one.

    Args:
        chances (Sequence[float]): hist-nat's chance of each event, e.g. 0.05.

    Returns:
        xr.DataArray: Quantile levels on (tail, chance), e.g. 0.95 (warm) and 0.05 (cold) for a chance of 0.05.
    """
    chances = xr.DataArray(list(chances), dims="chance", coords={"chance": list(chances)})
    #(c): The warm sign (+1) gives 0.5 + (0.5 - chance) = 1 - chance, the cold sign (-1) gives the chance
    levels = 0.5 + TAIL_SIGNS * (0.5 - chances)
    return np.round(levels, 10).rename("level")


@skip_empty
def thresholds(counterfactual, levels, dims=("member", "year")):
    """hist-nat's value at each level, its members and years pooled.

    Args:
        counterfactual (xr.Dataset | xr.DataArray): hist-nat over the years compared, on (member, year, ...).
        levels (xr.DataArray): ``threshold_levels`` output, on (tail, chance).
        dims (Sequence[str]): The dims pooled (those present).

    Returns:
        The same type, on (tail, chance, ...).
    """
    pooled = [dim for dim in dims if dim in counterfactual.dims]
    unique_levels = list(np.unique(levels.values))
    if isinstance(counterfactual, xr.Dataset):
        quantiles = counterfactual.map(nan_quantile, q=unique_levels, dim=pooled)
    else:
        quantiles = nan_quantile(counterfactual, unique_levels, pooled)
    #(c): Picking the levels out with a DataArray swaps the quantile dim for (tail, chance)
    return quantiles.sel(quantile=levels).drop_vars("quantile")


# ---------------------------------------------------------------------------
# 2. How often each member goes beyond them
# ---------------------------------------------------------------------------

@skip_empty
def member_exceedance(values, threshold, dim="year"):
    """The share of each member's years beyond the threshold: above it for the warm tail, below it for the cold.

    Args:
        values (xr.Dataset | xr.DataArray): Seasonal means on (member, year, ...).
        threshold (the same type): On (tail, chance, ...), from ``thresholds`` (or moved by a change).
        dim (str): The years.

    Returns:
        The same type, on (member, tail, chance, ...): 0 to 1, NaN where a member or the threshold
        has no data.
    """
    beyond = (TAIL_SIGNS * (values - threshold)) > 0
    #(c): A comparison with NaN is False, which would count as "not beyond"; keep missing data missing instead
    return beyond.where(values.notnull() & threshold.notnull()).mean(dim)


@skip_empty
def member_beyond(values, threshold, dim="year"):
    """How many of each member's years are beyond the threshold: above it for the warm tail, below it for the cold.

    For the seasons of one SAM phase, whose number differs from member to
    member: the chance is then ``member_beyond(...).sum('member')`` over
    ``values.count(['member', 'year'])``, every season counted once.

    Args:
        values (xr.Dataset | xr.DataArray): Seasonal means on (member, year, ...), NaN in the seasons left out.
        threshold (the same type): On (tail, chance, ...), from ``thresholds`` (or moved by a change).
        dim (str): The years.

    Returns:
        The same type, on (member, tail, chance, ...): a count, NaN where the threshold has no data.
    """
    beyond = (TAIL_SIGNS * (values - threshold)) > 0
    #(c): A season left out (NaN) compares False, so it is not counted; a missing threshold stays missing
    return beyond.sum(dim).where(threshold.notnull())


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
    ensemble (hist-nat's own chance and its shift-only chance, against the
    moved thresholds) are resampled together, and their ratio keeps its pairing.

    Args:
        members (xr.Dataset | xr.DataArray): On (member, ...), e.g. ``member_exceedance`` output.
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        The same type, with ``member`` replaced by ``trial`` (first, numbered from 0).
    """
    weights = resample_weights(members.sizes["member"], n_trials, np.random.default_rng(seed))
    return weighted_member_sums(members, weights)
