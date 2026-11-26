"""Is the observed (ERA5) change detectable, and which forcings is it consistent with?

The ERA5 evaluation asks whether ERA5 could be a member of the historical
ensemble. Attribution asks more: could ERA5 have happened *without* human
influence (hist-nat)?

Sections
--------
1. Detection classes    ERA5 outside hist-nat's range (detected) and inside historical's (consistent)
2. Record rates         how many record highs and lows ERA5 sets, against chance and the members

ERA5's trend among each experiment's members (detection) is worked out in notebook calc_03c, with
``obs_evaluation.locate`` for where ERA5 falls among the members.
"""

import numpy as np
import xarray as xr

from . import obs_evaluation as ev
from .statistics import records


# ---------------------------------------------------------------------------
# 1. Detection classes
# ---------------------------------------------------------------------------

#(t): Detection classes: is ERA5 outside hist-nat's range (detected) and inside historical's (consistent)?
DETECTION_CLASSES = {
    0: "not detected, consistent",      # ERA5 could be natural, and historical agrees
    1: "detected, consistent",          # human influence detected, and historical gets it right
    2: "detected, inconsistent",        # detected, but historical's change is too large or too small
    3: "not detected, inconsistent",    # looks natural, but historical has a different change
}


def detection_class(historical, hist_nat):
    """Classify each point from ERA5's tests against historical and against hist-nat.

    Args:
        historical, hist_nat (xr.Dataset): ``ev.locate`` outputs, e.g. of the trend at every grid point.

    Returns:
        xr.DataArray: Codes of ``DETECTION_CLASSES``; NaN where either test has no
        data, or too few members to flag anything (so "not detected" would mean nothing).
    """
    detected = hist_nat["verdict"] != 0
    consistent = historical["verdict"] == 0
    classes = xr.where(detected, xr.where(consistent, 1, 2), xr.where(consistent, 0, 3))
    usable = historical["verdict"].notnull() & hist_nat["verdict"].notnull()
    for test in (historical, hist_nat):
        if "testable" in test:
            usable = usable & (test["testable"] == 1)
    return classes.where(usable)


# ---------------------------------------------------------------------------
# 2. Record rates
# ---------------------------------------------------------------------------

def record_curves(ensembles, obs, pool_seasons=True):
    """Cumulative record highs and lows through the record, for ERA5 and each experiment's members.

    With no change at all, year k is a record with probability 1/k, so the
    expected count after n years is 1 + 1/2 + ... + 1/n. Warming makes record
    highs more frequent and record lows rarer. Pooling the four seasons adds
    their counts (4x the expected count), which makes the comparison far less
    noisy than one season's ~4 records.

    Args:
        ensembles (dict[str, xr.DataArray]): Experiment -> members on (member, year, season), over the years compared.
        obs (xr.DataArray): ERA5 on (year, season), over the same years.
        pool_seasons (bool): Sum the counts over seasons.

    Returns:
        xr.Dataset: ``obs`` (kind, year), ``members`` (experiment, member, kind, year),
        and ``expected`` (year), with ``kind`` = ["highs", "lows"].
    """
    def curves(da):
        both = xr.concat([records.cumulative_records(da, high=True), records.cumulative_records(da, high=False)],
                         dim=xr.Variable("kind", ["highs", "lows"]))
        return both.sum("season") if pool_seasons else both

    members, observed = {}, None
    for experiment, ensemble in ensembles.items():
        ensemble, aligned_obs = ev.align(ensemble, obs)
        members[experiment] = curves(ensemble).assign_coords(member=np.arange(ensemble.sizes["member"]))
        observed = curves(aligned_obs) if observed is None else observed
    stacked = xr.concat(list(members.values()), dim=xr.Variable("experiment", list(members)), join="outer")

    #(c): Expected cumulative count for a stationary series, from each season's valid years
    valid = aligned_obs.notnull()
    rank = valid.cumsum("year").where(valid)
    expected = (1 / rank).fillna(0).cumsum("year")
    expected = expected.sum("season") if pool_seasons else expected
    return xr.Dataset({"obs": observed, "members": stacked, "expected": expected})
