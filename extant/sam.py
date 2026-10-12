"""The Southern Annular Mode (SAM): its forced change, and how much of each forcing's temperature change goes with it.

The index is not computed here. LEADER-EPESC working group 1 made a monthly
SAM index for every LESFMIP member (``paths.SAM_RAW``, the files
archive/draft_current_DAILY.ipynb opened): the zonal-mean index of Gong & Wang
(1999), the difference between standardised zonal means at 40°S and 65°S, one
file per member with ``SAM`` on (time_mm, level). Notebook 01 (section 6) turns
it into seasonal means; notebook 03 (section 11) does the rest:

    the forced change     each experiment's final-years ensemble-mean SAM minus hist-nat's (as for
                          temperature, notebook 03 section 4)
    the regression β(x)   temperature on the SAM, at every grid point, from internal variability alone:
                          both as each member minus its own ensemble's mean, so the forced changes of
                          both are gone and only how they vary together is left
    the split             SAM-congruent change β(x) x ΔSAM, and the residual Δmean - β(x) x ΔSAM
    by phase              the width of temperature in seasons with a positive and with a negative SAM

The split assumes the forced change of the SAM has the same temperature pattern
as its internal variability. That is the standard assumption, but it is one.
Ozone, the SAM and sea ice are linked, so the split does not fully separate them.

Sections
--------
1. Opening the index files
2. The regression of temperature on the SAM
3. Temperature in each SAM phase
4. A synthetic index, so the steps run on the sample data
"""

import re
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr

from . import paths
from .datatree import skip_empty

# ---------------------------------------------------------------------------
# 1. Opening the index files
# ---------------------------------------------------------------------------

#(t): The member ID in a file name, e.g. SAM_zonal_mon_hist-nat_HadGEM3-GC31-LL_r10i1p1f3_interp.nc -> r10i1p1f3
MEMBER_PATTERN = r"_(r\d+i\d+p\d+f\d+)_"


def index_files(experiment, model, root=paths.SAM_RAW):
    """Every member's SAM file for one model and experiment, sorted by name (empty if there are none)."""
    return sorted((Path(root) / experiment / model).glob("*.nc"))


def member_of(path, pattern=MEMBER_PATTERN):
    """The member ID in a file's name."""
    return re.search(pattern, Path(path).name).group(1)


def _one_member(ds, level):
    """One file's index at one level, renamed ``sam`` on ``time``, with its member ID from the file name."""
    member = member_of(ds.encoding["source"])
    index = ds["SAM"].isel(level=level)
    level_value = index["level"].values.item() if "level" in index.coords else level
    return (index.drop_vars("level", errors="ignore").rename(time_mm="time").rename("sam")
            .assign_attrs(level=level_value).expand_dims(member=[member]).to_dataset())


def open_index(files, level=0):
    """The monthly SAM index of every file, members along ``member`` (lazily).

    archive/draft_current_DAILY.ipynb took ``level=3``, and a later version of
    its reader ``level=0``, called the surface; the notebook shows the levels
    in a file so the right one can be chosen.

    Args:
        files (Sequence[Path]): One file per member, from ``index_files``.
        level (int): Position along the files' ``level`` dim.

    Returns:
        xr.Dataset: ``sam`` on (member, time), the level kept as an attribute.
    """
    #(c): join='outer': members whose time axes differ are padded with NaN rather than silently overwritten
    return xr.open_mfdataset(files, preprocess=partial(_one_member, level=level), combine="nested",
                             concat_dim="member", join="outer", coords="minimal", compat="override", chunks={})


# ---------------------------------------------------------------------------
# 2. The regression of temperature on the SAM
# ---------------------------------------------------------------------------

@skip_empty
def regression_sums(temperature, index, variable="tas", dims=("member", "year")):
    """The two sums of a regression through the origin of temperature on the SAM index, over ``dims``.

    Give it anomalies with a mean of zero (each member minus its ensemble's
    mean): then the slope is Σ T'S' / Σ S'², and keeping the two sums apart
    lets several experiments' be added before dividing. Only the members and
    years both have are used, and only where both have data.

    Args:
        temperature (xr.Dataset): ``variable`` on (member, year, season, lat, lon).
        index (xr.Dataset): ``sam`` on (member, year, season).
        variable (str): The temperature variable.
        dims (Sequence[str]): Summed over.

    Returns:
        xr.Dataset: ``products`` Σ T'S' and ``squares`` Σ S'², on the remaining dims.
    """
    t, s = xr.align(temperature[variable], index["sam"], join="inner")
    valid = t.notnull() & s.notnull()
    return xr.Dataset({"products": (t * s).where(valid).sum(dims), "squares": (s ** 2).where(valid).sum(dims)})


# ---------------------------------------------------------------------------
# 3. Temperature in each SAM phase
# ---------------------------------------------------------------------------

#(t): The two phases: the index above zero, and below it
PHASES = ("positive", "negative")


@skip_empty
def by_phase(values, index):
    """``values`` in the seasons with a positive SAM, and in those with a negative one, along a new ``phase`` dim.

    Every other season is NaN in each, so pooled statistics of each phase use
    only its own seasons. Only the members and years both have are kept.

    Args:
        values (xr.Dataset | xr.DataArray): On (member, year, season, ...).
        index (xr.Dataset): ``sam`` on (member, year, season).

    Returns:
        The type of ``values``, with ``phase`` first.
    """
    values, sam = xr.align(values, index["sam"], join="inner")
    return xr.concat([values.where(sam > 0), values.where(sam < 0)], dim=xr.Variable("phase", list(PHASES)))


# ---------------------------------------------------------------------------
# 4. A synthetic index for the sample data
# ---------------------------------------------------------------------------

#(t): The synthetic index's trend from 1970 (per decade) in each experiment: a stand-in for the summer trend of the
#(t): ozone hole and a weaker greenhouse-gas one. Made up: only so the steps have something to find
SYNTHETIC_TRENDS = {"historical": 0.3, "hist-totalO3": 0.25, "hist-GHG": 0.1, "hist-aer": 0.0, "hist-nat": 0.0}

#(t): How much of that trend each season gets in the synthetic index (most in summer, as for the real SAM)
SYNTHETIC_SEASON_WEIGHTS = {"DJF": 1.0, "MAM": 0.5, "JJA": 0.2, "SON": 0.5}


def _synthetic_node(ds, experiment, rng, trends, start_year, variable):
    da = ds[variable]
    noise = rng.standard_normal((da.sizes["member"], da.sizes["year"], da.sizes["season"]))
    ramp = np.clip(da["year"].values - start_year, 0, None) / 10 * trends.get(experiment, 0.0)
    weight = np.array([SYNTHETIC_SEASON_WEIGHTS.get(season, 0.0) for season in da["season"].values])
    values = noise + ramp[None, :, None] * weight[None, None, :]
    coords = {dim: da[dim].values for dim in ("member", "year", "season")}
    return xr.Dataset({"sam": (("member", "year", "season"), values)}, coords=coords)


def synthetic_index(like, seed=0, trends=SYNTHETIC_TRENDS, start_year=1970, variable="tas"):
    """A made-up seasonal SAM index with the members, years and seasons of every node of ``like``.

    Not data. Each value is random noise, N(0, 1), plus a trend from
    ``start_year`` that depends on the experiment and the season. It only
    lets the SAM steps run, and be checked, on the sample data, which has no
    SAM index; nothing found with it means anything.

    Args:
        like (xr.DataTree): /<model>/<experiment> on (member, year, season, ...), e.g. the sample's seasonal means.
        seed (int): Random seed.
        trends (dict): Experiment -> trend per decade.
        start_year (int): First year of the trend.
        variable (str): A variable of ``like`` to copy the members, years and seasons from.

    Returns:
        xr.DataTree: The same nodes, each with ``sam`` on (member, year, season).
    """
    rng = np.random.default_rng(seed)
    return xr.DataTree.from_dict({
        node.relative_to(like): _synthetic_node(node.to_dataset(), node.name, rng, trends, start_year, variable)
        for node in like.leaves if node.data_vars
    })
