"""The Southern Annular Mode (SAM): which phase each season is in, so every result can be split by it.

The index is not computed here. LEADER-EPESC working group 1 made a monthly
SAM index for every LESFMIP member (``paths.SAM_RAW``, the files
archive/draft_current_DAILY.ipynb opened): the zonal-mean index of Gong & Wang
(1999), the difference between standardised zonal means at 40°S and 65°S, one
file per member with ``SAM`` on (time_mm, level). Notebook 01 (section 6) turns
it into seasonal means; notebook 03 (*SAM Phases*) cuts it into phases:

    negative   the index below hist-nat's lower tercile
    neutral    between hist-nat's two terciles
    positive   above its upper tercile

hist-nat's terciles over the final years, per model and season, members and
years pooled, and the same for every experiment of the model: hist-nat has a
third of its seasons in each phase, and an experiment whose SAM has shifted
has more in one. Each section of notebook 03 then has a *By SAM phase*
subsection that redoes its result with the seasons of one phase alone.

Sections
--------
1. Opening the index files
2. The phases
3. A synthetic index, so the steps run on the sample data
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
# 2. The phases
# ---------------------------------------------------------------------------

#(t): The three phases, from the lowest index to the highest
PHASES = ("negative", "neutral", "positive")

#(t): Where hist-nat's index is cut into the phases: its terciles, so a third of its seasons fall in each
TERCILES = (1 / 3, 2 / 3)


@skip_empty
def phase_masks(index, thresholds):
    """Which phase each season is in: True in one of the three along ``phase``, by where its index falls.

    negative: below the lower threshold; neutral: between the two, both
    included; positive: above the upper one. A season with no index is in none.

    Args:
        index (xr.Dataset): ``sam`` on (member, year, season).
        thresholds (xr.Dataset): ``sam`` on (quantile, season): the lower and upper boundary, hist-nat's
            ``TERCILES``.

    Returns:
        xr.Dataset: ``in_phase`` on (phase, member, year, season), boolean.
    """
    sam_index = index["sam"]
    lower = thresholds["sam"].isel(quantile=0, drop=True)
    upper = thresholds["sam"].isel(quantile=1, drop=True)
    masks = [sam_index < lower, (sam_index >= lower) & (sam_index <= upper), sam_index > upper]
    return xr.Dataset({"in_phase": xr.concat(masks, dim=xr.Variable("phase", list(PHASES)))})


@skip_empty
def by_phase(values, masks):
    """``values`` in the seasons of each phase alone, along a new ``phase`` dim; NaN in every other season.

    Only the members and years both have are kept, so the final years of
    ``values`` take the masks of those years.

    Args:
        values (xr.Dataset | xr.DataArray): On (member, year, season, ...).
        masks (xr.Dataset): ``phase_masks`` output.

    Returns:
        The type of ``values``, with ``phase`` added.
    """
    return values.where(masks["in_phase"])


# ---------------------------------------------------------------------------
# 3. A synthetic index for the sample data
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
