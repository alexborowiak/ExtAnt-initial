import re
from functools import partial
from pathlib import Path

import xarray as xr

OPEN_KWARGS_LESFMIP_DERIVED = dict(
    compat='override',
    coords='minimal',
    join='override',
    parallel=True,
    chunks={'time': -1},
)

MEMBER_PATTERN = r'_(r\d+i\d+p\d+f\d+)_'

def preprocess_sam(ds, pattern=MEMBER_PATTERN):
    """Select the surface SAM index and attach the ensemble member ID from the file's name.

    Args:
        ds (xr.Dataset): Dataset as opened from a single file.
        pattern (str): Regex with one capture group matching the member ID in the file name.

    Returns:
        xr.DataArray: Surface SAM on a ``time`` dimension, with a scalar ``member`` coordinate.
    """
    name = Path(ds.encoding['source']).name
    member = re.search(pattern, name).group(1)
    return ds.SAM.isel(level=0).rename(time_mm='time').assign_coords(member=member)


def tag_member(ds, variable, pattern=MEMBER_PATTERN):
    """Extract the ensemble member ID from a file's name and attach it as a coordinate.

    Args:
        ds (xr.Dataset): Dataset as opened from a single file.
        variable (str): Name of the variable to select.
        pattern (str): Regex with one capture group matching the member ID in the file name.

    Returns:
        xr.DataArray: The selected variable, squeezed, with a scalar ``member`` coordinate.
    """
    name = Path(ds.encoding['source']).name
    member = re.search(pattern, name).group(1)
    return ds.squeeze().assign_coords(member=member)[variable]


def open_members(paths, preprocess=None, **kwargs):
    """Open per-file datasets and concatenate them along a ``member`` dimension.

    Args:
        paths (Sequence[str | Path]): File paths, one per ensemble member.
        preprocess (Callable | None): Applied to each dataset before concatenation.
        **kwargs: Passed to ``xr.open_mfdataset``.

    Returns:
        xr.Dataset | xr.DataArray: Concatenated along ``member``.
    """
    return xr.open_mfdataset(
        paths,
        preprocess=preprocess,
        combine='nested',
        concat_dim='member',
        **kwargs,
    )