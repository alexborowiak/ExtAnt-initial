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
    combine='nested',
    concat_dim='member',
)

MEMBER_PATTERN = r'_(r\d+i\d+p\d+f\d+)_'

#(t): The small test sample (a 3x3 block of grid cells, every model and experiment): see open_lesfmip_sample
SAMPLE_PATH = Path(__file__).resolve().parent / 'tests' / 'data'

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
        **kwargs,
    )


def open_lesfmip_sample(path=SAMPLE_PATH, models=None, experiments=None):
    """Open the sample LESFMIP files as a /<model>/<experiment> DataTree, like the notebook's ``build_tree``.

    The files are ``<model>_<experiment>_sample.nc``: monthly ``tas`` in K on
    (member, time, lat, lon), each model on its own calendar. None means 'all'.

    Args:
        path (str | Path): Directory holding the sample files.
        models, experiments (Sequence[str] | None): Subsets to open.

    Returns:
        xr.DataTree: One node per model and experiment, loaded into memory.
    """
    nodes = {}
    for file in sorted(Path(path).glob('*_sample.nc')):
        if file.name.startswith('era5'):
            continue
        model, experiment = file.stem.removesuffix('_sample').rsplit('_', 1)
        if (models is None or model in models) and (experiments is None or experiment in experiments):
            nodes[f'{model}/{experiment}'] = xr.open_dataset(file).load()
    if not nodes:
        raise ValueError(f'no sample files matched {models=} {experiments=} in {path}')
    return xr.DataTree.from_dict(nodes)


def open_era5_sample(path=SAMPLE_PATH):
    """Open the sample ERA5 file as the notebook's ERA5 section leaves ``era5_ds``: monthly ``tas`` (°C) from 1979.

    The file was saved after the notebook had converted ERA5 to a 360-day
    calendar with ``align_on='year'``, which shifted its dates (1 March 1979
    reads 29 February, 1 December 30 November). They are relabelled with month
    starts on the standard calendar.

    Returns:
        xr.DataArray: (time, lat, lon), loaded into memory.
    """
    tas = xr.open_dataset(Path(path) / 'era5_sample.nc').tas.load()
    #(c): One value per month from the first
    start = f'{int(tas.time.dt.year[0]):04d}-{int(tas.time.dt.month[0]):02d}-01'
    return tas.assign_coords(time=xr.date_range(start, periods=tas.sizes['time'], freq='MS'))
