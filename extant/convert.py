"""Raw files to monthly zarr stores: LESFMIP (one netCDF file per member) and ERA5 (hourly files).

Used by notebook 01 on JASMIN, once per variable. Only the pieces are here
(finding the files, opening one model's members together, a month of ERA5,
joining the ERA5 years); the loops, what is left to do and the checks are in
the notebook. Everything is written under ``paths.SCRATCH``; move it to
``paths.DATA_DIR`` afterwards (see ``paths``). Runs on the Dask cluster, so
send the package to the workers first (``jasmin.upload_package(client)``).
"""

import logging
import re
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, paths, storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LESFMIP
# ---------------------------------------------------------------------------

#(t): How the member files are opened and stacked along a new ``member`` dim
OPEN_KWARGS = dict(
    compat="override",
    coords="minimal",
    join="override",
    parallel=True,
    chunks={"time": -1},
    combine="nested",
    concat_dim="member",
)

#(t): Member labels in the different modelling centres' file names
MEMBER_PATTERNS = (
    re.compile(r"(?:^|[_/])(r\d+i\d+p\d+(?:f\d+)?)(?=[_./]|$)"),  # CMIP DRS: r1i1p1f1
    re.compile(r"(?:^|[_/])(\d{4}\.\d{3})(?=[_.]|$)"),            # CESM2 LENS2: 1001.001
    re.compile(r"(?:^|[_/])(\d{2,3})(?=[_.]|$)"),                 # SFLE: 009
)

#(t): Folders of raw files (``paths.lesfmip_raw``) that are not experiments
NOT_EXPERIMENTS = ("nhsum", "reanalysis")


def member_from_path(source):
    """The member label in a file's name, or failing that in the folders above it."""
    path = Path(source)
    for pattern in MEMBER_PATTERNS:
        for part in (path.name, *reversed(path.parts[:-1])):
            match = pattern.search(part)
            if match:
                return match.group(1)
    raise ValueError(f"no member label found in {source}")


def preprocess_member(ds, variable):
    """One member's file: ``variable`` south of ``config.LAT_MAX``, with its member label as a coordinate."""
    source = ds.encoding.get("source")
    if source is None:
        raise ValueError("dataset has no source encoding, so its member cannot be found")
    ds = ds.squeeze().assign_coords(member=member_from_path(source))[[variable]]
    if "lon_2" in ds.coords:
        ds = ds.rename({"lon_2": "lon"})
    return ds.sel(lat=slice(-90, config.LAT_MAX))


def availability(variable, experiments=None):
    """Which models have raw files for which experiment: a True/False table, the best-covered models first.

    Args:
        variable (str): Key of ``config.VARIABLES``.
        experiments (Sequence[str] | None): Experiments to check; every folder if None.

    Returns:
        pd.DataFrame: models x experiments.
    """
    raw = paths.lesfmip_raw(variable)
    if experiments is None:
        experiments = sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name not in NOT_EXPERIMENTS)
    models = {experiment: [p.name for p in (raw / experiment).iterdir() if p.is_dir()] for experiment in experiments}
    table = (pd.Series(models).explode().rename_axis("experiment").reset_index(name="model")
             .pipe(lambda d: pd.crosstab(d["model"], d["experiment"])).astype(bool))
    return table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]


def raw_files(variable, experiment, model):
    """One model's raw files for one experiment (the ``config.GROUP`` ones), one per member, sorted."""
    return sorted((paths.lesfmip_raw(variable) / experiment / model).rglob(f"*_{config.GROUP}.nc"))


def open_members(files, variable, open_kwargs=None):
    """Open one model's member files for one experiment, stacked along ``member`` and chunked for saving (lazily).

    Args:
        files (Sequence[Path]): One file per member.
        variable (str): Variable name.
        open_kwargs (dict | None): Passed to ``xr.open_mfdataset``; ``OPEN_KWARGS`` if None.

    Returns:
        xr.Dataset: ``variable`` on (member, time, lat, lon), south of ``config.LAT_MAX``.
    """
    ds = xr.open_mfdataset(files, preprocess=partial(preprocess_member, variable=variable),
                           **(open_kwargs or OPEN_KWARGS))
    return ds.chunk({"time": -1, "lat": max(ds.sizes["lat"] // 2, 1), "lon": -1, "member": 5})


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------

#(t): ERA5 is cut a little north of the LESFMIP edge, so the conservative regridding covers the edge cells
ERA5_LAT_MAX = -38.0


def era5_south(ds):
    """ERA5 sorted south to north and cut at ``ERA5_LAT_MAX``."""
    return ds.sortby("latitude").sel(latitude=slice(-90.0, ERA5_LAT_MAX))


def era5_files(variable, year, month=None):
    """The hourly ERA5 files of one year (or month), sorted."""
    code = config.VARIABLES[variable].era5_code
    if code is None:
        raise ValueError(f"no ERA5 recipe for {variable} yet: see config.VARIABLES")
    months = f"{month:02d}" if month else "*"
    return sorted(Path(paths.ERA5_RAW).glob(f"{year}/{months}/*/*.{code}.nc"))


def era5_month_mean(files, name):
    """Mean of hourly ERA5 files (one month), south of 38°S, as a float32 array (lat, lon).

    Runs on a worker, one month each; sums in float64 so ~700 hours lose no precision.
    """
    total, n = None, 0
    for file in files:
        with xr.open_dataset(file) as ds:
            values = era5_south(ds)[name].values
        subtotal = values.sum(axis=0, dtype="float64")
        total = subtotal if total is None else total + subtotal
        n += values.shape[0]
    return (total / n).astype("float32")


def combine_era5_years(variable):
    """Join the year stores (and any existing combined store) into ``paths.era5_monthly``, check it, delete them."""
    years_dir = paths.era5_years_dir(variable)
    year_stores = paths.find_all(years_dir, "*.zarr")
    parts = [xr.open_zarr(p)[variable] for p in year_stores]
    if storage.exists(paths.era5_monthly(variable)):
        parts.insert(0, storage.open_dataarray(paths.era5_monthly(variable)))
    monthly = xr.concat(parts, dim="time").sortby("time").load()

    #(c): Reopened from where it was just written: paths.find would prefer an older copy in DATA_DIR
    store = storage.save(monthly.rename(variable), paths.era5_monthly(variable))
    saved = xr.open_zarr(store)[variable].load()
    assert saved.indexes["time"].is_unique, "a month appears twice"
    assert saved.indexes["time"].is_monotonic_increasing, "months out of order"
    assert np.allclose(saved.values, monthly.values, equal_nan=True), "the saved store differs from the data"

    for year_store in year_stores:
        shutil.rmtree(year_store)
    logger.info(f"ERA5 {variable}: {dict(saved.sizes)} saved, {len(year_stores)} year stores removed")
    return saved
