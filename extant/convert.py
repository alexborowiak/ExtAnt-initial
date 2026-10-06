"""Raw files to monthly zarr stores: LESFMIP (one netCDF file per member) and ERA5 (hourly files).

Run from notebook 01 on JASMIN, once per variable. Everything is written under
``paths.SCRATCH``; move it to ``paths.DATA_DIR`` afterwards (see ``paths``).
Runs on the Dask cluster, so send the package to the workers first
(``jasmin.upload_package(client)``).
"""

import logging
import re
import shutil
from functools import partial
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.utils import format_bytes

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

#(t): Folders in paths.LESFMIP_RAW that are not experiments
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
        variable (str): Variable name (a folder of ``paths.LESFMIP_RAW``).
        experiments (Sequence[str] | None): Experiments to check; every folder if None.

    Returns:
        pd.DataFrame: models x experiments.
    """
    raw = paths.LESFMIP_RAW / variable
    if experiments is None:
        experiments = sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name not in NOT_EXPERIMENTS)
    models = {experiment: [p.name for p in (raw / experiment).iterdir() if p.is_dir()] for experiment in experiments}
    table = (pd.Series(models).explode().rename_axis("experiment").reset_index(name="model")
             .pipe(lambda d: pd.crosstab(d["model"], d["experiment"])).astype(bool))
    return table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]


def save_members(files, relative, variable, open_kwargs=None):
    """Open one model's member files for one experiment, stack them along ``member`` and save them.

    Args:
        files (Sequence[Path]): One file per member.
        relative (Path): Where to save, relative to the output roots (``paths.lesfmip_monthly``).
        variable (str): Variable name.
        open_kwargs (dict | None): Passed to ``xr.open_mfdataset``; ``OPEN_KWARGS`` if None.

    Returns:
        Path: The store written.
    """
    ds = xr.open_mfdataset(files, preprocess=partial(preprocess_member, variable=variable),
                           **(open_kwargs or OPEN_KWARGS))
    ds = ds.chunk({"time": -1, "lat": max(ds.sizes["lat"] // 2, 1), "lon": -1, "member": 5})
    logger.info(f"{relative.name}: {dict(ds.sizes)} {format_bytes(ds.nbytes)}")
    return storage.save(ds, relative, consolidated=True)


def convert_experiment(variable, model, experiment, overwrite=False, open_kwargs=None):
    """Every group of one model's files for one experiment, to one store per group.

    The group is the last part of each file name ('interp' for the files on
    the common grid). A group that fails to open is reported, not raised, so
    one broken model does not stop the rest.

    Returns:
        dict: group -> {'error', 'files', 'store'} for each group that failed.
    """
    files = sorted((paths.LESFMIP_RAW / variable / experiment / model).rglob("*.nc"))

    def group_of(file):
        return file.name.removesuffix(".nc").split("_")[-1]

    groups = {group: list(fs) for group, fs in groupby(sorted(files, key=group_of), key=group_of)}
    logger.info(f"{experiment}/{model}: {len(files)} files, "
                + ", ".join(f"{group}={len(fs)}" for group, fs in groups.items()))

    failed = {}
    for group, group_files in groups.items():
        store = paths.lesfmip_monthly(model, variable, experiment, group)
        if not overwrite and storage.exists(store):
            logger.info(f"{store.name} exists, skipped")
            continue
        try:
            save_members(group_files, store, variable, open_kwargs)
        except (ValueError, KeyError) as error:
            failed[group] = {"error": error, "files": group_files, "store": store}
    return failed


def convert_lesfmip(variable, models, experiments, overwrite=False, open_kwargs=None):
    """``convert_experiment`` for every model and experiment.

    Returns:
        dict: (experiment, model) -> that call's failures, for those with any.
    """
    failures = {}
    for experiment in experiments:
        for model in models:
            if not (paths.LESFMIP_RAW / variable / experiment / model).exists():
                continue
            failed = convert_experiment(variable, model, experiment, overwrite, open_kwargs)
            if failed:
                failures[(experiment, model)] = failed
    return failures


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------

#(t): ERA5 is cut a little north of the LESFMIP edge, so the conservative regridding covers the edge cells
ERA5_LAT_MAX = -38.0


def _era5_south(ds):
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
            values = _era5_south(ds)[name].values
        subtotal = values.sum(axis=0, dtype="float64")
        total = subtotal if total is None else total + subtotal
        n += values.shape[0]
    return (total / n).astype("float32")


def era5_regridder(variable, target, year):
    """A conservative xesmf regridder from the ERA5 grid to ``target``'s (the LESFMIP 2.5° grid)."""
    import cf_xarray  # noqa: F401  (adds .cf)
    import xesmf

    first = _era5_south(xr.open_dataset(era5_files(variable, year)[0]))
    source = first[["latitude", "longitude"]].rename({"latitude": "lat", "longitude": "lon"}).cf.add_bounds(["lat", "lon"])
    source["lat_bounds"] = source["lat_bounds"].clip(-90.0, 90.0)
    destination = target[["lat", "lon"]].cf.add_bounds(["lat", "lon"])
    return xesmf.Regridder(source, destination, "conservative", periodic=True), first


def era5_monthly(variable, target, years, client):
    """Hourly ERA5 to monthly means on ``target``'s grid, saved as ``paths.era5_monthly(variable)``.

    Each month is averaged on a worker. Each year is saved as soon as it is
    complete, so an interrupted run picks up where it stopped; years already in
    the combined store, or saved as a year store, are skipped. Once every year
    is done, the year stores are combined with any existing combined store,
    checked, and deleted.

    Args:
        variable (str): Key of ``config.VARIABLES`` (with an ERA5 recipe).
        target (xr.Dataset | xr.DataArray): Anything on the LESFMIP grid, e.g. one monthly store.
        years (Iterable[int]): Years wanted.
        client (distributed.Client): The cluster.

    Returns:
        xr.DataArray: The combined monthly means (raw units), loaded.
    """
    var = config.VARIABLES[variable]
    years_dir = paths.era5_years_dir(variable)
    done = {int(p.name.removesuffix(".zarr")) for p in paths.find_all(years_dir, "*.zarr")}
    if storage.exists(paths.era5_monthly(variable)):
        done |= set(storage.open_dataarray(paths.era5_monthly(variable)).time.dt.year.values.tolist())
    todo = [year for year in years if year not in done]
    logger.info(f"ERA5 {variable}: {len(todo)} years to do")

    if todo:
        regridder, first = era5_regridder(variable, target, todo[0])
        futures = {(year, month): client.submit(era5_month_mean, era5_files(variable, year, month), var.era5_name,
                                                pure=False)
                   for year in todo for month in range(1, 13)}
        for year in todo:
            arrays = client.gather([futures[(year, month)] for month in range(1, 13)])
            monthly = xr.DataArray(
                np.stack(arrays), dims=("time", "lat", "lon"),
                coords={"time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
                        "lat": first["latitude"].values, "lon": first["longitude"].values},
            )
            storage.save(regridder(monthly).astype("float32").rename(variable), years_dir / f"{year}.zarr")
            logger.info(f"ERA5 {variable}: {year} saved")

    return combine_era5_years(variable)


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
