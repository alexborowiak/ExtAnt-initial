"""Raw files to monthly zarr stores: LESFMIP (one netCDF file per member) and ERA5 (hourly files).

Used by notebook 01 on JASMIN, once per variable, and by notebook 09 for the
sea ice, which is in CEDA's copy of the CMIP6 archive rather than the
LEADER-EPESC folders (the last section finds its files). Only the pieces are here
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


def era5_files(variable, year, month=None, hour=None):
    """The hourly ERA5 files of one year (or month), sorted; with ``hour``, only that hour (UTC) of each day."""
    code = config.VARIABLES[variable].era5_code
    if code is None:
        raise ValueError(f"no ERA5 recipe for {variable} yet: see config.VARIABLES")
    months = f"{month:02d}" if month else "*"
    hours = "" if hour is None else f"{hour:02d}00"
    return sorted(Path(paths.ERA5_RAW).glob(f"{year}/{months}/*/*{hours}.{code}.nc"))


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


# ---------------------------------------------------------------------------
# CEDA's CMIP6 archive: the sea ice
# ---------------------------------------------------------------------------

#(t): The sea ice, on the ocean grid (siconc) or the atmosphere grid (siconca), and the cell areas of each grid
#(t): (areacello, areacella), as (table, variable). A model's cell areas are the same in every run, so they are
#(t): also looked for in piControl, which often has them when the other runs do not
CEDA_SEA_ICE = (("SImon", "siconc"), ("SImon", "siconca"))
CEDA_AREAS = (("Ofx", "areacello"), ("fx", "areacella"))


def file_years(path):
    """First and last year of a CMIP file, from the dates at the end of its name (..._185001-201412.nc)."""
    match = re.search(r"_(\d{4})\d*-(\d{4})\d*\.nc$", Path(path).name)
    return (int(match[1]), int(match[2])) if match else (None, None)


def member_number(member):
    """The n of r<n> in a member label, so that r2 sorts before r10."""
    return int(re.search(r"r(\d+)i", member)[1])


def ceda_folder_row(folder):
    """One inventory row for a .../<member>/<table>/<variable>/<grid>/latest folder of CEDA's archive."""
    folder = Path(folder)
    files = sorted(folder.glob("*.nc"))
    spans = [file_years(path) for path in files]
    starts = [start for start, _ in spans if start is not None]
    ends = [end for _, end in spans if end is not None]
    institution, model, experiment, member, table, variable, grid = folder.parts[-8:-1]
    root = folder.parents[7]
    return {"project": root.parent.name, "activity": root.name, "institution": institution, "model": model,
            "experiment": experiment, "member": member, "table": table, "variable": variable, "grid": grid,
            "n_files": len(files), "start": min(starts, default=None), "end": max(ends, default=None),
            "folder": str(folder)}


def ceda_inventory(models, experiments, roots=None):
    """Every folder of the sea ice and the cell areas for ``models`` and ``experiments`` in CEDA's archive.

    Args:
        models, experiments (Sequence[str]): What to look for (the cell areas also in piControl).
        roots (Sequence[Path] | None): The trees searched; every one of ``paths.CEDA_CMIP6`` if None.

    Returns:
        pd.DataFrame: One row per member and variable: project, activity, institution, model, experiment,
        member, table, variable, grid, the number of files, the first and last year, and the folder.
    """
    roots = list(paths.CEDA_CMIP6.values()) if roots is None else roots
    wanted = [(table, variable, list(experiments)) for table, variable in CEDA_SEA_ICE]
    wanted += [(table, variable, [*experiments, "piControl"]) for table, variable in CEDA_AREAS]
    rows = []
    for root in roots:
        for model in models:
            for table, variable, searched in wanted:
                for experiment in searched:
                    pattern = Path(root) / "*" / model / experiment / "*" / table / variable / "*" / "latest"
                    folders = sorted(Path(root).glob(str(pattern.relative_to(root))))
                    rows += [ceda_folder_row(folder) for folder in folders]
    columns = ["project", "activity", "institution", "model", "experiment", "member", "table", "variable", "grid",
               "n_files", "start", "end", "folder"]
    table_df = pd.DataFrame(rows, columns=columns)
    order = table_df["member"].map(member_number) if len(table_df) else table_df["member"]
    return table_df.assign(number=order).sort_values(["model", "experiment", "variable", "number"]).drop(
        columns="number").reset_index(drop=True)


def ceda_member_files(rows):
    """Member label -> its files, for inventory rows of one model, experiment and variable (``ceda_inventory``)."""
    return {row.member: sorted(Path(row.folder).glob("*.nc")) for row in rows.itertuples()}


def ceda_area_file(inventory, model, variable, grid):
    """The cell-area file (``areacello`` or ``areacella``) of one model's grid, from any run that has it."""
    rows = inventory[(inventory["model"] == model) & (inventory["variable"] == variable)
                     & (inventory["grid"] == grid)]
    if rows.empty:
        raise FileNotFoundError(f"no {variable} on grid {grid} for {model} in the inventory")
    return sorted(Path(rows["folder"].iloc[0]).glob("*.nc"))[0]
