"""Raw files to monthly zarr stores: LESFMIP (one netCDF file per member) and ERA5 (hourly files).

Used by notebook calc_01 on JASMIN, once per variable, and by notebook calc_04 for the
sea ice, which is in CEDA's copy of the CMIP6 archive rather than the
LEADER-EPESC folders (the last section lists its folders). Only the pieces are
here (opening one model's member files together, a month of ERA5, the CEDA
inventory); the notebooks list the files, run the loops, save and check. Runs on
the Dask cluster, so send the package to the workers first
(``jasmin.upload_package(client)``).
"""

import re
from functools import partial
from pathlib import Path

import pandas as pd
import xarray as xr

from . import config

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


def ceda_inventory(models, experiments, roots):
    """Every folder of the sea ice and the cell areas for ``models`` and ``experiments`` in CEDA's archive.

    Args:
        models, experiments (Sequence[str]): What to look for (the cell areas also in piControl).
        roots (Sequence[Path]): The trees searched, e.g. CEDA's CMIP6/DAMIP, CMIP6/CMIP and CMIP6Plus/LESFMIP.

    Returns:
        pd.DataFrame: One row per member and variable: project, activity, institution, model, experiment,
        member, table, variable, grid, the number of files, the first and last year, and the folder.
    """
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
