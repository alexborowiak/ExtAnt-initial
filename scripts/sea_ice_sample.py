"""Find the LESFMIP sea-ice concentration on CEDA and save a small sample of it, with ERA5's sea-ice cover.

The sea ice is not in the LEADER-EPESC folders. It is in CEDA's copy of the CMIP6 archive, on each model's own
ocean grid (siconc) or atmosphere grid (siconca), in three trees:

    /badc/cmip6/data/CMIP6/DAMIP/          hist-nat, hist-aer, hist-GHG, hist-totalO3 / hist-stratO3
    /badc/cmip6/data/CMIP6/CMIP/           historical
    /badc/cmip6/data/CMIP6Plus/LESFMIP/    the newer LESFMIP runs (members f2023-r<n>...; not to be pooled with DAMIP)

each laid out as <root>/<institution>/<model>/<experiment>/<member>/<table>/<variable>/<grid>/latest/*.nc.
Only the models and experiments of the analysis are looked for. Run on a JASMIN sci server:

    module load jaspy
    python sea_ice_sample.py

Everything is written to OUT_DIR:

    inventory.csv                                   one row per folder found: member, grid, files, first and last year
    member_counts.csv                               members per model and experiment, for each variable
    lesfmip/<model>_<experiment>_<variable>_<member>_<grid>.nc   first member, SAMPLE_YEARS, south of LAT_MAX
    lesfmip/<model>_<area>_<grid>.nc                the cell areas (areacello / areacella), south of LAT_MAX
    era5_ci.nc                                      ERA5 sea-ice cover, 00 UTC on the 15th of each month of SAMPLE_YEARS
"""

import glob
import re
from pathlib import Path

import pandas as pd
import xarray as xr

#(t): Where the tables and the sample are written
OUT_DIR = Path("/work/scratch-pw4/aborow/ExtAnt/sea_ice_sample")

#(t): The models analysed (notebook 01's member counts) and their experiments. The ozone run is hist-totalO3 in
#(t): some models and hist-stratO3 in others, so both are looked for
MODELS = ["CanESM5", "GISS-E2-1-G", "HadGEM3-GC31-LL", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM"]
EXPERIMENTS = ["hist-nat", "hist-aer", "hist-GHG", "hist-totalO3", "hist-stratO3", "historical"]

ROOTS = [Path("/badc/cmip6/data/CMIP6/DAMIP"), Path("/badc/cmip6/data/CMIP6/CMIP"),
         Path("/badc/cmip6/data/CMIP6Plus/LESFMIP")]

#(t): What is looked for: (table, variable, experiments). siconc is on the ocean grid, whose cell areas are
#(t): areacello (Ofx); siconca on the atmosphere grid, areacella (fx). The areas are the same in every run of a
#(t): model, so any run will do, and piControl often has them when the others do not
TABLE = "SImon"
AREAS = {"siconc": "areacello", "siconca": "areacella"}
WANTED = [("SImon", "siconc", EXPERIMENTS), ("SImon", "siconca", EXPERIMENTS),
          ("Ofx", "areacello", EXPERIMENTS + ["piControl"]), ("fx", "areacella", EXPERIMENTS + ["piControl"])]

#(t): The sample: two years (a seasonal cycle and a bit), everything south of LAT_MAX
SAMPLE_YEARS = ("2013", "2014")
LAT_MAX = -40

#(t): Hourly ERA5 analyses, /<year>/<month>/<day>/ecmwf-era5_oper_an_sfc_<YYYYMMDDHHMM>.<code>.nc; ci is sea-ice cover
ERA5_RAW = Path("/badc/ecmwf-era5/data/oper/an_sfc")
ERA5_CODE = "ci"


# ---------------------------------------------------------------------------
# 1. What is there
# ---------------------------------------------------------------------------

def file_years(path):
    """First and last year of a CMIP file, from the dates at the end of its name (..._185001-201412.nc)."""
    match = re.search(r"_(\d{4})\d*-(\d{4})\d*\.nc$", path.name)
    return (int(match[1]), int(match[2])) if match else (None, None)


def member_number(member):
    """The n of r<n> in a member name, so that r2 sorts before r10."""
    return int(re.search(r"r(\d+)i", member)[1])


def folder_row(folder):
    """One inventory row for a .../<member>/<table>/<variable>/<grid>/latest folder."""
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


def inventory():
    """Every folder of the WANTED variables for MODELS, in every root."""
    rows = []
    for root in ROOTS:
        for model in MODELS:
            for table, variable, experiments in WANTED:
                for experiment in experiments:
                    pattern = root / "*" / model / experiment / "*" / table / variable / "*" / "latest"
                    rows += [folder_row(Path(folder)) for folder in sorted(glob.glob(str(pattern)))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. The sample
# ---------------------------------------------------------------------------

def latitude(ds):
    """The latitude of ``ds``: 1-D on a regular grid, 2-D (j, i) on an ocean model's own grid."""
    for name in ("latitude", "lat", "nav_lat"):
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"no latitude among {list(ds.variables)}")


def south(ds, lat_max=LAT_MAX):
    """``ds`` cut to the rows (or cells) with any latitude south of ``lat_max``, on 1-D and 2-D grids alike."""
    is_south = latitude(ds) <= lat_max
    for dim in is_south.dims:
        others = [other for other in is_south.dims if other != dim]
        keep = is_south.any(others) if others else is_south
        ds = ds.isel({dim: keep.values})
    return ds


def save(ds, path):
    """Write ``ds`` to netCDF, compressed (most of the sea ice is 0 or missing, so it shrinks a lot)."""
    ds = ds.drop_encoding()
    encoding = {name: {"zlib": True, "complevel": 4} for name, da in ds.data_vars.items() if da.dtype.kind == "f"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding)
    print(f"saved {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def open_years(folder, years=SAMPLE_YEARS):
    """The files of one folder that cover ``years``, opened together and cut to those years."""
    first, last = int(years[0]), int(years[1])
    files = [path for path in sorted(Path(folder).glob("*.nc"))
             if file_years(path)[0] is not None and file_years(path)[0] <= last and file_years(path)[1] >= first]
    if not files:
        raise FileNotFoundError(f"no file in {folder} covers {years[0]}-{years[1]}")
    for path in files:
        print(f"opening {path}")
    ds = xr.concat([xr.open_dataset(path) for path in files], dim="time", data_vars="minimal", coords="minimal",
                   compat="override")
    return ds.sel(time=slice(*years))


def first_members(inventory_df):
    """For each model, experiment and variable, the inventory row of the first member (DAMIP/CMIP before CMIP6Plus)."""
    ice_df = inventory_df[inventory_df["table"] == TABLE]
    ice_df = ice_df.assign(number=ice_df["member"].map(member_number)).sort_values(["project", "number"])
    return ice_df.groupby(["model", "experiment", "variable"]).head(1).drop(columns="number")


def save_member(row):
    """Save SAMPLE_YEARS of one member, south of LAT_MAX."""
    ds = south(open_years(row["folder"])).load()
    save(ds, OUT_DIR / "lesfmip" / f"{row['model']}_{row['experiment']}_{row['variable']}_{row['member']}_{row['grid']}.nc")


def save_area(inventory_df, model, variable, grid):
    """Save the cell areas of the grid ``variable`` is on, south of LAT_MAX."""
    area = AREAS[variable]
    area_df = inventory_df[(inventory_df["model"] == model) & (inventory_df["variable"] == area)
                           & (inventory_df["grid"] == grid)]
    if area_df.empty:
        print(f"no {area} on grid {grid} for {model}")
        return
    path = sorted(Path(area_df["folder"].iloc[0]).glob("*.nc"))[0]
    print(f"opening {path}")
    save(south(xr.open_dataset(path)).load(), OUT_DIR / "lesfmip" / f"{model}_{area}_{grid}.nc")


# ---------------------------------------------------------------------------
# 3. ERA5
# ---------------------------------------------------------------------------

def era5_codes(year="2014", month="01", day="15"):
    """The variable codes in the ERA5 file names of one day (2t, msl, ci, ...)."""
    return sorted({path.name.split(".")[-2] for path in (ERA5_RAW / year / month / day).glob("*.nc")})


def save_era5(years=SAMPLE_YEARS, code=ERA5_CODE):
    """ERA5 at 00 UTC on the 15th of each month of ``years``, south of LAT_MAX (the ice changes once a day)."""
    files = []
    for year in range(int(years[0]), int(years[1]) + 1):
        for month in range(1, 13):
            day = ERA5_RAW / str(year) / f"{month:02d}" / "15"
            files += sorted(day.glob(f"*{year}{month:02d}150000.{code}.nc"))
    for path in files:
        print(f"opening {path}")
    ds = xr.concat([south(xr.open_dataset(path)) for path in files], dim="time").load()
    save(ds, OUT_DIR / f"era5_{code}.nc")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory_df = inventory()
    inventory_df.to_csv(OUT_DIR / "inventory.csv", index=False)
    print(f"saved {OUT_DIR / 'inventory.csv'}  ({len(inventory_df)} folders)")
    if inventory_df.empty:
        print(f"nothing found under {[str(root) for root in ROOTS]}")
        return

    #(t): Members per model (rows) and variable/experiment (columns); CMIP6Plus kept apart from CMIP6
    member_counts_df = inventory_df[inventory_df["table"] == TABLE].pivot_table(
        index=["project", "model"], columns=["variable", "experiment"], values="member", aggfunc="nunique")
    member_counts_df.to_csv(OUT_DIR / "member_counts.csv")
    print(member_counts_df.to_string())

    #(t): The models with no sea ice at all
    ice_models = set(inventory_df.loc[inventory_df["table"] == TABLE, "model"])
    print("no sea ice found for:", sorted(set(MODELS) - ice_models) or "none")

    first_df = first_members(inventory_df)
    for _, row in first_df.iterrows():
        try:
            save_member(row)
        except Exception as error:  # noqa: BLE001 - one bad folder should not stop the rest
            print(f"FAILED {row['folder']}: {error!r}")

    for model, variable, grid in first_df[["model", "variable", "grid"]].drop_duplicates().itertuples(index=False):
        try:
            save_area(inventory_df, model, variable, grid)
        except Exception as error:  # noqa: BLE001
            print(f"FAILED areas of {model} {variable} {grid}: {error!r}")

    codes = era5_codes()
    print(f"ERA5 codes in {ERA5_RAW / '2014' / '01' / '15'}: {codes}")
    if ERA5_CODE in codes:
        save_era5()
    else:
        print(f"no '{ERA5_CODE}' files in ERA5 an_sfc: sea-ice cover is somewhere else")

    total = sum(path.stat().st_size for path in OUT_DIR.rglob("*.nc")) / 1e6
    print(f"done: {total:.0f} MB of netCDF in {OUT_DIR}")


if __name__ == "__main__":
    main()
