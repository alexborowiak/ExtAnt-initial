"""Opening the data: the monthly stores, the saved seasonal means, and the test sample.

Every function finds its files through ``paths``, so they work the same
whether a store is still on scratch or has been moved to the group workspace.
The stores themselves are written by ``convert`` (monthly) and notebook 01
(seasonal).
"""

from pathlib import Path

import pandas as pd
import xarray as xr

from . import config, paths, storage


# ---------------------------------------------------------------------------
# Monthly LESFMIP stores
# ---------------------------------------------------------------------------

def parse_store(store):
    """Model, variable, experiment and group from a monthly store's name (see ``paths.lesfmip_monthly``)."""
    model, variable, experiment, group = Path(store).stem.rsplit("_", 4)[:4]
    return {"model": model, "variable": variable, "experiment": experiment, "group": group, "path": Path(store)}


def monthly_stores(variable, models=None, experiments=None, groups=None):
    """Every monthly LESFMIP store of ``variable``, as ``parse_store`` dicts. None means 'all'."""
    entries = [parse_store(s) for s in paths.find_all(paths.lesfmip_monthly_dir(variable), "*_monthly.zarr")]
    return [e for e in entries
            if (models is None or e["model"] in models)
            and (experiments is None or e["experiment"] in experiments)
            and (groups is None or e["group"] in groups)]


def store_table(variable, groups=None):
    """Which models have a store for which experiment: a 0/1 table, models down, experiments across."""
    entries = monthly_stores(variable, groups=groups)
    if not entries:
        raise FileNotFoundError(f"no monthly {variable} stores in {paths.lesfmip_monthly_dir(variable)}")
    df = pd.DataFrame(entries).drop(columns="path")
    return df.pivot_table(index="model", columns="experiment", values="group", aggfunc="size",
                          fill_value=0).astype(bool).astype(int)


def models_with_most_experiments(table, n, experiments=None):
    """The ``n`` models with stores for the most experiments (of ``experiments``, if given)."""
    if experiments is not None:
        table = table[[e for e in experiments if e in table.columns]]
    return list(table.sum(axis=1).sort_values(ascending=False, kind="stable").index[:n])


def open_lesfmip_monthly(variable, models=None, experiments=None, groups=None, levels=("model", "experiment")):
    """Open the matching monthly stores (lazily) as a DataTree. None means 'all'.

    Args:
        variable (str): Variable name.
        models, experiments, groups (Sequence[str] | None): Subsets to open.
        levels (tuple[str, ...]): Store fields giving the tree's levels, outermost first.

    Returns:
        xr.DataTree: e.g. /<model>/<experiment>, monthly, on (member, time, lat, lon), raw units.
    """
    entries = monthly_stores(variable, models, experiments, groups)
    if not entries:
        raise ValueError(f"no {variable} stores matched {models=} {experiments=} {groups=}")
    nodes = {}
    for entry in entries:
        path = "/".join(entry[level] for level in levels)
        if path in nodes:
            raise ValueError(f"two stores for {path!r}; pass groups= to choose one")
        nodes[path] = xr.open_zarr(entry["path"], consolidated=True)
    return xr.DataTree.from_dict(nodes)


# ---------------------------------------------------------------------------
# ERA5 and the seasonal means
# ---------------------------------------------------------------------------

def open_era5_monthly(variable):
    """Monthly ERA5 means on the LESFMIP grid, in raw units (as ``convert.era5_monthly`` saved them)."""
    return storage.open_dataarray(paths.era5_monthly(variable))


def open_seasonal(variable, models=None, experiments=None):
    """The saved seasonal LESFMIP means (lazily), optionally only some models and experiments.

    Returns:
        xr.DataTree: /<model>/<experiment> on (member, year, season, lat, lon), analysis units.
    """
    tree = storage.open_tree(paths.seasonal("lesfmip", variable))
    if models is None and experiments is None:
        return tree
    return xr.DataTree.from_dict({
        node.relative_to(tree): node.to_dataset()
        for node in tree.leaves
        if (models is None or node.parent.name in models) and (experiments is None or node.name in experiments)
    })


def open_era5_seasonal(variable):
    """The saved seasonal ERA5 means on the LESFMIP grid, on (year, season, lat, lon), analysis units."""
    return storage.open_dataarray(paths.seasonal("era5", variable))


# ---------------------------------------------------------------------------
# The test sample
# ---------------------------------------------------------------------------

def open_lesfmip_sample(path=paths.SAMPLE_DIR, models=None, experiments=None):
    """The sample LESFMIP files as a /<model>/<experiment> DataTree, like ``open_lesfmip_monthly``.

    The files are ``<model>_<experiment>_sample.nc``: monthly ``tas`` in K on
    (member, time, lat, lon) for a 3x3 block of cells, each model on its own
    calendar. None means 'all'.

    Returns:
        xr.DataTree: Loaded into memory.
    """
    nodes = {}
    for file in sorted(Path(path).glob("*_sample.nc")):
        if file.name.startswith("era5"):
            continue
        model, experiment = file.stem.removesuffix("_sample").rsplit("_", 1)
        if (models is None or model in models) and (experiments is None or experiment in experiments):
            nodes[f"{model}/{experiment}"] = xr.open_dataset(file).load()
    if not nodes:
        raise ValueError(f"no sample files matched {models=} {experiments=} in {path}")
    return xr.DataTree.from_dict(nodes)


def open_era5_sample(path=paths.SAMPLE_DIR):
    """The sample ERA5 file, like ``open_era5_monthly``: monthly ``tas`` in K from 1979, on (time, lat, lon)."""
    return xr.open_dataarray(Path(path) / "era5_sample.nc").load()


def write_sample(tree, era5, point=None, path=paths.SAMPLE_DIR):
    """Save the 3x3 block of cells around ``point`` as the test sample, one file per model and experiment.

    Args:
        tree (xr.DataTree): Monthly LESFMIP, /<model>/<experiment>, raw units.
        era5 (xr.DataArray): Monthly ERA5 on the same grid, raw units.
        point (dict | None): ``lat`` and ``lon`` of the centre cell; ``config.POINT`` if None.
        path (Path): Folder to write to.

    Returns:
        list[Path]: The files written.
    """
    point = point or config.POINT
    ilat = era5.indexes["lat"].get_indexer([point["lat"]], method="nearest")[0]
    ilon = era5.indexes["lon"].get_indexer([point["lon"]], method="nearest")[0]
    cells = dict(lat=era5.lat.values[ilat - 1:ilat + 2], lon=era5.lon.values[ilon - 1:ilon + 2])

    def sample(obj):
        #(c): tolerance makes a grid mismatch fail loudly instead of silently picking other cells
        return obj.sel(cells, method="nearest", tolerance=0.01).drop_encoding().load()

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    written = []
    for node in tree.leaves:
        if node.data_vars:
            file = path / f"{node.parent.name}_{node.name}_sample.nc"
            sample(node.to_dataset()).to_netcdf(file)
            written.append(file)
    file = path / "era5_sample.nc"
    sample(era5).to_netcdf(file)
    return written + [file]
