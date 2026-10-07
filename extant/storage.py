"""Saving to zarr and opening again, so slow steps only ever run once.

``save`` writes under ``paths.SCRATCH`` and ``open_*`` read through
``paths.find`` (DATA_DIR first, then SCRATCH); see ``paths`` for why the two
differ on JASMIN. Every function takes a path relative to those roots, as
built by ``paths.seasonal``, ``paths.result`` and friends.

A store is written as ``<name>.part`` and only renamed to ``<name>`` once it is
complete, so a crashed or interrupted write never leaves a half-written store
under the real name.
"""

import logging
import os
import shutil
import time

import xarray as xr

from . import paths

logger = logging.getLogger(__name__)


def _describe(obj):
    """e.g. 'Dataset (model: 6, season: 4, lat: 21, lon: 144)' or 'DataTree of 30 nodes', for the log."""
    if isinstance(obj, xr.DataTree):
        return f"DataTree of {sum(1 for node in obj.leaves if node.data_vars)} nodes"
    return f"{type(obj).__name__} ({', '.join(f'{dim}: {n}' for dim, n in obj.sizes.items())})"


def _uniform_chunks(ds):
    """Zarr needs equal chunks along each dim (bar the last), which slicing and resampling often break."""
    if not ds.chunks:
        return ds
    ds = ds.unify_chunks()
    return ds.chunk({dim: max(sizes) for dim, sizes in ds.chunks.items()})


def _prepare(ds):
    """Drop the encodings copied from the source stores (their chunks no longer fit) and even out the chunks."""
    return _uniform_chunks(ds.drop_encoding())


def _rename(source, target, attempts=10):
    """``os.replace``, retried for a moment: on Windows a virus scanner can hold files that were just written."""
    for attempt in range(attempts):
        try:
            return os.replace(source, target)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def _save_tree(tree, store, **to_zarr_kwargs):
    """Write every node of a DataTree as a zarr group, computing all of their data at once.

    Each node's write is deferred and then everything is computed together,
    so the whole cluster works on the whole tree. (Older xarray versions of
    ``DataTree.to_zarr`` compute node by node, which leaves most workers idle.)
    Only the root keeps inherited coordinates, as ``DataTree.to_zarr`` does.
    """
    import dask
    import zarr

    writes = []
    for node in tree.subtree:
        at_root = node is tree
        dataset = _prepare(node.to_dataset(inherit=at_root))
        writes.append(dataset.to_zarr(store, group=None if at_root else node.relative_to(tree),
                                      mode="w" if at_root else "a", compute=False, consolidated=False,
                                      **to_zarr_kwargs))
    dask.compute(*writes)
    zarr.consolidate_metadata(str(store))


def save(obj, relative, **to_zarr_kwargs):
    """Write a Dataset, DataArray or DataTree to ``SCRATCH / relative``, replacing any earlier copy.

    Dask-backed data are computed as they are written, in parallel on the
    cluster if there is one.

    Args:
        obj (xr.Dataset | xr.DataArray | xr.DataTree): What to save. A DataArray needs a name.
        relative (Path | str): Path relative to the output roots, e.g. ``paths.result('ttest', 'tas', 'full')``.
        **to_zarr_kwargs: Passed to ``to_zarr``.

    Returns:
        Path: Where the store was written.
    """
    path = paths.output(relative)
    part = path.with_name(path.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    logger.info(f"writing {_describe(obj)} to {path}")

    if isinstance(obj, xr.DataTree):
        _save_tree(obj, part, **to_zarr_kwargs)
    elif isinstance(obj, xr.DataArray):
        if obj.name is None:
            raise ValueError("name the DataArray (da.rename('...')) so it can be opened again")
        _prepare(obj.to_dataset()).to_zarr(part, mode="w", **to_zarr_kwargs)
    else:
        _prepare(obj).to_zarr(part, mode="w", **to_zarr_kwargs)

    shutil.rmtree(path, ignore_errors=True)
    _rename(part, path)
    logger.info(f"saved {path}")
    if paths.DATA_DIR != paths.SCRATCH and (paths.DATA_DIR / relative).exists():
        logger.warning(f"an older {relative} in {paths.DATA_DIR} is read before this one: move this one over it")
    return path


def open_dataset(relative, load=False, **kwargs):
    """Open a saved Dataset (lazily, unless ``load``)."""
    path = paths.find(relative)
    logger.info(f"opening {path}")
    ds = xr.open_zarr(path, **kwargs)
    return ds.load() if load else ds


def open_dataarray(relative, load=False, **kwargs):
    """Open a saved DataArray (a store holding one data variable)."""
    ds = open_dataset(relative, load=load, **kwargs)
    (name,) = ds.data_vars
    return ds[name]


def open_tree(relative, load=False, **kwargs):
    """Open a saved DataTree (lazily, unless ``load``)."""
    kwargs.setdefault("chunks", {})
    path = paths.find(relative)
    logger.info(f"opening {path}")
    tree = xr.open_datatree(path, engine="zarr", **kwargs)
    return tree.load() if load else tree


def exists(relative):
    """Whether ``relative`` has been saved (in either root)."""
    try:
        paths.find(relative)
    except FileNotFoundError:
        return False
    return True
