from xarray import DataTree
import xarray as xr
import pandas as pd
import numpy as np

from functools import wraps

def skip_empty(func):
    @wraps(func)
    def wrapper(obj, *args, **kwargs):

        # Empty Dataset: no data variables
        if isinstance(obj, xr.Dataset):
            if len(obj.data_vars) == 0:
                return obj

        # Empty DataArray: one or more dimensions have length 0
        elif isinstance(obj, xr.DataArray):
            if obj.size == 0:
                return obj

        return func(obj, *args, **kwargs)

    return wrapper


@skip_empty
def coord_info(ds):
    name = next(iter(ds.data_vars))
    arr = ds[name].data
    return {
        **{("size", d): n for d, n in ds.sizes.items()},
        **{("chunk", d): n for d, n in zip(ds[name].dims, arr.chunksize)},
        ("memory", "mbytes"): int(ds.nbytes / 1024**2),
        ("memory", "chunk_mbytes"): round(float(np.prod(arr.chunksize) * arr.dtype.itemsize / 1024**2), 1),
        ("memory", "nchunks"): arr.npartitions
    }



def permute_levels(dt, order, name=None, inherit=False):
    """Reorder the hierarchy levels of a uniformly-nested DataTree.
    Parameters
    ----------
    dt : DataTree
        Tree whose leaves all sit at the same depth.
    order : sequence of int
        Permutation of ``range(depth)``; ``order[i]`` is the old level placed at level ``i``.
    name : str, optional
        Name for the new root, defaults to ``dt.name``.
    inherit : bool, default False
        Whether leaves keep coordinates inherited from their old parents.
    Returns
    -------
    DataTree
        New tree with the levels reordered.
    """
    order = tuple(order)
    out = {}
    for leaf in dt.leaves:
        parts = leaf.path.strip("/").split("/")
        if len(parts) != len(order):
            raise ValueError(f"leaf {leaf.path!r} has depth {len(parts)}, expected {len(order)}")
        out["/".join(parts[i] for i in order)] = leaf.to_dataset(inherit=inherit)
    return DataTree.from_dict(out, name=dt.name if name is None else name)


def swap_levels(dt, i=0, j=1, **kwargs):
    """Swap two levels of a DataTree hierarchy.
    Parameters
    ----------
    dt : DataTree
        Tree whose leaves all sit at the same depth.
    i, j : int, default 0, 1
        Levels to exchange.
    **kwargs
        Passed to :func:`permute_levels`.
    Returns
    -------
    DataTree
        New tree with levels ``i`` and ``j`` swapped.
    """
    depth = max(len(leaf.path.strip("/").split("/")) for leaf in dt.leaves)
    order = list(range(depth))
    order[i], order[j] = order[j], order[i]
    return permute_levels(dt, order, **kwargs)


@skip_empty
def to_dataset(branch, concat_dim):

    # to_concat = []
    # for key, leaf in branch.items():
    #     if not isinstance(leaf, xr.DataArray): leaf = leaf.to_dataset().to_dataarray()
    #     to_concat.append(leaf.assign_coords({concat_dim:key}))
    # return xr.concat(to_concat, dim=concat_dim).to_dataset()

    
    return xr.concat(
    [leaf.tas.assign_coords({concat_dim:key}) for key, leaf in branch.items()],
    dim='experiment')


def tree_to_dataset(dt: xr.DataTree, dims: list[str], **concat_kwargs) -> xr.Dataset:
    """Collapse a DataTree into a Dataset, mapping each path level to a dimension.

    e.g. a tree laid out as /<model>/<experiment> with dims=['model', 'experiment']
    """
    leaves = [node for node in dt.leaves if node.has_data]
    keys = [tuple(node.relative_to(dt).split("/")) for node in leaves]

    if any(len(k) != len(dims) for k in keys):
        raise ValueError(f"Not all leaves are at depth {len(dims)}")

    concat_kwargs.setdefault("join", "outer")
    ds = xr.concat([node.to_dataset() for node in leaves], dim="_stacked",
                   **concat_kwargs)

    idx = pd.MultiIndex.from_tuples(keys, names=dims)
    ds = ds.assign_coords(xr.Coordinates.from_pandas_multiindex(idx, "_stacked"))
    return ds.unstack("_stacked")