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
    #(c): Data in memory (e.g. the sample) is one chunk
    chunksize = getattr(arr, "chunksize", arr.shape)
    return {
        **{("size", d): n for d, n in ds.sizes.items()},
        **{("chunk", d): n for d, n in zip(ds[name].dims, chunksize)},
        ("memory", "mbytes"): int(ds.nbytes / 1024**2),
        ("memory", "chunk_mbytes"): round(float(np.prod(chunksize) * arr.dtype.itemsize / 1024**2), 1),
        ("memory", "nchunks"): getattr(arr, "npartitions", 1)
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

    Each leaf's path below `dt` becomes its coordinate labels: the leaf at
    /CanESM5/historical gets model='CanESM5', experiment='historical'.
    `dims` names those levels, top first, and must be a list even for one level.

    Examples
    --------
    Whole tree, laid out as /<model>/<experiment>:

    >>> lesfmip_ds = tree_to_dataset(lesfmip_season_tree, dims=['model', 'experiment'])
    >>> lesfmip_ds.tas.dims
    ('member', 'year', 'season', 'lat', 'lon', 'model', 'experiment')

    One model's subtree, laid out as /<experiment>, so there's one level:

    >>> canesm_ds = tree_to_dataset(lesfmip_season_tree['CanESM5'].isel(member=0),
    ...                             dims=['experiment'])          # not dims='experiment'
    >>> canesm_ds.experiment.values
    array(['hist-GHG', 'hist-aer', 'historical'], dtype=object)   # sorted by unstack
    """
    # Keep only nodes that hold actual variables. Skip has_data: it's also True for
    # a node that only defines coordinates, and that node would become a spurious entry.
    leaves = [node for node in dt.leaves if node.data_vars]

    # Each leaf's path relative to the root we were given, split into one label per level:
    # 'CanESM5/historical' -> ('CanESM5', 'historical'); in a one-model subtree 'historical' -> ('historical',)
    keys = [tuple(node.relative_to(dt).split("/")) for node in leaves]

    # Every path must have one label per name in dims. A string passed as dims
    # fails here, since len('experiment') == 10.
    if any(len(k) != len(dims) for k in keys):
        raise ValueError(f"Not all leaves are at depth {len(dims)}")

    # Stack every leaf along a temporary dimension, one entry per leaf.
    # join='outer' keeps the union of coordinates, so leaves with different
    # years or members are padded with NaN rather than raising or being trimmed.
    concat_kwargs.setdefault("join", "outer")
    ds = xr.concat([node.to_dataset() for node in leaves], dim="_stacked",
                   **concat_kwargs)

    # Label each entry of _stacked with its path tuple, as a MultiIndex whose
    # levels are named by dims: entry i -> (model=..., experiment=...).
    idx = pd.MultiIndex.from_tuples(keys, names=dims)
    ds = ds.assign_coords(xr.Coordinates.from_pandas_multiindex(idx, "_stacked"))

    # Split the MultiIndex into real dimensions (model, experiment). Combinations
    # the tree doesn't contain (a model missing an experiment) become NaN.
    return ds.unstack("_stacked")


def reduce_to_dataset(dt: xr.DataTree, func, dims=("model", "experiment"), **concat_kwargs) -> xr.Dataset:
    """Apply a member-reducing function to every node and collapse the result into one Dataset.

    Nodes only differ in their number of members, which is why the data sits in
    a tree. Once ``func`` removes ``member``, every node shares the same
    coordinates, so the results stack along ``dims`` instead of staying a tree.
    Missing combinations are NaN. ``concat_kwargs`` go to ``xr.concat``.
    """
    return tree_to_dataset(dt.map_over_datasets(skip_empty(func)), list(dims), **concat_kwargs)


def hist_nat_like(tree: xr.DataTree) -> xr.DataTree:
    """A tree shaped like ``tree`` (/<model>/<experiment>) in which every experiment's node holds its model's hist-nat.

    For mapping a function of an experiment and the hist-nat it is compared
    with over every experiment: ``xr.map_over_datasets(func, tree, hist_nat_like(tree))``.
    """
    return DataTree.from_dict({node.relative_to(tree): node.parent["hist-nat"].to_dataset()
                               for node in tree.leaves if node.data_vars})


def dataset_to_tree(ds: xr.Dataset, like: xr.DataTree, dims=("model", "experiment")) -> xr.DataTree:
    """Split a Dataset back into a tree with the same data nodes as ``like``.

    The inverse of :func:`tree_to_dataset`, for combining a member-free result
    with member-level data, e.g. ``tree - dataset_to_tree(signal, like=tree)``.
    """
    nodes = {}
    for node in like.leaves:
        if node.data_vars:
            path = node.relative_to(like)
            nodes[path] = ds.sel(dict(zip(dims, path.split("/"))), drop=True)
    return DataTree.from_dict(nodes)