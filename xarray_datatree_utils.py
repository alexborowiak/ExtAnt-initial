from xarray import DataTree

def take_level(dt, key, name=None, skip_missing=False):
    """Promote the node named `key` out of every child of `dt`."""
    out = {}
    for child_name, child in dt.items():
        if key in child.children:
            out[child_name] = child[key]
        elif not skip_missing:
            raise KeyError(f"{child_name!r} has no node named {key!r}; "
                           f"its children are {list(child.children)}")
    if not out:
        raise KeyError(f"no child of {dt.path!r} has a node named {key!r}")
    return DataTree.from_dict(out, name=name or key)