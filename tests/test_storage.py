"""Where outputs are written and read (paths), saving and reopening (storage), and the package zip for the cluster."""

import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from extant import jasmin, paths, storage

pytest.importorskip("zarr")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Separate scratch and data roots, as on JASMIN."""
    scratch, data = tmp_path / "scratch", tmp_path / "data"
    monkeypatch.setattr(paths, "SCRATCH", scratch)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    return scratch, data


def _tree():
    rng = np.random.default_rng(0)
    nodes = {f"{model}/{experiment}": xr.Dataset({"tas": (("member", "year"), rng.standard_normal((n, 5)))},
                                                 coords={"year": np.arange(2000, 2005)})
             for model, experiment, n in [("A", "hist-nat", 3), ("A", "historical", 4), ("B", "historical", 2)]}
    return xr.DataTree.from_dict(nodes).chunk({"member": -1, "year": 2})


def test_written_to_scratch_and_found_there(roots):
    scratch, _ = roots
    relative = paths.result("width", "tas", "full")
    written = storage.save(xr.Dataset({"w": ("x", np.arange(3.0))}), relative)
    assert written == scratch / relative and not (written.parent / (written.name + ".part")).exists()
    assert paths.find(relative) == written


def test_data_dir_is_read_first(roots):
    scratch, data = roots
    relative = paths.seasonal("era5", "tas")
    storage.save(xr.DataArray(np.zeros(3), dims="x", name="tas"), relative)
    #(c): Move it to DATA_DIR by hand, then write a newer copy to scratch
    (data / relative).parent.mkdir(parents=True)
    (scratch / relative).rename(data / relative)
    storage.save(xr.DataArray(np.ones(3), dims="x", name="tas"), relative)
    assert paths.find(relative) == data / relative
    assert float(storage.open_dataarray(relative).sum()) == 0


def test_missing_store_names_both_roots(roots):
    with pytest.raises(FileNotFoundError, match="neither"):
        paths.find(paths.result("nothing", "tas", "full"))
    assert not storage.exists(paths.result("nothing", "tas", "full"))


def test_tree_round_trip_with_ragged_members(roots):
    tree = _tree()
    storage.save(tree, paths.seasonal("lesfmip", "tas"))
    back = storage.open_tree(paths.seasonal("lesfmip", "tas"), load=True)
    for node in tree.leaves:
        xr.testing.assert_allclose(back[node.path].to_dataset(), node.to_dataset().load())


def test_uneven_chunks_are_evened_out(roots):
    da = xr.DataArray(np.arange(10.0), dims="x", name="v").chunk({"x": (3, 3, 4)})
    storage.save(da, paths.result("v", "tas", "sample"))
    np.testing.assert_array_equal(storage.open_dataarray(paths.result("v", "tas", "sample")).values, da.values)


def test_unnamed_dataarray_is_refused(roots):
    with pytest.raises(ValueError, match="name"):
        storage.save(xr.DataArray(np.zeros(2), dims="x"), paths.result("x", "tas", "sample"))


def test_find_all_prefers_data_dir(roots):
    scratch, data = roots
    for root, names in [(scratch, ["a.zarr", "b.zarr"]), (data, ["b.zarr", "c.zarr"])]:
        for name in names:
            (root / "monthly" / name).mkdir(parents=True)
    found = paths.find_all("monthly", "*.zarr")
    assert [p.name for p in found] == ["a.zarr", "b.zarr", "c.zarr"]
    assert found[1].parent.parent == data


def test_package_zip_imports_on_its_own(tmp_path):
    """What the workers get: the package as a zip, importable with nothing else on the path."""
    archive = jasmin.package_zip(tmp_path)
    check = (f"import sys; sys.path.insert(0, {str(archive)!r}); "
             "from extant import quantiles, significance, convert; print(quantiles.__file__)")
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert archive.name in out.stdout
