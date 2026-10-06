"""The JASMIN-only path, on fake raw files: member files -> monthly stores -> seasonal means -> saved and reopened.

Built from the sample, laid out as on JASMIN (one netCDF file per member under
LESFMIP_RAW/<table>/<variable>/<experiment>/<model>/), so everything but the Dask
Gateway cluster and the xesmf regridding runs as it does there.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extant import config, convert, loading, paths, preprocessing as pp, storage

pytest.importorskip("zarr")

MODELS = ["CanESM5", "HadGEM3-GC31-LL"]
EXPERIMENTS = ["hist-nat", "historical", "hist-GHG"]


@pytest.fixture(scope="module")
def jasmin(tmp_path_factory):
    """Fake raw files and separate scratch and data roots."""
    root = tmp_path_factory.mktemp("jasmin")
    sample = loading.open_lesfmip_sample(models=MODELS, experiments=EXPERIMENTS)
    for node in sample.leaves:
        folder = root / "raw" / "Amon" / "tas" / node.name / node.parent.name
        folder.mkdir(parents=True)
        ds = node.to_dataset()
        for k, member in enumerate(ds.member.values[:12]):
            label = f"r{k + 1}i1p1f1"
            one = ds.sel(member=member).drop_vars("member")
            one.to_netcdf(folder / f"tas_mon_{node.name}_{node.parent.name}_{label}_interp.nc")
    (root / "raw" / "Amon" / "tas" / "reanalysis").mkdir()
    return root


@pytest.fixture
def roots(jasmin, monkeypatch):
    monkeypatch.setattr(paths, "LESFMIP_RAW", jasmin / "raw")
    monkeypatch.setattr(paths, "SCRATCH", jasmin / "scratch")
    monkeypatch.setattr(paths, "DATA_DIR", jasmin / "data")
    return jasmin


def test_member_labels():
    assert convert.member_from_path("/a/tas_mon_historical_GISS-E2-1-G_r9i1p5f1_interp.nc") == "r9i1p5f1"
    assert convert.member_from_path("/a/tas_mon_historical_CESM2_1001.001_interp.nc") == "1001.001"
    assert convert.member_from_path("/a/009/tas_interp.nc") == "009"


def test_raw_files_to_seasonal_means(roots):
    table = convert.availability("tas")
    assert set(table.columns) == set(EXPERIMENTS) and table.all().all()   # 'reanalysis' is not an experiment

    failures = convert.convert_lesfmip("tas", MODELS, EXPERIMENTS)
    assert failures == {}
    store = paths.find(paths.lesfmip_monthly("CanESM5", "tas", "historical", "interp"))
    assert store.parent.parent.parent.parent == roots / "scratch"
    monthly = xr.open_zarr(store)
    assert monthly.sizes["member"] == 12 and set(monthly.member.values) == {f"r{k}i1p1f1" for k in range(1, 13)}
    assert float(monthly.lat.max()) <= config.LAT_MAX

    #(c): A second run skips what exists
    assert convert.convert_experiment("tas", "CanESM5", "historical") == {}

    stores = loading.store_table("tas")
    assert stores.loc["CanESM5"].sum() == len(EXPERIMENTS)
    models = loading.models_with_most_experiments(stores, 1)
    tree = loading.open_lesfmip_monthly("tas", models=models, experiments=EXPERIMENTS, groups=["interp"])
    assert set(tree.children) == set(models) and set(tree[models[0]].children) == set(EXPERIMENTS)

    seasons = pp.seasonal_tree(tree, "tas")
    era5 = pp.seasonal_era5(loading.open_era5_sample(), "tas", like=seasons)
    storage.save(seasons, paths.seasonal("lesfmip", "tas"))
    storage.save(era5, paths.seasonal("era5", "tas"))

    reopened = loading.open_seasonal("tas", experiments=["historical"])
    assert [node.name for node in reopened.leaves] == ["historical"]
    node = reopened[f"{models[0]}/historical"].tas
    assert set(node.dims) == {"member", "year", "season", "lat", "lon"}
    assert int(node.year.max()) == 2013 and -80 < float(node.mean()) < 0
    xr.testing.assert_allclose(loading.open_era5_seasonal("tas").load(), era5.load())


def test_era5_month_mean_and_combining_years(roots, tmp_path):
    lat, lon = np.linspace(-90, -30, 25), np.arange(0, 360, 30.0)
    files = []
    for hour in range(3):
        values = np.full((1, lat.size, lon.size), 250.0 + hour)
        ds = xr.Dataset({"t2m": (("time", "latitude", "longitude"), values)},
                        coords={"time": [np.datetime64("2000-01-01") + np.timedelta64(hour, "h")],
                                "latitude": lat[::-1], "longitude": lon})
        files.append(tmp_path / f"{hour}.2t.nc")
        ds.to_netcdf(files[-1])
    mean = convert.era5_month_mean(files, "t2m")
    assert mean.dtype == np.float32 and mean.shape == ((lat <= -38).sum(), lon.size)
    np.testing.assert_allclose(mean, 251.0)

    #(c): Two year stores, as era5_monthly leaves them, combined into one checked store
    for year in (2001, 2000):
        months = xr.DataArray(np.full((12, 2, 2), float(year)), dims=("time", "lat", "lon"),
                              coords={"time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
                                      "lat": [-80.0, -77.5], "lon": [0.0, 2.5]}, name="tas")
        storage.save(months, paths.era5_years_dir("tas") / f"{year}.zarr")
    combined = convert.combine_era5_years("tas")
    assert combined.sizes["time"] == 24 and combined.indexes["time"].is_monotonic_increasing
    assert paths.find_all(paths.era5_years_dir("tas"), "*.zarr") == []
    assert loading.open_era5_monthly("tas").sizes["time"] == 24
