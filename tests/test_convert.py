"""The JASMIN-only path of notebook 01, on fake raw files: member files -> monthly stores -> seasonal means -> saved.

Built from the sample, laid out as on JASMIN (one netCDF file per member under
<raw>/<experiment>/<model>/), and run as notebook 01 writes it out: the files
listed, opened together (convert.open_members), saved with to_zarr, the stores
listed and opened into a DataTree. Everything but the Dask Gateway cluster and
the xesmf regridding runs as it does there.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extant import config, convert, preprocessing as pp

pytest.importorskip("zarr")

#(t): The sample in this repository: <model>_<experiment>_sample.nc (monthly tas in K) and era5_sample.nc
SAMPLE_DIR = Path(__file__).resolve().parent / "data"
MODELS = ["CanESM5", "HadGEM3-GC31-LL"]
EXPERIMENTS = ["hist-nat", "historical", "hist-GHG"]


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    """Fake raw files, 12 members of each model and experiment: <raw>/<experiment>/<model>/*_interp.nc."""
    raw = tmp_path_factory.mktemp("jasmin") / "raw"
    for model in MODELS:
        for experiment in EXPERIMENTS:
            ds = xr.open_dataset(SAMPLE_DIR / f"{model}_{experiment}_sample.nc").load()
            folder = raw / experiment / model
            folder.mkdir(parents=True)
            for k, member in enumerate(ds.member.values[:12]):
                one = ds.sel(member=member).drop_vars("member")
                one.to_netcdf(folder / f"tas_mon_{experiment}_{model}_r{k + 1}i1p1f1_interp.nc")
    return raw


def test_member_labels():
    assert convert.member_from_path("/a/tas_mon_historical_GISS-E2-1-G_r9i1p5f1_interp.nc") == "r9i1p5f1"
    assert convert.member_from_path("/a/tas_mon_historical_CESM2_1001.001_interp.nc") == "1001.001"
    assert convert.member_from_path("/a/009/tas_interp.nc") == "009"


def test_raw_files_to_seasonal_means(raw_dir, tmp_path):
    monthly_dir, scratch_dir = tmp_path / "monthly", tmp_path / "scratch"

    #(c): Notebook 01, section 1: each model's member files for each experiment, opened together and saved
    for model in MODELS:
        for experiment in EXPERIMENTS:
            member_paths = sorted((raw_dir / experiment / model).rglob(f"*_{config.GROUP}.nc"))
            assert len(member_paths) == 12
            store_path = monthly_dir / f"{model}_tas_{experiment}_{config.GROUP}_monthly.zarr"
            convert.open_members(member_paths, "tas").drop_encoding().to_zarr(store_path, mode="w", consolidated=True)
    monthly_ds = xr.open_zarr(monthly_dir / f"CanESM5_tas_historical_{config.GROUP}_monthly.zarr")
    assert monthly_ds.sizes["member"] == 12 and set(monthly_ds.member.values) == {f"r{k}i1p1f1" for k in range(1, 13)}
    assert float(monthly_ds.lat.max()) <= config.LAT_MAX

    #(c): Section 3: the stores listed, the model and experiment read from each name, opened into a DataTree
    store_paths = sorted(monthly_dir.glob(f"*_tas_*_{config.GROUP}_monthly.zarr"))
    names = [path.name.removesuffix("_monthly.zarr").rsplit("_", 3) for path in store_paths]
    assert sorted({model for model, _, _, _ in names}) == MODELS
    lesfmip_tree = xr.DataTree.from_dict({f"{model}/{experiment}": xr.open_zarr(path, consolidated=True)
                                          for path, (model, _, experiment, _) in zip(store_paths, names)})
    assert set(lesfmip_tree.children) == set(MODELS) and set(lesfmip_tree["CanESM5"].children) == set(EXPERIMENTS)

    #(c): The seasonal means, saved with to_zarr and opened again
    season_tree = pp.seasonal_tree(lesfmip_tree, "tas")
    era5_da = pp.seasonal_era5(xr.open_dataarray(SAMPLE_DIR / "era5_sample.nc").load(), "tas", like=season_tree)
    season_tree.to_zarr(scratch_dir / "lesfmip_tas.zarr", mode="w")
    era5_da.to_dataset(name="tas").to_zarr(scratch_dir / "era5_tas.zarr", mode="w")

    reopened_tree = xr.open_datatree(scratch_dir / "lesfmip_tas.zarr", engine="zarr")
    node_da = reopened_tree["CanESM5/historical"].tas
    assert set(node_da.dims) == {"member", "year", "season", "lat", "lon"}
    assert int(node_da.year.max()) == 2013 and -80 < float(node_da.mean()) < 0
    xr.testing.assert_allclose(node_da.load(), season_tree["CanESM5/historical"].tas.load())
    xr.testing.assert_allclose(xr.open_zarr(scratch_dir / "era5_tas.zarr")["tas"].load(), era5_da.load())


def test_era5_month_mean(tmp_path):
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
