"""The SAM steps: opening the index files, the regression, the phases and the synthetic index."""

import numpy as np
import xarray as xr

from extant import config, sam


def test_open_index(tmp_path):
    folder = tmp_path / "hist-nat" / "MODEL"
    folder.mkdir(parents=True)
    times = xr.date_range("1850-01-16", periods=24, freq="MS", calendar="360_day", use_cftime=True)
    for member in ("r1i1p1f1", "r2i1p1f1"):
        ds = xr.Dataset({"SAM": (("time_mm", "level"), np.random.default_rng(0).normal(size=(24, 3))),
                         "P": ("level", [1000.0, 850.0, 500.0])},
                        coords={"time_mm": times, "level": [1000.0, 850.0, 500.0]})
        ds.to_netcdf(folder / f"SAM_zonal_mon_hist-nat_MODEL_{member}_interp.nc")
    files = sam.index_files("hist-nat", "MODEL", root=tmp_path)
    assert [sam.member_of(f) for f in files] == ["r1i1p1f1", "r2i1p1f1"]
    index = sam.open_index(files, level=1).load()
    assert index["sam"].dims == ("member", "time")
    assert list(index["member"].values) == ["r1i1p1f1", "r2i1p1f1"]
    assert index["sam"].attrs["level"] == 850.0


def test_regression_sums_recover_the_slope():
    rng = np.random.default_rng(1)
    index = xr.Dataset({"sam": (("member", "year"), rng.normal(size=(40, 30)))},
                       coords={"member": [f"r{i}" for i in range(40)], "year": np.arange(30)})
    temperature = xr.Dataset({"tas": 2.0 * index["sam"] + 0.1 * rng.normal(size=(40, 30))})
    #(c): One member the index does not have, which is left out
    temperature = xr.concat([temperature, temperature.isel(member=[0]).assign_coords(member=["extra"])], "member")
    sums = sam.regression_sums(temperature, index)
    np.testing.assert_allclose(float(sums["products"] / sums["squares"]), 2.0, atol=0.01)


def test_by_phase():
    index = xr.Dataset({"sam": (("member", "year"), [[1.0, -1.0, 2.0]])}, coords={"member": ["r1"], "year": [1, 2, 3]})
    values = xr.DataArray([[10.0, 20.0, 30.0]], dims=("member", "year"), coords=index.coords)
    split = sam.by_phase(values, index)
    np.testing.assert_allclose(split.sel(phase="positive").values, [[10.0, np.nan, 30.0]])
    np.testing.assert_allclose(split.sel(phase="negative").values, [[np.nan, 20.0, np.nan]])


def test_synthetic_index_matches_the_tree():
    like = xr.DataTree.from_dict({
        f"M/{experiment}": xr.Dataset({"tas": (("member", "year", "season"), np.zeros((3, 60, 4)))},
                                      coords={"member": ["a", "b", "c"], "year": np.arange(1950, 2010),
                                              "season": ["DJF", "JJA", "MAM", "SON"]})
        for experiment in config.MAIN_EXPERIMENTS})
    index = sam.synthetic_index(like, seed=0)
    node = index["M/historical"].to_dataset()
    assert node["sam"].dims == ("member", "year", "season")
    #(c): A summer trend in historical, none in hist-nat
    late = index["M/historical"]["sam"].sel(year=slice(2000, None), season="DJF").mean()
    natural = index["M/hist-nat"]["sam"].sel(year=slice(2000, None), season="DJF").mean()
    assert float(late) > 0.6 > abs(float(natural))


def test_notebook_01_chain(tmp_path):
    #(c): Notebook 01, section 6, on fake files: open, cut where historical ends, seasonal means, year x season
    from extant import datatree, preprocessing as pp
    times = xr.date_range("1850-01-16", "2020-12-16", freq="MS", calendar="360_day", use_cftime=True)
    for experiment in ("historical", "hist-nat"):
        folder = tmp_path / experiment / "MODEL"
        folder.mkdir(parents=True)
        for member in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1"):
            ds = xr.Dataset({"SAM": (("time_mm", "level"), np.random.default_rng(0).normal(size=(times.size, 2)))},
                            coords={"time_mm": times, "level": [1000.0, 500.0]})
            ds.to_netcdf(folder / f"SAM_zonal_mon_{experiment}_MODEL_{member}_interp.nc")
    monthly = xr.DataTree.from_dict({f"MODEL/{experiment}": sam.open_index(sam.index_files(experiment, "MODEL",
                                                                                          root=tmp_path))
                                     for experiment in ("historical", "hist-nat")})
    cut = monthly.sel(time=slice(None, config.LAST_MONTH))
    seasonal = cut.map_over_datasets(pp.seasonal_mean)
    seasons = seasonal.map_over_datasets(datatree.skip_empty(pp.split_time),
                                         kwargs=dict(parts=("year", "season"), dim="time")).compute()
    node = seasons["MODEL/historical"]
    assert node["sam"].dims == ("member", "year", "season")
    #(c): The Jan-Feb stub of 1850 has no December, and DJF 2014 runs into 2015, so 1850-2013 as for temperature
    assert int(node["year"][0]) == 1850 and int(node["year"][-1]) == 2013
