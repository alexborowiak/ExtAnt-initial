"""The SAM steps: opening the index files, the phases and the synthetic index."""

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
    #(c): Listed as notebook calc_01 lists them: /<experiment>/<model>/*.nc
    files = sorted((tmp_path / "hist-nat" / "MODEL").glob("*.nc"))
    assert [sam.member_of(f) for f in files] == ["r1i1p1f1", "r2i1p1f1"]
    index = sam.open_index(files, level=1).load()
    assert index["sam"].dims == ("member", "time")
    assert list(index["member"].values) == ["r1i1p1f1", "r2i1p1f1"]
    assert index["sam"].attrs["level"] == 850.0


def test_phase_masks_cut_at_the_thresholds():
    index = xr.Dataset({"sam": (("member", "year", "season"), [[[-2.0], [-1.0], [0.0], [1.0], [2.0], [np.nan]]])},
                       coords={"member": ["r1"], "year": np.arange(6), "season": ["DJF"]})
    thresholds = xr.Dataset({"sam": (("quantile", "season"), [[-1.0], [1.0]])},
                            coords={"quantile": list(sam.TERCILES), "season": ["DJF"]})
    masks = sam.phase_masks(index, thresholds)["in_phase"].sel(member="r1", season="DJF")
    assert list(masks["phase"].values) == list(sam.PHASES)
    #(c): The boundaries themselves are neutral; a season with no index is in no phase
    np.testing.assert_array_equal(masks.sel(phase="negative"), [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(masks.sel(phase="neutral"), [0, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(masks.sel(phase="positive"), [0, 0, 0, 0, 1, 0])


def test_by_phase_keeps_each_phase_s_seasons_and_the_years_both_have():
    coords = {"member": ["r1"], "year": [1, 2, 3], "season": ["DJF"]}
    index = xr.Dataset({"sam": (("member", "year", "season"), [[[1.0], [-1.0], [0.0]]])}, coords=coords)
    thresholds = xr.Dataset({"sam": (("quantile", "season"), [[-0.5], [0.5]])},
                            coords={"quantile": list(sam.TERCILES), "season": ["DJF"]})
    masks = sam.phase_masks(index, thresholds)
    #(c): Values over years 2 and 3 only, as the final years would be
    values = xr.DataArray([[[20.0], [30.0]]], dims=("member", "year", "season"),
                          coords={"member": ["r1"], "year": [2, 3], "season": ["DJF"]})
    split = sam.by_phase(values, masks).sel(member="r1", season="DJF")
    assert list(split["year"].values) == [2, 3]
    np.testing.assert_array_equal(split.sel(phase="negative"), [20.0, np.nan])
    np.testing.assert_array_equal(split.sel(phase="neutral"), [np.nan, 30.0])
    assert split.sel(phase="positive").isnull().all()


def test_hist_nat_terciles_put_a_third_in_each_phase():
    from functools import partial
    from extant import quantiles as qc
    rng = np.random.default_rng(3)
    index = xr.Dataset({"sam": (("member", "year", "season"), rng.normal(size=(30, 21, 4)))},
                       coords={"member": np.arange(30), "year": np.arange(21), "season": ["DJF", "JJA", "MAM", "SON"]})
    thresholds = partial(qc.pooled_quantiles, quantiles=sam.TERCILES)(index)
    shares = sam.phase_masks(index, thresholds)["in_phase"].mean(["member", "year"])
    np.testing.assert_allclose(shares, 1 / 3, atol=0.01)


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
    #(c): Notebook calc_01, section 6, on fake files: open, cut where historical ends, seasonal means, year x season
    from extant import datatree, preprocessing as pp
    times = xr.date_range("1850-01-16", "2020-12-16", freq="MS", calendar="360_day", use_cftime=True)
    for experiment in ("historical", "hist-nat"):
        folder = tmp_path / experiment / "MODEL"
        folder.mkdir(parents=True)
        for member in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1"):
            ds = xr.Dataset({"SAM": (("time_mm", "level"), np.random.default_rng(0).normal(size=(times.size, 2)))},
                            coords={"time_mm": times, "level": [1000.0, 500.0]})
            ds.to_netcdf(folder / f"SAM_zonal_mon_{experiment}_MODEL_{member}_interp.nc")
    monthly = xr.DataTree.from_dict({
        f"MODEL/{experiment}": sam.open_index(sorted((tmp_path / experiment / "MODEL").glob("*.nc")))
        for experiment in ("historical", "hist-nat")
    })
    cut = monthly.sel(time=slice(None, config.LAST_MONTH))
    seasonal = cut.map_over_datasets(pp.seasonal_mean)
    seasons = seasonal.map_over_datasets(datatree.skip_empty(pp.split_time),
                                         kwargs=dict(parts=("year", "season"), dim="time")).compute()
    node = seasons["MODEL/historical"]
    assert node["sam"].dims == ("member", "year", "season")
    #(c): The Jan-Feb stub of 1850 has no December, and DJF 2014 runs into 2015, so 1850-2013 as for temperature
    assert int(node["year"][0]) == 1850 and int(node["year"][-1]) == 2013
