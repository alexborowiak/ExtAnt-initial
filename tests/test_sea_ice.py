"""Checks of sea_ice on fields whose extent and edge are known exactly.

On the 2.5° grid, cell edges fall on multiples of 2.5°, so ice in every cell
south of -65° covers exactly the cap south of -65° (less the land), and the
equivalent latitude comes out at -65.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from extant import sea_ice  # noqa: E402

LAT = np.arange(-88.75, -40, 2.5)
LON = np.arange(-180.0, 180.0, 2.5)
LAND_EDGE = -75.0
#(t): Ice edge by month: furthest north in JJA, furthest south in DJF
EDGE = {12: -72.5, 1: -72.5, 2: -72.5, 6: -65.0, 7: -65.0, 8: -65.0}
OTHER_EDGE = -70.0


def _field(edge):
    """100% ice south of ``edge``, open water north of it, land (NaN) south of LAND_EDGE."""
    lat = xr.DataArray(LAT, dims="lat", coords={"lat": LAT})
    field = xr.where(lat < edge, 100.0, 0.0).where(lat > LAND_EDGE)
    return field.expand_dims(lon=LON).transpose("lat", "lon")


def test_areas_on_the_sphere():
    whole = sea_ice.cell_area(np.arange(-88.75, 90, 2.5), LON)
    np.testing.assert_allclose(float(whole.sum()), 4 * np.pi * sea_ice.EARTH_RADIUS ** 2, rtol=1e-12)
    south = sea_ice.cell_area(LAT, LON)
    np.testing.assert_allclose(float(south.sum()), sea_ice.cap_area(-40.0), rtol=1e-12)
    assert sea_ice.cap_area(-90.0) == 0 and np.isclose(sea_ice.cap_area(0.0), sea_ice.HEMISPHERE)
    phi = np.array([-90.0, -71.0, -60.0, -40.0, 0.0])
    np.testing.assert_allclose(sea_ice.equivalent_latitude(sea_ice.cap_area(phi)), phi, atol=1e-9)
    #(c): The paper's "about 71°S" with no sea ice: Antarctica (~14 million km²) alone
    assert -72 < sea_ice.equivalent_latitude(14.0e6) < -70


def test_extent_and_edge_of_one_field():
    example = sea_ice.edge_example(_field(-65.0))
    np.testing.assert_allclose(float(example.extent), sea_ice.cap_area(-65.0) - sea_ice.cap_area(LAND_EDGE),
                               rtol=1e-12)
    np.testing.assert_allclose(float(example.continent_area), sea_ice.cap_area(LAND_EDGE), rtol=1e-12)
    np.testing.assert_allclose(float(example.edge_latitude), -65.0, atol=1e-9)
    assert not (example.ice & example.continent).any()

    #(c): Land north of 60°S (an island) is not part of the continent
    island = _field(-65.0).where(~((_field(-65.0).lat == -51.25) & (_field(-65.0).lon == 0.0)))
    assert float(sea_ice.edge_example(island).continent_area) == pytest.approx(float(example.continent_area))


@pytest.fixture(scope="module")
def extents():
    time = xr.date_range("1850-01-01", periods=12 * 6, freq="MS", calendar="360_day", use_cftime=True)
    fields = xr.concat([_field(EDGE.get(month, OTHER_EDGE)) for month in time.month], dim="time")
    siconc = fields.assign_coords(time=time).expand_dims(member=[0, 1]).rename("siconc")
    monthly = xr.DataTree.from_dict({f"M/{experiment}": siconc.to_dataset() for experiment in ("hist-nat", "hist-GHG")})
    return sea_ice.seasonal_extent(monthly)


def test_seasonal_extent(extents):
    node = extents["M/hist-GHG"].to_dataset()
    #(c): Complete seasons in whole years only: the Jan-Feb stub and the last December are dropped
    assert list(node.year.values) == list(range(1850, 1855)) and node.extent.dims == ("member", "year", "season")
    np.testing.assert_allclose(float(node.continent_area), sea_ice.cap_area(LAND_EDGE), rtol=1e-6)
    edges = sea_ice.edge_latitude(extents, years=3)
    assert edges.dims == ("model", "experiment", "season")
    expected = {"DJF": -72.5, "JJA": -65.0, "MAM": OTHER_EDGE, "SON": OTHER_EDGE}
    for season, latitude in expected.items():
        np.testing.assert_allclose(edges.sel(model="M", season=season).values, latitude, atol=1e-4)


def test_summaries(extents):
    summary = sea_ice.extent_summary(extents)
    assert set(summary.data_vars) == {"mean", "low", "high", "continent_area"}
    assert (summary["low"] <= summary["mean"]).all() and (summary["mean"] <= summary["high"]).all()

    seasonal = xr.DataTree.from_dict({
        "M/hist-nat": xr.Dataset({"siconc": _field(-65.0).expand_dims(member=[0, 1], year=np.arange(1850, 1860),
                                                                       season=["DJF", "JJA"])})})
    climatology = sea_ice.concentration_climatology(seasonal, years=5)
    assert climatology.dims == ("model", "experiment", "season", "lat", "lon")
    np.testing.assert_allclose(climatology.sel(lat=-66.25).values, 100.0)


# ---------------------------------------------------------------------------
# Onto the 2.5° grid
# ---------------------------------------------------------------------------

def test_box_index():
    lat = xr.DataArray([-63.0, -88.75, -41.25, -38.0, np.nan, 1e30, -88.8], dims="point")
    lon = xr.DataArray([2.0, 0.0, 178.75, 0.0, 0.0, 0.0, 0.0], dims="point")
    boxes = sea_ice.box_index(lat, lon)
    n_lon = sea_ice.GRID_LON.size
    #(c): The docstring's example; then the edges, closed on the south and west: the first row, the last row and the
    #(c): last column (178.75°E is the 180° cell's west edge, so it goes to the cell centred on 180°W, column 0)
    assert int(boxes[0]) == 10 * n_lon + 73
    assert int(boxes[1]) == 0 * n_lon + 72
    assert int(boxes[2]) == 19 * n_lon + 0
    #(c): North of the grid, no position, a fill value, south of the first row's edge
    assert (boxes[3:].values == -1).all()
    #(c): 0-360 and -180-180 longitudes give the same cells
    assert int(sea_ice.box_index(xr.DataArray(-63.0), xr.DataArray(-178.0))) == \
        int(sea_ice.box_index(xr.DataArray(-63.0), xr.DataArray(182.0)))


def _fine_field():
    """A 0.25° field (points on multiples of 0.25°, as ERA5's) south of 40°S: 80% south of 61.25°S (the edge of a
    2.5° cell), 0% north of it, land south of 75°S and in the western half of the cell centred on (62.5°S, 10°E)."""
    lat = np.arange(-40.0, -90.01, -0.25)
    lon = np.arange(0.0, 360.0, 0.25)
    lat_da = xr.DataArray(lat, dims="lat", coords={"lat": lat})
    lon_da = xr.DataArray(lon, dims="lon", coords={"lon": lon})
    field = xr.where(lat_da < -61.25, 80.0, 0.0).broadcast_like(lon_da).transpose("lat", "lon")
    half_land = (lat_da < -61.25) & (lat_da >= -63.75) & (lon_da >= 8.75) & (lon_da < 10.0)
    return field.where(~((lat_da < -75) | half_land))


def test_to_grid_and_ocean_area():
    fine = _fine_field()
    area = sea_ice.cell_area(fine["lat"], fine["lon"])
    boxes = sea_ice.box_index(fine["lat"], fine["lon"])
    coarse = sea_ice.to_grid(fine, boxes, area)
    assert coarse.dims == ("lat", "lon") and list(coarse.lat.values) == list(sea_ice.GRID_LAT)
    #(c): The point on 61.25°S goes to the cell north of it (cells are closed on the south side), so that cell is 0%
    np.testing.assert_allclose(coarse.sel(lat=-65.0).values, 80.0)
    np.testing.assert_allclose(coarse.sel(lat=slice(-60.0, None)).values, 0.0)
    #(c): All land: missing; half land: the mean over its ocean
    assert coarse.sel(lat=-80.0).isnull().all()
    assert float(coarse.sel(lat=-62.5, lon=10.0)) == pytest.approx(80.0)

    ocean = sea_ice.ocean_area_on_grid(fine.notnull(), boxes, area)
    #(c): The ocean areas add up to the fine field's, and the half-land cell has half its area
    assert float(ocean.sum()) == pytest.approx(float(area.where(fine.notnull()).sum()), rel=1e-9)
    whole = sea_ice.cell_area(ocean["lat"], ocean["lon"])
    assert float(ocean.sel(lat=-62.5, lon=10.0)) == pytest.approx(0.5 * float(whole.sel(lat=-62.5, lon=10.0)),
                                                                   rel=0.02)

    #(c): On a curvilinear grid (2-D latitudes and longitudes), the same answer
    lat2d, lon2d = xr.broadcast(fine["lat"], fine["lon"])
    curvilinear = fine.rename(lat="j", lon="i").drop_vars(["j", "i"])
    boxes2d = sea_ice.box_index(lat2d.rename(lat="j", lon="i").drop_vars(["j", "i"]),
                                lon2d.rename(lat="j", lon="i").drop_vars(["j", "i"]))
    coarse2d = sea_ice.to_grid(curvilinear.expand_dims(time=2), boxes2d,
                               area.rename(lat="j", lon="i").drop_vars(["j", "i"]))
    xr.testing.assert_allclose(coarse2d.isel(time=1, drop=True), coarse)


def test_extent_with_ocean_area():
    fine = _fine_field()
    area = sea_ice.cell_area(fine["lat"], fine["lon"])
    boxes = sea_ice.box_index(fine["lat"], fine["lon"])
    ds = xr.Dataset({"siconc": sea_ice.to_grid(fine, boxes, area).expand_dims(time=[np.datetime64("2000-01-01")]),
                     "ocean_area": sea_ice.ocean_area_on_grid(fine.notnull(), boxes, area)})
    result = sea_ice.monthly_extent(ds)
    #(c): The ice edge is on a cell edge, so the 2.5° extent is the fine field's (a 2.5° cell the 15% line cut
    #(c): through would count all of its ocean)
    fine_extent = float(sea_ice.extent(fine, area))
    assert float(result["extent"].isel(time=0)) == pytest.approx(fine_extent, rel=1e-9)
    #(c): Within 3%: points on a cell edge (ERA5's) reach half a point into the next cell, so the cells round the
    #(c): continent's edge get a 0.125° strip too much or too little land (here about 0.2 million km², 0.1° of latitude)
    fine_continent = float(area.where(fine.isnull() & (fine["lat"] <= -60)).sum())
    assert float(result["continent_area"]) == pytest.approx(fine_continent, rel=0.03)
    #(c): Without the ocean areas, the half-land cell counts whole
    whole = sea_ice.monthly_extent(ds[["siconc"]])
    assert float(whole["extent"].isel(time=0)) > float(result["extent"].isel(time=0))


def _member_file(path, values, n_months=24, start="2013-01-16"):
    """A member's file on a small curvilinear (j, i) grid, as the CMIP files are."""
    lat = np.array([[-70.0, -70.0], [-62.0, -62.0]])
    lon = np.array([[0.0, 3.0], [0.0, 3.0]])
    time = xr.date_range(start, periods=n_months, freq="MS", calendar="noleap", use_cftime=True)
    ds = xr.Dataset({"siconc": (("time", "j", "i"), np.broadcast_to(values, (n_months, 2, 2)).astype("float32"))},
                    coords={"time": time, "j": [0, 1], "i": [0, 1], "latitude": (("j", "i"), lat),
                            "longitude": (("j", "i"), lon)})
    ds.to_netcdf(path)
    return path


def test_open_on_grid(tmp_path):
    values = np.array([[100.0, np.nan], [50.0, 0.0]])
    files = {"r1i1p1f1": [_member_file(tmp_path / "a.nc", values)], "r2i1p1f1": [_member_file(tmp_path / "b.nc", values)]}
    area = xr.Dataset({"areacello": (("j", "i"), np.full((2, 2), 1e9))},
                      coords={"j": [0, 1], "i": [0, 1], "latitude": (("j", "i"), [[-70.0, -70.0], [-62.0, -62.0]]),
                              "longitude": (("j", "i"), [[0.0, 3.0], [0.0, 3.0]])})
    area.to_netcdf(tmp_path / "area.nc")

    ds = sea_ice.open_on_grid(files, tmp_path / "area.nc", "siconc").load()
    assert list(ds.member.values) == ["r1i1p1f1", "r2i1p1f1"]
    assert ds["siconc"].dims == ("member", "time", "lat", "lon")
    assert float(ds["siconc"].isel(member=0, time=0).sel(lat=-70.0, lon=0.0)) == 100.0
    #(c): The point with no value is land: its cell is missing and has no ocean (areas in km²)
    assert ds["siconc"].isel(member=0, time=0).sel(lat=-70.0, lon=2.5).isnull()
    assert float(ds["ocean_area"].sum()) == pytest.approx(3 * 1e3)

    #(c): Members whose records differ in length are refused
    files["r3i1p1f1"] = [_member_file(tmp_path / "c.nc", values, n_months=12)]
    with pytest.raises(ValueError, match="differ in length"):
        sea_ice.open_on_grid(files, tmp_path / "area.nc", "siconc")


# ---------------------------------------------------------------------------
# Against ERA5, and where the ice changed
# ---------------------------------------------------------------------------

def test_era5_extent_test():
    rng = np.random.default_rng(3)
    years = np.arange(1979, 2014)
    members = xr.DataArray(10e6 + 0.5e6 * rng.standard_normal((30, years.size, 2)), dims=("member", "year", "season"),
                           coords={"year": years, "season": ["DJF", "JJA"]})
    #(c): ERA5 like a member in DJF, 3 million km² less ice in JJA
    era5 = members.isel(member=0, drop=True) + xr.DataArray([0.0, -3e6], dims="season")
    result = sea_ice.era5_extent_test(xr.Dataset({"extent": members}), era5, years=slice(1979, 2013))
    assert set(result.statistic.values) == set(sea_ice.ERA5_DIAGNOSTICS)
    assert int(result["verdict"].sel(statistic="mean", season="JJA")) == -1
    assert int(result["verdict"].sel(statistic="mean", season="DJF")) == 0
    assert bool(result["testable"].sel(statistic="mean", season="DJF"))
    assert str(result["units"].sel(statistic="trend").values) == "million km² per decade"
    #(c): With five members nothing can be flagged
    few = sea_ice.era5_extent_test(xr.Dataset({"extent": members.isel(member=slice(0, 5))}), era5)
    assert not few["testable"].any()


def test_zones():
    lat = xr.DataArray(sea_ice.GRID_LAT, dims="lat", coords={"lat": sea_ice.GRID_LAT})
    lon = xr.DataArray(sea_ice.GRID_LON, dims="lon", coords={"lon": sea_ice.GRID_LON})
    #(c): hist-nat: ice south of 60°S; the experiment's edge at 65°S, and ice gained at one longitude; land south of 75°S
    hist_nat = xr.where(lat < -60, 90.0, 0.0).broadcast_like(lon).where(lat > -75)
    experiment = xr.where(lat < -65, 90.0, 0.0).broadcast_like(lon).where(lat > -75)
    experiment = xr.where((lat == -55.0) & (lon == 0.0), 50.0, experiment)
    zones = sea_ice.ice_zones(hist_nat, experiment)
    named = {name: zones == k for k, name in enumerate(sea_ice.ZONES)}
    assert bool(named["ice lost"].sel(lat=-62.5).all()) and bool(named["ice in both"].sel(lat=-70.0).all())
    assert bool(named["ice gained"].sel(lat=-55.0, lon=0.0)) and bool(named["open water"].sel(lat=-45.0).all())
    assert bool(named["continent"].sel(lat=-80.0).all())

    #(c): A field equal to its zone's number has that mean in each zone; areas add up to the whole grid
    means = sea_ice.zone_means(zones.astype(float), zones)
    np.testing.assert_allclose(means.values, np.arange(len(sea_ice.ZONES)))
    areas = sea_ice.zone_areas(zones)
    whole = sea_ice.cell_area(lat, lon)
    assert float(areas.sum()) == pytest.approx(float(whole.sum()) / 1e6)

    #(c): No data at all for one of the two: no zones
    assert sea_ice.ice_zones(hist_nat * np.nan, experiment).isnull().all()


def test_edges_on_map_grids():
    pytest.importorskip("cartopy")
    from extant.plots import forced_response as fr, sea_ice as sip

    lat = xr.DataArray(sea_ice.GRID_LAT, dims="lat", coords={"lat": sea_ice.GRID_LAT})
    lon = xr.DataArray(sea_ice.GRID_LON, dims="lon", coords={"lon": sea_ice.GRID_LON})
    field = xr.where(lat < -60, 90.0, 0.0).broadcast_like(lon).where(lat > -75)
    ice = xr.concat([field, field.shift(lat=-2)], dim=pd.Index(["hist-nat", "hist-GHG"], name="experiment"))
    ice = ice.expand_dims(model=["A", "B"], season=["DJF", "JJA"])

    one = sip.edge_pair(ice, "hist-GHG")
    assert list(one["edge"].values) == ["hist-nat", "experiment"] and "experiment" not in one.dims
    several = sip.edge_pair(ice.sel(model="A"), ["hist-GHG", "hist-aer"])
    assert several.sizes["experiment"] == 2 and several.sel(experiment="hist-aer", edge="experiment").isnull().all()
    assert sip.edge_pair(None, "hist-GHG") is None

    change = (ice.sel(experiment="hist-GHG") - ice.sel(experiment="hist-nat")).fillna(0)
    scale = fr.Scale(np.arange(-55, 56, 10), "RdBu", "Change (%)")
    panels = fr.map_grid(change, scale, row_dim="model", col_dim="season", edges=one)
    labels = [text.get_text() for legend in panels.fig.legends for text in legend.get_texts()]
    assert labels == [sip.EDGES["hist-nat"]["label"], sip.EDGES["experiment"]["label"]]
    fr.count_grid(xr.zeros_like(change.sel(model="A")).expand_dims(direction=["increase", "decrease"]), 2, "t", "n",
                  above=[fr.Row(change.sel(model="A"), scale, "median", edges=one.sel(model="A"))], col_dim="season",
                  edges=one.sel(model="A"))
    plt.close("all")
