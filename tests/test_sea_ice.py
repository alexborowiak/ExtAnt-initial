"""Checks of sea_ice on fields whose extent and edge are known exactly.

On the 2.5° grid, cell edges fall on multiples of 2.5°, so ice in every cell
south of -65° covers exactly the cap south of -65° (less the land), and the
equivalent latitude comes out at -65.
"""

import numpy as np
import pytest
import xarray as xr

from extant import sea_ice

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
