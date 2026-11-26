"""Signal-to-noise on series with a known answer, and the rolling noise against the slow version."""

import numpy as np
import pytest
import xarray as xr

from extant.statistics import emergence

RNG = np.random.default_rng(5)


def test_self_signal_to_noise_of_a_known_trend():
    years = np.arange(1979, 2014)
    noise = RNG.standard_normal(years.size)
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    series = xr.DataArray(0.1 * (years - years[0]) + noise, dims="year", coords={"year": years})
    # Signal ~ 0.1 x 34 years = 3.4 over unit noise.
    np.testing.assert_allclose(float(emergence.self_signal_to_noise(series)), 3.4, rtol=0.15)


def _field(n_members=12, n_years=40, nan_fraction=0.05):
    """(member, year, season, lat) noise with scattered NaN and one all-NaN point."""
    coords = {"member": np.arange(n_members), "year": np.arange(n_years), "season": ["DJF", "JJA"],
              "lat": [-80.0, -70.0, -60.0]}
    values = RNG.standard_normal((n_members, n_years, 2, 3))
    values[RNG.random(values.shape) < nan_fraction] = np.nan
    values[:, :, 0, 2] = np.nan
    return xr.DataArray(values, dims=tuple(coords), coords=coords)


@pytest.mark.parametrize("dask", [False, True])
def test_rolling_std_matches_the_construct_version(dask):
    """Running sums give the window-copy numbers: ddof=0, NaN skipped, windows shortened at the ends."""
    da = _field(n_years=40) * 3 + 20
    expected = da.rolling(year=11, center=True).construct("window").std(["window", "member"])
    result = emergence.rolling_std(da.chunk({"lat": 1}) if dask else da, window=11).compute()
    assert result.dims == expected.dims
    np.testing.assert_allclose(result.values, expected.values, rtol=1e-9, atol=1e-9)
    # On a Dataset, as reduce_to_dataset passes it
    ds_result = emergence.rolling_std(da.to_dataset(name="tas"), window=11)
    np.testing.assert_allclose(ds_result.tas.values, expected.values, rtol=1e-9, atol=1e-9)
