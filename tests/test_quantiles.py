"""The fast quantile and smoothing kernels give the numbers of the slow reference implementations."""

import numpy as np
import pytest
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

from extant import quantiles as qc
from extant.stats import nan_quantile

RNG = np.random.default_rng(5)


def _field(n_members=12, n_years=40, nan_fraction=0.05):
    """(member, year, season, lat) noise with scattered NaN and one all-NaN point."""
    coords = {"member": np.arange(n_members), "year": np.arange(n_years), "season": ["DJF", "JJA"],
              "lat": [-80.0, -70.0, -60.0]}
    values = RNG.standard_normal((n_members, n_years, 2, 3))
    values[RNG.random(values.shape) < nan_fraction] = np.nan
    values[:, :, 0, 2] = np.nan
    return xr.DataArray(values, dims=tuple(coords), coords=coords)


@pytest.mark.parametrize("dask", [False, True])
def test_nan_quantile_over_several_dims_is_numpy_exactly(dask):
    da = _field()
    expected = da.quantile([0.05, 0.5, 0.95], dim=("member", "year"), skipna=True)
    result = nan_quantile(da.chunk({"lat": 1}) if dask else da, [0.05, 0.5, 0.95], ("member", "year"))
    np.testing.assert_array_equal(result.transpose(*expected.dims).values, expected.values)


def test_quantile_range_is_numpy_exactly():
    da = _field()
    q = da.quantile([0.05, 0.95], dim=("member", "year"), skipna=True)
    np.testing.assert_array_equal(qc.quantile_range(da).values, (q.sel(quantile=0.95) - q.sel(quantile=0.05)).values)


def _rolling_reference(da, window, quantiles):
    """Every window's pooled (year, member) values through np.quantile."""
    values = da.transpose("season", "lat", "year", "member").values
    half = window // 2
    out = np.full(values.shape[:2] + (len(quantiles), values.shape[2]), np.nan)
    for t in range(half, values.shape[2] - half):
        pool = values[:, :, t - half:t + half + 1].reshape(*values.shape[:2], -1)
        for i, j in np.ndindex(*values.shape[:2]):
            valid = pool[i, j][~np.isnan(pool[i, j])]
            if valid.size:
                out[i, j, :, t] = np.quantile(valid, quantiles)
    return out


@pytest.mark.parametrize("dask", [False, True])
def test_rolling_percentiles_equal_np_quantile(dask):
    da = _field()
    quantiles = [0.05, 0.5, 0.95]
    result = qc.centred_rolling_quantiles(da.chunk({"lat": 1}) if dask else da, window=11, quantiles=quantiles)
    expected = _rolling_reference(da, 11, quantiles)
    np.testing.assert_array_equal(result.transpose("season", "lat", "quantile", "year").values, expected)


def test_rolling_quantiles_are_centred():
    """Year t pools years t - 2 to t + 2 for a 5-year window; the first and last two years have no full window."""
    da = _field()
    result = qc.centred_rolling_quantiles(da, window=5, quantiles=[0.5])
    t = 10
    window = da.isel(year=slice(t - 2, t + 3), season=1, lat=1).values
    np.testing.assert_allclose(result.isel(year=t, season=1, lat=1, quantile=0), np.nanquantile(window, 0.5))
    valid = result.isel(season=1, lat=1, quantile=0).notnull().values
    assert not valid[:2].any() and not valid[-2:].any() and valid[2:-2].all()


def _statsmodels_rows(rows, window, it):
    frac = min(window / rows.shape[-1], 1.0)
    x = np.arange(rows.shape[-1], dtype=float)
    return np.array([np.full(row.size, np.nan) if np.isnan(row).all()
                     else lowess(row, x, frac=frac, it=it, return_sorted=False) for row in rows])


@pytest.mark.parametrize("window, it", [(81, 3), (81, 0), (20, 3), (400, 1)])
def test_lowess_equals_statsmodels(window, it):
    t = np.arange(165)
    rows = np.sin(t / 20) + 0.3 * RNG.standard_normal((30, t.size))
    rows[:5, [0, 1, 40, 164]] = np.nan          # missing values, including at the ends
    rows[5] = np.nan                            # an all-NaN series
    rows[6, 50] += 8                            # an outlier for the robustness passes
    rows[7] = 1.0                               # a constant series (zero residuals)
    result = qc.lowess_xarray(xr.DataArray(rows, dims=("s", "year")), window=window, it=it)
    np.testing.assert_allclose(result.values, _statsmodels_rows(rows, window, it), rtol=1e-9, atol=1e-12)


def test_lowess_on_dask_equals_in_memory():
    da = _field(n_years=100).isel(member=0, drop=True)
    in_memory = qc.lowess_xarray(da, window=31)
    np.testing.assert_array_equal(qc.lowess_xarray(da.chunk({"lat": 1}), window=31).values, in_memory.values)


@pytest.mark.parametrize("dask", [False, True])
def test_rolling_std_matches_the_construct_version(dask):
    """Running sums give the window-copy numbers: ddof=0, NaN skipped, windows shortened at the ends."""
    da = _field(n_years=40) * 3 + 20
    expected = da.rolling(year=11, center=True).construct("window").std(["window", "member"])
    result = qc.rolling_std(da.chunk({"lat": 1}) if dask else da, window=11).compute()
    assert result.dims == expected.dims
    np.testing.assert_allclose(result.values, expected.values, rtol=1e-9, atol=1e-9)
    # On a Dataset, as reduce_to_dataset passes it
    ds_result = qc.rolling_std(da.to_dataset(name="tas"), window=11)
    np.testing.assert_allclose(ds_result.tas.values, expected.values, rtol=1e-9, atol=1e-9)


def test_pooled_quantiles_are_xarray_quantile_exactly():
    da = _field()
    ds = da.to_dataset(name="tas")
    expected = ds.quantile([0.05, 0.5, 0.95], dim=("member", "year"), skipna=True)
    result = qc.pooled_quantiles(ds, [0.05, 0.5, 0.95])
    np.testing.assert_array_equal(result.tas.transpose(*expected.tas.dims).values, expected.tas.values)

