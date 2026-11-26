"""Trends, smoothing and the forced response, against numpy and on series with a known answer."""

from functools import partial

import numpy as np
import xarray as xr

from extant.statistics import trend

RNG = np.random.default_rng(5)


def test_linear_fit_and_detrend_match_polyfit_with_nan():
    years = np.arange(1979, 2014)
    y = 0.03 * (years - 1979) + RNG.standard_normal(years.size)
    y[[3, 10]] = np.nan
    da = xr.DataArray(y, dims="year", coords={"year": years})
    ok = ~np.isnan(y)
    slope, intercept = np.polyfit(years[ok], y[ok], 1)
    np.testing.assert_allclose(trend.trend(da), slope)
    residual = trend.detrend(da)
    np.testing.assert_allclose(residual.values[ok], y[ok] - (slope * years[ok] + intercept))
    assert np.isnan(residual.values[3])


def test_pooled_trend_uses_year_as_regressor():
    years = np.arange(1979, 2014)
    da = xr.DataArray(np.repeat(0.05 * (years - 1979)[:, None], 4, axis=1), dims=("year", "season"),
                      coords={"year": years, "season": ["DJF", "MAM", "JJA", "SON"]})
    np.testing.assert_allclose(trend.trend(da, ("year", "season")), 0.05)


def test_trend_on_a_datetime_axis_is_per_calendar_year():
    """The same series on dates as on years: the x values are each date's calendar year, so the slope is per year."""
    y = 0.03 * np.arange(35) + RNG.standard_normal(35)
    y[[3, 10]] = np.nan
    on_years = xr.DataArray(y, dims="year", coords={"year": np.arange(1979, 2014)})
    on_dates = xr.DataArray(y, dims="time", coords={"time": xr.date_range("1979-12-01", periods=35, freq="YS-DEC")})
    np.testing.assert_allclose(trend.trend(on_dates, "time"), trend.trend(on_years))
    np.testing.assert_allclose(trend.linear_fit(on_dates, "time"), trend.linear_fit(on_years))


def test_mann_kendall_leaves_the_missing_years_out():
    """A series with missing years has the same S and p as the series without them; too short a one has none."""
    y = 0.05 * np.arange(30) + RNG.standard_normal(30)
    gaps = np.where(np.isin(np.arange(30), [2, 7, 20]), np.nan, y)
    short = np.where(np.arange(30) < 5, y, np.nan)
    da = xr.DataArray(np.stack([gaps, short]), dims=("station", "year"), coords={"year": np.arange(1980, 2010)})
    result = trend.grid_mk(da, "year")
    without = trend.grid_mk(da.isel(station=0).dropna("year"), "year")
    for name in ("S", "p", "n_eff"):
        np.testing.assert_allclose(result[name].isel(station=0), without[name])
    assert np.isnan(result["p"].isel(station=1))


def test_lowess_keeps_a_straight_line_and_the_missing_years():
    """A local straight-line fit of a straight line is the line; a missing year stays missing."""
    t = np.arange(165.0)
    rows = np.stack([2.0 + 0.03 * t, -1.0 - 0.01 * t])
    rows[1, 40] = np.nan
    result = trend.lowess(xr.DataArray(rows, dims=("s", "year")), window=81).values
    np.testing.assert_allclose(result[0], rows[0], atol=1e-10)
    np.testing.assert_allclose(np.delete(result[1], 40), np.delete(rows[1], 40), atol=1e-10)
    assert np.isnan(result[1, 40])


def test_lowess_on_dask_equals_in_memory():
    da = xr.DataArray(RNG.standard_normal((100, 3)), dims=("year", "lat"))
    in_memory = trend.lowess(da, window=31)
    np.testing.assert_allclose(trend.lowess(da.chunk({"lat": 1}), window=31).values, in_memory.values, rtol=1e-12)


def test_members_minus_the_forced_response_leave_internal_variability():
    shift, spread = 2.0, 1.5
    members = xr.DataArray(shift + spread * RNG.standard_normal((30, 25, 4, 3)), dims=("member", "year", "season", "lat"))
    tree = xr.DataTree.from_dict({"A/historical": xr.Dataset({"tas": members})})
    internal = tree - tree.map_over_datasets(partial(trend.forced_response, centre='median', window=21))
    anomalies = internal["A/historical"].tas
    # The forced shift is gone and the spread is the members' own, not shrunk by removing it.
    np.testing.assert_allclose(float(anomalies.mean()), 0, atol=0.05)
    np.testing.assert_allclose(float(anomalies.std()), spread, rtol=0.05)
