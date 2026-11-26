"""Months into seasons, time into year and season, and back again."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extant.utils import time_axis

RNG = np.random.default_rng(42)


def test_seasonal_means_matches_lesfmip_pipeline():
    """ERA5 (datetime64, month starts) gets the same (year, season) labels and values as the
    LESFMIP pipeline applied to the same numbers on a cftime mid-month axis."""
    n = 12 * 6
    values = RNG.standard_normal(n)
    era5 = xr.DataArray(values, dims="time", coords={"time": pd.date_range("1979-01-01", periods=n, freq="MS")})
    model_time = [cftime.DatetimeNoLeap(1979 + i // 12, i % 12 + 1, 16, 12) for i in range(n)]
    model = xr.DataArray(values, dims="time", coords={"time": model_time})

    ours = time_axis.seasonal_means(era5)
    theirs = time_axis.split_time(time_axis.seasonal_mean(model), ("year", "season"), "time")

    # The Jan-Feb stub (DJF 1978) is dropped as incomplete, and so is 1984, which has no complete DJF.
    assert list(ours.year.values) == [1979, 1980, 1981, 1982, 1983] and not ours.isnull().any()
    xr.testing.assert_allclose(ours, theirs)
    # DJF 1979 = Dec 1979 + Jan 1980 + Feb 1980, labelled by December's year.
    np.testing.assert_allclose(ours.sel(year=1979, season="DJF"), values[[11, 12, 13]].mean())


def test_seasonal_mean_keeps_whole_years_on_a_360_day_calendar():
    """A model record cut at the end of 2014 ends with DJF 2013, with every season covering the same years."""
    time = [cftime.Datetime360Day(y, m, 16) for y in range(2000, 2015) for m in range(1, 13)]
    da = xr.DataArray(np.arange(len(time), dtype=float), dims="time", coords={"time": time})
    seasons = time_axis.split_time(time_axis.seasonal_mean(da), ("year", "season"), "time")
    assert list(seasons.year.values) == list(range(2000, 2014)) and not seasons.isnull().any()


def test_seasonal_means_refuses_shifted_months():
    """convert_calendar('360_day', align_on='year') moves month starts into the previous month: 1 March reads 29 Feb."""
    era5 = xr.DataArray(np.arange(48.0), dims="time", coords={"time": pd.date_range("1979-01-01", periods=48, freq="MS")})
    shifted = era5.convert_calendar("360_day", align_on="year")
    assert list(shifted.time.dt.month.values[:4]) == [1, 2, 2, 3]
    with pytest.raises(ValueError, match="more than once"):
        time_axis.seasonal_means(shifted)
    repaired = time_axis.monthly_time_axis(shifted)
    xr.testing.assert_equal(time_axis.seasonal_means(repaired), time_axis.seasonal_means(era5))


def test_seasons_in_time_is_split_time_backwards():
    """DJF 1980 sits at January 1981; every (year, season) comes back once, in order, with its own value."""
    da = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("year", "season"),
                      coords={"year": [1980, 1981, 1982], "season": ["DJF", "JJA", "MAM", "SON"]})
    on_time = time_axis.seasons_in_time(da)
    assert on_time.dims == ("time",) and on_time.indexes["time"].is_monotonic_increasing and on_time.size == 12
    djf_1980 = on_time.where((on_time["year"] == 1980) & (on_time["season"] == "DJF"), drop=True)
    np.testing.assert_allclose(djf_1980["time"], 1981 + 0.5 / 12)
    assert djf_1980.item() == float(da.sel(year=1980, season="DJF"))


def test_decimal_year_is_the_middle_of_the_month_on_any_calendar():
    standard = xr.DataArray(pd.to_datetime(["1990-07-15", "1990-07-01"]), dims="time")
    thirty_days = xr.DataArray([cftime.Datetime360Day(1990, 7, 30)], dims="time")
    np.testing.assert_allclose(time_axis.decimal_year(standard), 1990 + 6.5 / 12)
    np.testing.assert_allclose(time_axis.decimal_year(thirty_days), 1990 + 6.5 / 12)
