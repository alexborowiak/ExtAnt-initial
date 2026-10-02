"""Checks of era5_evaluation against independent reference implementations.

Run with ``pytest tests``. Running this file directly also prints a calibration
table (how often each test flags a statistic in synthetic worlds where the
answer is known): ``python tests/test_era5_evaluation.py 200``.
"""

import sys

import cftime
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr

if __name__ == "__main__":
    import conftest  # noqa: F401  (sets the import path when run as a script)

import era5_evaluation as ev
from quantile_calc import seasonal_mean
from significance import pvalue_two_sided
from xarray_calc import split_time

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Preparing ERA5
# ---------------------------------------------------------------------------

def test_seasonal_means_matches_lesfmip_pipeline():
    """ERA5 (datetime64, month starts) gets the same (year, season) labels and values as the
    LESFMIP pipeline applied to the same numbers on a cftime mid-month axis."""
    n = 12 * 6
    values = RNG.standard_normal(n)
    era5 = xr.DataArray(values, dims="time", coords={"time": pd.date_range("1979-01-01", periods=n, freq="MS")})
    model_time = [cftime.DatetimeNoLeap(1979 + i // 12, i % 12 + 1, 16, 12) for i in range(n)]
    model = xr.DataArray(values, dims="time", coords={"time": model_time})

    ours = ev.seasonal_means(era5)
    theirs = split_time(seasonal_mean(model), ("year", "season"), "time")

    # The Jan-Feb stub (DJF 1978) is dropped as incomplete, and so is 1984, which has no complete DJF.
    assert list(ours.year.values) == [1979, 1980, 1981, 1982, 1983] and not ours.isnull().any()
    xr.testing.assert_allclose(ours, theirs)
    # DJF 1979 = Dec 1979 + Jan 1980 + Feb 1980, labelled by December's year.
    np.testing.assert_allclose(ours.sel(year=1979, season="DJF"), values[[11, 12, 13]].mean())


def test_seasonal_mean_keeps_whole_years_on_a_360_day_calendar():
    """A model record cut at the end of 2014 ends with DJF 2013, with every season covering the same years."""
    time = [cftime.Datetime360Day(y, m, 16) for y in range(2000, 2015) for m in range(1, 13)]
    da = xr.DataArray(np.arange(len(time), dtype=float), dims="time", coords={"time": time})
    seasons = split_time(seasonal_mean(da), ("year", "season"), "time")
    assert list(seasons.year.values) == list(range(2000, 2014)) and not seasons.isnull().any()


def test_seasonal_means_refuses_shifted_months():
    """convert_calendar('360_day', align_on='year') moves month starts into the previous month: 1 March reads 29 Feb."""
    era5 = xr.DataArray(np.arange(48.0), dims="time", coords={"time": pd.date_range("1979-01-01", periods=48, freq="MS")})
    shifted = era5.convert_calendar("360_day", align_on="year")
    assert list(shifted.time.dt.month.values[:4]) == [1, 2, 2, 3]
    with pytest.raises(ValueError, match="more than once"):
        ev.seasonal_means(shifted)
    repaired = ev.monthly_time_axis(shifted)
    xr.testing.assert_equal(ev.seasonal_means(repaired), ev.seasonal_means(era5))


def test_match_grid_snaps_matching_points():
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-87.5, -85.0, -82.5], "lon": [-180.0, -177.5, -175.0, -172.5]})
    obs = like.copy(data=RNG.standard_normal((3, 4))).assign_coords(lat=like.lat + 1e-6, lon=like.lon - 1e-6)
    snapped = ev.match_grid(obs, like)
    assert (snapped.lat.values == like.lat.values).all()
    np.testing.assert_array_equal(snapped.values, obs.values)


def test_match_grid_converts_longitudes_and_latitude_order():
    """ERA5 on 0-360 longitudes, north to south, and larger than the target: the same values, on the target's labels."""
    lat, lon = np.arange(-40.0, -91.0, -2.5), np.arange(0.0, 360.0, 2.5)
    obs = xr.DataArray(np.random.default_rng(0).standard_normal((lat.size, lon.size)), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    like = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon"),
                        coords={"lat": [-85.0, -82.5, -80.0], "lon": [-180.0, -177.5, 117.5, 177.5]})
    matched = ev.match_grid(obs, like)
    np.testing.assert_array_equal(matched.lon.values, like.lon.values)
    expected = obs.sel(lat=like.lat, lon=[180.0, 182.5, 117.5, 177.5]).values
    np.testing.assert_array_equal(matched.values, expected)


def test_match_grid_interpolates_other_points():
    """Points between the source points are bilinearly interpolated, wrapping round in longitude."""
    lat, lon = np.arange(-90.0, -39.0, 2.0), np.arange(0.0, 360.0, 2.0)
    #(c): A field linear in latitude and in the longitude's cosine is interpolated almost exactly at 2° spacing
    field = (lat[:, None] + 10 * np.cos(np.deg2rad(lon))[None, :])
    obs = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    like = xr.DataArray(np.zeros((2, 4)), dims=("lat", "lon"),
                        coords={"lat": [-81.25, -61.25], "lon": [-179.0, -1.0, 90.0, 179.0]})
    matched = ev.match_grid(obs, like)
    expected = like.lat + 10 * np.cos(np.deg2rad(like.lon))
    np.testing.assert_allclose(matched.transpose("lat", "lon"), expected.transpose("lat", "lon"), atol=0.01)


# ---------------------------------------------------------------------------
# 2. Aligning
# ---------------------------------------------------------------------------

def test_align_masks_and_drops():
    ens, obs = ev.simulate_ensemble(n_members=5, years=np.arange(1979, 1990))
    ens = ens.copy()
    ens[0] = np.nan                                            # an all-NaN member
    ens.loc[dict(member=2, year=1983, season="JJA")] = np.nan  # one missing sample
    obs = obs.sel(year=slice(1981, None))                      # a shorter record
    e, o = ev.align(ens, obs)
    assert e.sizes["member"] == 4 and e.year.min() == 1981
    assert np.isnan(o.sel(year=1983, season="JJA")) and e.sel(year=1983, season="JJA").isnull().all()
    assert o.notnull().sum() == e.isel(member=0).notnull().sum() == 9 * 4 - 1


def test_align_refuses_kelvin_against_celsius():
    ens, obs = ev.simulate_ensemble(n_members=5, years=np.arange(1979, 1990))
    with pytest.raises(ValueError, match="K"):
        ev.align(ens + 273.15, obs)


# ---------------------------------------------------------------------------
# 3-4. Treatments and statistics
# ---------------------------------------------------------------------------

def test_linear_fit_and_detrend_match_polyfit_with_nan():
    years = np.arange(1979, 2014)
    y = 0.03 * (years - 1979) + RNG.standard_normal(years.size)
    y[[3, 10]] = np.nan
    da = xr.DataArray(y, dims="year", coords={"year": years})
    ok = ~np.isnan(y)
    slope, intercept = np.polyfit(years[ok], y[ok], 1)
    np.testing.assert_allclose(ev.trend(da), 10 * slope)
    residual = ev.detrend(da)
    np.testing.assert_allclose(residual.values[ok], y[ok] - (slope * years[ok] + intercept))
    assert np.isnan(residual.values[3])


def test_pooled_trend_uses_year_as_regressor():
    ens, _ = ev.simulate_ensemble(n_members=2, trend=0.5, noise=0.0)
    np.testing.assert_allclose(ev.trend(ens, ("year", "season")), 0.5)


def test_moments_match_scipy():
    da = xr.DataArray(RNG.gamma(2.0, size=(3, 40)), dims=("member", "year"))
    np.testing.assert_allclose(ev.skewness(da), scipy.stats.skew(da.values, axis=1))
    np.testing.assert_allclose(ev.excess_kurtosis(da), scipy.stats.kurtosis(da.values, axis=1))
    np.testing.assert_allclose(ev.std(da), da.values.std(axis=1, ddof=1))
    x = da.values[0]
    np.testing.assert_allclose(ev.lag1_autocorrelation(da.isel(member=0)), np.corrcoef(x[1:], x[:-1])[0, 1])


def test_width_and_tails_match_numpy():
    da = xr.DataArray(RNG.standard_normal((3, 40)), dims=("member", "year"))
    q05, q50, q95 = np.quantile(da.values, [0.05, 0.5, 0.95], axis=1)
    np.testing.assert_allclose(ev.quantile_width(da), q95 - q05)
    np.testing.assert_allclose(ev.upper_tail_width(da), q95 - q50)
    np.testing.assert_allclose(ev.lower_tail_width(da), q50 - q05)


@pytest.mark.parametrize("q", [0.5, [0.05, 0.5, 0.95], 0.0, 1.0])
def test_nan_quantile_matches_xarray(q):
    da = xr.DataArray(RNG.standard_normal((6, 36, 4, 3)), dims=("member", "year", "season", "lat"))
    da = da.where(~((da.year == 35) & (da.season != 1)))  # a shared missing-sample mask
    da[0, :, 0, 0] = np.nan                                # an all-NaN series
    expected = da.quantile(q, dim="year")
    np.testing.assert_allclose(ev.nan_quantile(da, q, "year").transpose(*expected.dims), expected, equal_nan=True)


def test_record_counts_match_brute_force_and_theory():
    x = RNG.standard_normal((400, 35))
    x[:5, -1] = np.nan
    da = xr.DataArray(x, dims=("member", "year"), coords={"year": np.arange(1979, 2014)})

    def brute(row, high=True):
        row = row[np.isfinite(row)]
        row = row if high else -row
        return sum(True if i == 0 else row[i] > row[:i].max() for i in range(row.size))

    np.testing.assert_array_equal(ev.record_highs(da).values, [brute(r) for r in x])
    np.testing.assert_array_equal(ev.record_lows(da).values, [brute(r, False) for r in x])
    np.testing.assert_allclose(float(ev.record_highs(da).mean()), ev.expected_records(35), atol=0.15)
    np.testing.assert_array_equal(ev.cumulative_records(da).isel(year=-1).values, ev.record_highs(da).values)


def test_self_signal_to_noise_of_a_known_trend():
    years = np.arange(1979, 2014)
    noise = RNG.standard_normal(years.size)
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    series = xr.DataArray(0.1 * (years - years[0]) + noise, dims="year", coords={"year": years})
    # Signal ~ 0.1 x 34 years = 3.4 over unit noise.
    np.testing.assert_allclose(float(ev.self_signal_to_noise(series)), 3.4, rtol=0.15)


def test_anomalies_reference_period():
    da = xr.DataArray(np.arange(10.0), dims="year", coords={"year": np.arange(2000, 2010)})
    np.testing.assert_allclose(ev.anomalies(da, reference=slice(2000, 2001)).values, np.arange(10.0) - 0.5)


# ---------------------------------------------------------------------------
# 5. Locating ERA5
# ---------------------------------------------------------------------------

def test_locate():
    members = xr.DataArray(np.arange(1.0, 21.0), dims="member")  # 1..20
    above = ev.locate(members, xr.DataArray(25.0))
    assert float(above.percentile) == 100 and float(above.verdict) == 1
    np.testing.assert_allclose(above.pvalue, 2 / 21)
    np.testing.assert_allclose(above.pvalue, pvalue_two_sided(members, xr.DataArray(25.0), dim="member"))
    middle = ev.locate(members, xr.DataArray(10.5))
    assert float(middle.percentile) == 50 and float(middle.verdict) == 0
    assert float(ev.locate(members, xr.DataArray(3.0)).percentile) == 100 * 2.5 / 20
    assert np.isnan(ev.locate(members, xr.DataArray(np.nan)).percentile)
    # 20 members can just reach p < 0.1 (2/21); 10 members cannot (2/11), so ERA5 beyond them all is untestable.
    assert float(above.testable) == 1
    few = ev.locate(members.isel(member=slice(0, 10)), xr.DataArray(25.0))
    assert float(few.percentile) == 100 and float(few.verdict) == 0 and float(few.testable) == 0


# ---------------------------------------------------------------------------
# 6. Moment test and field test
# ---------------------------------------------------------------------------

def test_moment_test_is_the_four_moments():
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=20))
    assert list(ev.moment_test(ens, obs).statistic.values) == ["mean", "std", "skewness", "kurtosis"]
    assert not set(ev.statistic_test(ens, obs).statistic.values) & set(ev.MOMENTS)


def _map_world(n_members=25, noise=1.0, seed=0):
    """Independent (ensemble, obs) at every point of a small polar grid: a perfect model if noise == 1."""
    rng = np.random.default_rng(seed)
    coords = {"year": np.arange(1979, 2014), "season": list(ev.SEASONS),
              "lat": np.linspace(-85.0, -50.0, 5), "lon": np.arange(0.0, 360.0, 60.0)}
    shape = tuple(len(v) for v in coords.values())
    ens = xr.DataArray(noise * rng.standard_normal((n_members, *shape)), dims=("member", *coords),
                       coords=coords)
    obs = xr.DataArray(rng.standard_normal(shape), dims=tuple(coords), coords=coords)
    return ev.align(ens, obs)


def test_field_test_areas_equal_explicit_leave_one_out():
    ens, obs = _map_world(n_members=23)
    result = ev.moment_test(ens, obs)
    field = ev.field_test(result)
    weights = np.cos(np.deg2rad(result.lat))
    n = result.sizes["member"]

    def area(flagged):
        return 100 * flagged.astype(float).weighted(weights).mean(("lat", "lon"))

    members = result["members"]
    era5 = sum(ev.locate(members.drop_isel(member=j), result["era5"]).verdict != 0 for j in range(n)) / n
    np.testing.assert_allclose(field["flagged_area"], area(era5))
    for i in (0, 7, n - 1):
        pseudo = ev.locate(members.drop_isel(member=i), members.isel(member=i, drop=True)).verdict != 0
        np.testing.assert_allclose(field["pseudo_flagged_area"].isel(member=i), area(pseudo))


def test_field_test_calibration():
    # A perfect model fails the field test about alpha of the time; one with too much noise fails for std.
    perfect = [ev.field_test(ev.moment_test(*_map_world(seed=seed))).field_verdict for seed in range(60)]
    assert float(xr.concat(perfect, "world").mean()) < 2 * ev.ALPHA
    noisy = ev.field_test(ev.moment_test(*_map_world(noise=1.6)))
    assert bool((noisy.field_verdict.sel(statistic="std") == 1).all())


def test_field_test_untestable_is_nan():
    field = ev.field_test(ev.moment_test(*_map_world(n_members=15)))
    assert bool(field.field_verdict.isnull().all()) and bool(field.flagged_area.isnull().all())


# ---------------------------------------------------------------------------
# 7. Rank histograms
# ---------------------------------------------------------------------------

def _explicit_leave_one_out_counts(ens, obs):
    """Reference: rank obs against each leave-one-out sub-ensemble and average the histograms."""
    n = ens.sizes["member"]
    total = 0
    for m in range(n):
        total = total + ev.rank_counts(ev.ensemble_rank(ens.drop_isel(member=m), obs), n, "year")
    return total / n


def test_leave_one_out_counts_equal_explicit_loop():
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=7, noise=1.0, obs_noise=1.5))
    fast = ev.leave_one_out_counts(ev.ensemble_rank(ens, obs), 7, "year")
    explicit = _explicit_leave_one_out_counts(ens, obs)
    np.testing.assert_allclose(fast.transpose(*explicit.dims), explicit)


def test_pseudo_observation_ranks_equal_explicit():
    ens, _ = ev.simulate_ensemble(n_members=6)
    ranks = ev.pseudo_observation_ranks(ens)
    for m in range(6):
        explicit = (ens.drop_isel(member=m) < ens.isel(member=m)).sum("member")
        np.testing.assert_array_equal(ranks.isel(member=m).values, explicit.values)


def test_rank_statistics_flat_and_u_shaped():
    flat = xr.DataArray(np.full(11, 5.0), dims="rank", coords={"rank": np.arange(11)})
    stats = ev.rank_statistics(flat)
    np.testing.assert_allclose(stats.dispersion, 1.0)
    np.testing.assert_allclose(stats.mean_rank, 0.5)
    np.testing.assert_allclose(stats.outside, 200 / 11)
    u_shape = flat.copy(data=[20.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20.0])
    assert float(ev.rank_statistics(u_shape).dispersion) > 1


def test_regroup_ranks_keeps_flat_flat_and_total():
    for n in (7, 50, 63):
        flat = xr.DataArray(np.full(n, 100 / n), dims="rank", coords={"rank": np.arange(n)})
        grouped = ev.regroup_ranks(flat, 10)
        np.testing.assert_allclose(grouped, 10.0)
        np.testing.assert_allclose(grouped.sum(), 100.0)


def test_rank_histogram_test_shapes_and_frequencies():
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=30))
    pooled = ev.rank_histogram_test(ens, obs, pool_dims=("year", "season"))
    assert pooled.sizes["rank"] == 30 and pooled.sizes["statistic"] == len(ev.RANKS)
    np.testing.assert_allclose(pooled.obs_frequency.sum("rank"), 100)
    np.testing.assert_allclose(pooled.pseudo_frequency.sum("rank"), 100)
    assert "season" in ev.rank_histogram_test(ens, obs).percentile.dims


# ---------------------------------------------------------------------------
# 8-9. Distributions and plumes
# ---------------------------------------------------------------------------

def test_kde_matches_scipy_and_integrates_to_one():
    samples = RNG.standard_normal(35)
    da = xr.DataArray(samples, dims="year")
    x = xr.DataArray(np.linspace(-6, 6, 2001), dims="x")
    bandwidth = 0.4
    ours = ev.kde(da, x, bandwidth=bandwidth)
    reference = scipy.stats.gaussian_kde(samples, bw_method=bandwidth / samples.std(ddof=1))(x.values)
    np.testing.assert_allclose(ours, reference, rtol=1e-10)
    np.testing.assert_allclose(np.trapezoid(ours, x), 1, rtol=1e-4)


def test_pooled_kde_is_the_member_mean_and_anomalies_are_centred():
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=8, bias=2))
    summary = ev.distribution_summary(ens, obs)
    raw = summary["raw"]
    np.testing.assert_allclose(raw.ensemble_kde, raw.member_kde.mean("member"))
    pooled = ev.kde(ens, raw.x, ("year", "member"), raw.bandwidth)
    np.testing.assert_allclose(raw.ensemble_kde.transpose(*pooled.dims), pooled, rtol=1e-10)
    np.testing.assert_allclose(summary["anomaly"].obs.mean("year"), 0, atol=1e-12)


def test_plume_outside_frequency():
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=9, bias=10))
    plume = ev.ensemble_plume(ens, obs)
    frequency = ev.outside_frequency(plume, ("year", "season"))
    assert float(frequency.below) == 100 and float(frequency.above) == 0
    assert float(plume["expected_outside"]) == 20.0


def test_evaluate_and_summary_table():
    results = {name: ev.evaluate(*ev.align(*ev.simulate_ensemble(n_members=n, seed=n)))
               for name, n in (("a", 20), ("b", 25))}
    table = ev.summary_table(results)
    assert set(table.dims) == {"model", "statistic", "season"}
    assert "dispersion (anomaly)" in table.statistic and "rank_trend" in table.statistic
    assert set(ev.MOMENTS) | set(ev.STATISTICS) <= set(table.statistic.values)
    assert table.attrs["alpha"] == ev.ALPHA


# ---------------------------------------------------------------------------
# Calibration (run this file directly)
# ---------------------------------------------------------------------------

def calibration(n_worlds=100, n_members=40):
    """Fraction of synthetic worlds in which each test flags each scenario (a perfect model: ~ALPHA)."""
    table = {}
    for scenario, kwargs in ev.SCENARIOS.items():
        flags = []
        for seed in range(n_worlds):
            ens, obs = ev.align(*ev.simulate_ensemble(n_members=n_members, seed=seed, **kwargs))
            tests = ev.combine_tests(ev.moment_test(ens, obs), ev.statistic_test(ens, obs))
            ranks_a = ev.rank_histogram_test(ens, obs, "anomaly", ("year", "season"), keep_histograms=False)
            ranks_d = ev.rank_histogram_test(ens, obs, "detrended", ("year", "season"), keep_histograms=False)
            row = {f"{s} (per season)": float((tests.verdict.sel(statistic=s) != 0).mean())
                   for s in tests.statistic.values}
            row.update({
                "rank dispersion (anomaly)": float(ranks_a.verdict.sel(statistic="dispersion") != 0),
                "rank dispersion (detrended)": float(ranks_d.verdict.sel(statistic="dispersion") != 0),
                "rank trend (anomaly)": float(ranks_a.verdict.sel(statistic="rank_trend") != 0),
            })
            flags.append(row)
        table[scenario] = pd.DataFrame(flags).mean()
    return pd.DataFrame(table)


if __name__ == "__main__":
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"Fraction flagged over {worlds} synthetic worlds (40 members, 35 years x 4 seasons):")
    print(calibration(worlds).round(2))
