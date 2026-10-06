"""Checks of era5_evaluation against independent reference implementations.

Run with ``pytest tests``. Running this file directly also prints a calibration
table (how often each test flags a statistic in synthetic worlds where the
answer is known): ``python tests/test_era5_evaluation.py 200``.
"""

import sys

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr

if __name__ == "__main__":
    import conftest  # noqa: F401  (sets the import path when run as a script)

from extant import era5_evaluation as ev
from extant.significance import pvalue_two_sided

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Aligning
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
# 4. Locating ERA5
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
# 5. Moment test and field test
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
# 6. Rank histograms
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

# ---------------------------------------------------------------------------
# 9. Spatial evaluation (Suarez-Gutierrez et al., 2021, section 2.2.2)
# ---------------------------------------------------------------------------

#(t): 60 independent grid points, one per "season" of the toy ensembles
SPATIAL_CELLS = [f"cell{i}" for i in range(60)]


def _spatial(n_members=50, thresholds="fixed", **world):
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=n_members, seasons=SPATIAL_CELLS, seed=1, **world))
    whole = ev.spatial_evaluation(ens, obs, thresholds=thresholds)
    early = ev.spatial_evaluation(ens, obs, period=slice(1979, 1996), thresholds=thresholds)
    late = ev.spatial_evaluation(ens, obs, period=slice(1997, 2013), thresholds=thresholds)
    return whole, ev.period_comparison(whole, early, late)


@pytest.mark.parametrize("world, diagnosis, period", [
    ({}, "adequate", "no problem over the whole record"),
    ({"trend": 2.0}, "warms too much", "problem changes: forced response"),
    ({"trend": -1.5}, "warms too little", "problem changes: forced response"),
    ({"noise": 0.5}, "too little variability", "same problem in both: variability"),
    ({"noise": 2.0}, "too much variability", "same problem in both: variability"),
])
def test_spatial_evaluation_diagnoses_known_worlds(world, diagnosis, period):
    """Most grid points of each toy world get its diagnosis (step 7) and its period verdict (step 8)."""
    whole, periods = _spatial(**world)
    codes = {name: code for code, name in ev.DIAGNOSES.items()}
    period_codes = {name: code for code, name in ev.PERIOD_DIAGNOSES.items()}
    assert float((whole.diagnosis == codes[diagnosis]).mean()) > 0.6
    assert float((periods == period_codes[period]).mean()) > 0.6


def test_spatial_evaluation_of_a_perfect_model_matches_theory():
    whole, _ = _spatial()
    np.testing.assert_allclose(float(whole.below.mean()), 100 / 51, atol=1)
    np.testing.assert_allclose(float(whole.above.mean()), 100 / 51, atol=1)
    assert 65 < float(whole.central.mean()) < 80
    assert (whole.n_years == 35).all() and (whole.n_members == 50)


def test_perfect_model_thresholds():
    """Not in the paper: each member against the others sets the thresholds, so a perfect model passes ~90%."""
    whole, _ = _spatial(thresholds="perfect_model")
    assert float(whole.adequate.mean()) > 0.75
    assert float(whole.below_threshold.min()) >= 0 and float(whole.central_threshold.mean()) > 75
    # Any ensemble size can be tested this way; the fixed 10% needs at least MIN_MEMBERS
    assert _spatial(n_members=10, thresholds="perfect_model")[0].adequate.notnull().all()
    assert _spatial(n_members=10)[0].adequate.isnull().all()


def test_central_of_others_equals_explicit_loop():
    values = np.random.default_rng(0).standard_normal((7, 12))
    ours = ev._central_of_others(values, *ev.CENTRAL_RANGE)
    for m in range(values.shape[-1]):
        low, high = np.quantile(np.delete(values, m, axis=-1), ev.CENTRAL_RANGE, axis=-1)
        np.testing.assert_array_equal(ours[:, m], (values[:, m] >= low) & (values[:, m] <= high))


def test_adequate_area_and_count():
    adequate = xr.DataArray([[1.0, 0.0], [np.nan, 1.0]], dims=("lat", "lon"), coords={"lat": [-80.0, -60.0], "lon": [0, 90]})
    table = xr.Dataset({"adequate": adequate})
    weights = np.cos(np.deg2rad([-80.0, -60.0]))
    np.testing.assert_allclose(float(ev.adequate_area(table)), 100 * (weights[0] + weights[1]) / (2 * weights[0] + weights[1]))
    counts = ev.adequate_count(xr.concat([table, table.fillna(0)], "model"))
    np.testing.assert_array_equal(counts.n_adequate, [[2, 0], [0, 2]])
    np.testing.assert_array_equal(counts.n_tested, [[2, 2], [1, 2]])


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
