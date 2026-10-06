"""Checks of observed_change on synthetic data with a known answer."""

import numpy as np
import pytest
import xarray as xr

from extant import era5_evaluation as ev
from extant import observed_change as oc

RNG = np.random.default_rng(8)
YEARS = np.arange(1979, 2015)
SEASONS = ["DJF", "JJA", "MAM", "SON"]


def _series(n, signal, noise=0.3):
    """Members (or, with n=None, one series) of ``signal`` + noise, on ``signal``'s (year, season)."""
    if n is None:
        return signal + noise * xr.DataArray(RNG.standard_normal(signal.shape), dims=signal.dims, coords=signal.coords)
    members = xr.DataArray(RNG.standard_normal((n, *signal.shape)), dims=("member", *signal.dims),
                           coords=signal.coords)
    return (signal + noise * members).transpose("member", ...)


@pytest.fixture(scope="module")
def world():
    """Forced responses with distinct shapes: an accelerating ANT trend and NAT volcanic dips."""
    t = (YEARS - YEARS[0]) / 35
    ant = xr.DataArray(np.repeat((1.5 * t ** 2)[:, None], 4, axis=1), dims=("year", "season"),
                       coords={"year": YEARS, "season": SEASONS})
    dips = np.zeros(YEARS.size)
    dips[[3, 12, 13, 25]] = -1.0
    nat = xr.DataArray(np.repeat(dips[:, None], 4, axis=1), dims=("year", "season"),
                       coords={"year": YEARS, "season": SEASONS})
    ensembles = {"historical": _series(40, ant + nat), "hist-nat": _series(30, nat)}
    obs = _series(None, 0.7 * ant + 1.0 * nat)
    return ensembles, obs


def test_scaling_factors_recover_the_truth(world):
    ensembles, obs = world
    result = oc.scaling_factors(ensembles, obs, "ANT + NAT")
    truth = {"ANT": 0.7, "NAT": 1.0}
    for signal, value in truth.items():
        row = result.sel(signal=signal)
        assert float(row.lower) <= value <= float(row.upper), (signal, float(row.beta))
    np.testing.assert_allclose(result.perfect_model_beta, 1.0, atol=0.1)
    assert bool(result.detected.sel(signal="ANT")) and not bool(result.consistent.sel(signal="ANT"))
    # The attributable trends add up to (roughly) the observed trend.
    np.testing.assert_allclose(float(result.attributable_trend.sum()), float(result.observed_trend), atol=0.05)


def test_detection_class():
    verdicts = xr.DataArray([0, 0, 1, 1], dims="point")
    historical = xr.Dataset({"verdict": xr.DataArray([0, 1, 0, 1], dims="point")})
    hist_nat = xr.Dataset({"verdict": verdicts})
    np.testing.assert_array_equal(oc.detection_class(historical, hist_nat).values, [0, 3, 1, 2])
    # A test with too few members to flag anything leaves the point unclassified.
    hist_nat["testable"] = xr.DataArray([1, 0, 1, 1], dims="point")
    np.testing.assert_array_equal(oc.detection_class(historical, hist_nat).values, [0, np.nan, 1, 2])


def test_regional_mean_is_cos_lat_weighted():
    lat = np.array([-80.0, -70.0, -60.0])
    da = xr.DataArray(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": [0.0, 180.0]})
    weights = np.cos(np.deg2rad([-80.0, -70.0]))
    expected = (1 * weights[0] + 2 * weights[1]) / weights.sum()
    np.testing.assert_allclose(oc.regional_mean(da, {"lat": slice(-90, -65)}), expected)


def test_record_curves(world):
    ensembles, obs = world
    curves = oc.record_curves(ensembles, obs)
    final = curves["obs"].isel(year=-1)
    np.testing.assert_allclose(final.sel(kind="highs"), float(ev.record_highs(obs).sum()))
    np.testing.assert_allclose(final.sel(kind="lows"), float(ev.record_lows(obs).sum()))
    np.testing.assert_allclose(curves["expected"].isel(year=-1), 4 * ev.expected_records(YEARS.size))
    assert set(curves["members"].experiment.values) == set(ensembles)


def test_experiment_consistency_and_table(world):
    ensembles, obs = world
    results = {"A": oc.experiment_consistency(ensembles, obs), "B": oc.experiment_consistency(ensembles, obs)}
    # The observed trend is 0.7x the forced trend: below historical's members, above hist-nat's.
    assert float(results["A"]["historical"].sel(statistic="trend").verdict.mean()) == -1
    assert float(results["A"]["hist-nat"].sel(statistic="trend").verdict.mean()) == 1
    table = oc.consistency_table(results)
    assert set(table.dims) == {"model", "statistic", "season"} and set(table.statistic.values) == set(ensembles)


def test_detection_maps_on_a_tree():
    # Model A has both experiments (with different member counts); B lacks hist-nat and is left out.
    rng = np.random.default_rng(3)
    coords = {"year": YEARS, "season": SEASONS, "lat": [-80.0, -70.0], "lon": [0.0, 90.0, 180.0]}
    shape = tuple(len(v) for v in coords.values())
    trend = xr.DataArray(np.arange(YEARS.size) / 10, dims="year", coords={"year": YEARS})

    def run(n, slope):
        noise = xr.DataArray(0.3 * rng.standard_normal((n, *shape)), dims=("member", *coords), coords=coords)
        return xr.Dataset({"tas": slope * trend + noise})

    tree = xr.DataTree.from_dict({"A/historical": run(30, 1.0), "A/hist-nat": run(25, 0.0),
                                  "B/historical": run(30, 1.0)})
    obs = trend + xr.DataArray(0.3 * rng.standard_normal(shape), dims=tuple(coords), coords=coords)
    detections = oc.detection_maps(tree, obs)
    assert list(detections.model.values) == ["A"]
    # ERA5 warms like historical and unlike hist-nat: detected everywhere, and consistent bar chance flags.
    assert bool(detections.detection_class.isin([1, 2]).all())
    assert float((detections.detection_class == 1).mean()) > 0.7
