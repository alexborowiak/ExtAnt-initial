"""Smoke tests: every figure function draws without error on small synthetic inputs."""

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("plotting_modules")

import era5_evaluation as ev  # noqa: E402
import era5_evaluation_plots as evp  # noqa: E402
import response_change as rc  # noqa: E402
import response_change_plots as rcp  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def results():
    """``ev.evaluate`` output for two toy models at one point."""
    return {name: ev.evaluate(*ev.align(*ev.simulate_ensemble(n_members=n, seed=n, **kwargs)))
            for name, n, kwargs in (("A", 25, {}), ("B", 30, {"noise": 1.6}))}


def test_distribution_grids(results):
    one = results["A"]
    evp.distribution_grid(one["distributions"], moments=one["moments"], title="one model")
    evp.distribution_grid({m: r["distributions"] for m, r in results.items()}, season="DJF",
                          moments={m: r["moments"] for m, r in results.items()})


def test_plume_grids(results):
    evp.plume_grid(results["A"]["plumes"], context=results["B"]["plumes"].sel(treatment=["anomaly"]))
    plumes = xr.concat([r["plumes"].sel(treatment="anomaly") for r in results.values()], dim="model",
                       coords="minimal", compat="override").assign_coords(model=list(results))
    evp.plume_grid(plumes, row_dim="model", col_dim="season", sharey=True)


def test_moment_and_rank_figures(results):
    evp.moment_grid(results["A"]["moments"], statistics=ev.SIGNAL_STATISTICS + ev.NOISE_STATISTICS)
    evp.rank_histogram_row({t: r for t, r in results["A"]["ranks"].items()})
    evp.rank_histogram_grid({m: r["ranks"] for m, r in results.items()})
    ens, obs = ev.align(*ev.simulate_ensemble(n_members=20))
    season_ens, season_obs = ens.sel(season="DJF"), obs.sel(season="DJF")
    result = ev.rank_histogram_test(season_ens, season_obs, "anomaly")
    evp.rank_anatomy(ev.treat(season_ens, "anomaly"), ev.treat(season_obs, "anomaly"), result)


def test_summaries(results):
    table = ev.summary_table(results)
    evp.scorecard(table)
    evp.signal_noise_diagram(table)
    evp.signal_noise_diagram(table, y="width")
    demo = {name: ev.evaluate(*ev.align(*ev.simulate_ensemble(n_members=20, **kwargs)))
            for name, kwargs in ev.SCENARIOS.items()}
    evp.scenario_grid(demo)


def test_maps(results):
    pytest.importorskip("cartopy")
    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    field = xr.DataArray(np.random.default_rng(0).uniform(0, 100, (2, 4, lat.size, lon.size)),
                         dims=("model", "season", "lat", "lon"),
                         coords={"model": ["A", "B"], "season": ["DJF", "JJA", "MAM", "SON"], "lat": lat, "lon": lon})
    table = xr.Dataset({"percentile": field.expand_dims(statistic=["std"])}).assign_coords(
        label=("statistic", ["Std. deviation"]))
    evp.percentile_maps(table, "std")

    shape = ("model", "experiment", "season", "lat", "lon")
    rng = np.random.default_rng(1)
    coords = {"model": ["A", "B"], "experiment": ["hist-GHG", "historical"], "season": ["DJF", "JJA", "MAM", "SON"],
              "lat": lat, "lon": lon}
    random = lambda: xr.DataArray(rng.standard_normal((2, 2, 4, lat.size, lon.size)), dims=shape, coords=coords)
    summary = xr.Dataset({
        "mean_change": random() + 2,
        "width_change": random(),
        "mean_robust": random() > 0,
        "width_significant": random() > 0.5,
    })
    rcp.joint_change_maps(summary, "A")
    counts = rc.joint_change_counts(summary)
    rcp.joint_change_count_maps(counts, "DJF", n_models=2)


def test_shift_and_widen_and_scatter():
    rng = np.random.default_rng(2)
    samples = {e: xr.DataArray(shift + scale * rng.standard_normal((20, 11)), dims=("member", "year"))
               for e, shift, scale in (("hist-nat", 0, 1), ("historical", 2, 1.4), ("hist-GHG", 3, 1.2))}
    shifts = {e: rc.quantile_shift(samples[e], samples["hist-nat"], n_boot=100) for e in ("historical", "hist-GHG")}
    rcp.shift_and_widen(samples, shifts, "historical")

    coords = {"model": ["A", "B"], "experiment": ["hist-GHG", "historical"], "season": ["DJF", "JJA", "MAM", "SON"]}
    values = lambda: xr.DataArray(rng.standard_normal((2, 2, 4)), dims=tuple(coords), coords=coords)
    regional = xr.Dataset({"mean_change": values(), "width_change": values(), "tail_asymmetry": values()},
                          attrs={"lat_max": -60})
    rcp.response_scatter(regional)
    rcp.response_scatter(regional, y="tail_asymmetry")
