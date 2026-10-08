"""Smoke tests: every figure function draws without error on small synthetic inputs."""

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("plotting_modules")

from extant import era5_evaluation as ev  # noqa: E402
from extant.plots import era5_evaluation as evp  # noqa: E402
from extant import response_change as rc  # noqa: E402
from extant.plots import response_change as rcp  # noqa: E402


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


def test_statistic_and_rank_figures(results):
    evp.statistic_grid(results["A"]["moments"])
    evp.statistic_grid(results["A"]["statistics"], statistics=ev.SIGNAL_STATISTICS + ev.NOISE_STATISTICS)
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
    #(c): With the field test, and model B too small to test
    cell = xr.DataArray([[[10.0, 30.0, 12.0, 8.0]], [[np.nan] * 4]], dims=("model", "statistic", "season"),
                        coords={"model": ["A", "B"], "statistic": ["std"], "season": ["DJF", "JJA", "MAM", "SON"]})
    table = table.assign(
        testable=xr.DataArray([1.0, 0.0], dims="model", coords={"model": ["A", "B"]}),
        flagged_area=cell, perfect_area=cell * 0 + 15, field_verdict=(cell > 15).where(cell.notnull()),
    )
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


@pytest.fixture(scope="module")
def spatial_table():
    """``ev.spatial_evaluation`` with step 8 for two toy models on a small map (B with too little variability)."""
    rng = np.random.default_rng(5)
    coords = {"year": np.arange(1979, 2014), "season": ["DJF", "JJA"], "lat": [-80.0, -75.0, -70.0],
              "lon": [0.0, 90.0, 180.0, 270.0]}
    shape = tuple(len(v) for v in coords.values())
    obs = xr.DataArray(rng.standard_normal(shape), dims=tuple(coords), coords=coords)
    tables = []
    for model, noise in (("A", 1.0), ("B", 0.5)):
        ensemble, aligned = ev.align(xr.DataArray(noise * rng.standard_normal((25, *shape)),
                                                  dims=("member", *coords), coords=coords), obs)
        whole = ev.spatial_evaluation(ensemble, aligned)
        early, late = (ev.spatial_evaluation(ensemble, aligned, period=p)
                       for p in (slice(1979, 1996), slice(1997, 2013)))
        tables.append(whole.assign(period_diagnosis=ev.period_comparison(whole, early, late))
                      .expand_dims(model=[model]))
    return xr.concat(tables, "model")


def test_spatial_evaluation_figures(spatial_table):
    pytest.importorskip("cartopy")
    evp.spatial_maps(spatial_table)
    evp.diagnosis_maps(spatial_table)
    evp.diagnosis_maps(spatial_table, variable="period_diagnosis")
    evp.adequate_count_maps(ev.adequate_count(spatial_table))



def test_method_figures():
    """The step-by-step schematics and the small figures that follow the data, on synthetic ensembles."""
    from extant import quantiles as qc
    from extant import significance as sig
    from extant.plots import methods

    rng = np.random.default_rng(1)
    years = np.arange(1850, 2014)

    def members(n, trend=0.0, scale=1.0):
        values = trend * (years - 1850) / 164 + scale * rng.standard_normal((n, years.size))
        return xr.DataArray(values, dims=("member", "year"), coords={"year": years})

    hist, nat = members(30, 2.0, 1.2), members(30)
    final, nat_final = hist.isel(year=slice(-21, None)), nat.isel(year=slice(-21, None))
    selected, window_starts = sig.bootstrap_draws(30, years.size, n_trials=200)
    null = sig.bootstrap_qrange(nat, n_trials=200) - qc.quantile_range(nat)
    change = qc.quantile_range(final) - qc.quantile_range(nat)
    methods.bootstrap_schematic(nat, final, null, selected, window_starts, change, sig.pvalue_two_sided(null, change),
                                "historical")
    weights = rc.permutation_weights(200, 30, 30, np.random.default_rng(0))
    permutations = weights @ np.concatenate([final.mean("year").values, nat_final.mean("year").values])
    methods.permutation_schematic(final, nat_final, permutations, weights > 0, float(final.mean() - nat_final.mean()),
                                  0.01, "historical")
    rolling = qc.centred_rolling_quantiles(hist, quantiles=[0.05, 0.5, 0.95])
    methods.rolling_quantile_demo(hist, rolling, 1950, 21)
    smooth = qc.lowess_matrix_xarray(rolling)
    methods.smoothing_demo(rolling, smooth)
    both = xr.concat([smooth, qc.lowess_matrix_xarray(qc.centred_rolling_quantiles(nat, quantiles=[0.05, 0.5, 0.95]))],
                     dim="experiment").assign_coords(experiment=["historical", "hist-nat"])
    methods.quantile_change_demo(both, nat.quantile([0.05, 0.5, 0.95], dim=["member", "year"]), "historical")
    methods.forced_response_demo(hist, rc.forced_response(hist))
    methods.two_sample_demo(hist.isel(year=slice(-21, None)), nat.isel(year=slice(-21, None)), "historical", t=3.0, p=0.01)
    means = xr.concat([hist.mean("member"), nat.mean("member")], dim="experiment").assign_coords(
        experiment=["historical", "hist-nat"])
    noise = xr.full_like(hist.mean("member"), 0.5)
    methods.signal_to_noise_demo(means, qc.lowess_matrix_xarray(means), noise, (means[0] - means[1]) / noise, "historical")

    seasons = xr.DataArray(rng.standard_normal((years.size, 4)), dims=("year", "season"),
                           coords={"year": years, "season": ["DJF", "JJA", "MAM", "SON"]})
    methods.year_season_demo(seasons)
    import pandas as pd
    time = pd.date_range("1990-01-01", periods=72, freq="MS")
    methods.seasonal_mean_demo(xr.DataArray(rng.standard_normal(72), dims="time", coords={"time": time}),
                               seasons, slice(1990, 1994))
    methods.member_count_heatmap(pd.DataFrame({"hist-nat": [50, np.nan], "historical": [65, 10]}, index=["A", "B"]))
    tree = xr.DataTree.from_dict({e: xr.Dataset({"tas": m.expand_dims(season=["DJF"])}) for e, m in
                                  (("historical", hist), ("hist-nat", nat))})
    methods.members_by_experiment(tree, ["hist-nat", "historical", "hist-GHG"])


@pytest.fixture(scope="module")
def zonal_inputs():
    """A change summary with tails and zonal tests, sea-ice edges and a concentration climatology, all synthetic."""
    from extant import sea_ice, zonal

    rng = np.random.default_rng(7)
    lat, lon = np.arange(-87.5, -39.9, 5.0), np.arange(-180.0, 180.0, 30.0)
    coords = {"model": ["A", "B"], "experiment": ["hist-GHG", "historical"], "season": ["DJF", "JJA", "MAM", "SON"],
              "lat": lat, "lon": lon}
    dims = tuple(coords)
    random = lambda: xr.DataArray(rng.standard_normal(tuple(len(v) for v in coords.values())), dims=dims, coords=coords)
    summary = xr.Dataset({
        "mean_change": random() + 2, "width_change": random(), "upper_tail_change": random(),
        "lower_tail_change": random(), "mean_robust": random() > 0, "width_significant": random() > 0.5,
    })
    pvalue = abs(random().mean("lon")) / 3
    null = xr.full_like(pvalue.isel(experiment=0, drop=True), 0.3)
    width = xr.Dataset({"width_pvalue": pvalue, "hist_nat_width": null * 10, "null_lower": -null, "null_upper": null})
    zonal_ds = zonal.zonal_summary(summary, xr.Dataset({"pvalue": pvalue}), width)

    ice = (100 / (1 + np.exp((xr.DataArray(lat, dims="lat", coords={"lat": lat}) + 62) / 2))
           ).expand_dims(lon=lon, model=["A", "B"], experiment=["hist-nat", "hist-GHG", "historical"],
                         season=coords["season"]).transpose("model", "experiment", "season", "lat", "lon")
    ice = ice.where(ice.lat > -75)
    area = sea_ice.cell_area(ice.lat, ice.lon)
    edges = sea_ice.equivalent_latitude(sea_ice.extent(ice, area)
                                        + sea_ice.continent_area(ice.isel(model=0, experiment=0, season=0).isnull(), area))
    return summary, zonal_ds, ice, edges


def test_zonal_figures(zonal_inputs):
    from extant.plots import zonal as zp

    _, zonal_ds, _, edges = zonal_inputs
    zp.zonal_change_grid(zonal_ds, "JJA", edges=edges)
    zp.zonal_change_grid(zonal_ds, "DJF", rows="experiment", lines="model")
    #(c): One model, the seasons as rows
    zp.zonal_change_grid(zonal_ds, rows="season", lines="experiment", select={"model": "A"}, edges=edges)
    zp.zonal_reference_width(zonal_ds, edges=edges)


def test_sea_ice_figures(zonal_inputs):
    pytest.importorskip("cartopy")
    from extant import sea_ice
    from extant.plots import sea_ice as sip

    summary, _, ice, _ = zonal_inputs
    rcp.extreme_change_maps(summary, "A", "historical", ice=ice)
    rcp.joint_change_maps(summary, "B", ice=ice)
    sip.edge_schematic(sea_ice.edge_example(ice.sel(model="A", experiment="hist-nat", season="JJA")))

    years = np.arange(1950, 1960)
    extent = xr.DataArray(np.random.default_rng(8).uniform(2e6, 4e6, (5, years.size, 2)),
                          dims=("member", "year", "season"), coords={"year": years, "season": ["DJF", "JJA"]})
    extents = xr.DataTree.from_dict({f"A/{e}": xr.Dataset({"extent": extent, "continent_area": 13.9e6})
                                     for e in ("hist-nat", "historical")})
    sip.extent_timeseries(sea_ice.extent_summary(extents))


def test_forced_response_figures():
    pytest.importorskip("cartopy")
    from extant import significance as sig
    from extant.plots import forced_response as fr

    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    rng = np.random.default_rng(2)
    coords = {"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"], "lat": lat, "lon": lon}
    change = xr.DataArray(rng.standard_normal((3, 2, lat.size, lon.size)), dims=tuple(coords), coords=coords)
    significant = np.abs(change) > 1
    change_scale = fr.Scale(np.linspace(-3, 3, 25), "RdBu_r", "Change")

    #(c): Columns that share a Scale share one colour bar: two scales, two bars, plus the dots' legend
    panels = fr.field_grid(
        [fr.Column(change.sel(model="A"), change_scale, "Change", not_significant=~significant.sel(model="A")),
         fr.Column(change.sel(model="B"), change_scale, "Another\nchange"),
         fr.Column(np.abs(change.sel(model="A")) / 4, fr.pvalue_scale(), "p-value")],
        title="A title long enough to need wrapping onto a second line in a narrow figure of three columns",
        row_dim="experiment",
    )
    assert panels.axes.shape == (2, 3)
    assert len(panels.fig.axes) == 6 + 2

    #(c): A model without one experiment leaves its panels empty
    fr.map_grid(np.abs(change.where(change["model"] != "C")), fr.separation_scale(), title="Every model",
                not_significant=~significant)

    #(c): Rows that share a Scale share one colour bar: the two experiments' changes, then a p-value per experiment
    panels = fr.row_grid(
        [fr.Row(change, change_scale, "Change", not_significant=~significant),
         fr.Row(np.abs(change) / 4, fr.pvalue_scale(), "p-value")],
        title="Rows", row_dim="experiment", col_dim="model",
    )
    assert panels.axes.shape == (4, 3)
    assert len(panels.extras["cbars"]) == 2

    #(c): The counts under the median: three rows, two colour bars
    counts = sig.significant_counts(change, significant)
    panels = fr.count_grid(counts, 3, title="How many", label="Models",
                           above=[fr.Row(change.median("model"), change_scale, "Median")])
    assert panels.axes.shape == (3, 2)
    assert len(panels.extras["cbars"]) == 2
    thresholds = xr.DataArray([0.5, 1, 2], dims="threshold", coords={"threshold": [0.5, 1, 2]})
    panels = fr.count_grid((np.abs(change) > thresholds).sum("model"), 3, title="How many apart", label="Models",
                           row_dim="threshold", signed=False)
    assert panels.axes.shape == (3, 2)
    classes = fr.class_scale(["neither", "mean only", "width only", "both"], ["white", "orange", "violet", "purple"])
    fr.map_grid(xr.DataArray(rng.integers(0, 4, change.shape), dims=change.dims, coords=change.coords), classes,
                title="Classes")

    quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
    curves = xr.DataArray(rng.standard_normal((3, 2, quantiles.size)), dims=("model", "experiment", "quantile"),
                          coords={"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"],
                                  "quantile": quantiles})
    fr.quantile_curves(curves, title="Every quantile")
    fr.quantile_curves(curves.expand_dims(season=["DJF", "JJA"]), col_dim="season", row_dim="experiment")


def test_colour_bar_ticks_include_the_ends_and_zero():
    from extant.plots import forced_response as fr

    np.testing.assert_allclose(fr._ticks(fr.Scale(np.linspace(-3, 3, 25)), 1), [-3, -1.5, 0, 1.5, 3])
    np.testing.assert_allclose(fr._ticks(fr.Scale(np.linspace(0, 4, 17)), 1), [0, 1, 2, 3, 4])
    np.testing.assert_allclose(fr._ticks(fr.Scale([0, 1, 5], ticks=[0, 5]), 1), [0, 5])


def test_block_grid_paper_figures_and_the_talk_figures(tmp_path):
    pytest.importorskip("cartopy")
    from extant import significance as sig
    from extant.plots import forced_response as fr, slides

    scales = fr.colour_scales("tas")
    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    rng = np.random.default_rng(3)
    coords = {"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"], "lat": lat, "lon": lon}

    def field(offset=0.0, spread=1.0):
        return xr.DataArray(offset + spread * rng.standard_normal((3, 2, lat.size, lon.size)), dims=tuple(coords),
                            coords=coords)

    values = xr.Dataset({"Q05": field(-20, 3), "Mean": field(-15, 3), "Q95": field(-10, 3), "Q95 − Q05": field(6)})
    change = xr.Dataset({"Q05": field(), "Mean": field(), "Q95": field(), "Q95 − Q05": field(0, 0.5)})
    not_significant = xr.Dataset({"Mean": field() > 0, "Q95 − Q05": field() > 0})

    #(c): Bracegirdle's layout: two blocks, each with a colour bar for three columns and one for the width
    panels = fr.block_grid(
        [fr.Block({"hist-nat": values.isel(model=0, experiment=0), "historical": values.isel(model=0, experiment=1)},
                  [scales.temperature] * 3 + [scales.spread]),
         fr.Block({"Change": change.isel(model=0, experiment=1)}, [scales.change] * 3 + [scales.spread_change],
                  not_significant={"Change": not_significant.isel(model=0, experiment=1)})],
        style=fr.PAPER, tag=True, panel_size=1.4,
    )
    assert panels.axes.shape == (3, 4)
    assert len(panels.extras["cbars"]) == 4
    paths = fr.save(panels, tmp_path, "figure")
    assert all(path.exists() for path in paths)

    #(c): The joint classes on their 3 x 3 key
    signs = sig.significant_sign(change["Mean"], ~not_significant["Mean"])
    robust = sig.robust_sign(change["Mean"], ~not_significant["Mean"], threshold=0.66)
    from extant import response_change as rc
    fr.map_grid(rc.joint_class(robust, robust), scales.joint, row_dim=None, col_dim="experiment", style=fr.PAPER)
    fr.map_grid(rc.joint_class(signs, signs), scales.joint)

    regional = xr.Dataset({name: xr.DataArray(rng.standard_normal((3, 2, 2)), dims=("model", "experiment", "season"),
                                              coords={"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"],
                                                      "season": ["DJF", "JJA"]})
                           for name in ("mean_change", "width_change")})
    fr.forcing_dots(regional, {"mean_change": "Mean", "width_change": "Width"}, style=fr.PAPER)

    #(c): The talk figures
    point = {"lat": -77.5, "lon": 0.0}
    members = {model: xr.DataArray(rng.standard_normal((5, 21)), dims=("member", "year")) for model in "ABC"}
    shifted = {model: values + 1 for model, values in members.items()}
    at_point = change["Mean"].sel(experiment="historical", **point)
    counts = sig.significant_counts(change["Mean"].sel(experiment="historical"),
                                    ~not_significant["Mean"].sel(experiment="historical"))
    slides.count_schematic(members, shifted, at_point, at_point * 0 + 0.01, counts, point, "historical")
    slides.joint_key_explainer(scales.joint)
    frames = slides.reveal_frames(change["Mean"].isel(model=0).expand_dims(season=["DJF", "JJA"]), scales.change,
                                  tmp_path / "frames", col_dim="season")
    assert [frame.name for frame in frames] == ["01_hist-GHG.png", "02_historical.png"]


def test_paper_figures_fill_the_page_width_and_ranges_are_checked(tmp_path):
    pytest.importorskip("cartopy")
    from PIL import Image

    from extant.plots import forced_response as fr

    scales = fr.colour_scales("tas")
    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    models = ["CanESM5", "GISS-E2-1-G", "HadGEM3-GC31-LL", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM"]
    field = xr.DataArray(np.random.default_rng(4).normal(0, 2, (6, 4, lat.size, lon.size)),
                         dims=("model", "experiment", "lat", "lon"),
                         coords={"model": models, "experiment": ["hist-GHG", "hist-aer", "hist-totalO3", "historical"],
                                 "lat": lat, "lon": lon})
    #(c): Long model names and five columns still fit 183 mm, once trimmed of white space as when saved
    panels = fr.map_grid(field, scales.change, style=fr.PAPER, tag=True, width=fr.PAGE_WIDTH)
    png, _ = fr.save(panels, tmp_path, "page")[::-1]
    width_mm = Image.open(png).size[0] / 300 * 25.4
    assert 170 < width_mm <= 183

    shares = fr.saturation(field, scales.change)
    assert 0 < shares["above"] < 0.2 and 0 < shares["below"] < 0.2
    assert fr.saturation(field.clip(-1, 1), scales.change) == {"below": 0.0, "above": 0.0}
