"""Smoke tests: every figure function draws without error on small synthetic inputs."""

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("plotting_modules")

from extant import datatree, era5_evaluation as ev  # noqa: E402
from extant.plots import era5_evaluation as evp  # noqa: E402
from extant import response_change as rc  # noqa: E402
from extant.plots import response_change as rcp  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _tests(statistics):
    """``ev.locate`` of each statistic (members and ERA5 in one Dataset), along a statistic dim, as in notebook 02."""
    return xr.concat([ev.locate(values["members"], values["era5"]) for values in statistics.values()],
                     dim=xr.Variable("statistic", list(statistics)))


@pytest.fixture(scope="module")
def point():
    """Two toy models at one point, as notebook 02 has them: members and ERA5 on the same samples, side by side."""
    nodes = {}
    for name, n, kwargs in (("A", 25, {}), ("B", 30, {"noise": 1.6})):
        ens, obs = ev.align(*ev.simulate_ensemble(n_members=n, seed=n, **kwargs))
        nodes[f"{name}/historical"] = xr.Dataset({"members": ens, "era5": obs})
    return xr.DataTree.from_dict(nodes)


@pytest.fixture(scope="module")
def tests(point):
    """The moment test and the other statistics of every model, on (model, statistic, season)."""
    point_ds = datatree.tree_to_dataset(point, dims=["model", "experiment"]).sel(experiment="historical", drop=True)
    detrended_ds = point_ds.map(ev.detrend)
    moments = _tests({
        "mean": point_ds.mean("year"),
        "std": detrended_ds.std("year", ddof=1),
        "skewness": detrended_ds.map(ev.skewness),
        "kurtosis": detrended_ds.map(ev.excess_kurtosis),
    }).assign_coords(treatment=("statistic", ["raw", "detrended", "detrended", "detrended"]))
    others = _tests({
        "trend": point_ds.map(ev.trend),
        "width": detrended_ds.map(ev.quantile_width),
    }).assign_coords(treatment=("statistic", ["raw", "detrended"]))
    return moments, others


def _ranks(point, treatment, pool_dims=("year", "season"), keep_histograms=True):
    return point.map_over_datasets(lambda ds: ev.rank_histogram_test(
        ds["members"], ds["era5"], treatment, pool_dims=pool_dims, keep_histograms=keep_histograms)
        if ds.data_vars else ds)


def test_distribution_grids(point, tests):
    moments, _ = tests
    one = point["A/historical"].to_dataset()
    evp.distribution_grid(ev.distribution_summary(one["members"], one["era5"], ("raw", "anomaly", "detrended")),
                          moments=moments.sel(model="A"), title="one model")
    every = {node.parent.name: ev.distribution_summary(node["members"], node["era5"]) for node in point.leaves}
    evp.distribution_grid(every, season="DJF", moments=moments)


def test_plume_grids(point):
    plumes = {node.parent.name: xr.concat([ev.ensemble_plume(node["members"], node["era5"], t)
                                           for t in ("raw", "anomaly")], dim=xr.Variable("treatment", ["raw", "anomaly"]))
              for node in point.leaves}
    evp.plume_grid(plumes["A"], context=plumes["B"].sel(treatment=["anomaly"]))
    every = xr.concat([plume.sel(treatment="anomaly") for plume in plumes.values()], dim="model",
                      coords="minimal", compat="override").assign_coords(model=list(plumes))
    evp.plume_grid(every, row_dim="model", col_dim="season", sharey=True)


def test_statistic_and_rank_figures(point, tests):
    moments, others = tests
    evp.statistic_grid(moments.sel(model="A"))
    evp.statistic_grid(others.sel(model="A"), statistics=["trend", "width"])
    ranks = {label: _ranks(point, treatment) for label, treatment in
             (("Raw values", "raw"), ("Anomalies", "anomaly"), ("Detrended anomalies", "detrended"))}
    evp.rank_histogram_row({label: tree["A/historical"].to_dataset() for label, tree in ranks.items()})
    evp.rank_histogram_grid(ranks)
    one = point["A/historical"].to_dataset().sel(season="DJF")
    result = ev.rank_histogram_test(one["members"], one["era5"], "anomaly")
    evp.rank_anatomy(one["members"] - one["members"].mean("year"), one["era5"] - one["era5"].mean("year"), result)


def test_summaries(point, tests):
    moments, others = tests
    seasonal = datatree.tree_to_dataset(_ranks(point, "anomaly", ("year",), False), dims=["model", "experiment"])
    rank_rows = seasonal.sel(experiment="historical", drop=True)[["percentile", "verdict", "testable"]]
    rank_rows = rank_rows.sel(statistic=["rank_trend", "dispersion"]).drop_vars(["label", "units"])
    rank_rows = rank_rows.assign_coords(statistic=["rank_trend", "dispersion (anomaly)"])
    table = xr.concat([moments[["percentile", "verdict", "testable"]], others[["percentile", "verdict", "testable"]],
                       rank_rows], dim="statistic")
    evp.scorecard(table)
    evp.signal_noise_diagram(table)
    evp.signal_noise_diagram(table, y="width")
    #(c): An unknown statistic (an experiment, say) is its own label
    evp.scorecard(table.sel(statistic=["mean"]).assign_coords(statistic=["hist-nat"]))

    demo = xr.concat([xr.Dataset(dict(zip(("members", "era5"), ev.simulate_ensemble(n_members=20, **kwargs))))
                      for kwargs in ev.SCENARIOS.values()], dim=xr.Variable("scenario", list(ev.SCENARIOS)))
    detrended = demo.map(ev.detrend)
    demo_tests = _tests({"mean": demo.mean("year"), "std": detrended.std("year", ddof=1)})
    evp.scenario_grid(ev.ensemble_plume(demo["members"], demo["era5"], "raw"),
                      ev.rank_histogram_test(demo["members"], demo["era5"], "anomaly", ("year", "season")),
                      ev.rank_histogram_test(demo["members"], demo["era5"], "detrended", ("year", "season")),
                      demo_tests)


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


def test_edge_classes_of_the_spatial_evaluation(spatial_table):
    """Every tested point has one of the classes of the paper's Fig. 5; an untested point has none."""
    classes = evp.edge_class(spatial_table)
    assert bool((classes.notnull() == spatial_table["adequate"].notnull()).all())
    assert set(np.unique(classes.values[np.isfinite(classes.values)])) <= set(range(len(evp.EDGE_CLASSES)))


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
    selected = sig.bootstrap_draws(30, n_trials=200)
    trial_widths = sig.bootstrap_qrange(nat_final, n_trials=200)
    width = qc.quantile_range(final)
    methods.bootstrap_schematic(nat_final, final, trial_widths, selected, width,
                                sig.pvalue_two_sided(trial_widths, width), "historical")
    trial_means = sig.bootstrap_mean(nat_final, n_trials=200)
    mean = final.mean()
    methods.bootstrap_mean_schematic(nat_final, final, trial_means, selected, mean,
                                     sig.pvalue_two_sided(trial_means, mean), "historical")
    index = xr.DataArray(rng.standard_normal((30, 21)), dims=("member", "year"))
    phases = xr.concat([final.where(index < -0.4), final.where(abs(index) <= 0.4), final.where(index > 0.4)],
                       dim=xr.Variable("phase", ["negative", "neutral", "positive"]))
    methods.sam_phase_demo(index, index + 0.5, xr.DataArray([-0.4, 0.4], dims="quantile"), phases, phases + 1.0,
                           "historical")
    weights =rc.permutation_weights(200, 30, 30, np.random.default_rng(0))
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

    _, _, ice, _ = zonal_inputs
    example = sea_ice.edge_example(ice.sel(model="A", experiment="hist-nat", season="JJA"))
    sip.plot_cap_area(float(example["extent"]), float(example["continent_area"]), float(example["edge_latitude"]))

    years = np.arange(1950, 1960)
    extent = xr.DataArray(np.random.default_rng(8).uniform(2e6, 4e6, (5, years.size, 2)),
                          dims=("member", "year", "season"), coords={"year": years, "season": ["DJF", "JJA"]})
    extents = xr.DataTree.from_dict({f"A/{e}": xr.Dataset({"extent": extent, "continent_area": 13.9e6})
                                     for e in ("hist-nat", "historical")})
    sip.extent_timeseries(sea_ice.extent_summary(extents))


def test_quantile_curves():
    from extant.plots import forced_response as fr

    rng = np.random.default_rng(2)
    quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
    curves = xr.DataArray(rng.standard_normal((3, 2, quantiles.size)), dims=("model", "experiment", "quantile"),
                          coords={"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"],
                                  "quantile": quantiles})
    fr.quantile_curves(curves, title="Every quantile")
    fr.quantile_curves(curves.expand_dims(season=["DJF", "JJA"]), col_dim="season", row_dim="experiment")


def test_scale_colours_are_the_settings_of_a_heatmap():
    from extant.plots import forced_response as fr

    colours = fr.Scale(np.linspace(-3, 3, 25), "RdBu_r", "Change").colours()
    assert colours["extend"] == "both" and colours["ticks"] is None and colours["label"] == "Change"
    #(c): A discrete scale: no arrows, and a tick in the middle of each class's colour, named
    colours = fr.class_scale(["no", "yes"], ["white", "k"]).colours()
    assert colours["extend"] == "neither" and colours["ticklabels"] == ["no", "yes"]
    np.testing.assert_allclose(colours["ticks"], [0, 1])


def test_saving_and_the_dots_and_talk_figures(tmp_path):
    from extant.plots import forced_response as fr, slides

    scales = fr.colour_scales("tas")
    rng = np.random.default_rng(3)

    #(c): Saved as a PDF and a PNG
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    paths = fr.save(fig, tmp_path, "figure")
    assert all(path.exists() for path in paths)

    regional = xr.Dataset({name: xr.DataArray(rng.standard_normal((3, 2, 2)), dims=("model", "experiment", "season"),
                                              coords={"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"],
                                                      "season": ["DJF", "JJA"]})
                           for name in ("mean_change", "width_change")})
    fr.forcing_dots(regional, {"mean_change": "Mean", "width_change": "Width"}, style=fr.PAPER)

    #(c): The talk figures: one model's test at a point (a significant increase), and the joint key
    hist_nat_da = xr.DataArray(rng.standard_normal((5, 21)), dims=("member", "year"))
    ax = slides.plot_point_test(hist_nat_da, hist_nat_da + 1, change=1.0, pvalue=0.01, experiment="historical")
    assert "significant increase" in [text.get_text() for text in ax.texts]
    slides.joint_key_explainer(scales.joint)


def test_paper_figures_fit_the_page_and_ranges_are_checked(tmp_path):
    pytest.importorskip("cartopy")
    from PIL import Image

    from extant.plots import antarctica, forced_response as fr

    scales = fr.colour_scales("tas")
    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    experiments = ["hist-GHG", "hist-aer", "hist-totalO3", "historical"]
    models = ["CanESM5", "GISS-E2-1-G", "HadGEM3-GC31-LL", "MIROC6", "MPI-ESM1-2-LR", "NorESM2-LM"]
    field = xr.DataArray(np.random.default_rng(4).normal(0, 2, (4, 6, lat.size, lon.size)),
                         dims=("experiment", "model", "lat", "lon"),
                         coords={"experiment": experiments, "model": models, "lat": lat, "lon": lon})
    #(c): As notebook 06 draws them: maps as big as fit 183 mm, long model names across, letters; once trimmed of
    #(c): white space as when saved, no wider than the page
    panel_size = antarctica.panel_size_to_fit(4, 6, 1, width=fr.PAGE_WIDTH, height=fr.PAGE_HEIGHT, title_height=0.35)
    with plt.rc_context(fr.PAPER.rc):
        fig = plt.figure(figsize=antarctica.map_figure_size(4, 6, 1, panel_size=panel_size, title_height=0.35))
        axes, cbar_axes = antarctica.map_axes(fig, 4, 6, 1, title_height=0.35)
        antarctica.heatmaps_of_antarctica(field, "experiment", "model", axes=axes, cax=cbar_axes[0],
                                          **scales.change.colours())
        antarctica.label_columns(axes, models, fontsize=fr.PAPER.label_size)
        antarctica.label_rows(axes, experiments, fontsize=fr.PAPER.label_size)
        antarctica.tag_panels(axes, fontsize=fr.PAPER.tag_size)
    png, _ = fr.save(fig, tmp_path, "page")[::-1]
    width_mm = Image.open(png).size[0] / 300 * 25.4
    assert 160 < width_mm <= 183

    shares = fr.saturation(field, scales.change)
    assert 0 < shares["above"] < 0.2 and 0 < shares["below"] < 0.2
    assert fr.saturation(field.clip(-1, 1), scales.change) == {"below": 0.0, "above": 0.0}
