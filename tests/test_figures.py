"""Smoke tests: every panel draws without error on small synthetic inputs, on the axes of ``charts``, as the notebooks
draw them."""

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("plotting_modules")

from extant import datatree, era5_evaluation as ev  # noqa: E402
from extant.plots import charts, era5_evaluation as evp  # noqa: E402
from extant import response_change as rc  # noqa: E402
from extant.plots import response_change as rcp  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _axes(n_rows, n_cols, sharex=False, sharey=False, **size):
    """A figure of chart panels, as the notebooks make them: its size from charts.chart_figure_size (``size`` is what
    that takes), its panels from a GridSpec."""
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows, n_cols, **size), layout="constrained")
    return fig.add_gridspec(n_rows, n_cols).subplots(sharex=sharex, sharey=sharey, squeeze=False)


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


def test_distribution_panels(point, tests):
    moments, _ = tests
    one = point["A/historical"].to_dataset()
    summaries = ev.distribution_summary(one["members"], one["era5"], ("raw", "anomaly", "detrended"))
    axes = _axes(1, 3)
    evp.plot_kde(summaries["raw"].sel(season="DJF"), ax=axes[0, 0])
    evp.plot_kde(summaries["anomaly"].sel(season="DJF"), ax=axes[0, 1], members=False)
    evp.plot_qq(summaries["anomaly"].sel(season="DJF"), ax=axes[0, 2])
    ok, message = evp.verdict(moments.sel(model="A", season="DJF"), evp.KDE_STATISTIC["anomaly"])
    evp.status_line(axes[0, 1], 0.98, 0.97, ok, message, ha="right")
    axes[0, 0].figure.legend(handles=evp.distribution_legend_handles(), loc="outside lower center")
    #(c): On its own figure too
    assert evp.plot_qq(summaries["detrended"].sel(season="JJA")) is not None


def test_plume_panels(point):
    plumes = {node.parent.name: ev.ensemble_plume(node["members"], node["era5"], "anomaly") for node in point.leaves}
    axes = _axes(1, 2, sharey=True)
    evp.plot_plume(plumes["A"].sel(season="DJF"), ax=axes[0, 0], context=plumes["B"].sel(season="DJF"))
    evp.plot_plume(plumes["B"].sel(season="DJF"), ax=axes[0, 1], note=False)
    assert len(evp.plume_legend_handles(context=True)) == len(evp.plume_legend_handles()) + 2


def test_statistic_and_rank_panels(point, tests):
    moments, others = tests
    axes = _axes(2, 4)
    evp.member_strips_grid(moments.sel(model="A", season=["DJF", "JJA"]), "season", axes=axes)
    assert evp.statistic_title(moments, "std").endswith("detrended anomalies")
    evp.member_strips_grid(others.sel(model="A"), "season", statistics=["trend", "width"])

    ranks = {label: _ranks(point, treatment) for label, treatment in
             (("Raw values", "raw"), ("Anomalies", "anomaly"), ("Detrended anomalies", "detrended"))}
    axes = _axes(1, 3)
    for ax, tree in zip(axes[0], ranks.values()):
        evp.plot_rank_histogram(tree["A/historical"].to_dataset(), ax=ax)
    evp.plot_grouped_rank_histogram(ranks["Anomalies"]["B/historical"].to_dataset())

    #(c): The rank test step by step
    one = point["A/historical"].to_dataset().sel(season="DJF")
    members, era5 = one["members"] - one["members"].mean("year"), one["era5"] - one["era5"].mean("year")
    result = ev.rank_histogram_test(members, era5, "anomaly")
    years = evp.anatomy_years(result)
    assert years == sorted(years) and 1 <= len(years) <= 3
    evp.plot_rank_counting(members, era5, years)
    evp.plot_rank_through_time(result, years)


def test_summaries(point, tests):
    moments, others = tests
    seasonal = datatree.tree_to_dataset(_ranks(point, "anomaly", ("year",), False), dims=["model", "experiment"])
    rank_rows = seasonal.sel(experiment="historical", drop=True)[["percentile", "verdict", "testable"]]
    rank_rows = rank_rows.sel(statistic=["rank_trend", "dispersion"]).drop_vars(["label", "units"])
    rank_rows = rank_rows.assign_coords(statistic=["rank_trend", "dispersion (anomaly)"])
    table = xr.concat([moments[["percentile", "verdict", "testable"]], others[["percentile", "verdict", "testable"]],
                       rank_rows], dim="statistic")
    axes = _axes(1, 1, n_legends=0, n_colorbars=1)
    image = evp.plot_scorecard(table, ax=axes[0, 0])
    plt.colorbar(image, ax=axes, location="bottom", shrink=0.6)
    axes = _axes(1, 2)
    for ax, season in zip(axes[0], ["DJF", "JJA"]):
        evp.plot_signal_noise(table, season, ax=ax)
        evp.plot_signal_noise(table, season, y="width")
    assert "trend" in evp.signal_noise_label("trend").lower()
    #(c): An unknown statistic (an experiment, say) is its own label
    evp.plot_scorecard(table.sel(statistic=["mean"]).assign_coords(statistic=["hist-nat"]))

    #(c): The toy scenarios: one row, as notebook 02b draws each
    demo = xr.concat([xr.Dataset(dict(zip(("members", "era5"), ev.simulate_ensemble(n_members=20, **kwargs))))
                      for kwargs in ev.SCENARIOS.values()], dim=xr.Variable("scenario", list(ev.SCENARIOS)))
    detrended = demo.map(ev.detrend)
    demo_tests = _tests({"mean": demo.mean("year"), "std": detrended.std("year", ddof=1)})
    anomaly_ranks = ev.rank_histogram_test(demo["members"], demo["era5"], "anomaly", ("year", "season"))
    detrended_ranks = ev.rank_histogram_test(demo["members"], demo["era5"], "detrended", ("year", "season"))
    scenario = list(ev.SCENARIOS)[0]
    axes = _axes(1, 4)
    evp.plot_plume(ev.ensemble_plume(demo["members"], demo["era5"], "raw").sel(scenario=scenario, season="DJF"),
                   ax=axes[0, 0])
    evp.plot_grouped_rank_histogram(anomaly_ranks.sel(scenario=scenario), ax=axes[0, 1], statistics=("dispersion",))
    evp.plot_grouped_rank_histogram(detrended_ranks.sel(scenario=scenario), ax=axes[0, 2], statistics=("dispersion",))
    evp.plot_percentile_ladder(demo_tests.sel(scenario=scenario, season="DJF"), anomaly_ranks.sel(scenario=scenario),
                               detrended_ranks.sel(scenario=scenario), ax=axes[0, 3])
    assert len(evp.scenario_legend_handles()) == 7


def test_point_and_scatter_panels():
    rng = np.random.default_rng(2)
    samples = {e: xr.DataArray(shift + scale * rng.standard_normal((20, 11)), dims=("member", "year"))
               for e, shift, scale in (("hist-nat", 0, 1), ("historical", 2, 1.4), ("hist-GHG", 3, 1.2))}
    shifts = {e: rc.quantile_shift(samples[e], samples["hist-nat"], n_boot=100) for e in ("historical", "hist-GHG")}
    axes = _axes(1, 2, n_legends=0)
    rcp.plot_point_distributions(samples, shifts["historical"], "historical", ax=axes[0, 0])
    rcp.plot_quantile_shifts(shifts, "historical", ax=axes[0, 1])

    coords = {"model": ["A", "B"], "experiment": ["hist-GHG", "historical"], "season": ["DJF", "JJA", "MAM", "SON"]}
    values = lambda: xr.DataArray(rng.standard_normal((2, 2, 4)), dims=tuple(coords), coords=coords)
    regional = xr.Dataset({"mean_change": values(), "width_change": values(), "tail_asymmetry": values()})
    axes = _axes(1, 4, sharex=True, sharey=True)
    for ax, season in zip(axes[0], coords["season"]):
        charts.plot_scatter_by_forcing(regional["mean_change"].sel(season=season),
                                       regional["tail_asymmetry"].sel(season=season), ax=ax)
        rcp.plot_quadrant_labels("mean_change", "tail_asymmetry", ax=ax)
    #(c): The correlation over the points, in the panel
    assert any(text.get_text().startswith("r = ") for text in axes[0, 0].texts)


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

def test_method_panels():
    """The step-by-step schematics and the small figures that follow the data, on synthetic ensembles."""
    import pandas as pd

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

    #(c): The bootstrap of the width: a, the trials (checked against the test's), c
    trial_widths = sig.bootstrap_qrange(nat_final, n_trials=200)
    width = qc.quantile_range(final)
    fig = plt.figure(figsize=(11, 15))
    ax_a, flow_axes, trial_axes, ax_c = methods.schematic_axes(fig)
    assert trial_axes.shape == (3, 3) and len(flow_axes) == 2
    methods.plot_centred_ensembles(nat_final, final, "historical", width, ax=ax_a)
    methods.plot_flow("Draw ...", flow_axes[0])
    methods.plot_width_trials(nat_final, selected, trial_widths, trial_axes)
    methods.plot_trial_distribution(trial_widths, width, float(qc.quantile_range(nat_final)), "historical", 30, ax=ax_c)

    #(c): The bootstrap of the mean, and the permutation test
    trial_means = sig.bootstrap_mean(nat_final, n_trials=200)
    fig = plt.figure(figsize=(11, 15))
    ax_a, flow_axes, trial_axes, ax_c = methods.schematic_axes(fig)
    methods.plot_two_ensembles(nat_final, final, "historical", ax=ax_a)
    methods.plot_mean_trials(nat_final, selected, trial_means, trial_axes)
    methods.plot_trial_distribution(trial_means, final.mean(), float(nat_final.mean()), "historical", 30, ax=ax_c)
    weights = rc.permutation_weights(200, 30, 30, np.random.default_rng(0))
    permutations = weights @ np.concatenate([final.mean("year").values, nat_final.mean("year").values])
    fig = plt.figure(figsize=(11, 15))
    ax_a, flow_axes, trial_axes, ax_c = methods.schematic_axes(fig)
    methods.plot_relabellings(final, nat_final, weights > 0, permutations, "historical", trial_axes)
    methods.plot_permutation_distribution(permutations, float(final.mean() - nat_final.mean()), "historical", ax=ax_c)

    #(c): The SAM phases
    index = xr.DataArray(rng.standard_normal((30, 21)), dims=("member", "year"))
    phases = xr.concat([final.where(index < -0.4), final.where(abs(index) <= 0.4), final.where(index > 0.4)],
                       dim=xr.Variable("phase", ["negative", "neutral", "positive"]))
    methods.plot_sam_index(index, index + 0.5, xr.DataArray([-0.4, 0.4], dims="quantile"), "historical")
    for phase in phases["phase"].values:
        methods.plot_phase_temperatures(phases.sel(phase=phase), phases.sel(phase=phase) + 1.0, "historical")

    #(c): Following the data: a rolling quantile, the smoothing, the change in each quantile, the forced response
    rolling = qc.centred_rolling_quantiles(hist, quantiles=[0.05, 0.5, 0.95])
    methods.plot_window(hist, 1950, 21)
    methods.plot_window_values(hist, 1950, 21)
    methods.plot_rolling_quantiles(rolling, 1950)
    smooth = qc.lowess_matrix_xarray(rolling)
    methods.plot_smoothing(rolling, smooth)
    both = xr.concat([smooth, qc.lowess_matrix_xarray(qc.centred_rolling_quantiles(nat, quantiles=[0.05, 0.5, 0.95]))],
                     dim="experiment").assign_coords(experiment=["historical", "hist-nat"])
    base = nat.quantile([0.05, 0.5, 0.95], dim=["member", "year"])
    methods.plot_smoothed_quantiles(both, base, "historical")
    methods.plot_quantile_changes(both, base, "historical")
    methods.plot_two_samples(final, nat_final, "historical", t=3.0, p=0.01)

    seasons = xr.DataArray(rng.standard_normal((years.size, 4)), dims=("year", "season"),
                           coords={"year": years, "season": ["DJF", "JJA", "MAM", "SON"]})
    methods.plot_seasons_in_time(seasons)
    methods.plot_seasons_split(seasons)
    time = pd.date_range("1990-01-01", periods=72, freq="MS")
    methods.plot_seasonal_means(xr.DataArray(rng.standard_normal(72), dims="time", coords={"time": time}), seasons,
                                slice(1990, 1994))
    image = methods.plot_member_counts(pd.DataFrame({"hist-nat": [50, np.nan], "historical": [65, 10]},
                                                    index=["A", "B"]))
    #(c): B has no hist-nat members: its cell is masked (grey)
    assert image.get_array().mask[1, 0] and not image.get_array().mask[0, 0]

    #(c): The probability ratio's schematics: a, an arrow, b
    thresholds = xr.DataArray([1.6, -1.6], dims="tail", coords={"tail": ["warm", "cold"]})
    trials = xr.DataArray(np.exp(rng.normal(size=(100, 2))), dims=("trial", "tail"), coords={"tail": ["warm", "cold"]})
    fig = plt.figure(figsize=(11, 8.5))
    ax_a, flow_axes, trial_axes, ax_b = methods.schematic_axes(fig, with_trials=False)
    assert trial_axes.size == 0 and len(flow_axes) == 1
    methods.plot_thresholds(final, nat_final, thresholds, "historical", ax=ax_a)
    methods.plot_ratio_trials(trials, "probability ratio", ax=ax_b, alpha=0.45)
    methods.plot_shift_only(final, nat_final, thresholds, 1.0, "historical")
    assert methods.percentile_words(0.05) == ("95th", "5th")


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

def test_zonal_layers(zonal_inputs):
    """A zonal figure as notebook 04c draws it: the four changes across, one line per experiment, the dots, hist-nat's
    band and the ice edges."""
    from extant.plots import zonal as zp

    _, zonal_ds, _, edges = zonal_inputs
    columns = list(rc.EXTREMES)
    season_ds = zonal_ds.sel(season="JJA")
    axes = _axes(2, len(columns), sharex=True, sharey="col")
    charts.bands_grid(season_ds["null_lower"], season_ds["null_upper"], "lat", "model", None, axes=axes[:, 3],
                      color=zp.NULL_COLOR)
    for experiment in season_ds["experiment"].values:
        experiment_ds = season_ds.sel(experiment=experiment)
        charts.lines_grid(experiment_ds[columns].to_dataarray("column"), "lat", "model", "column", axes=axes)
        charts.significant_dots_grid(experiment_ds["mean_change"], experiment_ds["mean_pvalue"] < 0.05, "lat", "model",
                                     None, axes=axes[:, 0])
        zp.ice_edge_lines_grid(edges.sel(experiment=experiment, season="JJA").expand_dims(column=columns), "model",
                               "column", axes=axes)
    axes[0, 0].xaxis.set_major_formatter(zp.latitude_label)
    assert zp.latitude_label(-60) == "60°S"
    #(c): A line per experiment in each panel, and a dashed edge where there is one
    assert len(axes[0, 1].lines) >= season_ds.sizes["experiment"]
    assert zp.plot_ice_edge_line(xr.DataArray(np.nan)) is None


def test_sea_ice_panels(zonal_inputs):
    pytest.importorskip("cartopy")
    from extant import sea_ice
    from extant.plots import sea_ice as sip

    _, _, ice, _ = zonal_inputs
    example = sea_ice.edge_example(ice.sel(model="A", experiment="hist-nat", season="JJA"))
    sip.plot_cap_area(float(example["extent"]), float(example["continent_area"]), float(example["edge_latitude"]))

    #(c): The extent through time: each experiment's band and mean, seasons down and models across
    years = np.arange(1950, 1960)
    extent = xr.DataArray(np.random.default_rng(8).uniform(2e6, 4e6, (5, years.size, 2)),
                          dims=("member", "year", "season"), coords={"year": years, "season": ["DJF", "JJA"]})
    extents = xr.DataTree.from_dict({f"A/{e}": xr.Dataset({"extent": extent, "continent_area": 13.9e6})
                                     for e in ("hist-nat", "historical")})
    summary = sea_ice.extent_summary(extents)
    axes = _axes(2, 1, sharex=True, sharey="row")
    for experiment in summary["experiment"].values:
        experiment_ds = summary.sel(experiment=experiment) / 1e6
        charts.bands_grid(experiment_ds["low"], experiment_ds["high"], "year", "season", "model", axes=axes, alpha=0.15)
        charts.lines_grid(experiment_ds["mean"], "year", "season", "model", axes=axes)
    assert len(axes[0, 0].lines) == summary.sizes["experiment"]

    #(c): The edge as a latitude against ERA5's, every member of every model
    edges = xr.DataArray(np.random.default_rng(9).uniform(-70, -60, (4, 2)), dims=("member", "season"),
                         coords={"season": ["DJF", "JJA"]})
    edges_tree = xr.DataTree.from_dict({f"{m}/historical": xr.Dataset({"edge_latitude": edges}) for m in ("A", "B")})
    sip.plot_edge_latitudes(edges_tree, edges.mean("member"), ["DJF", "JJA"])


def test_lines_and_dots_of_every_model():
    """The change at every percentile, a line per model; the area means of each forcing, a marker per model."""
    rng = np.random.default_rng(2)
    quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
    curves = xr.DataArray(rng.standard_normal((3, 2, quantiles.size)), dims=("model", "experiment", "quantile"),
                          coords={"model": ["A", "B", "C"], "experiment": ["hist-GHG", "historical"],
                                  "quantile": quantiles})
    curves = curves.assign_coords(percentile=("quantile", 100 * quantiles)).swap_dims(quantile="percentile")
    axes = _axes(1, 2, sharey=True)
    models = list(curves["model"].values)
    for j, experiment in enumerate(curves["experiment"].values):
        model_styles = charts.model_lines(experiment, models)
        for k, model in enumerate(models):
            charts.plot_lines(curves.sel(experiment=experiment, model=model), "percentile", ax=axes[0, j],
                              marker=charts.MODEL_MARKERS[k], **model_styles[model])
    axes[0, 0].figure.legend(handles=charts.model_legend_handles(models, experiment="historical"),
                             loc="outside lower center")
    assert len(axes[0, 1].lines) == 3
    #(c): hist-GHG's models in its reds, each with its own line style
    assert [line.get_label() for line in axes[0, 0].lines] == models
    assert len({line.get_color() for line in axes[0, 0].lines}) == 3
    assert len({str(line.get_linestyle()) + str(line.get_linewidth()) for line in axes[0, 0].lines}) == 3

    regional = curves.isel(percentile=0, drop=True)
    axes = _axes(1, 1)
    charts.plot_model_dots(regional, "experiment", ax=axes[0, 0], color={"hist-GHG": "r"})
    #(c): A bar (the median) per forcing, a marker per model
    assert len(axes[0, 0].patches) == 2 and len(axes[0, 0].lines) == 1 + 2 * 3


def test_scale_colours_are_the_settings_of_a_heatmap():
    from extant.plots import forced_response as fr

    colours = fr.Scale(np.linspace(-3, 3, 25), "RdBu_r", "Change").colours()
    assert colours["extend"] == "both" and colours["ticks"] is None and colours["label"] == "Change"
    #(c): A discrete scale: no arrows, and a tick in the middle of each class's colour, named
    colours = fr.class_scale(["no", "yes"], ["white", "k"]).colours()
    assert colours["extend"] == "neither" and colours["ticklabels"] == ["no", "yes"]
    np.testing.assert_allclose(colours["ticks"], [0, 1])

def test_saving_and_the_talk_panels(tmp_path):
    from extant.plots import forced_response as fr, slides

    scales = fr.colour_scales("tas")
    rng = np.random.default_rng(3)

    #(c): Saved as a PDF and a PNG
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    paths = fr.save(fig, tmp_path, "figure")
    assert all(path.exists() for path in paths)

    #(c): The talk figures: one model's test at a point (a significant increase), the joint key's cells, the build-up
    hist_nat_da = xr.DataArray(rng.standard_normal((5, 21)), dims=("member", "year"))
    ax = slides.plot_point_test(hist_nat_da, hist_nat_da + 1, change=1.0, pvalue=0.01, experiment="historical")
    assert "significant increase" in [text.get_text() for text in ax.texts]
    with plt.rc_context(fr.SLIDES.rc):
        axes = _axes(3, 3, column_gap=0.1, row_gap=0.1)
        for i, shift in enumerate((-1.2, 0.0, 1.2)):
            for j, width in enumerate((0.65, 1.0, 1.5)):
                slides.plot_joint_key_cell(shift, width, scales.joint.cmap(i * 3 + j), fill=i * 3 + j != 4,
                                           ax=axes[2 - i, j])
        series = {f: xr.DataArray(rng.standard_normal(10), dims="year", coords={"year": np.arange(1850, 1860)})
                  for f in ("hist-nat", "historical")}
        axes = _axes(1, 1, n_legends=0, top_margin=1.0)
        fig = axes[0, 0].figure
        slides.plot_buildup(series, ["hist-nat"], (-3, 3), ax=axes[0, 0])
        legend = slides.plot_buildup_legend(["hist-nat", "historical"], ["hist-nat"], fig)
        #(c): Every forcing in the legend from the first frame, the one still to come faded
        assert [text.get_alpha() for text in legend.get_texts()] == [1.0, 0.22]


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
