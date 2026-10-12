"""The candidate panels (notebook 07), the fingerprinting figures (08) and the probability-ratio schematic (03) draw."""

from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from extant import fingerprinting as fp, significance as sig  # noqa: E402
from extant.plots import fingerprinting as fpp, forced_response as fr, methods, paper_panels as pp  # noqa: E402

pytest.importorskip("cartopy")

MODELS = ["A", "B", "C"]
EXPERIMENTS = ["hist-GHG", "historical"]
SEASONS = ["DJF", "JJA"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _maps(rng):
    lat, lon = np.arange(-90, -39, 10.0), np.arange(-180, 180, 30.0)
    values = rng.normal(size=(3, 2, 2, lat.size, lon.size))
    return xr.DataArray(values, dims=("model", "experiment", "season", "lat", "lon"),
                        coords=dict(model=MODELS, experiment=EXPERIMENTS, season=SEASONS, lat=lat, lon=lon))


def test_map_block_and_compose():
    rng = np.random.default_rng(0)
    change = _maps(rng)
    agreement = sig.model_agreement(change, np.abs(change) > 0.5)
    hatches = [pp.Hatch(agreement[name], *pp.AGREEMENT[name]) for name in pp.AGREEMENT]
    scales = fr.colour_scales("tas")
    block = partial(pp.map_block, agreement["median"], scales.change, row_dim="experiment", columns=SEASONS,
                    hatches=hatches, notes=agreement["n_models"].max(["lat", "lon"]),
                    contours=pp.Contours(change.median("model"), (-1, 1), "lines"))
    panels = block(tag_start=0)
    assert panels.axes.shape == (2, 2)
    assert panels.extras["next_tag"] == 4
    #(c): A discrete scale, a single row, and a block title
    classes = (np.abs(change.isel(model=0, experiment=0)) > 1).astype(float)
    panels = pp.map_block(classes, pp.class_scale(["no", "yes"], ["white", "k"]), columns=SEASONS, title="Classes")
    assert panels.axes.shape == (1, 2)
    #(c): Blocks put together: the letters run on
    ratios = xr.Dataset({"pr": (("model", "experiment", "tail", "season"), np.exp(rng.normal(size=(3, 2, 2, 2))))},
                        coords=dict(model=MODELS, experiment=EXPERIMENTS, tail=["warm", "cold"], season=SEASONS))
    ratios = ratios.assign(pr_lower=ratios["pr"] / 2, pr_upper=ratios["pr"] * 2, pr_mean_only=ratios["pr"] * 0.5)
    layout = pp.compose([[block], [partial(pp.pr_points, ratios, mean_only=True)]], heights=[4, 3])
    assert [b.extras["next_tag"] for b in layout.extras["blocks"]] == [4, 8]


def test_line_and_point_panels():
    rng = np.random.default_rng(1)
    regional = xr.DataArray(rng.normal(size=(3, 2, 2)), dims=("model", "experiment", "season"),
                            coords=dict(model=MODELS, experiment=EXPERIMENTS, season=SEASONS))
    assert pp.scatter_by_season(regional, regional * 2, "x", "y").axes.shape == (1, 2)
    assert pp.forcing_bars({"Δ": regional}).axes.shape == (1, 2)
    curves = xr.DataArray(rng.normal(size=(3, 2, 19, 2, 2)), dims=("model", "experiment", "quantile", "region", "season"),
                          coords=dict(model=MODELS, experiment=EXPERIMENTS, quantile=np.arange(1, 20) / 20,
                                      region=["R1", "R2"], season=SEASONS))
    assert pp.quantile_curves(curves).axes.shape == (2, 2)
    plume = xr.DataArray(rng.normal(size=(2, 3, 30, 2)), dims=("experiment", "quantile", "year", "season"),
                         coords=dict(experiment=EXPERIMENTS, quantile=[0.05, 0.5, 0.95], year=np.arange(1985, 2015),
                                     season=SEASONS))
    era5 = plume.isel(experiment=0, quantile=1)
    assert pp.plumes(plume, era5).axes.shape == (1, 2)


def test_verification_panels():
    rng = np.random.default_rng(2)
    ranks = xr.Dataset({name: (("model", "season", "rank"), rng.uniform(5, 15, (3, 2, 10)))
                        for name in ("obs_frequency", "pseudo_lower", "pseudo_upper")},
                       coords=dict(model=MODELS, season=SEASONS, rank=np.arange(0.05, 1, 0.1)))
    ranks["percentile"] = (("model", "season", "statistic"), rng.uniform(0, 100, (3, 2, 1)))
    ranks["verdict"] = (("model", "season", "statistic"), np.zeros((3, 2, 1)))
    ranks = ranks.assign_coords(statistic=["dispersion"])
    assert pp.rank_histograms(ranks).axes.shape == (3, 2)
    percentile = xr.DataArray(rng.uniform(0, 100, (3, 2, 2)), dims=("model", "experiment", "season"),
                              coords=dict(model=MODELS, experiment=EXPERIMENTS, season=SEASONS))
    assert pp.percentile_tables(percentile, (percentile > 95).astype(float)).axes.shape == (1, 2)


def test_fingerprinting_figures():
    rng = np.random.default_rng(3)
    table = xr.DataArray(rng.normal(size=(2, 2)), dims=("region", "season"),
                         coords=dict(region=["R1", "R2"], season=SEASONS))
    fpp.entry_tables({"y": table, "x": table}, fr.Scale(np.linspace(-1, 1, 5), "RdBu_r", "trend"))
    betas = fp.fit_table({("2 signals", "ols"): {"beta": [1, 0.5], "lower": [0.5, 0], "upper": [1.5, 1]},
                          ("2 signals", "tls"): {"beta": [1.1, 0.4], "lower": [0.4, -0.2], "upper": [1.8, 1.1]}},
                         {"2 signals": ["ANT", "NAT"]})
    fpp.scaling_factor_plot(betas)
    attributable = xr.Dataset({"trend": table.expand_dims(signal=["ANT", "NAT"])})
    attributable = attributable.assign(lower=attributable["trend"] - 0.1, upper=attributable["trend"] + 0.1)
    fpp.attributable_trends(table, attributable)
    beta = betas.sel(set="2 signals", fit="tls")
    each = xr.DataArray(rng.normal(size=(2, 2)), dims=("model", "signal"), coords=dict(model=["A", "B"],
                                                                                         signal=["ANT", "NAT"]))
    fpp.model_betas(beta, each, each)
    matrix = xr.DataArray(np.eye(2), dims=("signal", "signal_2"), coords=dict(signal=["ANT", "NAT"],
                                                                               signal_2=["ANT", "NAT"]))
    fpp.correlation_maps({"a": matrix})
    fpp.residual_null({"set": {"statistic": 3.0, "null": rng.chisquare(4, 200), "pvalue": 0.4}})


def test_probability_ratio_schematic():
    rng = np.random.default_rng(4)
    experiment = xr.DataArray(rng.normal(1, 1, (20, 21)), dims=("member", "year"))
    hist_nat = xr.DataArray(rng.normal(0, 1, (20, 21)), dims=("member", "year"))
    thresholds = xr.DataArray([1.6, -1.6], dims="tail", coords={"tail": ["warm", "cold"]})
    trials = xr.Dataset({name: (("trial", "tail"), np.exp(rng.normal(size=(100, 2)))) for name in ("pr", "pr_mean_only")},
                        coords={"tail": ["warm", "cold"]})
    trials["pr"][0, 1] = 0.0
    panels = methods.probability_ratio_schematic(experiment, hist_nat, thresholds, 1.0, trials, "historical")
    assert panels.axes.size == 3


def test_ridgelines():
    from extant.stats import fast_kde_xr
    rng = np.random.default_rng(5)
    values = xr.DataArray(rng.normal(size=(2, 3, 40, 2)) + np.array([0.0, 1.5])[:, None, None, None],
                          dims=("experiment", "model", "member", "season"),
                          coords=dict(experiment=["hist-nat", "historical"], model=MODELS, season=SEASONS))
    x = np.linspace(-5, 6, 111)
    densities = fast_kde_xr(values, x, dim=["model", "member"], bandwidth=0.3)
    means = values.mean(["model", "member"])
    lines = values.sel(experiment="hist-nat").quantile([0.05, 0.95], dim=["model", "member"])
    panels = pp.ridgelines(densities, means, columns=SEASONS, lines=lines, colorbar_label="Mean", tag_start=0)
    assert panels.axes.shape == (1, 2)
    assert panels.extras["next_tag"] == 2
