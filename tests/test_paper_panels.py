"""The paper's panels (notebooks 04d, 06, 07) and the fingerprinting panels (08) draw, on the axes of ``charts``."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from extant import fingerprinting as fp  # noqa: E402
from extant.plots import charts, fingerprinting as fpp, forced_response as fr, paper_panels as pp  # noqa: E402

MODELS = ["A", "B", "C"]
EXPERIMENTS = ["hist-GHG", "historical"]
SEASONS = ["DJF", "JJA"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_probability_ratios_in_a_layout():
    """Two blocks of ratios in one figure, each in its part (fig.subfigures): the letters run on from one to the next."""
    rng = np.random.default_rng(0)
    ratios = xr.Dataset({"pr": (("model", "experiment", "tail", "season"), np.exp(rng.normal(size=(3, 2, 2, 2))))},
                        coords=dict(model=MODELS, experiment=EXPERIMENTS, tail=["warm", "cold"], season=SEASONS))
    ratios = ratios.assign(pr_lower=ratios["pr"] / 2, pr_upper=ratios["pr"] * 2, pr_shift_only=ratios["pr"] * 0.5)
    #(c): A ratio of 0 is drawn at the bottom edge, not lost
    ratios["pr"][0, 0, 0, 0] = 0.0
    with plt.rc_context(fr.PAPER.rc):
        fig = plt.figure(figsize=(fr.PAGE_WIDTH, 8), layout="constrained")
        top_fig, bottom_fig = fig.subfigures(2, 1)
        next_tag = 0
        for part in (top_fig, bottom_fig):
            axes = part.add_gridspec(2, 2).subplots(sharex=True, sharey=True)
            pp.probability_ratios_grid(ratios, "tail", "season", axes=axes, shift_only=True)
            next_tag = charts.tag_panels(axes, start=next_tag)
    assert next_tag == 8
    assert axes[0, 0].get_ylim()[0] < pp.RATIO_LIMITS[0]


def test_verification_panels():
    rng = np.random.default_rng(2)
    ranks = xr.Dataset({name: (("model", "season", "rank"), rng.uniform(5, 15, (3, 2, 10)))
                        for name in ("obs_frequency", "pseudo_lower", "pseudo_upper")},
                       coords=dict(model=MODELS, season=SEASONS, rank=np.arange(0.05, 1, 0.1)))
    ranks["percentile"] = (("model", "season", "statistic"), rng.uniform(0, 100, (3, 2, 1)))
    ranks["verdict"] = (("model", "season", "statistic"), np.zeros((3, 2, 1)))
    ranks = ranks.assign_coords(statistic=["dispersion"])
    ax = pp.plot_regional_rank_histogram(ranks.sel(model="A", season="DJF"))
    assert len(ax.patches) == 10 and ax.texts[0].get_text().startswith("✓")
    assert len(pp.rank_histogram_legend_handles()) == 3
    percentile = xr.DataArray(rng.uniform(0, 100, (3, 2, 2)), dims=("model", "experiment", "season"),
                              coords=dict(model=MODELS, experiment=EXPERIMENTS, season=SEASONS))
    image = pp.plot_percentile_table(percentile.sel(season="DJF"), (percentile.sel(season="DJF") > 95).astype(float))
    assert image.get_array().shape == (3, 2)


def test_fingerprinting_panels():
    rng = np.random.default_rng(3)
    table = xr.DataArray(rng.normal(size=(2, 2)), dims=("region", "season"),
                         coords=dict(region=["R1", "R2"], season=SEASONS))
    image = fpp.plot_entry_table(table, fr.Scale(np.linspace(-1, 1, 5), "RdBu_r", "trend"))
    assert image.get_array().shape == (2, 2)
    betas = fp.fit_table({("2 signals", "ols"): {"beta": [1, 0.5], "lower": [0.5, 0], "upper": [1.5, 1]},
                          ("2 signals", "tls"): {"beta": [1.1, 0.4], "lower": [0.4, -0.2], "upper": [1.8, 1.1]}},
                         {"2 signals": ["ANT", "NAT"]})
    fpp.plot_scaling_factors(betas.sel(set="2 signals"))
    attributable = xr.Dataset({"trend": table.expand_dims(signal=["ANT", "NAT"])})
    attributable = attributable.assign(lower=attributable["trend"] - 0.1, upper=attributable["trend"] + 0.1)
    ax = fpp.plot_attributable_trends(table.sel(region="R1"), attributable.sel(region="R1"))
    assert sorted(ax.get_legend_handles_labels()[1]) == ["ANT", "ERA5", "NAT"]
    beta = betas.sel(set="2 signals", fit="tls")
    each = xr.DataArray(rng.normal(size=(2, 2)), dims=("model", "signal"), coords=dict(model=["A", "B"],
                                                                                         signal=["ANT", "NAT"]))
    fpp.plot_model_betas(beta, each, each)
    matrix = xr.DataArray(np.eye(2), dims=("signal", "signal_2"), coords=dict(signal=["ANT", "NAT"],
                                                                               signal_2=["ANT", "NAT"]))
    fpp.plot_correlations(matrix)
    fpp.plot_residual_null({"statistic": 3.0, "null": rng.chisquare(4, 200), "pvalue": 0.4})


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
    fig = plt.figure(figsize=charts.chart_figure_size(1, 2, n_legends=0, n_colorbars=1), layout="constrained")
    axes = fig.add_gridspec(1, 2).subplots(sharex=True, sharey=True, squeeze=False)
    height_scale = 1.8 / float(densities.max())
    for ax, season in zip(axes[0], SEASONS):
        pp.plot_ridgelines(densities.sel(season=season), means.sel(season=season), "experiment", height_scale,
                           lines=lines.sel(season=season), ax=ax)
    #(c): Each ridge's mean written beside it; the top ridge (hist-nat) on the highest baseline
    assert [text.get_text() for text in axes[0, 0].texts][0].startswith(("+", "-"))
    assert charts.tag_panels(axes) == 2
