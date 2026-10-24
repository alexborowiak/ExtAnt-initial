"""The charts' layers (extant.plots.charts): each styles its panel, and each grid fills a GridSpec from a DataTree or a
DataArray."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from extant.datatree import skip_empty  # noqa: E402
from extant.plots import charts  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_figure_size_and_the_width_that_fits_a_page():
    #(c): The docstrings' examples
    assert charts.chart_figure_size(n_rows=2, n_cols=3, n_legends=1) == pytest.approx((10.2, 6.6))
    assert charts.panel_width_to_fit(n_cols=4, width=7.2) == pytest.approx(1.2375)
    #(c): At that width the figure is the page's width
    width, _ = charts.chart_figure_size(2, 4, panel_width=charts.panel_width_to_fit(4, 7.2))
    assert width == pytest.approx(7.2)
    #(c): Room in inches as a GridSpec's fractions, in a figure and in one part of it
    fig = plt.figure(figsize=(10, 5))
    assert charts.margins(fig, left=1.0, right=0.5, top=1.0, bottom=0.5) == pytest.approx(
        dict(left=0.1, right=0.95, top=0.8, bottom=0.1))
    top_fig, _ = fig.subfigures(2, 1)
    assert charts.margins(top_fig, bottom=0.5)["bottom"] == pytest.approx(0.2)


def _cycle_tree():
    """/<model>/historical, each model's extent on (member, month), with a different number of members each."""
    rng = np.random.default_rng(0)
    months = np.arange(1, 13)
    return xr.DataTree.from_dict({
        f"{model}/historical": xr.Dataset({"extent": (("member", "month"), rng.normal(size=(n_members, 12)))},
                                          coords={"month": months})
        for model, n_members in [("A", 3), ("B", 2), ("C", 5), ("D", 1), ("E", 4), ("F", 2)]})


def test_a_layer_styles_its_panel_once():
    #(c): No axis given: a new figure with one panel, styled
    lines = charts.plot_lines(_cycle_tree()["A/historical"]["extent"], "month", title="A")
    ax = lines[0].axes
    assert len(lines) == 3 and ax.get_title(loc="left") == "A"
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    #(c): A layer drawn later leaves what an earlier one changed: the model dots' x gridlines stay off
    ax = plt.figure(layout="constrained").add_gridspec(1, 1).subplots()
    charts.plot_model_dots(xr.DataArray([[1.0, 2.0]], dims=("model", "zone"), coords={"model": ["A"], "zone": [0, 1]}),
                           "zone", ax=ax)
    charts.plot_lines(xr.DataArray([0.0, 1.0], dims="zone", coords={"zone": [0, 1]}), "zone", ax=ax)
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert any(line.get_visible() for line in ax.get_ygridlines())


def test_grid_panels_of_a_tree_and_of_a_dataarray():
    #(c): A tree: one panel per leaf, in its order, titled with the leaf's path
    panels, shape = charts.grid_panels(_cycle_tree(), variable="extent")
    assert [title for _, title in panels] == [f"{model}/historical" for model in "ABCDEF"] and shape == (1, 6)
    assert panels[2][0].sizes["member"] == 5
    #(c): A DataArray: row by row, untitled
    da = xr.DataArray(np.arange(24.0).reshape(2, 3, 4), dims=("season", "model", "year"),
                      coords={"season": ["DJF", "JJA"], "model": ["a", "b", "c"], "year": np.arange(4)})
    panels, shape = charts.grid_panels(da, "season", "model")
    assert shape == (2, 3) and panels[4][1] is None
    assert float(panels[4][0][0]) == float(da.sel(season="JJA", model="b")[0])
    #(c): A Dataset's variable, a single row
    panels, shape = charts.grid_panels(da.to_dataset(name="x"), None, "model", variable="x")
    assert shape == (1, 3)


def test_lines_grid_fills_a_gridspec_from_a_tree():
    tree = _cycle_tree()
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=2, n_cols=3), layout="constrained")
    axes = fig.add_gridspec(2, 3).subplots(sharex=True, sharey=True)
    charts.lines_grid(tree, "month", axes=axes, variable="extent", **charts.MEMBER_LINE)
    charts.lines_grid(tree.mean("member"), "month", axes=axes, variable="extent", **charts.ENSEMBLE_MEAN_LINE)
    for ax in axes.ravel():
        charts.plot_lines(xr.DataArray(np.zeros(12), dims="month", coords={"month": np.arange(1, 13)}), "month", ax=ax,
                          **charts.ERA5_LINE)
    #(c): Two rows of three, row by row: each model's members, their mean and ERA5
    assert [len(ax.lines) for ax in axes.ravel()] == [5, 4, 7, 3, 6, 4]
    assert axes[1, 0].get_title(loc="left") == "D/historical"
    #(c): Too few axes is an error, not a model left out
    with pytest.raises(ValueError, match="6 panels to draw but only 3 axes"):
        charts.lines_grid(tree, "month", axes=axes[0], variable="extent")
    #(c): No axes: a new figure, the leaves in one row
    assert charts.lines_grid(tree, "month", variable="extent").shape == (1, 6)


def test_the_other_layers_take_a_tree_too():
    tree = _cycle_tree()
    axes = plt.figure(layout="constrained").add_gridspec(2, 3).subplots()
    charts.bands_grid(tree.min("member"), tree.max("member"), "month", axes=axes, variable="extent", alpha=0.3)
    quantiles_tree = tree.map_over_datasets(skip_empty(lambda ds: ds.quantile([0.1, 0.9], "member")))
    charts.quantile_bands_grid(quantiles_tree, [(0.1, 0.9)], "month", axes=axes, variable="extent")
    mean_tree = tree.mean("member")
    charts.significant_dots_grid(mean_tree, mean_tree > 0, "month", axes=axes, variable="extent")
    assert all(len(ax.collections) == 2 for ax in axes.ravel())
    assert axes[0, 2].get_title(loc="left") == "C/historical"
    #(c): The dots only where the mean is above 0
    n_dots = len(axes[0, 0].lines[0].get_xdata())
    assert n_dots == int((mean_tree["A/historical"]["extent"] > 0).sum())
    #(c): Model dots: each leaf on (model, x_dim)
    dots_tree = xr.DataTree.from_dict({
        season: xr.Dataset({"change": (("model", "zone"), np.ones((2, 3)))}, coords={"model": ["a", "b"], "zone": [0, 1, 2]})
        for season in ["DJF", "JJA"]})
    dot_axes = charts.model_dots_grid(dots_tree, "zone", variable="change")
    assert dot_axes.shape == (1, 2) and dot_axes[0, 1].get_title(loc="left") == "JJA"


def test_panel_letters_carry_on():
    axes = plt.figure(layout="constrained").add_gridspec(1, 3).subplots(squeeze=False)
    assert charts.tag_panels(axes, start=2) == 5
    assert axes[0, 0].texts[0].get_text() == "c)"


def test_each_model_a_shade_of_its_experiments_colour_with_its_own_line():
    import matplotlib.colors as mcolors
    from plotting_modules.constants import FORCING_COLORS

    models = [f"M{k}" for k in range(8)]
    shades = charts.model_shades(FORCING_COLORS["hist-GHG"], len(models))
    #(c): Eight different reds, light to dark
    lightness = [sum(mcolors.to_rgb(shade)) for shade in shades]
    assert len(set(shades)) == 8 and lightness == sorted(lightness, reverse=True)
    #(c): Each model's line differs from every other model's by its style or width, and is the same in every
    #(c): experiment but for the colour
    ghg_styles, nat_styles = charts.model_lines("hist-GHG", models), charts.model_lines("hist-nat", models)
    assert len({(str(style["linestyle"]), style["linewidth"]) for style in ghg_styles.values()}) == 8
    assert all(ghg_styles[model]["linestyle"] == nat_styles[model]["linestyle"] for model in models)
    assert ghg_styles["M0"]["color"] != nat_styles["M0"]["color"]
    #(c): A single model keeps the experiment's own colour
    assert charts.model_shades(FORCING_COLORS["hist-GHG"], 1) == [FORCING_COLORS["hist-GHG"].lower()]
