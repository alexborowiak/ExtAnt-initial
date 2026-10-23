"""The axes of every chart (extant.plots.charts): the layout's sizes, sharing, legends, and a block in a bigger figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

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


def test_panels_come_out_at_their_size_with_the_legends_under_them():
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=2, n_cols=3, n_legends=2))
    axes, legend_axes = charts.chart_axes(fig, n_rows=2, n_cols=3, n_legends=2)
    assert axes.shape == (2, 3) and len(legend_axes) == 2
    position = axes[0, 0].get_position()
    assert position.width * fig.get_figwidth() == pytest.approx(charts.PANEL_WIDTH, rel=0.01)
    assert position.height * fig.get_figheight() == pytest.approx(charts.PANEL_HEIGHT, rel=0.01)
    #(c): The legends below the bottom row, top first, with nothing drawn round them
    tops = [ax.get_position().y1 for ax in legend_axes]
    assert tops == sorted(tops, reverse=True) and tops[0] < axes[-1, 0].get_position().y0
    assert not legend_axes[0].axison
    #(c): Every panel styled: no top or right spine
    assert not any(ax.spines["top"].get_visible() or ax.spines["right"].get_visible() for ax in axes.ravel())


def test_colour_bars_go_under_the_legends():
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=1, n_cols=2, n_legends=1, n_colorbars=1))
    axes, legend_axes = charts.chart_axes(fig, n_rows=1, n_cols=2, n_legends=1, n_colorbars=1)
    legend_ax, cbar_ax = legend_axes
    assert cbar_ax.get_position().y1 <= legend_ax.get_position().y0 + 1e-9
    assert cbar_ax.get_position().height * fig.get_figheight() == pytest.approx(0.23, rel=0.01)
    fig.colorbar(axes[0, 0].imshow([[0, 1]]), cax=cbar_ax, orientation="horizontal")


def test_small_panels_keep_their_gaps_in_inches():
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=3, n_cols=2, panel_width=1.5, panel_height=0.8))
    axes, _ = charts.chart_axes(fig, n_rows=3, n_cols=2)
    top, below = axes[0, 0].get_position(), axes[1, 0].get_position()
    assert top.height * fig.get_figheight() == pytest.approx(0.8, rel=0.01)
    assert (top.y0 - below.y1) * fig.get_figheight() == pytest.approx(charts.ROW_GAP, rel=0.01)
    assert (axes[0, 1].get_position().x0 - top.x1) * fig.get_figwidth() == pytest.approx(charts.COLUMN_GAP, rel=0.01)


def test_shared_axes_label_only_the_outer_panels():
    fig = plt.figure(figsize=charts.chart_figure_size(n_rows=2, n_cols=2))
    axes, _ = charts.chart_axes(fig, n_rows=2, n_cols=2, sharex=True, sharey=True)
    axes[1, 1].plot([0, 10], [0, 5])
    assert axes[0, 0].get_xlim() == axes[1, 1].get_xlim() and axes[0, 0].get_ylim() == axes[1, 1].get_ylim()
    fig.canvas.draw()
    assert not any(label.get_visible() for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels())
    assert any(label.get_visible() for label in axes[1, 0].get_xticklabels())
    assert any(label.get_visible() for label in axes[1, 0].get_yticklabels())


def test_a_block_of_panels_in_part_of_a_figure():
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(2, 1)
    other_ax = fig.add_subplot(grid[0])
    axes, legend_axes = charts.chart_axes(fig, n_rows=1, n_cols=3, n_legends=1, subplot_spec=grid[1])
    assert axes.shape == (1, 3) and axes[0, 0].get_position().y1 < other_ax.get_position().y0
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
