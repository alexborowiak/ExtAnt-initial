"""The maps of Antarctica, layer by layer (extant.plots.antarctica): the layout's sizes, and each layer on a grid."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

pytest.importorskip("cartopy")

from extant.plots import antarctica, forced_response as fr  # noqa: E402

EXPERIMENTS = ["hist-GHG", "historical"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _change():
    """Random changes on (experiment, season, lat, lon), on a coarse grid south of 40°S."""
    lat, lon = np.arange(-87.5, -39.9, 10.0), np.arange(-180.0, 180.0, 30.0)
    values = np.random.default_rng(0).standard_normal((len(EXPERIMENTS), len(SEASONS), lat.size, lon.size))
    return xr.DataArray(values, dims=("experiment", "season", "lat", "lon"),
                        coords={"experiment": EXPERIMENTS, "season": SEASONS, "lat": lat, "lon": lon})


def test_figure_size_and_the_size_that_fits_a_page():
    #(c): The docstrings' examples
    assert antarctica.map_figure_size(n_rows=3, n_cols=4, n_colorbars=2) == pytest.approx((9.7, 9.2))
    assert antarctica.panel_size_to_fit(3, 4, 4, width=7.2, height=9.0) == pytest.approx(1.675)
    #(c): A tall figure is fitted to the page's height, and is then no wider or taller than the page
    panel_size = antarctica.panel_size_to_fit(6, 4, 1, width=7.2, height=9.0)
    width, height = antarctica.map_figure_size(6, 4, 1, panel_size=panel_size)
    assert width < 7.2 and height == pytest.approx(9.0)


def test_map_axes_are_rows_by_columns_with_the_bars_and_keys_under_them():
    fig = plt.figure(figsize=antarctica.map_figure_size(n_rows=2, n_cols=4, n_colorbars=2, n_keys=1))
    axes, cbar_axes = antarctica.map_axes(fig, n_rows=2, n_cols=4, n_colorbars=2, n_keys=1)
    assert axes.shape == (2, 4) and len(cbar_axes) == 3
    #(c): The maps' cells are about PANEL_SIZE tall; the bars, then the key, below the bottom row
    cell_height = axes[0, 0].get_position(original=True).height * fig.get_figheight()
    assert cell_height == pytest.approx(antarctica.PANEL_SIZE, rel=0.1)
    tops = [ax.get_position(original=True).y1 for ax in cbar_axes]
    assert tops == sorted(tops, reverse=True) and tops[0] < axes[-1, 0].get_position(original=True).y0


def test_every_layer_on_a_grid():
    change_da = _change()
    #(c): Concentration: 100% south of 65°S, 0% north of it, so every map has an edge
    ice_da = xr.where(change_da["lat"] < -65, 100.0, 0.0).broadcast_like(change_da)
    fig = plt.figure(figsize=antarctica.map_figure_size(n_rows=3, n_cols=4, n_colorbars=2))
    axes, cbar_axes = antarctica.map_axes(fig, n_rows=3, n_cols=4, n_colorbars=2)

    #(c): Rows 1 and 2: the change, dotted, hatched, with its lines and the ice edges
    antarctica.heatmaps_of_antarctica(change_da, "experiment", "season", axes=axes[:2], cax=cbar_axes[0],
                                      **fr.colour_scales("tas").change.colours())
    antarctica.not_significant_dots_grid(np.abs(change_da) < 0.5, "experiment", "season", axes=axes[:2])
    antarctica.hatching_grid(change_da > 1, "experiment", "season", axes=axes[:2], pattern="xxxx")
    antarctica.contour_lines_grid(change_da, [-1, 1], "experiment", "season", axes=axes[:2])
    antarctica.ice_edges(ice_da, "experiment", "season", axes=axes[:2], linestyle="dashed")
    #(c): Row 3: classes on their own bar, a tick in the middle of each, named
    classes_da = (change_da.sel(experiment="historical") > 0).astype(float)
    antarctica.heatmaps_of_antarctica(classes_da, None, "season", axes=axes[2], cax=cbar_axes[1],
                                      **fr.class_scale(["no", "yes"], ["white", "k"]).colours())
    antarctica.label_columns(axes, SEASONS)
    antarctica.label_rows(axes, [*EXPERIMENTS, "classes"])
    assert antarctica.tag_panels(axes) == 12
    fig.canvas.draw()

    assert axes[0, 0].get_title() == "DJF"
    assert [text.get_text() for text in cbar_axes[1].get_xticklabels()] == ["no", "yes"]
    #(c): Each map of the change has its mesh, dots, hatching, lines and edge
    assert all(len(ax.collections) >= 2 for ax in axes[:2].ravel())


def test_a_slice_of_one_column_and_a_figure_of_its_own():
    change_da = _change()
    fig = plt.figure(figsize=antarctica.map_figure_size(n_rows=2, n_cols=2, n_colorbars=1))
    axes, cbar_axes = antarctica.map_axes(fig, n_rows=2, n_cols=2, n_colorbars=1)
    #(c): One column at a time: the first column's DJF, the second's JJA
    antarctica.heatmaps_of_antarctica(change_da.sel(season="DJF"), "experiment", None, axes=axes[:, 0],
                                      cax=cbar_axes[0], levels=np.linspace(-3, 3, 13))
    antarctica.heatmaps_of_antarctica(change_da.sel(season="JJA"), "experiment", None, axes=axes[:, 1],
                                      levels=np.linspace(-3, 3, 13))
    assert all(ax.collections for ax in axes.ravel())

    #(c): Without axes, a figure of its own with a map per experiment and season, and a colour bar
    cbar = antarctica.heatmaps_of_antarctica(change_da, "experiment", "season")
    assert len(cbar.ax.figure.axes) == len(EXPERIMENTS) * len(SEASONS) + 1


def test_empty_fields_and_masks_draw_nothing():
    """A model without an experiment has no edge or lines; a mask True nowhere has no hatching."""
    field_da = _change().isel(experiment=0, season=0)
    assert antarctica.ice_edge(field_da * np.nan) is None
    assert antarctica.contour_lines(field_da * np.nan, [0.0]) is None
    assert antarctica.hatching(field_da > 100) is None


def test_bivariate_key():
    fig = plt.figure(figsize=antarctica.map_figure_size(n_rows=1, n_cols=2, n_colorbars=0, n_keys=1))
    axes, cbar_axes = antarctica.map_axes(fig, n_rows=1, n_cols=2, n_colorbars=0, n_keys=1)
    joint = fr.colour_scales("tas").joint
    rows, columns, row_title, column_title = joint.key
    antarctica.bivariate_key(cbar_axes[0], joint.cmap, rows, columns, row_title, column_title)
    assert [text.get_text() for text in cbar_axes[0].get_xticklabels()] == list(columns)
    assert cbar_axes[0].get_ylabel() == row_title
