import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_modules import core


def test_panel_grid_reuses_a_vertical_axes_vector():
    fig, axes = plt.subplots(3, 1)

    try:
        panels = core.panel_grid(3, 1, axes=axes, sharex=True)

        assert panels.fig is fig
        assert panels.axes.shape == (3, 1)
        assert list(panels.axes[:, 0]) == list(axes)
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_panel_grid_creates_axes_ready_for_direct_plotting():
    panels = core.panel_grid(n_rows=2, n_cols=3)
    try:
        fig, axes = panels
        axes[1, 2].plot([0, 1], [2, 3])
        assert panels.axes.shape == (2, 3)
        assert len(fig.axes) == 6
        assert len(axes[1, 2].lines) == 1
    finally:
        plt.close(panels.fig)


def test_panel_grid_accepts_one_existing_axis():
    fig, ax = plt.subplots()

    try:
        panels = core.panel_grid(1, 1, ax=ax)

        assert panels.fig is fig
        assert panels.axes[0, 0] is ax
    finally:
        plt.close(fig)


def test_panel_grid_does_not_own_a_colorbar_for_caller_axes():
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="create cax in your GridSpec"):
            core.panel_grid(1, 1, ax=ax, colorbar=True)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_nested_layout_uses_relative_colorbar_height():
    fig = plt.figure()
    outer = fig.add_gridspec(1, 1)
    layout = core.NestedLayout(
        1, 2, outer[0], has_cbar=True, cbar_frac=0.25, hspace=0
    )

    try:
        gs = layout.make_gridspec(fig)
        panel = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(layout.colorbar_spec())

        assert cax.get_position().height / panel.get_position().height == pytest.approx(
            0.25
        )
    finally:
        plt.close(fig)


def test_grid_layout_keeps_automatic_colorbars_in_gridspec():
    panels = core.panel_grid(1, 2, colorbar=True, cbar_height=0.30, cbar_gap=0.20)

    try:
        cax = panels.colorbar_ax()
        panel = panels.axes[0, 0]
        layout = panels.layout

        assert cax.get_subplotspec() is not None
        assert cax.get_position().height == pytest.approx(layout.fy(0.30))
        assert panel.get_position().y0 - cax.get_position().y1 == pytest.approx(
            layout.fy(0.20)
        )
    finally:
        plt.close(panels.fig)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cbar_height": 0}, "cbar_height"),
        ({"cbar_height": -0.1}, "cbar_height"),
        ({"cbar_gap": -0.1}, "cbar_gap"),
    ],
)
def test_grid_layout_rejects_invalid_colorbar_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        core.GridLayout(1, 1, has_cbar=True, **kwargs)
