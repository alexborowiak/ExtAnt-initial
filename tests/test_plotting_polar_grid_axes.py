import numpy as np
import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

cartopy = pytest.importorskip("cartopy.crs")
xr = pytest.importorskip("xarray")

from plotting_modules import core, figures, maps


def test_polar_grid_uses_caller_axes_and_cax(monkeypatch):
    field = xr.DataArray(
        np.arange(24).reshape(3, 2, 2, 2),
        dims=("kind", "period", "lat", "lon"),
        coords={
            "kind": ["signal", "noise", "sn"],
            "period": ["early", "late"],
            "lat": [-80, -70],
            "lon": [0, 180],
        },
    )
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, height_ratios=[3, .08])
    axes = core.panel_grid(
        3, 2, fig=fig, spec=grid[0], projection=cartopy.SouthPolarStereo()
    ).axes
    cax = fig.add_subplot(grid[1])
    seen = []

    def draw_contour(ax, da, *args, **kwargs):
        seen.append((ax, da.kind.item(), da.period.item()))
        mappable = cm.ScalarMappable(norm=colors.Normalize(-3, 3), cmap="RdBu_r")
        mappable.set_array([-3, 3])
        return mappable

    monkeypatch.setattr(maps, "draw_polar_contour", draw_contour)
    try:
        panels = maps.polar_grid(
            field, row_dim="kind", col_dim="period", axes=axes, cax=cax,
            tag=False,
        )

        assert panels.fig is fig
        assert panels.axes[0, 0] is axes[0, 0]
        assert panels.axes[2, 1] is axes[2, 1]
        assert seen == [
            (axes[row, col], kind, period)
            for row, kind in enumerate(("signal", "noise", "sn"))
            for col, period in enumerate(("early", "late"))
        ]
        assert panels.extras["cbar"].ax is cax
    finally:
        plt.close(fig)


def test_polar_grid_exposes_automatic_colorbar_geometry(monkeypatch):
    field = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=("period", "lat", "lon"),
        coords={"period": ["early", "late"], "lat": [-80, -70], "lon": [0, 180]},
    )

    def draw_contour(*args, **kwargs):
        mappable = cm.ScalarMappable(norm=colors.Normalize(-3, 3), cmap="RdBu_r")
        mappable.set_array([-3, 3])
        return mappable

    monkeypatch.setattr(maps, "draw_polar_contour", draw_contour)
    panels = maps.polar_grid(
        field, col_dim="period", cbar_height=0.30, cbar_gap=0.20, tag=False
    )
    try:
        cbar = panels.extras["cbar"].ax.get_position()
        panel = panels.axes[0, 0].get_position()

        assert panels.layout.cbar_height == pytest.approx(0.30)
        assert cbar.height == pytest.approx(panels.layout.fy(0.30))
        assert panel.y0 - cbar.y1 == pytest.approx(panels.layout.fy(0.20))
    finally:
        plt.close(panels.fig)


def test_polar_grid_rejects_absolute_colorbar_geometry_in_nested_layout():
    field = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=("period", "lat", "lon"),
        coords={"period": ["early", "late"], "lat": [-80, -70], "lon": [0, 180]},
    )
    fig = plt.figure()

    try:
        with pytest.raises(ValueError, match="cbar_frac"):
            maps.polar_grid(
                field,
                col_dim="period",
                fig=fig,
                spec=fig.add_gridspec(1, 1)[0],
                cbar_height=0.30,
            )
    finally:
        plt.close(fig)


def test_polar_grid_rejects_cax_from_another_figure(monkeypatch):
    field = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=("period", "lat", "lon"),
        coords={"period": ["early", "late"], "lat": [-80, -70], "lon": [0, 180]},
    )
    fig, axes = plt.subplots(
        1, 2, squeeze=False,
        subplot_kw={"projection": cartopy.SouthPolarStereo()},
    )
    foreign_fig, foreign_cax = plt.subplots()

    def draw_contour(*args, **kwargs):
        mappable = cm.ScalarMappable(norm=colors.Normalize(-3, 3), cmap="RdBu_r")
        mappable.set_array([-3, 3])
        return mappable

    monkeypatch.setattr(maps, "draw_polar_contour", draw_contour)
    try:
        with pytest.raises(ValueError, match="cax must belong to fig"):
            maps.polar_grid(field, col_dim="period", axes=axes, cax=foreign_cax)
    finally:
        plt.close(fig)
        plt.close(foreign_fig)


@pytest.mark.parametrize("target", ("axes", "cax"))
def test_polar_grid_rejects_colorbar_geometry_for_caller_owned_targets(target):
    field = xr.DataArray(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dims=("period", "lat", "lon"),
        coords={"period": ["early", "late"], "lat": [-80, -70], "lon": [0, 180]},
    )
    if target == "axes":
        fig, axes = plt.subplots(
            1, 2, squeeze=False,
            subplot_kw={"projection": cartopy.SouthPolarStereo()},
        )
        kwargs = {"axes": axes}
    else:
        fig = plt.figure()
        grid = fig.add_gridspec(1, 1)
        kwargs = {"fig": fig, "cax": fig.add_subplot(grid[0, 0])}

    try:
        with pytest.raises(ValueError, match="caller-owned cax"):
            maps.polar_grid(
                field,
                col_dim="period",
                cbar_height=0.30,
                cbar_gap=0.20,
                **kwargs,
            )
    finally:
        plt.close(fig)


def _quantile_field():
    return xr.DataArray(
        np.arange(12).reshape(3, 2, 2),
        dims=("quantile", "lat", "lon"),
        coords={"quantile": [0.1, 0.5, 0.9], "lat": [-80, -70], "lon": [0, 180]},
    )


def _mappable(*args, **kwargs):
    mappable = cm.ScalarMappable(norm=colors.Normalize(-3, 3), cmap="RdBu_r")
    mappable.set_array([-3, 3])
    return mappable


def test_quantile_matrix_uses_caller_owned_colorbar_axes(monkeypatch):
    field = _quantile_field()
    fig = plt.figure()
    grid = fig.add_gridspec(2, 4, height_ratios=[1, .06])
    axes = np.array([[
        fig.add_subplot(grid[0, col], projection=cartopy.SouthPolarStereo())
        for col in range(4)
    ]], dtype=object)
    caxes = (
        fig.add_subplot(grid[1, :3]),
        fig.add_subplot(grid[1, 3]),
    )
    monkeypatch.setattr(figures.maps, "draw_polar_contour", _mappable)
    axes_count = len(fig.axes)

    try:
        panels = figures.quantile_matrix(field, axes=axes, caxes=caxes)

        assert panels.fig is fig
        assert panels.extras["cbars"][0].ax is caxes[0]
        assert panels.extras["cbars"][1].ax is caxes[1]
        assert len(fig.axes) == axes_count
        with pytest.raises(ValueError, match="caller-owned caxes"):
            figures.quantile_matrix(
                field, axes=axes, caxes=caxes, cbar_height=0.30
            )
    finally:
        plt.close(fig)


def test_quantile_matrix_routes_automatic_colorbar_geometry(monkeypatch):
    monkeypatch.setattr(figures.maps, "draw_polar_contour", _mappable)
    panels = figures.quantile_matrix(
        _quantile_field(), cbar_height=0.30, cbar_gap=0.20
    )
    try:
        cbar = panels.extras["cbars"][0].ax.get_position()
        panel = panels.axes[0, 0].get_position()

        assert panels.layout.cbar_height == pytest.approx(0.30)
        assert panels.layout.cbar_gap == pytest.approx(0.20)
        assert cbar.height == pytest.approx(panels.layout.fy(0.30))
        assert panel.y0 - cbar.y1 == pytest.approx(panels.layout.fy(0.20))
    finally:
        plt.close(panels.fig)
