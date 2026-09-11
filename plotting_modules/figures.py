"""Specific figures. One function per figure that appears in a paper or talk.

Everything here is allowed to know about seasons, forcings and experiments, and
is expected to be called once from a notebook. Everything reusable lives in the
`plotting_modules` package.

The house pattern: build a GridLayout, hand a `draw(ax, row_val, col_val)`
callback to `panel_grid`, then decorate. The callback closes over whatever data
structure the figure happens to have, which is why `panel_grid` takes a
callback rather than an array.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import cartopy.crs as ccrs

from . import core, maps, timeseries
from .constants import (
    FORCING_COLORS,
    FORCING_LEGEND_LABELS,
    FORCING_REVEAL_ORDER,
    SEASON_COLORS,
    SLIDE_RC,
)
from .utils import plot, save_frame, shared_ylim


def _row_values(ds, dim):
    """Panel values for `dim`: its coordinate values, a scalar, or [None]."""
    if dim in ds.dims:
        return list(np.atleast_1d(ds[dim].values))
    if dim in ds.coords:
        return [ds[dim].values.item()]
    return [None]


# --------------------------------------------------------------------------
# Quantile matrix: low, median, high quantile and their difference
# --------------------------------------------------------------------------

@plot("figure")
def quantile_matrix(ds, row_dim="season", quant_dim="quantile", q_low=0.1, q_high=0.9,
                    raw_levels=np.linspace(-3, 3, 13),
                    diff_levels=np.linspace(-2, 2, 9), cmap="RdBu_r",
                    lat_name="lat", lon_name="lon", title=None,
                    raw_label=None, diff_label=None,
                    fig=None, spec=None, layout=None, **layout_kwargs):
    """Polar panels: low, median and high quantile, and their difference.

    Parameters
    ----------
    ds : xarray.DataArray
        Field with a quantile dimension and optionally a row dimension.
    row_dim, quant_dim : str
        Dimensions mapped to rows, and holding the quantiles.
    q_low, q_high : float
        Quantiles drawn in the first and third columns.
    raw_levels, diff_levels : array_like
        Contour levels for the quantile columns and the difference column.
    cmap : str
        Colormap name.
    lat_name, lon_name : str
        Names of the latitude and longitude coordinates.
    title : str or None
        Figure title.
    raw_label, diff_label : str or None
        Labels under the two colorbars.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    row_vals = _row_values(ds, row_dim)
    columns = [("q", q_low), ("q", 0.5), ("q", q_high), ("diff", None)]
    col_titles = [f"p{int(q_low * 100)}", "p50", f"p{int(q_high * 100)}",
                  f"p{int(q_high * 100)} - p{int(q_low * 100)}"]

    def draw(ax, row_val, column):
        ds_row = ds.sel({row_dim: row_val}) if row_dim in ds.dims else ds
        kind, q = column
        if kind == "diff":
            da = (ds_row.sel({quant_dim: q_high}, method="nearest")
                  - ds_row.sel({quant_dim: q_low}, method="nearest"))
            levels = diff_levels
        else:
            da = ds_row.sel({quant_dim: q}, method="nearest")
            levels = raw_levels
        return maps.draw_polar_contour(ax, da, levels, cmap, lat_name, lon_name)

    layout_kwargs.setdefault("row_labels", any(v is not None for v in row_vals))
    layout_kwargs.setdefault("has_title", title is not None)
    layout_kwargs.setdefault("has_cbar", True)
    layout_kwargs.setdefault("has_cbar_label", raw_label is not None or diff_label is not None)

    panels = core.panel_grid(row_vals, columns, draw, fig=fig, spec=spec, layout=layout,
                             projection=ccrs.SouthPolarStereo(), **layout_kwargs)

    core.label_cols(panels.axes, col_titles)
    core.label_rows(panels.axes, row_vals)
    core.tag_panels(panels.axes)

    panels.extras["cbars"] = [
        core.add_colorbar(panels.fig, panels.artists[0, 0],
                          panels.layout.cbar_ax(panels.fig, 0, 2), raw_levels, raw_label),
        core.add_colorbar(panels.fig, panels.artists[0, 3],
                          panels.layout.cbar_ax(panels.fig, 3, 3),
                          np.round(diff_levels, 1), diff_label),
    ]
    if title and panels.layout.owns_figure:
        core.add_suptitle(panels.fig, panels.layout, title)

    return panels


# --------------------------------------------------------------------------
# Seasonal quantile summary: spread on top, plume below, one column per season
# --------------------------------------------------------------------------

def _draw_spread(ax, da_season, q_pairs, color, time_dim, label=False,
                 styles=("-", "--", ":"), alphas=(1.0, 0.75, 0.55)):
    """Top panel of a season column: the width of each quantile pair over time.

    `styles` and `alphas` are cycled, one per pair in `q_pairs`.
    """
    time = da_season[time_dim].values.squeeze().ravel()
    lines = []
    for idx, (q_lo, q_hi) in enumerate(q_pairs):
        spread = (da_season.sel(quantile=q_hi, method="nearest")
                  - da_season.sel(quantile=q_lo, method="nearest"))
        lines += ax.plot(
            time, spread.values.squeeze().ravel(), color=color,
            linestyle=styles[idx % len(styles)],
            alpha=alphas[idx % len(alphas)], linewidth=1.2,
            label=f"q{q_hi}-q{q_lo}" if label else None,
        )
    return lines

@plot("figure")
def quantile_summary(da_point, q_pairs=((0.01, 0.99), (0.1, 0.9), (0.25, 0.75)),
                     q_med=0.5, band_alphas=(0.08, 0.18, 0.32),
                     band_edge_widths=(0.0, 0.6, 1.0),
                     band_edge_styles=("-", ":", "--"),
                     spread_styles=("-", "--", ":"),
                     spread_alphas=(1.0, 0.75, 0.55),
                     colors=None,
                     fig=None, spec=None, layout=None, **layout_kwargs):
    """Per-season quantile spread over a nested plume, with a figure legend.

    The rows are different heights, which is a GridLayout row_heights argument
    rather than a reason to hand-roll a GridSpec.

    Parameters
    ----------
    da_point : xarray.DataArray
        Dims ('season', 'quantile', time).
    q_pairs : sequence of (float, float)
        Quantile pairs, widest first.
    q_med : float
        Median quantile level.
    band_alphas, band_edge_widths, band_edge_styles : sequence
        Plume band styling, one entry per pair in `q_pairs`. Cycled if shorter.
    spread_styles, spread_alphas : sequence
        Line styling for the spread panel, one entry per pair. Cycled if shorter.
    colors : dict or None
        Season -> colour. Defaults to constants.SEASON_COLORS.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
        `axes` is (2, n_seasons): the spread row, then the plume row.
    """
    seasons = list(da_point.season.values)
    colors = SEASON_COLORS if colors is None else colors

    def draw(ax, row, season):
        da_season = da_point.sel(season=season)
        time_dim = "year" if "year" in da_season.coords else da_season.dims[0]
        first = season == seasons[0]
        color = colors.get(season, "C0")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

        if row == "spread":
            artists = _draw_spread(ax, da_season, q_pairs, color, time_dim,
                                   label=first, styles=spread_styles,
                                   alphas=spread_alphas)
            ax.set_title(season, fontsize=11, fontweight="bold", pad=4)
            ax.tick_params(labelbottom=False)
            ax.set_ylabel("Δ Quantiles" if first else "", fontsize=9)
            if first:
                ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.8)
            return artists

        line = timeseries.draw_plume(
            ax, da_season, pairs=q_pairs, median=q_med, color=color,
            alphas=band_alphas, edge_widths=band_edge_widths,
            edge_styles=band_edge_styles, edge_alpha=0.6, lw=1.8,
            label="Median" if first else None, x_dim=time_dim,
        )
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Difference (hist-nat - hist)" if first else "", fontsize=9)
        return line

    layout_kwargs.setdefault("row_heights", [1.6, 3.5])
    layout_kwargs.setdefault("panel_w", 3.2)
    layout_kwargs.setdefault("hspace", 0.35)
    layout_kwargs.setdefault("wspace", 0.55)
    layout_kwargs.setdefault("bottom", 0.85)
    panels = core.panel_grid(["spread", "plume"], seasons, draw,
                             fig=fig, spec=spec, layout=layout, sharex=True,
                             **layout_kwargs)

    handles = (panels.axes[1, 0].get_legend_handles_labels()[0]
               + timeseries.plume_handles(q_pairs, band_alphas))
    panels.extras["legend"] = panels.fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4,
        fontsize=10, frameon=True, facecolor="white", framealpha=0.95,
    )
    return panels


# --------------------------------------------------------------------------
# Slide buildup: forcings revealed one at a time
# --------------------------------------------------------------------------


def _style_buildup_ax(ax, ylim, baseline=0.0, xlim=(1850, 2014),
                      ylabel="Near-surface air temperature [K]"):
    """Frame styling shared by every buildup frame.

    Every number here is fixed across the sequence, which is what makes the
    frames line up when they are flipped through.
    """
    ax.axhline(baseline, color="0.75", lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.tick_params(length=4, width=1.0)
    core.style_ax(ax, grid=False)

@plot("figure")
def tas_buildup_figure(tas_by_forcing, revealed, ylim, order=None, fade_earlier=True,
                       baseline=0.0, xlim=(1850, 2014), frame_size=(9, 5),
                       fig=None, spec=None, layout=None, **layout_kwargs):
    """One frame of the forcing buildup: an ordinary figure function.

    Split out from the frame loop so a single frame can be inspected, tweaked
    or dropped into a larger figure like any other plot here.

    Parameters
    ----------
    tas_by_forcing : dict[str, xarray.DataArray]
        Forcing -> temperature series with a 'year' dimension.
    revealed : sequence
        Forcings drawn in this frame, in reveal order.
    ylim : (float, float)
        Shared across the whole sequence; see utils.shared_ylim.
    order : sequence or None
        Full reveal order, for the legend. Defaults to FORCING_REVEAL_ORDER.
    fade_earlier : bool
        Dim all but the newest line.
    baseline : float
        Value of the dashed reference line.
    xlim : (float, float)
        Shared x limits.
    frame_size : (float, float)
        Overall frame size in inches; the panel is sized to fit inside it.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels
    """
    order = list(order or FORCING_REVEAL_ORDER)
    layout_kwargs.setdefault("panel_w", frame_size[0] - 1.0)
    layout_kwargs.setdefault("panel_h", frame_size[1] - 1.4)
    layout_kwargs.setdefault("left", 0.85)
    layout_kwargs.setdefault("bottom", 0.6)
    layout_kwargs.setdefault("top", 0.9)

    def draw(ax, _row, _col):
        artists = timeseries.draw_reveal(
            ax, tas_by_forcing, revealed, FORCING_COLORS, fade=fade_earlier
        )
        _style_buildup_ax(ax, ylim, baseline=baseline, xlim=xlim)
        return artists

    panels = core.panel_grid([None], [None], draw, fig=fig, spec=spec,
                             layout=layout, **layout_kwargs)
    panels.extras["legend"] = core.reveal_legend(
        panels.fig, order, FORCING_COLORS, FORCING_LEGEND_LABELS, revealed,
        ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
    )
    return panels


def tas_buildup_frames(tas_by_forcing, directory="slides_tas_buildup",
                       fade_earlier=True, include_empty=True, order=None,
                       baseline=0.0, rc=None, **figure_kwargs):
    """Write one PNG per reveal step, all sharing the same axes geometry.

    Parameters
    ----------
    tas_by_forcing : dict[str, xarray.DataArray]
        Forcing -> temperature series with a 'year' dimension.
    directory : str or pathlib.Path
        Output directory; created on first write, not on import.
    fade_earlier : bool
        Dim all but the newest line in each frame.
    include_empty : bool
        Write an opening frame with axes and a fully dimmed legend and no data.
    order : sequence or None
        Reveal order; defaults to FORCING_REVEAL_ORDER, filtered to the
        forcings actually present.
    baseline : float
        Value of the dashed reference line, kept inside the shared y limits.
    rc : dict or None
        rcParams for the frames. Defaults to constants.SLIDE_RC.
    **figure_kwargs
        Passed to `tas_buildup_figure`.

    Returns
    -------
    list of pathlib.Path
        The frames written, in slide order.
    """
    order = [f for f in (order or FORCING_REVEAL_ORDER) if f in tas_by_forcing]
    missing = set(tas_by_forcing) - set(order)
    if missing:
        raise ValueError(f"no reveal position for {sorted(missing)}")

    # One y limit for the whole sequence. Per-frame autoscaling is what makes a
    # buildup jump between slides.
    ylim = shared_ylim([tas_by_forcing[f].values for f in order], include=[baseline])

    paths = []
    with plt.rc_context(SLIDE_RC if rc is None else rc):
        for index, step in enumerate(range(0 if include_empty else 1, len(order) + 1)):
            revealed = order[:step]
            panels = tas_buildup_figure(tas_by_forcing, revealed, ylim, order=order,
                                        fade_earlier=fade_earlier, baseline=baseline,
                                        **figure_kwargs)
            name = revealed[-1] if revealed else "empty"
            paths.append(save_frame(panels.fig, directory, index, name))

    return paths
