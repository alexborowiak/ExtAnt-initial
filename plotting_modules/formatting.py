def format_latlon(point, precision=2):
    lat, lon = point["lat"], point["lon"]
    lat_str = f"{abs(lat):.{precision}f}".rstrip("0").rstrip(".")
    lon_str = f"{abs(lon):.{precision}f}".rstrip("0").rstrip(".")
    return (
        f"{lat_str}°{'S' if lat < 0 else 'N'}, "
        f"{lon_str}°{'W' if lon < 0 else 'E'}"
    )


def format_sel(sel, labels=None, formats=None, sep=", ", assign=" = "):
    """Compact one-line label for a selection dict. Lat and lon get hemisphere suffixes.
    Args:
        sel (dict): Coordinate name to selected value.
        labels (dict): Coordinate name to display name; the key itself if absent.
        formats (dict): Coordinate name to value formatter; overrides the built-in
            lat/lon handling, str for anything missing.
        sep (str): Separator between entries.
        assign (str): Separator between a name and its value.
    """
    labels = labels or {}
    formats = formats or {}
    hemis = {"lat": "SN", "lon": "WE"}
    parts = []
    for k, v in sel.items():
        if k in formats:
            text = formats[k](v)
        elif k in hemis:
            text = f"{abs(v):.2f}°{hemis[k][v >= 0]}"
        else:
            text = str(v)
        parts.append(f"{labels.get(k, k)}{assign}{text}")
    return sep.join(parts)



def style_ax(ax, scale=1.0):
    """Apply gridlines behind the data, drop top and right spines, and scale any text present.

    Call after labels, title and legend are set: `set_title` and `legend` reapply
    the `rcParams` font sizes and would undo the scaling. Text that is absent
    (empty title, empty axis label, no ticks, no legend) is left alone.

    Args:
        ax (plt.Axes): Axis to format.
        scale (float): Multiplier applied to the current axis label, tick label,
            title and legend entry font sizes.
    """
    ax.grid(axis="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_title():
        ax.title.set_fontsize(ax.title.get_fontsize() * scale)
    if ax.get_xlabel():
        ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * scale)
    if ax.get_ylabel():
        ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * scale)
    xticklabels = ax.get_xticklabels()
    if xticklabels:
        ax.tick_params(axis="x", labelsize=xticklabels[0].get_fontsize() * scale)
    yticklabels = ax.get_yticklabels()
    if yticklabels:
        ax.tick_params(axis="y", labelsize=yticklabels[0].get_fontsize() * scale)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * scale)
    ax.autoscale(axis='x', tight=True)

# def style_ax(ax):
#     """Grid behind the data, top/right spines dropped, muted ticks and spines."""
#     ax.set_axisbelow(True)
#     ax.grid(True, axis="y", color="0.85", lw=0.6)
#     ax.grid(True, axis="x", color="0.93", lw=0.5)
#     for side in ("top", "right"):
#         ax.spines[side].set_visible(False)
#     for side in ("left", "bottom"):
#         ax.spines[side].set_color("0.45")
#         ax.spines[side].set_linewidth(0.8)
#     ax.tick_params(colors="0.3", length=4, width=0.8)
#     ax.margins(y=0.08)