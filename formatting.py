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