def format_latlon(point, precision=2):
    lat, lon = point["lat"], point["lon"]
    lat_str = f"{abs(lat):.{precision}f}".rstrip("0").rstrip(".")
    lon_str = f"{abs(lon):.{precision}f}".rstrip("0").rstrip(".")
    return (
        f"{lat_str}°{'S' if lat < 0 else 'N'}, "
        f"{lon_str}°{'W' if lon < 0 else 'E'}"
    )