"""One grid onto another."""

import numpy as np
import xarray as xr


def match_grid(obs, like, tolerance=0.01, lat="lat", lon="lon"):
    """Put ``obs`` on the latitudes and longitudes of ``like``.

    ERA5's grid need not be the LESFMIP grid: it may use 0-360 longitudes where
    LESFMIP uses -180-180, run north to south, cover a different area, or sit
    on different points. So this:

    1. puts ``obs``'s longitudes in ``like``'s convention and sorts both axes;
    2. if every point of ``like`` has a partner in ``obs`` within ``tolerance``
       degrees, takes those values unchanged, relabelled with ``like``'s exact
       coordinates so arithmetic between the two aligns;
    3. otherwise interpolates bilinearly onto ``like``'s points, wrapping round
       in longitude. Points outside ``obs``'s coverage are NaN.

    Interpolation suits data at a similar resolution, like the ERA5 store
    (conservatively regridded to 2.5° in notebook calc_01, section *ERA5 Hourly Files to Monthly Means*). Regrid much
    finer data conservatively first.

    Args:
        obs (xr.DataArray): Data to put on the grid.
        like (xr.DataArray): Data on the target grid.
        tolerance (float): Largest coordinate difference, in degrees, still treated as the same point.

    Returns:
        xr.DataArray: ``obs`` on ``like``'s ``lat`` and ``lon``.
    """
    target = {name: np.asarray(like[name].values, dtype=float) for name in (lat, lon)}
    #(c): Longitudes into the target's convention: -180-180 if it has any negative longitude, else 0-360.
    #(c): The range starts `tolerance` early, so -180.000001 stays next to -180 rather than wrapping to +180
    start = (-180.0 if target[lon].min() < 0 else 0.0) - tolerance
    wrapped = (obs[lon] - start) % 360 + start
    obs = obs.assign_coords({lon: wrapped.astype(float), lat: obs[lat].astype(float)}).sortby([lat, lon])

    def gap(name):
        return np.abs(obs[name].values[:, None] - target[name][None, :]).min(0).max()

    if gap(lat) <= tolerance and gap(lon) <= tolerance:
        nearest = {name: obs.indexes[name].get_indexer(target[name], method="nearest") for name in (lat, lon)}
        return obs.isel(nearest).assign_coords(target)

    #(c): One extra column at each end, a full turn away, so interpolation wraps round in longitude
    padded = xr.concat([obs.isel({lon: [-1]}).assign_coords({lon: obs[lon][-1:] - 360}), obs,
                        obs.isel({lon: [0]}).assign_coords({lon: obs[lon][:1] + 360})], dim=lon)
    return padded.interp(target)
