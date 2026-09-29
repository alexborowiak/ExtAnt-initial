import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _fast_kde_block(arr, x, bandwidth):
    """
    arr has shape (..., sample)
    """

    original_shape = arr.shape[:-1]
    nsample = arr.shape[-1]

    arr = arr.reshape(-1, nsample)

    nseries = arr.shape[0]
    nx = len(x)

    dx = x[1] - x[0]

    edges = np.concatenate([
        [x[0] - dx / 2],
        (x[:-1] + x[1:]) / 2,
        [x[-1] + dx / 2],
    ])

    # Flatten all values
    values = arr.ravel()

    valid = (
        np.isfinite(values)
        & (values >= edges[0])
        & (values <= edges[-1])
    )

    positions = np.flatnonzero(valid)

    # Which KDE does each value belong to?
    rows = positions // nsample

    # Which x bin?
    bins = np.searchsorted(
        edges,
        values[valid],
        side="right",
    ) - 1

    bins = np.clip(bins, 0, nx - 1)

    # Convert (row, bin) -> one-dimensional index
    indices = rows * nx + bins

    hist = np.bincount(
        indices,
        minlength=nseries * nx,
    ).reshape(nseries, nx)

    # Number of finite values in each KDE
    n = np.isfinite(arr).sum(axis=-1)

    # Avoid division by zero
    density_hist = np.divide(
        hist,
        n[:, None] * dx,
        out=np.full_like(hist, np.nan, dtype=float),
        where=n[:, None] > 0,
    )

    sigma = bandwidth / dx

    density = gaussian_filter1d(
        density_hist,
        sigma=sigma,
        axis=-1,
        mode="constant",
    )

    return density.reshape(*original_shape, nx)

def fast_kde_xr(da, x, dim="time", bandwidth="scott"):

    x = np.asarray(x)

    # --------------------------------------------
    # Accept either:
    #
    # dim="time"
    #
    # or:
    #
    # dim=["window", "member"]
    # --------------------------------------------
    if isinstance(da, xr.Dataset):
        var_name = list(da.data_vars)[0]
        da = da[var_name]
    else:
        var_name = da.name or "kde"
    
    if isinstance(da, xr.Dataset): da = da.to_array().squeeze('variable', drop=True)

    if isinstance(dim, str):
        dims = [dim]
    else:
        dims = list(dim)

    # --------------------------------------------
    # Stack sample dimensions
    # --------------------------------------------

    if len(dims) > 1:

        sample_dim = "__sample__"

        da = da.stack(
            {sample_dim: dims}
        )

    else:
        sample_dim = dims[0]

    # Make sure complete sample axis is available
    # within each dask block
    if da.chunks is not None:
        da = da.chunk({
            sample_dim: -1
        })

    # --------------------------------------------
    # Common bandwidth
    # --------------------------------------------

    if bandwidth == "scott":

        n = da.count(
            dim=sample_dim
        )

        std = da.std(
            dim=sample_dim,
            skipna=True,
            ddof=1,
        )

        bw = std * n ** (-1 / 5)

        # One bandwidth shared across all KDEs
        bandwidth = float(
            np.nanmedian(
                bw.compute().values
            )
        )

    else:

        bandwidth = float(bandwidth)

    # --------------------------------------------
    # KDE
    # --------------------------------------------

    result = xr.apply_ufunc(
        _fast_kde_block,
        da,

        input_core_dims=[
            [sample_dim]
        ],

        output_core_dims=[
            ["x"]
        ],

        kwargs={
            "x": x,
            "bandwidth": bandwidth,
        },

        vectorize=False,
        dask="parallelized",

        output_dtypes=[
            float
        ],

        dask_gufunc_kwargs={
            "output_sizes": {
                "x": len(x)
            }
        },
    )

    result = result.assign_coords(x=x)
    # result = result.xarray_datatree_utils.to_dataset(name = var_name)

    return result