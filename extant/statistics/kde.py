"""Kernel density estimates of every series at once (Gaussian kernels)."""

import numpy as np
import scipy.stats
import xarray as xr

SAMPLE_DIM = "year"
MEMBER_DIM = "member"


def kde_bandwidth(ensemble, dim=SAMPLE_DIM, member_dim=MEMBER_DIM):
    """One Gaussian-kernel bandwidth for ERA5 and all members: the members' median Scott's rule.

    Scott's rule (std x n^(-1/5)) is scipy.stats.gaussian_kde's default. With
    the same bandwidth for ERA5 and every member, a difference between curves
    comes from the data rather than the smoothing, and the pooled ensemble
    density is exactly the mean of the member densities.
    """
    return (ensemble.std(dim, ddof=1) * ensemble.count(dim) ** (-1 / 5)).median(member_dim)


def kde(da, x, dim=SAMPLE_DIM, bandwidth=None):
    """Gaussian kernel density estimate over ``dim``, evaluated at ``x``, ignoring NaN.

    Args:
        da (xr.DataArray): Samples.
        x (xr.DataArray): Evaluation points, on their own dim (e.g. "x").
        dim (str | Sequence[str]): Sample dimension(s).
        bandwidth (float | xr.DataArray | None): Kernel standard deviation; may
            vary over the broadcast dims (e.g. one per season). Scott's rule if None.

    Returns:
        xr.DataArray: Density at ``x``, integrating to 1 over ``x``.
    """
    if bandwidth is None:
        bandwidth = da.std(dim, ddof=1) * da.count(dim) ** (-1 / 5)
    z = (x - da) / bandwidth
    kernel = np.exp(-0.5 * z**2) / (bandwidth * np.sqrt(2 * np.pi))
    return kernel.sum(dim) / da.count(dim)
