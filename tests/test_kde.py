"""The kernel density estimate against scipy's."""

import numpy as np
import scipy.stats
import xarray as xr

from extant.statistics import kde

RNG = np.random.default_rng(42)


def test_kde_matches_scipy_and_integrates_to_one():
    samples = RNG.standard_normal(35)
    da = xr.DataArray(samples, dims="year")
    x = xr.DataArray(np.linspace(-6, 6, 2001), dims="x")
    bandwidth = 0.4
    ours = kde.kde(da, x, bandwidth=bandwidth)
    reference = scipy.stats.gaussian_kde(samples, bw_method=bandwidth / samples.std(ddof=1))(x.values)
    np.testing.assert_allclose(ours, reference, rtol=1e-10)
    np.testing.assert_allclose(np.trapezoid(ours, x), 1, rtol=1e-4)
