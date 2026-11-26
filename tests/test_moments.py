"""Skewness and kurtosis against scipy's."""

import numpy as np
import scipy.stats
import xarray as xr

from extant.statistics import moments


def test_moments_match_scipy():
    da = xr.DataArray(np.random.default_rng(42).gamma(2.0, size=(3, 40)), dims=("member", "year"))
    np.testing.assert_allclose(moments.skewness(da), scipy.stats.skew(da.values, axis=1))
    np.testing.assert_allclose(moments.excess_kurtosis(da), scipy.stats.kurtosis(da.values, axis=1))
