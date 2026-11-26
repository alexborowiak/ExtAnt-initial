"""Record highs and lows: years warmer (or colder) than every earlier year in the record."""

import numpy as np
import xarray as xr


def _record_flags(values, high=True):
    """True in each year that beats every earlier year (the first year counts), along the last axis."""
    x = np.where(np.isfinite(values), values if high else -values, -np.inf)
    earlier_best = np.concatenate([np.full(x.shape[:-1] + (1,), -np.inf),
                                   np.maximum.accumulate(x, axis=-1)[..., :-1]], axis=-1)
    return (x > earlier_best) & np.isfinite(values)


def _record_count(values, high=True):
    return np.where(np.isfinite(values).any(axis=-1), _record_flags(values, high).sum(axis=-1), np.nan)


def record_highs(da, dim='year'):
    """Number of record highs: years warmer than every earlier year in the record (the first year counts)."""
    return xr.apply_ufunc(_record_count, da, input_core_dims=[[dim]], kwargs={"high": True},
                          dask="parallelized", output_dtypes=[float])


def record_lows(da, dim='year'):
    """Number of record lows: years colder than every earlier year in the record (the first year counts)."""
    return xr.apply_ufunc(_record_count, da, input_core_dims=[[dim]], kwargs={"high": False},
                          dask="parallelized", output_dtypes=[float])


def cumulative_records(da, dim='year', high=True):
    """Running number of record highs (or lows) up to each year, for plotting record rates."""
    return xr.apply_ufunc(lambda v: np.cumsum(_record_flags(v, high), axis=-1).astype(float), da,
                          input_core_dims=[[dim]], output_core_dims=[[dim]], dask="parallelized",
                          output_dtypes=[float])


def expected_records(n_years):
    """Expected number of records in n years of a stationary, independent series: 1 + 1/2 + ... + 1/n.

    In year k every one of the k years so far is equally likely to be the
    largest, so year k is a record with probability 1/k.
    """
    return float(np.sum(1 / np.arange(1, n_years + 1)))
