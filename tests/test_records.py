"""Record highs and lows against a plain loop and against theory."""

import numpy as np
import xarray as xr

from extant.statistics import records

RNG = np.random.default_rng(42)


def test_record_counts_match_brute_force_and_theory():
    x = RNG.standard_normal((400, 35))
    x[:5, -1] = np.nan
    da = xr.DataArray(x, dims=("member", "year"), coords={"year": np.arange(1979, 2014)})

    def brute(row, high=True):
        row = row[np.isfinite(row)]
        row = row if high else -row
        return sum(True if i == 0 else row[i] > row[:i].max() for i in range(row.size))

    np.testing.assert_array_equal(records.record_highs(da).values, [brute(r) for r in x])
    np.testing.assert_array_equal(records.record_lows(da).values, [brute(r, False) for r in x])
    np.testing.assert_allclose(float(records.record_highs(da).mean()), records.expected_records(35), atol=0.15)
    np.testing.assert_array_equal(records.cumulative_records(da).isel(year=-1).values, records.record_highs(da).values)
