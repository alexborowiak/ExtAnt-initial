"""The shape of a distribution: skewness and excess kurtosis of every series at once.

The mean and the standard deviation, the other two moments, are xarray's own ``.mean`` and ``.std``. These two are the
plain moment estimators. They are biased for short records, but identically so for every series of the same length.
"""


def _standardised_moment(da, dim, order):
    deviation = da - da.mean(dim)
    variance = (deviation**2).mean(dim)
    return (deviation**order).mean(dim) / variance ** (order / 2)


def skewness(da, dim='year'):
    """Skewness: > 0 when the warm tail is longer than the cold tail."""
    return _standardised_moment(da, dim, 3)


def excess_kurtosis(da, dim='year'):
    """Excess kurtosis: 0 for a normal distribution, > 0 for heavier tails."""
    return _standardised_moment(da, dim, 4) - 3
