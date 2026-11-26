"""Trends and smoothing of any series.

    trend               the least-squares slope, per year
    linear_fit          the straight line itself
    detrend             the series minus its line
    lowess              the local-regression (LOWESS) smoother, the only one in the repository
    forced_response     an ensemble's mean or median, smoothed with it
    grid_mk             the Mann-Kendall trend test at every grid point
"""

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.stats import norm

from ..utils.datatree import skip_empty


def _years(da, dims):
    """The years of a series' samples, as numbers, missing wherever the data are, so both means use the same samples."""
    x = da['year' if 'year' in dims else dims[0]]
    #(c): Datetime coordinates regress on the calendar year, giving a gradient per year
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.dt.year
    return x.astype(float).broadcast_like(da).where(da.notnull())


def trend(da, dim='year'):
    """The least-squares slope of each series, in the data's units per year. Missing samples are ignored.

    Only the slope: no line, no intercept. For per decade, multiply where it is used: ``10 * trend.trend(da)``.

    Example: the series 1, 3, 2, 4 over the years 2000-2003. The years' mean is 2001.5 and the values' mean 2.5, so
    dx = -1.5, -0.5, 0.5, 1.5 and dy = -1.5, 0.5, -0.5, 1.5, and the slope is sum(dx * dy) / sum(dx²) = 4 / 5 = 0.8
    per year.

    Args:
        da (xr.DataArray | xr.Dataset): The series (each variable of a Dataset on its own).
        dim (str | Sequence[str]): The dim(s) holding the samples: "year", ("year", "season") for one slope through
            the four seasons pooled, or a datetime dim ("time"). The x values are always the years (a date's calendar
            year).

    Returns:
        Same type as ``da``, with ``dim`` reduced.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    x = _years(da, dims)
    dx = x - x.mean(dims)
    dy = da - da.mean(dims)
    return (dx * dy).sum(dims) / (dx ** 2).sum(dims)


def linear_fit(da, dim='year'):
    """The least-squares straight line through each series, on the series' own years (NaN where it is missing).

    Example: the series 1, 3, 2, 4 over the years 2000-2003 has the slope 0.8 per year (``trend``) through its means
    (2001.5, 2.5), so the line is 2.5 + 0.8 * (year - 2001.5): 1.3, 2.1, 2.9, 3.7.

    Args:
        da (xr.DataArray | xr.Dataset): The series.
        dim (str | Sequence[str]): As ``trend``'s.

    Returns:
        Same type and shape as ``da``.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    x = _years(da, dims)
    return da.mean(dims) + trend(da, dim) * (x - x.mean(dims))


def detrend(da, dim='year'):
    """Each series minus its own least-squares line (``linear_fit``), leaving only the year-to-year variability.

    Example: the series 1, 3, 2, 4 minus its line 1.3, 2.1, 2.9, 3.7 is -0.3, 0.9, -0.9, 0.3.
    """
    return da - linear_fit(da, dim)


# ---------------------------------------------------------------------------
# LOWESS smoothing
# ---------------------------------------------------------------------------

def _lowess_weights(n, frac):
    """Sparse weight matrices for local linear regression with tricube weights."""
    r = max(int(np.ceil(frac * n)), 2)
    x = np.arange(n, dtype=float)
    rows, cols, w_vals, wu_vals, wu2_vals = [], [], [], [], []

    for i in range(n):
        lo = min(max(i - r // 2, 0), n - r)
        idx = np.arange(lo, lo + r)
        u = x[idx] - x[i]

        d = np.abs(u).max()
        w = np.ones_like(u) if d == 0 else (1 - np.abs(u / d) ** 3) ** 3

        rows.append(np.full(r, i))
        cols.append(idx)
        w_vals.append(w)
        wu_vals.append(w * u)
        wu2_vals.append(w * u**2)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    W = sparse.csr_matrix((np.concatenate(w_vals), (rows, cols)), shape=(n, n))
    WU = sparse.csr_matrix((np.concatenate(wu_vals), (rows, cols)), shape=(n, n))
    WU2 = sparse.csr_matrix((np.concatenate(wu2_vals), (rows, cols)), shape=(n, n))

    return W, WU, WU2


@skip_empty
def lowess(da, core_dims='year', window: int = 81):
    """LOWESS along ``core_dims``: each year's value is a straight line fitted to the ``window`` years around it,
    nearer years weighted more (tricube weights), evaluated at that year.

    The one smoother of the repository: the forced response (``forced_response``, below), the forced S/N
    (``emergence.proper_sn``) and a series' S/N against itself (``emergence.self_signal_to_noise``) all use it. Not
    robust: no second pass that down-weights outliers (an ensemble mean has few). Every series is smoothed at once,
    as one sparse matrix product; missing years are left out of each fit and stay missing.

    Example: a straight line comes back unchanged (a local straight-line fit of a straight line is the line). With
    ``window=81`` over 1850-2014, 1900's value is the line fitted to 1860-1940; near the ends the 81 years shift
    inwards (1850's uses 1850-1930).

    Args:
        da (xr.DataArray | xr.Dataset): Data with ``core_dims``.
        core_dims (str): The dimension to smooth along.
        window (int): Years in each local fit.

    Returns:
        The same type and shape, smoothed.
    """
    #(c): One block along the smoothed dimension, as each fit needs the whole series
    if da.chunks:
        da = da.chunk({core_dims: -1})
    frac = window / da.sizes[core_dims]
    W, WU, WU2 = _lowess_weights(da.sizes[core_dims], frac=frac)

    def smooth(y):
        shape = y.shape
        y = y.reshape(-1, shape[-1])

        valid = np.isfinite(y)
        y0 = np.where(valid, y, 0.0)
        m = valid.astype(float)

        s0 = (W @ m.T).T
        s1 = (WU @ m.T).T
        s2 = (WU2 @ m.T).T

        wy = (W @ y0.T).T
        wuy = (WU @ y0.T).T

        denom = s0 * s2 - s1**2

        out = np.full_like(y0, np.nan, dtype=float)
        np.divide(
            s2 * wy - s1 * wuy,
            denom,
            out=out,
            where=denom != 0,
        )

        out[~valid] = np.nan

        return out.reshape(shape)

    return xr.apply_ufunc(
        smooth,
        da,
        input_core_dims=[[core_dims]],
        output_core_dims=[[core_dims]],
        dask="parallelized",
        output_dtypes=[float],
    )


@skip_empty
def forced_response(members, centre, window):
    """The forced response: the ensemble median or mean in each year, smoothed with LOWESS.

    Subtracted from every member, it leaves each member's internal
    variability, so a quantile range of what is left is the variability, free
    of any forced warming inside the analysis window. Smoothing matters:
    subtracting the raw ensemble mean would also remove its sampling noise from
    every member, shrinking the spread by sqrt(1 - 1/N), most for the smallest
    ensembles. Forced year-to-year signals too quick for the smoother (a
    volcanic eruption) stay in the members.

    Args:
        members (xr.Dataset | xr.DataArray): With ``member`` and ``year`` dims.
        centre (str): "median" or "mean" across members. Notebook calc_02a removes the median's (so what is removed
            is the shift of the middle, Q50), calc_02b's S/N uses the mean's.
        window (int): LOWESS window in years (81 in every notebook).

    Returns:
        The same type, without ``member``.
    """
    if centre not in ("median", "mean"):
        raise ValueError(f"centre must be 'median' or 'mean', not {centre!r}")
    forced = members.mean("member") if centre == "mean" else members.median("member")
    if forced.chunks:
        forced = forced.chunk({"year": -1})
    return lowess(forced, core_dims="year", window=window)


# ---------------------------------------------------------------------------
# Mann-Kendall trend test
# ---------------------------------------------------------------------------

def mk_S(y, i, j):
    return np.nansum(np.sign(y[..., j] - y[..., i]), axis=-1)

def grid_mk(da, dim='time', alpha=0.05, min_n=10):
    n = da.sizes[dim]
    i, j = np.triu_indices(n, k=1)

    S = xr.apply_ufunc(lambda y: mk_S(y, i, j), da,
                       input_core_dims=[[dim]],
                       dask='parallelized', output_dtypes=[float])

    # per-cell effective sample size and variance
    n_eff = da.notnull().sum(dim)
    var = n_eff * (n_eff - 1) * (2 * n_eff + 5) / 18.0
    sd = np.sqrt(var)

    Z = xr.where(S > 0, (S - 1) / sd, xr.where(S < 0, (S + 1) / sd, 0.0))
    Z = Z.where(n_eff >= min_n)          # NaN for masked / too-short cells
    p = xr.apply_ufunc(lambda z: 2 * norm.sf(np.abs(z)), Z,
                       dask='parallelized', output_dtypes=[float])

    return xr.Dataset({'S': S, 'Z': Z, 'p': p,
                       'sig': (p < alpha).where(p.notnull()),
                       'n_eff': n_eff})
