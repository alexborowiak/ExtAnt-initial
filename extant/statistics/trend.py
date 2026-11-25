import numpy as np
import xarray as xr

from ..utils.datatree import skip_empty
from scipy import sparse
from scipy.stats import norm


SAMPLE_DIM = 'year'


def _slope_and_line(da, dim=SAMPLE_DIM, x=None):
    """The least-squares slope and fitted line of each series (each variable of a Dataset too), ignoring missing samples."""
    dims = [dim] if isinstance(dim, str) else list(dim)
    if x is None:
        x = da[SAMPLE_DIM if SAMPLE_DIM in dims else dims[0]]

    #(c): Datetime coordinates regress on the calendar year, giving a gradient per year
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.dt.year

    #(c): Mask x wherever y is missing, so both means use the same samples
    x = x.astype(float).broadcast_like(da).where(da.notnull())
    dx = x - x.mean(dims)
    dy = da - da.mean(dims)
    slope = (dx * dy).sum(dims) / (dx**2).sum(dims)
    return slope, da.mean(dims) + slope * dx


def linear_fit(da, dim=SAMPLE_DIM, x=None):
    """Least-squares straight line through each series, ignoring missing samples.

    Example: the series 1, 3, 2, 4 over the years 2000-2003 has the line 1.3 + 0.8 * (year - 2000): ``gradient`` 0.8
    per year and ``fit`` 1.3, 2.1, 2.9, 3.7.

    Args:
        da (xr.DataArray): Data to fit (``trend`` and ``detrend`` take a Dataset too).
        dim (str | Sequence[str]): Dimension(s) holding the samples, e.g. "year",
            or ("year", "season") for one line through all four seasons.
        x (xr.DataArray | None): The regressor. Defaults to the "year"
            coordinate (or the coordinate of the first of ``dim``).

    Returns:
        xr.Dataset: ``fit``, the fitted line on ``da``'s shape, and ``gradient``,
        the slope per unit of ``x`` with ``dim`` reduced.
    """
    if isinstance(da, xr.Dataset): da = da.to_array()
    slope, line = _slope_and_line(da, dim, x)
    return line.to_dataset(name='fit').assign(gradient=slope)


def trend(da, dim=SAMPLE_DIM):
    """Least-squares trend, per decade: the gradient (per year) times 10, of a DataArray or each variable of a Dataset."""
    slope, _ = _slope_and_line(da, dim)
    return slope


def detrend(da, dim=SAMPLE_DIM):
    """Each series minus its own least-squares line (``linear_fit``'s fit), leaving only the year-to-year variability.

    Takes a DataArray or a Dataset (each variable detrended on its own)."""
    _, line = _slope_and_line(da, dim)
    return da - line


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

    The one smoother of the repository: the forced response (``response_change.forced_response``), the forced S/N
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