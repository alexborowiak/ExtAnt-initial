"""Signal-to-noise: when and where a forced change stands out from natural variability.

Two definitions of the forced S/N, side by side, for comparison:

    proper_sn     historical against hist-nat, one value of noise (notebook calc_03c, section 1)
                  signal  the historical ensemble mean's anomaly from hist-nat's climatology (its mean over every year
                          and member), LOWESS-smoothed
                  noise   hist-nat's standard deviation over every year and member: one number per grid point and
                          season
    rolling_sn    any experiment against hist-nat, noise that changes through time (notebook calc_02b, section 2)
                  signal  the experiment's forced response minus hist-nat's, each its ensemble mean LOWESS-smoothed
                  noise   each experiment's members about their forced response, pooled over the WINDOW years centred
                          on each year (rolling_std), the experiment's and hist-nat's variances averaged

Both smooth with ``trend.lowess`` (81-year window). Then the S/N of single series, the same for ERA5 and each member:
``linear_sn`` (the rise of a linear trend over the standard deviation about it) and ``self_signal_to_noise`` (the
change of a LOWESS fit over the deviations from it); and ``emergence_surface_area_percent``, the share of the map
where an S/N is past a threshold.
"""

from typing import Optional

import numpy as np
import xarray as xr

from ..config import WINDOW
from ..utils.datatree import skip_empty
from . import trend
from .spatial import area_weights

def linear_sn(input_da):
    '''S/N ratio from a straight line through each series (each member, or ERA5).

    signal = the line's gradient × years since the first year; noise = the standard deviation of the series
    about the line. Example: 0.03 °C/year over 1979–2013 is a signal of 0.03 × 34 = 1.02 °C by 2013, and
    with a noise of 0.51 °C the S/N in 2013 is 2.
    '''
    #(c): A tree node's Dataset (one variable) becomes its DataArray, without a size-1 'variable' dim
    if isinstance(input_da, xr.Dataset): input_da = input_da.to_dataarray().squeeze('variable', drop=True)

    output_da = trend.linear_fit(input_da, dim='year')
    output_da['linear_signal'] = output_da.gradient * (output_da.year - output_da.year.min())
    output_da['linear_detrend'] = input_da - output_da.fit
    output_da['linear_noise'] = output_da.linear_detrend.std('year')
    output_da['linear_sn'] = output_da.linear_signal / output_da.linear_noise

    return output_da


@skip_empty
def proper_sn(input_da, base_period:Optional=(1850, 1880), histnat_da:Optional=None):
    '''Forced S/N of an ensemble: the LOWESS-smoothed (81-year window) ensemble-mean anomaly over the noise.

    The anomaly is from hist-nat's climatology (``histnat_da``'s mean over the years and members) and the noise is
    hist-nat's standard deviation over the years and members. Without ``histnat_da``, both come from
    ``base_period`` of ``input_da`` (the noise from the series minus the signal).
    '''
    #(c): A tree node's Dataset (one variable) becomes its DataArray, without a size-1 'variable' dim
    if isinstance(input_da, xr.Dataset): input_da = input_da.to_dataarray().squeeze('variable', drop=True)

    if histnat_da is not None:
        if isinstance(histnat_da, xr.Dataset): histnat_da = histnat_da.to_dataarray().squeeze('variable', drop=True)
        #(c): Over the members too: hist-nat's members are not historical's (CanESM5: 50 and 65), and a climatology
        #(c): kept per member would line up with historical's only where the names match, dropping the others
        climatology = histnat_da.mean(dim=['year', 'member'])
    else:
        climatology = input_da.sel(year=slice(*base_period)).mean(dim='year')

    # Signal calculated with reference to the climatology (can be base period of hist-nat)
    anomalies = input_da - climatology
    unsmooth_signal = anomalies.mean(dim='member').to_dataset(name='signal')
    output_da = trend.lowess(unsmooth_signal)

    # Noise: Relevenat which one is used - hist-nat or base period
    if histnat_da is not None:
        da_for_noise = histnat_da
    else:
        detrend = input_da - output_da['signal']
        da_for_noise = detrend.sel(year=slice(*base_period))
    
    output_da['noise'] = da_for_noise.std(['year', 'member'])
    
    output_da['sn'] = output_da['signal']/output_da['noise']

    return output_da


@skip_empty
def rolling_sn(input_da, histnat_da, window=WINDOW, lowess_window=81):
    """Forced S/N of one experiment against hist-nat, with a noise that changes through time.

    For each of the experiment and hist-nat: its forced response is its ensemble mean, LOWESS-smoothed (as an anomaly
    from its own mean over every year and member, which is then added back), and its noise is the standard deviation
    of its members about that forced response, pooled over the ``window`` years centred on each year (``rolling_std``).
    Then

        signal = forced(experiment) - forced(hist-nat)
        noise  = sqrt((noise(experiment)² + noise(hist-nat)²) / 2)
        S/N    = signal / noise

    Example: in 2000 the experiment's forced response is 0.8 °C above hist-nat's, and their members scatter about
    their forced responses by 0.5 °C and 0.7 °C; the noise is sqrt((0.25 + 0.49) / 2) = 0.61 °C and the S/N 1.3.

    The steps one at a time, on every model and experiment at once, are notebook calc_02b's section 2.

    Args:
        input_da, histnat_da (xr.DataArray | xr.Dataset): The experiment's and hist-nat's members on (member, year,
            ...); a Dataset of one variable (a tree node) is taken as its DataArray.
        window (int): Years pooled for the noise.
        lowess_window (int): Years in each LOWESS fit of the forced response.

    Returns:
        xr.Dataset: ``forced`` (the experiment's forced response), ``signal``, ``noise`` and ``sn``, on (year, ...).
    """
    #(c): A tree node's Dataset (one variable) becomes its DataArray, without a size-1 'variable' dim
    if isinstance(input_da, xr.Dataset): input_da = input_da.to_dataarray().squeeze('variable', drop=True)
    if isinstance(histnat_da, xr.Dataset): histnat_da = histnat_da.to_dataarray().squeeze('variable', drop=True)

    def forced_and_noise(members_da):
        #(c): The ensemble mean smoothed as an anomaly from the whole-record mean, which is then added back
        record_mean_da = members_da.mean(['year', 'member'])
        anomaly_da = members_da - record_mean_da
        smooth_da = trend.lowess(anomaly_da.mean('member'), core_dims='year', window=lowess_window)
        noise_da = rolling_std(anomaly_da - smooth_da, window=window)
        return smooth_da + record_mean_da, noise_da

    forced_da, noise_da = forced_and_noise(input_da)
    histnat_forced_da, histnat_noise_da = forced_and_noise(histnat_da)
    signal_da = forced_da - histnat_forced_da
    pooled_noise_da = np.sqrt(0.5 * (noise_da ** 2 + histnat_noise_da ** 2))
    return xr.Dataset({'forced': forced_da, 'signal': signal_da, 'noise': pooled_noise_da,
                       'sn': signal_da / pooled_noise_da})


def emergence_surface_area_percent(da, thresholds):
    """Percent of the area where ``da`` is greater than each threshold, each grid cell weighted by cos(latitude).

    Example: an S/N of 2 at 70°S and 0.5 at 80°S, threshold 1: the weights are cos 70° = 0.34 and cos 80° = 0.17,
    so 100 × 0.34 / (0.34 + 0.17) = 66% of the area has emerged. Missing values count as not emerged.

    Args:
        da (xr.DataArray): On (..., lat, lon), e.g. the S/N on (model, year, season, lat, lon).
        thresholds (float | list[float]): E.g. [0.5, 1, 2].

    Returns:
        xr.DataArray: On (threshold, ...), lat and lon gone.
    """
    thresholds_da = xr.DataArray(np.atleast_1d(thresholds), dims='threshold')
    thresholds_da = thresholds_da.assign_coords(threshold=thresholds_da)
    weights_da = area_weights(da['lat'])
    exceeds_da = (da > thresholds_da).where(da.notnull())
    return 100 * exceeds_da.weighted(weights_da).mean(['lat', 'lon'])


#(t): LOWESS window for the emergence statistic, in years (a ~35-year record cannot carry the 81 used elsewhere)
EMERGENCE_WINDOW = 25


def _last_minus_first(values):
    """Last finite value minus the first, along the last axis."""
    finite = np.isfinite(values)
    first = np.argmax(finite, axis=-1)[..., None]
    last = (values.shape[-1] - 1 - np.argmax(finite[..., ::-1], axis=-1))[..., None]
    change = np.take_along_axis(values, last, -1)[..., 0] - np.take_along_axis(values, first, -1)[..., 0]
    return np.where(finite.any(axis=-1), change, np.nan)


def self_signal_to_noise(da, dim='year', window=EMERGENCE_WINDOW):
    """S/N of a series against itself: how far it has moved relative to its own year-to-year noise.

    signal   the change of its LOWESS-smoothed curve from the start of the record to the end
    noise    the standard deviation of its deviations from that curve

    ERA5 has no counterfactual, so this is the emergence measure that can be
    computed for ERA5 and, identically, for every member.
    """
    smooth = trend.lowess(da, core_dims=dim, window=window)
    signal = xr.apply_ufunc(_last_minus_first, smooth, input_core_dims=[[dim]], dask="parallelized",
                            output_dtypes=[float])
    return signal / (da - smooth).std(dim, ddof=1)


# ---------------------------------------------------------------------------
# The noise of a S/N that changes through time (notebook calc_02b, section 2)
# ---------------------------------------------------------------------------

def _centred_sums(x, window):
    """Sums over centred ``window``-long windows along the last axis, shortened at the ends.

    From one running total: a window's sum is the total to its last value minus
    the total before its first.

    Example: x = [2, 3, 1, 4] and window = 3 have the running totals
    [0, 2, 5, 6, 10]. The window centred on the third value runs from the second
    to the fourth: 10 - 2 = 8 (3 + 1 + 4). Centred on the first it is cut short
    to the first two: 5 - 0 = 5. Notebook calc_02b (section 2) goes through it.
    """
    half = window // 2
    n = x.shape[-1]
    #(c): cumulative[i] is the sum of the first i values, so cumulative[0] = 0 is the total before the first
    cumulative = np.concatenate([np.zeros_like(x[..., :1]), np.cumsum(x, axis=-1)], axis=-1)
    position = np.arange(n)
    #(c): Each window's last value is half after its centre and its first half before, both kept inside the record
    return cumulative[..., np.minimum(position + half + 1, n)] - cumulative[..., np.maximum(position - half, 0)]


def _rolling_std_block(x, window):
    """Standard deviation (ddof=0) of every member's values in each centred window; x is (..., member, year)."""
    valid = np.isfinite(x)
    x = np.where(valid, x, 0.0).astype(np.float64)
    #(c): Each window's count, sum and sum of squares, from each year's over the members (axis -2)
    count = _centred_sums(valid.sum(axis=-2).astype(np.float64), window)
    mean = _centred_sums(x.sum(axis=-2), window)
    square = _centred_sums((x * x).sum(axis=-2), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean /= count
        #(c): The variance is the mean square minus the squared mean: [1, 2, 3] gives 14 / 3 - 2² = 0.67
        #(c): (rounding can take it a hair below 0, hence the maximum)
        variance = np.maximum(square / count - mean * mean, 0.0)
    return np.where(count > 0, np.sqrt(variance), np.nan)


@skip_empty
def rolling_std(da, window=WINDOW, dim="year", member_dim="member"):
    """Standard deviation of every member's values in a centred ``window``-year window, for each year.

    The same numbers as ``da.rolling(year=window, center=True).construct('window').std(['window', 'member'])``
    (ddof=0; NaN skipped; windows shortened at the ends of the record), but
    from running sums of x and x², so no copy of the data per window: the
    construct version holds ``window`` copies of the data in memory at once.

    Args:
        da (xr.DataArray | xr.Dataset): With ``member_dim`` and ``dim``.
        window (int): Window length in years.
        dim, member_dim (str): The rolling and the pooled dimension.

    Returns:
        The same type, without ``member_dim``.
    """
    if da.chunks:
        da = da.chunk({member_dim: -1, dim: -1})
    out = xr.apply_ufunc(
        _rolling_std_block, da,
        input_core_dims=[[member_dim, dim]], output_core_dims=[[dim]],
        kwargs={"window": window}, dask="parallelized", output_dtypes=[np.float64],
    )
    return out.transpose(dim, ...)
