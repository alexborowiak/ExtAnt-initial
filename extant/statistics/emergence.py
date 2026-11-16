from typing import Optional

import numpy as np
import xarray as xr

from extant.datatree import skip_empty
from extant.statistics import trend

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
    output_da = trend.lowess_matrix_xarray(unsmooth_signal)

    # Noise: Relevenat which one is used - hist-nat or base period
    if histnat_da is not None:
        da_for_noise = histnat_da
    else:
        detrend = input_da - output_da['signal']
        da_for_noise = detrend.sel(year=slice(*base_period))
    
    output_da['noise'] = da_for_noise.std(['year', 'member'])
    
    output_da['sn'] = output_da['signal']/output_da['noise']

    return output_da


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
    weights_da = np.cos(np.deg2rad(da['lat']))
    exceeds_da = (da > thresholds_da).astype(float)
    return 100 * exceeds_da.weighted(weights_da).mean(['lat', 'lon'])
