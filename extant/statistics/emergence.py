from typing import Optional

import xarray as xr

from extant.datatree import skip_empty

def linear_sn(input_da):
    '''S/N ratio calculated using a linear trend fit for each member.'''
    if isinstance(input_da, xr.Dataset): input_da = input_da.to_array()

    output_da = trend.linear_fit(input_da, dim='year')
    output_da['linear_signal'] = output_da.gradient * (output_da.year - output_da.year.min())
    output_da['linear_detrend'] = input_da - output_da.fit
    output_da['linear_noise'] = output_da.linear_detrend.std('year')
    output_da['linear_sn'] = output_da.linear_signal / output_da.linear_noise

    return output_da


@skip_empty
def proper_sn(input_da, base_period:Optional=(1850, 1880), histnat_da:Optional=None):
    '''S/N ratio calculated with linear trend fit with respect to base period'''
    if isinstance(input_da, xr.Dataset): input_da = input_da.to_array()
        
    if histnat_da is not None:
        if isinstance(histnat_da, xr.Dataset): histnat_da = histnat_da.to_array()
        climatology = histnat_da.mean(dim='year')
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