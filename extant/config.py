"""Settings shared by every notebook: experiments, windows, the example point, and the variables.

Change a setting here and every notebook picks it up. Settings that only matter
to one figure stay in that figure's notebook. File locations are in ``paths``.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Experiments and models
# ---------------------------------------------------------------------------

#(t): The experiments analysed. hist-nat is the counterfactual every other experiment is compared with
MAIN_EXPERIMENTS = ["hist-nat", "hist-aer", "hist-totalO3", "hist-GHG", "historical"]
FORCED_EXPERIMENTS = ["hist-aer", "hist-totalO3", "hist-GHG", "historical"]

#(t): Which version of the raw files to convert: 'interp' is every model on the common 2.5° grid
GROUP = "interp"

#(t): Models are only converted with at least MIN_EXPERIMENTS of the experiments; the N_MODELS with most are analysed
MIN_EXPERIMENTS = 4
N_MODELS = 6

# ---------------------------------------------------------------------------
# Periods and windows
# ---------------------------------------------------------------------------

#(t): Length in years of every window: the rolling quantiles and noise, and the final period compared with hist-nat
WINDOW = 21

#(t): Index of the last year whose centred WINDOW-year window is complete: the centre of the final WINDOW years
LAST_FULL_YEAR = -(WINDOW // 2 + 1)

#(t): historical ends in 2014, so every experiment is cut there and "the final years" are the same years in all
#(c): '2014-12', not '2014-12-31': 360-day calendars have no 31st
LAST_MONTH = "2014-12"

#(t): ERA5 is only trustworthy over Antarctica from 1979 (the satellite era)
ERA5_START_YEAR = 1979

#(t): Southern edge of the stored data: everything south of 40°S
LAT_MAX = -40

# ---------------------------------------------------------------------------
# What the figures show
# ---------------------------------------------------------------------------

#(t): The example grid point (the sample data are the 3x3 block of cells around it)
POINT = dict(lat=-81.88, lon=120.9, method="nearest")

#(t): Model and experiment shown in single-model detail plots
PLOT_SEL = dict(model="CanESM5", experiment="historical")

#(t): Quantiles for the rolling-quantile analysis: the median and three symmetric pairs
QUANTILES = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])

#(t): Multi-model agreement thresholds (IPCC AR6 advanced approach)
SIGNIFICANT_THRESHOLD = 0.66   # fraction of models that must be significant
SIGN_THRESHOLD = 0.8           # fraction of models that must agree on the sign of the median

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variable:
    """How to read one variable and which units to analyse it in.

    Attributes:
        name: CMIP name. Also the name of the data variable in every store this project writes.
        long_name: For figure labels.
        units: Units after ``to_analysis_units``.
        scale, offset: analysis value = raw value * scale + offset.
        era5_code: Code in the hourly ERA5 file names (``*.2t.nc``); None if there is no ERA5 recipe yet.
        era5_name: Name of the variable inside those files.
        table: CMIP table, the folder of ``paths.LESFMIP_RAW`` the raw files are in.
    """

    name: str
    long_name: str
    units: str
    scale: float = 1.0
    offset: float = 0.0
    era5_code: str | None = None
    era5_name: str | None = None
    table: str = "Amon"

    def to_analysis_units(self, obj):
        """Convert from the units of the raw files to ``units``."""
        if self.scale != 1.0:
            obj = obj * self.scale
        if self.offset != 0.0:
            obj = obj + self.offset
        return obj

    @property
    def label(self):
        """e.g. 'Near-surface air temperature (°C)'."""
        return f"{self.long_name} ({self.units})"


#(c): pr and sfcWind follow the CMIP conventions (kg m-2 s-1 and m s-1) but have not been run yet. ERA5
#(c): precipitation is a forecast accumulation (fc_sfc, not an_sfc) and wind speed has to be built from 10u
#(c): and 10v, so neither has an ERA5 recipe in convert.py yet.
#(c): siconc (sea-ice concentration, %) is not analysed itself: it gives the sea-ice edge (``sea_ice``). Its
#(c): table, SImon, is a guess at where the interpolated files are; check it against the folders on JASMIN.
VARIABLES = {
    "tas": Variable("tas", "Near-surface air temperature", "°C", offset=-273.15, era5_code="2t", era5_name="t2m"),
    "pr": Variable("pr", "Precipitation", "mm day⁻¹", scale=86400.0),
    "sfcWind": Variable("sfcWind", "Near-surface wind speed", "m s⁻¹"),
    "siconc": Variable("siconc", "Sea-ice concentration", "%", table="SImon"),
}
