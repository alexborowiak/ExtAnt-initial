"""Settings shared by every notebook: experiments, windows, thresholds, and the variables.

Change a setting here and every notebook picks it up. Settings that only matter
to one figure stay in that figure's notebook. The folders on JASMIN are in ``src/paths.py``.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Experiments and models
# ---------------------------------------------------------------------------

#(t): The experiments analysed. hist-nat is the counterfactual every other experiment is compared with
MAIN_EXPERIMENTS = ["hist-nat", "hist-aer", "hist-totalO3", "hist-GHG", "historical"]

#(t): Every model with LESFMIP data is analysed except these, e.g. a model whose sea ice is too far from ERA5's (notebook
#(t): calc_04). The analysis notebooks drop them as they open the data; calc_01 and calc_04 still process every model
EXCLUDED_MODELS = []

#(t): The experiments compared with hist-nat, in the order of every figure
FORCED_EXPERIMENTS = ["hist-GHG", "hist-aer", "hist-totalO3", "historical"]

#(t): Which version of the raw files to convert: 'interp' is every model on the common 2.5° grid
GROUP = "interp"

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

#(t): The years ERA5 and LESFMIP share, for every comparison with ERA5: from ERA5_START_YEAR to LESFMIP's last whole
#(t): season, DJF 2013 (historical ends in December 2014, so DJF 2014 is not complete)
SATELLITE_YEAR_RANGE = slice(ERA5_START_YEAR, 2013)

#(t): Southern edge of the stored data: everything south of 40°S
LAT_MAX = -40

#(t): Concentration (%) from which a cell counts as sea ice: the usual definition of the extent, and of the edge on the
#(t): maps
ICE_THRESHOLD = 15.0

# ---------------------------------------------------------------------------
# The tests and their settings
# ---------------------------------------------------------------------------

#(t): Quantiles for the rolling-quantile analysis: the median and three symmetric pairs
QUANTILES = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])

#(t): The significance level of every test of the forced response (notebooks calc_02a to calc_02c, and their figures)
ALPHA = 0.05

#(t): Members in each trial of the hist-nat bootstrap (the width and the mean tests, notebooks calc_02a and calc_02b):
#(t): the smallest ensemble
N_BOOTSTRAP_MEMBERS = 10

#(t): Trials of that bootstrap, the same in both tests (trial k draws the same members in each)
N_TRIALS = 1000

#(t): Which test says the mean has changed, where the mean and the width are put together (notebooks plot_03c and
#(t): plot_05): 'significant' (the hist-nat bootstrap, calc_02b section *Mean Change and Signal to Noise*), 'emerged' (S/N) or 'robust' (both)
MEAN_TEST = "significant"

#(t): |S/N| counted as emerged, and the S/N thresholds of the areas and of the model counts
SN_THRESHOLD = 2
SN_THRESHOLDS = [0.5, 1, 2]

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
        table: CMIP table, the folder the raw files are in (notebook calc_01's ``RAW_DIR``).
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
#(c): siconc (sea-ice concentration, %) is not in the LEADER-EPESC folders: notebook calc_04 takes it from CEDA's copy of
#(c): CMIP6 (calc_04's ``CEDA_DIRS``, table SImon), each model on its own grid, and puts it on the 2.5° grid. ERA5's
#(c): sea-ice cover (code ci) is a fraction from 0 to 1; notebook calc_04 stores it in % like the models.
#(c): sam is the Southern Annular Mode index, already standardised, from ``paths.SAM_RAW_DIR`` rather than the
#(c): LESFMIP files (``sam.open_index``); it sorts the seasons into SAM phases (notebook calc_02a, *SAM Phases*).
VARIABLES = {
    "tas": Variable("tas", "Near-surface air temperature", "°C", offset=-273.15, era5_code="2t", era5_name="t2m"),
    "pr": Variable("pr", "Precipitation", "mm day⁻¹", scale=86400.0),
    "sfcWind": Variable("sfcWind", "Near-surface wind speed", "m s⁻¹"),
    "siconc": Variable("siconc", "Sea-ice concentration", "%", era5_code="ci", era5_name="siconc", table="SImon"),
    "sam": Variable("sam", "Southern Annular Mode index", "standard deviations"),
}
