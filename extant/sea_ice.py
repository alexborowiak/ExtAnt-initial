"""The sea ice's own names and numbers, shared by notebook calc_04 (which works everything out, step by step) and the
figures that draw what it saves.

Bracegirdle et al. (2024, npj Clim. Atmos. Sci. 7, 276, doi:10.1038/s41612-024-00822-y) mark the ice edge on their
zonal-mean figures with an *equivalent latitude* (Eisenman, 2010; Bracegirdle et al., 2018): the latitude the edge would
be at if the sea ice and the continent it surrounds were a cap round the pole. The area of the Earth south of latitude
phi is

    A(phi) = 2 pi R^2 (1 + sin phi)          (``cap_area``)

so the equivalent latitude is the phi for which A(phi) equals the sea-ice extent plus the area of the continent. The
extent is the ocean of the cells with at least 15% ice, and the continent the land south of 60 degrees S. Notebook
calc_04 (section *Seasonal Means, Extent and Ice Edge*) puts the sea ice on the 2.5 degree grid, works out the extent and the edge, and saves both.
"""

import numpy as np

from . import obs_evaluation as ev

#(t): Mean radius of the Earth (km); areas are in km²
EARTH_RADIUS = 6371.0


def cap_area(lat):
    """Area (km²) of the Earth south of ``lat`` (degrees): 2πR²(1 + sin φ)."""
    return 2 * np.pi * EARTH_RADIUS ** 2 * (1 + np.sin(np.deg2rad(lat)))


#(t): How the figures label ERA5's tests of the extent (notebook calc_04, section *Comparison with ERA5*), and what each means for the model
#(t): when ERA5 falls below (low) or above (high) its members: the mean of the extent, the year-to-year spread of
#(t): the detrended extent, and the linear trend
ERA5_DIAGNOSTICS = {
    "mean": ev.Diagnostic("Mean extent", "million km²", "model has too much ice", "model has too little ice",
                          treatment="raw"),
    "std": ev.Diagnostic("Year-to-year std", "million km²", "model too variable", "model not variable enough",
                         treatment="detrended"),
    "trend": ev.Diagnostic("Trend", "million km² per decade", "model's trend too positive",
                           "model's trend too negative", treatment="raw"),
}

#(t): The zones a cell can be in, comparing an experiment's final years with hist-nat's: open water in both, ice (at
#(t): least config.ICE_THRESHOLD %) in both, ice in hist-nat only (lost), in the experiment only (gained), or the
#(t): continent
ZONES = ("open water", "ice in both", "ice lost", "ice gained", "continent")
