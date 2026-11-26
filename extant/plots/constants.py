"""Domain vocabulary: names, colours and orderings that mean something in the science.

Only things with a stable meaning *outside* any one figure belong here; a number that exists because one plot looks
better with it is a default argument of that plot. Figure sizes and styles are not here either: every chart's layout
is extant/plots/charts.py's, every map's antarctica.py's, and their text sizes those of styles.py
(NOTEBOOK, PAPER, SLIDES), and the maps' colour scales are in colour_scales.py.
"""

# --------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------

#(c): A list, not a tuple: xarray takes a list in .sel(season=SEASONS), but a tuple as one label
SEASONS = ["DJF", "MAM", "JJA", "SON"]

SEASON_COLORS = {
    "DJF": "#1f77b4",
    "MAM": "#2ca02c",
    "JJA": "#d62728",
    "SON": "#ff7f0e",
}

# --------------------------------------------------------------------------
# Single-forcing experiments. The reveal order is also the order they are
# introduced in talks, so keep the three mappings below in step.
# --------------------------------------------------------------------------

FORCING_REVEAL_ORDER = ("hist-nat", "hist-GHG", "hist-aer", "hist-totalO3", "historical")

FORCING_COLORS = {
    "hist-nat": "#2E8B57",
    "hist-GHG": "#D1495B",
    "hist-aer": "#3A86C8",
    "hist-totalO3": "#8A5FBF",
    "historical": "#1A1A1A",
}

FORCING_LEGEND_LABELS = {
    "hist-nat": "hist-nat (natural only)",
    "hist-GHG": "hist-GHG (greenhouse gases)",
    "hist-aer": "hist-aer (aerosols)",
    "hist-totalO3": "hist-totalO3 (ozone)",
    "historical": "historical (all forcings)",
}

assert set(FORCING_COLORS) == set(FORCING_REVEAL_ORDER) == set(FORCING_LEGEND_LABELS), (
    "the three forcing mappings have drifted apart"
)

# --------------------------------------------------------------------------
# Colours every figure shares
# --------------------------------------------------------------------------

INK = "#0b0b0b"             # primary text
INK_2 = "#52514e"           # secondary text
MUTED = "#a3a29d"           # reference lines

#(t): ERA5, black in every figure; and where ERA5 is outside the ensemble (an accent in the method figures)
ERA5_COLOR = "#111111"
OUTSIDE_COLOR = "#eb6834"

#(t): The shading of the years ERA5 covers, behind a figure of the whole period
ERA5_YEARS_COLOR = "#dbe9f6"

#(t): One blue ramp for the ensemble, light (wide ranges) to dark (central values)
ENSEMBLE_COLORS = {
    "range": "#cde2fb",    # full member range
    "outer": "#9ec5f4",    # 5-95%
    "member": "#86b6ef",   # individual members
    "inner": "#5598e7",    # central 75%, 12.5-87.5%
    "pooled": "#1c5cab",   # pooled ensemble
    "median": "#104281",   # ensemble median
}

#(t): What is not a forcing, drawn beside the forcings
OTHER_COLORS = {"sum of parts": "#c49a00", "residual": "#c9c8c4", "ERA5": ERA5_COLOR}
