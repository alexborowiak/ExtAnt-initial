"""Domain vocabulary: names, colours and orderings that mean something in the science.

Only things with a stable meaning *outside* any one figure belong here; a number that exists because one plot looks
better with it is a default argument of that plot. Figure sizes and styles are not here either: every chart's layout
is extant/plots/charts.py's, every map's antarctica.py's, and their text sizes the styles of forced_response
(NOTEBOOK, PAPER, SLIDES).
"""

# --------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------

SEASONS = ("DJF", "MAM", "JJA", "SON")

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
