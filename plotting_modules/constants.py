"""Domain vocabulary: names, colours and orderings that mean something in the science.

Only things with a stable meaning *outside* any one figure belong here. A number
that exists because one plot happens to look better with it is a default argument
on that plot, not a constant. Two examples of what that rules out:

  PLUME_ALPHAS, SPREAD_STYLES and friends were module-level tuples that only
  quantile_summary ever read. They are now its default arguments, where you can
  see them next to the thing they style and override them per call.

  BUILDUP_XLIM, BUILDUP_FRAME_DIR and BUILDUP_FRAME_SIZE were parameters of one
  figure wearing constant names. Same treatment.

The layout defaults (PANEL_W, CBAR_HEIGHT and the rest) deliberately stay in
core.py, next to the GridLayout parameters they are the defaults for. Moving
them here would separate a default from its argument.
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

# --------------------------------------------------------------------------
# Output styling
# --------------------------------------------------------------------------

SLIDE_RC = {
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 200,
}
