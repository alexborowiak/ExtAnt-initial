"""Plot pieces several notebooks share: legend entries, the sea-ice zones' colours and the SAM phases' row labels.

Used as ``plot_constants.<name>``, e.g. ``fig.legend(handles=[plot_constants.hist_nat_edge_handle,
plot_constants.experiment_edge_handle])``, so a change here changes every figure that uses it. A legend copies a
handle's look, so one handle can go in any number of legends.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from extant import config, sam, sea_ice
from extant.plots import forced_response as fr

# --------------------------------------------------------------------------
# Legend entries of the maps
# --------------------------------------------------------------------------

#(t): The dots where a change is not significant (p ≥ config.ALPHA)
not_significant_handle = plt.Line2D([], [], color='k', marker='.', linestyle='none',
                                    label=f'Not significant (p ≥ {config.ALPHA:g})')

#(t): The dots where the models don't agree: fewer than two-thirds (config.SIGNIFICANT_THRESHOLD) significant with one sign
not_robust_handle = plt.Line2D([], [], color='k', marker='.', linestyle='none',
                               label='Not robust (fewer than two-thirds\nof the models significant with one sign)')

#(t): The two ice edges: hist-nat's dashed, the experiment's solid
hist_nat_edge_handle = plt.Line2D([], [], color='k', linestyle='dashed', label="hist-nat's ice edge")
experiment_edge_handle = plt.Line2D([], [], color='k', label="The experiment's ice edge")

# --------------------------------------------------------------------------
# Legend entries of the charts
# --------------------------------------------------------------------------

#(t): The dots on a line where it is significant (p < config.ALPHA)
significant_handle = plt.Line2D([], [], linestyle='none', marker='o', markersize=5, color='k', markeredgecolor='white',
                                label=f'Significant (p < {config.ALPHA:g})')

#(t): The bar of the models' median
multi_model_median_patch = Patch(color='0.6', alpha=0.5, label='Multi-model median')

#(t): The band of an ensemble's members, 5% to 95%
members_range_patch = Patch(color='0.6', alpha=0.4, label='5–95% of members')

# --------------------------------------------------------------------------
# Scales and labels
# --------------------------------------------------------------------------

#(t): The sea-ice zones (notebook calc_04, section 7): one colour each, in the order of their numbers (sea_ice.ZONES)
zone_scale = fr.class_scale(sea_ice.ZONES, ['#f4f3ef', '#2c7fb8', '#e6550d', '#31a354', '#bdbdbd'], 'Zone')

#(t): Each SAM phase's row label, e.g. 'SAM\npositive'
phase_labels = {phase: f'SAM\n{phase}' for phase in sam.PHASES}
