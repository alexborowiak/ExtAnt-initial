"""Plot pieces several notebooks share: legend entries, the look of ERA5's ice edge, the sea-ice zones' colours and the
SAM phases' row labels.

Used as ``plot_constants.<name>``, e.g. ``fig.legend(handles=[plot_constants.hist_nat_edge_handle,
plot_constants.experiment_edge_handle])``, so a change here changes every figure that uses it. A legend copies a
handle's look, so one handle can go in any number of legends.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .. import config, obs_evaluation as ev, sam, sea_ice
from . import charts, colour_scales, constants, zonal

# --------------------------------------------------------------------------
# Legend entries of the maps
# --------------------------------------------------------------------------


experiment_plotting_ordre = ['historical', 'hist-GHG', 'hist-aer', 'hist-totalO3']

#(t): The dots where a change is not significant (p ≥ config.ALPHA)
not_significant_handle = plt.Line2D([], [], color='k', marker='.', linestyle='none',
                                    label=f'Not significant (p ≥ {config.ALPHA:g})')

#(t): The dots where the models don't agree: fewer than two-thirds (config.SIGNIFICANT_THRESHOLD) significant with one sign
not_robust_handle = plt.Line2D([], [], color='k', marker='.', linestyle='none',
                               label='Not robust (fewer than two-thirds\nof the models significant with one sign)')

#(t): The two ice edges: hist-nat's dashed, the experiment's solid
hist_nat_edge_handle = plt.Line2D([], [], color='k', linestyle='dashed', label="Hist-nat ice edge")
experiment_edge_handle = plt.Line2D([], [], color='k', label="Protocal ice edge")

#(t): ERA5's ice edge: black, as ERA5 is in every figure, and dotted (its line plots have dots), apart from the models'
#(t): solid edges; and the two edges' legend entries
era5_edge_style = dict(color=charts.ERA5_LINE['color'], linestyle=':', linewidth=1.6)
model_edge_handle = plt.Line2D([], [], color='k', label="The model's edge")
era5_edge_handle = plt.Line2D([], [], **era5_edge_style, label="ERA5's edge")

#(t): The hatching where ERA5 is in the members' central 75% too often (the spatial evaluation), and far too often
central_75_handle = Patch(facecolor='white', edgecolor='0.35', hatch='////',
                          label=f"ERA5 in the members' central 75% in > {ev.CENTRAL_THRESHOLD}% of years")
central_75_in_90_handle = Patch(facecolor='white', edgecolor='0.35', hatch='xxxx', label='... in > 90% of years')

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

#(t): The band of zonal-mean width changes hist-nat's own variability gives
hist_nat_variability_patch = Patch(color=zonal.NULL_COLOR, label='hist-nat variability alone (95%)')

#(t): The area emerged by its two S/N: ERA5's since 1979, the models' total (solid) and theirs since 1979 (dashed)
era5_linear_sn_handle = plt.Line2D([], [], color='k', linewidth=1.8, linestyle='--', label='ERA5, since 1979')
proper_sn_handle = plt.Line2D([], [], color='0.4', linewidth=1.6,
                              label='Total: forced signal since 1850 ÷ natural variability')
linear_sn_handle = plt.Line2D([], [], color='0.4', linewidth=1.2, linestyle='--',
                              label='Since 1979: linear trend ÷ variability about it')

#(t): The shaded years ERA5 covers, behind a figure of the whole period
era5_years_handle = Patch(color=constants.ERA5_YEARS_COLOR, label=f'ERA5 years ({config.SATELLITE_YEAR_RANGE.start}–'
                                                                  f'{config.SATELLITE_YEAR_RANGE.stop})')

# --------------------------------------------------------------------------
# Scales and labels
# --------------------------------------------------------------------------

#(t): The sea-ice zones (notebook calc_04, section *Sea Ice and Temperature Change*): one colour each, in the order of their numbers (sea_ice.ZONES)
zone_scale = colour_scales.class_scale(sea_ice.ZONES, ['#f4f3ef', '#2c7fb8', '#e6550d', '#31a354', '#bdbdbd'],
                                       'Zone', wrap=8)

#(t): Each SAM phase's row label, e.g. 'SAM\npositive'
phase_labels = {phase: f'SAM\n{phase}' for phase in sam.PHASES}
