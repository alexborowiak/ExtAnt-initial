# ExtAnt-initial

How greenhouse gases, aerosols, ozone and natural forcing have each changed Antarctic near-surface climate in the LESFMIP single-forcing large ensembles. The analysis covers both the mean and the spread of the seasonal distribution, and checks the models against ERA5 to see where they can be trusted.

The data are on JASMIN. A small sample (a 3×3 block of grid cells from two models, plus ERA5) is in `tests/data`, and a sea-ice sample (one member of each model and experiment for 2013–2014, plus ERA5) in `tests/data/sea_ice`, so all of the code can be run and tested off JASMIN.

## Layout

```
notebooks/
    calc_01_process_data.ipynb               raw files -> monthly stores -> seasonal means (once per variable); Berkeley Earth on the LESFMIP grid; the SAM index (once)
    calc_02a_width_change.ipynb              the forced response, each method explained, saving every result: the SAM phases, the quantiles, the width (hist-nat bootstrap) and its tails, zonal means and additivity
    calc_02b_mean_change.ipynb               ... the mean (four tests), signal to noise, the additivity of the mean and its zonal means
    calc_02c_probability_ratios.ipynb        ... probability ratios and the shift-only ratio, on maps and by SAM phase
    calc_03a_era5_moment_test.ipynb          are the models consistent with ERA5? the climatologies, then each statistic (four moments, trend, width, tails) of every member and ERA5 at every grid point, saved as values
    calc_03b_era5_rank_histograms.ipynb      ... the rank histograms, the paper's grid-point evaluation, and every test in one table, saved
    calc_03c_observed_change.ipynb           is the observed change detectable, and which forcings is it consistent with? saved
    calc_04_sea_ice.ipynb                    the sea ice from CEDA onto the 2.5° grid, extent and edge, against ERA5, with the temperature changes
    plot_01a_era5_distributions_and_plumes.ipynb   the figures of calc_03a, from the saved results: the bias maps (the distributions and plumes at the point are drawn in calc_03a)
    plot_01b_era5_point_tests.ipynb          ... the moment test, rank histograms and scorecard at the point (the step-by-step and toy-ensemble demonstrations are in calc_03b)
    plot_01c_era5_maps.ipynb                 ... the moment test maps with the field test, and the spatial evaluation
    plot_01d_observed_change.ipynb           ... the observed change: the detection maps (calc_03c draws its S/N and record figures itself)
    plot_02_sea_ice.ipynb                    the figures of calc_04, from the saved results
    plot_03a_mean_and_signal_to_noise.ipynb  the forced response figures, from calc_02a to calc_02c's results (four for every result, from what goes in to how many models agree): the mean change and S/N
    plot_03b_quantiles_and_width.ipynb       ... the quantile response, and the change in width and which tail
    plot_03c_distribution_additivity_zonal.ipynb   ... where the mean and the width changed together, additivity, zonal means and the sea-ice edge
    plot_03d_probability_ratios.ipynb        ... probability ratios and the shift-only ratio
    plot_04_slides.ipynb                     figures for talks, including how the forced-response figures are made
    plot_05_paper.ipynb                      the paper's main and supplementary figures, in print style (PDF and PNG)
    plot_06_figure_candidates.ipynb          the candidate figures of the planning notes (A, B, C, E, F), the verification against ERA5 (V), the supplement (S) and the candidate layout left (L3)
    fingerprinting_01_calculations.ipynb     are the observed regional seasonal trends explained by the forcings, and is each forcing's size right? every step, saved
    fingerprinting_02_figures.ipynb          ... its figures, from the saved results
    ensemble_size.ipynb                      how many members are needed (after Milinski et al.)
extant/                                       the code the notebooks use
src/paths.py                                  the folders on JASMIN
tests/                                        pytest suite, the sample data, and a script that runs every notebook
scripts/                                      sea_ice_sample.py: what sea ice CEDA has, and the sea-ice sample (run on JASMIN)
archive/                                      the old draft notebooks
drafts/                                       work in progress: the stations against ERA5 and the models (statation_obs_01), the other observation products and reanalyses (obs_and_reanalysis), the IMBIE basins (regions)
docs/                                         the stations' metadata table, and reference figures (images/)
```

## Where things are in `extant/`

Four questions find anything:

1. Is it about the **shape of the data** (a tree, a time axis, a grid)? → `utils/`
2. Is it **maths any analysis could use**? → `statistics/`
3. Does it **draw**? → `plots/`
4. Is it **one analysis's own**? → the module named after that analysis

So nothing in `utils/` or `statistics/` imports `config` or an analysis module (`tests/test_layout.py` checks it), and an analysis module holds nothing another analysis would want. What is worked out once is written out in its notebook, not in a module; a setting, a colour scale or a legend entry that two notebooks need goes in a module rather than in both.

**General** (the short name in brackets is the one every notebook uses)

| Looking for | Where |
|---|---|
| months → seasons, time → year and season, and back; decimal years | `utils/time_axis`: `seasonal_mean`, `split_time`, `seasonal_means`, `seasons_in_time`, `decimal_year`, `monthly_time_axis` |
| one grid onto another | `utils/regrid`: `match_grid` |
| DataTrees: tree → dataset and back, each experiment's hist-nat, empty nodes | `utils/datatree`: `tree_to_dataset`, `reduce_to_dataset`, `dataset_to_tree`, `hist_nat_like`, `skip_empty`, `streamline_tree`, `swap_levels`, `permute_levels`, `coord_info` |
| the log; the JASMIN cluster | `utils/logger`, `utils/jasmin` |
| a trend, its line, detrending, LOWESS, an ensemble's forced response, Mann-Kendall | `statistics/trend`: `trend` (per year), `linear_fit`, `detrend`, `lowess`, `forced_response`, `grid_mk` |
| skewness, kurtosis | `statistics/moments`: `skewness`, `excess_kurtosis` |
| quantiles, a width, quantiles in a moving window | `statistics/quantiles` (`qc`): `nan_quantile`, `pooled_quantiles`, `quantile_range`, `centred_rolling_quantiles` |
| p-values, bootstraps, the permutation test, the t-test, how many models agree | `statistics/significance` (`sig`): `pvalue_two_sided`, `bootstrap_qrange`, `bootstrap_mean`, `bootstrap_draws`, `bootstrap_member_mean`, `resample_weights`, `resample_members`, `weighted_member_sums`, `quantile_shift`, `member_block_pvalue`, `permutation_weights`, `ttest_welch`, `model_agreement`, `significant_counts`, `significant_sign`, `robust_sign` |
| signal-to-noise, the noise through time, the area where an S/N has emerged | `statistics/emergence`: `proper_sn`, `linear_sn`, `self_signal_to_noise`, `rolling_std`, `emergence_surface_area_percent` |
| area weights, area means, the mean of named regions | `statistics/spatial`: `area_weights`, `area_mean`, `regional_mean`, `regional_means` |
| kernel density estimates | `statistics/kde`: `kde`, `kde_bandwidth` |
| record highs and lows | `statistics/records` |

**This project's own**

| Looking for | Where |
|---|---|
| settings several notebooks share: experiments, models left out, years, window, levels, the tests' trials and thresholds, the sea-ice threshold, variables and units | `config` |
| folders on JASMIN | `src/paths.py` |
| each model's climate sensitivity (ECS, TCR) | `constants` (`extant/constants.py`; the figures' names and colours are in `plots/constants`) |
| opening the raw files: a model's members, a month of ERA5, what is on CEDA | `convert` |
| monthly → the seasonal means every analysis starts from (units, years, chunks, ERA5's grid) | `preprocessing` (`pp`): `seasonal_tree`, `seasonal_era5` |
| the forced change put side by side: one summary, classes, tails, additivity | `response_change` (`rc`) |
| probability ratios: hist-nat's chances, the thresholds, and how often members go beyond them | `probability_ratio` (`pr`) |
| SAM: opening the index, its phases | `sam` |
| zonal means of the change with their tests | `zonal` |
| sea ice: the zones' names, its tests' labels, the Earth's radius and a polar cap's area (calc_04 does the rest, step by step) | `sea_ice` |
| the ERA5 tests (Suarez-Gutierrez et al., 2021): treatments, where ERA5 falls, the field test, rank histograms, the spatial evaluation and its classes | `obs_evaluation` (`ev`) |
| detection classes, record curves | `observed_change` (`oc`) |
| optimal fingerprinting: regions, signal sets, the final fit, vectors, fingerprints, noise, fits, residual test | `fingerprinting` (`fp`) |
| IMBIE basins and sea-ice sectors as masks | `region_shape_files` (`rsf`) |

**Drawing** (`plots/`)

| Looking for | Where |
|---|---|
| season and forcing names, orders and colours; ERA5's, the ensemble's and the text's colours | `constants` |
| legend entries several notebooks share, the look of ERA5's ice edge, the sea-ice zones' colours, the SAM phases' row labels | `plot_constants` |
| text sizes (notebook, paper, slides), page sizes, saving | `styles` |
| a map's colours (one scale per quantity); making a colormap | `colour_scales`; `colormap_creators` (`cc`) |
| maps; every other chart | `antarctica`; `charts` |
| labels of a latitude, a point, a selection, a period | `formatting` |
| the step-by-step method figures; talk figures and their frames | `methods`; `slides` |
| one analysis's panels | the module of the same name: `obs_evaluation` (`evp`), `response_change` (`rcp`), `observed_change` (`ocp`), `sea_ice` (`sip`), `zonal` (`zp`), `fingerprinting` (`fpp`) |

Every figure is made the same way: its axes first, then each layer drawn on them, then its labels. Maps take their axes from `antarctica` (`map_figure_size`, `map_axes`, then `colorbar_axis_below` for each colour bar, under the maps it is for, in a figure made with `layout='constrained'`). Every other plot gets its size from `charts.chart_figure_size` and its axes from a GridSpec (`fig.add_gridspec(...).subplots()`, also constrained). `charts` has the layers shared by many plots (lines, bands, a marker per model, a point per model and forcing), each a one-axis `plot_...` that styles its panel and a `..._grid` over a DataTree or a DataArray.

## Running it

Everything is in this repository: the `extant` package (moved here from the separate extant-functions repository on 7 October 2026; the figures' vocabulary, once `plotting_modules`, is in `extant/plots` since 16 November). Run the notebooks from the `notebooks` folder, in the order of their names: the `calc_` notebooks work everything out and save it, and the `plot_` notebooks draw the figures from what they saved. calc_01 builds the seasonal means; calc_02a to calc_02c, calc_03a to calc_03c and calc_04 use those (calc_02b also reads calc_02a's results, calc_02c both of theirs, calc_03b calc_03a's, and calc_04 calc_02a's and calc_02b's); plot_01a to plot_01d draw calc_03a to calc_03c (plot_01d calc_03c), plot_02 calc_04, plot_03a to plot_03d and plot_04 calc_02a to calc_02c (plot_01c and plot_03a to plot_03c with calc_04's sea ice), and plot_05 and plot_06 everything. The optimal fingerprinting is a pair of its own (fingerprinting_01 works it out from calc_01's seasonal means, fingerprinting_02 draws it), and ensemble_size works out and draws in one notebook. Each notebook opens the data in either a *Sample data* section or a *Full data* section (on JASMIN); run whichever one you want. Every folder on JASMIN is in `src/paths.py`, and each notebook's Settings shows the ones it uses (`SCRATCH_DIR = paths.SCRATCH_DIR`): everything is read from and written to JASMIN's scratch (`SCRATCH_DIR`, `/work/scratch-pw4/aborow/ExtAnt`), and the results of a run go to `SCRATCH_DIR/results/<variable>/<sample or full>`. After moving a folder, change it once in `src/paths.py`.

**Where the data go.** Dask writes zarr stores from many workers at once, and the group workspaces do not support parallel writes. So everything is written to scratch (`SCRATCH_DIR` in `src/paths.py`; the monthly LESFMIP stores to `MONTHLY_DIR`) and read back from there. A store moved by hand to the group workspace keeps the same layout; point `SCRATCH_DIR` there in `src/paths.py`:

```
monthly/era5_<variable>.zarr
seasonal/lesfmip_<variable>.zarr
seasonal/era5_<variable>.zarr
seasonal/lesfmip_sam.zarr                              the SAM index (notebook calc_01, section *SAM Index*)
results/<variable>/<full|sample>/<name>.zarr
results/siconc/<full|sample>/extent.zarr, climatology.zarr, ...     the sea ice (notebook calc_04)
```

The notebook test runner (`tests/run_notebooks.py`) points `SCRATCH_DIR` and the figures folder at a temporary folder, so a test run writes nothing on JASMIN or into the repository.

**Following a notebook.** Names end in their type: `_tree` (DataTree), `_ds` (Dataset), `_da` (DataArray), `_arr` (numpy array), `_df` (DataFrame). Each step shows what it made, and each method figure has a *quick look* before it: the arrays it is given, shown and drawn plainly with xarray's `.plot`, with all the code in the notebook. Every file is opened and saved in the notebook itself, with its full path written out and shown.

**Observations as a tree.** ERA5 is opened as `obs_tree`, a DataTree with one node per observation product (only `/era5` so far): `obs_tree = xr.DataTree.from_dict({'era5': xr.open_zarr(era5_path)[[VARIABLE.name]].load()})`. A new product is one more entry there. Every ERA5 step works on the tree, and where the observations meet the models they are stacked along an `obs` dim, so saved ERA5 results carry `obs` (`.sel(obs='era5')`; in `statistics.zarr` the observations are at `/obs/<product>`).

**The SAM index.** Notebook calc_02a's SAM phases use LEADER-EPESC working group 1's monthly SAM index of every member (`paths.SAM_RAW_DIR`, the files `archive/draft_current_DAILY.ipynb` opened), not one made here from sea-level pressure. Notebook calc_01, section *SAM Index* turns it into seasonal means; it shows the levels in a file so that `SAM_LEVEL` can be set to the right one (the draft took position 3, a later version of its reader position 0). The sample has no SAM index, so it uses a made-up one (`sam.synthetic_index`) that only lets the code run.

**Another variable.** Set `VARIABLE` at the top of each notebook. The variable needs an entry in `config.VARIABLES`, which says how to convert its units; `pr` and `sfcWind` are there but have not been run yet. Two things are not done for them yet. There is no ERA5 recipe: precipitation is a forecast accumulation, and wind speed has to be built from its u and v components. And some figure labels in `extant/plots` still talk about warming and cold or warm tails.

**Sea ice.** The sea ice is not in the LEADER-EPESC folders but in CEDA's copy of the CMIP6 archive (calc_04's `CEDA_DIRS`: DAMIP for the single forcings, CMIP for historical), each model on its own ocean grid, with far fewer members than the temperature. Notebook calc_04 puts it on the 2.5° grid (each cell the area-weighted mean of the points in it, and its ocean area, so the extent counts only the ocean of coastal cells), saves it as monthly stores next to the temperature's, checks it against ERA5's sea-ice cover, and compares the change in sea ice with the mean and width changes. Its `extent` and `climatology` give the sea-ice edge on the maps of notebooks plot_01c, plot_03a–plot_03c and plot_05 (hist-nat's dashed, the experiment's solid), as Bracegirdle et al. (2024, *npj Clim. Atmos. Sci.* 7, 276) mark it. `scripts/sea_ice_sample.py` lists what CEDA has and makes the sample. Without the sea ice, run *Without sea ice* in the notebooks that have it and the figures leave the edge out; only plot_01c's maps and a few of plot_03a's and plot_03b's always draw it.

## Tests

With an environment that has xarray, dask, numba, scipy, statsmodels, cftime, netCDF4, zarr, matplotlib and cartopy:

```
python -m pytest tests
python tests/run_notebooks.py
```

The second runs every notebook on the sample data, skipping the JASMIN-only sections, and writes its outputs to a temporary folder. It takes a few minutes.

## Notes

- Seasons are DJF/MAM/JJA/SON, and DJF is labelled by the year of its December. Every experiment is cut at the end of 2014, so the record ends with DJF 2013 and "the final 21 years" are 1993–2013 in every experiment.
- The width and mean tests (calc_02a's section *Quantiles, Width and Tails* and calc_02b's section *Mean Change and Signal to Noise*) compare an experiment's final 21 years with 10 random hist-nat members over the same years (a bootstrap from hist-nat alone), so one null distribution serves every experiment of a model.
- Every section of calc_02a to calc_02c that has a *By SAM phase* subsection (the mean, the width and tails, the probability ratios) redoes its result in the seasons of each SAM phase: negative, neutral and positive, cut at hist-nat's terciles (in calc_02a, which saves them for the other two).
- The zonal-mean figures (notebook plot_03c, section *Zonal Means and the Sea-Ice Edge*) follow the layout of Bracegirdle et al. (2024), [doi:10.1038/s41612-024-00822-y](https://doi.org/10.1038/s41612-024-00822-y), Figs 1–3, 6 and 8, but show every model on its own and this project's changes against hist-nat.
- ERA5 evaluation follows Suarez-Gutierrez, Milinski and Maher (2021), *Clim. Dyn.* 57, 2557–2580, [doi:10.1007/s00382-021-05821-w](https://doi.org/10.1007/s00382-021-05821-w).

## Moved on 24 November 2026

Names are unchanged unless the new column says so. A cell that fails with `AttributeError: module ... has no attribute ...` is using the old place (after a pull, restart the kernel first).

| Was | Now |
|---|---|
| `pp.seasonal_mean`, `pp.split_time`, `pp.seasonal_means`, `pp.monthly_time_axis` | `time_axis.` |
| `seasons_in_time` (in calc_03a) | `time_axis.seasons_in_time` |
| `pp.match_grid` | `regrid.match_grid` |
| `rc.forced_response` | `trend.forced_response` |
| `trend.linear_fit(x).gradient`, `trend.linear_fit(x).fit` (on 18 November) | `trend.trend(x)`, `trend.linear_fit(x)` |
| `rc.permutation_weights`, `rc.member_block_pvalue`, `rc.quantile_shift` | `sig.` |
| `pr.resample_weights`, `pr.bootstrap_member_mean` | `sig.` |
| `ev.skewness`, `ev.excess_kurtosis` | `moments.` |
| `ev.anomalies(da)` | `ev.treat(da, 'anomaly')` |
| `oc.regional_mean`, `oc.regional_means` | `spatial.` |
| `oc.REGIONS` | `fp.REGIONS` |
| `zonal.zonal_mean(x)` | `x.mean('lon')` |
| `sam.member_of` | `convert.member_from_path` |
| `sea_ice.edge_latitude`, `sea_ice.extent_summary` | calc_04's saved `edge_latitude.zarr`; written out, as in plot_02 |
| `evp.ERA5_COLOR`, `evp.INK`, `evp.INK_2`, `evp.MUTED`, `evp.OUTSIDE_COLOR` | `constants.` |
| `evp.ENSEMBLE` | `constants.ENSEMBLE_COLORS` |
| `paper_panels.OTHER_COLORS` | `constants.OTHER_COLORS` |
| `evp.PERCENTILE_BOUNDS`, `evp.PERCENTILE_EDGES`, `evp.percentile_colormap` | `colour_scales.` |
| `evp.EVAL_RC` | `styles.BASE_RC` |
| `rcp.count_colormap`, `plots.utils.make_diverging_colormap` | `cc.` |
| `zp.latitude_label`, `plots.utils.period_label` | `formatting.` |
| `plots.utils.shared_ylim`, `plots.utils.save_frame` | `slides.` |
| `charts.SIGNAL_COLORS` | `fpp.SIGNAL_COLORS` |
| `evp.EDGE_CLASSES`, `evp.edge_class` | `ev.` |
| `evp.order_seasons(x)` | `x.sel(season=constants.SEASONS)` |
| `antarctica.ICE_EDGE_LEVEL` | `config.ICE_THRESHOLD` |

`ev.DIAGNOSES` now has the names plot_01c and plot_05 showed ("signal overestimated" rather than "warms too much"), and `sig.model_agreement` and `sig.robust_sign` have no thresholds of their own: give them `config`'s, as every notebook does.

Settings, scales and legend entries that two notebooks each wrote out:

| Was | Now |
|---|---|
| `N_TRIALS` (calc_02a, calc_02b) | `config.N_TRIALS` |
| `SN_THRESHOLDS` (calc_03c, plot_05), `SN_LEVELS` (plot_03a) | `config.SN_THRESHOLDS` |
| `SN_THRESHOLD` (plot_05; 2 in plot_03c) | `config.SN_THRESHOLD` |
| `MEAN_TEST` (plot_03c, plot_05) | `config.MEAN_TEST` |
| `CHANGE_EXPERIMENTS` (calc_04) | `config.FORCED_EXPERIMENTS` |
| `THRESHOLD` (calc_04) | `config.ICE_THRESHOLD` |
| `EARTH_RADIUS` (calc_04) | `sea_ice.EARTH_RADIUS` |
| `CHANCE` (plot_03d, plot_06) | `pr.CHANCE` |
| `MAIN_SET`, `METHOD` (fingerprinting_01, fingerprinting_02) | `fp.` |
| `SCORECARD_STATISTICS` (plot_01b, plot_05) | `evp.SCORECARD_STATISTICS` |
| `class_names` of the diagnosis maps (plot_01c, plot_05) | `ev.DIAGNOSES` |
| `PAPER_TITLE_HEIGHT` (plot_05, plot_06) | `styles.PAPER_TITLE_HEIGHT` |
| `concentration_colours`, `concentration_change_colours`, `correlation_colours` (plot_02, plot_05; `..._scale` in plot_03c) | `colour_scales.CONCENTRATION`, `CONCENTRATION_CHANGE`, `ICE_CORRELATION` |
| `variance_scale` (plot_05, plot_06) | `colour_scales.VARIANCE_CHANGE` |
| `signal_levels`, `signal_cmap` (calc_03c, plot_05) | `colour_scales.FORCED_WARMING` |
| the scales of the linear S/N's two parts (calc_03c, plot_05) | `colour_scales.LINEAR_SIGNAL`, `LINEAR_NOISE` |
| `count_cmap`, `adequate_cmap` (plot_01c, plot_05) | `colour_scales.consistent_scale`, `adequate_scale` |
| `era5_edge_style` (plot_02, plot_05), `era5_years_handle` (calc_03c, plot_05) | `plot_constants.` |
| the legend entries of ERA5's and a model's ice edge (plot_02, plot_05), of hist-nat's variability (calc_02b, plot_03c), of the hatching (plot_01c, plot_05) and of the two S/N (calc_03c, plot_05) | `plot_constants.era5_edge_handle`, `model_edge_handle`, `hist_nat_variability_patch`, `central_75_handle`, `central_75_in_90_handle`, `era5_linear_sn_handle`, `proper_sn_handle`, `linear_sn_handle` |
| `ERA5_YEARS_COLOR` (calc_03c, plot_05) | `constants.ERA5_YEARS_COLOR` |

Deleted, because nothing used them: the older copy of calc_04's regridding, extent and zones in `sea_ice`; `paper_panels` (the panels of the dropped regional analysis); `ev`'s lag-1 autocorrelation and width functions (`qc.quantile_range` is the one width); second copies of `SEASONS`, the year range, the signal sets and the density estimate.
