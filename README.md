# ExtAnt-initial

How greenhouse gases, aerosols, ozone and natural forcing have each changed Antarctic near-surface climate in the LESFMIP single-forcing large ensembles. The analysis covers both the mean and the spread of the seasonal distribution, and checks the models against ERA5 to see where they can be trusted.

The data are on JASMIN. A small sample (a 3×3 block of grid cells from two models, plus ERA5) is in `tests/data`, and a sea-ice sample (one member of each model and experiment for 2013–2014, plus ERA5) in `tests/data/sea_ice`, so all of the code can be run and tested off JASMIN.

## Layout

```
notebooks/
    calc_01_process_data.ipynb               raw files -> monthly stores -> seasonal means (once per variable); the SAM index (once)
    calc_02a_width_change.ipynb              the forced response, each method explained, saving every result: the SAM phases, the quantiles, the width (hist-nat bootstrap) and its tails, zonal means and additivity
    calc_02b_mean_change.ipynb               ... the mean (four tests), signal to noise, the additivity of the mean and its zonal means
    calc_02c_probability_ratios.ipynb        ... regional means, probability ratios and the shift-only ratio
    calc_03a_era5_moment_test.ipynb          are the models consistent with ERA5? the climatologies, then each statistic (four moments, trend, width, tails) of every member and ERA5 at every grid point, saved as values
    calc_03b_era5_rank_histograms.ipynb      ... the rank histograms, the paper's grid-point evaluation, and every test in one table, saved
    calc_03c_observed_change.ipynb           is the observed change detectable, and which forcings is it consistent with? saved
    calc_04_sea_ice.ipynb                    the sea ice from CEDA onto the 2.5° grid, extent and edge, against ERA5, with the temperature changes
    plot_01a_era5_distributions_and_plumes.ipynb   the figures of calc_03a to calc_03c, from the saved results: the bias maps, and the distributions and plumes at the point
    plot_01b_era5_point_tests.ipynb          ... where ERA5 falls among the members, the toy ensembles, the moment test, rank histograms and scorecard at the point
    plot_01c_era5_maps.ipynb                 ... the moment test maps with the field test, and the spatial evaluation
    plot_01d_observed_change.ipynb           ... the observed change: consistency with each experiment, record rates, detection maps, scaling factors
    plot_02_sea_ice.ipynb                    the figures of calc_04, from the saved results
    plot_03a_mean_and_signal_to_noise.ipynb  the forced response figures, from calc_02a to calc_02c's results (four for every result, from what goes in to how many models agree): the mean change and S/N
    plot_03b_quantiles_and_width.ipynb       ... the quantile response, and the change in width and which tail
    plot_03c_distribution_additivity_zonal.ipynb   ... where the mean and the width changed together, additivity, zonal means and the sea-ice edge
    plot_03d_probability_ratios.ipynb        ... probability ratios and the shift-only ratio
    plot_04_slides.ipynb                     figures for talks, including how the forced-response figures are made
    plot_05_paper.ipynb                      the paper's main and supplementary figures, in print style (PDF and PNG)
    plot_06_figure_candidates.ipynb          every candidate figure of the planning notes (A-F), the verification against ERA5, and the layouts L1-L5
    fingerprinting_01_calculations.ipynb     are the observed regional seasonal trends explained by the forcings, and is each forcing's size right? every step, saved
    fingerprinting_02_figures.ipynb          ... its figures, from the saved results
    ensemble_size.ipynb                      how many members are needed (after Milinski et al.)
extant/                                       the code the notebooks use
plotting_modules/                             the forcing and season vocabulary (names, colours, orders), and small helpers
tests/                                        pytest suite, the sample data, and a script that runs every notebook
scripts/                                      sea_ice_sample.py: what sea ice CEDA has, and the sea-ice sample (run on JASMIN)
archive/                                      the old draft notebooks
docs/images/                                  reference figures
```

What is in `extant/`:

| Module | |
|---|---|
| `config` | the settings every notebook shares: experiments (`MAIN_EXPERIMENTS`, `FORCED_EXPERIMENTS` in figure order), window length, the years shared with ERA5 (`SATELLITE_YEAR_RANGE`), the significance level (`ALPHA`), example point, thresholds, and the variables with their units (`VARIABLES`) |
| `convert` | opening one model's raw member files together, a month of ERA5, and the inventory of CEDA's sea ice (the file lists, loops, saves and regridding are in notebooks calc_01 and calc_04) |
| `preprocessing` | monthly to seasonal means, and ERA5 onto the model grid |
| `quantiles` | rolling quantiles and LOWESS (numba) |
| `significance` | the hist-nat bootstrap (10 hist-nat members over the same final years) of the width and of the mean, Welch t-test, multi-model agreement |
| `response_change` | mean, width and tail changes, the member-block permutation test, additivity |
| `probability_ratio` | how much more often a season beyond hist-nat's percentiles (warm or cold), the shift-only ratio, and the bootstrap of members |
| `sam` | the SAM index files, the SAM phases (hist-nat's terciles) every result is split by, and a synthetic index for the sample |
| `fingerprinting` | optimal fingerprinting: the vectors, fingerprints, noise covariance (Ledoit-Wolf), OLS and TLS fits, Monte Carlo ranges and the residual test |
| `zonal` | zonal means of the change (mean, low and high extremes, width) and tests of them |
| `sea_ice` | the sea ice onto the 2.5° grid (with each cell's ocean area), extent and the equivalent latitude of the ice edge, the test against ERA5, and the zones where the ice changed |
| `era5_evaluation` | the ERA5 tests (Suarez-Gutierrez et al., 2021) |
| `observed_change` | detection, consistency, records and scaling factors |
| `datatree`, `stats` | DataTree helpers, fast quantiles and KDEs |
| `jasmin` | starting the Dask cluster and sending this package to its workers |
| `plots/` | the figures. Every figure is made the same way: its axes first, then each layer drawn on them, then its labels. Maps take their axes from `antarctica` (`map_figure_size`, `map_axes`, then `colorbar_axis_below` for each colour bar, under the maps it is for, in a figure made with `layout='constrained'`), every other plot gets its size from `charts.chart_figure_size` and its axes from a GridSpec (`fig.add_gridspec(...).subplots()`, in a figure made with `layout='constrained'`); `charts` also has the layers shared by many plots (lines, bands, a marker per model, a point per model and forcing), each a one-axis `plot_...` that styles its panel and a `..._grid` over a DataTree or a DataArray. The other modules hold one-axis panels, `plot_...(data, ax=None)`, one module per analysis module, plus `forced_response` (the colour scales and the notebook, paper and slide styles), `methods` (the step-by-step method figures), `paper_panels` (the paper's panels), `slides` and `fingerprinting` (notebook fingerprinting_02) |

## Running it

Everything is in this repository: the `extant` package and `plotting_modules` (moved here from the separate extant-functions repository on 7 October 2026). Run the notebooks from the `notebooks` folder, in the order of their names: the `calc_` notebooks work everything out and save it, and the `plot_` notebooks draw the figures from what they saved. calc_01 builds the seasonal means; calc_02a to calc_02c, calc_03a to calc_03c and calc_04 use those (calc_02b also reads calc_02a's results, calc_02c both of theirs, calc_03b calc_03a's, and calc_04 calc_02a's and calc_02b's); plot_01a to plot_01d draw calc_03a to calc_03c (plot_01d calc_03c), plot_02 calc_04, and plot_03a to plot_06 calc_02a to calc_02c's results (plot_06 everything). The optimal fingerprinting is a pair of its own (fingerprinting_01 works it out from calc_01's seasonal means, fingerprinting_02 draws it), and ensemble_size works out and draws in one notebook. Each notebook opens the data in either a *Sample data* section or a *Full data* section (on JASMIN); run whichever one you want. The folders are plain `Path`s in each notebook's Settings: everything is read from and written to JASMIN's scratch (`SCRATCH_DIR`, `/work/scratch-pw4/aborow/ExtAnt`), and the results of a run go to `SCRATCH_DIR/results/<variable>/<sample or full>`. After moving a store to the group workspace, change `SCRATCH_DIR` in the notebooks that read it.

**Where the data go.** Dask writes zarr stores from many workers at once, and the group workspaces do not support parallel writes. So everything is written to scratch (`SCRATCH_DIR` in each notebook's Settings; the monthly LESFMIP stores to `MONTHLY_DIR`, notebook calc_01) and read back from there. A store moved by hand to the group workspace keeps the same layout; point `SCRATCH_DIR` there in the notebooks that read it:

```
monthly/era5_<variable>.zarr
seasonal/lesfmip_<variable>.zarr
seasonal/era5_<variable>.zarr
seasonal/lesfmip_sam.zarr                              the SAM index (notebook calc_01, section 6)
results/<variable>/<full|sample>/<name>.zarr
results/siconc/<full|sample>/extent.zarr, climatology.zarr, ...     the sea ice (notebook calc_04)
```

The notebook test runner (`tests/run_notebooks.py`) points `SCRATCH_DIR` and the figures folder at a temporary folder, so a test run writes nothing on JASMIN or into the repository.

**Following a notebook.** Names end in their type: `_tree` (DataTree), `_ds` (Dataset), `_da` (DataArray), `_arr` (numpy array), `_df` (DataFrame). Each step shows what it made, and each method figure has a *quick look* before it: the arrays it is given, shown and drawn plainly with xarray's `.plot`, with all the code in the notebook. Every file is opened and saved in the notebook itself, with its full path written out and shown.

**The SAM index.** Notebook calc_02a's SAM phases use LEADER-EPESC working group 1's monthly SAM index of every member (`SAM_RAW_DIR` in notebook calc_01, the files `archive/draft_current_DAILY.ipynb` opened), not one made here from sea-level pressure. Notebook calc_01, section 6 turns it into seasonal means; it shows the levels in a file so that `SAM_LEVEL` can be set to the right one (the draft took position 3, a later version of its reader position 0). The sample has no SAM index, so it uses a made-up one (`sam.synthetic_index`) that only lets the code run.

**Another variable.** Set `VARIABLE` at the top of each notebook. The variable needs an entry in `config.VARIABLES`, which says how to convert its units; `pr` and `sfcWind` are there but have not been run yet. Two things are not done for them yet. There is no ERA5 recipe: precipitation is a forecast accumulation, and wind speed has to be built from its u and v components. And some figure labels in `extant/plots` still talk about warming and cold or warm tails.

**Sea ice.** The sea ice is not in the LEADER-EPESC folders but in CEDA's copy of the CMIP6 archive (calc_04's `CEDA_DIRS`: DAMIP for the single forcings, CMIP for historical), each model on its own ocean grid, with far fewer members than the temperature. Notebook calc_04 puts it on the 2.5° grid (each cell the area-weighted mean of the points in it, and its ocean area, so the extent counts only the ocean of coastal cells), saves it as monthly stores next to the temperature's, checks it against ERA5's sea-ice cover, and compares the change in sea ice with the mean and width changes. Its `extent` and `climatology` give the sea-ice edge on the maps of notebook plot_03 (sections 1, 4, 5 and 7: hist-nat's dashed, the experiment's solid) and notebook plot_06's D panels, as Bracegirdle et al. (2024, *npj Clim. Atmos. Sci.* 7, 276) mark it. `scripts/sea_ice_sample.py` lists what CEDA has and makes the sample. Without the sea ice, everything else runs and the figures leave the edge out.

## Tests

With an environment that has xarray, dask, numba, scipy, statsmodels, cftime, netCDF4, zarr, matplotlib and cartopy:

```
python -m pytest tests
python tests/run_notebooks.py
```

The second runs every notebook on the sample data, skipping the JASMIN-only sections, and writes its outputs to a temporary folder. It takes a few minutes.

## Notes

- Seasons are DJF/MAM/JJA/SON, and DJF is labelled by the year of its December. Every experiment is cut at the end of 2014, so the record ends with DJF 2013 and "the final 21 years" are 1993–2013 in every experiment.
- The width and mean tests (calc_02a, section 2, and calc_02b, section 1) compare an experiment's final 21 years with 10 random hist-nat members over the same years (a bootstrap from hist-nat alone), so one null distribution serves every experiment of a model.
- Every section of calc_02a to calc_02c that has a *By SAM phase* subsection (the mean, the width and tails, the probability ratios) redoes its result in the seasons of each SAM phase: negative, neutral and positive, cut at hist-nat's terciles (in calc_02a, which saves them for the other two).
- The zonal-mean figures (notebook plot_03c, section 7) follow the layout of Bracegirdle et al. (2024), [doi:10.1038/s41612-024-00822-y](https://doi.org/10.1038/s41612-024-00822-y), Figs 1–3, 6 and 8, but show every model on its own and this project's changes against hist-nat.
- ERA5 evaluation follows Suarez-Gutierrez, Milinski and Maher (2021), *Clim. Dyn.* 57, 2557–2580, [doi:10.1007/s00382-021-05821-w](https://doi.org/10.1007/s00382-021-05821-w).
