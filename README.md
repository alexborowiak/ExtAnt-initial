# ExtAnt-initial

How greenhouse gases, aerosols, ozone and natural forcing have each changed Antarctic near-surface climate in the LESFMIP single-forcing large ensembles. The analysis covers both the mean and the spread of the seasonal distribution, and checks the models against ERA5 to see where they can be trusted.

The data are on JASMIN. A small sample (a 3×3 block of grid cells from two models, plus ERA5) is in `tests/data`, so all of the code can be run and tested off JASMIN.

## Layout

```
notebooks/
    01_process_data.ipynb              raw files -> monthly stores -> seasonal means (once per variable); sea ice (once)
    02_era5_evaluation.ipynb           are the models consistent with ERA5? is the observed change detectable?
    03_forced_response.ipynb           the calculations, each method explained; saves every result
    04_forced_response_figures.ipynb   figures, from the saved results: four for every result, from what goes in to how many models agree
    05_slide_figures.ipynb             figures for talks, including how the forced-response figures are made
    06_paper_figures.ipynb             the paper's main and supplementary figures, in print style (PDF and PNG)
extant/                                the code the notebooks use
plotting_modules/                      the general plotting package: panel grids, polar maps, colour bars
tests/                                 pytest suite, the sample data, and a script that runs every notebook
archive/                               the old draft notebooks
docs/images/                           reference figures
```

What is in `extant/`:

| Module | |
|---|---|
| `config` | experiments, window length, example point, thresholds, and the variables with their units |
| `paths` | every input and output location |
| `storage` | saving to zarr and opening again |
| `convert` | raw LESFMIP files to monthly zarr stores, and the pieces of the ERA5 step (its loop and regridding are in notebook 01) |
| `loading` | opening the monthly stores, the seasonal means and the sample |
| `preprocessing` | monthly to seasonal means, and ERA5 onto the model grid |
| `quantiles` | rolling quantiles and LOWESS (numba) |
| `significance` | the hist-nat bootstrap, Welch t-test, multi-model agreement |
| `response_change` | mean, width and tail changes, the member-block permutation test, additivity |
| `zonal` | zonal means of the change (mean, low and high extremes, width) and tests of them |
| `sea_ice` | sea-ice extent and the equivalent latitude of the ice edge |
| `era5_evaluation` | the ERA5 tests (Suarez-Gutierrez et al., 2021) |
| `observed_change` | detection, consistency, records and scaling factors |
| `datatree`, `stats` | DataTree helpers, fast quantiles and KDEs |
| `jasmin` | starting the Dask cluster and sending this package to its workers |
| `plots/` | the figures, one module per analysis module, plus `forced_response` (the map grids of notebook 04), `methods` (the step-by-step method figures) and `multimodel` (used only by the archived notebook 04) |

## Running it

Everything is in this repository: the `extant` package and `plotting_modules`, the general plotting package (moved here from the separate extant-functions repository on 7 October 2026). Run the notebooks from the `notebooks` folder, in order. 01 builds the seasonal means. 02 and 03 use those, and 04, 05 and 06 use what 03 saved. Each notebook opens the data in either a *Sample data* section or a *Full data (JASMIN)* section; run whichever one you want.

**Where the data go.** Dask writes zarr stores from many workers at once, and the group workspaces do not support parallel writes. So everything is written to scratch (`paths.SCRATCH`) and then moved by hand to the group workspace (`paths.DATA_DIR`). Reading always tries the group workspace first and falls back to scratch, so nothing needs changing after a move. The layout is the same under both:

```
monthly/lesfmip/<variable>/<model>_<variable>_<experiment>_interp_monthly.zarr
monthly/era5_<variable>.zarr
seasonal/lesfmip_<variable>.zarr
seasonal/era5_<variable>.zarr
results/<variable>/<full|sample>/<name>.zarr
results/siconc/full/extent.zarr, climatology.zarr     the sea ice (notebook 01, section 5)
```

Off JASMIN both are `output/` in this repository.

**Following a notebook.** Names end in their type: `_tree` (DataTree), `_ds` (Dataset), `_da` (DataArray), `_arr` (numpy array), `_df` (DataFrame). Each step prints what it made, and each method figure has a *quick look* before it: the arrays it is given, printed and drawn plainly with xarray's `.plot`, with all the code in the notebook. Every file opened or saved through `extant` is logged with its full path, and a store that cannot be found raises an error naming both places it looked and what each folder holds instead.

**Another variable.** Set `VARIABLE` at the top of each notebook. The variable needs an entry in `config.VARIABLES`, which says how to convert its units; `pr` and `sfcWind` are there but have not been run yet. Two things are not done for them yet. There is no ERA5 recipe: precipitation is a forecast accumulation, and wind speed has to be built from its u and v components. And some figure labels in `extant/plots` still talk about warming and cold or warm tails.

**Sea ice.** The figures in notebook 04, section 7 mark the sea-ice edge, as Bracegirdle et al. (2024, *npj Clim. Atmos. Sci.* 7, 276) do, from the sea-ice concentration `siconc`. Notebook 01, section 5 converts it and saves each member's extent and the final-years mean concentration; run it once. The raw files are looked for in `paths.lesfmip_raw('siconc')`, i.e. under the `SImon` folder next to `Amon`. That folder is a guess: set the right one as the `table` of `siconc` in `config.VARIABLES`. Without the sea ice, everything else runs and the figures leave the edge out.

## Tests

With an environment that has xarray, dask, numba, scipy, statsmodels, cftime, netCDF4, zarr, matplotlib and cartopy:

```
python -m pytest tests
python tests/run_notebooks.py
```

The second runs every notebook on the sample data, skipping the JASMIN-only sections, and writes its outputs to a temporary folder. It takes a few minutes.

## Notes

- Seasons are DJF/MAM/JJA/SON, and DJF is labelled by the year of its December. Every experiment is cut at the end of 2014, so the record ends with DJF 2013 and "the final 21 years" are 1993–2013 in every experiment.
- The width test (03, section 2) is a bootstrap from hist-nat alone, with 10 members, so one null distribution serves every experiment of a model.
- The zonal-mean figures (notebook 04, section 7) follow the layout of Bracegirdle et al. (2024), [doi:10.1038/s41612-024-00822-y](https://doi.org/10.1038/s41612-024-00822-y), Figs 1–3, 6 and 8, but show every model on its own and this project's changes against hist-nat.
- ERA5 evaluation follows Suarez-Gutierrez, Milinski and Maher (2021), *Clim. Dyn.* 57, 2557–2580, [doi:10.1007/s00382-021-05821-w](https://doi.org/10.1007/s00382-021-05821-w).
