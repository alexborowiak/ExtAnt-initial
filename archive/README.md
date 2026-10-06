# Archive

Earlier drafts, kept for reference. Nothing in `notebooks/` or `extant/` depends on them.

- `draft_current_seasonal.ipynb`: the single notebook that `notebooks/01`–`05` were split out of. It also holds exploratory work that was not carried over. That includes the "S/N ratio variation" section, the smoothing drafts (Lanczos), the MIROC6 bootstrapping, the member bootstrap of the rolling Q-range at one point, and the KDE method demonstration.
- `draft_current_DAILY.ipynb`: the daily prototype (day-of-year climatologies, the first KDE and rank-histogram comparisons with ERA5).
- `draft_01.ipynb`–`draft_05.ipynb`: older drafts.
- `xarray_calc.py`, `draft_current_plotting_functions.py`, `timeseries.py`: code that only these drafts used. `timeseries.py` is an old copy of `plotting_modules/timeseries.py`.

They import modules by their old names, from the repository's old flat layout. To run one, check out commit `7956b46` (the last commit before the reorganisation).
