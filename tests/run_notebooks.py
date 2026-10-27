"""Run every notebook on the sample data, in order, and report the cells that fail.

    python tests/run_notebooks.py [notebook ...] [--trials N] [--keep]

The sections that need JASMIN (the cluster, the raw-data conversion, the
full-data opening sections) are skipped: those named in ``JASMIN_SECTIONS``, by
notebook, and any whose heading still says (JASMIN). So the notebooks run on the
sample in tests/data instead; notebook 10, which has no sample section, is
skipped whole (``JASMIN_NOTEBOOKS``). IPython magics, shell lines and help lines
(``pr.thresholds?``) are dropped. Everything the notebooks save goes to a
temporary folder: the scratch folder they write out as a plain path is replaced
by a temporary scratch folder, and this repository's figures folder
(``Path('../figures')``) by a temporary figures folder. So notebooks 04a-08b
read what 02a, 02c, 03 and 09a saved, just as they do on JASMIN; 02c runs
straight after 02a, so that 02b can draw its results, and 09a straight after 03.
Cells run in order in one namespace per notebook, as in Jupyter, from the
notebooks folder; a failing cell is reported and the run carries on. Resampling cells use at most ``--trials`` trials (default 500),
and figures are drawn and closed. The whole run takes a few minutes.

Exits with 1 if any cell failed.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xarray  # noqa: E402, F401  (imported before the stand-ins below, which it checks for)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"

#(t): A heading with this in it marks a section that needs the data or the cluster on JASMIN
JASMIN = "(JASMIN)"

#(t): The sections that need JASMIN, by notebook (headings no longer say so): the cluster, the raw data and the
#(t): full-data opening sections, notebook 01's conversion, test sample and SAM index, and the year-to-year correlation
#(t): (which needs the full seasonal temperature) where it is made (09a) and drawn (09b, 04c)
JASMIN_SECTIONS = {
    "01_process_data": {"Dask Cluster", "1. LESFMIP Files to Monthly Stores", "2. ERA5 Hourly Files to Monthly Means",
                        "Full data", "Save", "4. Test Sample", "6. SAM Index"},
    "02a_era5_evaluation_calculations": {"Full data"},
    "02b_era5_evaluation_figures": {"Full data"},
    "02c_observed_change_calculations": {"Full data"},
    "03_forced_response": {"Dask Cluster", "Full data"},
    "09a_sea_ice_calculations": {"Dask cluster", "Daily values to monthly means", "What is on CEDA", "Example processing",
                                 "Processing full data and saving", "Opening Data", "Year-to-year correlation"},
    "09b_sea_ice_figures": {"Full data", "Year-to-year correlation"},
    "04a_mean_and_signal_to_noise": {"Full data"},
    "04b_quantiles_and_width": {"Full data"},
    "04c_distribution_additivity_zonal": {"Full data", "7.6 Sea ice itself", "7.8 Year-to-year correlation"},
    "04d_probability_ratios": {"Full data"},
    "05_slide_figures": {"Full data"},
    "06_paper_figures": {"Full data"},
    "07_figure_candidates": {"Full data", "D2 and D4", "Layouts with sea ice"},
    "08a_optimal_fingerprinting_calculations": {"Full data"},
    "08b_optimal_fingerprinting_figures": {"Full data"},
}

#(t): A section inside one of those that runs on the sample too: 09a's look at what was opened, either way
SAMPLE_TOO = {"What was opened"}

#(t): Notebooks that need JASMIN throughout: 10 opens the full seasonal store from its first cells, with no sample
#(t): section
JASMIN_NOTEBOOKS = {"10_ensemble_size"}

#(t): IPython's help on a name, e.g. ``pr.thresholds?`` (or ``??`` for its source): a line of its own
HELP_LINE = re.compile(r"\s*[\w.]+\?{1,2}\s*")

#(t): The scratch folder on JASMIN as the notebooks write it out; here it is EXTANT_SCRATCH
JASMIN_SCRATCH = "Path('/work/scratch-pw4/aborow/ExtAnt')"
#(t): This repository's figures folder, as notebooks 05-07 write it out (from the notebooks folder); here it is
#(t): EXTANT_FIGURES, so a test run never writes into the repository
REPO_FIGURES = "Path('../figures')"

#(t): Resampling settings in the notebooks, and what they become here
TRIAL_CAPS = {"n_trials=10_000": "n_trials={}", "N_TRIALS = 1000": "N_TRIALS = {}", "n_permutations=5000": "n_permutations={}",
              "DEMO_TRIALS = 10_000": "DEMO_TRIALS = {}", "N_PERMUTATIONS = 5000": "N_PERMUTATIONS = {}",
              "PR_TRIALS = 1000": "PR_TRIALS = {}", "N_MONTE_CARLO = 1000": "N_MONTE_CARLO = {}"}


def stand_in_modules():
    """Empty modules for the JASMIN-only imports that are not installed here, and ipywidgets' sliders (09a)."""
    missing = [name for name in ("nc_time_axis", "zarr", "distributed", "dask_gateway")
               if importlib.util.find_spec(name) is None]
    for name in missing:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__version__ = "not installed"
        sys.modules[name] = module
    if "distributed" in missing:
        #(c): Without a cluster, persisted data is already computed, so there is nothing to wait for
        stub = types.ModuleType("dask.distributed")
        stub.wait = lambda *args, **kwargs: None
        sys.modules["dask.distributed"] = stub
    if importlib.util.find_spec("ipywidgets") is None:
        #(c): A slider is its first value, and ``@interact(i=IntSlider(...))`` draws the figure once, at that value
        widgets = types.ModuleType("ipywidgets")
        widgets.IntSlider = lambda value=0, *args, **kwargs: value

        def interact(**sliders):
            def draw_once(function):
                function(**sliders)
                return function
            return draw_once

        widgets.interact = interact
        sys.modules["ipywidgets"] = widgets


def code_cells(notebook):
    """Yield (index, headings, source) for each code cell, with the markdown headings it sits under."""
    headings = {}
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            for line in source.splitlines():
                level = len(line) - len(line.lstrip("#"))
                if 0 < level <= 6 and line[level:level + 1] == " ":
                    headings = {k: v for k, v in headings.items() if k < level}
                    headings[level] = line[level:].strip()
        elif cell["cell_type"] == "code":
            yield index, [headings[k] for k in sorted(headings)], source


def needs_jasmin(name, headings):
    """Whether a cell needs JASMIN: one of its headings says (JASMIN), or is one of ``JASMIN_SECTIONS[name]``.

    Args:
        name (str): The notebook's name, e.g. '09a_sea_ice_calculations'.
        headings (list[str]): The headings the cell sits under, outermost first.

    Example: under 09a's "Opening Data", a cell in "ERA5" needs JASMIN, one in "What was opened" does not.
    """
    if headings and headings[-1] in SAMPLE_TOO:
        return False
    sections = JASMIN_SECTIONS.get(name, set())
    return any(JASMIN in heading or heading in sections for heading in headings)


def prepare(source, trials):
    """Drop IPython magics, shell lines and help lines, point the JASMIN scratch folder at EXTANT_SCRATCH, and cap the
    number of resampling trials."""
    lines = [line for line in source.splitlines()
             if not line.lstrip().startswith(("%", "!")) and not HELP_LINE.fullmatch(line)]
    source = "\n".join(lines)
    source = source.replace(JASMIN_SCRATCH, f"Path(r'{os.environ['EXTANT_SCRATCH']}')")
    source = source.replace(REPO_FIGURES, f"Path(r'{os.environ['EXTANT_FIGURES']}')")
    if importlib.util.find_spec("IPython") is None:
        source = source.replace("from IPython.display import display", "display = print")
    for big, small in TRIAL_CAPS.items():
        source = source.replace(big, small.format(trials))
    return source


def run(path, trials):
    """Run one notebook's cells; return the failures."""
    if path.stem in JASMIN_NOTEBOOKS:
        print("  skipped: needs the data on JASMIN throughout (no sample section)", flush=True)
        return []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    failures = []
    for index, headings, source in code_cells(notebook):
        if needs_jasmin(path.stem, headings):
            continue
        start = time.time()
        try:
            exec(compile(prepare(source, trials), f"<{path.stem} cell {index}>", "exec"), namespace)
            status = "ok"
        except Exception as error:  # noqa: BLE001  (report every failing cell, then carry on)
            if isinstance(error, ImportError) and "nc-time-axis" in str(error):
                #(c): Plots against cftime dates need nc-time-axis, which JASMIN has and this environment does not
                status = "skipped: plotting cftime dates needs nc-time-axis (on JASMIN)"
            else:
                status = f"FAILED {type(error).__name__}: {str(error)[:200]}"
                failures.append((path.stem, index, source, traceback.format_exc()))
        plt.close("all")
        print(f"  [{index:3d}] {(headings[-1] if headings else '')[:40]:40s} {time.time() - start:6.1f}s  {status}",
              flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks to run (default: all, in order)")
    parser.add_argument("--trials", type=int, default=500, help="most resampling trials per test")
    parser.add_argument("--keep", action="store_true", help="keep the temporary output folder and print it")
    args = parser.parse_args()

    output = Path(tempfile.mkdtemp(prefix="extant_notebooks_"))
    for variable in ("EXTANT_SCRATCH", "EXTANT_FIGURES"):
        os.environ[variable] = str(output / variable.split("_")[1].lower())
    sys.path[:0] = [str(REPO)]
    stand_in_modules()

    if args.notebooks:
        notebooks = [p.resolve() for p in args.notebooks]
    else:
        #(c): In order, but 02c straight after 02a, as 02b draws its results too; and 09a straight after 03: it reads
        #(c): notebook 03's results, and 04a-04c read its sea ice
        notebooks = sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))
        observed_change = NOTEBOOKS / "02c_observed_change_calculations.ipynb"
        notebooks.remove(observed_change)
        notebooks.insert(notebooks.index(NOTEBOOKS / "02a_era5_evaluation_calculations.ipynb") + 1, observed_change)
        sea_ice = NOTEBOOKS / "09a_sea_ice_calculations.ipynb"
        notebooks.remove(sea_ice)
        notebooks.insert(notebooks.index(NOTEBOOKS / "03_forced_response.ipynb") + 1, sea_ice)
    #(c): The notebooks put '..' on the path, relative to the notebooks folder
    os.chdir(NOTEBOOKS)
    failures = []
    for path in notebooks:
        print(f"\n{path.name}", flush=True)
        failures += run(path, args.trials)

    for name, index, source, trace in failures:
        print(f"\n----- {name} cell {index} -----\n{source[:800]}\n{trace[-3000:]}")
    print(f"\n{len(failures)} failing cell(s)" + (f"; outputs in {output}" if args.keep else ""))
    if not args.keep:
        shutil.rmtree(output, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
