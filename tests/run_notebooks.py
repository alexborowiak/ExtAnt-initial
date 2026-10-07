"""Run every notebook on the sample data, in order, and report the cells that fail.

    python tests/run_notebooks.py [notebook ...] [--trials N] [--keep]

Sections whose heading says (JASMIN) are skipped (the cluster, the raw-data
conversion, the full-data opening sections, saving to the group workspace),
so the notebooks run on the sample in tests/data instead. Everything the
notebooks save goes to a temporary folder (EXTANT_SCRATCH, EXTANT_DATA and
EXTANT_FIGURES point there), so notebook 04 reads what notebook 03 saved, just
as it does on JASMIN. Cells run in order in one namespace per notebook, as in
Jupyter, from the notebooks folder; a failing cell is reported and the run
carries on. Resampling cells use at most ``--trials`` trials (default 500),
and figures are drawn and closed. The whole run takes a few minutes.

Exits with 1 if any cell failed.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
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

#(t): Resampling settings in the notebooks, and what they become here
TRIAL_CAPS = {"n_trials=10_000": "n_trials={}", "N_TRIALS = 1000": "N_TRIALS = {}", "n_permutations=5000": "n_permutations={}",
              "DEMO_TRIALS = 10_000": "DEMO_TRIALS = {}", "N_PERMUTATIONS = 5000": "N_PERMUTATIONS = {}"}


def stand_in_modules():
    """Empty modules for the JASMIN-only imports that are not installed here."""
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


def prepare(source, trials):
    """Drop IPython magics and shell lines, and cap the number of resampling trials."""
    lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
    source = "\n".join(lines)
    if importlib.util.find_spec("IPython") is None:
        source = source.replace("from IPython.display import display", "display = print")
    for big, small in TRIAL_CAPS.items():
        source = source.replace(big, small.format(trials))
    return source


def run(path, trials):
    """Run one notebook's cells; return the failures."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    failures = []
    for index, headings, source in code_cells(notebook):
        if any(JASMIN in heading for heading in headings):
            continue
        start = time.time()
        try:
            exec(compile(prepare(source, trials), f"<{path.stem} cell {index}>", "exec"), namespace)
            status = "ok"
        except Exception as error:  # noqa: BLE001  (report every failing cell, then carry on)
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
    for variable in ("EXTANT_SCRATCH", "EXTANT_DATA", "EXTANT_FIGURES"):
        os.environ[variable] = str(output / variable.split("_")[1].lower())
    sys.path[:0] = [str(REPO), str(REPO.parent / "extant-functions")]
    stand_in_modules()

    notebooks = [p.resolve() for p in args.notebooks] or sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))
    #(c): The notebooks put '..' and '../../extant-functions' on the path, relative to the notebooks folder
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
