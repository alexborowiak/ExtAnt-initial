"""Run every analysis cell of draft_current_seasonal.ipynb on the sample data, and report the cells that fail.

    python tests/run_notebook_sample.py [notebook] [--trials N]

Opens the data with the notebook's *Sample data* section and skips the
sections that need JASMIN (the Dask cluster, Processing Data, the full-data
ERA5 and LESFMIP opening sections, Subset saving), the Archive, and cells
that save to disk (``to_zarr``). Cells run
in order in one namespace, as in Jupyter; a failing cell is reported and the
run carries on. Resampling cells use at most ``--trials`` trials (default 500),
so the whole run takes a few minutes. Figures are drawn and closed.

Exits with 1 if any cell failed.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
import traceback
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xarray  # noqa: E402, F401  (imported before the stand-ins below, which it checks for)

REPO = Path(__file__).resolve().parents[1]

#(t): Sections skipped: ("# heading", "## heading"), None matching every subsection
SKIP_SECTIONS = {
    ("Dask cluster", None), ("Processing Data", None), ("Subset saving", None), ("Archive", None),
    ("Opening Data", "ERA5"), ("Opening Data", "LESFMIP"),
}


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
    """Yield (index, section, subsection, source) for each code cell, tracking the markdown headings."""
    section = subsection = None
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            for line in source.splitlines():
                if line.startswith("# "):
                    section, subsection = line[2:].strip(), None
                elif line.startswith("## "):
                    subsection = line[3:].strip()
        elif cell["cell_type"] == "code":
            yield index, section, subsection, source


def prepare(source, trials):
    """Drop IPython magics and shell lines, and cap the number of resampling trials."""
    lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
    source = "\n".join(lines)
    if importlib.util.find_spec("IPython") is None:
        source = source.replace("from IPython.display import display", "display = print")
    for big in ("n_trials = 10_000", "n_trials = 1000"):
        source = source.replace(big, f"n_trials = {trials}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", nargs="?", default=REPO / "draft_current_seasonal.ipynb", type=Path)
    parser.add_argument("--trials", type=int, default=500, help="most resampling trials per test")
    args = parser.parse_args()

    sys.path[:0] = [str(REPO), str(REPO.parent / "extant-functions")]
    stand_in_modules()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))

    namespace = {"__name__": "__main__"}
    failures = []
    for index, section, subsection, source in code_cells(notebook):
        if (section, None) in SKIP_SECTIONS or (section, subsection) in SKIP_SECTIONS or ".to_zarr(" in source:
            continue
        start = time.time()
        try:
            exec(compile(prepare(source, args.trials), f"<cell {index}>", "exec"), namespace)
            status = "ok"
        except Exception as error:  # noqa: BLE001  (report every failing cell, then carry on)
            status = f"FAILED {type(error).__name__}: {str(error)[:200]}"
            failures.append((index, source, traceback.format_exc()))
        plt.close("all")
        print(f"[{index:3d}] {(subsection or section or '')[:30]:30s} {time.time() - start:6.1f}s  {status}", flush=True)

    for index, source, trace in failures:
        print(f"\n----- cell {index} -----\n{source[:800]}\n{trace[-3000:]}")
    print(f"\n{len(failures)} failing cell(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
