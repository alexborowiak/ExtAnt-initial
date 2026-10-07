"""Every file location the project reads or writes.

Writing and reading go to different places. Dask writes a zarr store from many
workers at once, and JASMIN's group workspaces do not support parallel writes,
so everything is written under ``SCRATCH`` (a parallel-write scratch area) and
moved by hand to ``DATA_DIR`` afterwards, keeping the same layout::

    mv /work/scratch-pw4/aborow/ExtAnt/seasonal/lesfmip_tas.zarr /gws/.../ExtAnt/seasonal/

``find`` looks in ``DATA_DIR`` first and then in ``SCRATCH``, so a store can be
used as soon as it is written and keeps working once it has been moved.

The monthly LESFMIP stores are the exception: on JASMIN they are all in one
folder, ``LESFMIP_MONTHLY``::

    /work/scratch-pw4/aborow/LESFMIP/monthly_v2/<model>_<variable>_<experiment>_<group>_monthly.zarr

The layout of everything else, the same under both roots::

    monthly/era5_<variable>.zarr
    seasonal/lesfmip_<variable>.zarr       a DataTree, /<model>/<experiment>
    seasonal/era5_<variable>.zarr
    results/<variable>/<run>/<name>.zarr   run: 'full' or 'sample'

Off JASMIN (the sample data on a laptop) both roots are ``output/`` in this
repository. The environment variables EXTANT_SCRATCH, EXTANT_DATA and
EXTANT_FIGURES override them; the notebook test runner uses that to write to a
temporary folder.
"""

import os
from pathlib import Path

from . import config

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Inputs (read only)
# ---------------------------------------------------------------------------

LEADER_EPESC = Path("/gws/ssde/j25b/leader_epesc")

#(t): The LESFMIP model output on the common grid: /<table>/<variable>/<experiment>/<model>/*.nc, monthly, one
#(t): file per member; <table> is the CMIP table of the variable (``config.VARIABLES``), e.g. Amon
LESFMIP_RAW = LEADER_EPESC / "CMIP6_SinglForcHistSimul" / "InterpolatedFlds"
LESFMIP_CATALOG = LEADER_EPESC / "catalogs" / "all-catalog_v1.csv"

#(t): Hourly ERA5 analyses: /<year>/<month>/<day>/*.<code>.nc
ERA5_RAW = Path("/badc/ecmwf-era5/data/oper/an_sfc")

#(t): The 3x3-cell sample used by the tests and the notebooks' sample sections
SAMPLE_DIR = REPO / "tests" / "data"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

ON_JASMIN = LEADER_EPESC.exists()

#(t): The monthly LESFMIP stores of every variable, one per model, experiment and group (None off JASMIN)
LESFMIP_MONTHLY = Path("/work/scratch-pw4/aborow/LESFMIP/monthly_v2") if ON_JASMIN else None

#(t): Where everything is written (parallel writes are fine on scratch-pw)
JASMIN_SCRATCH = Path("/work/scratch-pw4/aborow/ExtAnt")
#(t): Where outputs are moved to by hand, and read from first
JASMIN_DATA = LEADER_EPESC / "aborow" / "ExtAnt"

LOCAL_OUTPUT = REPO / "output"

SCRATCH = Path(os.environ.get("EXTANT_SCRATCH", JASMIN_SCRATCH if ON_JASMIN else LOCAL_OUTPUT))
DATA_DIR = Path(os.environ.get("EXTANT_DATA", JASMIN_DATA if ON_JASMIN else LOCAL_OUTPUT))

#(t): Figures saved for slides and papers (small, so straight into the repository's figures/ folder)
FIGURE_DIR = Path(os.environ.get("EXTANT_FIGURES", REPO / "figures"))


def lesfmip_raw(variable):
    """Folder of one variable's raw LESFMIP files: /<experiment>/<model>/*.nc."""
    return LESFMIP_RAW / config.VARIABLES[variable].table / variable


# ---------------------------------------------------------------------------
# The layout, relative to either root
# ---------------------------------------------------------------------------

def lesfmip_monthly_dir(variable):
    """Folder of the monthly LESFMIP stores, one per model, experiment and group.

    On JASMIN, ``LESFMIP_MONTHLY``. It is an absolute path, so joining it to
    SCRATCH or DATA_DIR gives it back unchanged. Off JASMIN (the tests),
    monthly/lesfmip/<variable> under the output roots.
    """
    if LESFMIP_MONTHLY is not None:
        return LESFMIP_MONTHLY
    return Path("monthly", "lesfmip", variable)


def lesfmip_monthly(model, variable, experiment, group):
    """One model's monthly store for one experiment, members along ``member``."""
    return lesfmip_monthly_dir(variable) / f"{model}_{variable}_{experiment}_{group}_monthly.zarr"


def era5_monthly(variable):
    """Monthly ERA5 means on the LESFMIP grid, in the units of the raw files."""
    return Path("monthly", f"era5_{variable}.zarr")


def era5_years_dir(variable):
    """One store per year while the ERA5 monthly means are being built; deleted once they are combined."""
    return Path("monthly", f"era5_{variable}_years")


def seasonal(source, variable):
    """Seasonal means on (year, season), in analysis units. ``source`` is 'lesfmip' or 'era5'."""
    return Path("seasonal", f"{source}_{variable}.zarr")


def result(name, variable, run):
    """A saved result. ``run`` keeps results from the sample apart from those from the full data."""
    return Path("results", variable, run, f"{name}.zarr")


# ---------------------------------------------------------------------------
# Writing and finding
# ---------------------------------------------------------------------------

def output(relative):
    """Where to write ``relative``: under SCRATCH, with its parent folders created."""
    path = SCRATCH / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _contents(folder, most=10):
    """What ``folder`` holds, for error messages: its first ``most`` names, or that it does not exist."""
    if not folder.is_dir():
        return "folder does not exist"
    names = sorted(path.name for path in folder.iterdir())
    if not names:
        return "folder is empty"
    more = f" and {len(names) - most} more" if len(names) > most else ""
    return "folder holds " + ", ".join(names[:most]) + more


def find(relative):
    """Where to read ``relative`` from: DATA_DIR if it has been moved there, otherwise SCRATCH."""
    candidates = list(dict.fromkeys(root / relative for root in (DATA_DIR, SCRATCH)))
    for path in candidates:
        if path.exists():
            return path
    #(c): Every full path tried, and what is there instead, so a wrong name or a missing step is plain to see
    looked = "\n".join(f"  {path}  ({_contents(path.parent)})" for path in candidates)
    raise FileNotFoundError(f"{relative} is in neither DATA_DIR nor SCRATCH. Looked for:\n{looked}")


def find_all(folder, pattern):
    """Every file matching ``pattern`` in ``folder`` under either root; a name in DATA_DIR hides the same name in SCRATCH."""
    found = {}
    for root in (SCRATCH, DATA_DIR):
        for path in sorted((root / folder).glob(pattern)):
            found[path.name] = path
    return [found[name] for name in sorted(found)]


def figure(name, variable):
    """Where to save a figure, e.g. ``figure('tas_buildup', 'tas')``."""
    path = FIGURE_DIR / variable / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
