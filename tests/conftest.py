"""Put the repository and the sibling extant-functions checkout (for plotting_modules) on the import path."""

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "extant-functions"))

#(c): All-NaN slices and 0/0 at masked samples are expected in these tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

#(c): Import dask before netCDF4 is first used: the other way round, with pip wheels (netCDF4 1.7, Python 3.13),
#(c): netCDF4 segfaults when the sample files are opened after test_convert. The notebooks import dask first anyway
import dask  # noqa: E402, F401


def pytest_configure(config):
    #(c): Zarr 3 warns that consolidated metadata is not in its spec; xarray writes it anyway, for zarr 2 readers
    config.addinivalue_line("filterwarnings", "ignore:Consolidated metadata:UserWarning")
