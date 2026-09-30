"""Put the repository and the sibling extant-functions checkout (for plotting_modules) on the import path."""

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "extant-functions"))

#(c): All-NaN slices and 0/0 at masked samples are expected in these tests
warnings.filterwarnings("ignore", category=RuntimeWarning)
