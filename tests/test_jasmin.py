"""The package zip the cluster's workers get."""

import subprocess
import sys

from extant import jasmin


def test_package_zip_imports_on_its_own(tmp_path):
    """What the workers get: the package as a zip, importable with nothing else on the path."""
    archive = jasmin.package_zip(tmp_path)
    check = (f"import sys; sys.path.insert(0, {str(archive)!r}); "
             "from extant import quantiles, significance, convert; print(quantiles.__file__)")
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert archive.name in out.stdout
