"""The forced change at each latitude: zonal means of the change, and of its tests.

After Bracegirdle et al. (2024, npj Clim. Atmos. Sci. 7, 276, doi:10.1038/s41612-024-00822-y),
Figs 2, 6 and 8, which plot the change in the low extremes, the background
climate, the high extremes and the range between them against latitude, with
the sea-ice edge marked (``sea_ice``). What differs here:

- every model on its own; the paper averages five;
- the changes are this project's: each experiment's final ``WINDOW`` years
  against hist-nat, split as in ``response_change.extreme_changes``;
- the zonal mean is tested, with the same tests as the maps (notebook 03, section 8).

A latitude's value is the mean over longitude of the change at each grid
point. So the zonal-mean width is the average of the local widths (how much a
season varies at a point), not the width of the zonal-mean series, in which
anomalies at different longitudes would cancel.

The tests
---------
mean    the member-block permutation test (``rc.member_block_pvalue``) of the
        zonal means of the member means. A mean is linear, so the change in the
        zonal mean is the zonal mean of the change: the test is of exactly the
        plotted line.
width   the zonal mean of the hist-nat bootstrap trials (``sig.bootstrap_qrange``).
        Each trial draws the same members and years at every grid point, so the
        trial's changes averaged over longitude are one draw of the zonal-mean
        width change that natural variability gives on its own.

The low and high extremes have no test of their own.
"""

import xarray as xr

from . import response_change as rc
from .datatree import skip_empty


def zonal_mean(obj, lon="lon"):
    """Mean over longitude, skipping missing cells, of a Dataset, DataArray or (every node of) a DataTree."""
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(skip_empty(lambda ds: ds.mean(lon)))
    return obj.mean(lon)


def zonal_summary(summary, mean_tested=None, width_tested=None):
    """The four parts of ``rc.extreme_changes``, zonally averaged, with the zonal tests' results.

    Args:
        summary (xr.Dataset): ``rc.change_summary`` output, made with ``tails``.
        mean_tested (xr.Dataset | None): Notebook 03's ``zonal_mean_ds``, with ``pvalue``.
        width_tested (xr.Dataset | None): Notebook 03's ``zonal_width_ds``, with ``width_pvalue``,
            ``hist_nat_width``, ``null_lower`` and ``null_upper``.

    Returns:
        xr.Dataset: ``rc.EXTREMES`` on (model, experiment, season, lat); with the tests,
        ``mean_pvalue``, ``width_pvalue``, and the width test's ``hist_nat_width``,
        ``null_lower`` and ``null_upper``.
    """
    out = zonal_mean(rc.extreme_changes(summary))
    if mean_tested is not None:
        out["mean_pvalue"] = mean_tested["pvalue"]
    if width_tested is not None:
        for name in ("width_pvalue", "hist_nat_width", "null_lower", "null_upper"):
            out[name] = width_tested[name]
    return out.assign_attrs(summary.attrs)
