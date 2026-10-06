"""The forced change at each latitude: zonal means of the change, and tests of them.

After Bracegirdle et al. (2024, npj Clim. Atmos. Sci. 7, 276, doi:10.1038/s41612-024-00822-y),
Figs 2, 6 and 8, which plot the change in the low extremes, the background
climate, the high extremes and the range between them against latitude, with
the sea-ice edge marked (``sea_ice``). What differs here:

- every model on its own; the paper averages five;
- the changes are this project's: each experiment's final ``WINDOW`` years
  against hist-nat, split as in ``response_change.extreme_changes``;
- the zonal mean is tested, with the same tests as the maps.

A latitude's value is the mean over longitude of the change at each grid
point. So the zonal-mean width is the average of the local widths (how much a
season varies at a point), not the width of the zonal-mean series, in which
anomalies at different longitudes would cancel.

The tests
---------
mean    ``mean_test``: the member-block permutation test (``rc.member_block_test``)
        on zonal-mean data. A mean is linear, so the change in the zonal mean is
        the zonal mean of the change: the test is of exactly the plotted line.
width   ``width_test``: the hist-nat bootstrap (``sig.sample_hist_nat_qrange_changes``).
        Each trial draws the same members and years at every grid point, so the
        trial's changes averaged over longitude are one draw of the zonal-mean
        width change that natural variability gives on its own.

The low and high extremes have no test of their own.
"""

import logging

import xarray as xr

from . import response_change as rc
from . import significance as sig
from .config import WINDOW
from .datatree import skip_empty
from .quantiles import quantile_range
from .stats import nan_quantile

logger = logging.getLogger(__name__)

REFERENCE = rc.REFERENCE


def zonal_mean(obj, lon="lon"):
    """Mean over longitude, skipping missing cells, of a Dataset, DataArray or (every node of) a DataTree."""
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(skip_empty(lambda ds: ds.mean(lon)))
    return obj.mean(lon)


def mean_test(tree, years=WINDOW, reference=REFERENCE, n_permutations=5000, seed=0, variable="tas"):
    """Member-block permutation test of the zonal-mean change in the mean.

    Args:
        tree (xr.DataTree): /<model>/<experiment> on (member, year, season, lat, lon).
        years, reference, n_permutations, seed, variable: As in ``rc.member_block_test``.

    Returns:
        xr.Dataset: ``mean_change`` and ``pvalue`` on (model, experiment, season, lat).
    """
    return rc.member_block_test(zonal_mean(tree), years=years, reference=reference, n_permutations=n_permutations,
                                seed=seed, variable=variable)


def width_test(tree, years=WINDOW, reference=REFERENCE, n_members=sig.N_BOOTSTRAP_MEMBERS, quantiles=(0.05, 0.95),
               n_trials=1000, alpha=0.05, seed=0, variable="tas"):
    """The hist-nat bootstrap test of the zonal-mean change in width.

    The change is ``sig.qrange_significance``'s at every grid point (the final
    ``years`` against hist-nat's full record), averaged over longitude, and so
    is each bootstrap trial. Use the same tree as the width test of the maps
    (``rc.remove_forced_response`` output in notebook 03).

    Args:
        tree (xr.DataTree): /<model>/<experiment> on (member, year, season, lat, lon).
        years, reference, n_members, quantiles, n_trials, seed: As in ``sig.qrange_significance``.
        alpha (float): Two-sided level of ``null_lower`` and ``null_upper``.
        variable (str): Variable name.

    Returns:
        xr.Dataset:
            width_change     (model, experiment, season, lat) the zonal-mean change in Q95 - Q05
            width_pvalue     (model, experiment, season, lat) its two-sided bootstrap p-value
            hist_nat_width   (model, season, lat) the zonal-mean width of hist-nat's full record
            null_lower       (model, season, lat) the alpha/2 quantile of the trials
            null_upper       (model, season, lat) and their 1 - alpha/2 quantile
    """
    by_model = {}
    #(c): A loop on purpose: one bootstrap per model, shared by all of its experiments
    for model, branch in tree.children.items():
        if reference not in branch.children:
            continue
        hist_nat = branch[reference][variable].dropna("member", how="all")
        if hist_nat.sizes["member"] < n_members:
            logger.warning(f"{model}: only {hist_nat.sizes['member']} {reference} members, skipped")
            continue
        hist_nat_qrange = quantile_range(hist_nat, quantiles).compute()
        null = zonal_mean(sig.sample_hist_nat_qrange_changes(
            hist_nat, hist_nat_qrange, n_members, years=years, quantiles=quantiles, n_trials=n_trials, seed=seed))
        bounds = nan_quantile(null, [alpha / 2, 1 - alpha / 2], "trial")

        experiments = {}
        for experiment, node in branch.children.items():
            if experiment == reference or not node.data_vars:
                continue
            final = node[variable].dropna("member", how="all").isel(year=slice(-years, None))
            change = zonal_mean(quantile_range(final, quantiles) - hist_nat_qrange).compute()
            experiments[experiment] = xr.Dataset({
                "width_change": change,
                "width_pvalue": sig.pvalue_two_sided(null, change).where(change.notnull()),
            })
        by_model[model] = xr.concat(list(experiments.values()), dim="experiment").assign_coords(
            experiment=list(experiments)).assign(
            hist_nat_width=zonal_mean(hist_nat_qrange),
            null_lower=bounds.isel(quantile=0, drop=True),
            null_upper=bounds.isel(quantile=1, drop=True),
        )

    out = xr.concat(list(by_model.values()), dim="model", join="outer").assign_coords(model=list(by_model))
    return out.drop_vars("height", errors="ignore").assign_attrs(
        n_members=n_members, years=years, n_trials=n_trials, alpha=alpha, quantiles=list(quantiles))


def zonal_summary(summary, mean_tested=None, width_tested=None):
    """The four parts of ``rc.extreme_changes``, zonally averaged, with the zonal tests' results.

    Args:
        summary (xr.Dataset): ``rc.change_summary`` output, made with ``tails``.
        mean_tested (xr.Dataset | None): ``mean_test`` output.
        width_tested (xr.Dataset | None): ``width_test`` output.

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
