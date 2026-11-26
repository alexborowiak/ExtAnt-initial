"""Where and how each forcing changes the mean and the variability of temperature.

Notebooks calc_02a to calc_02c calculate each part of the forced change step by step, with the general statistics
(``statistics/trend.forced_response``, ``statistics/significance``, ``statistics/quantiles``). Here is what is this
analysis's own: putting the results side by side, each experiment's final ``WINDOW`` years against hist-nat, so the
figures can answer the combined question: where has the mean shifted, where has the distribution widened or narrowed,
which tail has stretched, and where have both happened.

    the mean     the mean response, its tests (``significance.bootstrap_mean``, ``significance.member_block_pvalue``) and S/N
    the width    the change in Q95 - Q05 and its hist-nat bootstrap test (``significance.bootstrap_qrange``)
    the tails    how each tail's length changes, measured from the median

Sections
--------
1. One summary dataset    every change and test on (model, experiment, season, lat, lon), and its
                          split into mean, low and high extremes and width (``extreme_changes``);
                          the width split into its two tails (``width_parts``)
2. Classifying changes    neither / mean only / width only / both
3. Additivity             does historical equal the sum of the single forcings?
4. One grid point         where both changed most, and each experiment's values there
"""

import numpy as np
import xarray as xr

from .statistics.spatial import area_mean


# ---------------------------------------------------------------------------
# 1. One summary dataset
# ---------------------------------------------------------------------------

def change_summary(mean_change, mean_pvalue, signal_to_noise, qrange_change, tails=None, *, alpha, sn_threshold):
    """Every change and its test in one Dataset, on the grid points and experiments all inputs share.

    Args:
        mean_change (xr.DataArray): The change in the mean that ``mean_pvalue`` tests, final years against
            hist-nat's final years: ``mean_bootstrap_ds["mean_change"]`` (as the figures use) or
            ``mean_block_ds["mean_change"]`` (the same numbers).
        mean_pvalue (xr.DataArray): Its p-values: the hist-nat bootstrap's ``mean_bootstrap_ds["pvalue"]``, the
            member-block test's ``mean_block_ds["pvalue"]``, or the pooled Welch t-test's ``ttest_ds["p"]``.
        signal_to_noise (xr.DataArray): S/N against hist-nat at the year of interest (``sn_final``).
        qrange_change (xr.Dataset): ``qrange_change_ds``, with ``qrange_change`` and ``qrange_pvalue``
            (final years against hist-nat's final years, tested by the hist-nat bootstrap).
        tails (xr.Dataset | None): Notebook calc_02a's ``tails_ds``: upper_tail_change, lower_tail_change,
            width_change, tail_asymmetry and q05/q50/q95_change.
        alpha (float): Significance level for the mean test and the width test (``config.ALPHA``).
        sn_threshold (float): |S/N| at which a mean change counts as emerged.

    Returns:
        xr.Dataset:
            mean_change        the ensemble-mean change (°C)
            mean_pvalue        p-value of the mean test
            signal_to_noise    S/N against hist-nat
            mean_significant   p < alpha: the ensemble mean has shifted
            mean_emerged       |S/N| >= sn_threshold: the shift stands out from year-to-year noise
            mean_robust        both
            width_change       change in Q95 - Q05 (°C), as tested by the hist-nat bootstrap
            width_pvalue       its bootstrap p-value
            width_significant  p < alpha
            and with ``tails``: upper_tail_change, lower_tail_change, tail_asymmetry,
            q05_change, q50_change, q95_change
    """
    parts = [
        mean_change.rename("mean_change"),
        mean_pvalue.rename("mean_pvalue"),
        signal_to_noise.rename("signal_to_noise"),
        qrange_change["qrange_change"].rename("width_change"),
        qrange_change["qrange_pvalue"].rename("width_pvalue"),
    ]
    if tails is not None:
        parts += [tails[name] for name in tails.data_vars if name != "width_change"]
    parts = [part.drop_vars(["height", "year", "quantile"], errors="ignore") for part in parts]
    summary = xr.merge(parts, join="inner", compat="override")

    summary["mean_significant"] = summary["mean_pvalue"] < alpha
    summary["mean_emerged"] = np.abs(summary["signal_to_noise"]) >= sn_threshold
    summary["mean_robust"] = summary["mean_significant"] & summary["mean_emerged"]
    summary["width_significant"] = summary["width_pvalue"] < alpha
    summary.attrs.update(alpha=alpha, sn_threshold=sn_threshold)
    return summary


#(t): The four parts of the change in ``extreme_changes``, in the order the figures show them, with their formulas
EXTREMES = {
    "mean_change": "Δ mean",
    "low_extreme_change": "Δ(Q05 − Q50)",
    "high_extreme_change": "Δ(Q95 − Q50)",
    "width_change": "Δ(Q95 − Q05)",
}


def extreme_changes(summary):
    """The change in the mean, the low extremes, the high extremes and the width, as Bracegirdle et al. (2024) split it.

    Their low and high extremes are the 10th and 90th percentiles of the
    residuals about the background climate (the smoothed ensemble mean),
    added back to it, so p10 - background is how far the low extremes sit
    from the middle. Here the extremes are Q05 and Q95 of the internal
    variability (the members minus their ``trend.forced_response``), measured from the median:

        mean_change          the mean response (their background climate)
        low_extreme_change   Δ(Q05 - Q50) > 0: the low extremes have moved up towards the middle
                             (cold extremes warming faster than the median); = -lower_tail_change
        high_extreme_change  Δ(Q95 - Q50) > 0: the high extremes have moved away from the middle;
                             = upper_tail_change
        width_change         Δ(Q95 - Q05) = high_extreme_change - low_extreme_change

    so the two extremes correspond to the paper's p10 and p90 changes minus
    its background change, and the width to its p90 - p10.

    Args:
        summary (xr.Dataset): ``change_summary`` output, made with ``tails``.

    Returns:
        xr.Dataset: The four variables of ``EXTREMES``, on the dims of ``summary``.
    """
    return xr.Dataset({
        "mean_change": summary["mean_change"],
        "low_extreme_change": -summary["lower_tail_change"],
        "high_extreme_change": summary["upper_tail_change"],
        "width_change": summary["width_change"],
    })


#(t): The three parts of the width, as ``width_parts`` names them, with their formulas
WIDTH_PARTS = {"lower tail": "Q50 − Q05", "upper tail": "Q95 − Q50", "width": "Q95 − Q05"}


def width_parts(quantiles, low=0.05, high=0.95):
    """Split the width at the median: the lower tail Q50 - Q05, the upper tail Q95 - Q50, and the width Q95 - Q05.

    The two tails add up to the width, so the change in each says which tail
    did the widening (or narrowing).

    Args:
        quantiles (xr.DataArray): With a ``quantile`` dim holding ``low``, 0.5 and ``high``.
        low, high (float): The quantiles at the two ends of the width.

    Returns:
        xr.DataArray: The three on a new ``part`` dim, named as in ``WIDTH_PARTS``.
    """
    q_low = quantiles.sel(quantile=low, drop=True)
    q_middle = quantiles.sel(quantile=0.5, drop=True)
    q_high = quantiles.sel(quantile=high, drop=True)
    return xr.concat([q_middle - q_low, q_high - q_middle, q_high - q_low], dim=xr.Variable("part", list(WIDTH_PARTS)))


# ---------------------------------------------------------------------------
# 2. Classifying changes
# ---------------------------------------------------------------------------

#(t): What changed at each grid point
CHANGE_CLASSES = {0: "neither", 1: "mean only", 2: "width only", 3: "mean and width"}


def joint_class(mean_sign, width_sign):
    """The mean and the width together, as one of nine classes: ``3 * (mean_sign + 1) + (width_sign + 1)``.

    Each sign is -1 (down, or narrower), 0 (no change) or +1 (up, or wider),
    e.g. from ``significance.significant_sign`` (one model) or
    ``significance.robust_sign`` (the models together). Class 0 is a lower
    mean and a narrower distribution, 4 no change in either, 8 a higher mean
    and a wider distribution: the order of ``plots.colour_scales.JOINT``.
    NaN where either sign is.
    """
    return 3 * (mean_sign + 1) + (width_sign + 1)


def change_class(summary, mean_test="robust"):
    """0 neither, 1 mean only, 2 width only, 3 both (``CHANGE_CLASSES``); NaN where there is no data.

    Args:
        summary (xr.Dataset): Output of ``change_summary``.
        mean_test (str): Which test decides that the mean changed: "significant" (p < alpha), "emerged" (|S/N| at least
            the threshold) or "robust" (both).
    """
    mean_changed = summary[f"mean_{mean_test}"].astype(int)
    width_changed = summary["width_significant"].astype(int)
    return (mean_changed + 2 * width_changed).where(summary["mean_change"].notnull())


def joint_change_counts(summary, mean_test="robust", model_dim="model"):
    """How many models show a mean change together with a significant widening, or a narrowing.

    Returns:
        xr.DataArray: Model counts with a ``change`` dim of
        ["mean and wider", "mean and narrower"], and ``model_dim`` reduced.
    """
    both = summary[f"mean_{mean_test}"] & summary["width_significant"]
    wider = (both & (summary["width_change"] > 0)).sum(model_dim)
    narrower = (both & (summary["width_change"] < 0)).sum(model_dim)
    return xr.concat([wider, narrower], dim=xr.Variable("change", ["mean and wider", "mean and narrower"]))


AREA_SUMMARY_VARIABLES = ("mean_change", "width_change", "upper_tail_change", "lower_tail_change", "tail_asymmetry")


def area_summary(summary, lat_max=-60.0, variables=AREA_SUMMARY_VARIABLES, mean_test="robust"):
    """Cos(lat)-weighted means south of ``lat_max``, plus the fraction of that area in each change class.

    Args:
        summary (xr.Dataset): Output of ``change_summary``.
        lat_max (float): Northern edge of the area averaged.
        variables (Sequence[str]): Variables averaged.
        mean_test (str): Which test decides that the mean changed, as ``change_class``'s, for the classes.

    Returns:
        xr.Dataset: The area means of the ``variables`` present in ``summary``, and
        ``fraction_<class>`` for each of ``CHANGE_CLASSES``.
    """
    region = summary.sel(lat=slice(None, lat_max))
    out = xr.Dataset({name: area_mean(region[name]) for name in variables if name in region})
    classes = change_class(region, mean_test=mean_test)
    for code, name in CHANGE_CLASSES.items():
        out[f"fraction_{name.replace(' ', '_')}"] = area_mean((classes == code).where(classes.notnull()))
    out.attrs.update(summary.attrs, lat_max=lat_max)
    return out


# ---------------------------------------------------------------------------
# 3. Additivity
# ---------------------------------------------------------------------------

#(t): The single-forcing experiments whose responses should add up to historical
ADDITIVE_PARTS = ("hist-nat", "hist-GHG", "hist-aer", "hist-totalO3")


def additivity(changes, parts=ADDITIVE_PARTS, total="historical", min_total=0.25):
    """Does the historical change equal the sum of the single-forcing changes?

    Args:
        changes (xr.DataArray): Each experiment's change from its own baseline, notebook calc_02b's ``own_changes_da``.
        parts (Sequence[str]): Single-forcing experiments to add up.
        total (str): The all-forcing experiment.
        min_total (float): Fractions of the historical change are only given
            where |historical| is at least this (°C); elsewhere they blow up.

    Returns:
        xr.Dataset:
            response  (term, ...) each part, "sum of parts", "historical", and
                      "residual" = historical - sum. The residual holds
                      everything the parts leave out: land use (hist-lu), any
                      other forcing, and non-additive interaction between forcings.
            fraction  (term, ...) each part and the residual as a fraction of
                      historical, NaN where |historical| < min_total
    """
    part_changes = changes.sel(experiment=list(parts))
    #(c): min_count: the sum is NaN unless every part is present, rather than silently partial (or 0)
    parts_sum = part_changes.sum("experiment", min_count=len(parts))
    historical = changes.sel(experiment=total, drop=True)
    residual = historical - parts_sum
    names = [*parts, "sum of parts", "historical", "residual"]
    response = xr.concat([*[part_changes.sel(experiment=p, drop=True) for p in parts], parts_sum, historical, residual],
                         dim=xr.Variable("term", names))
    big_enough = np.abs(historical) >= min_total
    fraction = (response.sel(term=[*parts, "residual"]) / historical).where(big_enough)
    #(c): Reindex to the full term list first, or aligning the two would sort the terms alphabetically
    fraction = fraction.reindex(term=names)
    return xr.Dataset({"response": response, "fraction": fraction}, attrs={"min_total": min_total})


# ---------------------------------------------------------------------------
# 4. One grid point
# ---------------------------------------------------------------------------

def strongest_joint_change(summary, model, experiment, season=None, lat_max=-60.0, mean_test="robust"):
    """The grid point where the mean and the width both changed and the width changed most.

    Falls back to the largest significant width change, then to the largest
    width change, if no point has both. Restricted to south of ``lat_max``.
    With ``season`` None every season is searched, and the point says which.

    Returns:
        dict: ``{"lat": ..., "lon": ...}`` (with ``"season"`` first when it was searched), ready for ``.sel(point)``.
    """
    cell = summary.sel(model=model, experiment=experiment)
    if season is not None:
        cell = cell.sel(season=season)
    cell = cell.sel(lat=slice(None, lat_max))
    size = np.abs(cell["width_change"])
    dims = [dim for dim in ("season", "lat", "lon") if dim in size.dims]
    for candidates in (cell[f"mean_{mean_test}"] & cell["width_significant"], cell["width_significant"], size.notnull()):
        score = size.where(candidates).stack(point=dims)
        if int(score.notnull().sum()):
            best = dict(zip(dims, score["point"].values[int(score.argmax("point", skipna=True))]))
            return {dim: str(value) if dim == "season" else float(value) for dim, value in best.items()}
    raise ValueError(f"no data for {model} {experiment} {season or 'in any season'}")


def final_years(tree, model, experiments, point, season=None, *, years, variable):
    """Each experiment's members over the final ``years`` at one point and season.

    Args:
        point (dict): ``lat`` and ``lon``, and ``season`` unless it is given on its own, as
            ``strongest_joint_change`` returns it.
        years (int): How many final years, e.g. ``config.WINDOW``.
        variable (str): The variable, e.g. ``VARIABLE.name``.

    Returns:
        dict[str, xr.DataArray]: Experiment -> data on (member, year), members
        with no data dropped.
    """
    season = point["season"] if season is None else season
    location = {"lat": point["lat"], "lon": point["lon"]}
    samples = {}
    for experiment in experiments:
        da = tree[model][experiment][variable].sel(location, method="nearest").sel(season=season)
        samples[experiment] = da.isel(year=slice(-years, None)).dropna("member", how="all").load()
    return samples
