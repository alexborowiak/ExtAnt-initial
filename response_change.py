"""Where and how each forcing changes the mean and the variability of temperature.

The seasonal notebook tests the parts of the forced change separately. This
module puts them side by side, each experiment's final ``WINDOW`` years
against hist-nat, so the figures can answer the combined
question: where has the mean shifted, where has the distribution widened or
narrowed, which tail has stretched, and where have both happened.

    the mean     ``member_block_test`` (or ttest_ds) and the S/N section's sn_final
    the width    qrange_change_ds (hist-nat bootstrap test of the change in Q95 - Q05)
    the tails    ``tail_changes``: how each tail's length changes, relative to the median

Sections
--------
1. Internal variability   members minus the smoothed forced response
2. The mean response      final-years change against hist-nat, and a member-block test of it
3. The tails              the change in the length of each tail
4. One summary dataset    every change and test on (model, experiment, season, lat, lon)
5. Classifying changes    neither / mean only / width only / both
6. Additivity             does historical equal the sum of the single forcings?
7. Local distributions    samples, quantiles and quantile shifts at one grid point
"""

import numpy as np
import xarray as xr

from quantile_calc import WINDOW, lowess_matrix_xarray
from significance import area_mean, resample_members
from xarray_datatree_utils import reduce_to_dataset, skip_empty
from xarray_stats import nan_quantile

REFERENCE = "hist-nat"

#(t): How remove_forced_response estimates the forced response across members before LOWESS smoothing:
#(c): the median, as in draft_05's S/N, so what is removed is the shift of the middle (Q50)
FORCED_CENTRE = "median"

#(t): Default significance for the mean and width tests, and the S/N emergence threshold, as in their sections
ALPHA = 0.05
SN_THRESHOLD = 2

#(t): Quantile levels for the quantile-shift curve
SHIFT_QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ---------------------------------------------------------------------------
# 1. Internal variability
# ---------------------------------------------------------------------------

def remove_forced_response(tree, window=81, centre=FORCED_CENTRE):
    """Every member minus its own experiment's smoothed ensemble median (or mean): internal variability only.

    The forced response is the ensemble median (or mean) in each year,
    smoothed with LOWESS over ``window`` years, as in the S/N section.
    Smoothing matters: subtracting the raw ensemble mean would also remove its
    sampling noise from every member, shrinking the spread by sqrt(1 - 1/N),
    most for the smallest ensembles. What is left is each member's internal
    variability, so a quantile range computed from it is the variability,
    free of any forced warming inside the analysis window. Forced year-to-year
    signals too quick for the smoother (a volcanic eruption) stay in.

    Args:
        tree (xr.DataTree): /<model>/<experiment>, with ``member`` and ``year`` dims.
        window (int): LOWESS window in years.
        centre (str): "median" or "mean" across members.

    Returns:
        xr.DataTree: Same layout, internal-variability anomalies.
    """
    def remove(ds):
        forced = ds.mean("member") if centre == "mean" else ds.median("member")
        if forced.chunks:
            forced = forced.chunk({"year": -1})
        return ds - lowess_matrix_xarray(forced, core_dims="year", window=window)

    return tree.map_over_datasets(skip_empty(remove))


# ---------------------------------------------------------------------------
# 2. The mean response
# ---------------------------------------------------------------------------

def mean_response(tree, years=WINDOW, reference=REFERENCE, variable="tas"):
    """Ensemble-mean change over the final ``years``: each experiment minus the reference.

    Both are averaged over members and the same final years, which is exactly
    the difference the Welch t-test in ``ttest_ds`` tests.

    Args:
        tree (xr.DataTree): Laid out as /<model>/<experiment>, with ``member`` and ``year`` dims.
        years (int): Number of final years.
        reference (str): Experiment subtracted, dropped from the result.
        variable (str): Variable to return.

    Returns:
        xr.DataArray: The mean change on (model, experiment, season, lat, lon).
    """
    final = reduce_to_dataset(tree.isel(year=slice(-years, None)), lambda ds: ds.mean(("year", "member")))
    change = final - final.sel(experiment=reference)
    return change[variable].drop_sel(experiment=reference).drop_vars("height", errors="ignore")


def member_block_test(tree, years=WINDOW, reference=REFERENCE, n_permutations=5000, batch_size=500, seed=0,
                      variable="tas"):
    """Permutation test of the mean change, resampling whole members as blocks.

    The Welch t-test in ``ttest_ds`` treats every member-year as independent,
    but a member's years are serially correlated, so its p-values are too
    small. Members, though, are independent realisations. Permuting whole
    members between the experiment and hist-nat keeps each member's years
    together (a block) and so respects that correlation.

    Because every member contributes the same final years, resampling whole
    members is the same as resampling each member's ``years``-mean, so the
    test runs on member means: each batch of permutations is one matrix
    product over the whole grid.

    Args:
        tree (xr.DataTree): /<model>/<experiment>, with ``member`` and ``year`` dims.
        years (int): Number of final years, as in the t-test.
        reference (str): The counterfactual experiment.
        n_permutations (int): Random relabellings of the members.
        batch_size (int): Permutations per matrix product.
        seed (int): Random seed.
        variable (str): Variable name.

    Returns:
        xr.Dataset: ``mean_change`` and two-sided ``pvalue`` (with the +1
        correction, as in ``significance.pvalue_two_sided``) on
        (model, experiment, season, lat, lon), without the reference.
    """
    rng = np.random.default_rng(seed)
    member_means = tree.isel(year=slice(-years, None)).map_over_datasets(
        skip_empty(lambda ds: ds[[variable]].mean("year")))

    results = []
    for model, branch in member_means.children.items():
        if reference not in branch.children:
            continue
        ref = branch[reference][variable].dropna("member", how="all").load()
        for experiment, node in branch.children.items():
            if experiment == reference:
                continue
            exp = node[variable].dropna("member", how="all").load()
            n_exp, n_ref = exp.sizes["member"], ref.sizes["member"]
            pooled = np.concatenate([exp.transpose("member", ...).values, ref.transpose("member", ...).values])
            shape = pooled.shape[1:]
            pooled = pooled.reshape(n_exp + n_ref, -1)
            observed = pooled[:n_exp].mean(0) - pooled[n_exp:].mean(0)

            #(c): A permutation is a weight row: +1/n_exp on the members labelled experiment, -1/n_ref on the rest
            at_least, at_most, done = np.zeros_like(observed), np.zeros_like(observed), 0
            while done < n_permutations:
                n = min(batch_size, n_permutations - done)
                order = np.argsort(rng.random((n, n_exp + n_ref)), axis=1)
                weights = np.full((n, n_exp + n_ref), -1 / n_ref)
                np.put_along_axis(weights, order[:, :n_exp], 1 / n_exp, axis=1)
                differences = weights @ np.nan_to_num(pooled)
                at_least += (differences >= observed).sum(0)
                at_most += (differences <= observed).sum(0)
                done += n
            pvalue = np.minimum(1, 2 * np.minimum(at_least + 1, at_most + 1) / (n_permutations + 1))
            pvalue = np.where(np.isfinite(observed), pvalue, np.nan)

            template = exp.isel(member=0, drop=True).transpose(*exp.transpose("member", ...).dims[1:])
            results.append(xr.Dataset({
                "mean_change": template.copy(data=observed.reshape(shape)),
                "pvalue": template.copy(data=pvalue.reshape(shape)),
            }).drop_vars("height", errors="ignore").expand_dims(model=[model], experiment=[experiment]))
    #(c): drop_conflicts: models carry different variable attributes (comments, histories)
    return xr.combine_by_coords(results, combine_attrs="drop_conflicts").assign_attrs(
        n_permutations=n_permutations, years=years)


# ---------------------------------------------------------------------------
# 3. The tails
# ---------------------------------------------------------------------------

def tail_changes(tree, years=WINDOW, reference=REFERENCE, variable="tas", low=0.05, high=0.95):
    """How each tail's length changes: each experiment's final ``years`` against hist-nat's full record.

    The change in width, Δ(Q95 - Q05), is the same thing as ΔQ95 - ΔQ05: "does
    the warm tail warm faster than the cold tail". So ΔQ05 - ΔQ95 on its own
    adds nothing to the width. What it does not say is *which* tail did the
    widening. Measuring each tail from the median splits it:

        upper_tail_change   Δ(Q95 - Q50): the warm tail stretching (+) or shrinking (-)
        lower_tail_change   Δ(Q50 - Q05): the cold tail
        width_change        their sum, Δ(Q95 - Q05)
        tail_asymmetry      upper minus lower: > 0 when the warm tail stretches
                            more than the cold tail (the distribution skews warm)

    Quantiles pool members and years. The reference is hist-nat's full
    record, as in the width test (``significance.qrange_significance``), so
    with the same tree (e.g. ``remove_forced_response`` output) ``width_change``
    equals its ``qrange_change``.

    Returns:
        xr.Dataset: The four variables above, plus q05/q50/q95_change, on
        (model, experiment, season, lat, lon), without the reference.
    """
    levels = [low, 0.5, high]

    def pooled_quantiles(ds):
        return nan_quantile(ds[variable], levels, ("member", "year")).to_dataset(name=variable)

    final = reduce_to_dataset(tree.isel(year=slice(-years, None)), pooled_quantiles)[variable]
    full = reduce_to_dataset(tree.match(f"*/{reference}"), pooled_quantiles)[variable].sel(experiment=reference, drop=True)
    change = (final - full).drop_sel(experiment=reference)
    q_low, q_mid, q_high = (change.sel(quantile=level, drop=True) for level in levels)
    upper, lower = q_high - q_mid, q_mid - q_low
    out = xr.Dataset({
        "upper_tail_change": upper,
        "lower_tail_change": lower,
        "width_change": upper + lower,
        "tail_asymmetry": upper - lower,
        "q05_change": q_low,
        "q50_change": q_mid,
        "q95_change": q_high,
    })
    return out.drop_vars("height", errors="ignore")


# ---------------------------------------------------------------------------
# 4. One summary dataset
# ---------------------------------------------------------------------------

def change_summary(mean_change, mean_pvalue, signal_to_noise, qrange_change, tails=None,
                   alpha=ALPHA, sn_threshold=SN_THRESHOLD):
    """Every change and its test in one Dataset, on the grid points and experiments all inputs share.

    Args:
        mean_change (xr.DataArray): Output of ``mean_response``.
        mean_pvalue (xr.DataArray): p-values for the mean change: ``member_block_test(...)["pvalue"]``,
            or the pooled Welch t-test's ``ttest_ds["p"]``.
        signal_to_noise (xr.DataArray): S/N against hist-nat at the year of interest (``sn_final``).
        qrange_change (xr.Dataset): ``qrange_change_ds``, with ``qrange_change`` and ``qrange_pvalue``
            (final years against hist-nat's full record, tested by the hist-nat bootstrap).
        tails (xr.Dataset | None): Output of ``tail_changes``.
        alpha (float): Significance level for the mean test and the width test.
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


# ---------------------------------------------------------------------------
# 5. Classifying changes
# ---------------------------------------------------------------------------

#(t): What changed at each grid point
CHANGE_CLASSES = {0: "neither", 1: "mean only", 2: "width only", 3: "mean and width"}

#(t): Which test decides that the mean has changed
MEAN_TESTS = {
    "significant": "mean test p < alpha",
    "emerged": "|S/N| ≥ threshold",
    "robust": "mean test p < alpha and |S/N| ≥ threshold",
}


def change_class(summary, mean_test="robust"):
    """0 neither, 1 mean only, 2 width only, 3 both (``CHANGE_CLASSES``); NaN where there is no data.

    Args:
        summary (xr.Dataset): Output of ``change_summary``.
        mean_test (str): Key of ``MEAN_TESTS``: which test decides that the mean changed.
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


REGIONAL_VARIABLES = ("mean_change", "width_change", "upper_tail_change", "lower_tail_change", "tail_asymmetry")


def regional_mean(summary, lat_max=-60.0, variables=REGIONAL_VARIABLES):
    """Cos(lat)-weighted means south of ``lat_max``, plus the fraction of that area in each change class.

    Returns:
        xr.Dataset: The regional-mean ``variables`` present in ``summary``, and
        ``fraction_<class>`` for each of ``CHANGE_CLASSES`` (robust mean test).
    """
    region = summary.sel(lat=slice(None, lat_max))
    out = xr.Dataset({name: area_mean(region[name]) for name in variables if name in region})
    classes = change_class(region)
    for code, name in CHANGE_CLASSES.items():
        out[f"fraction_{name.replace(' ', '_')}"] = area_mean((classes == code).where(classes.notnull()))
    out.attrs.update(summary.attrs, lat_max=lat_max)
    return out


# ---------------------------------------------------------------------------
# 6. Additivity
# ---------------------------------------------------------------------------

#(t): The single-forcing experiments whose responses should add up to historical
ADDITIVE_PARTS = ("hist-nat", "hist-GHG", "hist-aer", "hist-totalO3")


def own_baseline_change(tree, years=WINDOW, baseline=slice(1850, 1900), variable="tas"):
    """Each experiment's ensemble-mean change: its final ``years`` minus its own ``baseline`` years.

    A single-forcing run contains only its own forcing, so relative to its own
    pre-industrial baseline this is that forcing's response. Measuring every
    run from hist-nat's full-record mean instead would subtract hist-nat's
    average volcanic cooling from each one, so a sum of four single forcings
    would subtract it four times while historical subtracts it once.

    Returns:
        xr.DataArray: Change on (model, experiment, season, lat, lon).
    """
    def change(ds):
        return ds.isel(year=slice(-years, None)).mean(("year", "member")) - ds.sel(year=baseline).mean(("year", "member"))

    return reduce_to_dataset(tree, change)[variable].drop_vars("height", errors="ignore")


def additivity(changes, parts=ADDITIVE_PARTS, total="historical", min_total=0.25):
    """Does the historical change equal the sum of the single-forcing changes?

    Args:
        changes (xr.DataArray): Output of ``own_baseline_change``.
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
# 7. Local distributions at one grid point
# ---------------------------------------------------------------------------

def strongest_joint_change(summary, model, experiment, season, lat_max=-60.0, mean_test="robust"):
    """The grid point where the mean and the width both changed and the width changed most.

    Falls back to the largest significant width change, then to the largest
    width change, if no point has both. Restricted to south of ``lat_max``.

    Returns:
        dict: ``{"lat": ..., "lon": ...}``, ready for ``.sel(**point)``.
    """
    cell = summary.sel(model=model, experiment=experiment, season=season).sel(lat=slice(None, lat_max))
    size = np.abs(cell["width_change"])
    for candidates in (cell[f"mean_{mean_test}"] & cell["width_significant"], cell["width_significant"], size.notnull()):
        score = size.where(candidates).stack(point=("lat", "lon"))
        if int(score.notnull().sum()):
            lat, lon = score["point"].values[int(score.argmax("point", skipna=True))]
            return {"lat": float(lat), "lon": float(lon)}
    raise ValueError(f"no data for {model} {experiment} {season}")


def final_years(tree, model, experiments, point, season, years=WINDOW, variable="tas"):
    """Each experiment's members over the final ``years`` at one point and season.

    Returns:
        dict[str, xr.DataArray]: Experiment -> data on (member, year), members
        with no data dropped.
    """
    samples = {}
    for experiment in experiments:
        da = tree[model][experiment][variable].sel(point, method="nearest").sel(season=season)
        samples[experiment] = da.isel(year=slice(-years, None)).dropna("member", how="all").load()
    return samples


def quantile_shift(experiment, reference, quantiles=SHIFT_QUANTILES, n_boot=1000, seed=0):
    """Change in each quantile, experiment minus reference, pooling members and years, with a bootstrap range.

    The bootstrap resamples whole members (with replacement) in both
    ensembles, as in ``significance.resample_members``, so each member's years
    stay together.

    Args:
        experiment, reference (xr.DataArray): Samples on (member, year), e.g. from ``final_years``.
        quantiles (Sequence[float]): Quantile levels.
        n_boot (int): Bootstrap resamples.
        seed (int): Random seed.

    Returns:
        xr.Dataset:
            experiment_quantiles, reference_quantiles   (quantile)
            shift                                       (quantile) experiment minus reference
            shift_lower, shift_upper                    (quantile) bootstrap 5-95% range
    """
    pool = ("member", "year")
    rng = np.random.default_rng(seed)
    exp_q = experiment.quantile(quantiles, dim=pool)
    ref_q = reference.quantile(quantiles, dim=pool)
    boot = (resample_members(experiment, n_boot, rng).quantile(quantiles, dim=pool)
            - resample_members(reference, n_boot, rng).quantile(quantiles, dim=pool))
    #(c): Move the quantile levels aside so xarray's (fast) quantile over trials can add its own dim
    band = (boot.rename(quantile="level").quantile([0.05, 0.95], dim="trial")
            .rename(quantile="bootstrap", level="quantile"))
    return xr.Dataset({
        "experiment_quantiles": exp_q,
        "reference_quantiles": ref_q,
        "shift": exp_q - ref_q,
        "shift_lower": band.sel(bootstrap=0.05, drop=True),
        "shift_upper": band.sel(bootstrap=0.95, drop=True),
    })
