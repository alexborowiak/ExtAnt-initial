"""Where and how each forcing changes the mean and the variability of temperature.

Notebook 03 calculates each part of the forced change step by step, with the
pieces here; this module also puts the results side by side, each
experiment's final ``WINDOW`` years against hist-nat, so the figures can
answer the combined question: where has the mean shifted, where has the
distribution widened or narrowed, which tail has stretched, and where have
both happened.

    the mean     the mean response, its member-block test (``member_block_pvalue``) and S/N
    the width    the change in Q95 - Q05 and its hist-nat bootstrap test (``significance.bootstrap_qrange``)
    the tails    how each tail's length changes, measured from the median

Sections
--------
1. Internal variability   the forced response, which notebook 03 removes from every member
2. The mean response      the member-block permutation test of the change in the mean
3. One summary dataset    every change and test on (model, experiment, season, lat, lon), and its
                          split into mean, low and high extremes and width (``extreme_changes``)
4. Classifying changes    neither / mean only / width only / both
5. Additivity             does historical equal the sum of the single forcings?
6. Local distributions    samples, quantiles and quantile shifts at one grid point
"""

import numpy as np
import xarray as xr

from .config import WINDOW
from .quantiles import lowess_matrix_xarray
from .significance import area_mean, resample_members
from .datatree import skip_empty

#(t): How forced_response estimates the forced response across members before LOWESS smoothing:
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

@skip_empty
def forced_response(members, window=81, centre=FORCED_CENTRE):
    """The forced response: the ensemble median (or mean) in each year, smoothed with LOWESS.

    Subtracted from every member, it leaves each member's internal
    variability, so a quantile range of what is left is the variability, free
    of any forced warming inside the analysis window. Smoothing matters:
    subtracting the raw ensemble mean would also remove its sampling noise from
    every member, shrinking the spread by sqrt(1 - 1/N), most for the smallest
    ensembles. Forced year-to-year signals too quick for the smoother (a
    volcanic eruption) stay in the members.

    Args:
        members (xr.Dataset | xr.DataArray): With ``member`` and ``year`` dims.
        window (int): LOWESS window in years.
        centre (str): "median" or "mean" across members.

    Returns:
        The same type, without ``member``.
    """
    forced = members.mean("member") if centre == "mean" else members.median("member")
    if forced.chunks:
        forced = forced.chunk({"year": -1})
    return lowess_matrix_xarray(forced, core_dims="year", window=window)


# ---------------------------------------------------------------------------
# 2. The mean response: member-block permutation test
# ---------------------------------------------------------------------------

def permutation_weights(n_permutations, n_exp, n_nat, rng):
    """Random relabellings of ``n_exp + n_nat`` members, each as a row of weights.

    A row has +1/n_exp on the members relabelled as the experiment and
    -1/n_nat on the rest, so ``weights @ member_means`` is each relabelling's
    difference in means. The experiment's members come first in the pool.

    Returns:
        np.ndarray: (n_permutations, n_exp + n_nat).
    """
    order = np.argsort(rng.random((n_permutations, n_exp + n_nat)), axis=1)
    weights = np.full((n_permutations, n_exp + n_nat), -1 / n_nat)
    np.put_along_axis(weights, order[:, :n_exp], 1 / n_exp, axis=1)
    return weights


def _member_block_pvalue(experiment, hist_nat, n_permutations, batch_size, seed):
    """Two-sided permutation p-values of the difference in means, for arrays on (..., member)."""
    n_exp, n_nat = experiment.shape[-1], hist_nat.shape[-1]
    observed = experiment.mean(-1) - hist_nat.mean(-1)
    pooled = np.nan_to_num(np.concatenate([experiment, hist_nat], axis=-1))
    rng = np.random.default_rng(seed)
    at_least = np.zeros(observed.shape)
    at_most = np.zeros(observed.shape)
    for start in range(0, n_permutations, batch_size):
        weights = permutation_weights(min(batch_size, n_permutations - start), n_exp, n_nat, rng)
        differences = pooled @ weights.T
        at_least += (differences >= observed[..., None]).sum(-1)
        at_most += (differences <= observed[..., None]).sum(-1)
    pvalue = np.minimum(1, 2 * np.minimum(at_least + 1, at_most + 1) / (n_permutations + 1))
    return np.where(np.isfinite(observed), pvalue, np.nan)


@skip_empty
def member_block_pvalue(experiment, hist_nat, n_permutations=5000, batch_size=500, seed=0):
    """Permutation p-value of the difference in means, experiment minus hist-nat, permuting whole members.

    Give it each member's mean over the years tested (e.g. the final
    ``WINDOW`` years): permuting member means is permuting whole members, each
    keeping its years together. A pooled t-test treats every member-year as
    independent, but a member's years are serially correlated; members are
    independent realisations, so this test respects that correlation.

    Each permutation relabels the pooled members, as many as the experiment
    has as the experiment and the rest as hist-nat (``permutation_weights``,
    the experiment's members first), and takes the difference in means. The
    p-value is two-sided, with the +1 correction, as in
    ``significance.pvalue_two_sided``. The permutations are drawn in batches of
    ``batch_size``, one matrix product each, from ``np.random.default_rng(seed)``.

    Args:
        experiment, hist_nat (xr.DataArray | xr.Dataset): Member means on (member, ...); the numbers of
            members may differ.
        n_permutations (int): Relabellings.
        batch_size (int): Relabellings per matrix product.
        seed (int): Random seed.

    Returns:
        The same type, without ``member``; NaN where the difference in means is NaN.
    """
    if experiment.chunks:
        experiment = experiment.chunk({"member": -1})
    if hist_nat.chunks:
        hist_nat = hist_nat.chunk({"member": -1})
    return xr.apply_ufunc(
        _member_block_pvalue, experiment, hist_nat,
        input_core_dims=[["member"], ["member"]], exclude_dims={"member"},
        kwargs=dict(n_permutations=n_permutations, batch_size=batch_size, seed=seed),
        dask="parallelized", output_dtypes=[float],
    )


# ---------------------------------------------------------------------------
# 3. One summary dataset
# ---------------------------------------------------------------------------

def change_summary(mean_change, mean_pvalue, signal_to_noise, qrange_change, tails=None,
                   alpha=ALPHA, sn_threshold=SN_THRESHOLD):
    """Every change and its test in one Dataset, on the grid points and experiments all inputs share.

    Args:
        mean_change (xr.DataArray): The mean response, notebook 03's ``mean_response_da``.
        mean_pvalue (xr.DataArray): p-values for the mean change: the member-block test's
            ``mean_block_ds["pvalue"]``, or the pooled Welch t-test's ``ttest_ds["p"]``.
        signal_to_noise (xr.DataArray): S/N against hist-nat at the year of interest (``sn_final``).
        qrange_change (xr.Dataset): ``qrange_change_ds``, with ``qrange_change`` and ``qrange_pvalue``
            (final years against hist-nat's full record, tested by the hist-nat bootstrap).
        tails (xr.Dataset | None): Notebook 03's ``tails_ds``: upper_tail_change, lower_tail_change,
            width_change, tail_asymmetry and q05/q50/q95_change.
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
    variability (the members minus their ``forced_response``), measured from the median:

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


# ---------------------------------------------------------------------------
# 4. Classifying changes
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
# 5. Additivity
# ---------------------------------------------------------------------------

#(t): The single-forcing experiments whose responses should add up to historical
ADDITIVE_PARTS = ("hist-nat", "hist-GHG", "hist-aer", "hist-totalO3")


def additivity(changes, parts=ADDITIVE_PARTS, total="historical", min_total=0.25):
    """Does the historical change equal the sum of the single-forcing changes?

    Args:
        changes (xr.DataArray): Each experiment's change from its own baseline, notebook 03's ``own_changes_da``.
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
# 6. Local distributions at one grid point
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
