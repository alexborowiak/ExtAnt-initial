"""Where and how each forcing changes the mean and the variability of temperature.

The seasonal notebook tests the two parts of the forced change separately:
    the mean       ttest_ds (Welch t-test, final years vs hist-nat) and the S/N ratio
    the width      qrange_change_ds (permutation test of the change in Q95 - Q05)
    the tails      quantile_response (the change in Q05, Q50 and Q95)
This module puts them side by side, so the figures can answer the combined
question: where has the mean shifted, where has the distribution widened or
narrowed, where have both happened, and which tail moves faster.

Sections
--------
1. The mean response      final-years ensemble mean minus hist-nat, as the t-test compares
2. One summary dataset    every change and test on (model, experiment, season, lat, lon)
3. Classifying changes    neither / mean only / width only / both
4. Local distributions    samples, quantiles and quantile shifts at one grid point

Everything here combines results computed elsewhere; the only new calculation
is the bootstrap of the quantile shifts at a single point (section 4).
"""

import numpy as np
import xarray as xr

from significance import area_mean, resample_members
from xarray_datatree_utils import reduce_to_dataset

REFERENCE = "hist-nat"

#(t): Default significance for the mean (t-test) and the width (permutation), as used in their sections
ALPHA = 0.05
SN_THRESHOLD = 2

#(t): Quantile levels for the quantile-shift curve
SHIFT_QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ---------------------------------------------------------------------------
# 1. The mean response
# ---------------------------------------------------------------------------

def mean_response(tree, years=11, reference=REFERENCE, variable="tas"):
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


# ---------------------------------------------------------------------------
# 2. One summary dataset
# ---------------------------------------------------------------------------

def change_summary(mean_change, ttest, signal_to_noise, qrange_change, quantile_response=None,
                   alpha=ALPHA, sn_threshold=SN_THRESHOLD):
    """Every change and its test in one Dataset, on the grid points and experiments all inputs share.

    Args:
        mean_change (xr.DataArray): Output of ``mean_response``.
        ttest (xr.Dataset): ``ttest_ds``, with the t-test ``p``.
        signal_to_noise (xr.DataArray): S/N at the year of interest (e.g. ``sn_final``).
        qrange_change (xr.Dataset): ``qrange_change_ds``, with ``qrange_period_difference``
            and ``qrange_permutation_pvalue`` (experiment vs hist-nat over the same final years).
        quantile_response (xr.DataArray | None): ``quantile_response_da`` with Q05, Q50 and Q95.
        alpha (float): Significance level for the t-test and the permutation test.
        sn_threshold (float): |S/N| at which a mean change counts as emerged.

    Returns:
        xr.Dataset:
            mean_change        the ensemble-mean change (°C)
            mean_pvalue        t-test p-value
            signal_to_noise    S/N
            mean_significant   p < alpha: the ensemble mean has shifted
            mean_emerged       |S/N| >= sn_threshold: the shift stands out from year-to-year noise
            mean_robust        both
            width_change       change in Q95 - Q05 (°C)
            width_pvalue       permutation-test p-value
            width_significant  p < alpha
            q05_change, q50_change, q95_change, tail_asymmetry
                               with ``quantile_response``: the quantile changes, and
                               ΔQ05 - ΔQ95 (> 0: the cold tail warms faster)
    """
    parts = [
        mean_change.rename("mean_change"),
        ttest["p"].rename("mean_pvalue"),
        signal_to_noise.rename("signal_to_noise"),
        qrange_change["qrange_period_difference"].rename("width_change"),
        qrange_change["qrange_permutation_pvalue"].rename("width_pvalue"),
    ]
    if quantile_response is not None:
        for level in (0.05, 0.5, 0.95):
            parts.append(quantile_response.sel(quantile=level, drop=True).rename(f"q{round(100 * level):02d}_change"))
    parts = [part.drop_vars(["height", "year", "quantile"], errors="ignore") for part in parts]
    summary = xr.merge(parts, join="inner", compat="override")

    summary["mean_significant"] = summary["mean_pvalue"] < alpha
    summary["mean_emerged"] = np.abs(summary["signal_to_noise"]) >= sn_threshold
    summary["mean_robust"] = summary["mean_significant"] & summary["mean_emerged"]
    summary["width_significant"] = summary["width_pvalue"] < alpha
    if quantile_response is not None:
        summary["tail_asymmetry"] = summary["q05_change"] - summary["q95_change"]
    summary.attrs.update(alpha=alpha, sn_threshold=sn_threshold)
    return summary


# ---------------------------------------------------------------------------
# 3. Classifying changes
# ---------------------------------------------------------------------------

#(t): What changed at each grid point
CHANGE_CLASSES = {0: "neither", 1: "mean only", 2: "width only", 3: "mean and width"}

#(t): Which test decides that the mean has changed
MEAN_TESTS = {
    "significant": "t-test p < alpha",
    "emerged": "|S/N| ≥ threshold",
    "robust": "t-test p < alpha and |S/N| ≥ threshold",
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


def regional_mean(summary, lat_max=-60.0, variables=("mean_change", "width_change", "tail_asymmetry")):
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
# 4. Local distributions at one grid point
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
            lat, lon = score["point"].values[int(score.argmax(skipna=True))]
            return {"lat": float(lat), "lon": float(lon)}
    raise ValueError(f"no data for {model} {experiment} {season}")


def final_years(tree, model, experiments, point, season, years=11, variable="tas"):
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
