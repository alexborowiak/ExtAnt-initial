"""Is the observed (ERA5) change detectable, and which forcings is it consistent with?

The ERA5 evaluation asks whether ERA5 could be a member of the historical
ensemble. Attribution asks more: could ERA5 have happened *without* human
influence (hist-nat)? And how much of it does each forcing explain?

Sections
--------
1. Regions              area-weighted regional means (more signal, less noise than a grid point)
2. Consistency          ERA5's trend (or any other statistic) among each experiment's members
3. Detection maps       ERA5 outside hist-nat's range (detected) and inside historical's (consistent)
4. Record rates         how many record highs and lows ERA5 sets, against chance and the members
5. Scaling factors      regression of ERA5 on the models' forced responses ("fingerprinting"),
                        with the uncertainty from a perfect-model test

Everything reuses era5_evaluation: ``ev.align`` for common samples,
``ev.statistic_test`` and ``ev.locate`` for where ERA5 falls among members.
"""

import numpy as np
import xarray as xr

import era5_evaluation as ev
from xarray_datatree_utils import reduce_to_dataset

#(t): The observed period that ERA5 and LESFMIP historical share
YEARS = slice(1979, 2014)


# ---------------------------------------------------------------------------
# 1. Regions
# ---------------------------------------------------------------------------

#(t): Regions as lat/lon selections on the LESFMIP grid (boxes; edit to taste)
REGIONS = {
    "Antarctica (90–65°S)": {"lat": slice(-90, -65)},
    "Antarctic Peninsula": {"lat": slice(-75, -62.5), "lon": slice(-75, -55)},
    "Southern Ocean (65–50°S)": {"lat": slice(-65, -50)},
}


def regional_mean(obj, region):
    """Cos(latitude)-weighted mean over a region of ``REGIONS`` (or a selection dict).

    Works on DataArrays, Datasets and DataTrees alike.
    """
    selection = REGIONS[region] if isinstance(region, str) else region

    def average(ds):
        ds = ds.sel(selection)
        return ds.weighted(np.cos(np.deg2rad(ds["lat"]))).mean(("lat", "lon"))

    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(lambda ds: average(ds) if ds.data_vars else ds)
    return average(obj)


# ---------------------------------------------------------------------------
# 2. Consistency with each experiment
# ---------------------------------------------------------------------------

def experiment_consistency(ensembles, obs, statistics=("trend",), years=YEARS, alpha=ev.ALPHA):
    """Where ERA5 falls among each experiment's members, for one or more statistics.

    ERA5 inside hist-nat's range means the observed change could be natural;
    outside hist-nat but inside historical is the classic detected-and-
    consistent result; inside a single-forcing experiment's range means that
    forcing alone could produce it.

    Args:
        ensembles (dict[str, xr.DataArray]): Experiment -> members on (member, year, season, ...).
        obs (xr.DataArray): ERA5 on (year, season, ...).
        statistics (Sequence[str]): Keys of ``ev.MOMENTS`` or ``ev.STATISTICS``.
        years (slice): The shared period.
        alpha (float): Two-sided significance level.

    Returns:
        dict[str, xr.Dataset]: Experiment -> ``ev.statistic_test`` output.
    """
    return {experiment: ev.statistic_test(*ev.align(ensemble, obs, years=years), statistics=statistics, alpha=alpha)
            for experiment, ensemble in ensembles.items()}


def consistency_table(results, statistic="trend"):
    """Stack ``experiment_consistency`` outputs for many models into a scorecard table.

    Args:
        results (dict[str, dict[str, xr.Dataset]]): Model -> experiment -> ``ev.statistic_test`` output.
        statistic (str): The statistic to tabulate.

    Returns:
        xr.Dataset: ``ev.RESULT_VARIABLES`` (without members) on (model, statistic, season),
        where ``statistic`` holds the experiments, ready for ``evp.scorecard``.
    """
    tables = []
    for model, by_experiment in results.items():
        rows = [result.sel(statistic=[statistic]).drop_dims("member").assign_coords(
                    statistic=[experiment], label=("statistic", [experiment]))
                for experiment, result in by_experiment.items()]
        tables.append(xr.concat(rows, "statistic", coords="minimal", compat="override").expand_dims(model=[model]))
    table = xr.concat(tables, "model", coords="minimal", compat="override", join="outer")
    table.attrs = {"alpha": next(iter(next(iter(results.values())).values())).attrs["alpha"],
                   "statistic": statistic}
    return table


# ---------------------------------------------------------------------------
# 3. Detection maps
# ---------------------------------------------------------------------------

#(t): Detection classes: is ERA5 outside hist-nat's range (detected) and inside historical's (consistent)?
DETECTION_CLASSES = {
    0: "not detected, consistent",      # ERA5 could be natural, and historical agrees
    1: "detected, consistent",          # human influence detected, and historical gets it right
    2: "detected, inconsistent",        # detected, but historical's change is too large or too small
    3: "not detected, inconsistent",    # looks natural, but historical has a different change
}


def detection_class(historical, hist_nat):
    """Classify each point from ERA5's tests against historical and against hist-nat.

    Args:
        historical, hist_nat (xr.Dataset): ``ev.locate`` (or one ``ev.statistic_test`` statistic) outputs.

    Returns:
        xr.DataArray: Codes of ``DETECTION_CLASSES``; NaN where either test has no
        data, or too few members to flag anything (so "not detected" would mean nothing).
    """
    detected = hist_nat["verdict"] != 0
    consistent = historical["verdict"] == 0
    classes = xr.where(detected, xr.where(consistent, 1, 2), xr.where(consistent, 0, 3))
    usable = historical["verdict"].notnull() & hist_nat["verdict"].notnull()
    for test in (historical, hist_nat):
        if "testable" in test:
            usable = usable & (test["testable"] == 1)
    return classes.where(usable)


def detection_maps(tree, obs, statistic="trend", experiment="historical", reference="hist-nat", years=YEARS,
                   alpha=ev.ALPHA):
    """Detection classes on the whole grid for every model that has both experiments.

    Loads one experiment of one model at a time (1979-2014 only).

    Returns:
        xr.Dataset: ``detection_class`` and ERA5's percentile among each
        experiment's members, on (model, season, lat, lon).
    """
    def test(ds):
        ensemble, era5 = ev.align(ds["tas"].sel(year=years).load(), obs, years=years)
        return ev.statistic_test(ensemble, era5, (statistic,), alpha=alpha).sel(statistic=statistic).drop_dims("member")

    pair = tree.filter(lambda node: node.name in (experiment, reference)
                       and {experiment, reference} <= set(node.parent.children))
    tests = reduce_to_dataset(pair, test, coords="minimal", compat="override").drop_vars(
        ["statistic", "label", "units", "treatment"], errors="ignore")
    tested, natural = tests.sel(experiment=experiment, drop=True), tests.sel(experiment=reference, drop=True)
    return xr.Dataset({
        "detection_class": detection_class(tested, natural),
        f"{experiment}_percentile": tested["percentile"],
        f"{reference}_percentile": natural["percentile"],
    })


# ---------------------------------------------------------------------------
# 4. Record rates
# ---------------------------------------------------------------------------

def record_curves(ensembles, obs, years=YEARS, pool_seasons=True):
    """Cumulative record highs and lows through the record, for ERA5 and each experiment's members.

    With no change at all, year k is a record with probability 1/k, so the
    expected count after n years is 1 + 1/2 + ... + 1/n. Warming makes record
    highs more frequent and record lows rarer. Pooling the four seasons adds
    their counts (4x the expected count), which makes the comparison far less
    noisy than one season's ~4 records.

    Args:
        ensembles (dict[str, xr.DataArray]): Experiment -> members on (member, year, season).
        obs (xr.DataArray): ERA5 on (year, season).
        years (slice): The shared period.
        pool_seasons (bool): Sum the counts over seasons.

    Returns:
        xr.Dataset: ``obs`` (kind, year), ``members`` (experiment, member, kind, year),
        and ``expected`` (year), with ``kind`` = ["highs", "lows"].
    """
    def curves(da):
        both = xr.concat([ev.cumulative_records(da, high=True), ev.cumulative_records(da, high=False)],
                         dim=xr.Variable("kind", ["highs", "lows"]))
        return both.sum("season") if pool_seasons else both

    members, observed = {}, None
    for experiment, ensemble in ensembles.items():
        ensemble, aligned_obs = ev.align(ensemble, obs, years=years)
        members[experiment] = curves(ensemble).assign_coords(member=np.arange(ensemble.sizes["member"]))
        observed = curves(aligned_obs) if observed is None else observed
    stacked = xr.concat(list(members.values()), dim=xr.Variable("experiment", list(members)), join="outer")

    #(c): Expected cumulative count for a stationary series, from each season's valid years
    valid = aligned_obs.notnull()
    rank = valid.cumsum("year").where(valid)
    expected = (1 / rank).fillna(0).cumsum("year")
    expected = expected.sum("season") if pool_seasons else expected
    return xr.Dataset({"obs": observed, "members": stacked, "expected": expected})


# ---------------------------------------------------------------------------
# 5. Scaling factors (fingerprinting)
# ---------------------------------------------------------------------------

#(t): Signal sets: name -> (experiment, experiment subtracted or None)
SIGNAL_SETS = {
    "ANT + NAT": {"ANT": ("historical", "hist-nat"), "NAT": ("hist-nat", None)},
    "GHG + AER + O3 + NAT": {"GHG": ("hist-GHG", None), "AER": ("hist-aer", None),
                             "O3": ("hist-totalO3", None), "NAT": ("hist-nat", None)},
}


def _anomalies(da):
    """Each season measured from its own mean over the years, so only the time structure is fitted."""
    return da - da.mean("year")


def _least_squares(y, X):
    """Coefficients of y on the columns of X over the rows where everything is finite."""
    rows = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]


def scaling_factors(ensembles, obs, signals="ANT + NAT", pseudo="historical", years=YEARS, alpha=ev.ALPHA):
    """Regress ERA5 on the models' forced responses: ERA5 ≈ Σ β_k × signal_k + internal variability.

    Method:
        1. Each series becomes anomalies from its own mean in each season.
        2. Each signal's fingerprint is an ensemble-mean anomaly (for ANT,
           historical minus hist-nat).
        3. β for ERA5 by least squares over every year and season.
        4. Uncertainty from a perfect-model test: each ``pseudo`` member in
           turn stands in for ERA5, regressed on fingerprints that leave it
           out. Their β's show how far internal variability alone moves β;
           ERA5's interval is β minus the 5-95% range of those deviations
           from their median. The median itself shows any bias of the method
           (1 if unbiased).

    Reading the result:
        detected     the interval excludes 0: the signal is in the observations
        consistent   the interval includes 1: the model's response has the right size
        attributable_trend   β times the fingerprint's trend: how much of the
                     observed trend the signal explains

    Args:
        ensembles (dict[str, xr.DataArray]): Experiment -> members on (member, year, season),
            e.g. regional means from ``regional_mean``.
        obs (xr.DataArray): ERA5 on (year, season).
        signals (str | dict): A key of ``SIGNAL_SETS``, or {name: (experiment, subtracted or None)}.
        pseudo (str): Experiment whose members act as perfect-model observations.
        years (slice): The shared period.
        alpha (float): Two-sided level; 0.1 gives a 5-95% interval.

    Returns:
        xr.Dataset on ``signal``: beta, lower, upper, detected, consistent,
        perfect_model_beta, pseudo_beta (member, signal), fingerprint_trend,
        attributable_trend, attributable_lower, attributable_upper; and
        scalars observed_trend and n_samples.
    """
    signals = SIGNAL_SETS[signals] if isinstance(signals, str) else signals
    obs = obs.sel(year=years)
    ensembles = {name: ensembles[name].sel(year=years) for name in
                 {e for pair in signals.values() for e in pair if e} | {pseudo}}

    #(c): One shared mask: a sample counts only if ERA5 and every member of every ensemble have it
    valid = obs.notnull()
    for ensemble in ensembles.values():
        valid = valid & ensemble.notnull().all("member")
    obs = _anomalies(obs.where(valid))
    ensembles = {name: _anomalies(ensemble.where(valid)) for name, ensemble in ensembles.items()}

    def fingerprints(leave_out=None):
        means = {name: (ensemble.drop_isel(member=leave_out) if name == pseudo and leave_out is not None
                        else ensemble).mean("member") for name, ensemble in ensembles.items()}
        return {k: means[a] - (means[b] if b else 0) for k, (a, b) in signals.items()}

    def as_vector(da):
        return da.transpose("year", "season").values.ravel()

    names = list(signals)
    prints = fingerprints()
    X = np.column_stack([as_vector(prints[k]) for k in names])
    beta = _least_squares(as_vector(obs), X)

    pseudo_members = ensembles[pseudo]
    pseudo_beta = []
    for m in range(pseudo_members.sizes["member"]):
        loo = fingerprints(leave_out=m)
        X_m = np.column_stack([as_vector(loo[k]) for k in names])
        pseudo_beta.append(_least_squares(as_vector(pseudo_members.isel(member=m)), X_m))
    pseudo_beta = np.array(pseudo_beta)
    centre = np.median(pseudo_beta, axis=0)
    spread_low, spread_high = np.quantile(pseudo_beta - centre, [alpha / 2, 1 - alpha / 2], axis=0)
    lower, upper = beta - spread_high, beta - spread_low

    trends = np.array([float(ev.trend(prints[k], ("year", "season"))) for k in names])
    attributable = np.stack([beta * trends, lower * trends, upper * trends])
    coords = {"signal": names}
    return xr.Dataset({
        "beta": ("signal", beta),
        "lower": ("signal", lower),
        "upper": ("signal", upper),
        "detected": ("signal", (lower > 0) | (upper < 0)),
        "consistent": ("signal", (lower <= 1) & (upper >= 1)),
        "perfect_model_beta": ("signal", centre),
        "pseudo_beta": (("member", "signal"), pseudo_beta),
        "fingerprint_trend": ("signal", trends),
        "attributable_trend": ("signal", beta * trends),
        "attributable_lower": ("signal", attributable.min(axis=0)),
        "attributable_upper": ("signal", attributable.max(axis=0)),
        "observed_trend": float(ev.trend(obs, ("year", "season"))),
        "n_samples": int(valid.sum()),
    }, coords=coords, attrs={"signals": str(signals), "pseudo": pseudo, "alpha": alpha})
