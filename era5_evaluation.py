"""Local comparison of large ensembles with ERA5, at seasonal resolution.

Every test here asks the same question: *could ERA5 be one more member of this
ensemble?* If a model simulates both the forced response (the signal) and the
internal variability (the noise) correctly, ERA5 is statistically
indistinguishable from any one of its members. So each test computes the same
quantity for ERA5 and for every member, and asks where ERA5 falls among the
members. The members supply the null distribution, so the tests assume nothing
about the shape of the distribution, and they account for the short record
automatically: each member is exactly as long as ERA5 (~35 years).

This is the framework of Suarez-Gutierrez, Milinski and Maher (2021),
"Exploiting large ensembles for a better yet simpler climate model evaluation",
Clim. Dyn. 57, 2557-2580, doi:10.1007/s00382-021-05821-w.

Sections
--------
1.  Preparing ERA5      monthly means to seasonal means, on the model grid
2.  Aligning            the same years, and one shared missing-data mask
3.  Treatments          raw values, anomalies, detrended anomalies
4.  Statistics          mean, trend, std, skewness, kurtosis, lag-1 autocorrelation
5.  Locating ERA5       the percentile and p-value that every test reports
6.  Moment test         each section 4 statistic, ERA5 against every member
7.  Rank histograms     the Suarez-Gutierrez et al. (2021) test
8.  Distributions       kernel density estimates and quantiles
9.  Plume membership    is ERA5 inside the ensemble range?
10. Synthetic data      toy ensembles with a known answer, to demonstrate the tests

Conventions
-----------
Functions reduce over ``sample_dim`` ("year") and/or ``member_dim`` ("member")
and broadcast over everything else, so one call handles a grid point, all four
seasons at once, or a whole map. ``obs`` is ERA5 in practice, but nothing here
is ERA5-specific. Works on numpy- or dask-backed data; at a single point the
data are tiny, so ``.load()`` them first.

The two tests (sections 6 and 7) return datasets with the same layout, built
by ``locate``, so they can be stacked with ``combine_tests`` and drawn by the
same functions in ``era5_evaluation_plots``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from significance import pvalue_two_sided
from xarray_calc import split_time

MEMBER_DIM = "member"
SAMPLE_DIM = "year"
SEASONS = ("DJF", "MAM", "JJA", "SON")

#(t): Two-sided significance level for every test: ERA5 outside the members' 5-95% range
ALPHA = 0.1


# ---------------------------------------------------------------------------
# 1. Preparing ERA5
# ---------------------------------------------------------------------------

def seasonal_means(monthly, dim="time", min_months=3):
    """Monthly means to seasonal (DJF/MAM/JJA/SON) means, on ``year`` and ``season`` dims.

    Follows the LESFMIP pre-processing step for step (``QS-DEC`` bins, then
    ``xarray_calc.split_time``), so DJF is labelled by the year of its December
    and the result lines up with ``lesfmip_season_tree`` label for label.
    Seasons with fewer than ``min_months`` months are dropped rather than
    averaged, e.g. the January-February stub at the start of ERA5.

    Args:
        monthly (xr.DataArray): Monthly means with a datetime ``dim``.
        dim (str): Name of the time dimension.
        min_months (int): Months a season needs in order to be kept.

    Returns:
        xr.DataArray: Seasonal means with ``year`` and ``season`` in place of ``dim``.
    """
    seasonal = monthly.resample({dim: "QS-DEC"}).mean()
    n_months = monthly[dim].resample({dim: "QS-DEC"}).count()
    seasonal = seasonal.where(n_months >= min_months, drop=True)
    return split_time(seasonal, ("year", "season"), dim)


def match_grid(obs, like, tolerance=0.01, lat="lat", lon="lon"):
    """Put ``obs`` on exactly the latitudes and longitudes of ``like``.

    ERA5 was conservatively regridded onto the LESFMIP grid, so the two sets of
    coordinates should agree to floating-point noise. This snaps them together
    so arithmetic between the two aligns, and raises if any point is more than
    ``tolerance`` degrees from its partner, which would mean the grids really
    are different.

    Args:
        obs (xr.DataArray): Data to relabel.
        like (xr.DataArray): Data on the target grid.
        tolerance (float): Largest acceptable coordinate difference, in degrees.

    Returns:
        xr.DataArray: ``obs`` carrying ``like``'s ``lat`` and ``lon`` values.
    """
    for name in (lat, lon):
        target = np.asarray(like[name].values, dtype=float)
        source = obs.indexes[name]
        nearest = source.get_indexer(target, method="nearest")
        gap = np.abs(np.asarray(source[nearest], dtype=float) - target).max()
        if gap > tolerance:
            raise ValueError(f"{name} grids differ by up to {gap:.3g} degrees; regrid ERA5 first")
        obs = obs.isel({name: nearest}).assign_coords({name: like[name].values})
    return obs


# ---------------------------------------------------------------------------
# 2. Aligning
# ---------------------------------------------------------------------------

def align(ensemble, obs, years=None, member_dim=MEMBER_DIM, sample_dim=SAMPLE_DIM):
    """Put an ensemble and ERA5 on the same samples, with one shared missing-data mask.

    Each test compares a statistic of ERA5 with the same statistic of every
    member, which is only fair if they are all computed from exactly the same
    samples. So this:

    1. keeps the years both records cover (within ``years``, if given);
    2. drops members with no data at all in that window;
    3. masks any sample (a year-season) that is missing from ERA5 or from any
       member, in ERA5 and in every member. LESFMIP stops part-way through
       2014, for example, so the 2014 seasons it lacks are masked in ERA5 too.

    Every other dimension (season, lat, lon) must already match exactly; see
    ``match_grid``.

    Args:
        ensemble (xr.DataArray): Model data with ``member_dim`` and ``sample_dim``.
        obs (xr.DataArray): ERA5, with ``sample_dim`` and no ``member_dim``.
        years (slice | None): Optional window, e.g. ``slice(1979, 2014)``.
        member_dim, sample_dim (str): Dimension names.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: ``(ensemble, obs)``, NaN wherever a sample is incomplete.
    """
    ensemble = ensemble.drop_vars("height", errors="ignore")
    obs = obs.drop_vars("height", errors="ignore")
    if years is not None:
        obs = obs.sel({sample_dim: years})

    shared = np.intersect1d(ensemble[sample_dim].values, obs[sample_dim].values)
    if shared.size == 0:
        raise ValueError(f"the ensemble and obs share no values of {sample_dim!r}")
    ensemble = ensemble.sel({sample_dim: shared})
    obs = obs.sel({sample_dim: shared})

    #(c): 'exact' raises rather than silently intersecting mismatched grids or seasons
    ensemble, obs = xr.align(ensemble, obs, join="exact", exclude=[member_dim])
    ensemble = ensemble.dropna(member_dim, how="all")

    valid = obs.notnull() & ensemble.notnull().all(member_dim)
    return ensemble.where(valid), obs.where(valid)


# ---------------------------------------------------------------------------
# 3. Treatments
# ---------------------------------------------------------------------------

#(t): The three ways of treating each series before comparing, and what each one tests
TREATMENTS = {
    "raw": "Raw values",
    "anomaly": "Anomalies",
    "detrended": "Detrended anomalies",
}


def anomalies(da, sample_dim=SAMPLE_DIM, reference=None):
    """Subtract each series' own mean over the ``reference`` samples (all of them if None).

    Every member and ERA5 are each measured from their own climatology, so a
    mean-state bias disappears while the forced trend and the variability
    remain.
    """
    base = da if reference is None else da.sel({sample_dim: reference})
    return da - base.mean(sample_dim)


def linear_fit(da, dim=SAMPLE_DIM, x=None):
    """Least-squares straight line through each series, ignoring missing samples.

    Args:
        da (xr.DataArray): Data to fit.
        dim (str | Sequence[str]): Dimension(s) holding the samples, e.g. "year",
            or ("year", "season") for one line through all four seasons.
        x (xr.DataArray | None): The regressor. Defaults to the "year"
            coordinate (or the coordinate of the first of ``dim``).

    Returns:
        tuple[xr.DataArray, xr.DataArray]: ``(slope, fitted)``, the slope per unit
        of ``x`` with ``dim`` reduced, and the fitted line on ``da``'s shape.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    if x is None:
        x = da[SAMPLE_DIM if SAMPLE_DIM in dims else dims[0]]

    #(c): Mask x wherever y is missing, so both means use the same samples
    x = x.astype(float).broadcast_like(da).where(da.notnull())
    dx = x - x.mean(dims)
    dy = da - da.mean(dims)
    slope = (dx * dy).sum(dims) / (dx**2).sum(dims)
    return slope, da.mean(dims) + slope * dx


def detrend(da, sample_dim=SAMPLE_DIM):
    """Subtract each series' own least-squares line, leaving only year-to-year variability."""
    _, fitted = linear_fit(da, sample_dim)
    return da - fitted


def treat(da, treatment, sample_dim=SAMPLE_DIM, reference=None):
    """Apply one of ``TREATMENTS`` to every series (each member and ERA5 separately).

    raw        unchanged. A mean-state bias dominates any comparison.
    anomaly    minus the series' own mean over ``reference``. The bias is gone;
               the forced trend (signal) and the variability (noise) remain.
    detrended  minus the series' own least-squares line. Only the noise remains.

    Treating ERA5 and the members identically is what keeps every test fair:
    each series loses the same things, with the same sampling noise.
    """
    if treatment == "raw":
        return da
    if treatment == "anomaly":
        return anomalies(da, sample_dim, reference)
    if treatment == "detrended":
        return detrend(da, sample_dim)
    raise ValueError(f"treatment must be one of {list(TREATMENTS)}, got {treatment!r}")


# ---------------------------------------------------------------------------
# 4. Statistics
# ---------------------------------------------------------------------------
# Each takes the samples along ``dim`` and returns one number per series.
# Skewness and kurtosis use the plain moment estimators. They are biased for
# short records, but identically so for ERA5 and the members, so the
# comparison is unaffected.

def mean(da, dim=SAMPLE_DIM):
    """Mean over the samples."""
    return da.mean(dim)


def std(da, dim=SAMPLE_DIM):
    """Standard deviation over the samples (ddof=1)."""
    return da.std(dim, ddof=1)


def _standardised_moment(da, dim, order):
    deviation = da - da.mean(dim)
    variance = (deviation**2).mean(dim)
    return (deviation**order).mean(dim) / variance ** (order / 2)


def skewness(da, dim=SAMPLE_DIM):
    """Skewness: > 0 when the warm tail is longer than the cold tail."""
    return _standardised_moment(da, dim, 3)


def excess_kurtosis(da, dim=SAMPLE_DIM):
    """Excess kurtosis: 0 for a normal distribution, > 0 for heavier tails."""
    return _standardised_moment(da, dim, 4) - 3


def trend(da, dim=SAMPLE_DIM):
    """Least-squares trend, per decade."""
    slope, _ = linear_fit(da, dim)
    return 10 * slope


def lag1_autocorrelation(da, dim=SAMPLE_DIM):
    """Correlation between consecutive years: the year-to-year memory."""
    return xr.corr(da, da.shift({dim: 1}), dim=dim)


def _sorted_quantiles(values, q):
    """Linear-interpolation quantiles along the last axis, ignoring NaN, for every series at once."""
    ordered = np.sort(values, axis=-1)  # NaN sorts to the end
    n = np.sum(~np.isnan(ordered), axis=-1, keepdims=True)
    position = (n - 1) * q
    lower = np.clip(np.floor(position), 0, None).astype(int)
    upper = np.clip(lower + 1, None, np.maximum(n - 1, 0)).astype(int)
    fraction = position - np.floor(position)
    low_values = np.take_along_axis(ordered, lower, axis=-1)
    high_values = np.take_along_axis(ordered, upper, axis=-1)
    return np.where(n > 0, low_values + fraction * (high_values - low_values), np.nan)


def nan_quantile(da, q, dim):
    """Quantiles over ``dim`` ignoring NaN: the same numbers as ``da.quantile(q, dim)``, much faster on maps.

    numpy's nanquantile, which xarray uses for float data, loops over every
    series in Python, which takes minutes on a full grid. Sorting once and
    interpolating between order statistics gives identical results (numpy's
    default 'linear' method) in one vectorised step.

    Args:
        da (xr.DataArray): Data.
        q (float | Sequence[float]): Quantile level(s).
        dim (str): Dimension to reduce.

    Returns:
        xr.DataArray: With a ``quantile`` dim (last) if ``q`` is a sequence, without one if it is a scalar.
    """
    levels = np.atleast_1d(np.asarray(q, dtype=float))
    result = xr.apply_ufunc(
        _sorted_quantiles, da,
        input_core_dims=[[dim]], output_core_dims=[["quantile"]],
        kwargs={"q": levels}, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"quantile": levels.size}},
    ).assign_coords(quantile=levels)
    return result.isel(quantile=0, drop=True) if np.ndim(q) == 0 else result


#(t): The quantiles behind the width and tail statistics, matching the Q95 - Q05 analysis elsewhere
LOW_QUANTILE, HIGH_QUANTILE = 0.05, 0.95


def _quantile(da, q, dim):
    return nan_quantile(da, q, dim)


def quantile_width(da, dim=SAMPLE_DIM):
    """Width of the distribution, Q95 - Q05: the quantity whose change the Q-range analysis tests."""
    return _quantile(da, HIGH_QUANTILE, dim) - _quantile(da, LOW_QUANTILE, dim)


def upper_tail_width(da, dim=SAMPLE_DIM):
    """Length of the warm tail, Q95 - Q50."""
    return _quantile(da, HIGH_QUANTILE, dim) - _quantile(da, 0.5, dim)


def lower_tail_width(da, dim=SAMPLE_DIM):
    """Length of the cold tail, Q50 - Q05."""
    return _quantile(da, 0.5, dim) - _quantile(da, LOW_QUANTILE, dim)


@dataclass(frozen=True)
class Diagnostic:
    """How to compute, label and read one test statistic.

    ``low`` and ``high`` say what it means for the *model* when ERA5 falls
    significantly below or above the members; the plots print them as the
    verdict. ``func`` and ``treatment`` are used by the moment test only.
    """

    label: str
    units: str
    low: str
    high: str
    treatment: str | None = None
    func: Callable | None = None


#(t): Moment-test statistics, each paired with the treatment that isolates what it tests
#(c): Signal: mean and trend. Noise: std, width and the two tails, i.e. the quantities whose forced
#(c): changes the rest of the notebook analyses. Shape: skewness, kurtosis and memory.
MOMENTS = {
    "mean": Diagnostic("Mean", "°C", "model too warm", "model too cold",
                       treatment="raw", func=mean),
    "trend": Diagnostic("Trend", "°C/decade", "model trend too large", "model trend too small",
                        treatment="raw", func=trend),
    "std": Diagnostic("Std. deviation", "°C", "model too variable", "model not variable enough",
                      treatment="detrended", func=std),
    "width": Diagnostic("Width (Q95 − Q05)", "°C", "model distribution too wide", "model distribution too narrow",
                        treatment="detrended", func=quantile_width),
    "lower_tail": Diagnostic("Cold tail (Q50 − Q05)", "°C", "model cold tail too long", "model cold tail too short",
                             treatment="detrended", func=lower_tail_width),
    "upper_tail": Diagnostic("Warm tail (Q95 − Q50)", "°C", "model warm tail too long", "model warm tail too short",
                             treatment="detrended", func=upper_tail_width),
    "skewness": Diagnostic("Skewness", "", "model skewness too high", "model skewness too low",
                           treatment="detrended", func=skewness),
    "kurtosis": Diagnostic("Excess kurtosis", "", "model tails too heavy", "model tails too light",
                           treatment="detrended", func=excess_kurtosis),
    "lag1": Diagnostic("Lag-1 autocorrelation", "", "model too persistent", "model not persistent enough",
                       treatment="detrended", func=lag1_autocorrelation),
}

#(t): The moment statistics grouped by the question they answer
SIGNAL_STATISTICS = ("mean", "trend")
NOISE_STATISTICS = ("std", "width", "lower_tail", "upper_tail")
SHAPE_STATISTICS = ("skewness", "kurtosis", "lag1")

#(t): Rank-histogram summaries (section 7); the treatment is chosen when the test is run
RANKS = {
    "dispersion": Diagnostic("Rank dispersion", "", "spread too large (dome)", "spread too small (U shape)"),
    "outside": Diagnostic("Outside ensemble", "%", "ERA5 too rarely outside", "ERA5 too often outside"),
    "below": Diagnostic("Below all members", "%", "ERA5 too rarely below", "ERA5 too often below"),
    "above": Diagnostic("Above all members", "%", "ERA5 too rarely above", "ERA5 too often above"),
    "mean_rank": Diagnostic("Mean rank", "", "ERA5 sits low: model too warm", "ERA5 sits high: model too cold"),
    "rank_trend": Diagnostic("Rank trend", "/decade", "model warms faster than ERA5", "model warms slower than ERA5"),
}

DIAGNOSTICS = {**MOMENTS, **RANKS}


# ---------------------------------------------------------------------------
# 5. Locating ERA5 among the members
# ---------------------------------------------------------------------------

#(t): The variables every test result carries, so results can be stacked and plotted alike
RESULT_VARIABLES = ("era5", "members", "lower", "median", "upper", "percentile", "pvalue", "verdict")


def locate(members, observed, member_dim=MEMBER_DIM, alpha=ALPHA):
    """Where does ERA5's value of a statistic fall among the members' values?

    If the model were perfect, ERA5 would be exchangeable with the members, so
    its percentile among them would be uniform: every value from 0 to 100
    equally likely, and outside the members' central (1 - alpha) range only
    alpha of the time. A percentile near 0 or 100 (a small p-value) means ERA5
    does not behave like a member for this statistic.

    With N members the smallest two-sided p-value is 2 / (N + 1), so at
    alpha = 0.1 an ensemble needs at least 20 members to show anything.

    Args:
        members (xr.DataArray): The statistic for every member: the null distribution.
        observed (xr.DataArray): The same statistic for ERA5.
        member_dim (str): Member dimension of ``members``.
        alpha (float): Two-sided significance level; 0.1 is the 5-95% range.

    Returns:
        xr.Dataset:
            era5                  ERA5's value
            members               every member's value (kept for plotting)
            lower, median, upper  the members' alpha/2, 50% and 1 - alpha/2 quantiles
            percentile            % of members below ERA5 (ties count half); 50 is ideal
            pvalue                two-sided p-value (significance.pvalue_two_sided)
            verdict               -1 ERA5 significantly below the members, +1 above, 0 consistent
    """
    n = members.notnull().sum(member_dim)
    below = (members < observed).sum(member_dim)
    ties = (members == observed).sum(member_dim)
    percentile = 100 * (below + 0.5 * ties) / n

    bounds = nan_quantile(members, [alpha / 2, 0.5, 1 - alpha / 2], member_dim)
    pvalue = pvalue_two_sided(members, observed, dim=member_dim, method="tails")
    verdict = xr.where(pvalue < alpha, np.sign(percentile - 50), 0)

    result = xr.Dataset({
        "era5": observed,
        "members": members,
        "lower": bounds.isel(quantile=0, drop=True),
        "median": bounds.isel(quantile=1, drop=True),
        "upper": bounds.isel(quantile=2, drop=True),
        "percentile": percentile,
        "pvalue": pvalue,
        "verdict": verdict,
    })
    return result.where(observed.notnull())


def _stack(results, names, registry, **attrs):
    """Concatenate per-statistic ``locate`` results along a labelled ``statistic`` dim."""
    stacked = xr.concat(results, dim="statistic", coords="minimal", compat="override")
    stacked = stacked.assign_coords(
        statistic=list(names),
        label=("statistic", [registry[name].label for name in names]),
        units=("statistic", [registry[name].units for name in names]),
    )
    stacked.attrs.update(attrs)
    return stacked


def combine_tests(*results):
    """Stack test results along ``statistic``, e.g. for a scorecard.

    Only ``RESULT_VARIABLES`` are kept. A statistic that appears in several
    results (e.g. rank dispersion of anomalies and of detrended anomalies) is
    made unique by appending its treatment to its name and label.
    """
    parts = [result[list(RESULT_VARIABLES)] for result in results]
    names = [name for part in parts for name in part["statistic"].values]
    repeated = {name for name in names if names.count(name) > 1}

    relabelled = []
    for part in parts:
        treatments = part["treatment"].values
        relabelled.append(part.assign_coords(
            statistic=[f"{s} ({t})" if s in repeated else s
                       for s, t in zip(part["statistic"].values, treatments)],
            label=("statistic", [f"{lab} ({t})" if s in repeated else lab
                                 for s, lab, t in zip(part["statistic"].values, part["label"].values, treatments)]),
        ))
    return xr.concat(relabelled, dim="statistic", coords="minimal", compat="override")


# ---------------------------------------------------------------------------
# 6. Moment test
# ---------------------------------------------------------------------------

def moment_test(ensemble, obs, statistics=tuple(MOMENTS), sample_dim=SAMPLE_DIM,
                member_dim=MEMBER_DIM, reference=None, alpha=ALPHA):
    """Test ERA5 against the ensemble one statistic at a time.

    For each statistic in ``MOMENTS``: give ERA5 and every member the same
    treatment, compute the statistic over ``sample_dim`` for each, and
    ``locate`` ERA5 among the members. The treatment isolates what is tested:

        statistic   treatment   tests
        mean        raw         the climatology (mean-state bias)
        trend       raw         the forced response over the record (signal)
        std         detrended   the size of year-to-year variability (noise)
        width       detrended   the Q95 - Q05 width of the distribution
        lower_tail  detrended   the length of the cold tail, Q50 - Q05
        upper_tail  detrended   the length of the warm tail, Q95 - Q50
        skewness    detrended   the asymmetry of the noise
        kurtosis    detrended   the tail weight of the noise
        lag1        detrended   the year-to-year memory of the noise

    Args:
        ensemble, obs (xr.DataArray): Output of ``align``.
        statistics (Sequence[str]): Keys of ``MOMENTS``.
        sample_dim, member_dim (str): Dimension names.
        reference (slice | None): Anomaly reference years (only 'anomaly' uses it).
        alpha (float): Two-sided significance level.

    Returns:
        xr.Dataset: ``locate`` output on a ``statistic`` dim, with ``label``,
        ``units`` and ``treatment`` coordinates along it.
    """
    treated = {}
    results = []
    for name in statistics:
        diagnostic = MOMENTS[name]
        if diagnostic.treatment not in treated:
            treated[diagnostic.treatment] = (
                treat(ensemble, diagnostic.treatment, sample_dim, reference),
                treat(obs, diagnostic.treatment, sample_dim, reference),
            )
        ensemble_t, obs_t = treated[diagnostic.treatment]
        results.append(locate(diagnostic.func(ensemble_t, sample_dim), diagnostic.func(obs_t, sample_dim),
                              member_dim, alpha))

    stacked = _stack(results, statistics, MOMENTS, test="moment", alpha=alpha,
                     n_members=ensemble.sizes[member_dim])
    return stacked.assign_coords(treatment=("statistic", [MOMENTS[name].treatment for name in statistics]))


# ---------------------------------------------------------------------------
# 7. Rank histograms (Suarez-Gutierrez et al., 2021)
# ---------------------------------------------------------------------------

def ensemble_rank(ensemble, obs, member_dim=MEMBER_DIM):
    """ERA5's rank within the ensemble at each sample: how many members lie below it (0 to N)."""
    return (ensemble < obs).sum(member_dim).where(obs.notnull())


def _rank_among_others(values):
    """For each value along the last axis, how many of the others lie below it."""
    order = np.argsort(values, axis=-1, kind="stable")
    ranks = np.empty(values.shape, dtype=float)
    positions = np.broadcast_to(np.arange(values.shape[-1], dtype=float), values.shape)
    np.put_along_axis(ranks, order, positions, axis=-1)
    return np.where(np.isnan(values), np.nan, ranks)


def pseudo_observation_ranks(ensemble, member_dim=MEMBER_DIM):
    """Each member's rank among the other N - 1 members, at each sample (0 to N - 1).

    Treating each member in turn as if it were the observations ("model as
    truth") shows what the rank histogram of a *perfect* model looks like,
    for this ensemble size and this record length.
    """
    ranks = xr.apply_ufunc(
        _rank_among_others, ensemble,
        input_core_dims=[[member_dim]], output_core_dims=[[member_dim]],
        dask="parallelized", output_dtypes=[float],
    )
    return ranks.transpose(*ensemble.dims)


def _rank_bins(n_ranks, rank_dim):
    return xr.DataArray(np.arange(n_ranks), dims=rank_dim, coords={rank_dim: np.arange(n_ranks)})


def rank_counts(ranks, n_ranks, dims, rank_dim="rank"):
    """Histogram of integer ranks: how many samples along ``dims`` have each rank 0 to n_ranks - 1.

    Counts one rank at a time rather than broadcasting against every bin at
    once, which for pseudo-observations on a map would need N x N x samples x grid booleans.
    """
    counts = [(ranks == k).sum(dims) for k in range(n_ranks)]
    return xr.concat(counts, dim=rank_dim).assign_coords({rank_dim: np.arange(n_ranks)})


def leave_one_out_counts(obs_rank, n_members, dims, rank_dim="rank"):
    """ERA5's rank histogram against N - 1 members, to match the pseudo-observations exactly.

    A pseudo-observation is ranked against the N - 1 *other* members, so its
    rank runs from 0 to N - 1, whereas ERA5 ranked against all N runs from 0
    to N. To compare like with like, ERA5 is ranked against each of the N
    sub-ensembles that leave one member out, and the N histograms averaged.

    No loop is needed. If r members lie below ERA5, leaving out one of those
    r lowers its rank to r - 1, and leaving out one of the other N - r keeps
    it at r. So each sample adds r / N to bin r - 1 and (N - r) / N to bin r.

    Args:
        obs_rank (xr.DataArray): ``ensemble_rank`` output, 0 to N.
        n_members (int): N.
        dims (str | Sequence[str]): Dimensions pooled into the histogram.

    Returns:
        xr.DataArray: Expected sample count in each rank bin 0 to N - 1.
    """
    bins = _rank_bins(n_members, rank_dim)
    moved_down = (obs_rank / n_members) * (obs_rank - 1 == bins)
    stayed = ((n_members - obs_rank) / n_members) * (obs_rank == bins)
    return (moved_down + stayed).sum(dims)


def rank_statistics(counts, rank_dim="rank"):
    """Summarise rank histograms over ranks 0 to M.

    Returns:
        xr.Dataset:
            below, above  % of samples in the lowest / highest rank, i.e. outside the ensemble
            outside       their sum; a perfect model gives 200 / (M + 1) %
            mean_rank     mean rank scaled to 0-1; 0.5 when ERA5 is centred in the ensemble
            dispersion    variance of the scaled rank relative to a flat histogram:
                          1 flat, > 1 U-shaped (too little spread), < 1 dome-shaped (too much)
    """
    frequency = counts / counts.sum(rank_dim)
    n_top = counts.sizes[rank_dim] - 1
    scaled = counts[rank_dim] / n_top

    mean_rank = (frequency * scaled).sum(rank_dim)
    variance = (frequency * (scaled - mean_rank) ** 2).sum(rank_dim)
    #(c): Variance of a flat histogram, i.e. the discrete uniform on 0, 1/M, ..., 1
    flat_variance = (n_top + 2) / (12 * n_top)

    below = 100 * frequency.isel({rank_dim: 0}, drop=True)
    above = 100 * frequency.isel({rank_dim: -1}, drop=True)
    return xr.Dataset({
        "below": below,
        "above": above,
        "outside": below + above,
        "mean_rank": mean_rank,
        "dispersion": variance / flat_variance,
    })


def rank_histogram_test(ensemble, obs, treatment="anomaly", pool_dims=(SAMPLE_DIM,),
                        sample_dim=SAMPLE_DIM, member_dim=MEMBER_DIM, reference=None,
                        alpha=ALPHA, keep_histograms=True):
    """The rank-histogram test of Suarez-Gutierrez et al. (2021).

    Method:
        1. Give ERA5 and every member the same ``treatment`` (along ``sample_dim``).
        2. At every sample, rank ERA5 among the members.
        3. Count those ranks into ERA5's rank histogram, pooling over ``pool_dims``.
        4. Repeat 2-3 for each member in turn, ranked against the other N - 1
           ("pseudo-observations"). Their N histograms show what a *perfect*
           model's histogram looks like: never exactly flat with ~35 samples,
           but scattered around flat by a known amount.
        5. Summarise every histogram (``rank_statistics``, plus the trend of the
           rank through time) and ``locate`` ERA5's summaries among the
           pseudo-observations' summaries.

    Reading the histogram:
        flat    ERA5 behaves like a member: signal and noise both consistent.
        U       ERA5 too often outside the ensemble: spread too small (too
                little noise), or a wrong trend pushing ERA5 out at both ends.
        dome    ERA5 too rarely near the edges: spread too large (too much noise).
        slope   ERA5 mostly low (model too warm) or mostly high (model too cold).

    Running the test on 'anomaly' (signal + noise) and on 'detrended' (noise
    only) tells the two apart: a U shape in anomalies that disappears after
    detrending points at the forced response, not the variability. The
    ``rank_trend`` statistic tests the signal directly: if the model warms
    faster than ERA5, ERA5's rank drifts down through the record.

    Args:
        ensemble, obs (xr.DataArray): Output of ``align``.
        treatment (str): One of ``TREATMENTS``.
        pool_dims (Sequence[str]): Pooled into one histogram: ("year",) gives one
            per season; ("year", "season") pools all four seasons (4x the samples).
        sample_dim, member_dim (str): Dimension names; treatments act along ``sample_dim``.
        reference (slice | None): Anomaly reference years.
        alpha (float): Two-sided significance level.
        keep_histograms (bool): Also return the histograms and ranks, for plotting.
            Turn off for maps, where they are large.

    Returns:
        xr.Dataset: ``locate`` output on a ``statistic`` dim (keys of ``RANKS``),
        and with ``keep_histograms``:
            obs_frequency     (rank) ERA5's leave-one-out histogram, % of samples
            pseudo_frequency  (member, rank) each pseudo-observation's histogram, %
            obs_rank          ERA5's rank at each sample, scaled to 0-1
            pseudo_rank       (member, ...) each pseudo-observation's scaled rank
    """
    pool_dims = list(pool_dims)
    ensemble = treat(ensemble, treatment, sample_dim, reference)
    obs = treat(obs, treatment, sample_dim, reference)
    n_members = ensemble.sizes[member_dim]

    obs_rank = ensemble_rank(ensemble, obs, member_dim)
    pseudo_rank = pseudo_observation_ranks(ensemble, member_dim)

    #(t): Histograms over the same N rank bins for ERA5 and for the pseudo-observations
    obs_counts = leave_one_out_counts(obs_rank, n_members, pool_dims)
    pseudo_counts = rank_counts(pseudo_rank, n_members, pool_dims)
    obs_summary = rank_statistics(obs_counts)
    pseudo_summary = rank_statistics(pseudo_counts)

    #(t): Rank trend: does ERA5 drift through the ensemble over time?
    #(c): r / N is ERA5's leave-one-out expected scaled rank, matching r / (N - 1) for a pseudo-observation
    obs_scaled = obs_rank / n_members
    pseudo_scaled = pseudo_rank / (n_members - 1)
    obs_summary["rank_trend"] = trend(obs_scaled, pool_dims)
    pseudo_summary["rank_trend"] = trend(pseudo_scaled, pool_dims)

    names = list(RANKS)
    results = [locate(pseudo_summary[name], obs_summary[name], member_dim, alpha) for name in names]
    result = _stack(results, names, RANKS, test="rank histogram", alpha=alpha, n_members=n_members,
                    pool_dims=", ".join(pool_dims))
    result = result.assign_coords(treatment=("statistic", [treatment] * len(names)))

    if keep_histograms:
        result["obs_frequency"] = 100 * obs_counts / obs_counts.sum("rank")
        result["pseudo_frequency"] = 100 * pseudo_counts / pseudo_counts.sum("rank")
        result["obs_rank"] = obs_scaled
        result["pseudo_rank"] = pseudo_scaled
    return result


def running_mean(frequency, window=5, rank_dim="rank"):
    """Centred running mean over the interior ranks, as drawn on Suarez-Gutierrez et al.'s histograms.

    The lowest and highest ranks (ERA5 outside the ensemble) are excluded,
    because they measure something different and are shown separately.
    Ranks without a full window are NaN.
    """
    interior = frequency.isel({rank_dim: slice(1, -1)})
    return interior.rolling({rank_dim: window}, center=True).mean()


def regroup_ranks(frequency, n_bins=10, rank_dim="rank"):
    """Merge a rank histogram's N bins into ``n_bins`` equal-width bins of scaled rank.

    Rank r (of N bins) covers the interval [r / N, (r + 1) / N); its frequency
    is shared among the new bins in proportion to the overlap, so a flat
    histogram stays exactly flat whatever N is. This lets ensembles of
    different sizes be drawn on one axis.

    Returns:
        xr.DataArray: Frequencies on a ``rank_dim`` of bin centres, from 1/(2 n_bins) to 1 - 1/(2 n_bins).
    """
    n_ranks = frequency.sizes[rank_dim]
    rank_edges = np.linspace(0, 1, n_ranks + 1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    overlap = np.clip(
        np.minimum(rank_edges[1:, None], bin_edges[None, 1:])
        - np.maximum(rank_edges[:-1, None], bin_edges[None, :-1]),
        0, None,
    ) * n_ranks
    weights = xr.DataArray(overlap, dims=(rank_dim, "bin"))
    grouped = (frequency * weights).sum(rank_dim)
    centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    return grouped.rename(bin=rank_dim).assign_coords({rank_dim: centres})


# ---------------------------------------------------------------------------
# 8. Distributions
# ---------------------------------------------------------------------------

#(t): Quantile levels for Q-Q plots; ~35 samples cannot support levels further into the tails
QQ_QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)


def kde_bandwidth(ensemble, dim=SAMPLE_DIM, member_dim=MEMBER_DIM):
    """One Gaussian-kernel bandwidth for ERA5 and all members: the members' median Scott's rule.

    Scott's rule (std x n^(-1/5)) is scipy.stats.gaussian_kde's default. With
    the same bandwidth for ERA5 and every member, a difference between curves
    comes from the data rather than the smoothing, and the pooled ensemble
    density is exactly the mean of the member densities.
    """
    return (ensemble.std(dim, ddof=1) * ensemble.count(dim) ** (-1 / 5)).median(member_dim)


def kde(da, x, dim=SAMPLE_DIM, bandwidth=None):
    """Gaussian kernel density estimate over ``dim``, evaluated at ``x``, ignoring NaN.

    Args:
        da (xr.DataArray): Samples.
        x (xr.DataArray): Evaluation points, on their own dim (e.g. "x").
        dim (str | Sequence[str]): Sample dimension(s).
        bandwidth (float | xr.DataArray | None): Kernel standard deviation; may
            vary over the broadcast dims (e.g. one per season). Scott's rule if None.

    Returns:
        xr.DataArray: Density at ``x``, integrating to 1 over ``x``.
    """
    if bandwidth is None:
        bandwidth = da.std(dim, ddof=1) * da.count(dim) ** (-1 / 5)
    z = (x - da) / bandwidth
    kernel = np.exp(-0.5 * z**2) / (bandwidth * np.sqrt(2 * np.pi))
    return kernel.sum(dim) / da.count(dim)


def distribution_summary(ensemble, obs, treatments=("raw", "anomaly"), sample_dim=SAMPLE_DIM,
                         member_dim=MEMBER_DIM, reference=None, quantiles=QQ_QUANTILES, n_points=600):
    """KDEs and quantiles of ERA5 and every member, for each treatment, ready to plot.

    Args:
        ensemble, obs (xr.DataArray): Output of ``align``.
        treatments (Sequence[str]): Keys of ``TREATMENTS``.
        quantiles (Sequence[float]): Levels for Q-Q plots.
        n_points (int): Length of each treatment's evaluation grid, which spans
            every value (all seasons, all members) plus four bandwidths.

    Returns:
        dict[str, xr.Dataset]: Treatment -> dataset with
            obs               ERA5's treated values (for a rug)
            member_kde        (member, x) each member's density
            ensemble_kde      (x) the pooled ensemble density: the mean of member_kde
            obs_kde           (x) ERA5's density
            member_quantiles  (quantile, member), obs_quantiles (quantile)
            bandwidth         the shared kernel width
    """
    summaries = {}
    for treatment in treatments:
        ensemble_t = treat(ensemble, treatment, sample_dim, reference)
        obs_t = treat(obs, treatment, sample_dim, reference)
        bandwidth = kde_bandwidth(ensemble_t, sample_dim, member_dim)

        pad = 4 * float(bandwidth.max())
        low = float(min(ensemble_t.min(), obs_t.min())) - pad
        high = float(max(ensemble_t.max(), obs_t.max())) + pad
        x = xr.DataArray(np.linspace(low, high, n_points), dims="x")
        x = x.assign_coords(x=x.values)

        member_kde = kde(ensemble_t, x, sample_dim, bandwidth)
        summaries[treatment] = xr.Dataset({
            "obs": obs_t,
            "member_kde": member_kde,
            "ensemble_kde": member_kde.mean(member_dim),
            "obs_kde": kde(obs_t, x, sample_dim, bandwidth),
            "member_quantiles": ensemble_t.quantile(quantiles, dim=sample_dim),
            "obs_quantiles": obs_t.quantile(quantiles, dim=sample_dim),
            "bandwidth": bandwidth,
        }, attrs={"treatment": treatment})
    return summaries


# ---------------------------------------------------------------------------
# 9. Plume membership
# ---------------------------------------------------------------------------

#(t): Ensemble quantiles drawn as the plume, widest band first
PLUME_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def ensemble_plume(ensemble, obs, treatment="raw", quantiles=PLUME_QUANTILES, sample_dim=SAMPLE_DIM,
                   member_dim=MEMBER_DIM, reference=None):
    """Ensemble quantiles and range through time, and where ERA5 sits relative to them.

    If ERA5 were a member, it would fall below the lowest member (or above the
    highest) with probability 1 / (N + 1) at each sample, so
    ``expected_outside`` = 200 / (N + 1) % of samples should be outside the
    range by chance alone.

    Returns:
        xr.Dataset:
            quantiles         (quantile, ...) ensemble quantiles at each sample
            minimum, maximum  the ensemble range at each sample
            obs               ERA5 (treated)
            below, above      where ERA5 is outside the range (1.0 or 0.0; NaN where missing)
            n_members         N
            expected_outside  % expected outside by chance
        ``n_members`` and ``expected_outside`` are variables rather than
        attributes so they survive concatenating models of different sizes.
    """
    ensemble = treat(ensemble, treatment, sample_dim, reference)
    obs = treat(obs, treatment, sample_dim, reference)
    n_members = ensemble.sizes[member_dim]
    minimum = ensemble.min(member_dim)
    maximum = ensemble.max(member_dim)
    present = obs.notnull()
    return xr.Dataset({
        "quantiles": ensemble.quantile(list(quantiles), dim=member_dim),
        "minimum": minimum,
        "maximum": maximum,
        "obs": obs,
        "below": (obs < minimum).where(present),
        "above": (obs > maximum).where(present),
        "n_members": n_members,
        "expected_outside": 200 / (n_members + 1),
    }, attrs={"treatment": treatment})


def outside_frequency(plume, dims=SAMPLE_DIM):
    """% of samples with ERA5 below, above and outside the ensemble range, and the number of samples."""
    below = 100 * plume["below"].mean(dims)
    above = 100 * plume["above"].mean(dims)
    return xr.Dataset({
        "below": below,
        "above": above,
        "outside": below + above,
        "expected_outside": plume["expected_outside"],
        "n_samples": plume["below"].count(dims),
    })


# ---------------------------------------------------------------------------
# Running everything at once
# ---------------------------------------------------------------------------

def evaluate(ensemble, obs, pool_dims=(SAMPLE_DIM, "season"), sample_dim=SAMPLE_DIM,
             member_dim=MEMBER_DIM, reference=None, alpha=ALPHA):
    """Run every test on one aligned ensemble and ERA5; everything the figures need.

    Args:
        ensemble, obs (xr.DataArray): Output of ``align``.
        pool_dims (Sequence[str]): Pooled into the headline rank histograms; the
            default pools the four seasons for four times the samples.
        sample_dim, member_dim (str): Dimension names.
        reference (slice | None): Anomaly reference years.
        alpha (float): Two-sided significance level.

    Returns:
        dict:
            moments         moment_test, per season
            ranks           {treatment: rank_histogram_test pooled over ``pool_dims``}
            ranks_seasonal  {treatment: rank_histogram_test per season}, without histograms
            plumes          ensemble_plume for raw values and anomalies, along ``treatment``
            distributions   distribution_summary for raw values, anomalies and detrended
    """
    kwargs = dict(sample_dim=sample_dim, member_dim=member_dim, reference=reference)
    return {
        "moments": moment_test(ensemble, obs, alpha=alpha, **kwargs),
        "ranks": {t: rank_histogram_test(ensemble, obs, t, pool_dims, alpha=alpha, **kwargs)
                  for t in TREATMENTS},
        "ranks_seasonal": {t: rank_histogram_test(ensemble, obs, t, (sample_dim,), alpha=alpha,
                                                  keep_histograms=False, **kwargs)
                           for t in ("anomaly", "detrended")},
        "plumes": xr.concat([ensemble_plume(ensemble, obs, t, **kwargs) for t in ("raw", "anomaly")],
                            dim=xr.Variable("treatment", ["raw", "anomaly"])),
        "distributions": distribution_summary(ensemble, obs, tuple(TREATMENTS), **kwargs),
    }


#(t): Rank statistics worth a scorecard column, per treatment (a detrended series has no rank trend to speak of)
SUMMARY_RANKS = {"anomaly": ("dispersion", "rank_trend"), "detrended": ("dispersion",)}


def summary_table(results, rank_statistics=SUMMARY_RANKS, member_dim=MEMBER_DIM):
    """Stack every model's per-season test results into one (model, statistic, season) dataset.

    Args:
        results (dict[str, dict]): Model -> ``evaluate`` output.
        rank_statistics (dict[str, Sequence[str]]): Treatment -> keys of ``RANKS``
            to include from the per-season rank tests.
        member_dim (str): Member dimension, dropped because its size differs between models.

    Returns:
        xr.Dataset: ``RESULT_VARIABLES`` except ``members``, on ``model``,
        ``statistic`` and ``season``.
    """
    tables = []
    for model, result in results.items():
        ranks = [result["ranks_seasonal"][t].sel(statistic=list(names)) for t, names in rank_statistics.items()]
        table = combine_tests(result["moments"], *ranks).drop_dims(member_dim)
        tables.append(table.expand_dims(model=[model]))
    table = xr.concat(tables, dim="model", coords="minimal", compat="override")
    table.attrs = {"alpha": next(iter(results.values()))["moments"].attrs["alpha"]}
    return table


# ---------------------------------------------------------------------------
# 10. Synthetic ensembles with a known answer
# ---------------------------------------------------------------------------

#(t): Toy cases for demonstrating the tests: keyword arguments for simulate_ensemble
SCENARIOS = {
    "Consistent": {},
    "Too little noise": {"noise": 0.6},
    "Too much noise": {"noise": 1.6},
    "Signal too strong": {"trend": 0.9},
    "Mean-state bias": {"bias": 2.0},
}

#(t): What is wrong with the model in each toy case (ERA5 always has trend 0.3/decade, noise 1)
SCENARIO_DESCRIPTIONS = {
    "Consistent": "model = ERA5: nothing should be flagged (except ~10% by chance)",
    "Too little noise": "model noise 0.6x ERA5's",
    "Too much noise": "model noise 1.6x ERA5's",
    "Signal too strong": "model trend 0.9/decade vs ERA5's 0.3",
    "Mean-state bias": "model 2° too warm; otherwise perfect",
}


def simulate_ensemble(n_members=50, years=np.arange(1979, 2014), seasons=SEASONS, trend=0.3, noise=1.0,
                      obs_trend=0.3, obs_noise=1.0, bias=0.0, seed=0):
    """A toy ensemble and pseudo-ERA5 with chosen signal and noise, so the right answer is known.

    Each member is ``bias + trend * decades + noise * e`` and pseudo-ERA5 is
    ``obs_trend * decades + obs_noise * e``, with e independent standard
    normal draws and decades counted from the first year. The defaults make
    the model perfect; ``SCENARIOS`` holds the departures used to demonstrate
    each test.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: ``(ensemble, obs)`` on (member, year, season) and (year, season).
    """
    rng = np.random.default_rng(seed)
    years = np.asarray(years)
    coords = {"year": years, "season": list(seasons)}
    decades = xr.DataArray((years - years[0]) / 10, dims="year", coords={"year": years})
    shape = (len(years), len(seasons))

    members = xr.DataArray(rng.standard_normal((n_members, *shape)), dims=("member", "year", "season"),
                           coords={"member": np.arange(n_members), **coords})
    ensemble = (bias + trend * decades + noise * members).transpose("member", "year", "season")
    obs = obs_trend * decades + obs_noise * xr.DataArray(rng.standard_normal(shape), dims=("year", "season"),
                                                         coords=coords)
    return ensemble.rename("tas"), obs.transpose("year", "season").rename("tas")
