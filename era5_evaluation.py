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
4.  Statistics          the four moments (mean, std, skewness, kurtosis) and other statistics
                        (trend, width, tails, records, lag-1 autocorrelation, ...)
5.  Locating ERA5       the percentile and p-value that every test reports
6.  Moment test         the four moments, ERA5 against every member; the other statistics the
                        same way; and the field test: is ERA5 flagged over more of a map than
                        a perfect model would be?
7.  Rank histograms     the Suarez-Gutierrez et al. (2021) test
8.  Distributions       kernel density estimates and quantiles
9.  Plume membership    is ERA5 inside the ensemble range?
10. Spatial evaluation  Suarez-Gutierrez et al. (2021), section 2.2.2: at every grid point, how
                        often ERA5 is below, above and in the middle of the ensemble, and why
11. Synthetic data      toy ensembles with a known answer, to demonstrate the tests

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
import scipy.stats
import xarray as xr

from quantile_calc import lowess_matrix_xarray, seasonal_mean
from significance import area_mean, pvalue_two_sided
from xarray_calc import split_time
from xarray_stats import nan_quantile

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

    Follows the LESFMIP pre-processing step for step (``quantile_calc.seasonal_mean``,
    then ``xarray_calc.split_time``), so DJF is labelled by the year of its
    December and the result lines up with ``lesfmip_season_tree`` label for
    label. Incomplete seasons, and years without all four, are dropped, e.g.
    the January-February stub at the start of ERA5.

    ERA5 can stay on its own (standard) calendar: the seasons are matched by
    their (year, season) labels, not by date. Converting it to the models'
    360-day calendar with ``align_on='year'`` moves month-start dates into the
    previous month, which this refuses.

    Args:
        monthly (xr.DataArray): Monthly means with a datetime ``dim``.
        dim (str): Name of the time dimension.
        min_months (int): Months a season needs in order to be kept.

    Returns:
        xr.DataArray: Seasonal means with ``year`` and ``season`` in place of ``dim``.
    """
    month_index = monthly[dim].dt.year * 12 + monthly[dim].dt.month
    if np.unique(month_index).size != month_index.size:
        raise ValueError(
            "some months appear more than once, so seasons would get the wrong months. "
            "convert_calendar('360_day', align_on='year') does this to month-start dates (1 March becomes "
            "29 February); keep ERA5 on its own calendar, or relabel it with monthly_time_axis"
        )
    return split_time(seasonal_mean(monthly, min_months=min_months, dim=dim), ("year", "season"), dim)


def monthly_time_axis(monthly, dim="time"):
    """Relabel consecutive monthly means with month-start dates on the standard calendar.

    Repairs a record whose dates were shifted, e.g. by
    ``convert_calendar('360_day', align_on='year')``, which moves 1 March to
    29 February and 1 December to 30 November. Assumes one value per month,
    with no gaps, starting in the month of the first date.

    Args:
        monthly (xr.DataArray): Monthly means with a datetime ``dim``.
        dim (str): Name of the time dimension.

    Returns:
        xr.DataArray: The same values on month-start dates.
    """
    start = f"{int(monthly[dim].dt.year[0]):04d}-{int(monthly[dim].dt.month[0]):02d}-01"
    return monthly.assign_coords({dim: xr.date_range(start, periods=monthly.sizes[dim], freq="MS")})


def match_grid(obs, like, tolerance=0.01, lat="lat", lon="lon"):
    """Put ``obs`` on the latitudes and longitudes of ``like``.

    ERA5's grid need not be the LESFMIP grid: it may use 0-360 longitudes where
    LESFMIP uses -180-180, run north to south, cover a different area, or sit
    on different points. So this:

    1. puts ``obs``'s longitudes in ``like``'s convention and sorts both axes;
    2. if every point of ``like`` has a partner in ``obs`` within ``tolerance``
       degrees, takes those values unchanged, relabelled with ``like``'s exact
       coordinates so arithmetic between the two aligns;
    3. otherwise interpolates bilinearly onto ``like``'s points, wrapping round
       in longitude. Points outside ``obs``'s coverage are NaN.

    Interpolation suits data at a similar resolution, like the ERA5 store
    (conservatively regridded to 2.5°). Regrid much finer data conservatively
    first, as the ERA5 processing section does with xesmf.

    Args:
        obs (xr.DataArray): Data to put on the grid.
        like (xr.DataArray): Data on the target grid.
        tolerance (float): Largest coordinate difference, in degrees, still treated as the same point.

    Returns:
        xr.DataArray: ``obs`` on ``like``'s ``lat`` and ``lon``.
    """
    target = {name: np.asarray(like[name].values, dtype=float) for name in (lat, lon)}
    #(c): Longitudes into the target's convention: -180-180 if it has any negative longitude, else 0-360.
    #(c): The range starts `tolerance` early, so -180.000001 stays next to -180 rather than wrapping to +180
    start = (-180.0 if target[lon].min() < 0 else 0.0) - tolerance
    wrapped = (obs[lon] - start) % 360 + start
    obs = obs.assign_coords({lon: wrapped.astype(float), lat: obs[lat].astype(float)}).sortby([lat, lon])

    def gap(name):
        return np.abs(obs[name].values[:, None] - target[name][None, :]).min(0).max()

    if gap(lat) <= tolerance and gap(lon) <= tolerance:
        nearest = {name: obs.indexes[name].get_indexer(target[name], method="nearest") for name in (lat, lon)}
        return obs.isel(nearest).assign_coords(target)

    #(c): One extra column at each end, a full turn away, so interpolation wraps round in longitude
    padded = xr.concat([obs.isel({lon: [-1]}).assign_coords({lon: obs[lon][-1:] - 360}), obs,
                        obs.isel({lon: [0]}).assign_coords({lon: obs[lon][:1] + 360})], dim=lon)
    return padded.interp(target)


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
    ensemble, obs = ensemble.where(valid), obs.where(valid)
    #(c): A gap this size is a unit mismatch (K against °C), not a model bias
    offset = float(ensemble.mean()) - float(obs.mean())
    if abs(offset) > 100:
        raise ValueError(f"the ensemble is {offset:+.0f} from obs on average: is one in K and the other in °C?")
    return ensemble, obs


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


#(t): LOWESS window for the emergence statistic, in years (a ~35-year record cannot carry the 81 used elsewhere)
EMERGENCE_WINDOW = 25


def _last_minus_first(values):
    """Last finite value minus the first, along the last axis."""
    finite = np.isfinite(values)
    first = np.argmax(finite, axis=-1)[..., None]
    last = (values.shape[-1] - 1 - np.argmax(finite[..., ::-1], axis=-1))[..., None]
    change = np.take_along_axis(values, last, -1)[..., 0] - np.take_along_axis(values, first, -1)[..., 0]
    return np.where(finite.any(axis=-1), change, np.nan)


def self_signal_to_noise(da, dim=SAMPLE_DIM, window=EMERGENCE_WINDOW):
    """S/N of a series against itself: how far it has moved relative to its own year-to-year noise.

    signal   the change of its LOWESS-smoothed curve from the start of the record to the end
    noise    the standard deviation of its deviations from that curve

    ERA5 has no counterfactual, so this is the emergence measure that can be
    computed for ERA5 and, identically, for every member.
    """
    smooth = lowess_matrix_xarray(da, core_dims=dim, window=window)
    signal = xr.apply_ufunc(_last_minus_first, smooth, input_core_dims=[[dim]], dask="parallelized",
                            output_dtypes=[float])
    return signal / (da - smooth).std(dim, ddof=1)


def _record_flags(values, high=True):
    """True in each year that beats every earlier year (the first year counts), along the last axis."""
    x = np.where(np.isfinite(values), values if high else -values, -np.inf)
    earlier_best = np.concatenate([np.full(x.shape[:-1] + (1,), -np.inf),
                                   np.maximum.accumulate(x, axis=-1)[..., :-1]], axis=-1)
    return (x > earlier_best) & np.isfinite(values)


def _record_count(values, high=True):
    return np.where(np.isfinite(values).any(axis=-1), _record_flags(values, high).sum(axis=-1), np.nan)


def record_highs(da, dim=SAMPLE_DIM):
    """Number of record highs: years warmer than every earlier year in the record (the first year counts)."""
    return xr.apply_ufunc(_record_count, da, input_core_dims=[[dim]], kwargs={"high": True},
                          dask="parallelized", output_dtypes=[float])


def record_lows(da, dim=SAMPLE_DIM):
    """Number of record lows: years colder than every earlier year in the record (the first year counts)."""
    return xr.apply_ufunc(_record_count, da, input_core_dims=[[dim]], kwargs={"high": False},
                          dask="parallelized", output_dtypes=[float])


def cumulative_records(da, dim=SAMPLE_DIM, high=True):
    """Running number of record highs (or lows) up to each year, for plotting record rates."""
    return xr.apply_ufunc(lambda v: np.cumsum(_record_flags(v, high), axis=-1).astype(float), da,
                          input_core_dims=[[dim]], output_core_dims=[[dim]], dask="parallelized",
                          output_dtypes=[float])


def expected_records(n_years):
    """Expected number of records in n years of a stationary, independent series: 1 + 1/2 + ... + 1/n.

    In year k every one of the k years so far is equally likely to be the
    largest, so year k is a record with probability 1/k.
    """
    return float(np.sum(1 / np.arange(1, n_years + 1)))


@dataclass(frozen=True)
class Diagnostic:
    """How to compute, label and read one test statistic.

    ``low`` and ``high`` say what it means for the *model* when ERA5 falls
    significantly below or above the members; the plots print them as the
    verdict. ``func`` and ``treatment`` are used by ``statistic_test`` only.
    """

    label: str
    units: str
    low: str
    high: str
    treatment: str | None = None
    func: Callable | None = None


#(t): The four moments, each paired with the treatment that isolates what it tests
#(c): The mean is of the raw values (the climatology). The other three are of detrended values,
#(c): so the forced trend over the record neither inflates the spread nor bends the shape.
MOMENTS = {
    "mean": Diagnostic("Mean", "°C", "model too warm", "model too cold",
                       treatment="raw", func=mean),
    "std": Diagnostic("Std. deviation", "°C", "model too variable", "model not variable enough",
                      treatment="detrended", func=std),
    "skewness": Diagnostic("Skewness", "", "model skewness too high", "model skewness too low",
                           treatment="detrended", func=skewness),
    "kurtosis": Diagnostic("Excess kurtosis", "", "model tails too heavy", "model tails too light",
                           treatment="detrended", func=excess_kurtosis),
}

#(t): Other statistics, tested in exactly the same way but not moments
#(c): Signal: trend and emergence. Noise: width and the two tails, i.e. the quantities whose forced
#(c): changes the rest of the notebook analyses. Then record counts and year-to-year memory.
STATISTICS = {
    "trend": Diagnostic("Trend", "°C/decade", "model trend too large", "model trend too small",
                        treatment="raw", func=trend),
    "emergence": Diagnostic("S/N (own change)", "", "model emerges too strongly", "model emerges too weakly",
                            treatment="raw", func=self_signal_to_noise),
    "width": Diagnostic("Width (Q95 − Q05)", "°C", "model distribution too wide", "model distribution too narrow",
                        treatment="detrended", func=quantile_width),
    "lower_tail": Diagnostic("Cold tail (Q50 − Q05)", "°C", "model cold tail too long", "model cold tail too short",
                             treatment="detrended", func=lower_tail_width),
    "upper_tail": Diagnostic("Warm tail (Q95 − Q50)", "°C", "model warm tail too long", "model warm tail too short",
                             treatment="detrended", func=upper_tail_width),
    "records_high": Diagnostic("Record highs", "count", "model sets too many record highs",
                               "model sets too few record highs", treatment="raw", func=record_highs),
    "records_low": Diagnostic("Record lows", "count", "model sets too many record lows",
                              "model sets too few record lows", treatment="raw", func=record_lows),
    "lag1": Diagnostic("Lag-1 autocorrelation", "", "model too persistent", "model not persistent enough",
                       treatment="detrended", func=lag1_autocorrelation),
}

#(t): The other statistics grouped by the question they answer
SIGNAL_STATISTICS = ("trend", "emergence")
NOISE_STATISTICS = ("width", "lower_tail", "upper_tail")
RECORD_STATISTICS = ("records_high", "records_low")

#(t): Rank-histogram summaries (section 7); the treatment is chosen when the test is run
RANKS = {
    "dispersion": Diagnostic("Rank dispersion", "", "spread too large (dome)", "spread too small (U shape)"),
    "outside": Diagnostic("Outside ensemble", "%", "ERA5 too rarely outside", "ERA5 too often outside"),
    "below": Diagnostic("Below all members", "%", "ERA5 too rarely below", "ERA5 too often below"),
    "above": Diagnostic("Above all members", "%", "ERA5 too rarely above", "ERA5 too often above"),
    "mean_rank": Diagnostic("Mean rank", "", "ERA5 sits low: model too warm", "ERA5 sits high: model too cold"),
    "rank_trend": Diagnostic("Rank trend", "/decade", "model warms faster than ERA5", "model warms slower than ERA5"),
}

DIAGNOSTICS = {**MOMENTS, **STATISTICS, **RANKS}


# ---------------------------------------------------------------------------
# 5. Locating ERA5 among the members
# ---------------------------------------------------------------------------

#(t): The variables every test result carries, so results can be stacked and plotted alike
RESULT_VARIABLES = ("era5", "members", "lower", "median", "upper", "percentile", "pvalue", "verdict", "testable")


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
            verdict               -1 ERA5 significantly below the members, +1 above, 0 not flagged
            testable              whether N is large enough for anything to be flagged at all;
                                  where it is not, a verdict of 0 means "untestable", not "consistent"
    """
    n = members.notnull().sum(member_dim)
    below = (members < observed).sum(member_dim)
    ties = (members == observed).sum(member_dim)
    percentile = 100 * (below + 0.5 * ties) / n

    bounds = nan_quantile(members, [alpha / 2, 0.5, 1 - alpha / 2], member_dim)
    pvalue = pvalue_two_sided(members, observed, dim=member_dim, method="tails")
    verdict = xr.where(pvalue < alpha, np.sign(percentile - 50), 0)
    #(c): The smallest possible two-sided p-value is 2 / (N + 1); if that is not below alpha nothing can be flagged
    testable = (2 / (n + 1)) < alpha

    result = xr.Dataset({
        "era5": observed,
        "members": members,
        "lower": bounds.isel(quantile=0, drop=True),
        "median": bounds.isel(quantile=1, drop=True),
        "upper": bounds.isel(quantile=2, drop=True),
        "percentile": percentile,
        "pvalue": pvalue,
        "verdict": verdict,
        "testable": testable,
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
# 6. Moment test (and the other statistics, tested the same way)
# ---------------------------------------------------------------------------

def statistic_test(ensemble, obs, statistics=tuple(STATISTICS), sample_dim=SAMPLE_DIM,
                   member_dim=MEMBER_DIM, reference=None, alpha=ALPHA):
    """Test ERA5 against the ensemble one statistic at a time.

    For each statistic: give ERA5 and every member the same treatment, compute
    the statistic over ``sample_dim`` for each, and ``locate`` ERA5 among the
    members. The members' values are the sampling distribution of that
    statistic for this record length under the model's climate, so no
    parametric formula or effective sample size is needed. The treatment
    isolates what is tested:

        statistic   treatment   tests
        The moments (``MOMENTS``; ``moment_test`` runs these four)
        mean        raw         the climatology (mean-state bias)
        std         detrended   the size of year-to-year variability (noise)
        skewness    detrended   the asymmetry of the noise
        kurtosis    detrended   the tail weight of the noise (excess kurtosis)
        Other statistics (``STATISTICS``)
        trend       raw         the forced response over the record (signal)
        emergence   raw         the smoothed change over the record in units of the
                                series' own noise (its S/N against itself)
        width       detrended   the Q95 - Q05 width of the distribution
        lower_tail  detrended   the length of the cold tail, Q50 - Q05
        upper_tail  detrended   the length of the warm tail, Q95 - Q50
        records_high, records_low
                    raw         how many record highs / lows the record sets;
                                1/2 + ... + 1/n are expected with no change at all
        lag1        detrended   the year-to-year memory of the noise

    Args:
        ensemble, obs (xr.DataArray): Output of ``align``.
        statistics (Sequence[str]): Keys of ``MOMENTS`` or ``STATISTICS``.
        sample_dim, member_dim (str): Dimension names.
        reference (slice | None): Anomaly reference years (only 'anomaly' uses it).
        alpha (float): Two-sided significance level.

    Returns:
        xr.Dataset: ``locate`` output on a ``statistic`` dim, with ``label``,
        ``units`` and ``treatment`` coordinates along it.
    """
    registry = {**MOMENTS, **STATISTICS}
    treated = {}
    results = []
    for name in statistics:
        diagnostic = registry[name]
        if diagnostic.treatment not in treated:
            treated[diagnostic.treatment] = (
                treat(ensemble, diagnostic.treatment, sample_dim, reference),
                treat(obs, diagnostic.treatment, sample_dim, reference),
            )
        ensemble_t, obs_t = treated[diagnostic.treatment]
        results.append(locate(diagnostic.func(ensemble_t, sample_dim), diagnostic.func(obs_t, sample_dim),
                              member_dim, alpha))

    stacked = _stack(results, statistics, registry, test="statistic", alpha=alpha,
                     n_members=ensemble.sizes[member_dim])
    return stacked.assign_coords(treatment=("statistic", [registry[name].treatment for name in statistics]))


def moment_test(ensemble, obs, **kwargs):
    """The moment test: ERA5's mean, std, skewness and excess kurtosis among the members'.

    ``statistic_test`` on the four ``MOMENTS``; it takes the same keyword
    arguments. Use ``field_test`` on its output to ask whether a model is
    acceptable over a whole map rather than point by point.
    """
    return statistic_test(ensemble, obs, tuple(MOMENTS), **kwargs).assign_attrs(test="moment")


def _flagged(below, n_others, alpha):
    """Whether a value with ``below`` of ``n_others`` values under it is flagged by the two-sided rank test.

    The same rule ``locate`` applies through ``pvalue_two_sided`` (ties aside):
    p = 2 x (the smaller tail count + 1) / (n_others + 1).
    """
    above = n_others - below
    return 2 * (np.minimum(below, above) + 1) / (n_others + 1) < alpha


def field_test(result, dims=("lat", "lon"), member_dim=MEMBER_DIM, alpha=ALPHA):
    """Is ERA5 flagged over more of the map than it would be if the model were perfect?

    A map of point-wise verdicts cannot say by itself whether a model is
    acceptable: at alpha = 0.1 a perfect model is flagged at about 10% of grid
    points by chance, and neighbouring points are correlated, so the flagged
    area of a perfect model varies a lot from one realisation to the next.
    This is the Monte Carlo field-significance test of Livezey and Chen
    (1983), with the members supplying the null. Each member in turn plays
    ERA5 against the other N - 1 (the pseudo-observations of the rank-histogram
    test), and the areas they are flagged over show what a perfect model gives,
    with the model's own spatial correlation built in. The model fails for a
    statistic when ERA5 is flagged over more area than all but alpha of them.

    ERA5 is ranked against each of the N leave-one-out sub-ensembles of N - 1
    members and the verdicts averaged, as in ``leave_one_out_counts``, so ERA5
    and the pseudo-observations face exactly the same test.

    Args:
        result (xr.Dataset): ``moment_test`` or ``statistic_test`` output on a
            lat/lon grid, still with ``members``.
        dims (Sequence[str]): The map dimensions; areas are cos(latitude)-weighted.
        member_dim (str): Member dimension.
        alpha (float): Significance level, for the point-wise tests and the field test.

    Returns:
        xr.Dataset:
            flagged_area          % of the area where ERA5 is flagged
            pseudo_flagged_area   (member) the same for each pseudo-observation
            perfect_area          the (1 - alpha) quantile of pseudo_flagged_area:
                                  a perfect model is flagged over more only alpha of the time
            field_pvalue          one-sided: the share of pseudo-observations flagged over at
                                  least as much area as ERA5 (with the +1 correction)
            field_verdict         1 if field_pvalue < alpha (flagged over too much of the map), else 0
        NaN where the ensemble is too small for the point-wise test to flag anything.
    """
    members = result["members"]
    n = members.notnull().sum(member_dim)
    below = (members < result["era5"]).sum(member_dim)
    era5_flagged = ((below / n) * _flagged(below - 1, n - 1, alpha)
                    + ((n - below) / n) * _flagged(below, n - 1, alpha))
    pseudo_below = xr.apply_ufunc(_rank_among_others, members, input_core_dims=[[member_dim]],
                                  output_core_dims=[[member_dim]], dask="parallelized", output_dtypes=[float])
    pseudo_flagged = _flagged(pseudo_below, n - 1, alpha)

    #(c): A pseudo-observation has N - 1 others, so it can only be flagged if 2 / N < alpha
    present = result["era5"].notnull() & (2 / n < alpha)
    weights = np.cos(np.deg2rad(result["lat"]))

    def area(flagged):
        return 100 * flagged.astype(float).where(present).weighted(weights).mean(dims)

    flagged_area = area(era5_flagged)
    pseudo_area = area(pseudo_flagged)
    n_pseudo = pseudo_area.notnull().sum(member_dim)
    pvalue = ((pseudo_area >= flagged_area).sum(member_dim) + 1) / (n_pseudo + 1)
    valid = flagged_area.notnull()
    return xr.Dataset({
        "flagged_area": flagged_area,
        "pseudo_flagged_area": pseudo_area,
        "perfect_area": pseudo_area.quantile(1 - alpha, dim=member_dim, skipna=True).drop_vars("quantile"),
        "field_pvalue": pvalue.where(valid),
        "field_verdict": (pvalue < alpha).where(valid),
    }, attrs={"alpha": alpha})


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
#(c): 12.5-87.5% is the central 75% of Suarez-Gutierrez et al. (2021)
PLUME_QUANTILES = (0.05, 0.125, 0.5, 0.875, 0.95)


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
            central           where ERA5 is inside the members' central 75%, 12.5-87.5% (1.0 or 0.0)
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
    central = ensemble.quantile(list(CENTRAL_RANGE), dim=member_dim)
    return xr.Dataset({
        "quantiles": ensemble.quantile(list(quantiles), dim=member_dim),
        "minimum": minimum,
        "maximum": maximum,
        "obs": obs,
        "below": (obs < minimum).where(present),
        "above": (obs > maximum).where(present),
        "central": ((obs >= central.isel(quantile=0, drop=True))
                    & (obs <= central.isel(quantile=1, drop=True))).where(present),
        "n_members": n_members,
        "expected_outside": 200 / (n_members + 1),
    }, attrs={"treatment": treatment})


def outside_frequency(plume, dims=SAMPLE_DIM):
    """% of samples with ERA5 below, above and outside the ensemble range, and inside its central 75%."""
    below = 100 * plume["below"].mean(dims)
    above = 100 * plume["above"].mean(dims)
    return xr.Dataset({
        "below": below,
        "above": above,
        "outside": below + above,
        "central": 100 * plume["central"].mean(dims),
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
            moments         moment_test (mean, std, skewness, kurtosis), per season
            statistics      statistic_test of the other statistics, per season
            ranks           {treatment: rank_histogram_test pooled over ``pool_dims``}
            ranks_seasonal  {treatment: rank_histogram_test per season}, without histograms
            plumes          ensemble_plume for raw values and anomalies, along ``treatment``
            distributions   distribution_summary for raw values, anomalies and detrended
    """
    kwargs = dict(sample_dim=sample_dim, member_dim=member_dim, reference=reference)
    return {
        "moments": moment_test(ensemble, obs, alpha=alpha, **kwargs),
        "statistics": statistic_test(ensemble, obs, alpha=alpha, **kwargs),
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
        table = combine_tests(result["moments"], result["statistics"], *ranks).drop_dims(member_dim)
        tables.append(table.expand_dims(model=[model]))
    table = xr.concat(tables, dim="model", coords="minimal", compat="override")
    table.attrs = {"alpha": next(iter(results.values()))["moments"].attrs["alpha"]}
    return table


# ---------------------------------------------------------------------------
# 10. Spatial evaluation (Suarez-Gutierrez et al., 2021, section 2.2.2)
# ---------------------------------------------------------------------------
# At a grid point the paper does not test rank histograms against pseudo-
# observations (that is for the global mean). It reduces the histogram to three
# numbers, how often the observations are below every member, above every
# member, and inside the members' central 75%, and compares them with what a
# perfect model would give, using generous fixed thresholds. When the edge
# years happen then says whether the forced response or the variability is wrong.

#(t): The members' central 75%
CENTRAL_RANGE = (0.125, 0.875)

#(t): The paper's thresholds, generous because short records push the percentages up by chance:
#(c): a perfect model puts ERA5 below (or above) every member 100 / (N + 1) % of the time, and 75% in the middle
EDGE_THRESHOLD = 10       # % of years below (or above) every member: at least this is a problem
CENTRAL_THRESHOLD = 80    # % of years inside the central 75%: more than this is a problem

#(t): Too few years, or too few members for the fixed 10% (N = 10 puts ERA5 beyond a member 9% of the time
#(c): by chance), and a grid point is not tested. 20 is the smallest ensemble in the paper (5% by chance)
MIN_YEARS = 10
MIN_MEMBERS = 20

#(t): Share of a problem's years in one half of the period for it to count as happening "mostly" then
TIMING_SHARE = 2 / 3

#(t): Step 7, what is wrong at a grid point
DIAGNOSES = {
    0: "adequate",                 # none of the problems below
    1: "warms too much",           # ERA5 below the members late (or above them early): forced response too strong
    2: "warms too little",         # ERA5 above the members late (or below them early): forced response too weak
    3: "too little variability",   # ERA5 both below and above the members, throughout: spread too narrow
    4: "too much variability",     # ERA5 inside the central 75% too often: spread too wide
    5: "one tail too short",       # ERA5 beyond one edge only, throughout: the shape of the distribution
}

#(t): Step 8, for a problem over the whole record: does it stay the same in an early and a late period?
PERIOD_ALPHA = 0.1
PERIOD_DIAGNOSES = {
    0: "no problem over the whole record",
    1: "same problem in both: variability",
    2: "problem changes: forced response",
}


def _central_of_others(values, low, high):
    """For each member (last axis), whether it lies inside the [low, high] quantiles of the other members.

    The quantiles of the other N - 1 are read off the sorted full ensemble,
    skipping the member itself, so there is no loop over members.
    """
    ordered = np.sort(values, axis=-1)
    position = np.argsort(np.argsort(values, axis=-1), axis=-1)
    n_others = values.shape[-1] - 1

    def others_quantile(q):
        index = (n_others - 1) * q
        k = int(np.floor(index))

        def order_statistic(j):
            #(c): The j-th smallest of the others is the j-th of the ensemble, or the next one once past the member
            return np.take_along_axis(ordered, np.where(j < position, j, j + 1), axis=-1)

        lower, upper = order_statistic(k), order_statistic(min(k + 1, n_others - 1))
        return lower + (index - k) * (upper - lower)

    return (values >= others_quantile(low)) & (values <= others_quantile(high))


def _perfect_model_thresholds(ensemble, present, member_dim, sample_dim, quantile=0.95):
    """Optional step 6, not in the paper: the same three percentages for each member treated as ERA5.

    Each member is placed among the other N - 1; the ``quantile`` across
    members of each percentage is what a perfect model exceeds only
    1 - ``quantile`` of the time.
    """
    position = pseudo_observation_ranks(ensemble, member_dim)
    n_members = ensemble.sizes[member_dim]
    central = xr.apply_ufunc(
        _central_of_others, ensemble, input_core_dims=[[member_dim]], output_core_dims=[[member_dim]],
        kwargs={"low": CENTRAL_RANGE[0], "high": CENTRAL_RANGE[1]}, dask="parallelized", output_dtypes=[bool],
    )
    pseudo = {
        "below": (position == 0).where(present),
        "above": (position == n_members - 1).where(present),
        "central": central.where(present),
    }
    return {name: nan_quantile(100 * flag.mean(sample_dim), quantile, member_dim) for name, flag in pseudo.items()}


def spatial_evaluation(ensemble, obs, reference=None, period=None, thresholds="fixed",
                       sample_dim=SAMPLE_DIM, member_dim=MEMBER_DIM):
    """The grid-point evaluation of Suarez-Gutierrez et al. (2021), at every grid point (and season) at once.

    1-2. Anomalies: each member and ERA5 minus its own mean over ``reference``
         (all the years by default). The constant offset is gone; the forced
         change and the variability are left.
    3.   Only years ERA5 has (``align`` has already masked the rest, in ERA5
         and every member); with fewer than ``MIN_YEARS``, a point is untested.
    4.   Each year, is ERA5 below every member (rank 0) or above every member (rank N)?
    5.   % of years below, % above, and % inside the members' central 75% (12.5-87.5%).
    6.   Compare with a perfect model: ``thresholds="fixed"`` uses the paper's
         10%, 10% and 80%; ``"perfect_model"`` (not in the paper) treats each
         member in turn as ERA5 against the other N - 1 and flags a percentage
         above the 95th percentile of theirs.
    7.   Diagnose (``DIAGNOSES``), using when the edge years fall: in the first
         or the second half of the period. ERA5 below the members late, or
         above them early, means the model warms too much (anomalies are
         relative to the whole record, so a model that warms too fast looks too
         warm late and too cold early, relative to ERA5); the reverse, too
         little. Both edges throughout means too little variability; too often
         in the middle, too much; one edge throughout, a tail of the wrong length.

    Step 8 is ``period_comparison``; step 9 (the area score and the model
    count) is ``adequate_area`` and ``adequate_count``.

    Args:
        ensemble (xr.DataArray): Members with ``member_dim`` and ``sample_dim``, from ``align``.
        obs (xr.DataArray): ERA5 from ``align``.
        reference (slice | None): Years whose mean is removed from every series (all if None).
        period (slice | None): Years to evaluate, after the anomalies are taken, for step 8.
        thresholds (str): "fixed" (the paper) or "perfect_model".
        sample_dim, member_dim (str): Dimension names.

    Returns:
        xr.Dataset on the remaining dims (season, lat, lon):
            below, above, central                   % of years
            below_late, above_late                  share of those below / above years in the second half
            below_threshold, ...                    the thresholds used (step 6)
            below_flag, above_flag, central_flag    each beyond its threshold (1/0, NaN if untested)
            diagnosis                               key of ``DIAGNOSES`` (NaN if untested)
            adequate                                1 adequate, 0 not, NaN untested
            n_years, n_members
    """
    if thresholds not in ("fixed", "perfect_model"):
        raise ValueError(f"thresholds must be 'fixed' or 'perfect_model', got {thresholds!r}")
    ensemble = treat(ensemble, "anomaly", sample_dim, reference)
    obs = treat(obs, "anomaly", sample_dim, reference)
    if period is not None:
        ensemble, obs = ensemble.sel({sample_dim: period}), obs.sel({sample_dim: period})

    #(t): Steps 3-5
    present = obs.notnull()
    n_years = present.sum(sample_dim)
    n_members = ensemble.sizes[member_dim]
    central_range = nan_quantile(ensemble, list(CENTRAL_RANGE), member_dim)
    flags = {
        "below": (obs < ensemble.min(member_dim)).where(present),
        "above": (obs > ensemble.max(member_dim)).where(present),
        "central": ((obs >= central_range.isel(quantile=0, drop=True))
                    & (obs <= central_range.isel(quantile=1, drop=True))).where(present),
    }
    percent = {name: 100 * flag.mean(sample_dim) for name, flag in flags.items()}
    years = obs[sample_dim]
    late = years > (years.min() + years.max()) / 2
    late_share = {name: flags[name].where(late, 0).sum(sample_dim) / flags[name].sum(sample_dim)
                  for name in ("below", "above")}

    #(t): Step 6
    if thresholds == "fixed":
        limit = {"below": EDGE_THRESHOLD, "above": EDGE_THRESHOLD, "central": CENTRAL_THRESHOLD}
        beyond = {"below": percent["below"] >= EDGE_THRESHOLD, "above": percent["above"] >= EDGE_THRESHOLD,
                  "central": percent["central"] > CENTRAL_THRESHOLD}
        testable = (n_years >= MIN_YEARS) & (n_members >= MIN_MEMBERS)
    else:
        limit = _perfect_model_thresholds(ensemble, present, member_dim, sample_dim)
        beyond = {name: percent[name] > limit[name] for name in percent}
        testable = n_years >= MIN_YEARS

    #(t): Step 7, each later rule overriding the earlier ones, so a timed edge (the forced response) wins
    below, above = beyond["below"], beyond["above"]
    both = below & above
    below_late, below_early = late_share["below"] >= TIMING_SHARE, late_share["below"] <= 1 - TIMING_SHARE
    above_late, above_early = late_share["above"] >= TIMING_SHARE, late_share["above"] <= 1 - TIMING_SHARE
    #(c): With both edges beyond, both must point the same way: a few edge years bunching in one half by chance
    #(c): is common, and too little variability would otherwise often pass for a forced-response error
    too_much = xr.where(both, below_late & above_early, (below & below_late) | (above & above_early))
    too_little = xr.where(both, above_late & below_early, (above & above_late) | (below & below_early))
    diagnosis = xr.zeros_like(n_years, dtype=float)
    diagnosis = xr.where(beyond["central"], 4, diagnosis)
    diagnosis = xr.where(below ^ above, 5, diagnosis)
    diagnosis = xr.where(both, 3, diagnosis)
    diagnosis = xr.where(too_little, 2, diagnosis)
    diagnosis = xr.where(too_much, 1, diagnosis)

    def tested(da):
        return da.astype(float).where(testable)

    return xr.Dataset({
        **{name: value.where(testable) for name, value in percent.items()},
        **{f"{name}_late": share.where(testable) for name, share in late_share.items()},
        **{f"{name}_threshold": xr.full_like(n_years, value, dtype=float) if np.isscalar(value) else value
           for name, value in limit.items()},
        **{f"{name}_flag": tested(flag) for name, flag in beyond.items()},
        "diagnosis": diagnosis.where(testable),
        "adequate": tested(diagnosis == 0),
        "n_years": n_years,
        "n_members": n_members,
    }, attrs={"thresholds": thresholds})


def _binomial_pvalue(k, n, share):
    """Two-sided binomial p-value of k successes in n trials with success probability ``share``."""
    lower = scipy.stats.binom.cdf(k, n, share)
    upper = scipy.stats.binom.sf(k - 1, n, share)
    return np.minimum(1, 2 * np.minimum(lower, upper))


def period_comparison(whole, early, late, alpha=PERIOD_ALPHA):
    """Step 8: is a problem over the whole record the same in an early and a late period?

    A model's variability changes little over time, but the forcing does: a
    problem from the variability shows up as often in both periods, one from
    the forced response appears, grows or flips between them. Each period is
    short (~17 years per season), so which thresholds it crosses on its own is
    largely chance (in 17 years a perfect model is in the central 75% more than
    80% of the time in about a third of cases). So each problem the whole
    record flags is tested for an uneven split between the periods instead,
    with a two-sided binomial test (p < ``alpha``: the forced response):

    - ERA5 beyond the edges: a forced-response error puts ERA5 above the
      members in one period and below them in the other (or beyond one edge in
      one period only). So of all the years beyond either edge, the share that
      fits "too much warming" (above early, below late) is tested against what
      an even spread over the two periods gives. Pooling both edges is what
      gives the test its power: each edge alone has only a few years.
    - ERA5 too often in the middle: its years in the central 75% are tested
      for being shared between the periods in proportion to their lengths.

    Args:
        whole, early, late (xr.Dataset): ``spatial_evaluation`` of the whole
            record and of each period, with the same ``reference``.
        alpha (float): Significance level of the split tests.

    Returns:
        xr.DataArray: Key of ``PERIOD_DIAGNOSES``; NaN where untested.
    """
    n_early, n_late = early["n_years"], late["n_years"]
    count = {(name, when): np.round(period[name] * period["n_years"] / 100)
             for name in ("below", "above", "central") for when, period in (("early", early), ("late", late))}
    above, below = count["above", "early"] + count["above", "late"], count["below", "early"] + count["below", "late"]
    #(c): Under an even spread an edge year is early with probability n_early / n
    expected = (above * n_early + below * n_late) / ((n_early + n_late) * (above + below))
    edge_p = xr.apply_ufunc(_binomial_pvalue, count["above", "early"] + count["below", "late"], above + below, expected)
    central_p = xr.apply_ufunc(_binomial_pvalue, count["central", "late"],
                               count["central", "early"] + count["central", "late"], n_late / (n_early + n_late))
    edge_problem = (whole["below_flag"] == 1) | (whole["above_flag"] == 1)
    changed = (edge_problem & (edge_p < alpha)) | ((whole["central_flag"] == 1) & (central_p < alpha))
    out = xr.where(whole["adequate"] == 1, 0, xr.where(changed, 2, 1))
    return out.where(whole["adequate"].notnull()).rename("period_diagnosis")


def adequate_area(result, lat="lat", lon="lon"):
    """Step 9: % of the tested area (cos-latitude weighted) where the model has no problem."""
    return 100 * area_mean(result["adequate"], lat=lat, lon=lon)


def adequate_count(results, model_dim="model"):
    """Step 9 across models (Fig. 8 of the paper): how many tested models are adequate at each grid point.

    Returns:
        xr.Dataset: ``n_adequate`` and ``n_tested``, with ``model_dim`` reduced.
    """
    return xr.Dataset({
        "n_adequate": (results["adequate"] == 1).sum(model_dim),
        "n_tested": results["adequate"].notnull().sum(model_dim),
    })


# ---------------------------------------------------------------------------
# 11. Synthetic ensembles with a known answer
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
