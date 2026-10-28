"""Optimal fingerprinting: how much of each forcing's simulated pattern of trends is in the observed trends.

The observed trends, one per region and season, are written as a sum of the
models' forced trends plus internal variability:

    y = β₁x₁ + β₂x₂ + ... + ε

    y     the observed trends, as one vector (``flatten``: region-major, the seasons inside)
    xᵢ    forcing i's simulated trend in the same entries: its fingerprint
    βᵢ    its scaling factor: how much of that pattern the observations hold
    ε     internal variability, with covariance C

"Optimal" means the fit is weighted by C⁻¹: entries that vary a lot on their own
count less, quiet ones more. In practice the vectors are prewhitened by
P = C^(-1/2), after which the noise is uncorrelated with equal variance and the
fit is an ordinary regression (``fit_ols``). Total least squares (``fit_tls``)
also allows for the noise left in the fingerprints, which have finite ensembles.

Reading β (with its 5-95% range from ``monte_carlo``):
    excludes 0     the forcing is detected in the observations
    includes 1     consistent: the model's response has the right size
    above 1        the model underestimates the response; below 1, it overestimates it
The fit is only to be trusted if the residual consistency test passes
(``residual_test``): what is left over must look like internal variability.

The steps follow Allen & Tett (1999), Allen & Stott (2003) and Ribes et al.
(2013), with Ledoit & Wolf's (2004) regularised covariance. Everything here
works on numpy arrays with fixed shapes: y (n,), X (n, k), C (n, n), noise
samples (s, n) and effective ensemble sizes (k,); the notebook turns them back
into labelled xarray objects.

Sections
--------
1. Vectors: the order of the entries
2. Fingerprints, signal sets and effective ensemble sizes
3. Noise: samples, two independent halves, the covariance
4. Fitting: prewhitening, OLS and TLS
5. Uncertainty: Monte Carlo
6. The residual consistency test
"""

import numpy as np
import xarray as xr

from .datatree import skip_empty

# ---------------------------------------------------------------------------
# 1. Vectors: the order of the entries
# ---------------------------------------------------------------------------

#(t): The order of a vector's entries: region-major, the seasons inside each region
ENTRY_DIMS = ("region", "season")


@skip_empty
def flatten(table, dims=ENTRY_DIMS):
    """A table of regions x seasons as one vector, along ``entry``, region-major.

    Sort the regions and seasons into the order wanted first: the vector
    follows the order they are in. ``unflatten`` turns it back.

    Args:
        table (xr.DataArray | xr.Dataset): With ``dims`` (and any others, which are kept).
        dims (Sequence[str]): The dims to flatten, slowest-varying first.

    Returns:
        The same type, with ``dims`` replaced by ``entry`` (a MultiIndex of them), last.
    """
    return table.stack(entry=list(dims)).transpose(..., "entry")


def unflatten(vector):
    """``flatten`` backwards: the ``entry`` dim split into its regions and seasons again."""
    return vector.unstack("entry")


# ---------------------------------------------------------------------------
# 2. Fingerprints, signal sets and effective ensemble sizes
# ---------------------------------------------------------------------------

#(t): The signal sets, from one signal to four: each signal is a sum of experiments' responses, with coefficients.
#(t): ANT is historical minus hist-nat; OTH (aerosols and ozone) is hist-aer plus hist-totalO3. Every set with
#(t): more than one signal assumes the single forcings add up (notebook calc_02b, section 3)
SIGNAL_SETS = {
    "1 signal": {"ALL": {"historical": 1}},
    "2 signals": {"ANT": {"historical": 1, "hist-nat": -1}, "NAT": {"hist-nat": 1}},
    "3 signals": {"GHG": {"hist-GHG": 1}, "OTH": {"hist-aer": 1, "hist-totalO3": 1}, "NAT": {"hist-nat": 1}},
    "4 signals": {"GHG": {"hist-GHG": 1}, "AER": {"hist-aer": 1}, "O3": {"hist-totalO3": 1}, "NAT": {"hist-nat": 1}},
}


#(t): Every signal of every set, in the order the figures list them
SIGNALS = list(dict.fromkeys(signal for signals in SIGNAL_SETS.values() for signal in signals))


def coefficients(signals, experiments):
    """How much of each experiment's response goes into each signal, as a (signal, experiment) matrix.

    Args:
        signals (dict): Signal -> {experiment: coefficient}, e.g. ``SIGNAL_SETS["2 signals"]``.
        experiments (Sequence[str]): The experiments, in order.
    """
    return xr.DataArray([[float(signal.get(experiment, 0)) for experiment in experiments]
                         for signal in signals.values()],
                        dims=("signal", "experiment"), coords={"signal": list(signals), "experiment": list(experiments)})


def model_fingerprints(ensemble_means, signals):
    """Each signal's vector in each model: its experiments' ensemble means, times their coefficients, added up.

    A model without one of a signal's experiments has no fingerprint for it (NaN).

    Args:
        ensemble_means (xr.DataArray): Ensemble-mean vectors on (model, experiment, entry).
        signals (dict): Signal -> {experiment: coefficient}.

    Returns:
        xr.DataArray: On (signal, model, entry).
    """
    weights = coefficients(signals, list(ensemble_means["experiment"].values))
    used = weights != 0
    #(c): Missing experiments are filled with 0 for the sum, then the signals that needed one are blanked
    total = (weights * ensemble_means.fillna(0)).sum("experiment")
    complete = (ensemble_means.notnull() | ~used).all("experiment")
    return total.where(complete).transpose("signal", "model", ...)


def effective_members(members, signals):
    """The effective ensemble size of each signal's multi-model mean fingerprint.

    A signal Σ cₑ x̄ₑ made from ensemble means of mₑ members has noise variance
    σ² Σ cₑ² / mₑ in one model, and the equal-weight mean of M models has
    σ² (1/M²) Σⱼ Σₑ cₑ² / mₑⱼ, so

        n_eff = 1 / [(1/M²) Σⱼ Σₑ cₑ² / mₑⱼ],

    which for one experiment is 1 / [(1/M²) Σⱼ 1/mⱼ]. Only models with all of
    a signal's experiments count, as in ``model_fingerprints``.

    Args:
        members (xr.DataArray): Members of each model and experiment, on (model, experiment); NaN if missing.
        signals (dict): Signal -> {experiment: coefficient}.

    Returns:
        xr.DataArray: n_eff on (signal).
    """
    weights = coefficients(signals, list(members["experiment"].values))
    used = weights != 0
    complete = (members.notnull() | ~used).all("experiment")
    inverse = (weights ** 2 / members).where(used, 0).sum("experiment").where(complete)
    n_models = complete.sum("model")
    return (n_models ** 2 / inverse.sum("model")).rename("n_eff")


def model_effective_members(members, signals):
    """``effective_members`` of each model's own fingerprint: 1 / Σₑ cₑ² / mₑ, on (signal, model)."""
    weights = coefficients(signals, list(members["experiment"].values))
    used = weights != 0
    complete = (members.notnull() | ~used).all("experiment")
    return (1 / (weights ** 2 / members).where(used, 0).sum("experiment")).where(complete).rename("n_eff")


# ---------------------------------------------------------------------------
# 3. Noise: samples, two independent halves, the covariance
# ---------------------------------------------------------------------------

@skip_empty
def ensemble_anomalies(members, dim="member"):
    """Each member minus its ensemble's mean, scaled by √(m / (m - 1)): samples of internal variability alone.

    Removing the ensemble mean removes the forced response, but also a
    1/m share of each member's own noise, which the scaling puts back, so the
    samples have the variance of one member's internal variability.
    """
    m = members.sizes[dim]
    return (members - members.mean(dim)) * np.sqrt(m / (m - 1))


def pool_samples(tree, variable, entry="entry"):
    """Every member of every node of a tree as one set of noise samples, with the half each belongs to.

    The halves alternate within each ensemble (its first, third, ... members
    in half 0, its second, fourth, ... in half 1), so the two halves are
    independent and each has members of every model and experiment.

    Args:
        tree (xr.DataTree): ``ensemble_anomalies`` of every node, each ``variable`` on (member, entry).
        variable (str): The variable.
        entry (str): The vector's dim.

    Returns:
        xr.DataArray: On (sample, entry), with ``model``, ``experiment``, ``member`` and ``half`` along ``sample``.
    """
    pieces = []
    for node in tree.leaves:
        if not node.data_vars:
            continue
        da = node[variable].transpose("member", entry)
        n = da.sizes["member"]
        pieces.append(da.rename(member="sample").assign_coords(
            model=("sample", [node.parent.name] * n), experiment=("sample", [node.name] * n),
            member=("sample", [str(m) for m in da["member"].values]), half=("sample", np.arange(n) % 2)))
    samples = xr.concat(pieces, dim="sample")
    return samples.drop_vars("sample").assign_coords(sample=np.arange(samples.sizes["sample"]))


def ledoit_wolf(samples):
    """Ledoit & Wolf's (2004) shrunk covariance of noise samples with a mean of zero.

    The sample covariance S of a few hundred samples in a few dozen entries
    is noisy, and its smallest eigenvalues far too small, which inverting it
    turns into the largest weights. Shrinking it towards a multiple of the
    identity, Σ = (1 - λ) S + λ μ I with μ the mean variance, fixes that; λ
    is the amount that minimises the expected error, estimated from the
    samples themselves (the same numbers as
    ``sklearn.covariance.ledoit_wolf(samples, assume_centered=True)``).

    Example: λ = 0.2 and S's eigenvalues [4, 1, 0.01], whose mean is μ = 1.67,
    give 0.8 * [4, 1, 0.01] + 0.2 * 1.67 = [3.53, 1.13, 0.34]: the smallest is
    lifted from 0.01 to 0.34, so its weight in the fit drops from 100 to 3. If
    S is 2 from μ I (delta) and 0.5 of that is sampling noise (beta), λ = 0.25.
    Notebook fingerprinting_01 (section 5) goes through it on the first half.

    Args:
        samples (np.ndarray): (s, n): s samples of n entries, each already with a mean of zero.

    Returns:
        tuple[np.ndarray, float]: The (n, n) covariance, and the shrinkage λ (0 to 1).
    """
    X = np.asarray(samples, dtype=float)
    n_samples, n_entries = X.shape
    sample_covariance = X.T @ X / n_samples
    variances = np.diag(sample_covariance)
    mu = variances.sum() / n_entries
    #(c): delta: how far S is from μI; beta: how much of that distance is sampling noise (each over n_entries)
    squared = X ** 2
    beta_sum = np.sum(squared.T @ squared)
    delta_sum = np.sum((X.T @ X) ** 2) / n_samples ** 2
    beta = (beta_sum / n_samples - delta_sum) / (n_entries * n_samples)
    delta = (delta_sum - 2 * mu * variances.sum() + n_entries * mu ** 2) / n_entries
    shrinkage = 0.0 if beta == 0 else min(beta, delta) / delta
    covariance = (1 - shrinkage) * sample_covariance + shrinkage * mu * np.eye(n_entries)
    return covariance, float(shrinkage)


# ---------------------------------------------------------------------------
# 4. Fitting: prewhitening, OLS and TLS
# ---------------------------------------------------------------------------

def inverse_sqrt(covariance):
    """P = C^(-1/2), from C's eigendecomposition: after P, noise with covariance C has the identity covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T


def fit_ols(y, X, covariance):
    """Generalised least squares, β̂ = (Xᵀ C⁻¹ X)⁻¹ Xᵀ C⁻¹ y, as ordinary least squares after prewhitening.

    Treats the fingerprints as exact. With noisy fingerprints (a small
    ensemble, e.g. ozone) β̂ is biased towards 0: regression dilution.

    Args:
        y (np.ndarray): (n,) the observations.
        X (np.ndarray): (n, k) the fingerprints, one per column.
        covariance (np.ndarray): (n, n) the noise covariance C.

    Returns:
        np.ndarray: (k,) β̂.
    """
    P = inverse_sqrt(covariance)
    return np.linalg.lstsq(P @ X, P @ y, rcond=None)[0]


def fit_tls(y, X, covariance, n_eff):
    """Total least squares (Allen & Stott, 2003): the fingerprints are allowed noise too, 1/n_eff of y's.

    1. Prewhiten y and X with P = C^(-1/2), so y's noise has unit variance.
    2. Scale each fingerprint by √n_eff, so its noise has unit variance too.
    3. The best fit through noisy points in every direction is the plane
       orthogonal to the smallest right singular vector v of Z = [X̃ₛ, ỹ]:
       βₛ = -v[:k] / v[k].
    4. Undo the scaling: βᵢ = βₛ,ᵢ √n_eff,ᵢ.

    Example: 2 entries, one fingerprint, and y = 2 x exactly, so Z = [[1, 2], [2, 4]].
    Z times [2, -1] is [0, 0]: Z has no extent in that direction, so it is the
    smallest singular vector, v = [2, -1] / √5, and βₛ = -(2 / √5) / (-1 / √5) = 2.
    Notebook fingerprinting_01 (section 7) goes through the steps on the real fit.

    Args:
        y (np.ndarray): (n,) the observations.
        X (np.ndarray): (n, k) the fingerprints.
        covariance (np.ndarray): (n, n) the noise covariance C.
        n_eff (np.ndarray): (k,) each fingerprint's effective ensemble size (``effective_members``).

    Returns:
        np.ndarray: (k,) β.
    """
    P = inverse_sqrt(covariance)
    scale = np.sqrt(np.asarray(n_eff, dtype=float))
    Z = np.column_stack([(P @ X) * scale, P @ y])
    v = np.linalg.svd(Z)[2][-1]
    return -v[:-1] / v[-1] * scale


def fit(y, X, covariance, n_eff=None, method="ols"):
    """``fit_ols`` or ``fit_tls``, by name."""
    if method == "ols":
        return fit_ols(y, X, covariance)
    if method == "tls":
        return fit_tls(y, X, covariance, n_eff)
    raise ValueError(f"method must be 'ols' or 'tls', got {method!r}")


def fit_each(y, fingerprints, n_eff, covariance, method="ols"):
    """``fit`` with every set of fingerprints along any other dim, e.g. each model's own (a robustness check).

    Args:
        y (xr.DataArray): The observations on (entry).
        fingerprints (xr.DataArray): On (signal, entry) and any other dims, e.g. (signal, model, entry).
        n_eff (xr.DataArray): On (signal) and the same other dims.
        covariance (np.ndarray): (n, n) the noise covariance.
        method (str): "ols" or "tls".

    Returns:
        xr.DataArray: β on the other dims and (signal); NaN where a fingerprint is missing.
    """
    def one(X, n):
        if not np.all(np.isfinite(X)):
            return np.full(X.shape[1], np.nan)
        return fit(y.values, X, covariance, n, method)

    return xr.apply_ufunc(one, fingerprints, n_eff, input_core_dims=[["entry", "signal"], ["signal"]],
                          output_core_dims=[["signal"]], vectorize=True).rename("beta")


def ols_covariance(X, covariance_fit, covariance_test):
    """The analytic covariance of the OLS β̂: A C₂ Aᵀ, with A = (Xᵀ C₁⁻¹ X)⁻¹ Xᵀ C₁⁻¹.

    C₁ (``covariance_fit``) is the one the fit was weighted with, C₂
    (``covariance_test``) an independent estimate of the noise.
    """
    inverse = np.linalg.inv(covariance_fit)
    A = np.linalg.solve(X.T @ inverse @ X, X.T @ inverse)
    return A @ covariance_test @ A.T


# ---------------------------------------------------------------------------
# 5. Uncertainty: Monte Carlo
# ---------------------------------------------------------------------------

def _perturbed(y, X, noise, n_eff, rng):
    """y plus one noise sample, and each fingerprint plus another divided by √n_eff, all drawn with replacement."""
    picks = rng.integers(0, noise.shape[0], size=X.shape[1] + 1)
    y_new = y + noise[picks[0]]
    X_new = X + noise[picks[1:]].T / np.sqrt(np.asarray(n_eff, dtype=float))
    return y_new, X_new


def monte_carlo(y, X, covariance, noise, n_eff, method="ols", n_trials=1000, seed=0):
    """The fit repeated on noisy copies of the observations and the fingerprints: the sampling spread of β.

    Each trial adds a noise sample (drawn with replacement from ``noise``,
    the half not used for the fit's covariance) to y, and to each
    fingerprint another divided by √n_eff, then refits with the same
    covariance. The 5th to 95th percentiles of the trials' β are the usual
    90% range of detection and attribution.

    Args:
        y (np.ndarray): (n,) the observations.
        X (np.ndarray): (n, k) the fingerprints.
        covariance (np.ndarray): (n, n) C₁, the fit's covariance.
        noise (np.ndarray): (s, n) independent noise samples (the other half).
        n_eff (np.ndarray): (k,) effective ensemble sizes.
        method (str): "ols" or "tls".
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        np.ndarray: (n_trials, k) β of every trial.
    """
    rng = np.random.default_rng(seed)
    betas = np.empty((n_trials, X.shape[1]))
    for trial in range(n_trials):
        y_new, X_new = _perturbed(y, X, noise, n_eff, rng)
        betas[trial] = fit(y_new, X_new, covariance, n_eff, method)
    return betas


def fit_table(results, signal_names):
    """Several fits as one Dataset on (set, fit, signal), NaN where a set has no such signal.

    Args:
        results (dict): (set, method) -> {name: (k,) array}, e.g. {"beta": ..., "lower": ..., "upper": ...}.
        signal_names (dict): Set -> its k signal names, in the order of the arrays.

    Returns:
        xr.Dataset: One variable per name, the signals in the order of ``SIGNALS``; the dim of the methods is
        ``fit`` (``method`` is a keyword of ``sel``).
    """
    pieces = [xr.Dataset({name: ("signal", np.asarray(values)) for name, values in arrays.items()},
                         coords={"signal": list(signal_names[set_name])}).expand_dims(set=[set_name], fit=[method])
              for (set_name, method), arrays in results.items()]
    table = xr.merge(pieces, join="outer", compat="no_conflicts")
    order = [signal for signal in SIGNALS if signal in table["signal"].values]
    sets = list(dict.fromkeys(set_name for set_name, _ in results))
    methods = list(dict.fromkeys(method for _, method in results))
    return table.reindex(signal=order, set=sets, fit=methods)


# ---------------------------------------------------------------------------
# 6. The residual consistency test
# ---------------------------------------------------------------------------

def residual_statistic(y, X, beta, covariance):
    """rᵀ C⁻¹ r, with r = y - Xβ: how big what the fit leaves over is, in units of internal variability."""
    residual = y - X @ beta
    return float(residual @ np.linalg.solve(covariance, residual))


def residual_test(y, X, beta, covariance_fit, covariance_test, noise, n_eff, method="ols", n_trials=1000, seed=0):
    """Is what the fit leaves over consistent with internal variability? (after Allen & Tett, 1999)

    The observed statistic rᵀ C₂⁻¹ r is compared with its distribution when
    the fit is right: pseudo-observations Xβ̂ plus a noise sample, fitted the
    same way (with fingerprints perturbed as in ``monte_carlo``), each giving
    its own statistic. The p-value is the share of those at least as large,
    with the +1 correction; below 0.05 the fit is inconsistent and its β
    should not be trusted.

    Args:
        y, X, beta: The observations, the fingerprints and the fitted β.
        covariance_fit (np.ndarray): C₁, used in each refit.
        covariance_test (np.ndarray): C₂, used in the statistic.
        noise (np.ndarray): (s, n) noise samples to draw from (the second half).
        n_eff (np.ndarray): (k,) effective ensemble sizes.
        method (str): "ols" or "tls".
        n_trials (int): Trials.
        seed (int): Random seed.

    Returns:
        dict: ``statistic`` (observed), ``null`` (n_trials,) and ``pvalue``.
    """
    rng = np.random.default_rng(seed)
    observed = residual_statistic(y, X, beta, covariance_test)
    expected = X @ beta
    null = np.empty(n_trials)
    for trial in range(n_trials):
        y_new, X_new = _perturbed(expected, X, noise, n_eff, rng)
        beta_new = fit(y_new, X_new, covariance_fit, n_eff, method)
        null[trial] = residual_statistic(y_new, X_new, beta_new, covariance_test)
    pvalue = (1 + np.sum(null >= observed)) / (n_trials + 1)
    return {"statistic": observed, "null": null, "pvalue": float(pvalue)}
