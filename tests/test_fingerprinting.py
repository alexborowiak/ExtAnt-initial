"""The optimal fingerprinting steps, on cases with a known answer (the handover's unit tests 1-4, and more)."""

import numpy as np
import pytest
import xarray as xr

from extant import fingerprinting as fp


def test_toy_exact_recovery():
    #(c): 2 regions x 2 seasons, read region by region: Peninsula DJF, JJA, East Antarctica DJF, JJA
    y = np.array([0.4, 0.4, 0.1, 0.2])
    X = np.column_stack([[0.2, 0.4, 0.1, 0.2], [0.2, 0.0, 0.0, 0.0]])
    beta = fp.fit_ols(y, X, np.eye(4))
    np.testing.assert_allclose(beta, [1, 1], atol=1e-12)
    #(c): No noise anywhere, so TLS finds the same plane
    np.testing.assert_allclose(fp.fit_tls(y, X, np.eye(4), [10, 10]), [1, 1], atol=1e-10)


def test_weighted_single_forcing():
    y, X, C = np.array([2.0, 1.0]), np.array([[1.0], [1.0]]), np.diag([4.0, 1.0])
    #(c): Xᵀ C⁻¹ y = 2/4 + 1 = 1.5 and Xᵀ C⁻¹ X = 1/4 + 1 = 1.25
    np.testing.assert_allclose(fp.fit_ols(y, X, C), [1.2])
    np.testing.assert_allclose(fp.ols_covariance(X, C, C), [[0.8]])


def test_flatten_round_trip():
    table = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("region", "season"),
                         coords={"region": ["Peninsula", "East"], "season": ["DJF", "MAM", "JJA"]})
    vector = fp.flatten(table)
    #(c): Region-major: every season of the first region before the second
    assert list(vector.values) == [0, 1, 2, 3, 4, 5]
    assert list(vector["region"].values[:3]) == ["Peninsula"] * 3
    back = fp.unflatten(vector).sel(region=table["region"], season=table["season"])
    xr.testing.assert_identical(back, table)


def _synthetic_problem(rng, n=12, k=2, beta=(1.0, 0.5), n_eff=(20.0, 5.0)):
    X_true = rng.normal(size=(n, k)) + 1.0
    L = rng.normal(size=(n, n)) * 0.3
    C = L @ L.T / n + 0.5 * np.eye(n)
    chol = np.linalg.cholesky(C)
    return X_true, C, chol, np.asarray(beta), np.asarray(n_eff)


def test_monte_carlo_coverage():
    #(c): y = X β + noise from a known C, again and again: the 90% range should hold the true β about 90% of the time
    rng = np.random.default_rng(1)
    X, C, chol, beta_true, _ = _synthetic_problem(rng)
    noise_pool = (chol @ rng.normal(size=(C.shape[0], 4000))).T
    n_eff = np.array([1e8, 1e8])
    hits = []
    for _ in range(200):
        y = X @ beta_true + chol @ rng.normal(size=C.shape[0])
        beta = fp.fit_ols(y, X, C)
        draws = fp.monte_carlo(y, X, C, noise_pool, n_eff, n_trials=300, seed=int(rng.integers(1e9)))
        #(c): The draws scatter about β̂ as β̂ scatters about the truth, so their 5-95% range is the 90% range
        low, high = np.percentile(draws, [5, 95], axis=0)
        hits.append((low <= beta_true) & (beta_true <= high))
    coverage = np.mean(hits, axis=0)
    assert np.all((coverage > 0.82) & (coverage < 0.97))


def test_tls_less_biased_than_ols_with_noisy_fingerprints():
    rng = np.random.default_rng(2)
    n, n_eff = 30, 3.0
    X_true = rng.normal(size=(n, 1)) * 0.6
    ols, tls = [], []
    for _ in range(400):
        y = X_true[:, 0] + rng.normal(size=n)
        X = X_true + rng.normal(size=(n, 1)) / np.sqrt(n_eff)
        ols.append(fp.fit_ols(y, X, np.eye(n))[0])
        tls.append(fp.fit_tls(y, X, np.eye(n), [n_eff])[0])
    #(c): OLS is diluted towards 0; TLS is about right (its median, as TLS has heavy tails)
    assert np.median(ols) < 0.8
    assert abs(np.median(tls) - 1) < abs(np.median(ols) - 1)
    assert abs(np.median(tls) - 1) < 0.15


def test_ledoit_wolf():
    rng = np.random.default_rng(3)
    #(c): Fewer samples than entries: the sample covariance is singular, the shrunk one is not
    few = rng.normal(size=(10, 20))
    covariance, shrinkage = fp.ledoit_wolf(few)
    assert 0 < shrinkage <= 1
    assert np.linalg.eigvalsh(covariance).min() > 0
    #(c): Many samples of a strongly correlated truth: hardly any shrinkage, and close to the truth
    truth = np.array([[1.0, 0.9], [0.9, 1.0]])
    many = rng.multivariate_normal([0, 0], truth, size=20000)
    covariance, shrinkage = fp.ledoit_wolf(many)
    assert shrinkage < 0.01
    np.testing.assert_allclose(covariance, truth, atol=0.03)
    #(c): The trace is kept: shrinking moves weight between the entries, it does not add any
    np.testing.assert_allclose(np.trace(covariance), np.trace(many.T @ many / len(many)))


def test_inverse_sqrt_whitens():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(5, 5))
    C = A @ A.T + np.eye(5)
    P = fp.inverse_sqrt(C)
    np.testing.assert_allclose(P @ C @ P.T, np.eye(5), atol=1e-10)


def test_fingerprints_and_effective_members():
    means = xr.DataArray(
        np.array([[[1.0, 2.0], [0.5, 0.5], [0.1, 0.1]], [[2.0, 4.0], [1.5, 1.5], [np.nan, np.nan]]]),
        dims=("model", "experiment", "entry"),
        coords={"model": ["A", "B"], "experiment": ["historical", "hist-nat", "hist-totalO3"]})
    members = xr.DataArray([[10.0, 20.0, 5.0], [40.0, 40.0, np.nan]], dims=("model", "experiment"),
                           coords=means["model"].coords | means["experiment"].coords)
    signals = {"ANT": {"historical": 1, "hist-nat": -1}, "O3": {"hist-totalO3": 1}}
    prints = fp.model_fingerprints(means, signals)
    np.testing.assert_allclose(prints.sel(signal="ANT").values, [[0.5, 1.5], [0.5, 2.5]])
    #(c): Model B has no hist-totalO3, so no ozone fingerprint
    assert np.isnan(prints.sel(signal="O3", model="B")).all()
    n_eff = fp.effective_members(members, signals)
    #(c): ANT: 1/m_eff is 1/10 + 1/20 for A and 1/40 + 1/40 for B; two models, so n_eff = 4 / (0.15 + 0.05)
    np.testing.assert_allclose(float(n_eff.sel(signal="ANT")), 4 / 0.2)
    #(c): O3: only model A has it, so n_eff is its 5 members
    np.testing.assert_allclose(float(n_eff.sel(signal="O3")), 5)


def test_ensemble_anomalies_and_pool():
    rng = np.random.default_rng(5)
    nodes = {f"{model}/{experiment}": xr.Dataset({"tas": (("member", "entry"), rng.normal(size=(m, 3)))},
                                                  coords={"member": [f"r{i}" for i in range(m)]})
             for model, experiment, m in [("A", "historical", 4), ("A", "hist-nat", 3), ("B", "historical", 5)]}
    tree = xr.DataTree.from_dict(nodes).map_over_datasets(fp.ensemble_anomalies)
    pooled = fp.pool_samples(tree, "tas")
    assert pooled.sizes == {"sample": 12, "entry": 3}
    #(c): Each ensemble's anomalies average to 0, and the halves alternate within it
    np.testing.assert_allclose(pooled.groupby(pooled["model"] + pooled["experiment"]).mean().values, 0, atol=1e-12)
    assert list(pooled["half"].values[:4]) == [0, 1, 0, 1]


def test_residual_test_detects_a_wrong_pattern():
    rng = np.random.default_rng(6)
    n = 16
    X = rng.normal(size=(n, 1))
    noise = rng.normal(size=(2000, n))
    n_eff = [1e8]
    right, wrong = [], []
    for _ in range(40):
        #(c): Observations the fingerprint explains, plus noise: consistent, so p is uniform
        y_right = 0.8 * X[:, 0] + rng.normal(size=n)
        beta = fp.fit_ols(y_right, X, np.eye(n))
        right.append(fp.residual_test(y_right, X, beta, np.eye(n), np.eye(n), noise, n_eff, n_trials=200)["pvalue"])
        #(c): Observations with a pattern the fingerprint lacks: inconsistent
        y_wrong = y_right + 4 * np.sin(np.arange(n))
        beta = fp.fit_ols(y_wrong, X, np.eye(n))
        wrong.append(fp.residual_test(y_wrong, X, beta, np.eye(n), np.eye(n), noise, n_eff, n_trials=200)["pvalue"])
    assert np.mean(np.array(right) < 0.05) < 0.2
    assert np.all(np.array(wrong) < 0.01)


@pytest.mark.parametrize("method", ["ols", "tls"])
def test_fit_by_name(method):
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(fp.fit(y, X, np.eye(3), [100.0], method), [1.0], atol=1e-10)
