"""Known answers: OLS and ARIMAX recover the parameters of long simulated series.

The tolerance is set from each estimate's own standard error: every estimate
lies within Z standard errors of the true value. Z = 4 leaves a chance of
about 6e-5 per parameter that a correct estimator misses (two-sided normal),
while a wrong sign, a lost driver or an intercept on the wrong scale misses
by tens of standard errors. The series are seeded, so the outcome is fixed.
"""

import numpy as np
import pandas as pd
import pytest

from dazflow import models

N = 2000
Z = 4.0


def drivers(rng, n=N):
    index = pd.period_range("1700Q1", periods=n, freq="Q")
    return pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["x1", "x2"])


def within(fitted, truth):
    se = np.sqrt(np.diag(fitted.cov_params.to_numpy()))
    est = fitted.params[list(truth)].to_numpy()
    z = (
        np.abs(est - np.array(list(truth.values())))
        / se[[list(fitted.params.index).index(k) for k in truth]]
    )
    return dict(zip(truth, z, strict=True))


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_ols_recovers_its_coefficients(seed):
    rng = np.random.default_rng(seed)
    X = drivers(rng)
    truth = {"const": 1.0, "x1": 0.5, "x2": -0.3}
    y = truth["const"] + truth["x1"] * X["x1"] + truth["x2"] * X["x2"] + rng.normal(size=N)
    fitted = models.get("ols")().fit(y.rename("y"), X)

    z = within(fitted, truth)
    assert max(z.values()) < Z, z
    # the standard error itself matches theory: sigma / sqrt(n var(x)) = 1 / sqrt(2000)
    assert np.sqrt(fitted.cov_params.loc["x1", "x1"]) == pytest.approx(1 / np.sqrt(N), rel=0.1)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_arimax_recovers_regression_with_ar1_errors(seed):
    rng = np.random.default_rng(seed)
    X = drivers(rng)
    mu, beta, phi, sigma2 = 0.5, 0.8, 0.6, 1.0
    e = rng.normal(scale=np.sqrt(sigma2), size=N)
    u = np.zeros(N)
    u[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, N):
        u[t] = phi * u[t - 1] + e[t]
    y = pd.Series(mu + beta * X["x1"].to_numpy() + u, index=X.index, name="y")
    fitted = models.get("arimax")(order=(1, 0, 0), trend="c").fit(y, X[["x1"]])

    # SARIMAX puts the constant in the AR equation: intercept = mu (1 - phi)
    truth = {"intercept": mu * (1 - phi), "x1": beta, "ar.L1": phi, "sigma2": sigma2}
    z = within(fitted, truth)
    assert max(z.values()) < Z, z


def test_a_wrong_answer_misses_by_many_standard_errors():
    # the bound is not loose: a coefficient off by 0.15 is about 6.7 standard errors out
    rng = np.random.default_rng(11)
    X = drivers(rng)
    y = 1.0 + 0.5 * X["x1"] - 0.3 * X["x2"] + rng.normal(size=N)
    fitted = models.get("ols")().fit(y.rename("y"), X)

    assert within(fitted, {"x1": 0.65})["x1"] > Z
