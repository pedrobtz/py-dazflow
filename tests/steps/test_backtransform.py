"""Forecasts back on the target's scale: smearing for logs, simulation for logit, exact for linear chains.

Oracles: the log-normal mean ``exp(mu + sigma^2 / 2)``; the logit-normal mean
by Gauss-Hermite quadrature (scipy's nodes, not the step's draws); the
Monte Carlo error of the average over replications; hand arithmetic for
the intervals of a differenced forecast (independent errors add in
variance); `numpy.exp` of the bounds for a stateless chain; the
log-normal mean of a cumulated path for a chain with a lag.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import dazflow as dz
from dazflow import ForecastError
from dazflow.steps import forecast

REPS = 200
N = 200


def fit(z, x, x_new, h=1):
    """OLS of `z` on `x`, forecast `h` periods with the driver at `x_new`."""
    index = pd.period_range("1970Q1", periods=len(z), freq="Q")
    y = pd.Series(z, index=index, name="y")
    X = pd.DataFrame({"x": x}, index=index)
    fitted = dz.models.get("ols")().fit(y, X)
    return fitted, fitted.predict(h, pd.DataFrame({"x": np.full(h, x_new)}), levels=(0.8, 0.95))


def replicate(link, sigma, correction, seed=0, **kw):
    """The corrected and the plain back-transformed mean at x = 1, over REPS simulated samples."""
    rng = np.random.default_rng(seed)
    corrected, plain = [], []
    for r in range(REPS):
        x = rng.normal(size=N)
        z = 0.5 + 0.8 * x + sigma * rng.normal(size=N)
        fitted, fc = fit(z, x, 1.0)
        out = forecast.back_transform(
            fc, [link], residuals=fitted.residuals, correction=correction, seed=r, **kw
        )
        none = forecast.back_transform(fc, [link], correction="none")
        corrected.append(out.forecast.mean[0])
        plain.append(none.forecast.mean[0])
    return np.array(corrected), np.array(plain)


def within(values, truth, k=3.0):
    """|mean - truth| below k Monte Carlo standard errors of the mean over the replications."""
    se = values.std(ddof=1) / np.sqrt(len(values))
    return abs(values.mean() - truth) < k * se, abs(values.mean() - truth) / se


def test_smearing_recovers_the_log_normal_mean_and_the_plain_inverse_does_not():
    sigma = 0.5
    truth = np.exp(0.5 + 0.8 + sigma**2 / 2)
    corrected, plain = replicate("log", sigma, "smearing")
    ok, z = within(corrected, truth)
    assert ok, f"smearing is {z:.1f} Monte Carlo errors from the true mean"
    bad, z = within(plain, truth)
    assert not bad, f"the plain inverse is only {z:.1f} Monte Carlo errors away"
    assert plain.mean() < truth  # exp(E[log y]) is the median, below the mean


def test_smearing_is_duans_factor_for_a_log():
    rng = np.random.default_rng(3)
    x = rng.normal(size=N)
    fitted, fc = fit(0.5 + 0.8 * x + 0.5 * rng.normal(size=N), x, 0.3, h=3)
    out = forecast.back_transform(fc, ["log"], residuals=fitted.residuals, correction="smearing")
    factor = np.mean(np.exp(fitted.residuals.to_numpy()))
    np.testing.assert_allclose(out.forecast.mean, np.exp(fc.mean) * factor, rtol=1e-12)
    assert out.correction == "smearing" and out.intervals == "inverse"
    assert out.draws is None and out.seed is None


def logit_normal_mean(mu, sigma):
    nodes, weights = np.polynomial.hermite_e.hermegauss(120)
    return float(np.sum(weights * expit(mu + sigma * nodes)) / np.sqrt(2 * np.pi))


def test_simulation_recovers_the_logit_normal_mean_and_the_plain_inverse_does_not():
    sigma = 0.8
    truth = logit_normal_mean(1.3, sigma)
    corrected, plain = replicate("logit", sigma, "simulation", draws=2000)
    ok, z = within(corrected, truth)
    assert ok, f"the simulation is {z:.1f} Monte Carlo errors from the true mean"
    bad, z = within(plain, truth)
    assert not bad, f"the plain inverse is only {z:.1f} Monte Carlo errors away"
    assert plain.mean() > truth  # expit is concave above 0: the median is above the mean


def test_the_logit_quadrature_oracle_is_right():
    # Checked against a brute-force sample, so a wrong oracle cannot pass the test above.
    rng = np.random.default_rng(0)
    draws = expit(1.3 + 0.8 * rng.normal(size=2_000_000))
    assert logit_normal_mean(1.3, 0.8) == pytest.approx(draws.mean(), abs=5 * draws.std() / 1414)


def test_the_simulation_repeats_for_a_seed_and_moves_with_it():
    rng = np.random.default_rng(5)
    x = rng.normal(size=N)
    _, fc = fit(0.2 + 0.5 * x + 0.3 * rng.normal(size=N), x, 1.0)
    a = forecast.back_transform(fc, ["logit"], correction="simulation", draws=500, seed=7)
    b = forecast.back_transform(fc, ["logit"], correction="simulation", draws=500, seed=7)
    c = forecast.back_transform(fc, ["logit"], correction="simulation", draws=500, seed=8)
    assert a == b
    assert a.forecast.mean != c.forecast.mean
    assert a.draws == 500 and a.seed == 7 and a.correction == "simulation"


def test_a_stateless_chain_inverts_the_bounds():
    rng = np.random.default_rng(1)
    x = rng.normal(size=N)
    fitted, fc = fit(0.5 + 0.8 * x + 0.5 * rng.normal(size=N), x, 0.0, h=2)
    out = forecast.back_transform(fc, ["log"], residuals=fitted.residuals)
    for i in range(len(fc.levels)):
        np.testing.assert_allclose(out.forecast.lower[i], np.exp(fc.lower[i]), rtol=1e-12)
        np.testing.assert_allclose(out.forecast.upper[i], np.exp(fc.upper[i]), rtol=1e-12)
    assert out.correction == "smearing"  # auto, for a log


def test_auto_picks_simulation_for_logit_and_nothing_for_a_linear_chain():
    fc = flat(4)
    assert forecast.back_transform(fc, ["logit"]).correction == "simulation"
    hist = history()
    out = forecast.back_transform(fc, ["diff(lag=1)"], hist)
    assert out.correction == "none" and out.intervals == "linear"
    same = forecast.back_transform(fc, [])
    assert same.correction == "none" and same.intervals == "model"
    assert same.forecast.mean == fc.mean and same.forecast.lower == fc.lower


def flat(h, mean=0.1, half=(0.2, 0.4)):
    """A model-scale forecast with a constant mean and interval half-widths at 80% and 95%."""
    periods = tuple(str(p) for p in pd.period_range("2020Q1", periods=h, freq="Q"))
    m = tuple(float(mean + 0.01 * i) for i in range(h))
    return dz.models.Forecast(
        form="ols",
        frequency="Q-DEC",
        periods=periods,
        mean=m,
        levels=(0.8, 0.95),
        lower=tuple(tuple(v - w for v in m) for w in half),
        upper=tuple(tuple(v + w for v in m) for w in half),
    )


def history(last=5.0):
    index = pd.period_range("2017Q1", periods=12, freq="Q")
    return pd.Series(np.linspace(1.0, last, 12), index=index, name="y")


def test_a_differenced_forecast_cumulates_the_mean_and_adds_the_variances():
    fc = flat(4)
    out = forecast.back_transform(fc, ["diff(lag=1)"], history())
    mean = 5.0 + np.cumsum(fc.mean)
    np.testing.assert_allclose(out.forecast.mean, mean, rtol=1e-14)
    for i, half in enumerate((0.2, 0.4)):
        width = half * np.sqrt(np.arange(1, 5))  # independent errors: sd grows with sqrt(h)
        np.testing.assert_allclose(out.forecast.lower[i], mean - width, rtol=1e-13)
        np.testing.assert_allclose(out.forecast.upper[i], mean + width, rtol=1e-13)


def test_a_seasonal_difference_adds_only_the_same_quarters_errors():
    fc = flat(6)
    hist = history()
    out = forecast.back_transform(fc, ["seasonal_diff(period=4)"], hist)
    base = hist.to_numpy()[-4:]
    mean = np.array([base[0], base[1], base[2], base[3], 0, 0]) + np.array(fc.mean)
    mean[4] += mean[0]
    mean[5] += mean[1]
    np.testing.assert_allclose(out.forecast.mean, mean, rtol=1e-14)
    k = np.array([1, 1, 1, 1, 2, 2])
    np.testing.assert_allclose(out.forecast.upper[1], mean + 0.4 * np.sqrt(k), rtol=1e-13)


def test_a_log_with_a_difference_is_simulated_to_the_log_normal_mean():
    fc = flat(4, mean=0.02, half=(0.05, 0.08))
    hist = history(last=2.0)
    out = forecast.back_transform(fc, ["log", "diff(lag=1)"], hist, draws=40_000, seed=3)
    sigma = 0.08 / 1.959963984540054
    h = np.arange(1, 5)
    truth = 2.0 * np.exp(np.cumsum(fc.mean) + h * sigma**2 / 2)
    se = truth * np.sqrt(np.exp(h * sigma**2) - 1) / np.sqrt(40_000)
    assert np.all(np.abs(np.array(out.forecast.mean) - truth) < 4 * se)
    assert out.correction == "simulation" and out.intervals == "simulated"
    # The simulated 95% bounds are the log-normal quantiles, to Monte Carlo error.
    lo = 2.0 * np.exp(np.cumsum(fc.mean) - 1.959963984540054 * sigma * np.sqrt(h))
    np.testing.assert_allclose(out.forecast.lower[1], lo, rtol=5e-3)


def test_a_correction_that_does_not_fit_the_chain_is_refused():
    fc = flat(4)
    with pytest.raises(ForecastError, match="linear"):
        forecast.back_transform(fc, ["diff(lag=1)"], history(), correction="smearing")
    with pytest.raises(ForecastError, match="linear"):
        forecast.back_transform(fc, [], correction="simulation")
    with pytest.raises(ForecastError, match="smearing"):
        forecast.back_transform(fc, ["log", "diff(lag=1)"], history(), correction="smearing")
    with pytest.raises(ForecastError, match="residuals"):
        forecast.back_transform(fc, ["log"], correction="smearing")
