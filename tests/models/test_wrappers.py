"""Wrapper equality: every form's parameters, residuals and forecasts equal the direct
statsmodels call on the same data (benchmarks: the statsmodels model each one equals)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from dazflow import models

LEVELS = (0.8, 0.95)
H = 6
RTOL = 1e-10  # the same computation on the same data; 1e-10 allows for summation order


def close(a, b, rtol=RTOL, atol=1e-12):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol)


def rounded(values):
    """statsmodels' start values at 10 significant digits, as the wrappers start (D36)."""
    return np.array([float(f"{v:.10g}") for v in np.asarray(values, float)])


def layout(X, order):
    """`X` with the same values held row-major ("C") or column-major ("F")."""
    values = np.asarray(X.to_numpy(dtype=float), order=order)
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def assert_paths(fc, frame_at, lo_col, up_col):
    """`fc` equals the statsmodels summary frames at every level."""
    for i, lv in enumerate(LEVELS):
        frame = frame_at(1 - lv)
        close(fc.lower[i], frame[lo_col])
        close(fc.upper[i], frame[up_col])


def future(X, h=H):
    index = pd.period_range(X.index[-1] + 1, periods=h, freq=X.index.freq)
    rng = np.random.default_rng(99)
    return pd.DataFrame(rng.normal(size=(h, X.shape[1])), index=index, columns=X.columns)


def test_naive_equals_a_random_walk_sarimax(data):
    y, _ = data
    fitted = models.get("naive")().fit(y)
    fc = fitted.predict(H, levels=LEVELS)
    # ARIMA(0,1,0) with the scale concentrated out: no parameter to optimise
    res = SARIMAX(y, order=(0, 1, 0), concentrate_scale=True).filter(np.array([]))
    pred = res.get_forecast(H)

    close(fitted.params["sigma2"], res.scale)
    close(fitted.residuals, y.diff().iloc[1:])
    close(fc.mean, pred.predicted_mean)
    assert fc.periods == tuple(str(p) for p in pred.predicted_mean.index)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "mean_ci_lower", "mean_ci_upper")


def test_seasonal_naive_equals_a_seasonal_random_walk_sarimax(data):
    y, _ = data
    fitted = models.get("seasonal_naive")().fit(y)  # quarterly: season 4
    fc = fitted.predict(H, levels=LEVELS)
    res = SARIMAX(y, order=(0, 0, 0), seasonal_order=(0, 1, 0, 4), concentrate_scale=True).filter(
        np.array([])
    )
    pred = res.get_forecast(H)

    close(fitted.params["sigma2"], res.scale)
    close(fitted.residuals, y.diff(4).iloc[4:])
    close(fc.mean, pred.predicted_mean)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "mean_ci_lower", "mean_ci_upper")


def test_historic_mean_equals_ols_on_a_constant(data):
    y, _ = data
    fitted = models.get("mean")().fit(y)
    fc = fitted.predict(H, levels=LEVELS)
    res = sm.OLS(y, np.ones(len(y))).fit()
    pred = res.get_prediction(np.ones((H, 1)))

    close(fitted.params, res.params)
    close(fitted.cov_params, res.cov_params())
    close(fitted.residuals, res.resid)
    close(fc.mean, pred.predicted_mean)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "obs_ci_lower", "obs_ci_upper")


@pytest.mark.parametrize("cov_type", ["nonrobust", "HAC"])
def test_ols_equals_statsmodels_ols(data, cov_type):
    y, X = data
    fitted = models.get("ols")(cov_type=cov_type).fit(y, X)
    Xf = future(X)
    fc = fitted.predict(H, Xf, levels=LEVELS)
    kwargs = {"cov_type": cov_type}
    if cov_type == "HAC":
        kwargs["cov_kwds"] = {"maxlags": int(np.floor(4 * (len(y) / 100) ** (2 / 9)))}
    res = sm.OLS(y, sm.add_constant(X)).fit(**kwargs)
    pred = res.get_prediction(sm.add_constant(Xf, has_constant="add"))

    assert list(fitted.params.index) == ["const", "gdp", "rate"]
    close(fitted.params, res.params)
    close(fitted.cov_params, res.cov_params())
    close(fitted.residuals, res.resid)
    close(fc.mean, pred.predicted_mean)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "obs_ci_lower", "obs_ci_upper")


def test_glm_equals_statsmodels_binomial_glm(data):
    y, X = data
    fitted = models.get("glm")().fit(y, X)
    Xf = future(X)
    fc = fitted.predict(H, Xf, levels=LEVELS)
    res = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit(scale="X2")
    frame = res.get_prediction(sm.add_constant(Xf, has_constant="add")).summary_frame()
    mu = frame["mean"].to_numpy()
    sd = np.sqrt(res.scale * mu * (1 - mu) + frame["mean_se"].to_numpy() ** 2)

    close(fitted.params, res.params)
    close(fitted.cov_params, res.cov_params())
    close(fitted.residuals, res.resid_response)
    close(fc.mean, mu)
    for i, z in enumerate([1.2815515655446004, 1.959963984540054]):  # norm.ppf(0.9), (0.975)
        close(fc.lower[i], np.clip(mu - z * sd, 0, 1))
        close(fc.upper[i], np.clip(mu + z * sd, 0, 1))


def test_arimax_equals_statsmodels_sarimax(data):
    y, X = data
    fitted = models.get("arimax")(order=(1, 0, 0), trend="c").fit(y, X)
    Xf = future(X)
    fc = fitted.predict(H, Xf, levels=LEVELS)
    # The wrappers hold drivers row-major (#16); the direct call gets the same layout.
    model = SARIMAX(y, exog=layout(X, "C"), order=(1, 0, 0), trend="c")
    res = model.fit(start_params=rounded(model.start_params), disp=False, maxiter=200)
    pred = res.get_forecast(H, exog=Xf)

    assert list(fitted.params.index) == list(res.params.index)
    close(fitted.params, res.params)
    close(fitted.cov_params, res.cov_params())
    close(fitted.residuals, res.resid)
    close(fc.mean, pred.predicted_mean)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "mean_ci_lower", "mean_ci_upper")


def test_integrated_arimax_residuals_leave_out_the_diffuse_periods(data):
    y, X = data
    fitted = models.get("arimax")(order=(0, 1, 1), trend="n").fit(y, X)
    model = SARIMAX(y, exog=layout(X, "C"), order=(0, 1, 1), trend="n")
    res = model.fit(start_params=rounded(model.start_params), disp=False, maxiter=200)
    burn = max(res.loglikelihood_burn, res.nobs_diffuse)

    assert burn == 1
    close(fitted.params, res.params)
    close(fitted.residuals, res.resid.iloc[burn:])


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"trend": "add"},
        {"trend": "add", "damped_trend": True, "seasonal": "add"},
    ],
)
def test_ets_equals_statsmodels_ets_model(data, config):
    y, _ = data
    fitted = models.get("ets")(**config).fit(y)
    fc = fitted.predict(H, levels=LEVELS)
    kwargs = {"error": "add", **config}
    if "seasonal" in config:
        kwargs["seasonal_periods"] = 4
    model = ETSModel(y, **kwargs)
    res = model.fit(start_params=rounded(model.start_params), disp=False, maxiter=1000)
    n = len(y)
    pred = res.get_prediction(start=n, end=n + H - 1)

    assert list(fitted.params.index) == list(model.param_names)
    close(fitted.params, res.params)
    close(fitted.residuals, y - res.fittedvalues)
    close(fc.mean, pred.predicted_mean)
    assert_paths(fc, lambda a: pred.summary_frame(alpha=a), "pi_lower", "pi_upper")


@pytest.mark.parametrize(
    ("form", "config"),
    [
        ("ols", {}),
        ("glm", {}),
        ("arimax", {"order": (1, 0, 0), "trend": "c"}),
        ("arimax", {"order": (0, 1, 1), "trend": "n"}),
    ],
)
def test_a_fit_depends_on_the_drivers_values_not_their_memory_layout(data, form, config):
    # #16: statsmodels' likelihood takes a BLAS path that depends on the exog
    # array's strides; on statsmodels 0.14.2 / numpy 1.26.4 a row-major and a
    # column-major copy of the same drivers moved an ARIMAX fit by up to 1e-4.
    y, X = data
    by_row = models.get(form)(**config).fit(y, layout(X, "C"))
    by_column = models.get(form)(**config).fit(y, layout(X, "F"))

    assert np.asarray(layout(X, "F").to_numpy()).flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(by_row.params, by_column.params)
    np.testing.assert_array_equal(by_row.cov_params, by_column.cov_params)
    np.testing.assert_array_equal(by_row.residuals, by_column.residuals)
