"""What fit and predict refuse, each with a ModelError (or ConfigError) naming the problem."""

import numpy as np
import pandas as pd
import pytest

from dazflow import ConfigError, ModelError, models


def ols():
    return models.get("ols")()


def test_fit_refuses_data_it_cannot_use(data):
    y, X = data
    gap = y.copy()
    gap.iloc[3] = np.nan
    with pytest.raises(ModelError, match="missing") as err:
        ols().fit(gap, X)
    assert "2000Q4" in err.value.observed
    with pytest.raises(ModelError, match="PeriodIndex"):
        ols().fit(y.reset_index(drop=True), X)
    with pytest.raises(ModelError, match="not a pandas Series"):
        ols().fit(y.to_numpy(), X)
    with pytest.raises(ModelError, match="target's periods"):
        ols().fit(y, X.iloc[1:])
    with pytest.raises(ModelError, match="target's name"):
        ols().fit(y, X.rename(columns={"gdp": "loss"}))
    bad = X.copy()
    bad.iloc[0, 1] = np.inf
    with pytest.raises(ModelError, match="non-finite values in the drivers"):
        ols().fit(y, bad)
    with pytest.raises(ModelError, match="at least"):
        ols().fit(y.iloc[:3], X.iloc[:3])
    with pytest.raises(ModelError, match="nothing to regress on"):
        models.get("ols")(constant=False).fit(y)


def test_a_form_without_drivers_refuses_them(data):
    y, X = data
    for key in ("naive", "seasonal_naive", "mean", "ets"):
        with pytest.raises(ModelError, match="takes no drivers"):
            models.get(key)().fit(y, X)


def test_binomial_glm_needs_a_share(data):
    y, X = data
    with pytest.raises(ModelError, match="between 0 and 1"):
        models.get("glm")().fit(y * 4, X)
    assert models.get("glm")(family="gaussian").fit(y * 4, X)


def test_configs_refuse_inconsistent_parameters():
    with pytest.raises(ConfigError, match="damped_trend needs a trend"):
        models.get("ets")(damped_trend=True)
    with pytest.raises(ConfigError, match="season length"):
        models.get("arimax")(seasonal_order=(1, 0, 0, 0))
    with pytest.raises(ConfigError):
        models.get("ets")(error="mul")
    with pytest.raises(ConfigError, match="config and parameters"):
        models.get("ols")(models.get("ols").Config(), constant=False)


def test_seasonal_forms_need_a_season(data):
    y, _ = data
    annual = pd.Series(
        y.to_numpy()[:30], index=pd.period_range("1990", periods=30, freq="Y"), name="loss"
    )
    assert models.get("seasonal_naive")().fit(annual).predict(2).mean == (annual.iloc[-1],) * 2
    with pytest.raises(ModelError, match="season of at least 2"):
        models.get("ets")(seasonal="add").fit(annual)
    with pytest.raises(ModelError, match="at least 6"):
        models.get("seasonal_naive")().fit(y.iloc[:5])


def test_predict_refuses_a_horizon_or_drivers_it_cannot_use(data):
    y, X = data
    fitted = ols().fit(y, X)
    future = pd.DataFrame(np.zeros((3, 2)), index=fitted.future_index(3), columns=["gdp", "rate"])
    for h in (0, -1, 1.5, True):
        with pytest.raises(ModelError, match="positive integer"):
            fitted.predict(h, future)
    with pytest.raises(ModelError, match="no future values"):
        fitted.predict(3)
    with pytest.raises(ModelError, match="do not cover the horizon"):
        fitted.predict(4, future)
    with pytest.raises(ModelError, match="lack a fitted driver"):
        fitted.predict(3, future[["gdp"]])
    with pytest.raises(ModelError, match="not on the forecast periods"):
        fitted.predict(3, future.set_axis(future.index + 1))
    with pytest.raises(ModelError, match="levels"):
        fitted.predict(3, future, levels=(0.8, 1.0))
    # by position is accepted, and the result is the same
    assert fitted.predict(3, future.reset_index(drop=True)) == fitted.predict(3, future)
    with pytest.raises(ModelError, match="without drivers"):
        models.get("naive")().fit(y).predict(3, future)


def test_a_forecast_is_data_with_a_pandas_view(data):
    y, _ = data
    fc = models.get("naive")().fit(y).predict(4, levels=(0.95, 0.8))

    assert fc.levels == (0.8, 0.95)
    assert models.Forecast.model_validate_json(fc.model_dump_json()) == fc
    frame = fc.frame()
    assert list(frame.columns) == ["mean", "lower_0.8", "upper_0.8", "lower_0.95", "upper_0.95"]
    assert frame.index.equals(pd.period_range("2020Q1", periods=4, freq="Q"))
    lo, up = fc.interval(0.95)
    assert lo.tolist() == list(fc.lower[1]) and up.tolist() == list(fc.upper[1])
    with pytest.raises(ModelError, match=r"no interval at level 0\.9"):
        fc.interval(0.9)
    with pytest.raises(ModelError):
        models.Forecast(form="x", frequency="Q-DEC", periods=("2020Q1",), mean=(1.0, 2.0))


def test_fit_notes_keep_statsmodels_warnings(data):
    y, X = data
    fitted = models.get("arimax")(order=(2, 0, 2), maxiter=2).fit(y, X)

    assert any("ConvergenceWarning" in n for n in fitted.notes), fitted.notes
