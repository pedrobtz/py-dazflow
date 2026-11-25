"""Rolling-origin evaluation: each fold refits and forecasts as a direct fit at that origin."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dazflow import EvaluationError, evaluation, models

INDEX = pd.period_range("2000Q1", periods=20, freq="Q")


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=20), index=INDEX, name="x")
    y = pd.Series(1.0 + 0.5 * x + rng.normal(scale=0.3, size=20), index=INDEX, name="y")
    return y, x.to_frame()


def test_naive_scores_equal_the_hand_computed_ones(data):
    y, _ = data
    ev = evaluation.evaluate(
        models.get("naive")(), y, splitter=evaluation.Expanding(initial=16, horizon=2, step=2)
    )
    v = y.to_numpy()
    # origins at positions 15 and 17; the naive forecast is the value at the origin
    errors = np.array([v[16] - v[15], v[17] - v[15], v[18] - v[17], v[19] - v[17]])
    scale = [np.mean(np.abs(np.diff(v[:16])))] * 2 + [np.mean(np.abs(np.diff(v[:18])))] * 2

    assert [r.origin for r in ev.rows] == ["2003Q4", "2003Q4", "2004Q2", "2004Q2"]
    assert [r.horizon for r in ev.rows] == [1, 2, 1, 2]
    assert ev.score("mae") == pytest.approx(np.mean(np.abs(errors)), abs=1e-14)
    assert ev.score("rmse") == pytest.approx(np.sqrt(np.mean(errors**2)), abs=1e-14)
    assert ev.score("mase") == pytest.approx(np.mean(np.abs(errors) / scale), abs=1e-14)
    assert ev.score("mae", horizon=2) == pytest.approx(np.mean(np.abs(errors[[1, 3]])), abs=1e-14)


def test_ols_forecasts_equal_direct_statsmodels_fits_at_each_origin(data):
    y, X = data
    ev = evaluation.evaluate(
        models.get("ols")(), y, X, splitter=evaluation.Rolling(window=12, horizon=3)
    )
    assert len(ev.folds) == 6
    for fold in ev.folds:
        tr, te = slice(*fold.train), slice(*fold.test)
        res = sm.OLS(y.iloc[tr], sm.add_constant(X.iloc[tr])).fit()
        frame = res.get_prediction(sm.add_constant(X.iloc[te], has_constant="add")).summary_frame(
            alpha=0.05
        )
        rows = [r for r in ev.rows if r.fold == fold.number]
        np.testing.assert_allclose([r.forecast for r in rows], frame["mean"], rtol=1e-12)
        np.testing.assert_allclose([r.lower[1] for r in rows], frame["obs_ci_lower"], rtol=1e-12)
        assert [r.actual for r in rows] == y.iloc[te].tolist()

    inside = [r.lower[1] <= r.actual <= r.upper[1] for r in ev.rows]
    assert ev.score("coverage") == pytest.approx(np.mean(inside), abs=1e-15)


def test_the_evaluation_is_data(data):
    y, _X = data
    ev = evaluation.evaluate("mean", y, splitter=evaluation.Expanding(initial=15))
    again = evaluation.Evaluation.model_validate_json(ev.model_dump_json())

    assert again == ev
    assert ev.spec == models.get("mean")().fit(y).spec
    assert evaluation.evaluate(ev.spec, y, splitter=evaluation.Expanding(initial=15)) == ev
    assert ev.splitter == "expanding"
    assert list(ev.frame().columns[-4:]) == ["lower_0.8", "upper_0.8", "lower_0.95", "upper_0.95"]


def test_a_coverage_level_that_is_not_forecast_is_refused(data):
    y, _ = data
    with pytest.raises(EvaluationError, match="not forecast"):
        evaluation.evaluate(
            models.get("naive")(),
            y,
            metrics=[evaluation.Coverage(level=0.9)],
            levels=(0.8, 0.95),
        )
