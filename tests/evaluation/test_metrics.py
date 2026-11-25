"""Metrics: MASE on a hand-worked four-point example, and the other metrics by hand."""

import math

import numpy as np
import pytest

from dazflow import EvaluationError, evaluation

TRAIN = [2.0, 4.0, 3.0, 7.0]  # naive changes 2, 1, 4: in-sample naive MAE 7/3
ACTUAL = [5.0, 6.0, 4.0, 8.0]
FORECAST = [4.0, 6.0, 6.0, 5.0]  # errors 1, 0, -2, 3: MAE 6/4


def metric(key, **params):
    return evaluation.metrics.get(key)(**params)


def test_mase_on_the_hand_worked_example():
    # MASE = (6/4) / (7/3) = 9/14
    assert metric("mase").score(ACTUAL, FORECAST, train=TRAIN) == pytest.approx(9 / 14, abs=1e-15)
    assert metric("mase").scale(TRAIN) == pytest.approx(7 / 3, abs=1e-15)


def test_seasonal_mase_scales_by_the_seasonal_naive_error():
    # season 2: changes 3-2 = 1 and 7-4 = 3, mean 2
    assert metric("mase", season=2).score(ACTUAL, FORECAST, train=TRAIN) == pytest.approx(
        1.5 / 2, abs=1e-15
    )


def test_mase_with_a_scale_per_point():
    # the rolling evaluation divides each error by its own fold's scale
    got = metric("mase").score(ACTUAL, FORECAST, scale=[1.0, 1.0, 2.0, 3.0])
    assert got == pytest.approx((1 + 0 + 1 + 1) / 4, abs=1e-15)


def test_mae_and_rmse_by_hand():
    assert metric("mae").score(ACTUAL, FORECAST) == pytest.approx(1.5, abs=1e-15)
    assert metric("rmse").score(ACTUAL, FORECAST) == pytest.approx(math.sqrt(14 / 4), abs=1e-15)


def test_coverage_counts_the_bounds_as_inside():
    lower = [4.0, 6.0, 4.5, 7.0]
    upper = [5.0, 7.0, 6.0, 9.0]  # 5 on the bound, 6 on the bound, 4 below, 8 inside
    assert metric("coverage").score(ACTUAL, FORECAST, lower=lower, upper=upper) == 0.75


def test_metrics_refuse_what_they_cannot_score():
    with pytest.raises(EvaluationError, match="scale"):
        metric("mase").score(ACTUAL, FORECAST)
    with pytest.raises(EvaluationError, match="zero"):
        metric("mase").score(ACTUAL, FORECAST, train=[1.0, 1.0, 1.0])
    with pytest.raises(EvaluationError, match="interval"):
        metric("coverage").score(ACTUAL, FORECAST)
    with pytest.raises(EvaluationError, match="lengths"):
        metric("mae").score(ACTUAL, FORECAST[:3])
    with pytest.raises(EvaluationError, match="no value"):
        metric("rmse").score([], [])
    with pytest.raises(EvaluationError, match="missing"):
        metric("mae").score([1.0, np.nan], [1.0, 2.0])


def test_every_metric_key_is_registered():
    assert set(evaluation.metrics.keys()) >= {"mae", "rmse", "mase", "coverage"}
    assert set(evaluation.splitters.keys()) >= {"expanding", "rolling"}
