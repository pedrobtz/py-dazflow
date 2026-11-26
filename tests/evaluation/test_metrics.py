"""Metrics: MASE on a hand-worked four-point example, and the other metrics by hand."""

import math

import numpy as np
import pytest

from dazflow import ConfigError, EvaluationError, evaluation

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


# Mutation survivors (Stage 13): each test below was missing an oracle.


@pytest.mark.parametrize("key", ["mae", "rmse", "mase", "coverage"])
def test_a_missing_value_given_as_none_is_refused_as_an_evaluation_error(key):
    # None becomes NaN only when the values are read as floats; as objects it
    # would reach numpy as a TypeError instead of the classed refusal.
    kwargs = {"mase": {"scale": 1.0}, "coverage": {"lower": [0.0, 0.0], "upper": [9.0, 9.0]}}
    m = metric(key)
    with pytest.raises(EvaluationError, match="missing"):
        m.score([1.0, None], [1.0, 2.0], **kwargs.get(key, {}))
    with pytest.raises(EvaluationError, match="missing"):
        m.score([1.0, 2.0], [None, 2.0], **kwargs.get(key, {}))


def test_a_missing_bound_given_as_none_is_refused():
    with pytest.raises(EvaluationError, match="missing"):
        metric("coverage").score([1.0, 2.0], [1.0, 2.0], lower=[0.0, None], upper=[9.0, 9.0])


def test_coverage_needs_both_bounds():
    with pytest.raises(EvaluationError, match="lower given, upper missing"):
        metric("coverage").score(ACTUAL, FORECAST, lower=ACTUAL)
    with pytest.raises(EvaluationError, match="lower missing, upper given"):
        metric("coverage").score(ACTUAL, FORECAST, upper=ACTUAL)


@pytest.mark.parametrize("season", [1, 2, 3])
def test_the_mase_scale_straddles_a_training_series_as_long_as_its_season(season):
    train = [float(v) for v in range(1, season + 2)]  # season + 1 values: one change
    assert metric("mase", season=season).scale(train) == pytest.approx(season, abs=1e-15)
    with pytest.raises(EvaluationError, match="longer than its season"):
        metric("mase", season=season).scale(train[:season])  # season values: none
    with pytest.raises(EvaluationError, match="longer than its season"):
        metric("mase", season=season).scale(train[: season - 1])


def test_the_mase_scale_refuses_a_table_and_a_missing_value():
    with pytest.raises(EvaluationError, match="longer than its season"):
        metric("mase").scale([[1.0, 2.0], [3.0, 5.0]])
    with pytest.raises(EvaluationError, match="longer than its season"):
        metric("mase").scale([1.0, None, 3.0, 4.0])


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan, [1.0, 1.0, 0.0, 1.0]])
def test_a_mase_scale_that_is_not_positive_and_finite_is_refused(scale):
    with pytest.raises(EvaluationError, match="positive"):
        metric("mase").score(ACTUAL, FORECAST, scale=scale)


def test_a_mase_scale_given_as_none_per_point_is_refused():
    with pytest.raises(EvaluationError, match="positive"):
        metric("mase").score(ACTUAL, FORECAST, scale=[1.0, None, 1.0, 1.0])


def test_a_component_takes_its_config_as_an_object_a_mapping_or_keywords():
    config = evaluation.MASE.Config(season=2)
    by_object = evaluation.MASE(config)
    by_mapping = evaluation.MASE({"season": 2})

    assert by_object.config is config
    assert by_mapping.config == config
    assert evaluation.MASE(season=2).config == config
    assert by_mapping.score(ACTUAL, FORECAST, train=TRAIN) == pytest.approx(1.5 / 2, abs=1e-15)
    with pytest.raises(ConfigError, match="a config and parameters"):
        evaluation.MASE({"season": 2}, season=3)


def test_registered_components_carry_their_key():
    for registry in (evaluation.metrics, evaluation.splitters):
        for key in sorted(registry.keys()):
            assert registry.get(key).key == key
