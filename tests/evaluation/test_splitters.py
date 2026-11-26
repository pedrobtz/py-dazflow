"""Splitters: fold boundaries on a 20-period series equal folds listed by hand."""

import pandas as pd
import pytest

from dazflow import EvaluationError, evaluation

INDEX = pd.period_range("2000Q1", periods=20, freq="Q")  # 2000Q1 .. 2004Q4


def bounds(folds):
    """Each fold as (first train, last train, first test, last test) periods."""
    return [
        (
            str(INDEX[f.train[0]]),
            str(INDEX[f.train[1] - 1]),
            str(INDEX[f.test[0]]),
            str(INDEX[f.test[1] - 1]),
        )
        for f in folds
    ]


def split(key, **params):
    return evaluation.splitters.get(key)(**params).split(INDEX)


def test_expanding_folds_equal_the_hand_list():
    folds = split("expanding", initial=12, horizon=4, step=2)

    assert bounds(folds) == [
        ("2000Q1", "2002Q4", "2003Q1", "2003Q4"),
        ("2000Q1", "2003Q2", "2003Q3", "2004Q2"),
        ("2000Q1", "2003Q4", "2004Q1", "2004Q4"),
    ]
    assert [f.number for f in folds] == [1, 2, 3]
    assert [str(INDEX[f.origin]) for f in folds] == ["2002Q4", "2003Q2", "2003Q4"]


def test_rolling_folds_equal_the_hand_list():
    folds = split("rolling", window=8, horizon=4, step=2)

    assert bounds(folds) == [
        ("2000Q1", "2001Q4", "2002Q1", "2002Q4"),
        ("2000Q3", "2002Q2", "2002Q3", "2003Q2"),
        ("2001Q1", "2002Q4", "2003Q1", "2003Q4"),
        ("2001Q3", "2003Q2", "2003Q3", "2004Q2"),
        ("2002Q1", "2003Q4", "2004Q1", "2004Q4"),
    ]


def test_expanding_defaults_start_at_half_the_series_one_step_ahead():
    folds = split("expanding")

    assert bounds(folds) == [
        ("2000Q1", str(INDEX[t - 1]), str(INDEX[t]), str(INDEX[t])) for t in range(10, 20)
    ]


def test_folds_keeps_the_last_origins():
    assert bounds(split("expanding", initial=12, horizon=1, folds=3)) == [
        ("2000Q1", "2004Q1", "2004Q2", "2004Q2"),
        ("2000Q1", "2004Q2", "2004Q3", "2004Q3"),
        ("2000Q1", "2004Q3", "2004Q4", "2004Q4"),
    ]
    assert bounds(split("rolling", window=4, horizon=2, folds=1)) == [
        ("2003Q3", "2004Q2", "2004Q3", "2004Q4")
    ]


def test_the_last_test_period_is_the_last_period():
    # One period more of training than fits leaves no fold; one less leaves one.
    assert bounds(split("expanding", initial=16, horizon=4)) == [
        ("2000Q1", "2003Q4", "2004Q1", "2004Q4")
    ]
    with pytest.raises(EvaluationError, match="no fold"):
        split("expanding", initial=17, horizon=4)
    assert len(split("rolling", window=16, horizon=4)) == 1
    with pytest.raises(EvaluationError, match="no fold"):
        split("rolling", window=17, horizon=4)


def test_a_fold_never_trains_on_its_test_periods():
    for folds in (split("expanding", initial=5, horizon=3), split("rolling", window=5, horizon=3)):
        for f in folds:
            assert f.train[1] == f.test[0]
            assert f.test[1] - f.test[0] == 3


def test_the_index_must_be_periods():
    with pytest.raises(EvaluationError, match="PeriodIndex"):
        evaluation.splitters.get("expanding")().split(pd.RangeIndex(20))


def test_an_expanding_split_may_train_on_one_period():
    # initial is at least 1; the first origin then trains on the first period alone
    folds = evaluation.Expanding(initial=1).split(INDEX)
    assert (folds[0].train, folds[0].test) == ((0, 1), (1, 2))
    assert len(folds) == 19
    short = pd.period_range("2000Q1", periods=3, freq="Q")
    assert evaluation.Expanding().split(short)[0].train == (0, 1)  # half of 3 is 1


@pytest.mark.parametrize("key", ["expanding", "rolling"])
def test_a_splitter_that_leaves_no_fold_names_itself(key):
    params = {"initial": 20} if key == "expanding" else {"window": 20}
    with pytest.raises(EvaluationError, match=f"the {key} splitter leaves no fold"):
        split(key, **params)
