"""Out-of-domain input raises TransformError naming the series and the period."""

import numpy as np
import pandas as pd
import pytest

from dazflow import TransformError
from dazflow import transforms as tf


def series(values, name="gdp"):
    index = pd.period_range("2020Q1", periods=len(values), freq="Q")
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


@pytest.mark.parametrize(
    ("chain", "values", "period", "bad"),
    [
        (["log"], [1.0, 2.0, 0.0, 3.0], "2020Q3", "0.0"),
        (["log"], [-1.0, 2.0], "2020Q1", "-1.0"),
        (["logdiff"], [1.0, 0.0, 2.0], "2020Q2", "0.0"),
        (["boxcox(lmbda=0.5)"], [1.0, 2.0, 0.0], "2020Q3", "0.0"),
        (["logit"], [0.5, 1.0], "2020Q2", "1.0"),
        (["logit"], [0.0, 0.5], "2020Q1", "0.0"),
        (["logit"], [0.2, 1.5], "2020Q2", "1.5"),
        (["growth"], [1.0, 0.0, 2.0], "2020Q2", "0.0"),
        (["diff"], [1.0, np.inf], "2020Q2", "inf"),
    ],
)
def test_out_of_domain_names_the_series_and_period(chain, values, period, bad):
    with pytest.raises(TransformError) as info:
        tf.forward(series(values), chain)
    err = info.value
    assert err.series == "gdp"
    assert err.period == period
    assert "'gdp'" in str(err) and period in str(err) and bad in str(err)


def test_growth_accepts_a_zero_in_the_last_period():
    # The last value is never a denominator.
    y, _ = tf.forward(series([1.0, 2.0, 0.0]), ["growth"])
    assert y.iloc[-1] == -1.0


def test_logit_domain_is_checked_after_an_earlier_transform():
    # diff turns 0.2, 0.9 into 0.7: in (0, 1), so logit is fine; 0.1, 1.5 gives 1.4: not.
    tf.forward(series([0.2, 0.9]), ["diff", "logit"])
    with pytest.raises(TransformError) as info:
        tf.forward(series([0.1, 1.5]), ["diff", "logit"])
    assert info.value.period == "2020Q2"


def test_inverse_out_of_domain_names_the_period():
    y = series([1.0, -3.0])  # lambda * y + 1 = -0.5 at 2020Q2: no Box-Cox level gives it
    with pytest.raises(TransformError) as info:
        tf.inverse(y, ["boxcox(lmbda=0.5)"], history=y.iloc[:0])
    assert info.value.series == "gdp" and info.value.period == "2020Q2"


def test_missing_values_are_not_out_of_domain():
    y, _ = tf.forward(series([1.0, np.nan, 2.0]), ["log"])
    assert np.isnan(y.iloc[1])


def test_an_unresolved_parameter_cannot_be_inverted():
    with pytest.raises(TransformError, match="lmbda"):
        tf.inverse(series([1.0]), ["boxcox"], history=series([]))


def test_seasonal_period_needs_a_seasonal_frequency():
    x = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2000", periods=3, freq="Y"), name="a")
    with pytest.raises(TransformError, match="period"):
        tf.forward(x, ["seasonal_diff"])


def test_boxcox_cannot_estimate_lambda_from_a_constant_series():
    with pytest.raises(TransformError, match="lmbda"):
        tf.forward(series([2.0, 2.0, 2.0]), ["boxcox"])


def test_the_index_must_be_periods():
    with pytest.raises(TransformError, match="PeriodIndex"):
        tf.forward(pd.Series([1.0, 2.0], name="a"), ["log"])


# --- refusals ------------------------------------------------------------------


def test_text_values_and_malformed_parameters():
    from dazflow import ConfigError

    assert str(tf.parse("boxcox(lmbda=None)")) == "boxcox(bound=2.0)"
    with pytest.raises(ConfigError, match="lag"):
        tf.parse("diff(lag=abc)")
    with pytest.raises(TransformError, match="name=value"):
        tf.parse("diff(2)")


def test_a_single_transform_may_be_given_as_text():
    y, chain = tf.forward(series([1.0, 3.0]), "diff")
    assert y.iloc[1] == 2.0 and [str(s) for s in chain] == ["diff(lag=1)"]


@pytest.mark.parametrize(
    ("x", "match"),
    [
        ([1.0, 2.0], "not a pandas Series"),
        (
            pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2020Q2", "2020Q1"], freq="Q"), name="a"),
            "unsorted",
        ),
        (
            pd.Series(["a", "b"], index=pd.period_range("2020Q1", periods=2, freq="Q"), name="a"),
            "not numeric",
        ),
    ],
)
def test_input_that_is_not_a_sorted_numeric_period_series_is_refused(x, match):
    with pytest.raises(TransformError, match=match):
        tf.forward(x, ["log"])


def test_history_at_another_frequency_is_refused():
    months = pd.Series([1.0], index=pd.period_range("2020-01", periods=1, freq="M"), name="gdp")
    with pytest.raises(TransformError, match="frequencies"):
        tf.inverse(series([1.0]), ["diff"], history=months)


def test_apply_refuses_an_unknown_column_and_a_panel():
    import dazflow as dz

    ds = dz.Dataset(series([1.0, 2.0]).to_frame())
    with pytest.raises(TransformError, match="zz"):
        tf.apply(ds, "zz", ["log"])
    index = pd.MultiIndex.from_product(
        [["north", "south"], pd.period_range("2020Q1", periods=2, freq="Q")]
    )
    panel = dz.Dataset(pd.DataFrame({"gdp": [1.0, 2.0, 3.0, 4.0]}, index=index))
    with pytest.raises(TransformError, match="segment"):
        tf.apply(panel, "gdp", ["log"])
