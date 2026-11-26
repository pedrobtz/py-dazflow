"""Oracles the mutation run found missing in `transforms/` (roadmap Stage 13, design D46).

Each test names, in its comment, the mutants it kills. The scheduled
`Mutation` workflow is the check that they stay killed.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from dazflow import TransformError
from dazflow import transforms as tf


def series(values, name="gdp", freq="Q"):
    index = pd.period_range("2020Q1" if freq == "Q" else "2020-01", periods=len(values), freq=freq)
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


@pytest.mark.parametrize("seed", range(5))
def test_the_estimated_boxcox_lambda_is_scipys_maximum_likelihood_estimate(seed):
    # BoxCox.resolve mutant 39 (the likelihood maximised, not minimised) and 34/46 (the method).
    rng = np.random.default_rng(seed)
    x = series(np.exp(rng.normal(1.0, 0.4, 40)))
    _, (spec,) = tf.forward(x, ["boxcox"])
    assert spec.params["lmbda"] == pytest.approx(scipy.stats.boxcox(x.to_numpy())[1], abs=1e-4)


def test_an_estimate_past_the_bound_stops_at_the_lower_bound():
    # BoxCox.resolve mutant 44: the bounds are (-bound, bound), not (bound, bound).
    x = series([1.0, 1.01, 1.02, 1.05, 50.0])  # scipy's MLE is far below -1
    assert scipy.stats.boxcox(x.to_numpy())[1] < -1
    _, (spec,) = tf.forward(x, ["boxcox(bound=1)"])
    assert spec.params["lmbda"] == pytest.approx(-1.0, abs=1e-4)


def test_two_distinct_values_are_enough_to_estimate_lambda_and_one_is_not():
    # BoxCox.resolve mutants 10 and 11: the bound is two values.
    _, (spec,) = tf.forward(series([1.0, 2.0]), ["boxcox"])
    assert -2 <= spec.params["lmbda"] <= 2
    with pytest.raises(TransformError, match="cannot estimate lmbda"):
        tf.forward(series([2.0]), ["boxcox"])
    with pytest.raises(TransformError, match="1 values, 1 distinct"):
        tf.forward(series([2.0, np.nan]), ["boxcox"])


@pytest.mark.parametrize("lmbda", [1.0, 0.5, -0.5])
def test_the_boxcox_inverse_domain_straddles_lambda_times_value_plus_one_at_zero(lmbda):
    # BoxCox.inverse mutants 3 (lmbda 1 skipped the check) and 23 (<= 0 read as < 0).
    edge = -1.0 / lmbda  # lmbda * edge + 1 == 0, exactly for these lambdas
    inside = edge + (1e-9 if lmbda > 0 else -1e-9)
    chain = [f"boxcox(lmbda={lmbda!r})"]
    assert np.isfinite(tf.inverse(series([inside]), chain)).all()
    with pytest.raises(TransformError, match="the inverse of boxcox"):
        tf.inverse(series([edge]), chain)


@pytest.mark.parametrize(
    ("chain", "values", "rule"),
    [
        (["log"], [1.0, 0.0], "values are greater than 0"),
        (["logdiff"], [1.0, -2.0], "values are greater than 0"),
        (["boxcox(lmbda=0.5)"], [1.0, 0.0], "values are greater than 0"),
        (["boxcox"], [1.0, 0.0], "values are greater than 0"),
        (["logit"], [0.5, 1.0], "0 < value < 1"),
        (["growth"], [0.0, 1.0], "the earlier value is not 0"),
        (["diff"], [1.0, np.inf], "values are finite"),
    ],
)
def test_a_domain_error_names_the_transform_and_its_rule(chain, values, rule):
    # The domain_error arguments of log, logdiff, boxcox, logit, growth and _finite.
    key = chain[0].split("(")[0]
    with pytest.raises(TransformError) as info:
        tf.forward(series(values), chain)
    text = str(info.value)
    assert key in text and rule in text
    assert "inverse" not in text


@pytest.mark.parametrize("chain", [["diff"], ["log", "diff(lag=4)"], ["boxcox(lmbda=0.3)"]])
def test_forward_and_inverse_keep_the_series_name(chain):
    # forward mutant 17, inverse mutants 21, 59, 79, _like mutants 3 and 6, undo_lag 30 and 33.
    x = series(np.linspace(1.0, 3.0, 12), name="loss_rate")
    y, resolved = tf.forward(x, chain)
    start = tf.consumed(resolved)
    back = tf.inverse(y.iloc[start:], resolved, history=x.iloc[:start].rename("other"))
    assert y.name == "loss_rate"
    assert back.name == "loss_rate"


def test_apply_logs_the_resolved_chain_comma_separated():
    # apply mutant 48: the log entry lists every transform of the chain.
    import dazflow as dz

    ds = dz.Dataset(pd.DataFrame({"y": np.linspace(1.0, 3.0, 8)}, index=series([0] * 8).index))
    out = tf.apply(ds, "y", ["log", "diff"])
    assert out.series["y"].transforms == ("log", "diff(lag=1)")
    assert out.log[-1].reason == "applied log, diff(lag=1)"
    assert out.log[-1].columns == ("y",)


@pytest.mark.parametrize("text", [3, None, ["log"]])
def test_a_chain_entry_that_is_not_text_is_a_transform_error(text):
    # parse mutant 5: a non-string would reach the regular expression as a TypeError.
    with pytest.raises(TransformError):
        tf.parse(text)


def test_null_is_an_unset_parameter():
    # _value mutants 5 and 6: "null" and "None" both mean unset, as YAML writes it.
    assert tf.parse("boxcox(lmbda=null)").params.get("lmbda") is None
    assert tf.parse("boxcox(lmbda=None)").params.get("lmbda") is None
