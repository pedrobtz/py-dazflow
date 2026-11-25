"""Every transform followed by its inverse returns the input (roadmap Stage 3).

The oracle is the input itself: hypothesis draws finite series in each
transform's domain, the forward transform is applied, and the inverse,
seeded only with the periods the transform consumes, must rebuild every
later value.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from dazflow import transforms as tf

SETTINGS = settings(deadline=None, max_examples=60)


def series(values, freq="Q", name="y"):
    index = pd.period_range("2000Q1" if freq == "Q" else "2000-01", periods=len(values), freq=freq)
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


def levels(lo, hi, min_size=1):
    return st.lists(
        st.floats(lo, hi, allow_nan=False, allow_infinity=False), min_size=min_size, max_size=40
    )


def assert_round_trip(x, chain):
    y, resolved = tf.forward(x, chain)
    start = tf.consumed(resolved)
    back = tf.inverse(y.iloc[start:], resolved, history=x.iloc[:start])
    pd.testing.assert_index_equal(back.index, x.index[start:])
    # Lagged inverses add up rounding errors along the series: scale the absolute tolerance.
    scale = max(1.0, float(np.nanmax(np.abs(x.to_numpy()))))
    np.testing.assert_allclose(
        back.to_numpy(), x.iloc[start:].to_numpy(), rtol=1e-9, atol=1e-9 * scale
    )
    # The leading periods the transform consumed have no value.
    assert y.iloc[:start].isna().all()
    assert y.iloc[start:].notna().all()


@SETTINGS
@given(levels(-1e6, 1e6, min_size=5), st.integers(1, 4))
def test_diff_round_trip(values, lag):
    assert_round_trip(series(values), [f"diff(lag={lag})"])


@SETTINGS
@given(levels(-1e6, 1e6, min_size=13))
def test_seasonal_diff_round_trip_takes_the_period_from_the_frequency(values):
    x = series(values, freq="M")
    _, resolved = tf.forward(x, ["seasonal_diff"])
    assert [str(s) for s in resolved] == ["seasonal_diff(period=12)"]
    assert_round_trip(x, resolved)


@SETTINGS
@given(levels(1e-6, 1e6))
def test_log_round_trip(values):
    assert_round_trip(series(values), ["log"])


@SETTINGS
@given(levels(1e-3, 1e6, min_size=5), st.integers(1, 4))
def test_logdiff_round_trip(values, lag):
    assert_round_trip(series(values), [f"logdiff(lag={lag})"])


@SETTINGS
@given(
    st.lists(
        # Magnitudes within a factor of 100: near a ratio of 0, 1 + y loses the digits of x.
        st.floats(1, 100).flatmap(lambda m: st.sampled_from([m, -m])),
        min_size=5,
        max_size=40,
    ),
    st.integers(1, 4),
)
def test_growth_round_trip(values, lag):
    assert_round_trip(series(values), [f"growth(lag={lag})"])


@SETTINGS
@given(levels(1e-9, 1 - 1e-9))
def test_logit_round_trip_over_the_open_unit_interval(values):
    assert_round_trip(series(values), ["logit"])


@SETTINGS
@given(levels(1e-3, 1e3), st.floats(-2, 2))
@example(values=[2.0], lmbda=5e-324)  # scipy 1.13's inv_boxcox returned e for this
@example(values=[2.0, 1e3], lmbda=0.0)
@example(values=[2.0, 1e-3], lmbda=-1e-13)
def test_boxcox_round_trip(values, lmbda):
    assert_round_trip(series(values), [f"boxcox(lmbda={lmbda!r})"])


@SETTINGS
@given(levels(1e-3, 1e3, min_size=3).filter(lambda v: max(v) > min(v)))
def test_boxcox_estimates_lambda_and_records_it(values):
    x = series(values)
    _, resolved = tf.forward(x, ["boxcox"])
    (spec,) = resolved
    assert spec.key == "boxcox" and -2 <= spec.params["lmbda"] <= 2
    assert_round_trip(x, resolved)


@SETTINGS
@given(levels(1e-3, 1e4, min_size=8))
def test_a_chain_round_trips_through_its_intermediate_levels(values):
    assert_round_trip(series(values), ["log", "diff", "diff(lag=4)"])


@pytest.mark.parametrize("lmbda", [0.0, 1e-13, -1e-13, 1e-11, 0.5, -1.5])
def test_boxcox_matches_its_definition_in_50_digit_arithmetic(lmbda):
    from decimal import Decimal, localcontext

    x = series([0.5, 2.0, 30.0])
    y, _ = tf.forward(x, [f"boxcox(lmbda={lmbda!r})"])
    with localcontext() as ctx:
        ctx.prec = 50
        lam = Decimal(lmbda)
        expected = [
            float(Decimal(v).ln() if lam == 0 else ((lam * Decimal(v).ln()).exp() - 1) / lam)
            for v in x
        ]
    np.testing.assert_allclose(y, expected, rtol=1e-14)

    # Inside the inverse's domain, lmbda * v + 1 > 0.
    v = series([-0.6, 0.3, 0.6] if lmbda < -1 else [-0.7, 0.7, 3.4])
    back = tf.inverse(v, [f"boxcox(lmbda={lmbda!r})"])
    with localcontext() as ctx:
        ctx.prec = 50
        expected = [
            float(Decimal(w).exp() if lam == 0 else ((1 + lam * Decimal(w)).ln() / lam).exp())
            for w in v
        ]
    np.testing.assert_allclose(back, expected, rtol=1e-14)


def test_missing_values_stay_missing_and_do_not_break_the_inverse():
    x = series([1.0, 2.0, np.nan, 4.0, 8.0])
    y, resolved = tf.forward(x, ["log"])
    assert np.isnan(y.iloc[2])
    back = tf.inverse(y, resolved, history=x.iloc[:0])
    np.testing.assert_allclose(back.to_numpy(), x.to_numpy())


def test_inverse_extends_a_series_past_its_history():
    # Forecast use: future differences on top of the last observed levels.
    x = series([10.0, 11.0, 13.0])
    future = pd.Series([1.0, 2.0], index=pd.period_range("2000Q4", periods=2, freq="Q"))
    back = tf.inverse(future, ["diff"], history=x)
    assert back.tolist() == [14.0, 16.0]


def test_text_form_round_trips_and_names_every_built_in():
    assert set(tf.registry.keys()) >= {
        "diff",
        "seasonal_diff",
        "log",
        "logdiff",
        "growth",
        "logit",
        "boxcox",
    }
    for text in ["log", "diff(lag=2)", "boxcox(lmbda=0.25, bound=2.0)", "seasonal_diff(period=4)"]:
        assert str(tf.parse(text)) == text
    assert str(tf.parse("diff")) == "diff(lag=1)"


@pytest.mark.parametrize("text", ["diff(lag=0)", "diff(lags=1)", "nope", "diff(lag=", "log(x=1)"])
def test_bad_text_is_refused(text):
    from dazflow import ConfigError, RegistryError, TransformError

    with pytest.raises((TransformError, ConfigError, RegistryError)):
        tf.parse(text)


def test_apply_records_the_chain_in_the_series_metadata():
    import dazflow as dz

    x = series([1.0, 2.0, 4.0, 8.0], name="gdp")
    ds = dz.Dataset(x.to_frame(), {"gdp": {"role": "target"}})
    out = tf.apply(ds, "gdp", ["log", "diff"])
    assert out.series["gdp"].transforms == ("log", "diff(lag=1)")
    assert out.series["gdp"].role == "target"
    np.testing.assert_allclose(out.frame["gdp"].iloc[1:], np.log(2.0))
    assert out.log[-1].action == "transform" and out.log[-1].columns == ("gdp",)
    again = tf.apply(out, "gdp", ["diff"])
    assert again.series["gdp"].transforms == ("log", "diff(lag=1)", "diff(lag=1)")
    back = tf.inverse(out.frame["gdp"].iloc[1:], out.series["gdp"].transforms, history=x.iloc[:1])
    np.testing.assert_allclose(back, x.iloc[1:])


class TestTransformContract:
    """The contract every registered transform passes."""

    @pytest.mark.parametrize(
        "key", ["diff", "seasonal_diff", "log", "logdiff", "growth", "logit", "boxcox"]
    )
    def test_built_in_follows_the_protocol(self, key):
        from dazflow.core.registry import _protocol_members

        obj = tf.registry.get(key)
        assert not [m for m in _protocol_members(tf.Transform) if not hasattr(obj, m)]
        assert issubclass(obj.Config, __import__("dazflow").Config)
