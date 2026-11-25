"""The Chow test, dazflow's own code (design §7.8).

Oracles: statsmodels OLS fits of the same data, restricted (one regression
over the whole sample) and unrestricted (each coefficient allowed to change
at the break, as interactions with a step dummy), compared by statsmodels'
own `compare_f_test`, and the F statistic written from the three residual
sums of squares; a synthetic series with a break built in at a known
period.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import chosen_model, levels, scenarios

import dazflow as dz
from dazflow.steps import diagnose

BREAK = pd.Period("2008Q1", "Q")


def broken(n=100, seed=0, shift=1.5, at=BREAK):
    """y = 1 + 0.8 a - 0.5 b + noise, whose slope on `a` moves by `shift` from `at` on."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1995Q1", periods=n, freq="Q")
    X = pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["a", "b"])
    after = (index >= at).astype(float)
    y = 1.0 + (0.8 + shift * after) * X["a"] - 0.5 * X["b"] + 0.3 * rng.normal(size=n)
    return pd.Series(y, index=index, name="y"), X


def unrestricted(y, X, at):
    exog = sm.add_constant(X)
    step = (y.index >= at).astype(float)
    inter = exog.mul(step, axis=0).add_suffix("_after")
    return sm.OLS(y, pd.concat([exog, inter], axis=1)).fit(), sm.OLS(y, exog).fit()


@pytest.mark.parametrize("at", ["2000Q1", "2008Q1", "2015Q4"])
@pytest.mark.parametrize("shift", [0.0, 1.5])
def test_the_chow_statistic_equals_statsmodels_restricted_and_unrestricted_fits(at, shift):
    y, X = broken(shift=shift)
    at = pd.Period(at, "Q")
    out = diagnose.chow(y, X, at)
    full, pooled = unrestricted(y, X, at)
    F, p, df = full.compare_f_test(pooled)
    assert out.statistic == pytest.approx(F, rel=1e-10)
    assert out.pvalue == pytest.approx(p, rel=1e-9, abs=1e-300)
    assert out.df_num == df == 3
    assert out.df_denom == len(y) - 6
    # The same F from the residual sums of squares of the pooled and the two separate fits.
    first, second = y.index < at, y.index >= at
    exog = sm.add_constant(X)
    rss = [sm.OLS(y[m], exog[m]).fit().ssr for m in (first, second)]
    by_hand = ((pooled.ssr - sum(rss)) / 3) / (sum(rss) / (len(y) - 6))
    assert out.statistic == pytest.approx(by_hand, rel=1e-10)
    assert out.period == str(at)


def test_the_chow_test_without_a_constant_counts_only_the_drivers():
    y, X = broken()
    out = diagnose.chow(y, X, BREAK, constant=False)
    step = (y.index >= BREAK).astype(float)
    full = sm.OLS(y, pd.concat([X, X.mul(step, axis=0).add_suffix("_after")], axis=1)).fit()
    F, _, df = full.compare_f_test(sm.OLS(y, X).fit())
    assert out.statistic == pytest.approx(F, rel=1e-10) and out.df_num == df == 2


def test_it_detects_a_break_built_in_at_a_known_period():
    y, X = broken(shift=1.5)
    at = diagnose.chow(y, X, BREAK)
    assert at.pvalue < 1e-6
    # Of every date with room for both fits, the known one gives the largest F.
    dates = y.index[10:-10]
    stats = [diagnose.chow(y, X, d).statistic for d in dates]
    assert dates[int(np.argmax(stats))] == BREAK


def test_it_does_not_reject_a_series_without_a_break():
    y, X = broken(shift=0.0)
    assert diagnose.chow(y, X, BREAK).pvalue > 0.05


@pytest.mark.parametrize(
    "at", ["1995Q1", "1995Q3", "1995Q4", "2019Q2", "2019Q3", "2019Q4", "2030Q1"]
)
def test_a_break_without_room_for_two_fits_is_refused(at):
    y, X = broken()
    # Each part needs more observations than the three coefficients: three are refused.
    with pytest.raises(dz.DiagnoseError, match="break"):
        diagnose.chow(y, X, pd.Period(at, "Q"))


def test_the_smallest_parts_are_four_observations_each():
    y, X = broken()
    assert np.isfinite(diagnose.chow(y, X, y.index[4]).statistic)
    assert np.isfinite(diagnose.chow(y, X, y.index[-4]).statistic)


def test_the_step_tests_the_configured_breaks_and_flags_the_built_in_one():
    y, X = broken(shift=1.5)
    c = pd.Series(np.random.default_rng(9).normal(size=len(y)), index=y.index, name="c")
    frame = pd.concat([y, X, c], axis=1)
    meta = {
        "y": {"role": "target"},
        "a": {"role": "driver", "expected_sign": 1},
        "b": {"role": "driver", "expected_sign": -1},
        "c": {"role": "driver", "expected_sign": 1},
    }
    ds = scenarios(dz.Dataset(frame, meta))
    s = chosen_model(ds)
    r = diagnose.run(ds, {"chow": {"breaks": ["2000Q1", "2008Q1"]}}, select=s)
    assert not r.gate["chow:2008Q1"].passed
    want = diagnose.chow(y, X, BREAK)
    d = r.diagnostic("chow:2008Q1")
    assert d.statistic == pytest.approx(want.statistic, rel=1e-10)
    assert d.group == "stability" and d.threshold == 0.05


def test_the_candidate_breaks_are_fractions_of_the_sample():
    ds = scenarios(levels())
    s = chosen_model(ds)
    r = diagnose.run(ds, {"chow": {"candidates": [0.25, 0.5, 0.75]}}, select=s)
    # 100 quarters from 1995Q1: positions 25, 50 and 75.
    names = [d.name for d in r.diagnostics if d.name.startswith("chow:")]
    assert names == ["chow:2001Q2", "chow:2007Q3", "chow:2013Q4"]


def test_a_configured_break_outside_the_sample_is_not_applicable_with_its_reason():
    ds = scenarios(levels())
    s = chosen_model(ds)
    r = diagnose.run(ds, {"chow": {"breaks": ["1980Q1"]}}, select=s)
    d = r.diagnostic("chow:1980Q1")
    assert not d.applicable and "observations" in d.reason
    assert "chow:1980Q1" not in {c.name for c in r.gate.checks}
