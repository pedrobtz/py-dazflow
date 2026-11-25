"""Unit-root and stationarity tests, and the order of integration read from them.

Oracles: statsmodels' `adfuller` and `kpss` computed here on the same
series, trend and lag (a second implementation of the arch statistics),
and series simulated from known processes over fixed seeds.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, kpss

from dazflow.steps import explore

N = 200
SEEDS = range(10)
# The rule misreads 11 of 200 simulated random walks (design D31); seeds 1 and 2
# of the first ten are two of them, so the per-seed test names the others and
# the rate test below covers every seed.
WALK_SEEDS = (0, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def series(values, freq="M"):
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.period_range("2000-01", periods=len(values), freq=freq),
        name="x",
    )


def random_walk(seed, n=N):
    return series(np.cumsum(np.random.default_rng(seed).normal(size=n)))


def ar1(seed, phi=0.5, n=N):
    e = np.random.default_rng(seed).normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return series(x)


def trend_stationary(seed, n=N):
    return series(0.1 * np.arange(n) + ar1(seed, n=n).to_numpy())


def statsmodels_adf(x, trend, lags):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # its return type changes in 0.16
        return adfuller(x.to_numpy(), maxlag=lags, autolag=None, regression=trend)[0]


def statsmodels_kpss(x, trend, lags):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value outside its table
        return kpss(x.to_numpy(), regression=trend, nlags=lags)[0]


def by_test(results):
    return {r.test: r for r in results}


# --- cross-implementation ------------------------------------------------------


@pytest.mark.parametrize("make", [random_walk, ar1, trend_stationary])
@pytest.mark.parametrize("trend", ["c", "ct"])
@pytest.mark.parametrize("lags", [0, 2, 5])
def test_adf_and_kpss_statistics_equal_statsmodels_on_the_same_series_and_lags(make, trend, lags):
    x = make(3)
    got = by_test(explore.unit_root(x, trend=trend, lags=lags))
    assert got["adf"].lags == lags
    assert got["kpss"].lags == lags
    assert got["adf"].statistic == pytest.approx(statsmodels_adf(x, trend, lags), rel=1e-10)
    assert got["kpss"].statistic == pytest.approx(statsmodels_kpss(x, trend, lags), rel=1e-10)


def test_automatic_lags_are_recorded_and_reproduce_the_statistic():
    x = ar1(4)
    auto = by_test(explore.unit_root(x, trend="c"))
    fixed = by_test(explore.unit_root(x, trend="c", lags=auto["adf"].lags))
    assert fixed["adf"].statistic == pytest.approx(auto["adf"].statistic, rel=1e-12)
    assert auto["adf"].statistic == pytest.approx(
        statsmodels_adf(x, "c", auto["adf"].lags), rel=1e-10
    )


def test_the_four_tests_are_reported_in_order_with_their_null_hypotheses_read():
    x = ar1(5)
    tests = explore.unit_root(x, trend="c", alpha=0.05)
    assert [t.test for t in tests] == ["adf", "dfgls", "pp", "kpss"]
    for t in tests[:3]:  # a unit-root null: rejecting it says stationary
        assert t.stationary == (t.pvalue < 0.05)
    assert tests[3].stationary == (tests[3].pvalue >= 0.05)  # a stationarity null


def test_a_test_arch_cannot_run_is_noted_and_does_not_say_stationary():
    tests = explore.unit_root(series(np.ones(30)))
    assert all(t.statistic is None and t.note and not t.stationary for t in tests)
    got = explore.integration_order(series(np.ones(30)), {"max_order": 1})
    assert got.kind == "undetermined"
    assert all(level.votes == 0 for level in got.steps)


# --- known answers ---------------------------------------------------------------


@pytest.mark.parametrize("seed", WALK_SEEDS)
def test_a_random_walk_is_integrated_of_order_one(seed):
    got = explore.integration_order(random_walk(seed))
    assert (got.order, got.kind) == (1, "integrated")
    assert got.label == "I(1)"


@pytest.mark.parametrize("seed", SEEDS)
def test_an_ar1_with_phi_one_half_is_integrated_of_order_zero(seed):
    got = explore.integration_order(ar1(seed))
    assert (got.order, got.kind) == (0, "stationary")
    assert got.label == "I(0)"


@pytest.mark.parametrize("seed", SEEDS)
def test_a_trend_stationary_series_is_classed_as_trend_stationary(seed):
    got = explore.integration_order(trend_stationary(seed))
    assert (got.order, got.kind) == (0, "trend-stationary")
    # Not stationary around a constant, stationary around a trend.
    assert [(s.differences, s.trend, s.stationary) for s in got.steps] == [
        (0, "c", False),
        (0, "ct", True),
    ]


def test_the_rule_reads_known_processes_right_over_forty_seeds():
    seeds = range(40)
    walks = sum(explore.integration_order(random_walk(s)).label == "I(1)" for s in seeds)
    ar = sum(explore.integration_order(ar1(s)).label == "I(0)" for s in seeds)
    trend = sum(
        explore.integration_order(trend_stationary(s)).kind == "trend-stationary" for s in seeds
    )
    # Measured: 37, 40 and 40 of 40 (189, 200 and 200 of 200 seeds).
    assert walks >= 36
    assert ar >= 38
    assert trend >= 38


def test_a_twice_integrated_series_is_order_two():
    x = series(np.cumsum(np.cumsum(np.random.default_rng(7).normal(size=N))))
    got = explore.integration_order(x)
    assert (got.order, got.label) == (2, "I(2)")


def test_a_series_not_stationary_within_max_order_is_undetermined():
    x = series(np.cumsum(np.cumsum(np.random.default_rng(7).normal(size=N))))
    got = explore.integration_order(x, {"max_order": 1})
    assert (got.order, got.kind, got.label) == (None, "undetermined", "undetermined")
    assert max(s.differences for s in got.steps) == 1


# --- the configured rule -----------------------------------------------------------


def test_votes_straddle_the_number_of_tests_that_say_stationary():
    x = ar1(1, phi=0.9, n=80)
    first = explore.integration_order(x, {"votes": 1}).steps[0]
    k = first.votes
    assert 1 <= k <= 3, "the fixture needs a level the tests disagree on"
    assert explore.integration_order(x, {"votes": k}).steps[0].stationary  # at
    assert explore.integration_order(x, {"votes": k - 1 or 1}).steps[0].stationary  # below
    assert not explore.integration_order(x, {"votes": k + 1}).steps[0].stationary  # above


@pytest.mark.parametrize("test", ["adf", "dfgls", "pp"])
def test_a_unit_root_test_says_stationary_below_alpha_only(test):
    p = 0.0199479516283884
    at = explore.says_stationary(test, p, alpha=p)
    above = explore.says_stationary(test, p, alpha=p * (1 + 1e-12))
    below = explore.says_stationary(test, p, alpha=p * (1 - 1e-12))
    assert (below, at, above) == (False, False, True)


def test_kpss_says_stationary_at_and_above_alpha():
    p = 0.0199479516283884
    at = explore.says_stationary("kpss", p, alpha=p)
    above = explore.says_stationary("kpss", p, alpha=p * (1 + 1e-12))
    below = explore.says_stationary("kpss", p, alpha=p * (1 - 1e-12))
    assert (below, at, above) == (True, True, False)
    assert not explore.says_stationary("kpss", float("nan"), alpha=0.05)


def test_alpha_straddles_a_tests_p_value_end_to_end():
    # Reruns agree to about 1e-16 (BLAS paths), so the end-to-end straddle
    # leaves a margin; the exact point is tested on says_stationary above.
    x = ar1(1, phi=0.9, n=80)
    adf = by_test(explore.unit_root(x, trend="c"))["adf"]
    above = by_test(explore.unit_root(x, trend="c", alpha=adf.pvalue * (1 + 1e-6)))["adf"]
    below = by_test(explore.unit_root(x, trend="c", alpha=adf.pvalue * (1 - 1e-6)))["adf"]
    assert (below.stationary, above.stationary) == (False, True)


def test_differences_and_chain_follow_the_order():
    x = random_walk(0)
    got = explore.integration_order(x)
    assert got.chain == ("diff(lag=1)",)
    ts = explore.integration_order(trend_stationary(2))
    assert ts.chain == ("diff(lag=1)",)  # default: difference the trend away
    kept = explore.integration_order(trend_stationary(2), {"trend_stationary": "keep"})
    assert kept.chain == ()


def test_a_series_arch_cannot_test_gets_notes_not_an_error():
    tests = explore.unit_root(series(np.arange(12.0)))  # a straight line fits exactly
    noted = [t for t in tests if t.note]
    assert noted, "arch ran every test on a straight line"
    assert all(t.statistic is None and not t.stationary for t in noted)
