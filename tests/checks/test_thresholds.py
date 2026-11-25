"""Every check at, just below and just above its threshold (design §13).

Two thresholds meet in each check: its own parameter (a bound, a gap
length, a tolerance), which decides whether one period fails, and the
pointblank `error` threshold on how many periods may fail before the gate
check fails. Each is straddled here. The oracle for the count is the
construction (k failing periods are built in); for the level it is
pointblank's own `warning`, `error` and `critical`, which the gate check,
computed by dazflow from the counts, must agree with.
"""

import numpy as np
import pandas as pd
import pytest

from dazflow import ConfigError, checks

N = 40
KEYS = (
    "completeness",
    "gaps",
    "duplicates",
    "range",
    "revisions",
    "outliers",
    "representativeness",
)


def periods(n=N, start="2000Q1"):
    return pd.period_range(start, periods=n, freq="Q")


def wavy(n=N):
    """A series with no failing period for any check: distinct, positive, smooth."""
    t = np.arange(n, dtype=float)
    return pd.Series(100.0 + t + 0.5 * np.sin(t), index=periods(n), name="y")


# --- k failing periods, per check ----------------------------------------------
# Each builder returns (series, config, context) with exactly k failing periods.


def completeness(k):
    x = wavy()
    x.iloc[[3 + 5 * i for i in range(k)]] = np.nan
    return x, {}, {}


def gaps(k):
    x = wavy()
    x.iloc[[3 + 5 * i for i in range(k)]] = np.nan  # k gaps of one period
    return x, {"max_length": 0}, {}


def duplicates(k):
    x = wavy()
    for i in range(k):
        x.iloc[4 + 5 * i] = x.iloc[3 + 5 * i]  # repeats the previous value
    return x, {}, {}


def range_(k):
    x = wavy()
    x.iloc[[3 + 5 * i for i in range(k)]] = 500.0
    return x, {"bounds": {"y": [0.0, 200.0]}}, {}


def revisions(k):
    x = wavy()
    old = x.copy()
    x.iloc[[3 + 5 * i for i in range(k)]] *= 1.5
    return x, {"tolerance": 0.01}, {"previous": old}


def outliers(k):
    t = np.arange(N, dtype=float)
    rng = np.random.default_rng(7)
    x = pd.Series(
        100.0 + 0.3 * t + 10.0 * np.sin(np.pi * t / 2) + rng.normal(0.0, 1.0, N),
        index=periods(),
        name="y",
    )
    for i in range(k):
        x.iloc[6 + 9 * i] += 40.0 * (-1) ** i
    return x, {}, {}


def representativeness(k):
    x = wavy()
    x.iloc[[10 + i for i in range(k)]] = np.nan  # 2002Q3 onwards
    window = {"label": "downturn", "start": "2002Q3", "end": "2004Q2"}
    return x, {"downturns": [window]}, {}


BUILD = {
    "completeness": completeness,
    "gaps": gaps,
    "duplicates": duplicates,
    "range": range_,
    "revisions": revisions,
    "outliers": outliers,
    "representativeness": representativeness,
}


def evaluate(key, k, **thresholds):
    x, cfg, ctx = BUILD[key](k)
    return checks.evaluate(key, x, {**cfg, **thresholds}, checks.Context(**ctx))


def test_every_built_in_check_is_registered():
    assert set(KEYS) <= set(checks.registry.keys())


@pytest.mark.parametrize("key", KEYS)
def test_no_failing_period_passes(key):
    out = evaluate(key, 0, error=1)
    assert out.failed == 0
    assert out.flagged == ()
    assert out.check.passed
    assert out.level == "none"


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(("k", "passed"), [(2, True), (3, False), (4, False)])
def test_count_threshold_straddle(key, k, passed):
    # error = 3 periods: two pass, three (at) and four fail.
    out = evaluate(key, k, warning=1, error=3, critical=4)
    assert out.failed == k
    assert len(out.flagged) == k
    assert out.check.observed == k
    assert out.check.threshold == 3
    assert out.check.passed is passed
    assert out.check.status == ("pass" if passed else "fail")
    assert out.level == {2: "warning", 3: "error", 4: "critical"}[k]
    assert out.check.name == f"{key}:y"


@pytest.mark.parametrize(("k", "passed"), [(3, True), (4, False), (5, False)])
def test_share_threshold_straddle(k, passed):
    # error = 0.1 of 40 periods = 4: 3 (0.075) passes, 4 (at) and 5 fail.
    out = evaluate("completeness", k, error=0.1)
    assert out.units == N
    assert out.share == k / N
    assert out.check.observed == k / N
    assert out.check.passed is passed
    assert out.check.unit == "share of periods"


def test_the_level_is_pointblanks():
    out = evaluate("range", 1, warning=1, error=2)
    assert out.level == "warning"
    assert out.check.passed


# --- each check's own parameter ------------------------------------------------


def one(key, x, cfg, **ctx):
    return checks.evaluate(key, x, cfg, checks.Context(**ctx))


@pytest.mark.parametrize(
    ("value", "failed"), [(200.0, 0), (np.nextafter(200.0, 300.0), 1), (199.0, 0)]
)
def test_range_upper_bound_straddle(value, failed):
    x = wavy()
    x.iloc[5] = value
    assert one("range", x, {"bounds": {"y": [0.0, 200.0]}}).failed == failed


@pytest.mark.parametrize(("value", "failed"), [(50.0, 0), (np.nextafter(50.0, 0.0), 1), (51.0, 0)])
def test_range_lower_bound_straddle(value, failed):
    x = wavy()
    x.iloc[5] = value
    assert one("range", x, {"bounds": {"y": [50.0, None]}}).failed == failed


def test_range_without_bounds_for_the_series_is_not_applicable():
    out = one("range", wavy(), {"bounds": {"other": [0.0, 1.0]}})
    assert out.check is None
    assert "bounds" in out.reason


@pytest.mark.parametrize(("length", "failed"), [(1, 0), (2, 0), (3, 3)])
def test_gap_length_straddle(length, failed):
    # max_length = 2: a gap of 1 or 2 periods is accepted, every period of a 3-gap fails.
    x = wavy()
    x.iloc[10 : 10 + length] = np.nan
    out = one("gaps", x, {"max_length": 2})
    assert out.failed == failed
    assert out.flagged == tuple(str(p) for p in x.index[10 : 10 + length]) * (failed > 0)


def test_leading_and_trailing_missing_values_are_not_gaps():
    x = wavy()
    x.iloc[:3] = np.nan
    x.iloc[-2:] = np.nan
    assert one("gaps", x, {"max_length": 0}).failed == 0
    assert one("completeness", x, {}).failed == 5


@pytest.mark.parametrize(("towards", "failed"), [(None, 1), (np.inf, 0), (-np.inf, 0)])
def test_duplicate_straddle(towards, failed):
    # A value equal to the previous one is a repeat; the nearest other float is not.
    x = wavy()
    x.iloc[8] = x.iloc[7] if towards is None else np.nextafter(x.iloc[7], towards)
    assert one("duplicates", x, {}).failed == failed


@pytest.mark.parametrize(("new", "failed"), [(101.0, 0), (101.0000001, 1), (100.5, 0)])
def test_revision_tolerance_straddle(new, failed):
    # tolerance 0.01: a revision of exactly 1% of the old value is accepted.
    old = wavy()
    old.iloc[6] = 100.0
    x = old.copy()
    x.iloc[6] = new
    assert one("revisions", x, {"tolerance": 0.01}, previous=old).failed == failed


def test_a_revision_from_zero_fails_and_none_from_zero_passes():
    old = wavy()
    old.iloc[2] = 0.0
    x = old.copy()
    assert one("revisions", x, {"tolerance": 0.0}, previous=old).failed == 0
    x.iloc[2] = 1.0
    assert one("revisions", x, {"tolerance": 0.5}, previous=old).failed == 1


def test_revisions_without_a_previous_vintage_are_not_applicable():
    out = one("revisions", wavy(), {})
    assert out.check is None
    assert "previous" in out.reason


@pytest.mark.parametrize(("start", "failed"), [("2000Q1", 0), ("1999Q4", 1), ("2000Q2", 0)])
def test_downturn_window_straddle(start, failed):
    # The history starts in 2000Q1: a window from 1999Q4 has one uncovered period.
    window = {"label": "downturn", "start": start, "end": "2001Q4"}
    out = one("representativeness", wavy(), {"downturns": [window]})
    assert out.failed == failed
    assert out.flagged == (("1999Q4",) if failed else ())


def test_representativeness_without_a_downturn_is_not_applicable():
    out = one("representativeness", wavy(), {})
    assert out.check is None


def test_a_check_with_no_test_unit_passes():
    # pointblank 0.13 cannot interrogate an empty table; dazflow answers without it.
    x = pd.Series(np.nan, index=periods(), name="y")
    out = one("range", x, {"bounds": {"y": [0.0, 1.0]}})
    assert out.units == 0
    assert out.failed == 0
    assert out.check.passed


@pytest.mark.parametrize(
    "bad",
    [{"error": 0}, {"error": -1}, {"error": 2.5}, {"warning": 0}, {"critical": 1.5}],
)
def test_thresholds_are_positive_shares_or_whole_counts(bad):
    with pytest.raises(ConfigError):
        checks.evaluate("completeness", wavy(), bad)


def test_unknown_check_configuration_is_refused():
    with pytest.raises(ConfigError):
        checks.evaluate("gaps", wavy(), {"max_gap": 1})
