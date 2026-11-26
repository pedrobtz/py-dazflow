"""Outlier detection on STL residuals: a synthetic seasonal series with three spikes.

The oracle is the construction: the spikes are put at known periods, and
exactly those periods must be flagged, for every fixed seed.
"""

import numpy as np
import pandas as pd
import pytest

from dazflow import checks

SEEDS = range(10)


def seasonal(freq, n, seed, spikes=(), size=10.0):
    period = {"M": 12, "Q": 4}[freq]
    start = {"M": "2010-01", "Q": "2000Q1"}[freq]
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    x = 100.0 + 0.3 * t + 10.0 * np.sin(2 * np.pi * t / period) + rng.normal(0.0, 1.0, n)
    for i, at in enumerate(spikes):
        x[at] += size * (-1) ** i
    return pd.Series(x, index=pd.period_range(start, periods=n, freq=freq), name="y")


# Spikes in three different seasons, away from the first and last cycle.
CASES = {"M": (120, (25, 62, 99)), "Q": (80, (17, 42, 67))}


@pytest.mark.parametrize("freq", ["M", "Q"])
@pytest.mark.parametrize("seed", SEEDS)
def test_exactly_the_three_spikes_are_flagged(freq, seed):
    n, at = CASES[freq]
    x = seasonal(freq, n, seed, spikes=at)
    out = checks.evaluate("outliers", x)
    assert out.flagged == tuple(str(x.index[i]) for i in at)
    assert out.units == n


@pytest.mark.parametrize("freq", ["M", "Q"])
@pytest.mark.parametrize("seed", SEEDS)
def test_a_series_without_spikes_has_no_outlier(freq, seed):
    n, _ = CASES[freq]
    assert checks.evaluate("outliers", seasonal(freq, n, seed)).flagged == ()


def test_scores_are_robust_z_scores_of_the_residuals():
    n, at = CASES["M"]
    x = seasonal("M", n, 0, spikes=at)
    z = checks.outlier_scores(x)
    assert z.index.equals(x.index)
    assert (z >= 0).all()
    assert z.iloc[list(at)].min() > 4.0
    assert z.drop(x.index[list(at)]).max() < 4.0
    # Robust standardisation: half the scores are at most 0.6745 (the normal MAD).
    assert np.median(z) == pytest.approx(0.6745, rel=0.35)


def test_missing_values_are_neither_scored_nor_flagged():
    n, at = CASES["M"]
    x = seasonal("M", n, 1, spikes=at)
    x.iloc[[5, 40]] = np.nan
    out = checks.evaluate("outliers", x)
    assert out.flagged == tuple(str(x.index[i]) for i in at)
    assert out.units == n - 2


def test_an_annual_spike_is_flagged():
    t = np.arange(40, dtype=float)
    rng = np.random.default_rng(3)
    x = pd.Series(
        50.0 + 0.5 * t + rng.normal(0.0, 1.0, 40),
        index=pd.period_range("1980", periods=40, freq="Y"),
        name="y",
    )
    x.iloc[20] += 12.0
    assert "2000" in checks.evaluate("outliers", x).flagged
    z = checks.outlier_scores(x)
    assert z.idxmax() == pd.Period("2000", freq="Y")


def test_a_series_shorter_than_two_cycles_is_not_applicable():
    x = seasonal("Q", 7, 0)
    out = checks.evaluate("outliers", x)
    assert out.check is None
    assert "8" in out.reason


@pytest.mark.parametrize(("z", "flagged"), [(3.0, True), (100.0, False)])
def test_the_z_threshold_is_configuration(z, flagged):
    n, at = CASES["Q"]
    x = seasonal("Q", n, 0, spikes=at[:1], size=6.0)
    out = checks.evaluate("outliers", x, {"z": z})
    assert (str(x.index[at[0]]) in out.flagged) is flagged


# Detection rates on short series and small spikes (Stage 13, the Stage 4 open
# point). The exact-set tests above use 20 years of quarters; here the series
# are 10 years of quarters or 4 of months, or the spikes are 6 or 8 standard
# deviations, and the check is not exact on every seed. The oracle is still
# the construction, read as rates over 40 fixed seeds: the share of planted
# spikes flagged (recall) and the periods flagged that hold no spike. Measured
# on 2026-11-26 (statsmodels 0.15.0), with a margin for other releases:
#
#   series            spike  recall    false flags
#   40 quarters       10 sd  120/120   0
#   40 quarters        8 sd  101/120   0
#   40 quarters       none   -         2
#   48 months         10 sd  119/120   0
#   80 quarters        6 sd   91/120   1

RATE_SEEDS = range(40)
RATES = {
    "40Q-10sd": ("Q", 40, (9, 20, 31), 10.0, 114, 2),
    "40Q-8sd": ("Q", 40, (9, 20, 31), 8.0, 90, 2),
    "40Q-none": ("Q", 40, (), 10.0, 0, 4),
    "48M-10sd": ("M", 48, (13, 25, 37), 10.0, 114, 2),
    "80Q-6sd": ("Q", 80, (17, 42, 67), 6.0, 78, 3),
}


@pytest.mark.parametrize("case", RATES)
def test_detection_rate_on_short_series_and_small_spikes(case):
    freq, n, at, size, min_recall, max_false = RATES[case]
    recall = false = 0
    for seed in RATE_SEEDS:
        x = seasonal(freq, n, seed, spikes=at, size=size)
        flagged = set(checks.evaluate("outliers", x).flagged)
        planted = {str(x.index[i]) for i in at}
        recall += len(flagged & planted)
        false += len(flagged - planted)
    assert recall >= min_recall, f"{recall} of {len(at) * len(RATE_SEEDS)} spikes flagged"
    assert false <= max_false, f"{false} periods without a spike flagged"
