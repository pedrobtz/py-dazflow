"""Shared data for the model tests: a quarterly share driven by two series, with AR(1) noise."""

import numpy as np
import pandas as pd
import pytest


def share(n=80, seed=1, start="2000Q1"):
    """A share between 0 and 1 (`y`), two drivers (`X`), on a quarterly PeriodIndex."""
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, periods=n, freq="Q")
    X = pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["gdp", "rate"])
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal(scale=0.02)
    season = np.tile([0.01, -0.005, 0.0, -0.005], n // 4 + 1)[:n]
    y = 0.3 + 0.03 * X["gdp"] - 0.02 * X["rate"] + season + 0.0005 * np.arange(n) + e
    return y.rename("loss"), X


@pytest.fixture
def data():
    return share()
