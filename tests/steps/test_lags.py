"""Lag search: the lag of each driver whose values best explain the target.

Oracles: a driver built to lead the target by a known number of periods,
and correlations computed here with pandas on the shifted series.
"""

import numpy as np
import pandas as pd
import pytest

import dazflow as dz
from dazflow.steps import explore

N = 160


def leading(seed, lead, n=N, noise=0.3):
    """`x` leads `y` by `lead` periods: y[t] = 0.8 x[t - lead] + noise."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1980Q1", periods=n, freq="Q")
    x = pd.Series(rng.normal(size=n), index=index, name="x")
    y = (0.8 * x.shift(lead) + noise * rng.normal(size=n)).rename("y")
    return y, x


@pytest.mark.parametrize("seed", range(5))
def test_a_driver_leading_the_target_by_two_periods_is_found_at_lag_two(seed):
    y, x = leading(seed, 2)
    got = explore.lag_search(y, x, max_lag=6)
    assert got.lag == 2
    assert len(got.correlations) == 7


@pytest.mark.parametrize("lead", [0, 1, 3])
def test_other_leads_are_found_at_their_lag(lead):
    y, x = leading(11, lead)
    assert explore.lag_search(y, x, max_lag=4).lag == lead


def test_correlations_equal_pandas_on_the_common_sample():
    y, x = leading(1, 2)
    k = 4
    got = explore.lag_search(y, x, max_lag=k)
    frame = pd.DataFrame({"y": y, **{f"x{lag}": x.shift(lag) for lag in range(k + 1)}}).dropna()
    expected = [frame["y"].corr(frame[f"x{lag}"]) for lag in range(k + 1)]
    assert got.correlations == pytest.approx(expected, rel=1e-12)
    assert got.observations == len(frame)


def test_lag_zero_only_when_max_lag_is_zero():
    y, x = leading(1, 2)
    got = explore.lag_search(y, x, max_lag=0)
    assert (got.lag, len(got.correlations)) == (0, 1)


def test_run_screens_the_driver_at_the_lag_found_and_charts_every_lag():
    rng = np.random.default_rng(3)
    index = pd.period_range("1980Q1", periods=N, freq="Q")
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    y = 0.8 * pd.Series(x).shift(2).to_numpy() + 0.3 * rng.normal(size=N)
    frame = pd.DataFrame({"y": y, "x": x, "z": z}, index=index).iloc[2:]
    ds = dz.Dataset(
        frame, {"y": {"role": "target"}, "x": {"role": "driver"}, "z": {"role": "driver"}}
    )
    r = explore.run(ds, {"max_lag": 4})
    assert r.orders == {"y": 0, "x": 0, "z": 0}  # white noise: screened untransformed
    assert r.lags["x"] == 2
    assert {s.driver: s.lag for s in r.screens}["x"] == 2
    fig = r.artefacts["F3.1"]
    assert list(fig.x) == [0, 1, 2, 3, 4]
    lines = {layer.label: layer for layer in fig.layers if layer.kind == "line"}
    assert list(lines) == ["x", "z"]  # one line per driver, in the configured order
    searched = {p.driver: p for p in r.lag_profiles}
    assert lines["x"].y == pytest.approx(searched["x"].correlations)


def test_lag_search_refuses_a_negative_max_lag_and_too_short_a_sample():
    y, x = leading(1, 2, n=6)
    with pytest.raises(dz.ExploreError, match="max_lag"):
        explore.lag_search(y, x, max_lag=-1)
    with pytest.raises(dz.ExploreError, match="too few"):
        explore.lag_search(y, x, max_lag=4)
