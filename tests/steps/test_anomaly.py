"""Anomalies: observations outside the model's interval at a configured level.

Oracles: points injected far outside the interval are flagged and none
inside it is, on hand-made fitted values and on a fitted model with spikes
built into its data; the interval is the fitted value plus or minus the
normal quantile times the residual standard error (statsmodels' OLS
`scale`); the gate's share at, just below and just above its maximum.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import chosen_model, levels, scenarios

from dazflow import ConfigError
from dazflow.steps import diagnose

SPIKES = ("2001Q1", "2006Q3", "2014Q2")


def test_points_injected_outside_the_interval_are_flagged_and_none_inside():
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    fitted = pd.Series(np.linspace(0, 1, 40), index=index)
    z = NormalDist().inv_cdf(0.995)
    rng = np.random.default_rng(3)
    inside = rng.uniform(-0.999, 0.999, size=40) * z * 0.5  # within the half-width 0.5 z
    actual = fitted + inside
    out = [5, 17, 33]
    actual.iloc[out] = fitted.iloc[out] + np.array([1, -1, 1]) * z * 0.5 * 1.001
    found = diagnose.anomalies(actual, fitted, 0.5, 0.99)
    assert [a.period for a in found] == [str(index[i]) for i in out]
    for a in found:
        assert a.lower == pytest.approx(a.fitted - z * 0.5)
        assert a.upper == pytest.approx(a.fitted + z * 0.5)
        assert a.actual < a.lower or a.actual > a.upper
        assert a.z == pytest.approx(a.residual / 0.5)


def test_a_point_on_the_bound_is_inside():
    index = pd.period_range("2000Q1", periods=3, freq="Q")
    z = NormalDist().inv_cdf(0.975)
    fitted = pd.Series([0.0, 0.0, 0.0], index=index)
    actual = pd.Series([z, -z, np.nextafter(z, 10)], index=index)
    assert [a.period for a in diagnose.anomalies(actual, fitted, 1.0, 0.95)] == ["2000Q3"]


def spiked():
    ds = levels()
    f = ds.frame
    for p in SPIKES:
        f.loc[pd.Period(p, "Q"), "y"] += 4.0  # about 13 noise standard deviations
    return scenarios(ds.with_frame(f))


def test_spikes_built_into_the_data_are_the_anomalies_of_the_fitted_model():
    ds = spiked()
    s = chosen_model(ds)
    r = diagnose.run(ds, select=s)
    f = ds.frame
    res = sm.OLS(f["y"], sm.add_constant(f[["a", "b"]])).fit()
    s_e = np.sqrt(res.scale)
    assert r.scale == pytest.approx(s_e, rel=1e-10)
    assert r.config.anomalies.level == 0.99
    z = NormalDist().inv_cdf(0.995)
    outside = {str(p) for p, e in res.resid.items() if abs(e) > z * s_e}
    assert {a.period for a in r.anomalies} == outside
    assert set(SPIKES) <= outside
    for a in r.anomalies:
        assert a.residual == pytest.approx(res.resid[pd.Period(a.period, "Q")], rel=1e-10)


def test_the_level_is_configuration():
    ds = spiked()
    s = chosen_model(ds)
    wide = diagnose.run(ds, {"anomalies": {"level": 0.999999}}, select=s)
    narrow = diagnose.run(ds, {"anomalies": {"level": 0.5}}, select=s)
    assert {a.period for a in wide.anomalies} <= {a.period for a in narrow.anomalies}
    assert len(narrow.anomalies) > len(wide.anomalies)
    for bad in (0, 1, 1.5):
        with pytest.raises(ConfigError):
            diagnose.Config.model_validate({"anomalies": {"level": bad}})


def test_the_anomaly_share_check_passes_at_its_maximum_and_fails_above_it():
    ds = spiked()
    s = chosen_model(ds)
    r = diagnose.run(ds, select=s)
    n = len(s.design.frames(ds.frame)[0])
    share = len(r.anomalies) / n
    assert r.diagnostic("anomalies").observed == share
    for limit, ok in [(share * (1 - 1e-9), False), (share, True), (share * (1 + 1e-9), True)]:
        check = diagnose.run(ds, {"anomalies": {"max_share": limit}}, select=s).gate["anomalies"]
        assert check.observed == share and check.op == "<=" and check.passed is ok, limit


def test_the_anomaly_table_lists_each_flagged_period():
    ds = spiked()
    r = diagnose.run(ds, select=chosen_model(ds))
    table = r.artefacts["T8.3"]
    assert [row[0] for row in table.rows] == [a.period for a in r.anomalies]


def test_an_arimax_interval_takes_its_estimated_variance():
    ds = scenarios(levels())
    s = chosen_model(
        ds, form={"form": "arimax", "config": {"order": [1, 0, 0]}}, candidate="arimax:a+b"
    )
    r = diagnose.run(ds, select=s)
    assert r.scale == pytest.approx(np.sqrt(s.record.params["sigma2"]), rel=1e-12)
    z = NormalDist().inv_cdf(0.995)
    assert all(abs(a.residual) > z * r.scale for a in r.anomalies)
