"""One-at-a-time sensitivity: a shock in each driver, and each driver's scenario path alone.

Oracles: for a linear model the response to a unit shock is the
coefficient (statsmodels' OLS estimate; for a regression with ARMA errors
the SARIMAX coefficient), at every horizon; a one-standard-deviation shock
is the coefficient times the driver's sample standard deviation; a driver's
scenario path alone moves a linear forecast by the coefficient times the
path's departure from the base scenario, and the drivers' responses add up
to the whole scenario's; on a differenced target the level response
cumulates the coefficient, h times at horizon h.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import chosen_model, levels, scenarios

import dazflow as dz
from dazflow.steps import diagnose

H = 6


@pytest.fixture(scope="module")
def chosen():
    ds = scenarios(levels(), h=H)
    return ds, chosen_model(ds)


def ols(ds, s):
    f = ds.frame.loc[pd.Period(s.design.start, "Q") : pd.Period(s.design.end, "Q")]
    return sm.OLS(f["y"], sm.add_constant(f[["a", "b"]])).fit()


def rows(result, kind):
    return {(r.driver, r.scenario): r for r in result.sensitivity if r.kind == kind}


def test_the_response_to_a_unit_shock_is_the_coefficient(chosen):
    ds, s = chosen
    r = diagnose.run(ds, {"sensitivity": {"scale": "unit"}}, select=s)
    beta = ols(ds, s).params
    shocks = rows(r, "shock")
    assert {d for d, _ in shocks} == {"a", "b"}
    for (d, _), row in shocks.items():
        assert row.size == 1.0
        assert len(row.response) == H
        np.testing.assert_allclose(row.response, [beta[d]] * H, rtol=1e-10)
        np.testing.assert_allclose(row.level_response, [beta[d]] * H, rtol=1e-10)


def test_the_shock_size_is_configuration(chosen):
    ds, s = chosen
    r = diagnose.run(ds, {"sensitivity": {"scale": "unit", "shock": -2.5}}, select=s)
    beta = ols(ds, s).params
    for (d, _), row in rows(r, "shock").items():
        assert row.size == -2.5
        np.testing.assert_allclose(row.response, [-2.5 * beta[d]] * H, rtol=1e-10)


def test_a_one_standard_deviation_shock_is_the_default(chosen):
    ds, s = chosen
    r = diagnose.run(ds, select=s)
    assert r.config.sensitivity.scale == "sd"
    res = ols(ds, s)
    _, X = s.design.frames(ds.frame)
    for (d, _), row in rows(r, "shock").items():
        sd = float(X[d].std(ddof=1))
        assert row.size == pytest.approx(sd, rel=1e-12)
        np.testing.assert_allclose(row.response, [res.params[d] * sd] * H, rtol=1e-10)


def test_for_a_regression_with_arma_errors_the_unit_response_is_the_coefficient():
    ds = scenarios(levels(), h=H)
    s = chosen_model(
        ds, form={"form": "arimax", "config": {"order": [1, 0, 0]}}, candidate="arimax:a+b"
    )
    r = diagnose.run(ds, {"sensitivity": {"scale": "unit"}}, select=s)
    for (d, _), row in rows(r, "shock").items():
        np.testing.assert_allclose(row.response, [s.record.params[d]] * H, rtol=1e-8)


def test_each_drivers_path_alone_moves_the_forecast_by_coefficient_times_departure(chosen):
    ds, s = chosen
    r = diagnose.run(ds, select=s)
    beta = ols(ds, s).params
    base = ds.scenarios.xs("s0", level="scenario").iloc[:H]
    paths = rows(r, "scenario")
    assert {sc for _, sc in paths} == {"s1", "s2"}  # the base scenario, s0, is the reference
    for scen in ("s1", "s2"):
        other = ds.scenarios.xs(scen, level="scenario").iloc[:H]
        total = np.zeros(H)
        for d in ("a", "b"):
            want = beta[d] * (other[d].to_numpy() - base[d].to_numpy())
            np.testing.assert_allclose(paths[(d, scen)].response, want, rtol=1e-10)
            total += want
        np.testing.assert_allclose(paths[("all", scen)].response, total, rtol=1e-10)
        f = dz.forecast.run(ds, select=s)
        diff = np.array(f.paths[int(scen[1])].raw.mean) - np.array(f.paths[0].raw.mean)
        np.testing.assert_allclose(paths[("all", scen)].response, diff, rtol=1e-10)


def test_the_base_scenario_is_configuration(chosen):
    ds, s = chosen
    r = diagnose.run(ds, {"sensitivity": {"base": "s1"}}, select=s)
    assert {sc for _, sc in rows(r, "scenario")} == {"s0", "s2"}
    with pytest.raises(dz.DiagnoseError, match="base"):
        diagnose.run(ds, {"sensitivity": {"base": "nope"}}, select=s)


def test_the_horizon_is_forecasts_and_otherwise_the_scenarios_coverage(chosen):
    ds, s = chosen
    f = dz.forecast.run(ds, {"horizon": 3}, select=s)
    r = diagnose.run(ds, select=s, forecast=f)
    assert all(len(row.response) == 3 for row in r.sensitivity)
    r = diagnose.run(ds, {"sensitivity": {"horizon": 2}}, select=s, forecast=f)
    assert all(len(row.response) == 2 for row in r.sensitivity)
    assert all(len(row.response) == H for row in diagnose.run(ds, select=s).sensitivity)


def test_without_scenarios_the_drivers_are_held_at_their_last_values_and_shocked(chosen):
    ds, s = chosen
    plain = ds.with_scenarios(None)
    r = diagnose.run(plain, {"sensitivity": {"scale": "unit", "horizon": 4}}, select=s)
    beta = ols(ds, s).params
    assert not rows(r, "scenario")
    for (d, _), row in rows(r, "shock").items():
        np.testing.assert_allclose(row.response, [beta[d]] * 4, rtol=1e-10)


def test_on_a_differenced_target_the_level_response_cumulates_the_coefficient(ran_example):
    record = json.loads((ran_example / "reports" / "record.json").read_text(encoding="utf-8"))
    result = diagnose.Result.from_dict(record["results"]["diagnose"])
    assert result.design.chains["loss_rate"] == ("diff(lag=1)",)
    shocks = rows(result, "shock")
    assert shocks
    for (d, _), row in shocks.items():
        beta = result.record.params[d] * row.size
        h = np.arange(1, len(row.response) + 1)
        np.testing.assert_allclose(row.response, beta, rtol=1e-10)
        np.testing.assert_allclose(row.level_response, beta * h, rtol=1e-9)
