"""Driver screen: single-driver OLS with HAC errors, the expected-sign gate and the threshold.

Oracles: a direct `statsmodels.OLS(...).fit(cov_type="HAC")` call on the
same aligned data and maxlags, the Newey-West lag rule written out here,
and drivers built with a known sign.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import dazflow as dz
from dazflow import ConfigError, ExploreError, GateError
from dazflow.steps import explore

N = 120


def frame(seed=0, n=N, beta_x=0.8, beta_w=-0.6):
    """White-noise drivers: y = beta_x x[t-1] + beta_w w[t] + noise; v is unrelated."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990Q1", periods=n, freq="Q")
    x, w, v = rng.normal(size=(3, n))
    y = beta_x * pd.Series(x).shift(1).to_numpy() + beta_w * w + 0.5 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "w": w, "v": v}, index=index).iloc[1:]


def dataset(signs=None, **kw):
    signs = signs or {"x": 1, "w": -1}
    meta = {"y": {"role": "target"}}
    for d in ("x", "w"):
        meta[d] = {"role": "driver", "expected_sign": signs.get(d)}
    return dz.Dataset(frame(**kw)[["y", "x", "w"]], meta)


def direct_hac(y, x, lag, maxlags):
    data = pd.DataFrame({"y": y, "x": x.shift(lag)}).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data["x"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return fit, len(data)


# --- the HAC regression ------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 1, 3])
@pytest.mark.parametrize("maxlags", [0, 2, 6])
def test_coefficient_and_hac_standard_error_equal_a_direct_statsmodels_call(lag, maxlags):
    f = frame(2)
    got = explore.screen(f["y"], f["x"], lag=lag, maxlags=maxlags)
    fit, n = direct_hac(f["y"], f["x"], lag, maxlags)
    assert got.coefficient == pytest.approx(fit.params["x"], rel=1e-12)
    assert got.std_error == pytest.approx(fit.bse["x"], rel=1e-12)
    assert got.t == pytest.approx(fit.tvalues["x"], rel=1e-12)
    assert got.pvalue == pytest.approx(fit.pvalues["x"], rel=1e-10)
    assert (got.lag, got.maxlags, got.observations) == (lag, maxlags, n)


def test_default_maxlags_follow_newey_west():
    f = frame(2)
    got = explore.screen(f["y"], f["x"], lag=1)
    n = len(f) - 1
    assert got.maxlags == math.floor(4 * (n / 100) ** (2 / 9))
    fit, _ = direct_hac(f["y"], f["x"], 1, got.maxlags)
    assert got.std_error == pytest.approx(fit.bse["x"], rel=1e-12)


def test_run_screens_with_the_configured_maxlags():
    ds = dataset()
    r = explore.run(ds, {"hac_maxlags": 3, "max_lag": 2})
    got = {s.driver: s for s in r.screens}["x"]
    f = ds.frame
    fit, _ = direct_hac(f["y"], f["x"], got.lag, 3)
    assert got.std_error == pytest.approx(fit.bse["x"], rel=1e-12)


# --- the expected-sign gate --------------------------------------------------------


def test_drivers_with_the_expected_sign_and_significance_are_kept():
    r = explore.run(dataset(), {"max_lag": 2})
    assert r.gate.passed
    assert r.kept == ("x", "w")
    assert r.gate["sign:x"].status == "pass"
    assert r.gate["sign:w"].status == "pass"
    assert r.gate["significance:x"].status == "pass"


def test_a_driver_with_the_wrong_sign_fails_the_gate():
    r = explore.run(dataset({"x": 1, "w": 1}), {"max_lag": 2})
    assert not r.gate.passed
    assert r.gate["sign:w"].status == "fail"
    assert r.gate["sign:w"].observed < 0
    assert r.kept == ("x",)
    assert {s.driver: s.observed_sign for s in r.screens}["w"] == -1


def test_a_wrong_sign_passes_only_with_an_override_that_records_a_reason():
    reason = "Substitution effect; see the methodology note."
    r = explore.run(
        dataset({"x": 1, "w": 1}),
        {"max_lag": 2, "overrides": [{"check": "sign:w", "reason": reason}]},
    )
    assert r.gate.passed
    assert r.gate["sign:w"].status == "overridden"
    assert r.gate["sign:w"].override.reason == reason
    assert r.kept == ("x", "w")
    row = next(row for row in r.artefacts["T3.3"].rows if row[0] == "w")
    assert row[-1] == "overridden"


@pytest.mark.parametrize("reason", ["", "   "])
def test_an_override_without_a_reason_is_refused(reason):
    with pytest.raises((ConfigError, GateError)):
        explore.run(
            dataset({"x": 1, "w": 1}),
            {"max_lag": 2, "overrides": [{"check": "sign:w", "reason": reason}]},
        )


def test_the_configured_expected_sign_replaces_the_metadata():
    r = explore.run(dataset({"x": 1, "w": 1}), {"max_lag": 2, "expected_signs": {"w": -1}})
    assert r.gate["sign:w"].status == "pass"
    assert {s.driver: s.expected_sign for s in r.screens}["w"] == -1


def test_a_driver_without_an_expected_sign_has_no_sign_check():
    r = explore.run(dataset({"x": 1}), {"max_lag": 2})
    names = [c.name for c in r.gate.checks]
    assert "sign:w" not in names
    assert "significance:w" in names


# --- the significance threshold ------------------------------------------------------


def _screened(pvalue, coefficient=0.4, expected=1):
    return explore.Screen(
        driver="x",
        lag=0,
        coefficient=coefficient,
        pvalue=pvalue,
        observations=100,
        maxlags=4,
        expected_sign=expected,
    )


def test_the_significance_threshold_straddles_the_p_value():
    p = 0.0123456789
    for threshold, status in [(p, "pass"), (p * (1 + 1e-12), "pass"), (p * (1 - 1e-12), "fail")]:
        gate = dz.Gate.evaluate(explore.driver_checks(_screened(p), threshold))
        assert gate["significance:x"].status == status, threshold


def test_the_sign_check_straddles_zero():
    for coefficient, status in [(1e-12, "pass"), (0.0, "fail"), (-1e-12, "fail")]:
        gate = dz.Gate.evaluate(explore.driver_checks(_screened(0.01, coefficient), 0.05))
        assert gate["sign:x"].status == status, coefficient
    neg = dz.Gate.evaluate(explore.driver_checks(_screened(0.01, -0.3, expected=-1), 0.05))
    assert neg["sign:x"].status == "pass"


def test_the_significance_threshold_straddles_the_p_value_end_to_end():
    ds = dz.Dataset(
        frame(5, beta_x=0.15)[["y", "x"]],
        {"y": {"role": "target"}, "x": {"role": "driver", "expected_sign": 1}},
    )
    p = {s.driver: s for s in explore.run(ds, {"max_lag": 2}).screens}["x"].pvalue
    assert 1e-6 < p < 0.5

    def status(threshold):
        r = explore.run(ds, {"max_lag": 2, "significance": threshold})
        return r.gate["significance:x"].status

    # Reruns agree to about 1e-16, so the end-to-end straddle leaves a margin.
    assert status(p * (1 + 1e-6)) == "pass"
    assert status(p * (1 - 1e-6)) == "fail"


def test_an_unrelated_driver_fails_the_significance_check():
    ds = dz.Dataset(
        frame(0)[["y", "v"]],
        {"y": {"role": "target"}, "v": {"role": "driver"}},
    )
    r = explore.run(ds, {"max_lag": 0})
    assert r.gate["significance:v"].status == "fail"
    assert r.kept == ()


# --- refusals ------------------------------------------------------------------------


def test_unknown_drivers_and_signs_are_refused():
    with pytest.raises(ExploreError, match="drivers"):
        explore.run(dataset(), {"drivers": ["nope"]})
    with pytest.raises(ExploreError, match="expected sign"):
        explore.run(dataset(), {"expected_signs": {"nope": 1}})


def test_drivers_without_a_target_are_refused():
    ds = dz.Dataset(frame()[["x", "w"]], {"x": {"role": "driver"}, "w": {"role": "driver"}})
    with pytest.raises(ExploreError, match="target"):
        explore.run(ds)


@pytest.mark.parametrize(
    "cfg",
    [{"significance": 0}, {"significance": 1}, {"max_lag": -1}, {"hac_maxlags": -1}],
)
def test_out_of_range_settings_are_refused(cfg):
    with pytest.raises(ConfigError):
        explore.Config.model_validate(cfg)


def test_a_driver_with_no_variation_has_no_coefficient_and_fails_the_gate():
    f = frame(0)[["y", "x"]].assign(x=1.0)
    got = explore.screen(f["y"], f["x"])
    assert (got.coefficient, got.pvalue) == (None, None)
    gate = dz.Gate.evaluate(
        explore.driver_checks(got.model_copy(update={"expected_sign": 1}), 0.05)
    )
    assert [c.status for c in gate.checks] == ["fail", "fail"]


def test_screen_refuses_too_short_a_sample():
    f = frame(0).iloc[:4]
    with pytest.raises(ExploreError, match="too few"):
        explore.screen(f["y"], f["x"], lag=2)
