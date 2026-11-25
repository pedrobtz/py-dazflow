"""The diagnose step: residual tests, stability, back-testing and the gate.

Oracles: the direct statsmodels calls (`acorr_ljungbox`,
`acorr_breusch_godfrey`, `het_breuschpagan`, `het_white`, `het_arch`,
`jarque_bera`, `linear_reset`, `breaks_cusumolsresid`, `RecursiveLS`,
`RollingOLS`) on a statsmodels OLS fit of the same data; `dz.evaluation.
evaluate` with the same splitter for the back-test; Brown, Durbin and
Evans' bounds by hand; every threshold at, just below and just above the
observed value; a deep copy of the caller's dataset.
"""

import copy
import inspect

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import chosen_model, close, levels, scenarios
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats import diagnostic as smd
from statsmodels.stats.stattools import jarque_bera

import dazflow as dz
from dazflow import ConfigError, DiagnoseError
from dazflow.steps import diagnose

H = 8
RESIDUAL = (
    "ljung-box",
    "breusch-godfrey",
    "breusch-pagan",
    "white",
    "arch",
    "jarque-bera",
    "reset",
)


def plain(f, *args, **kwargs):
    # statsmodels 0.15 warns unless told the tuple form is wanted; 0.14 has no such argument.
    if "result_object" in inspect.signature(f).parameters:
        kwargs["result_object"] = False
    return f(*args, **kwargs)


@pytest.fixture(scope="module")
def chosen():
    ds = scenarios(levels(), h=H)
    return ds, chosen_model(ds)


@pytest.fixture(scope="module")
def diagnosed(chosen):
    ds, s = chosen
    return diagnose.run(ds, select=s)


def direct_ols(ds, s):
    f = ds.frame.loc[pd.Period(s.design.start, "Q") : pd.Period(s.design.end, "Q")]
    return sm.OLS(f["y"], sm.add_constant(f[["a", "b"]])).fit()


def lags(n, season=4):
    """Hyndman's rule: twice the season (ten without one), at most a fifth of the sample."""
    return min(2 * season if season > 1 else 10, n // 5)


# --- residual tests equal the direct statsmodels calls ------------------------------


def test_the_residual_tests_equal_the_direct_statsmodels_calls(chosen, diagnosed):
    ds, s = chosen
    res = direct_ols(ds, s)
    n = int(res.nobs)
    L = lags(n)
    lb = smd.acorr_ljungbox(res.resid, lags=[L], model_df=0, return_df=True)
    bg = plain(smd.acorr_breusch_godfrey, res, nlags=L)
    bp = smd.het_breuschpagan(res.resid, res.model.exog)
    wh = smd.het_white(res.resid, res.model.exog)
    ar = plain(smd.het_arch, res.resid, nlags=L)
    jb = jarque_bera(res.resid)
    rs = smd.linear_reset(res, power=3)
    want = {
        "ljung-box": (lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]),
        "breusch-godfrey": (bg[0], bg[1]),
        "breusch-pagan": (bp[0], bp[1]),
        "white": (wh[0], wh[1]),
        "arch": (ar[0], ar[1]),
        "jarque-bera": (jb[0], jb[1]),
        "reset": (float(np.squeeze(rs.statistic)), float(rs.pvalue)),
    }
    for name, (stat, p) in want.items():
        d = diagnosed.diagnostic(name)
        assert d.applicable and d.group == "residuals", name
        assert d.statistic == pytest.approx(float(stat), rel=1e-10), name
        assert d.pvalue == pytest.approx(float(p), rel=1e-10, abs=1e-300), name
        assert d.observed == d.pvalue and d.op == ">=" and d.threshold == 0.05, name
        check = diagnosed.gate[name]
        assert check.observed == d.pvalue and check.threshold == 0.05 and check.op == ">="


def test_the_residual_tests_take_the_configured_lags_and_power(chosen):
    ds, s = chosen
    cfg = {
        "ljung_box": {"lags": 4},
        "breusch_godfrey": {"lags": 2},
        "arch": {"lags": 1},
        "reset": {"power": 2},
    }
    r = diagnose.run(ds, cfg, select=s)
    res = direct_ols(ds, s)
    lb = smd.acorr_ljungbox(res.resid, lags=[4], model_df=0, return_df=True)
    assert r.diagnostic("ljung-box").statistic == pytest.approx(lb["lb_stat"].iloc[0], rel=1e-10)
    bg = plain(smd.acorr_breusch_godfrey, res, nlags=2)
    assert r.diagnostic("breusch-godfrey").statistic == pytest.approx(bg[0], rel=1e-10)
    ar = plain(smd.het_arch, res.resid, nlags=1)
    assert r.diagnostic("arch").statistic == pytest.approx(ar[0], rel=1e-10)
    rs = smd.linear_reset(res, power=2)
    assert r.diagnostic("reset").statistic == pytest.approx(float(np.squeeze(rs.statistic)))


@pytest.fixture(scope="module")
def arimax():
    ds = scenarios(levels(), h=H)
    s = chosen_model(
        ds, form={"form": "arimax", "config": {"order": [1, 0, 1]}}, candidate="arimax:a+b"
    )
    return ds, s, diagnose.run(ds, select=s)


def test_ljung_box_on_an_arimax_takes_its_arma_order_and_regression_tests_do_not_apply(arimax):
    ds, s, r = arimax
    y, X = s.design.frames(ds.frame)
    resid = dz.models.refit(s.record, y, X).residuals
    lb = smd.acorr_ljungbox(resid, lags=[lags(len(resid))], model_df=2, return_df=True)
    d = r.diagnostic("ljung-box")
    assert d.applicable
    assert d.statistic == pytest.approx(lb["lb_stat"].iloc[0], rel=1e-10)
    assert d.pvalue == pytest.approx(lb["lb_pvalue"].iloc[0], rel=1e-10)
    jb = jarque_bera(resid)
    assert r.diagnostic("jarque-bera").statistic == pytest.approx(jb[0], rel=1e-10)
    for name in ("breusch-godfrey", "breusch-pagan", "white", "reset", "cusum", "cusum-recursive"):
        d = r.diagnostic(name)
        assert not d.applicable and "ols" in d.reason, name
        assert name not in {c.name for c in r.gate.checks}
    assert not any(d.name.startswith("chow") and d.applicable for d in r.diagnostics)
    assert r.cusum is None and r.rolling is None
    assert "F8.2" not in r.artefacts and "F8.3" not in r.artefacts


@pytest.mark.parametrize("key", ["mean", "naive"])
def test_no_residual_test_applies_to_a_benchmark(key):
    y = levels().frame["y"]
    fitted = dz.models.get(key)().fit(y)
    out = diagnose.residual_tests(fitted, y, None)
    assert {d.name for d in out} == set(RESIDUAL)
    assert all(not d.applicable and "benchmark" in d.reason for d in out)
    assert all(d.statistic is None and d.pvalue is None for d in out)


def test_ets_takes_the_residual_tests_that_need_no_regression():
    y = levels().frame["y"]
    fitted = dz.models.get("ets")().fit(y)
    out = {d.name: d for d in diagnose.residual_tests(fitted, y, None)}
    resid = fitted.residuals
    jb = jarque_bera(resid)
    assert out["jarque-bera"].statistic == pytest.approx(jb[0], rel=1e-10)
    lb = smd.acorr_ljungbox(resid, lags=[lags(len(resid))], model_df=0, return_df=True)
    assert out["ljung-box"].statistic == pytest.approx(lb["lb_stat"].iloc[0], rel=1e-10)
    for name in ("breusch-godfrey", "breusch-pagan", "white", "reset"):
        assert not out[name].applicable and "ols" in out[name].reason


# --- stability: CUSUM and rolling coefficients ------------------------------------


def test_cusum_equals_statsmodels_and_its_bounds_are_brown_durbin_evans(chosen, diagnosed):
    ds, s = chosen
    res = direct_ols(ds, s)
    k = res.model.exog.shape[1]
    sup, p, _ = smd.breaks_cusumolsresid(res.resid, ddof=k)
    d = diagnosed.diagnostic("cusum")
    assert d.group == "stability"
    assert d.statistic == pytest.approx(sup, rel=1e-10) and d.pvalue == pytest.approx(p, rel=1e-10)
    rls = RecursiveLS(res.model.endog, res.model.exog).fit()
    c = diagnosed.cusum
    np.testing.assert_allclose(c.values, rls.cusum, rtol=1e-8, atol=1e-10)
    n = int(res.nobs)
    t = np.arange(k, n)
    root = np.sqrt(n - k)
    upper = 0.948 * root + 2 * 0.948 * (t - k) / root
    np.testing.assert_allclose(c.upper, upper, rtol=1e-12)
    np.testing.assert_allclose(c.lower, -upper, rtol=1e-12)
    assert c.periods == tuple(str(q) for q in res.model.data.row_labels[k:])
    outside = int(np.sum(np.abs(rls.cusum) > upper))
    assert diagnosed.diagnostic("cusum-recursive").observed == outside


@pytest.mark.parametrize(("alpha", "a"), [(0.01, 1.143), (0.05, 0.948), (0.10, 0.850)])
def test_the_cusum_bounds_take_the_configured_level(chosen, alpha, a):
    ds, s = chosen
    r = diagnose.run(ds, {"cusum": {"alpha": alpha}}, select=s)
    n = len(r.cusum.values) + 3
    root = np.sqrt(n - 3)
    assert r.cusum.upper[0] == pytest.approx(a * root, rel=1e-12)
    assert r.diagnostic("cusum").threshold == alpha


def test_the_cusum_level_is_one_brown_durbin_and_evans_tabulated():
    with pytest.raises(ConfigError):
        diagnose.Config.model_validate({"cusum": {"alpha": 0.2}})


def test_rolling_coefficients_equal_rolling_ols(chosen, diagnosed):
    ds, s = chosen
    res = direct_ols(ds, s)
    n = int(res.nobs)
    window = max(2 * res.model.exog.shape[1], n // 3)
    direct = RollingOLS(res.model.endog, res.model.exog, window=window).fit().params[window - 1 :]
    r = diagnosed.rolling
    assert r.window == window
    assert r.periods == tuple(str(q) for q in res.model.data.row_labels[window - 1 :])
    for j, name in enumerate(("const", "a", "b")):
        np.testing.assert_allclose(r.params[name], direct[:, j], rtol=1e-10)


def test_the_rolling_window_is_configuration(chosen):
    ds, s = chosen
    r = diagnose.run(ds, {"rolling": {"window": 20}}, select=s)
    assert (
        r.rolling.window == 20 and len(r.rolling.periods) == len(s.design.frames(ds.frame)[0]) - 19
    )


# --- back-testing reuses dz.evaluation ------------------------------------------------


def test_the_back_test_is_dz_evaluation_with_the_configured_splitter(chosen, diagnosed):
    ds, s = chosen
    y, X = s.design.frames(ds.frame)
    split = dz.evaluation.splitters.get("expanding")({"horizon": 4})
    cov = dz.evaluation.metrics.get("coverage")(level=0.95)
    want = dz.evaluation.evaluate(
        s.record.spec, y, X, splitter=split, metrics=("mae", "rmse", "mase", cov), levels=[0.95]
    )
    got = diagnosed.backtest
    # statsmodels' OLS intervals move by an ulp between runs, so compare to 1e-12.
    close(got.model_dump(mode="json"), want.model_dump(mode="json"))
    d = diagnosed.diagnostic("backtest-coverage")
    assert d.group == "backtest" and d.op == ">=" and d.threshold == 0.8
    assert d.observed == pytest.approx(want.score("coverage"), abs=1e-12)


def test_a_back_test_too_short_for_its_splitter_is_a_diagnose_error(chosen):
    ds, s = chosen
    with pytest.raises(DiagnoseError, match="back-test"):
        diagnose.run(ds, {"backtest": {"splitter_config": {"horizon": 400}}}, select=s)


# --- every threshold at, just below and just above ------------------------------------


def straddle(value):
    """Thresholds just below, at and just above `value`."""
    return value * (1 - 1e-9), value, value * (1 + 1e-9)


@pytest.mark.parametrize(
    ("name", "section"),
    [
        ("ljung-box", "ljung_box"),
        ("breusch-godfrey", "breusch_godfrey"),
        ("breusch-pagan", "breusch_pagan"),
        ("white", "white"),
        ("arch", "arch"),
        ("jarque-bera", "jarque_bera"),
        ("reset", "reset"),
        ("chow:2007Q3", "chow"),
    ],
)
def test_a_p_value_check_passes_at_its_alpha_and_above_it(chosen, diagnosed, name, section):
    # A rerun repeats a p-value to about 1e-15 (statsmodels moves by an ulp), so the run
    # is checked a relative 1e-9 either side, and the value itself on the gate's reading.
    ds, s = chosen
    d = diagnosed.diagnostic(name)
    p = d.pvalue
    assert 0 < p < 1
    below, at, above = straddle(p)
    for alpha, ok in [(below, True), (above, False)]:
        cfg = {section: {"alpha": alpha}}
        if section == "chow":
            cfg[section]["breaks"] = ["2007Q3"]
        check = diagnose.run(ds, cfg, select=s).gate[name]
        assert check.threshold == alpha and check.op == ">="
        assert check.observed == pytest.approx(p, rel=1e-12)
        assert check.passed is ok, alpha
    assert d.model_copy(update={"threshold": at}).check().passed
    assert diagnosed.gate[name] == d.check()


def test_the_cusum_check_passes_at_its_alpha_and_above_it(chosen, diagnosed):
    # The CUSUM level must be one of three, so its p-value check is read on the
    # reading function the gate uses.
    p = diagnosed.diagnostic("cusum").pvalue
    for alpha, ok in [(p * (1 - 1e-9), True), (p, True), (p * (1 + 1e-9), False)]:
        d = diagnosed.diagnostic("cusum").model_copy(update={"threshold": alpha})
        assert d.check().passed is ok


@pytest.mark.parametrize("crossings", [0, 1, 2])
def test_the_recursive_cusum_check_allows_the_configured_crossings(chosen, crossings):
    ds, s = chosen
    observed = diagnose.run(ds, select=s).diagnostic("cusum-recursive").observed
    for limit, ok in [(observed - 1, False), (observed, True), (observed + 1, True)]:
        if limit < 0:
            with pytest.raises(ConfigError):
                diagnose.Config.model_validate({"cusum": {"max_crossings": limit}})
            continue
        check = diagnose.run(ds, {"cusum": {"max_crossings": limit}}, select=s).gate[
            "cusum-recursive"
        ]
        assert check.passed is ok and check.op == "<="


def test_the_back_test_coverage_check_passes_at_its_minimum_and_below_it(chosen, diagnosed):
    ds, s = chosen
    cov = diagnosed.diagnostic("backtest-coverage").observed
    assert 0 < cov < 1
    for minimum, ok in [(cov * (1 - 1e-9), True), (cov, True), (cov * (1 + 1e-9), False)]:
        check = diagnose.run(ds, {"backtest": {"min_coverage": minimum}}, select=s).gate[
            "backtest-coverage"
        ]
        assert check.observed == cov and check.passed is ok, minimum


def test_a_failed_check_fails_the_gate_and_an_override_with_a_reason_accepts_it(chosen, diagnosed):
    ds, s = chosen
    p = diagnosed.diagnostic("jarque-bera").pvalue
    cfg = {"jarque_bera": {"alpha": min(0.99, p * 2)}}
    r = diagnose.run(ds, cfg, select=s)
    assert not r.gate.passed and r.gate["jarque-bera"].status == "fail"
    cfg["overrides"] = [{"check": "jarque-bera", "reason": "Accepted for the test."}]
    r = diagnose.run(ds, cfg, select=s)
    assert r.gate.passed and r.gate["jarque-bera"].status == "overridden"


def test_the_gate_holds_every_applicable_diagnostic(diagnosed):
    applicable = [d for d in diagnosed.diagnostics if d.applicable]
    assert [c.name for c in diagnosed.gate.checks] == [d.name for d in applicable]
    assert diagnosed.gate.passed is all(c.passed for c in diagnosed.gate.checks)
    names = {c.name for c in diagnosed.gate.checks}
    assert set(RESIDUAL) | {"cusum", "cusum-recursive", "backtest-coverage", "anomalies"} <= names
    assert "chow:2007Q3" in names  # the default candidate: the sample's midpoint


# --- the step as a whole ------------------------------------------------------------


def test_the_callers_dataset_is_unchanged(chosen):
    ds, s = chosen
    frame, scen, fp = copy.deepcopy(ds.frame), copy.deepcopy(ds.scenarios), ds.fingerprint()
    meta, log = dict(ds.series), ds.log
    diagnose.run(ds, select=s)
    pd.testing.assert_frame_equal(ds.frame, frame)
    pd.testing.assert_frame_equal(ds.scenarios, scen)
    assert ds.fingerprint() == fp and dict(ds.series) == meta and ds.log == log


def test_another_dataset_than_selects_is_refused(chosen):
    _, s = chosen
    other = levels(seed=1)
    with pytest.raises(DiagnoseError, match="select ran on"):
        diagnose.run(other, select=s)


def test_a_forecast_of_another_model_is_refused(chosen):
    ds, s = chosen
    f = dz.forecast.run(ds, select=s)
    with pytest.raises(DiagnoseError, match="forecast"):
        diagnose.run(ds, select=s, forecast=f.model_copy(update={"chosen": "ols:a"}))


def test_select_without_a_chosen_model_is_refused(chosen):
    ds, s = chosen
    with pytest.raises(DiagnoseError, match="no model"):
        diagnose.run(ds, select=s.model_copy(update={"chosen": None}))


def test_the_artefacts_have_their_declared_ids(diagnosed):
    assert set(diagnosed.artefacts) == {
        "T8.1",
        "T8.2",
        "T8.3",
        "T8.4",
        "T8.5",
        "F8.1",
        "F8.2",
        "F8.3",
        "S8.1",
    }
    assert diagnosed.finding.key == "diagnose.summary"
    table = diagnosed.artefacts["T8.1"]
    names = [row[0] for row in table.rows]
    assert names[: len(RESIDUAL)] == [diagnosed.diagnostic(n).label for n in RESIDUAL]


def test_a_test_that_does_not_apply_is_noted_under_the_table_with_its_reason(arimax):
    _, _, r = arimax
    table = r.artefacts["T8.1"]
    reset = r.diagnostic("reset")
    assert reset.label not in [row[0] for row in table.rows]
    assert f"Not applicable: {reset.label}, as {reset.reason}." in table.notes
    applicable = [d for d in r.diagnostics if d.applicable]
    assert [row[0] for row in table.rows] == [d.label for d in applicable]
    assert [row[-1] for row in table.rows] == [r.gate[d.name].status for d in applicable]


def test_the_result_round_trips_through_json(diagnosed):
    again = diagnose.Result.from_dict(diagnosed.to_dict())
    assert again.diagnostics == diagnosed.diagnostics
    assert again.gate == diagnosed.gate


def test_diagnose_is_a_registered_step_with_an_adapter():
    from dazflow.pipeline import DEFAULT_STEPS, adapters
    from dazflow.steps import registry

    assert registry.get("diagnose") is diagnose
    assert "diagnose" in adapters
    assert DEFAULT_STEPS[-1] == "diagnose"


# --- refusals and edges ------------------------------------------------------------


def test_an_unknown_diagnostic_is_a_diagnose_error(diagnosed):
    with pytest.raises(DiagnoseError, match="no diagnostic"):
        diagnosed.diagnostic("durbin-watson")


@pytest.mark.parametrize(
    ("freq", "n", "want"),
    [("Q", 100, 8), ("M", 200, 24), ("M", 60, 12), ("Y", 100, 10), ("Y", 30, 6), ("D", 100, 10)],
)
def test_the_default_lags_are_twice_the_season_at_most_a_fifth_of_the_sample(freq, n, want):
    index = pd.period_range("2000-01-01", periods=n, freq=freq)
    assert diagnose.default_lags(n, index) == want


def test_a_glm_takes_only_ljung_box():
    y = levels().frame["y"]
    share = (y - y.min() + 0.1) / (y.max() - y.min() + 0.2)
    X = levels().frame[["a", "b"]]
    fitted = dz.models.get("glm")().fit(share, X)
    out = {d.name: d for d in diagnose.residual_tests(fitted, share, X)}
    assert out["ljung-box"].applicable
    for name in ("jarque-bera", "arch"):
        assert not out[name].applicable and "GLM" in out[name].reason
    assert not out["reset"].applicable and "ols" in out["reset"].reason


def test_an_ols_without_drivers_takes_no_test_of_its_regressors():
    y = levels().frame["y"]
    fitted = dz.models.get("ols")().fit(y)
    out = {d.name: d for d in diagnose.residual_tests(fitted, y, None)}
    for name in ("breusch-pagan", "white", "reset"):
        assert not out[name].applicable and out[name].reason == "the model has no drivers"
    assert out["breusch-godfrey"].applicable and out["ljung-box"].applicable


def test_ljung_box_needs_more_lags_than_the_arma_order():
    f = levels().frame
    fitted = dz.models.get("arimax")(order=(1, 0, 1)).fit(f["y"], f[["a", "b"]])
    out = {d.name: d for d in diagnose.residual_tests(fitted, f["y"], f[["a", "b"]])}
    assert out["ljung-box"].applicable
    short = diagnose.residual_tests(fitted, f["y"], f[["a", "b"]], {"ljung_box": {"lags": 2}})
    lb = next(d for d in short if d.name == "ljung-box")
    assert not lb.applicable and "ARMA order" in lb.reason
    three = diagnose.residual_tests(fitted, f["y"], f[["a", "b"]], {"ljung_box": {"lags": 3}})
    assert next(d for d in three if d.name == "ljung-box").applicable


def test_a_chow_test_with_nothing_to_regress_on_is_refused():
    y = levels().frame["y"]
    with pytest.raises(DiagnoseError, match="no coefficients"):
        diagnose.chow(y, None, y.index[50], constant=False)


@pytest.mark.parametrize("window", [3, 101])
def test_a_rolling_window_outside_the_sample_is_refused(chosen, window):
    ds, s = chosen
    with pytest.raises(DiagnoseError, match="rolling window"):
        diagnose.run(ds, {"rolling": {"window": window}}, select=s)
    # Its neighbours inside fit: one more than the coefficients, and the whole sample.
    inside = 4 if window == 3 else 100
    assert diagnose.run(ds, {"rolling": {"window": inside}}, select=s).rolling.window == inside


def test_the_sensitivity_horizon_cannot_pass_the_scenarios(chosen):
    ds, s = chosen
    with pytest.raises(DiagnoseError, match="cover"):
        diagnose.run(ds, {"sensitivity": {"horizon": H + 1}}, select=s)


def test_without_scenarios_or_a_forecast_the_horizon_must_be_given(chosen):
    ds, s = chosen
    with pytest.raises(DiagnoseError, match="horizon must be given"):
        diagnose.run(ds.with_scenarios(None), select=s)


def test_a_scenario_without_a_driver_cannot_drive_the_model(chosen):
    ds, s = chosen
    scen = ds.scenarios
    scen.loc["s2", "a"] = np.nan
    broken = ds.with_scenarios(scen, list(ds.severity))
    with pytest.raises(DiagnoseError, match="cannot drive"):
        diagnose.run(broken, select=s)


def test_scenarios_come_from_selects_configuration_when_the_dataset_has_none(tmp_path, chosen):
    ds, s = chosen
    frame = ds.scenarios.reset_index()
    frame["period"] = frame["period"].astype(str)
    path = tmp_path / "scenarios.csv"
    frame[["scenario", "period", "a", "b", "c"]].to_csv(path, index=False)
    spec = {"path": str(path), "severity": list(ds.severity)}
    with_file = s.model_copy(
        update={
            "config": s.config.model_copy(update={"scenarios": dz.readers.ScenarioInput(**spec)})
        }
    )
    plain = ds.with_scenarios(None)
    r = diagnose.run(plain, select=with_file)
    close(
        [x.response for x in r.sensitivity],
        [x.response for x in diagnose.run(ds, select=s).sensitivity],
    )
    path.unlink()
    with pytest.raises(DiagnoseError):
        diagnose.run(plain, select=with_file)


def test_a_benchmark_takes_the_back_test_and_the_anomalies_only(chosen):
    ds, s = chosen
    b = dz.build.run(ds, {"max_size": 2, "forms": ["ols"]})
    mean = b.candidate("mean")
    bench = s.model_copy(update={"chosen": "mean", "record": mean.record, "design": mean.design})
    r = diagnose.run(ds, select=bench)
    applicable = {d.name for d in r.diagnostics if d.applicable}
    assert applicable == {"backtest-coverage", "anomalies"}
    assert all("benchmark" in d.reason for d in r.diagnostics if not d.applicable)
    assert r.sensitivity == () and r.cusum is None and r.rolling is None
    assert {"F8.2", "F8.3"}.isdisjoint(r.artefacts)
