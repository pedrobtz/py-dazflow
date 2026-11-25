"""The forecast step: scenario paths with intervals, anchored, mean-reverting, on the target's scale.

Oracles: hand-worked four-period paths for every anchoring rule and for
mean reversion; statsmodels' own forecast from the same driver paths (OLS
`get_prediction`, exact; SARIMAX `get_forecast` from the same parameters);
statsmodels' OLS residual at the origin for the jump-off gap, and the
gate at, just below and just above it; hand arithmetic (last level plus
the cumulated changes) for a differenced target; a deep copy of the
caller's dataset; a second run with the same seed.
"""

import copy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import close, levels, scenarios
from scipy.special import expit
from statsmodels.tsa.statespace.sarimax import SARIMAX

import dazflow as dz
from dazflow import ConfigError, ForecastError
from dazflow.steps import build, forecast, select

H = 8


def quick(**extra):
    return {"mcs": {"reps": 200}, "splitter_config": {"folds": 10}, **extra}


def chosen_model(ds, form="ols", candidate="ols:a+b", explore=None, **select_extra):
    b = build.run(ds, {"max_size": 2, "forms": [form]}, explore=explore)
    s = select.run(
        ds,
        quick(choice={"candidate": candidate, "reason": "Fixed for the test."}, **select_extra),
        build=b,
    )
    assert s.chosen == candidate
    return s


@pytest.fixture(scope="module")
def chosen():
    ds = scenarios(levels(), h=H)
    return ds, chosen_model(ds)


def sample(ds, s):
    f = ds.frame
    return f.loc[pd.Period(s.design.start, "Q") : pd.Period(s.design.end, "Q")]


def ols_direct(ds, s):
    f = sample(ds, s)
    return sm.OLS(f["y"], sm.add_constant(f[["a", "b"]])).fit()


# --- anchoring and mean reversion, by hand --------------------------------------------


@pytest.mark.parametrize(
    ("rule", "decay", "expected"),
    [
        ("carry", 0.5, [2.0, 2.0, 2.0, 2.0]),
        ("decay", 0.5, [1.0, 0.5, 0.25, 0.125]),
        ("decay", 0.25, [1.5, 1.125, 0.84375, 0.6328125]),
        ("decay", 1.0, [0.0, 0.0, 0.0, 0.0]),
        ("none", 0.5, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_anchoring_adds_the_gap_by_its_rule_on_a_four_period_path(rule, decay, expected):
    adj = forecast.adjustment(2.0, 4, rule, decay)
    assert list(adj) == expected
    path = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(path + adj) == [p + e for p, e in zip(path, expected, strict=True)]


def test_a_negative_gap_is_carried_and_decayed_with_its_sign():
    assert list(forecast.adjustment(-4.0, 4, "carry")) == [-4.0] * 4
    assert list(forecast.adjustment(-4.0, 4, "decay", 0.5)) == [-2.0, -1.0, -0.5, -0.25]


@pytest.mark.parametrize(
    ("path", "level", "periods", "expected"),
    [
        ([4.0, 4.0, 4.0, 4.0], 0.0, None, [-1.0, -2.0, -3.0, -4.0]),  # reaches 0 at the end
        ([4.0, 4.0, 4.0, 4.0], 0.0, 2, [-2.0, -4.0, -4.0, -4.0]),  # reaches it after two
        ([1.0, 2.0, 3.0, 4.0], 10.0, None, [2.25, 4.0, 5.25, 6.0]),
        ([1.0, 2.0, 3.0, 4.0], 10.0, 8, [1.125, 2.0, 2.625, 3.0]),
    ],
)
def test_mean_reversion_moves_a_four_period_path_towards_the_level(path, level, periods, expected):
    shift = forecast.revert(path, level, periods)
    assert list(shift) == expected
    if periods is None or periods <= len(path):
        assert (np.array(path) + shift)[-1] == level


def test_the_step_anchors_then_reverts_and_moves_the_intervals_with_the_mean(chosen):
    ds, s = chosen
    cfg = {"anchor": {"rule": "carry"}, "reversion": {"level": 1.0, "periods": 4}}
    r = forecast.run(ds, cfg, select=s)
    resid = ols_direct(ds, s).resid.iloc[-1]
    assert r.gap == pytest.approx(resid, rel=1e-10)
    w = np.minimum(np.arange(1, H + 1) / 4, 1.0)
    for p in r.paths:
        anchored = np.array(p.raw.mean) + resid
        np.testing.assert_allclose(p.anchored.mean, anchored, rtol=1e-12)
        shift = w * (1.0 - anchored)
        np.testing.assert_allclose(p.forecast.mean, anchored + shift, rtol=1e-12, atol=1e-14)
        for i in range(len(p.raw.levels)):
            np.testing.assert_allclose(
                p.forecast.lower[i], np.array(p.raw.lower[i]) + resid + shift, rtol=1e-12
            )
            np.testing.assert_allclose(
                p.forecast.upper[i], np.array(p.raw.upper[i]) + resid + shift, rtol=1e-12
            )
        assert p.forecast.mean[3:] == pytest.approx([1.0] * (H - 3), abs=1e-12)


def test_decay_is_the_default_rule(chosen):
    ds, s = chosen
    r = forecast.run(ds, select=s)
    assert r.config.anchor.rule == "decay" and r.config.anchor.decay == 0.5
    expected = r.gap * 0.5 ** np.arange(1, H + 1)
    np.testing.assert_allclose(r.adjustments, expected, rtol=1e-14)


# --- forecast equality: before anchoring, statsmodels' forecast --------------------


def test_before_anchoring_the_mean_equals_statsmodels_ols_prediction(chosen):
    ds, s = chosen
    r = forecast.run(ds, select=s)
    ols = ols_direct(ds, s)
    assert r.origin == s.design.end and r.horizon == H
    assert {p.scenario for p in r.paths} == set(ds.severity)
    for p in r.paths:
        paths = ds.scenarios.xs(p.scenario, level="scenario").iloc[:H]
        direct = ols.get_prediction(sm.add_constant(paths[["a", "b"]], has_constant="add"))
        np.testing.assert_allclose(p.raw.mean, direct.predicted_mean, rtol=1e-10)
        assert p.raw.periods == tuple(str(q) for q in paths.index)


def test_before_anchoring_the_mean_equals_sarimax_get_forecast():
    # From the same parameters, so the optimiser's run-to-run variation at the
    # lowest dependency versions (issue #16) does not enter the comparison.
    ds = scenarios(levels(), h=H)
    s = chosen_model(
        ds, form={"form": "arimax", "config": {"order": [1, 0, 0]}}, candidate="arimax:a+b"
    )
    r = forecast.run(ds, select=s)
    f = sample(ds, s)
    model = SARIMAX(f["y"], exog=f[["a", "b"]], order=(1, 0, 0), trend="c")
    direct = model.smooth(np.array(list(s.record.params.values())))
    for p in r.paths:
        paths = ds.scenarios.xs(p.scenario, level="scenario").iloc[:H]
        want = direct.get_forecast(H, exog=paths[["a", "b"]].to_numpy()).predicted_mean
        np.testing.assert_allclose(p.raw.mean, want, rtol=1e-8)


# --- the reconciliation gate --------------------------------------------------------


def test_the_jump_off_gap_is_the_origin_residual_and_the_gate_straddles_it(chosen):
    ds, s = chosen
    none = {"anchor": {"rule": "none"}}
    r = forecast.run(ds, none, select=s)
    gap = abs(ols_direct(ds, s).resid.iloc[-1])
    assert abs(r.origin_gap) == pytest.approx(gap, rel=1e-10)
    assert not r.gate["reconcile"].passed  # the default tolerance is 1e-6
    at = abs(r.origin_gap)
    for tol, ok in [(at, True), (at * (1 - 1e-9), False), (at * (1 + 1e-9), True)]:
        g = forecast.run(ds, {**none, "tolerance": tol}, select=s).gate["reconcile"]
        assert g.passed is ok, tol
        assert g.observed == at and g.threshold == tol and g.op == "<="


@pytest.mark.parametrize("rule", ["carry", "decay"])
def test_an_anchored_forecast_reconciles_to_the_origin(chosen, rule):
    ds, s = chosen
    r = forecast.run(ds, {"anchor": {"rule": rule}}, select=s)
    assert abs(r.origin_gap) < 1e-12
    assert r.gate["reconcile"].passed


def unit_root():
    ds = scenarios(levels(), h=H)
    f = ds.frame
    f["y"] = 50 + np.cumsum(f["y"])
    return ds.with_frame(f)


def test_a_differenced_target_is_forecast_in_levels_from_the_origin():
    ds = unit_root()
    ex = dz.explore.run(ds)
    assert ex.chains["y"] == ("diff(lag=1)",)
    s = chosen_model(ds, explore=ex)
    r = forecast.run(ds, {"anchor": {"rule": "none"}}, select=s)
    y = ds.frame["y"]
    origin = pd.Period(r.origin, "Q")
    assert r.chain == ("diff(lag=1)",) and r.correction == "none" and r.intervals == "linear"
    # The gap in levels: the observed level against last level plus the fitted change.
    dy, X = s.design.frames(ds.frame)
    fitted_change = sm.OLS(dy, sm.add_constant(X)).fit().fittedvalues.iloc[-1]
    prev = y.loc[origin - 1]
    assert r.origin_gap == pytest.approx(prev + fitted_change - y.loc[origin], rel=1e-10)
    assert r.observed == y.loc[origin]
    for p in r.paths:
        np.testing.assert_allclose(
            p.forecast.mean, y.loc[origin] + np.cumsum(p.raw.mean), rtol=1e-12
        )


# --- bounds -------------------------------------------------------------------------


def test_bounds_count_the_mean_values_outside_them(chosen):
    ds, s = chosen
    r = forecast.run(ds, {"bounds": [None, 1.5]}, select=s)
    values = [v for p in r.paths for v in p.forecast.mean]
    out = sum(1 for v in values if v > 1.5)
    assert out > 0
    assert r.gate["forecast-bounds"].observed == out and not r.gate["forecast-bounds"].passed
    assert forecast.run(ds, {"bounds": [-100, 100]}, select=s).gate["forecast-bounds"].passed


def test_unset_bounds_are_selects_and_without_any_there_is_no_check(chosen):
    ds, s = chosen
    assert "forecast-bounds" not in {c.name for c in forecast.run(ds, select=s).gate.checks}
    bounded = s.model_copy(update={"config": s.config.model_copy(update={"bounds": (None, 1.5)})})
    r = forecast.run(ds, select=bounded)
    assert r.bounds == (None, 1.5) and not r.gate["forecast-bounds"].passed
    r = forecast.run(ds, {"bounds": [None, None]}, select=bounded)
    assert "forecast-bounds" not in {c.name for c in r.gate.checks}


# --- back-transform in the step ---------------------------------------------------


def positive(link):
    ds = scenarios(levels(), h=H)
    f = ds.frame
    f["y"] = np.exp(0.3 * f["y"]) if link == "log" else expit(0.5 * f["y"] - 0.5)
    return ds.with_frame(f)


def test_a_log_target_is_corrected_by_smearing():
    ds = positive("log")
    ex = dz.explore.run(ds, {"transforms": {"y": ["log"]}})
    assert ex.chains["y"] == ("log",)
    s = chosen_model(ds, explore=ex)
    r = forecast.run(ds, {"anchor": {"rule": "none"}}, select=s)
    assert r.correction == "smearing" and r.intervals == "inverse"
    fitted = dz.models.refit(s.record, *s.design.frames(ds.frame))
    factor = np.mean(np.exp(fitted.residuals.to_numpy()))
    for p in r.paths:
        np.testing.assert_allclose(p.forecast.mean, np.exp(p.raw.mean) * factor, rtol=1e-12)


def test_a_logit_target_is_simulated_and_repeats_for_a_seed():
    ds = positive("logit")
    ex = dz.explore.run(ds, {"transforms": {"y": ["logit"]}})
    assert ex.chains["y"] == ("logit",)
    s = chosen_model(ds, explore=ex)
    cfg = {"simulation": {"draws": 500}}
    a = forecast.run(ds, cfg, select=s, seed=4)
    b = forecast.run(ds, cfg, select=s, seed=4)
    close(a.to_dict(), b.to_dict())
    assert a.correction == "simulation" and a.seed == 4 and a.draws == 500
    c = forecast.run(ds, {"simulation": {"draws": 500, "seed": 9}}, select=s, seed=4)
    assert c.seed == 9 and c.paths[0].forecast.mean != a.paths[0].forecast.mean
    assert forecast.run(ds, cfg, select=s).seed == 0


# --- inputs, purity, artefacts ----------------------------------------------------


def test_the_callers_dataset_is_unchanged_and_a_rerun_repeats(chosen):
    ds, s = chosen
    frame, scen, fp = copy.deepcopy(ds.frame), copy.deepcopy(ds.scenarios), ds.fingerprint()
    a = forecast.run(ds, select=s)
    pd.testing.assert_frame_equal(ds.frame, frame)
    pd.testing.assert_frame_equal(ds.scenarios, scen)
    assert ds.fingerprint() == fp
    close(a.to_dict(), forecast.run(ds, select=s).to_dict())


def test_scenarios_come_from_selects_configuration_when_the_dataset_has_none(tmp_path, chosen):
    ds, s = chosen
    frame = ds.scenarios.reset_index()
    frame["period"] = frame["period"].astype(str)
    path = tmp_path / "scenarios.csv"
    frame[["scenario", "period", "a", "b", "c"]].to_csv(path, index=False)
    plain = ds.with_scenarios(None)
    spec = {"path": str(path), "severity": list(ds.severity)}
    from_file = s.model_copy(
        update={"config": s.config.model_copy(update={"scenarios": select.ScenarioInput(**spec)})}
    )
    r = forecast.run(plain, select=from_file)
    want = forecast.run(ds, select=s)
    assert [p.scenario for p in r.paths] == [p.scenario for p in want.paths]
    close([p.forecast.mean for p in r.paths], [p.forecast.mean for p in want.paths])
    with pytest.raises(ForecastError, match="scenarios"):
        forecast.run(plain, select=s.model_copy(update={"config": s.config}))


def test_the_horizon_shortens_the_paths_and_cannot_pass_the_scenarios(chosen):
    ds, s = chosen
    r = forecast.run(ds, {"horizon": 3}, select=s)
    assert r.horizon == 3 and all(p.forecast.h == 3 for p in r.paths)
    with pytest.raises(ForecastError, match="cover"):
        forecast.run(ds, {"horizon": H + 1}, select=s)


def test_refusals(chosen):
    ds, s = chosen
    with pytest.raises(ForecastError, match="select ran on"):
        forecast.run(scenarios(levels(seed=2), h=H), select=s)
    with pytest.raises(ForecastError, match="no model"):
        forecast.run(ds, select=s.model_copy(update={"chosen": None, "record": None}))
    with pytest.raises(ForecastError, match="linear"):
        forecast.run(ds, {"correction": "smearing"}, select=s)
    with pytest.raises(ConfigError):
        forecast.Config.model_validate({"anchor": {"rule": "decay", "decay": 0}})
    with pytest.raises(ConfigError):
        forecast.Config.model_validate({"levels": []})
    with pytest.raises(ConfigError):
        forecast.Config.model_validate({"simulation": {"draws": 10}})


def test_artefacts_finding_and_record(chosen):
    ds, s = chosen
    r = forecast.run(ds, select=s)
    assert list(r.artefacts) == ["T7.1", "T7.2", "T7.3", "F7.1", "S7.1"]
    table = r.artefacts["T7.1"]
    assert len(table.rows) == len(ds.severity) * H
    first = table.rows[0]
    assert list(first[:3]) == ["s0", 1, r.paths[0].forecast.periods[0]]
    assert first[3] == r.paths[0].forecast.mean[0]
    fig = r.artefacts["F7.1"]
    assert {layer.kind for layer in fig.layers} == {"line", "band"}
    assert s.chosen in r.finding.text and r.finding.key == "forecast.summary"
    assert r.chosen == s.chosen and r.record == s.record and r.form == "ols"
    back = forecast.Result.from_dict(r.to_dict())
    assert back.to_dict() == r.to_dict()
    frame = r.frame()
    assert list(frame.columns[:3]) == ["scenario", "horizon", "period"]
    assert len(frame) == len(ds.severity) * H
    assert r.path("s2") == r.paths[2].forecast
