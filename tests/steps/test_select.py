"""The select step: benchmark-scaled scores, the model confidence set, scenarios, the default rule.

Oracles: a direct `arch.bootstrap.MCS` call on the same losses with the same
seed; hand-made candidate tables for the default rule (design D38);
hand-ordered projections for the scenario check; the evaluation rows'
own errors for the scaled scores; hand arithmetic (last level plus the
cumulated changes) for a projection back in levels; a deep copy of the
caller's dataset.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import MCS
from candidate_data import close, levels, scenarios

import dazflow as dz
from dazflow import ConfigError, SelectError
from dazflow.steps import build, select

# --- the model confidence set ----------------------------------------------------------


def losses(seed=0, t=60):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(t, 1))
    noise = rng.normal(size=(t, 4))
    return pd.DataFrame(
        (base + noise) ** 2 + np.array([0.0, 0.05, 0.6, 1.5]), columns=["m1", "m2", "m3", "m4"]
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("size", [0.10, 0.25])
def test_the_confidence_set_equals_a_direct_arch_mcs_call(seed, size):
    L = losses(seed)
    got = select.confidence_set(L, size=size, reps=500, seed=seed)
    direct = MCS(L, size, reps=500, block_size=None, method="R", bootstrap="stationary", seed=seed)
    direct.compute()
    assert set(got.included) == set(direct.included)
    assert got.included == tuple(c for c in L.columns if c in set(direct.included))
    pv = direct.pvalues["Pvalue"]
    assert got.pvalues == pytest.approx({k: float(pv[k]) for k in L.columns})


def test_the_confidence_set_repeats_for_a_seed():
    L = losses(3)
    a = select.confidence_set(L, size=0.1, reps=300, seed=11)
    b = select.confidence_set(L, size=0.1, reps=300, seed=11)
    assert a == b


# --- the default rule (design D38) ------------------------------------------------------


def rows(*specs):
    return [
        select.RuleRow(candidate=c, drivers=k, error=e, passed=p, in_set=s)
        for c, k, e, p, s in specs
    ]


def test_the_lowest_error_wins_when_the_set_is_a_singleton():
    table = rows(
        ("ols:a", 1, 0.80, True, False),
        ("ols:a+b", 2, 0.60, True, True),
        ("ols:b", 1, 0.90, True, False),
    )
    assert select.default_rule(table) == "ols:a+b"


def test_a_smaller_model_in_the_set_wins_over_a_larger_lower_error_one():
    table = rows(
        ("ols:a+b", 2, 0.60, True, True),
        ("ols:a", 1, 0.65, True, True),
        ("ols:b", 1, 0.70, True, True),
        ("ols:a+b+c", 3, 0.55, True, True),
    )
    # Fewest drivers first (a and b), then the lower scaled error.
    assert select.default_rule(table) == "ols:a"


def test_a_failing_model_with_the_lowest_error_is_skipped():
    table = rows(
        ("ols:a", 1, 0.50, False, True),
        ("ols:b", 1, 0.70, True, True),
        ("ols:a+b", 2, 0.65, True, True),
    )
    assert select.default_rule(table) == "ols:b"


def test_no_gate_passing_candidate_in_the_set_proposes_none():
    table = rows(
        ("ols:a", 1, 0.50, False, True),
        ("ols:b", 1, 0.70, True, False),
    )
    assert select.default_rule(table) is None
    assert select.default_rule([]) is None


def test_a_tie_on_drivers_and_error_goes_to_the_first_listed():
    table = rows(("ols:b", 1, 0.7, True, True), ("ols:a", 1, 0.7, True, True))
    assert select.default_rule(table) == "ols:b"


# --- the scenario ordering check ----------------------------------------------------------


SEVERITY = ("baseline", "adverse", "severe")


def test_projections_in_severity_order_pass():
    assert select.out_of_order({"baseline": 1.0, "adverse": 2.0, "severe": 3.5}, SEVERITY) == 0
    check = select.order_check({"baseline": 1.0, "adverse": 2.0, "severe": 3.5}, SEVERITY)
    assert check.passed


def test_one_swapped_pair_fails():
    values = {"baseline": 1.0, "adverse": 3.5, "severe": 2.0}
    assert select.out_of_order(values, SEVERITY) == 1
    assert not select.order_check(values, SEVERITY).passed


def test_a_tie_is_not_out_of_order():
    values = {"baseline": 1.0, "adverse": 2.0, "severe": 2.0}
    assert select.out_of_order(values, SEVERITY) == 0
    assert select.order_check(values, SEVERITY).passed


def test_a_falling_target_reads_severity_downwards():
    values = {"baseline": 3.0, "adverse": 2.0, "severe": 1.0}
    assert select.out_of_order(values, SEVERITY, direction=-1) == 0
    assert select.out_of_order(values, SEVERITY, direction=1) == 2


# --- the step ---------------------------------------------------------------------------

BUILD = {"max_size": 2, "forms": ["ols"]}


@pytest.fixture(scope="module")
def built():
    ds = scenarios(levels())
    return ds, build.run(ds, BUILD)


def quick(**extra):
    """Fewer origins and replications than the defaults, to keep the tests quick."""
    return {"mcs": {"reps": 300}, "splitter_config": {"folds": 20}, **extra}


def test_scores_are_scaled_by_the_benchmarks_and_rank_the_candidates(built):
    ds, b = built
    r = select.run(ds, quick(), build=b)
    bench = r.evaluation(r.benchmark)
    for ranked in r.ranking:
        if ranked.scaled_error is None:
            continue
        ev = r.evaluation(ranked.candidate)
        err = [row.actual - row.forecast for row in ev.rows]
        ref = [row.actual - row.forecast for row in bench.rows]
        assert ranked.scaled_error == pytest.approx(
            np.sqrt(np.mean(np.square(err))) / np.sqrt(np.mean(np.square(ref))), rel=1e-12
        )
    errors = [x.scaled_error for x in r.ranking]
    assert errors == sorted(errors)


def test_the_steps_set_is_arch_mcs_on_its_losses(built):
    ds, b = built
    r = select.run(ds, quick(), build=b, seed=5)
    L = pd.DataFrame(r.losses, index=r.origins)
    assert len(r.origins) == 20
    direct = MCS(L, 0.10, reps=300, method="R", bootstrap="stationary", seed=5)
    direct.compute()
    assert set(r.confidence.included) == set(direct.included)
    assert r.seed == 5
    # Each origin's loss is the squared error of that fold's forecast (one-step folds).
    for cid, values in r.losses.items():
        ev = r.evaluation(cid)
        want = [(row.actual - row.forecast) ** 2 for row in ev.rows]
        assert values == pytest.approx(want, rel=1e-12)


def test_the_seed_comes_from_the_config_before_the_project():
    ds, b = scenarios(levels()), None
    b = build.run(ds, BUILD)
    r = select.run(ds, quick(mcs={"reps": 300, "seed": 9}), build=b, seed=5)
    assert r.seed == 9
    assert select.run(ds, quick(), build=b).seed == 0


def test_the_default_rule_chooses_and_the_gate_passes(built):
    ds, b = built
    r = select.run(ds, quick(), build=b)
    table = [
        select.RuleRow(
            candidate=x.candidate,
            drivers=x.size,
            error=x.scaled_error,
            passed=x.eligible,
            in_set=x.in_set,
        )
        for x in r.ranking
        if x.scaled_error is not None
    ]
    assert r.proposal == select.default_rule(table) is not None
    assert r.chosen == r.proposal and r.decision == "proposed"
    chosen = next(c for c in b.candidates if c.id == r.chosen)
    assert r.record == chosen.record and r.drivers == chosen.drivers
    assert r.design == chosen.design
    assert r.gate.passed
    assert {c.name for c in r.gate.checks} == {
        "confidence-set",
        "build",
        "benchmark",
        "scenario-order",
    }


def test_each_candidate_is_forecast_under_every_scenario(built):
    ds, b = built
    r = select.run(ds, quick(), build=b)
    evaluated = [x.candidate for x in r.ranking if x.scaled_error is not None]
    pairs = {(s.candidate, s.scenario) for s in r.scenario_forecasts}
    assert pairs == {(c, s) for c in evaluated for s in ds.severity}
    # The data are stationary (no transform), so the projection in levels is the mean.
    one = next(s for s in r.scenario_forecasts if s.candidate == r.chosen and s.scenario == "s2")
    assert one.levels == one.forecast.mean
    assert one.forecast.periods[0] == str(ds.frame.index[-1] + 1)


def test_the_projection_is_mapped_back_to_levels_through_the_chain():
    ds = scenarios(levels())
    f = ds.frame
    f["y"] = 50 + np.cumsum(f["y"])  # a target in levels with a unit root
    ds = ds.with_frame(f)
    ex = dz.explore.run(ds)
    assert ex.chains["y"] == ("diff(lag=1)",)
    b = build.run(ds, BUILD, explore=ex)
    r = select.run(ds, quick(), build=b)
    one = next(s for s in r.scenario_forecasts if s.candidate == r.chosen and s.scenario == "s1")
    expected = f["y"].iloc[-1] + np.cumsum(one.forecast.mean)
    np.testing.assert_allclose(one.levels, expected, rtol=1e-12)


def test_scenario_order_is_checked_on_the_chosen_model(built):
    ds, b = built
    r = select.run(ds, quick(), build=b)
    assert r.gate["scenario-order"].passed  # a rises and b falls with severity
    flipped = ds.with_scenarios(ds.scenarios, ["s2", "s1", "s0"])
    r = select.run(flipped, quick(), build=b)
    assert r.proposal is None  # every driver model now projects out of order
    assert not r.gate["confidence-set"].passed
    assert "no candidate" in r.finding.text


def test_bounds_fail_a_projection_outside_them(built):
    ds, b = built
    r = select.run(ds, quick(bounds=[None, 1.5]), build=b)
    for x in r.ranking:
        if x.scaled_error is None:
            continue
        values = [v for f in r.scenario_forecasts if f.candidate == x.candidate for v in f.levels]
        assert x.out_of_bounds == sum(1 for v in values if v > 1.5)
    # Models with a or b project about 2.4 and 3.7 under s1 and s2: out of bounds.
    moved = [x for x in r.ranking if {"a", "b"} & set(x.drivers)]
    assert moved and all(x.out_of_bounds > 0 for x in moved)
    assert r.proposal is None
    r = select.run(ds, quick(bounds=[-100, 100]), build=b)
    assert r.gate["scenario-bounds"].observed == 0 and r.gate["scenario-bounds"].passed


def test_without_scenarios_there_is_no_scenario_check(built):
    ds, b = built
    plain = ds.with_scenarios(None)
    r = select.run(plain, quick(), build=b)
    assert r.scenario_forecasts == ()
    assert "scenario-order" not in {c.name for c in r.gate.checks}


def test_an_override_without_a_reason_is_refused():
    with pytest.raises(ConfigError, match="reason"):
        select.Config.model_validate({"choice": {"candidate": "ols:a"}})
    with pytest.raises(ConfigError, match="reason"):
        select.Config.model_validate({"choice": {"candidate": "ols:a", "reason": "  "}})


def test_the_modeller_confirms_or_overrides_with_a_reason(built):
    ds, b = built
    proposed = select.run(ds, quick(), build=b)
    same = select.run(
        ds, quick(choice={"candidate": proposed.proposal, "reason": "Agreed."}), build=b
    )
    assert same.decision == "confirmed" and same.chosen == proposed.proposal
    other = next(
        x.candidate
        for x in proposed.ranking
        if x.role == "candidate" and x.candidate != proposed.proposal
    )
    over = select.run(ds, quick(choice={"candidate": other, "reason": "Simpler story."}), build=b)
    assert over.decision == "overridden" and over.chosen == other
    assert over.reason == "Simpler story."
    assert over.record == next(c for c in b.candidates if c.id == other).record
    with pytest.raises(SelectError, match="nope"):
        select.run(ds, quick(choice={"candidate": "nope", "reason": "x"}), build=b)


def test_a_candidate_whose_fold_is_too_short_is_dropped_with_its_reason(built):
    ds, _ = built
    b = build.run(
        ds, {"max_size": 1, "forms": [{"form": "arimax", "config": {"order": [6, 0, 0]}}]}
    )
    r = select.run(ds, quick(splitter_config={"initial": 8}), build=b)
    dropped = [x for x in r.ranking if x.role == "candidate"]
    assert dropped and all(x.scaled_error is None and not x.in_set for x in dropped)
    assert all("needs at least" in x.note for x in dropped)
    assert all(x.candidate not in r.losses for x in dropped)
    assert r.proposal is None


def test_the_benchmark_must_be_one_build_fitted(built):
    ds, b = built
    with pytest.raises(SelectError, match="seasonal_naive"):
        select.run(ds, quick(benchmark="seasonal_naive"), build=b)


def test_the_dataset_must_be_the_one_build_ran_on(built):
    _, b = built
    with pytest.raises(SelectError, match="build"):
        select.run(levels(seed=1), quick(), build=b)


def test_artefacts_are_declared(built):
    ds, b = built
    r = select.run(ds, quick(), build=b)
    assert set(r.artefacts) == {
        "T6.1",
        "T6.2",
        "T6.3",
        "T6.4",
        "T6.5",
        "F6.1",
        "F6.2",
        "S6.1",
    }


def test_the_callers_dataset_is_unchanged_and_the_run_is_deterministic(built):
    ds, b = built
    frame, scen, fp = copy.deepcopy(ds.frame), copy.deepcopy(ds.scenarios), ds.fingerprint()
    first = select.run(ds, quick(), build=b, seed=3)
    second = select.run(ds, quick(), build=b, seed=3)
    pd.testing.assert_frame_equal(ds.frame, frame)
    pd.testing.assert_frame_equal(ds.scenarios, scen)
    assert ds.fingerprint() == fp
    # The choice, the set and its p-values repeat exactly; every number to 1e-12.
    assert (first.proposal, first.chosen, first.confidence) == (
        second.proposal,
        second.chosen,
        second.confidence,
    )
    assert first.losses == second.losses
    close(first.to_dict(), second.to_dict())
    assert select.Result.from_dict(first.to_dict()) == first


def test_scenarios_are_read_from_a_file(tmp_path, built):
    ds, b = built
    path = tmp_path / "scen.csv"
    frame = ds.scenarios.reset_index()
    frame["period"] = frame["period"].astype(str)
    frame.rename(columns={"period": "date"}).to_csv(path, index=False)
    plain = ds.with_scenarios(None)
    cfg = quick(scenarios={"path": str(path), "severity": ["s0", "s1", "s2"]})
    r = select.run(plain, cfg, build=b)
    direct = select.run(ds, quick(), build=b)
    close(
        [f.model_dump() for f in r.scenario_forecasts],
        [f.model_dump() for f in direct.scenario_forecasts],
    )
    assert r.severity == ("s0", "s1", "s2")


def test_one_model_is_its_own_confidence_set():
    got = select.confidence_set(losses()[["m1"]], seed=0)
    assert got.included == ("m1",) and got.pvalues == {"m1": 1.0}


def test_absolute_loss_scales_by_the_mae(built):
    ds, b = built
    r = select.run(ds, quick(loss="absolute"), build=b)
    bench = r.evaluation("mean")
    for x in r.ranking:
        ev = r.evaluation(x.candidate)
        mae = np.mean([abs(row.actual - row.forecast) for row in ev.rows])
        ref = np.mean([abs(row.actual - row.forecast) for row in bench.rows])
        assert x.scaled_error == pytest.approx(mae / ref, rel=1e-12)


def test_a_candidate_build_could_not_fit_is_listed_with_its_error(built):
    ds, _ = built
    big = {"form": "arimax", "config": {"order": [98, 0, 0]}}
    b = build.run(ds, {"max_size": 1, "forms": [big, "ols"]})
    r = select.run(ds, quick(), build=b)
    row = next(x for x in r.ranking if x.candidate == "arimax:a")
    assert row.scaled_error is None and "needs at least" in row.note
    assert r.evaluation("arimax:a") is None


def test_a_benchmark_that_cannot_be_evaluated_is_refused(built):
    ds, b = built
    with pytest.raises(SelectError, match="benchmark mean"):
        select.run(ds, quick(splitter_config={"initial": 1}), build=b)


def test_scenarios_must_reach_past_the_sample(built):
    ds, b = built
    early = ds.scenarios.reset_index()
    early["period"] = early["period"] - 20
    early = early.set_index(["scenario", "period"])
    with pytest.raises(SelectError, match="no period after"):
        select.run(ds.with_scenarios(early, ds.severity), quick(), build=b)


def test_a_scenarios_file_must_exist_and_name_its_label_column(tmp_path, built):
    ds, b = built
    plain = ds.with_scenarios(None)
    with pytest.raises(SelectError, match="does not exist"):
        select.run(plain, quick(scenarios={"path": str(tmp_path / "none.csv")}), build=b)
    path = tmp_path / "scen.csv"
    path.write_text("date,a,b,c\n2020Q1,1,1,1\n")
    with pytest.raises(SelectError, match="no column 'scenario'"):
        select.run(plain, quick(scenarios={"path": str(path)}), build=b)
