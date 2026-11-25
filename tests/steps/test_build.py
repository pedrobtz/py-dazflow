"""The build step: combination search, VIF, forms by complexity, the coefficient gate.

Oracles: `math.comb` and a brute-force enumeration over every subset for the
combination search; statsmodels' `variance_inflation_factor` for VIF;
statsmodels' OLS (HAC) p-values for the significance gate; the explore
step's own transformed dataset for the design matrices; a deep copy of the
caller's dataset.
"""

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from candidate_data import close, levels
from statsmodels.stats.outliers_influence import variance_inflation_factor

import dazflow as dz
from dazflow import BuildError, ConfigError
from dazflow.steps import build

NAMES = ["a", "b", "c", "d", "e", "f"]


# --- combination search ------------------------------------------------------------


@pytest.mark.parametrize(("n", "k"), [(n, k) for n in range(1, 7) for k in range(1, n + 2)])
def test_the_number_of_combinations_is_the_binomial_sum(n, k):
    combos = build.combinations(NAMES[:n], k)
    assert len(combos) == sum(math.comb(n, j) for j in range(1, min(k, n) + 1))
    assert len(set(combos)) == len(combos)
    assert all(1 <= len(c) <= k for c in combos)


def test_combinations_are_ordered_by_size_then_by_the_drivers_order():
    assert build.combinations(["x", "y", "z"], 2) == (
        ("x",),
        ("y",),
        ("z",),
        ("x", "y"),
        ("x", "z"),
        ("y", "z"),
    )


@pytest.mark.parametrize("seed", range(5))
def test_excluded_pairs_never_appear_and_every_other_set_does(seed):
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(NAMES, 2))
    excluded = [pairs[i] for i in rng.choice(len(pairs), size=4, replace=False)]
    combos = build.combinations(NAMES, 4, excluded)
    for c in combos:
        for p, q in excluded:
            assert not (p in c and q in c), (c, (p, q))
    # Brute force: every subset of size 1..4 holding no excluded pair, by bitmask.
    expected = set()
    for mask in range(1, 1 << len(NAMES)):
        chosen = tuple(NAMES[i] for i in range(len(NAMES)) if mask >> i & 1)
        if len(chosen) <= 4 and not any(p in chosen and q in chosen for p, q in excluded):
            expected.add(chosen)
    assert set(combos) == expected


def test_an_exclusion_naming_an_unknown_driver_is_refused():
    with pytest.raises(BuildError, match="zz"):
        build.combinations(["a", "b"], 2, [("a", "zz")])


def test_stepwise_adds_the_driver_with_the_lowest_aic_first():
    ds = levels(seed=3)
    f = ds.frame
    y, X = f["y"], f[["a", "b", "c"]]
    path = build.stepwise(y, X, 2)
    aic = {d: sm.OLS(y, sm.add_constant(X[[d]])).fit().aic for d in X}
    first = min(aic, key=aic.get)
    assert path[0] == (first,)
    pair = {d: sm.OLS(y, sm.add_constant(X[[first, d]])).fit().aic for d in X if d != first}
    assert path[1] == (first, min(pair, key=pair.get))


def test_stepwise_never_joins_an_excluded_pair():
    ds = levels(seed=3)
    f = ds.frame
    path = build.stepwise(f["y"], f[["a", "b", "c"]], 3, [("a", "b")])
    assert all(not ("a" in s and "b" in s) for s in path)


# --- VIF and correlation -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_vif_equals_statsmodels_variance_inflation_factor(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(80, 4))
    X = pd.DataFrame(
        {"a": z[:, 0], "b": 0.7 * z[:, 0] + z[:, 1], "c": z[:, 2] - 0.4 * z[:, 1], "d": z[:, 3]}
    )
    got = build.vif(X)
    exog = sm.add_constant(X).to_numpy()
    expected = [variance_inflation_factor(exog, i + 1) for i in range(4)]
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-10)
    assert list(got.index) == list(X.columns)


def test_vif_of_one_driver_is_one():
    X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=30)})
    assert build.vif(X).iloc[0] == pytest.approx(1.0, abs=1e-12)


# --- the step ------------------------------------------------------------------------


def test_every_combination_is_a_candidate_and_benchmarks_are_always_fitted():
    result = build.run(levels(), {"max_size": 2, "forms": ["ols"]})
    sets = [c.drivers for c in result.candidates if c.role == "candidate"]
    assert sets == list(build.combinations(["a", "b", "c"], 2))
    bench = [c for c in result.candidates if c.role == "benchmark"]
    assert [c.form for c in bench] == ["naive", "mean"]
    assert all(c.record is not None and c.passed for c in bench)


def test_benchmarks_are_fitted_even_with_no_driver():
    ds = levels().select(["y"])
    result = build.run(ds)
    assert [c.id for c in result.candidates] == ["naive", "mean"]
    assert not result.gate.passed  # no candidate with drivers passes


def test_excluded_pairs_are_never_candidates():
    result = build.run(levels(), {"max_size": 3, "forms": ["ols"], "exclusions": [["a", "b"]]})
    sets = [set(c.drivers) for c in result.candidates]
    assert sets and not any({"a", "b"} <= s for s in sets)


def test_significance_equals_statsmodels_ols_hac_pvalues():
    ds = levels()
    result = build.run(ds, {"max_size": 2})  # the default forms: ols with HAC errors first
    cand = result.candidate("ols:a+c")
    assert cand.record.spec.config["cov_type"] == "HAC"
    f = ds.frame
    X = sm.add_constant(f[["a", "c"]])
    lags = dz.models.newey_west_lags(len(f))
    fit = sm.OLS(f["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    got = {c.term: c.pvalue for c in cand.coefficients}
    np.testing.assert_allclose([got["a"], got["c"]], fit.pvalues[["a", "c"]], rtol=1e-10)
    # c is noise: its significance check fails, so the candidate fails the gate.
    assert cand.gate["significance:c"].status == "fail"
    assert not cand.passed


def test_a_wrong_sign_fails_the_candidate():
    result = build.run(levels(), {"max_size": 1, "forms": ["ols"], "expected_signs": {"a": -1}})
    a = next(c for c in result.candidates if c.drivers == ("a",))
    assert a.gate["sign:a"].status == "fail" and not a.passed
    b = next(c for c in result.candidates if c.drivers == ("b",))
    assert b.passed


def test_forms_move_up_only_when_the_simpler_form_fails():
    result = build.run(levels(), {"max_size": 1, "forms": ["ols", "arimax"]})
    by_set = {}
    for c in result.candidates:
        by_set.setdefault(c.drivers, []).append(c.form)
    assert by_set[("a",)] == ["ols"]  # ols passes: arimax is not fitted
    assert by_set[("b",)] == ["ols"]
    assert by_set[("c",)] == ["ols", "arimax"]  # c is noise: ols fails, arimax is tried


def test_the_multicollinearity_gate_fails_nearly_collinear_drivers():
    ds = levels()
    f = ds.frame
    f["a2"] = f["a"] + 1e-3 * np.random.default_rng(9).normal(size=len(f))
    meta = {**{k: v.model_dump() for k, v in ds.series.items()}, "a2": {"role": "driver"}}
    ds = dz.Dataset(f, meta)
    result = build.run(ds, {"drivers": ["a", "a2"], "max_size": 2, "forms": ["ols"]})
    pair = next(c for c in result.candidates if c.drivers == ("a", "a2"))
    assert pair.max_vif > 1000
    assert pair.gate["vif"].status == "fail"
    assert result.vif["a"] == pytest.approx(pair.max_vif, rel=1e-9)


def test_a_candidate_check_can_be_overridden_with_a_reason():
    result = build.run(
        levels(),
        {
            "max_size": 1,
            "forms": ["ols"],
            "expected_signs": {"c": 0},
            "overrides": [{"check": "ols:c/significance:c", "reason": "Kept for the document."}],
        },
    )
    c = next(c for c in result.candidates if c.id == "ols:c")
    assert c.gate["significance:c"].status == "overridden" and c.passed


def test_an_override_naming_no_check_is_refused():
    with pytest.raises(dz.GateError, match="nothing"):
        build.run(levels(), {"forms": ["ols"], "overrides": [{"check": "nothing", "reason": "x"}]})


def test_the_gate_passes_with_one_passing_candidate_and_counts_them():
    result = build.run(levels(), {"max_size": 2, "forms": ["ols"]})
    passing = [c.id for c in result.candidates if c.role == "candidate" and c.passed]
    assert result.passing == tuple(passing)
    assert result.gate["candidates"].observed == len(passing) >= 1
    assert result.gate.passed


def test_artefacts_are_declared_and_complete():
    result = build.run(levels(), {"max_size": 2, "forms": ["ols"]})
    assert set(result.artefacts) == {"T4.1", "T4.2", "T4.3", "T4.4", "F4.1", "S4.1"}
    table = result.artefacts["T4.2"]
    assert len(table.rows) == len(result.candidates)
    corr = result.artefacts["T4.1"]
    assert [r[0] for r in corr.rows] == ["a", "b", "c"]


def test_the_callers_dataset_is_unchanged_and_the_run_repeats():
    ds = levels()
    before = copy.deepcopy(ds.frame), ds.fingerprint()
    first = build.run(ds, {"max_size": 2, "forms": ["ols", "arimax"]})
    second = build.run(ds, {"max_size": 2, "forms": ["ols", "arimax"]})
    pd.testing.assert_frame_equal(ds.frame, before[0])
    assert ds.fingerprint() == before[1]
    assert [c.id for c in first.candidates] == [c.id for c in second.candidates]
    assert first.passing == second.passing
    close(first.to_dict(), second.to_dict())  # floats to 1e-12: an ulp moves between runs
    assert build.Result.from_dict(first.to_dict()) == first


def test_a_bad_form_or_size_is_a_config_error():
    with pytest.raises(ConfigError):
        build.Config(max_size=0)
    with pytest.raises(BuildError, match="no_such_form"):
        build.run(levels(), {"forms": ["no_such_form"]})
    with pytest.raises(BuildError, match="naive"):
        build.run(levels(), {"forms": ["naive"]})  # takes no drivers


# --- with explore: transforms, lags and trend terms ------------------------------------


def integrated(n=120, seed=0):
    """A target in levels with a unit root, driven by a stationary `g` and the changes in `u`."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990Q1", periods=n, freq="Q")
    g = rng.normal(size=n)
    du = rng.normal(size=n)
    u = 5 + np.cumsum(du)
    y = 10 + np.cumsum(-0.4 * g + 0.6 * du + 0.2 * rng.normal(size=n))
    frame = pd.DataFrame({"y": y, "g": g, "u": u}, index=index)
    meta = {
        "y": {"role": "target"},
        "g": {"role": "driver", "expected_sign": -1},
        "u": {"role": "driver", "expected_sign": 1},
    }
    return dz.Dataset(frame, meta)


def test_build_fits_on_the_series_explore_transformed():
    ds = integrated()
    ex = dz.explore.run(ds)
    result = build.run(ds, {"forms": ["ols"]}, explore=ex)
    assert result.chains == ex.chains and result.drivers == ex.kept
    cand = next(c for c in result.candidates if c.drivers == ("g", "u"))
    y, X = cand.design.frames(ds.frame)
    moved = ex.dataset.frame
    pd.testing.assert_series_equal(y, moved["y"].loc[y.index], check_names=False)
    pd.testing.assert_frame_equal(X, moved[["g", "u"]].loc[y.index])
    assert cand.record.data == dz.models.fingerprint(y, X)
    assert str(y.index[0]) == result.sample.start


def test_build_refuses_a_dataset_explore_did_not_run_on():
    ds = integrated()
    ex = dz.explore.run(ds)
    with pytest.raises(BuildError, match="explore"):
        build.run(integrated(seed=1), explore=ex)


def test_lags_from_explore_shift_the_driver():
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    y = 0.8 * np.roll(x, 2) + 0.3 * rng.normal(size=n)
    index = pd.period_range("1990Q1", periods=n, freq="Q")
    ds = dz.Dataset(
        pd.DataFrame({"y": y, "x": x}, index=index).iloc[2:],
        {"y": {"role": "target"}, "x": {"role": "driver", "expected_sign": 1}},
    )
    ex = dz.explore.run(ds)
    assert ex.lags == {"x": 2}
    result = build.run(ds, {"forms": ["ols"]}, explore=ex)
    cand = next(c for c in result.candidates if c.drivers == ("x",))
    _, X = cand.design.frames(ds.frame)
    f = ds.frame
    np.testing.assert_array_equal(X["x"].to_numpy(), f["x"].shift(2).loc[X.index].to_numpy())
    assert cand.design.lags == {"x": 2}


def trending(n=120, seed=0):
    """A trend-stationary target and driver: both a line plus stationary noise."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990Q1", periods=n, freq="Q")
    t = np.arange(n)
    x = 0.05 * t + rng.normal(size=n)
    y = 2 + 0.03 * t + 0.7 * (x - 0.05 * t) + 0.3 * rng.normal(size=n)
    return dz.Dataset(
        pd.DataFrame({"y": y, "x": x}, index=index),
        {"y": {"role": "target"}, "x": {"role": "driver", "expected_sign": 1}},
    )


def test_a_trend_stationary_series_kept_in_levels_gets_a_trend_term():
    ds = trending()
    ex = dz.explore.run(ds, {"stationarity": {"trend_stationary": "keep"}})
    kinds = {i.series: i.kind for i in ex.integration}
    assert kinds["y"] == "trend-stationary" and ex.chains["y"] == ()
    result = build.run(ds, {"forms": ["ols"]}, explore=ex)
    cand = next(c for c in result.candidates if c.drivers == ("x",))
    assert cand.design.trend
    assert cand.record.spec.drivers == ("x", "trend")
    _y, X = cand.design.frames(ds.frame)
    pos = np.array([(p - ds.frame.index[0]).n + 1 for p in X.index], dtype=float)
    np.testing.assert_array_equal(X["trend"].to_numpy(), pos)
    assert "trend" not in {c.term for c in cand.coefficients if c.gated}
    # Differenced instead (the default), no series needs a trend.
    ex = dz.explore.run(ds)
    result = build.run(ds, {"forms": ["ols"]}, explore=ex)
    assert not any(c.design.trend for c in result.candidates)


def test_the_design_continues_the_trend_and_lags_into_the_future():
    ds = trending()
    ex = dz.explore.run(ds, {"stationarity": {"trend_stationary": "keep"}})
    cand = build.run(ds, {"forms": ["ols"]}, explore=ex).candidate("ols:x")
    last = ds.frame.index[-1]
    periods = pd.period_range(last + 1, periods=3, freq=last.freq)
    paths = pd.DataFrame({"x": [7.0, 8.0, 9.0]}, index=periods)
    Xf = cand.design.future(ds.frame, paths, 3)
    np.testing.assert_array_equal(Xf["x"].to_numpy(), [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(Xf["trend"].to_numpy(), [121.0, 122.0, 123.0])
    with pytest.raises(dz.ModelError, match="does not cover"):
        cand.design.future(ds.frame, paths.iloc[:2], 3)
    with pytest.raises(dz.ModelError, match="no driver 'x'"):
        cand.design.future(ds.frame, paths.rename(columns={"x": "z"}), 3)
    with pytest.raises(dz.ModelError, match="no series 'x'"):
        cand.design.frames(ds.frame[["y"]])


def test_stepwise_search_gives_the_candidates():
    ds = levels(seed=3)
    result = build.run(ds, {"search": "stepwise", "max_size": 3, "forms": ["ols"]})
    f = ds.frame
    path = build.stepwise(f["y"], f[["a", "b", "c"]], 3)
    assert result.combinations == path
    assert [c.drivers for c in result.candidates if c.role == "candidate"] == list(path)


def test_a_form_that_cannot_be_fitted_is_recorded_and_the_next_one_tried():
    big = {"form": "arimax", "config": {"order": [98, 0, 0]}}
    result = build.run(levels(), {"max_size": 1, "forms": [big, "ols"]})
    first = result.candidate("arimax:a")
    assert first.record is None and "needs at least" in first.error and not first.passed
    assert result.candidate("ols:a").passed


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"target": "nope"}, "nope"),
        ({"drivers": ["a", "a"]}, "repeated"),
        ({"drivers": ["y"]}, "the target"),
        ({"expected_signs": {"zz": 1}}, "zz"),
        ({"lags": {"zz": 1}}, "zz"),
    ],
)
def test_names_that_are_not_drivers_are_refused(config, match):
    with pytest.raises(BuildError, match=match):
        build.run(levels(), config)


def test_two_targets_and_no_name_is_refused():
    ds = levels().with_series("c", role="target")
    with pytest.raises(BuildError, match="one target"):
        build.run(ds)


def test_a_repeated_benchmark_is_a_config_error():
    with pytest.raises(ConfigError, match="once"):
        build.Config(benchmarks=["mean", "mean"])


def test_a_gap_inside_the_sample_is_refused():
    ds = levels()
    f = ds.frame
    f.iloc[50, f.columns.get_loc("a")] = np.nan
    with pytest.raises(BuildError, match="missing inside"):
        build.run(ds.with_frame(f))


def test_an_override_for_a_candidate_not_fitted_is_refused():
    with pytest.raises(dz.GateError, match="arimax:a"):
        build.run(
            levels(),
            {"forms": ["ols", "arimax"], "overrides": [{"check": "arimax:a/vif", "reason": "x"}]},
        )
