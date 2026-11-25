"""The explore step: series profile, chosen transforms, artefacts and the caller's data.

Oracles: profile measures written out here from their definitions (STL
strength as in Wang, Smith and Hyndman 2006;
standard deviation of changes; share of zeros), the transforms module's
own `forward` for the recorded chains, statsmodels' `acf`, and a deep copy of the input.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa import stattools
from statsmodels.tsa.seasonal import STL

import dazflow as dz
from dazflow import ExploreError
from dazflow.steps import explore

N = 120


def monthly(values, name="x"):
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.period_range("2000-01", periods=len(values), freq="M"),
        name=name,
    )


def project(seed=0, n=N):
    """A target in levels (a random walk with drift) and two drivers: one I(1), one I(0)."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990Q1", periods=n, freq="Q")
    g = rng.normal(size=n)
    level = 100 + np.cumsum(0.2 + g)
    u = rng.normal(size=n)
    y = 50 + np.cumsum(0.1 + 0.7 * g + 0.3 * rng.normal(size=n))
    frame = pd.DataFrame({"y": y, "level": level, "u": u}, index=index)
    meta = {
        "y": {"role": "target", "unit": "index"},
        "level": {"role": "driver", "expected_sign": 1},
        "u": {"role": "driver"},
    }
    return dz.Dataset(frame, meta)


# --- profile ----------------------------------------------------------------------


def test_profile_strengths_equal_their_definition_on_an_stl_fit():
    t = np.arange(N)
    rng = np.random.default_rng(1)
    x = monthly(0.05 * t + 2 * np.sin(2 * np.pi * t / 12) + 0.3 * rng.normal(size=N))
    got = explore.profile(x, "M")
    fit = STL(x.to_numpy(), period=12, seasonal=7).fit()
    r = fit.resid
    assert got.trend_strength == pytest.approx(1 - np.var(r) / np.var(fit.trend + r))
    assert got.seasonal_strength == pytest.approx(1 - np.var(r) / np.var(fit.seasonal + r))
    assert got.trend_strength > 0.9
    assert got.seasonal_strength > 0.9


def test_white_noise_has_weak_trend_and_season():
    got = explore.profile(monthly(np.random.default_rng(2).normal(size=N)), "M")
    assert got.trend_strength < 0.5
    assert got.seasonal_strength < 0.5


def test_profile_autocorrelation_volatility_and_intermittency():
    rng = np.random.default_rng(3)
    values = rng.normal(size=N)
    values[::4] = 0.0  # a quarter of the values are zero
    x = monthly(values)
    got = explore.profile(x, "M")
    acf = stattools.acf(values, nlags=12, fft=False)
    assert got.acf1 == pytest.approx(acf[1], rel=1e-12)
    assert got.acf_seasonal == pytest.approx(acf[12], rel=1e-12)
    assert got.volatility == pytest.approx(np.std(np.diff(values), ddof=1))
    assert got.intermittency == pytest.approx(0.25)
    assert (got.observations, got.missing) == (N, 0)


def test_annual_series_have_no_seasonal_measures():
    x = pd.Series(
        np.random.default_rng(4).normal(size=40),
        index=pd.period_range("1980", periods=40, freq="Y"),
        name="x",
    )
    got = explore.profile(x, "Y")
    assert got.seasonal_strength is None
    assert got.acf_seasonal is None
    assert 0 <= got.trend_strength <= 1


def test_interior_missing_values_are_counted_and_interpolated_for_the_fit():
    rng = np.random.default_rng(5)
    values = np.cumsum(rng.normal(size=N))
    values[[10, 11, 50]] = np.nan
    got = explore.profile(monthly(values), "M")
    assert (got.observations, got.missing) == (N - 3, 3)
    assert got.trend_strength is not None


# --- the step ----------------------------------------------------------------------


def test_the_callers_dataset_is_unchanged_after_run():
    ds = project()
    frame, series, log, fp = (
        ds.frame.copy(),
        copy.deepcopy(dict(ds.series)),
        ds.log,
        ds.fingerprint(),
    )
    explore.run(ds, {"transforms": {"y": ["log"]}})
    pd.testing.assert_frame_equal(ds.frame, frame)
    assert dict(ds.series) == series
    assert ds.log == log
    assert ds.fingerprint() == fp


def test_each_series_is_transformed_by_its_chain_and_the_chain_is_recorded():
    ds = project()
    r = explore.run(ds, {"transforms": {"y": ["log"]}})
    assert r.orders == {"y": 1, "level": 1, "u": 0}
    assert r.chains == {"y": ("log", "diff(lag=1)"), "level": ("diff(lag=1)",), "u": ()}
    for name, chain in r.chains.items():
        assert r.dataset.series[name].transforms == chain
        expected, _ = dz.transforms.forward(ds.frame[name], list(chain))
        pd.testing.assert_series_equal(r.dataset.frame[name], expected, check_names=False)
    assert r.fingerprint == r.dataset.fingerprint()
    assert r.input_fingerprint == ds.fingerprint()


def test_the_stationarity_is_tested_after_the_configured_transforms():
    ds = project()
    r = explore.run(ds, {"transforms": {"y": ["log"]}})
    got = {i.series: i for i in r.integration}["y"]
    direct = explore.integration_order(np.log(ds.frame["y"]).rename("y"))
    assert [s.votes for s in got.steps] == [s.votes for s in direct.steps]


def test_artefacts_have_their_declared_identifiers():
    r = explore.run(project())
    assert list(r.artefacts) == ["T3.1", "T3.2", "T3.3", "T3.4", "F3.1", "S3.1"]
    assert explore.IDS.table("profile") == "T3.1"
    assert explore.IDS.table("stationarity") == "T3.2"
    assert explore.IDS.table("screen") == "T3.3"
    assert explore.IDS.table("gate") == "T3.4"
    assert explore.IDS.figure("lags") == "F3.1"
    assert explore.IDS.finding("summary") == "S3.1"
    assert r.finding == r.artefacts["S3.1"]
    assert [row[0] for row in r.artefacts["T3.1"].rows] == ["y", "level", "u"]


def test_ids_do_not_depend_on_the_drivers_or_their_order():
    ds = project()
    a = explore.run(ds, {"drivers": ["u", "level"]})
    b = explore.run(ds, {"drivers": ["level"]})
    assert list(a.artefacts) == list(b.artefacts)
    assert [layer.label for layer in a.artefacts["F3.1"].layers if layer.kind == "line"] == [
        "u",
        "level",
    ]


def test_without_drivers_there_is_no_screen_figure():
    ds = project().select(["y"])
    r = explore.run(ds)
    assert "F3.1" not in r.artefacts
    assert r.screens == ()
    assert r.gate.passed


def test_the_result_round_trips_without_its_dataset():
    r = explore.run(project())
    back = explore.Result.from_dict(r.to_dict())
    assert back.dataset is None
    assert back.to_dict() == r.to_dict()
    assert "dataset" not in r.to_dict()


def test_the_step_is_registered():
    assert dz.steps.registry.get("explore") is explore
    assert dz.explore is explore


def test_the_finding_counts_the_kept_drivers():
    r = explore.run(project())
    assert r.finding.values["drivers"] == 2
    assert r.finding.values["kept"] == len(r.kept)


def test_panels_and_unknown_transform_series_are_refused():
    ds = project()
    with pytest.raises(ExploreError, match="transforms"):
        explore.run(ds, {"transforms": {"nope": ["log"]}})
    frame = pd.concat({"a": ds.frame, "b": ds.frame}, names=["segment", "period"])
    with pytest.raises(ExploreError, match="panel"):
        explore.run(dz.Dataset(frame, dict(ds.series)))


def test_a_series_too_short_to_test_is_refused():
    ds = project().select(["y"])
    short = dz.Dataset(ds.frame.iloc[:15], dict(ds.series))
    with pytest.raises(ExploreError, match="too short"):
        explore.run(short)


# --- degenerate input ----------------------------------------------------------------


def test_a_constant_or_empty_series_gets_no_strength_or_autocorrelation():
    flat = explore.profile(monthly(np.full(N, 3.0)))  # frequency read from the index
    assert (flat.trend_strength, flat.seasonal_strength, flat.acf1) == (None, None, None)
    assert flat.volatility == 0.0
    empty = explore.profile(monthly(np.full(N, np.nan)))
    assert (empty.observations, empty.trend_strength) == (0, None)


def test_a_constant_series_is_noted_in_the_stationarity_table():
    ds = project()
    frame = ds.frame.assign(flat=1.0)
    r = explore.run(dz.Dataset(frame, {**dict(ds.series), "flat": {"role": "other"}}))
    assert r.orders["flat"] is None
    notes = r.artefacts["T3.2"].notes
    assert any("could not be run on flat" in n and "constant" in n for n in notes)


def test_the_target_is_named_when_there_are_several():
    ds = project()
    two = dz.Dataset(ds.frame, {**dict(ds.series), "level": {"role": "target"}})
    with pytest.raises(ExploreError, match="several targets"):
        explore.run(two)
    r = explore.run(two, {"target": "y", "drivers": ["u"]})
    assert (r.target, r.drivers) == ("y", ("u",))
    with pytest.raises(ExploreError, match="target"):
        explore.run(ds, {"target": "nope"})
