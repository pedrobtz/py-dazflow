"""The clean step: checks before and after treatment, the treatment log, the gate, artefacts.

The treatment-log oracle is a replay written here, independently of the
library: start from the raw frame, write each logged cell's recorded value,
and the cleaned frame must come out, with no cell changed that the log does
not name.
"""

import numpy as np
import pandas as pd
import pytest

import dazflow as dz
from dazflow import CleanError, ConfigError
from dazflow.steps import clean

NAN = np.nan
TREATMENTS = ("fill", "interpolate", "winsorise", "replace", "exclude")


def raw(n=40):
    index = pd.period_range("2000Q1", periods=n, freq="Q")
    t = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "y": 100.0 + t + 0.5 * np.sin(t),
            "x": 50.0 + 0.5 * t + 0.3 * np.cos(t),
        },
        index=index,
    )
    frame.loc["2003Q2", "y"] = NAN  # a gap of one period
    frame.loc["2005Q1", "y"] = 400.0  # a spike
    frame.loc["2006Q1":"2006Q2", "x"] = NAN  # a gap of two periods
    return dz.Dataset(frame, {"y": {"role": "target"}, "x": {"role": "driver"}})


TREATED = [
    {"action": "interpolate", "series": ["y"], "reason": "One quarter lost in the source."},
    {"action": "fill", "series": ["x"], "method": "previous", "reason": "Carried forward."},
    {
        "action": "replace",
        "series": ["y"],
        "periods": ["2005Q1"],
        "reason": "Keying error; confirmed with the source.",
    },
    {
        "action": "winsorise",
        "series": ["x"],
        "lower": 0.05,
        "upper": 0.95,
        "reason": "Tails damped.",
    },
    {
        "action": "exclude",
        "start": "2008Q4",
        "end": "2009Q1",
        "reason": "Strike quarters; not representative.",
    },
]


def run(treatments=(), **cfg):
    return clean.run(raw(), {"treatments": list(treatments), **cfg})


# --- the treatment log -----------------------------------------------------------


def test_applying_the_logged_treatments_to_the_raw_frame_reproduces_the_cleaned_frame():
    before = raw()
    r = clean.run(before, {"treatments": TREATED})
    cleaned = r.dataset.frame
    logged = [e for e in r.dataset.log if e.action in TREATMENTS]
    assert [e.action for e in logged] == [
        "interpolate",
        "fill",
        "replace",
        "winsorise",
        "exclude",
        "exclude",
    ]
    replay = before.frame
    touched = set()
    for entry in logged:
        (column,) = entry.columns
        for period in entry.rows:
            (rec,) = [
                t
                for t in r.treatments
                if t.action == entry.action and t.series == column and t.period == period
            ]
            replay.loc[period, column] = NAN if rec.after is None else rec.after
            touched.add((period, column))
    pd.testing.assert_frame_equal(replay, cleaned)
    # Every changed cell is logged, and every logged cell changed.
    diff = before.frame.compare(cleaned, keep_shape=True)
    changed = {
        (str(p), c)
        for p in before.frame.index
        for c in before.frame.columns
        if not _same(before.frame.loc[p, c], cleaned.loc[p, c])
    }
    assert changed == touched
    assert diff.notna().any().any()


def _same(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


def test_each_treatment_is_logged_with_its_reason_rows_and_values():
    r = run(TREATED)
    entries = {(e.action, e.columns): e for e in r.dataset.log if e.action in TREATMENTS}
    assert entries[("interpolate", ("y",))].rows == ("2003Q2",)
    assert entries[("interpolate", ("y",))].reason == "One quarter lost in the source."
    assert entries[("fill", ("x",))].rows == ("2006Q1", "2006Q2")
    assert entries[("replace", ("y",))].rows == ("2005Q1",)
    assert entries[("exclude", ("y",))].rows == ("2008Q4", "2009Q1")
    frame = r.dataset.frame
    # Interpolated halfway between its neighbours; filled from the last value.
    assert frame.loc["2003Q2", "y"] == pytest.approx(
        (frame.loc["2003Q1", "y"] + frame.loc["2003Q3", "y"]) / 2
    )
    assert frame.loc["2006Q2", "x"] == raw().frame.loc["2005Q4", "x"]
    # Replaced without a value: interpolated from its neighbours.
    assert frame.loc["2005Q1", "y"] == pytest.approx(
        (frame.loc["2004Q4", "y"] + frame.loc["2005Q2", "y"]) / 2
    )
    assert frame.loc["2008Q4":"2009Q1"].isna().all().all()
    rec = [t for t in r.treatments if t.action == "replace"]
    assert [(t.period, t.before) for t in rec] == [("2005Q1", 400.0)]


def test_winsorise_clips_to_the_quantiles_of_the_series():
    r = run([TREATED[3]])
    x = raw().frame["x"]
    lo, hi = np.nanquantile(x, [0.05, 0.95])
    out = r.dataset.frame["x"]
    assert out.min() == pytest.approx(lo)
    assert out.max() == pytest.approx(hi)
    expected = x.clip(lo, hi)
    pd.testing.assert_series_equal(out, expected)


def test_fill_with_a_value_and_from_the_next_period():
    r = run([{"action": "fill", "series": ["x"], "method": "value", "value": 0.0, "reason": "r"}])
    assert r.dataset.frame.loc["2006Q1":"2006Q2", "x"].tolist() == [0.0, 0.0]
    r = run([{"action": "fill", "series": ["x"], "method": "next", "reason": "r"}])
    assert r.dataset.frame.loc["2006Q1", "x"] == raw().frame.loc["2006Q3", "x"]


def test_a_treatment_that_changes_nothing_is_still_logged():
    r = run(
        [{"action": "fill", "series": ["y"], "start": "2000Q1", "end": "2000Q4", "reason": "r"}]
    )
    (entry,) = [e for e in r.dataset.log if e.action == "fill"]
    assert entry.rows == ()
    assert [t.period for t in r.treatments] == [None]


def test_the_callers_dataset_is_unchanged():
    ds = raw()
    frame, fingerprint, log = ds.frame, ds.fingerprint(), ds.log
    r = clean.run(ds, {"treatments": TREATED})
    pd.testing.assert_frame_equal(ds.frame, frame)
    assert ds.fingerprint() == fingerprint
    assert ds.log == log
    assert r.dataset.fingerprint() != fingerprint
    assert r.input_fingerprint == fingerprint


# --- configuration -----------------------------------------------------------------


@pytest.mark.parametrize("action", TREATMENTS)
@pytest.mark.parametrize("reason", [None, "", "   "])
def test_a_treatment_without_a_reason_is_refused(action, reason):
    t = {"action": action, "periods": ["2005Q1"]}
    if reason is not None:
        t["reason"] = reason
    with pytest.raises(ConfigError, match="reason"):
        clean.Config.model_validate({"treatments": [t]})
    with pytest.raises(ConfigError, match="reason"):
        clean.run(raw(), {"treatments": [t]})


def test_a_treatment_from_yaml_without_a_reason_is_refused():
    text = "treatments:\n  - action: exclude\n    periods: ['2005Q1']\n"
    with pytest.raises(ConfigError, match="reason"):
        clean.Config.from_yaml(text)


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "replace", "reason": "r"},  # no periods
        {"action": "exclude", "reason": "r"},
        {"action": "fill", "method": "value", "reason": "r"},  # no value
        {"action": "winsorise", "lower": 0.9, "upper": 0.1, "reason": "r"},
        {"action": "exclude", "periods": ["2005Q1"], "start": "2004Q1", "reason": "r"},
        {"action": "trim", "reason": "r"},
    ],
)
def test_invalid_treatments_are_refused(bad):
    with pytest.raises(ConfigError):
        clean.Config.model_validate({"treatments": [bad]})


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"action": "exclude", "series": ["z"], "periods": ["2005Q1"], "reason": "r"}, "z"),
        ({"action": "exclude", "periods": ["1990Q1"], "reason": "r"}, "1990Q1"),
        ({"action": "exclude", "periods": ["not a period"], "reason": "r"}, "not a period"),
    ],
)
def test_a_treatment_naming_what_the_data_lack_is_refused(bad, match):
    with pytest.raises(CleanError, match=match):
        run([bad])


def test_a_replace_without_neighbours_is_refused():
    with pytest.raises(CleanError, match="2000Q1"):
        run([{"action": "replace", "series": ["y"], "periods": ["2000Q1"], "reason": "r"}])


def test_a_check_naming_an_unknown_series_is_refused():
    with pytest.raises(CleanError, match="z"):
        run(completeness={"columns": ["z"]})


def test_a_panel_is_refused():
    index = pd.MultiIndex.from_product([["a"], pd.period_range("2000Q1", periods=4, freq="Q")])
    ds = dz.Dataset(pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}, index=index))
    with pytest.raises(CleanError, match="panel"):
        clean.run(ds, {})


# --- checks and the gate ---------------------------------------------------------


def test_checks_run_before_and_after_treatment_and_the_gate_uses_after():
    r = run(TREATED[:3], outliers={"enabled": False})
    before = {(o.key, o.column): o for o in r.before}
    after = {(o.key, o.column): o for o in r.after}
    assert before[("gaps", "y")].flagged == ("2003Q2",)
    assert before[("gaps", "x")].flagged == ("2006Q1", "2006Q2")
    assert not before[("gaps", "y")].check.passed
    assert after[("gaps", "y")].check.passed
    assert after[("gaps", "x")].check.passed
    assert r.gate["gaps:y"].passed
    assert r.gate.passed
    assert r.flagged["y"] == ("2003Q2",)


def test_untreated_failures_fail_the_gate_and_overrides_accept_them():
    r = run(outliers={"enabled": False})
    assert not r.gate.passed
    assert {c.name for c in r.gate.failed} == {"gaps:y", "gaps:x"}
    r = run(
        outliers={"enabled": False},
        overrides=[
            {"check": "gaps:y", "reason": "Accepted."},
            {"check": "gaps:x", "reason": "Accepted."},
        ],
    )
    assert r.gate.passed
    assert r.gate["gaps:y"].status == "overridden"


def test_excluded_periods_are_not_counted_as_missing():
    r = run(
        [{"action": "exclude", "start": "2007Q1", "end": "2007Q4", "reason": "r"}],
        outliers={"enabled": False},
        gaps={"max_length": 2},
    )
    after = {(o.key, o.column): o for o in r.after}
    before = {(o.key, o.column): o for o in r.before}
    assert after[("completeness", "y")].failed == before[("completeness", "y")].failed
    assert after[("gaps", "y")].flagged == ()
    assert after[("completeness", "y")].units == 36


def test_the_spike_is_flagged_by_the_outlier_check():
    r = run()
    before = {(o.key, o.column): o for o in r.before}
    assert "2005Q1" in before[("outliers", "y")].flagged
    assert "2005Q1" in r.flagged["y"]


def test_range_and_revisions_and_representativeness_from_configuration():
    previous = raw()
    frame = previous.frame
    frame.loc["2001Q1", "x"] *= 1.2
    r = clean.run(
        raw(),
        {
            "range": {"bounds": {"y": [0.0, 300.0]}},
            "revisions": {"tolerance": 0.05},
            "representativeness": {
                "downturns": [{"label": "GFC", "start": "1999Q1", "end": "1999Q4"}]
            },
            "outliers": {"enabled": False},
        },
        previous=previous.with_frame(frame),
    )
    before = {(o.key, o.column): o for o in r.before}
    assert before[("range", "y")].flagged == ("2005Q1",)
    assert before[("range", "x")].check is None
    assert before[("revisions", "x")].flagged == ("2001Q1",)
    assert before[("revisions", "y")].flagged == ()
    assert before[("representativeness", "y")].failed == 4


def test_a_disabled_check_does_not_run():
    r = run(gaps={"enabled": False})
    assert "gaps" not in {o.key for o in r.before}


def test_a_registered_plugin_check_runs_from_configuration():
    @dz.checks.register("positive_test_only")
    class Positive:
        class Config(dz.checks.CheckConfig):
            pass

        def units(self, x, config, context):
            return dz.checks.Units(values=x.dropna(), test="above", bound=0.0)

    try:
        r = run(custom={"positive_test_only": {"error": 1}}, outliers={"enabled": False})
        keys = {o.key for o in r.after}
        assert "positive_test_only" in keys
        assert r.gate["positive_test_only:y"].passed
    finally:
        dz.checks.registry._items.pop("positive_test_only")


def test_an_unknown_custom_check_is_refused():
    with pytest.raises(dz.UnknownKeyError):
        run(custom={"nope": {}})


# --- result and artefacts --------------------------------------------------------


def test_artefacts_have_their_declared_identifiers():
    r = run(TREATED)
    assert set(r.artefacts) == {"T2.1", "T2.2", "T2.3", "F2.1", "S2.1"}
    assert clean.IDS.table("quality") == "T2.1"
    assert clean.IDS.table("treatments") == "T2.2"
    assert clean.IDS.table("gate") == "T2.3"
    assert clean.IDS.figure("history") == "F2.1"
    assert clean.IDS.finding("summary") == "S2.1"
    report = r.artefacts["T2.1"]
    applicable = [o for o in r.after if o.check is not None]
    assert len(report.rows) == len(applicable)
    assert any("range is not checked on y, x" in note for note in report.notes)
    table = r.artefacts["T2.2"]
    assert len(table.rows) == len(r.treatments)


def test_the_history_chart_marks_the_flagged_periods():
    r = run()
    fig = r.artefacts["F2.1"]
    assert fig.x[0] == "2000Q1"
    line, flags = fig.layers
    assert line.kind == "line"
    assert flags.kind == "scatter"
    marked = [fig.x[i] for i, v in enumerate(flags.y) if v is not None]
    assert "2005Q1" in marked
    assert flags.y[fig.x.index("2005Q1")] == 400.0
    assert set(marked) == set(r.flagged["y"]) - {"2003Q2"}  # a missing value has no point


def test_no_target_no_figure():
    ds = raw().with_series("y", role="driver")
    assert "F2.1" not in clean.run(ds, {}).artefacts


def test_the_finding_counts_what_was_found_and_changed():
    r = run(TREATED)
    assert r.finding.key == "clean.summary"
    assert r.finding.values["treated"] == len([t for t in r.treatments if t.period is not None])
    assert r.finding.values["failing"] == len(r.gate.failed)


def test_the_result_round_trips_without_the_dataset():
    r = run(TREATED)
    d = r.to_dict()
    assert "dataset" not in d
    back = clean.Result.from_dict(d)
    assert back.dataset is None
    assert back.fingerprint == r.dataset.fingerprint()
    assert back.before == r.before
    assert back.treatments == r.treatments


def test_clean_is_a_registered_step():
    from dazflow.steps import registry

    assert registry.get("clean") is clean
    assert dz.clean is clean


def test_the_artefacts_render():
    r = run(TREATED)
    for key in ("T2.1", "T2.2", "T2.3", "S2.1"):
        assert dz.render.render(r.artefacts[key], "markdown")
    assert dz.render.render(r.artefacts["F2.1"], "svg").startswith("<?xml")
