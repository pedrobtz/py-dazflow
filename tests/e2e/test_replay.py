"""Replay: every step rerun from the run record and every recorded number compared (design §7.10, §10.1).

Oracles: the record the run wrote (`reports/record.json`); a change made on
purpose to one input value, one configuration setting or one recorded
number, which the replay must name with its path and both values; and the
tolerance's definition, one step inside and one step outside it.
"""

import copy
import json

import pytest
from typer.testing import CliRunner

from dazflow import project
from dazflow.cli import EXIT_GATE, EXIT_OK, app
from dazflow.pipeline import ATOL, RTOL, RunRecord, compare_records

STEPS = ("ingest", "clean", "explore", "build", "select", "forecast", "diagnose", "report")

runner = CliRunner()


def record_of(root):
    return json.loads((root / "reports" / "record.json").read_text(encoding="utf-8"))


def test_the_example_from_ingest_to_report_replayed_matches_its_record(ran_example, replayed):
    assert record_of(ran_example)["run"]["steps"] == list(STEPS)
    assert replayed.differences == ()
    assert replayed.passed
    assert replayed.steps == STEPS
    assert replayed.record_id == record_of(ran_example)["run"]["id"]
    assert (replayed.rtol, replayed.atol) == (RTOL, ATOL)
    # The replay compared what the record holds, not an empty run.
    assert set(replayed.record.results) == set(STEPS)


def test_dazflow_replay_exits_0_on_a_match(ran_example):
    result = runner.invoke(app, ["replay", "-p", str(ran_example), "--json"])
    assert result.exit_code == EXIT_OK, result.output
    got = json.loads(result.stdout)
    assert got["passed"] is True
    assert got["differences"] == []
    assert got["steps"] == list(STEPS)
    assert "record" not in got


def test_altering_one_input_value_fails_replay_and_names_the_difference(ran_copy, monkeypatch):
    series = ran_copy / "data" / "raw" / "series.csv"
    text = series.read_text(encoding="utf-8")
    assert "1995Q2,2.009," in text
    series.write_text(text.replace("1995Q2,2.009,", "1995Q2,2.019,"), encoding="utf-8")

    rep = project.load(ran_copy).replay()
    assert not rep.passed
    paths = [d.path for d in rep.differences]
    assert paths[0] == "inputs.ingest.macro.fingerprint"
    first = rep.differences[0]
    assert first.recorded == record_of(ran_copy)["inputs"][0]["fingerprint"]
    assert first.found != first.recorded
    # The changed value reaches the results: ingest's dataset is another one.
    assert "results.ingest.fingerprint" in paths
    assert any(p.startswith("results.diagnose.") for p in paths)

    monkeypatch.setattr("dazflow.cli._SHOWN", 3)
    result = runner.invoke(app, ["replay", "-p", str(ran_copy)])
    assert result.exit_code == EXIT_GATE
    lines = result.output.splitlines()
    assert lines[0].startswith("inputs.ingest.macro.fingerprint: recorded")
    assert lines[3] == f"... and {len(rep.differences) - 3} more"
    assert lines[-1].endswith(
        f"{len(rep.differences)} differences from its record (rtol 1e-09, atol 1e-12)"
    )


def test_altering_one_configuration_default_fails_replay_and_names_the_difference(ran_copy):
    path = ran_copy / "config" / "diagnose.yaml"
    text = path.read_text(encoding="utf-8")
    assert "  level: 0.99\n" in text
    path.write_text(text.replace("  level: 0.99\n", "  level: 0.95\n"), encoding="utf-8")

    rep = project.load(ran_copy).replay()
    assert not rep.passed
    assert [(d.path, d.recorded, d.found) for d in rep.differences] == [
        ("config.diagnose.anomalies.level", 0.99, 0.95)
    ]
    assert "config/diagnose.yaml" in rep.differences[0].note


def test_the_rerun_uses_the_recorded_configuration(ran_copy):
    # The record, not config/, says how each step ran: a setting changed in the record
    # changes what the rerun computes, and the replay names the results it moved.
    path = ran_copy / "reports" / "record.json"
    data = record_of(ran_copy)
    data["config"]["diagnose"]["anomalies"]["level"] = 0.95
    path.write_text(json.dumps(data), encoding="utf-8")

    rep = project.load(ran_copy).replay()
    paths = [d.path for d in rep.differences]
    assert "config.diagnose.anomalies.level" in paths  # config/ still says 0.99
    assert any(p.startswith("results.diagnose.") for p in paths)
    assert not any(p.startswith(("results.ingest.", "results.select.")) for p in paths)


@pytest.fixture
def number(replayed):
    """A recorded float of select's ranking, with its path, in a copy of the record."""
    data = copy.deepcopy(replayed.record.to_dict())
    row = data["results"]["select"]["ranking"][0]
    key = next(k for k, v in row.items() if isinstance(v, float) and v != 0)
    return data, row, key


def test_a_recorded_number_moved_just_inside_the_tolerance_still_matches(replayed, number):
    data, row, key = number
    row[key] *= 1 + 0.5 * RTOL
    assert compare_records(RunRecord.from_dict(data), replayed.record) == ()


def test_a_recorded_number_moved_just_outside_the_tolerance_is_named(replayed, number):
    data, row, key = number
    original = row[key]
    row[key] *= 1 + 2 * RTOL
    diffs = compare_records(RunRecord.from_dict(data), replayed.record)
    assert [(d.path, d.recorded, d.found) for d in diffs] == [
        (f"results.select.ranking[0].{key}", row[key], original)
    ]


def test_replay_without_a_record_exits_1(example):
    result = runner.invoke(app, ["replay", "-p", str(example)])
    assert result.exit_code == 1
    assert "has not run" in result.stderr
