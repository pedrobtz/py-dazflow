"""Exit codes: 0 on success, 1 on an error, 2 when a step completed with a failed gate (design §11).

Oracle: the design's three codes, observed through typer's CliRunner, and
the files the run leaves behind.
"""

import json
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from dazflow.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, app

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_the_codes_are_the_designs():
    assert (EXIT_OK, EXIT_ERROR, EXIT_GATE) == (0, 1, 2)


def test_run_on_the_example_exits_0_and_writes_the_record(example):
    result = invoke("run", "-p", example)
    assert result.exit_code == 0, result.output
    record = json.loads((example / "reports" / "record.json").read_text(encoding="utf-8"))
    assert record["run"]["status"] == "passed"
    assert record["run"]["steps"] == [
        "ingest",
        "clean",
        "explore",
        "build",
        "select",
        "forecast",
        "diagnose",
    ]


def test_a_configuration_error_exits_1(example):
    (example / "config" / "clean.yaml").write_text("no_such_setting: 1\n")
    result = invoke("run", "-p", example)
    assert result.exit_code == 1
    assert "no_such_setting" in result.stderr
    assert not (example / "reports" / "record.json").exists()


def test_a_folder_without_project_yaml_exits_1(tmp_path):
    result = invoke("validate", "-p", tmp_path)
    assert result.exit_code == 1
    assert "not a dazflow project" in result.stderr


@pytest.mark.parametrize("args", [["--from", "nowhere"], ["--from", "explore", "--to", "ingest"]])
def test_a_step_range_outside_the_pipeline_exits_1(example, args):
    assert invoke("run", "-p", example, *args).exit_code == 1


def test_a_failed_gate_exits_2_and_the_run_stops_there(example):
    path = example / "config" / "ingest.yaml"
    path.write_text(path.read_text().replace("min_history: 40", "min_history: 1000"))
    result = invoke("run", "-p", example)
    assert result.exit_code == 2, result.output
    record = json.loads((example / "reports" / "record.json").read_text(encoding="utf-8"))
    assert record["run"]["status"] == "gate_failed"
    assert record["run"]["stopped_at"] == "ingest"
    assert list(record["results"]) == ["ingest"]
    assert [c["name"] for c in record["checks"] if c["status"] == "fail"] == ["history"]
    assert not (example / "reports" / "clean").exists()


def test_a_failed_gate_in_a_step_command_exits_2(example):
    assert invoke("ingest", "-p", example).exit_code == 0
    path = example / "config" / "explore.yaml"
    path.write_text(path.read_text().replace("significance: 0.05", "significance: 1.0e-300"))
    assert invoke("clean", "-p", example).exit_code == 0
    assert invoke("explore", "-p", example).exit_code == 2


def test_an_overridden_gate_exits_0(example):
    path = example / "config" / "ingest.yaml"
    text = path.read_text().replace("min_history: 40", "min_history: 1000")
    text = text.replace(
        "overrides: []", "overrides:\n  - check: history\n    reason: a short history is accepted"
    )
    path.write_text(text)
    assert invoke("run", "-p", example, "--to", "ingest").exit_code == 0


def test_a_step_before_its_inputs_exist_exits_1(example):
    result = invoke("clean", "-p", example)
    assert result.exit_code == 1
    assert "ingest" in result.stderr


def test_version_exits_0_with_the_installed_version():
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"dazflow {version('dazflow')}"


def test_show_of_an_unknown_identifier_exits_1(ran_example):
    assert invoke("show", "T9.9", "-p", ran_example).exit_code == 1


def test_show_before_any_run_exits_1(example):
    assert invoke("show", "T1.1", "-p", example).exit_code == 1


def test_init_into_a_folder_that_has_files_exits_1(example):
    assert invoke("init", example).exit_code == 1
