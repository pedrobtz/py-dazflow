"""report: the document, its tables and its gate, built from the run record alone (design §7.10).

Oracles: the run record itself (every documented step, artefact, input,
setting and package in it), the report result the pipeline recorded, and
the source of the report module (it imports no other step).
"""

import ast
import copy
import inspect
import json

import pytest

import dazflow as dz
from dazflow.core.artefact import STEP_NUMBERS
from dazflow.steps import report

DOCUMENTED = ["ingest", "clean", "explore", "build", "select", "forecast", "diagnose"]


@pytest.fixture(scope="module")
def recorded(ran_example):
    return json.loads((ran_example / "reports" / "record.json").read_text(encoding="utf-8"))


@pytest.fixture
def before(recorded):
    """The record as the report step saw it: every step before report."""
    data = copy.deepcopy(recorded)
    for part in ("results", "config"):
        data[part].pop("report")
    data["checks"] = [c for c in data["checks"] if c["step"] != "report"]
    return data


def test_report_is_a_registered_step_at_the_end_of_the_default_pipeline():
    from dazflow.pipeline import DEFAULT_STEPS, adapters
    from dazflow.steps import registry

    assert registry.get("report") is report is dz.report
    assert "report" in adapters
    assert (*DOCUMENTED, "report") == DEFAULT_STEPS


def test_the_report_imports_no_other_step():
    tree = ast.parse(inspect.getsource(report))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    assert not [m for m in imported if m.startswith("dazflow.steps") or m == "dazflow.pipeline"]


def test_the_result_from_the_record_alone_equals_the_recorded_one(recorded, before):
    result = report.run(before, recorded["config"]["report"])
    assert result.to_dict() == recorded["results"]["report"]


def test_the_identifiers_are_the_reports_section_number(recorded):
    assert STEP_NUMBERS["report"] == 10
    ids = list(recorded["results"]["report"]["artefacts"])
    assert ids == ["T10.1", "T10.2", "T10.3", "T10.4", "T10.5", "S10.1"]


def test_the_sections_are_contents_summary_each_step_in_order_and_the_appendices(before):
    result = report.run(before)
    titles = [s.title for s in result.document.sections]
    assert titles == list(result.sections)
    assert titles[:2] == ["Contents", "Summary"]
    assert titles[2:9] == [report.TITLES[s] for s in DOCUMENTED]
    assert titles[9:] == ["Appendix A. Configuration", "Appendix B. Environment and inputs"]
    assert result.steps == tuple(DOCUMENTED)
    contents = result.document.sections[0].items
    assert list(contents) == [f"{i}. {t}" for i, t in enumerate(titles[1:], start=1)]


def test_each_step_section_holds_that_steps_artefacts(before):
    result = report.run(before)
    for section, step in zip(result.document.sections[2:9], DOCUMENTED, strict=True):
        tables = [i.id for i in section.items if isinstance(i, dz.Table)]
        figures = [i.id for i in section.items if isinstance(i, dz.Figure)]
        labels = [i for i in section.items if isinstance(i, str)]
        findings = [i for i in section.items if isinstance(i, dz.Finding)]
        ids = list(before["results"][step]["artefacts"])
        assert tables == [i for i in ids if i.startswith("T")]
        assert figures == [i for i in ids if i.startswith("F")]
        assert labels == [f"Finding {i}." for i in ids if i.startswith("S")]
        assert [f.text for f in findings] == [
            before["results"][step]["artefacts"][i]["text"] for i in ids if i.startswith("S")
        ]


def test_the_summary_table_has_one_row_per_step_with_its_gate(before):
    table = report.run(before).artefacts["T10.1"]
    assert [r[0] for r in table.rows] == DOCUMENTED
    for row, step in zip(table.rows, DOCUMENTED, strict=True):
        checks = before["results"][step]["gate"]["checks"]
        named = dict(zip([c.name for c in table.columns], row, strict=True))
        assert named["checks"] == len(checks)
        assert named["failed"] == sum(c["status"] == "fail" for c in checks)
        assert named["overridden"] == sum(c["status"] == "overridden" for c in checks)
        assert named["artefacts"] == len(before["results"][step]["artefacts"])
        assert named["gate"] == ("overridden" if named["overridden"] else "pass")


def test_the_environment_table_is_the_records_environment(before):
    table = report.run(before).artefacts["T10.2"]
    rows = [list(r) for r in table.rows]
    env = before["environment"]
    assert rows[:4] == [
        ["Python", env["python"]],
        ["Implementation", env["implementation"]],
        ["Platform", env["platform"]],
        ["Machine", env["machine"]],
    ]
    assert rows[4:] == [[name, v] for name, v in env["packages"].items()]


def test_the_input_table_lists_every_recorded_input(before):
    table = report.run(before).artefacts["T10.3"]
    assert [list(r) for r in table.rows] == [
        [i["step"], i["name"], i["path"], i["kind"], i["fingerprint"]] for i in before["inputs"]
    ]


def test_the_configuration_appendix_holds_every_setting_of_every_step(before):
    table = report.run(before).artefacts["T10.4"]

    def leaves(node, prefix):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                yield from leaves(v, f"{prefix}.{k}" if prefix else k)
        else:
            yield prefix, node

    expected = [
        [step, key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)]
        for step in DOCUMENTED
        for key, value in leaves(before["config"][step], "")
    ]
    rows = [list(r) for r in table.rows]
    assert rows == expected
    assert ["diagnose", "anomalies.level", "0.99"] in rows


def test_the_gate_passes_on_a_complete_record_whose_gates_passed(before):
    result = report.run(before)
    assert result.gate.passed
    assert [c.name for c in result.gate.checks] == ["complete", "gates"]


def test_a_step_of_the_pipeline_missing_from_the_record_fails_complete(before):
    before["results"].pop("forecast")
    result = report.run(before)
    check = next(c for c in result.gate.checks if c.name == "complete")
    assert check.status == "fail"
    assert check.observed == 1
    assert result.missing == ("forecast",)
    assert "forecast" not in result.steps


def test_a_documented_step_whose_gate_failed_fails_gates(before):
    before["results"]["explore"]["gate"]["checks"] = [
        {"name": "screen", "observed": 1.0, "threshold": 0.0, "op": "<="}
    ]
    result = report.run(before)
    check = next(c for c in result.gate.checks if c.name == "gates")
    assert (check.status, check.observed) == ("fail", 1)
    assert result.failed == ("explore",)


def test_a_failed_check_can_be_overridden_with_a_reason(before):
    before["results"].pop("forecast")
    cfg = {"overrides": [{"check": "complete", "reason": "Forecast is reported separately."}]}
    assert report.run(before, cfg).gate.passed


def test_the_projects_description_opens_the_summary(before):
    before["project"]["description"] = "Loss rate of the example portfolio."
    summary = report.run(before).document.sections[1]
    assert summary.items[0] == "Loss rate of the example portfolio."
    assert summary.items[1] == "Finding S10.1."


def test_the_title_is_the_project_name_unless_configured(before):
    assert report.run(before).document.title == "loss-rate"
    assert report.run(before, {"title": "Loss model"}).document.title == "Loss model"


def test_the_replay_script_carries_the_configured_tolerance(before):
    script = report.run(before, {"replay": {"rtol": 1e-7, "atol": 1e-10}}).script
    compile(script, "replay.py", "exec")
    assert "RTOL = 1e-07" in script
    assert "ATOL = 1e-10" in script


def test_the_reports_tolerance_defaults_to_replays():
    from dazflow.pipeline import ATOL, RTOL

    cfg = report.Config()
    assert (cfg.replay.rtol, cfg.replay.atol) == (RTOL, ATOL)


def test_a_record_object_and_its_dictionary_give_the_same_report(before):
    from dazflow.pipeline import RunRecord

    rec = RunRecord.from_dict(before)
    assert report.run(rec).to_dict() == report.run(before).to_dict()


def test_something_that_is_not_a_run_record_is_refused():
    with pytest.raises(dz.ReportError, match="not a dazflow run record"):
        report.run({"kind": "something else"})


def test_the_document_and_script_are_not_in_the_results_dictionary(before):
    data = report.run(before).to_dict()
    assert "document" not in data
    assert "script" not in data


def test_the_finding_counts_the_steps_artefacts_and_overrides(before):
    finding = report.run(before).finding
    every = [a for s in DOCUMENTED for a in before["results"][s]["artefacts"]]
    assert finding.values["tables"] == sum(a.startswith("T") for a in every)
    assert finding.values["figures"] == sum(a.startswith("F") for a in every)
    assert finding.values["findings"] == sum(a.startswith("S") for a in every)
    assert finding.text.endswith("Every step's gate passed, 1 check by override with a reason.")
