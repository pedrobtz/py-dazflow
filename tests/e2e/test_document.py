"""The Word document, read back with python-docx (design §7.10).

Oracles: the run record (its steps in pipeline order, every artefact
identifier each step produced, the run id) and python-docx's own reading
of the file the run wrote.
"""

import json

import docx
import pytest
from typer.testing import CliRunner

from dazflow.steps import report

DOCUMENTED = ["ingest", "clean", "explore", "build", "select", "forecast", "diagnose"]


@pytest.fixture(scope="module")
def record(ran_example):
    return json.loads((ran_example / "reports" / "record.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def paragraphs(ran_example):
    path = ran_example / "reports" / "document" / "document.docx"
    return [(p.style.name, p.text) for p in docx.Document(str(path)).paragraphs]


def sections(paragraphs):
    """Heading 1 title to the text of every paragraph under it."""
    out, current = {}, None
    for style, text in paragraphs:
        if style == "Heading 1":
            current = text
            out[current] = []
        elif current is not None:
            out[current].append(text)
    return out


def test_the_document_has_every_steps_section_in_pipeline_order(paragraphs, record):
    headings = [text for style, text in paragraphs if style == "Heading 1"]
    steps = [s for s in record["pipeline"] if s != "report"]
    assert steps == DOCUMENTED
    assert headings == [
        "Contents",
        "Summary",
        *[report.TITLES[s] for s in steps],
        "Appendix A. Configuration",
        "Appendix B. Environment and inputs",
    ]
    assert paragraphs[0] == ("Title", "loss-rate")


def test_the_contents_list_names_every_section_after_it(paragraphs):
    found = sections(paragraphs)
    titles = list(found)[1:]
    assert found["Contents"] == [f"{i}. {t}" for i, t in enumerate(titles, start=1)]


def test_every_artefact_identifier_the_run_produced_is_in_its_steps_section(paragraphs, record):
    found = sections(paragraphs)
    where = {s: report.TITLES[s] for s in DOCUMENTED}
    where["report"] = None  # the report's own artefacts: summary and appendices
    for step, result in record["results"].items():
        for aid in result["artefacts"]:
            label = {"T": "Table", "F": "Figure", "S": "Finding"}[aid[0]] + f" {aid}."
            titles = [where[step]] if where[step] else list(found)
            assert any(t.startswith(label) for title in titles for t in found[title]), aid


def test_tables_and_figures_carry_the_run_id_in_their_source_notes(paragraphs, record):
    notes = [t for _, t in paragraphs if t.startswith("Source:")]
    assert notes
    assert all(f"run {record['run']['id']}" in n for n in notes)
    # One note per table and figure of every step, each with its step's dataset.
    explore = sections(paragraphs)[report.TITLES["explore"]]
    fp = record["results"]["explore"]["fingerprint"][:12]
    assert all(fp in t for t in explore if t.startswith("Source:"))


def test_the_document_folder_holds_the_document_and_the_replay_script(ran_example):
    folder = ran_example / "reports" / "document"
    assert sorted(p.name for p in folder.iterdir()) == ["document.docx", "replay.py"]


def test_dazflow_report_alone_rebuilds_the_document_from_the_record(ran_copy):
    from dazflow.cli import app

    folder = ran_copy / "reports" / "document"
    (folder / "stale.txt").write_text("from an earlier run", encoding="utf-8")
    before = json.loads((ran_copy / "reports" / "record.json").read_text(encoding="utf-8"))
    result = CliRunner().invoke(app, ["report", "-p", str(ran_copy)])
    assert result.exit_code == 0, result.output
    after = json.loads((ran_copy / "reports" / "record.json").read_text(encoding="utf-8"))
    assert after["run"]["steps"] == ["report"]
    assert after["run"]["carried"] == DOCUMENTED
    assert after["results"]["report"] == before["results"]["report"]
    assert sorted(p.name for p in folder.iterdir()) == ["document.docx", "replay.py"]
    heads = [p.text for p in docx.Document(str(folder / "document.docx")).paragraphs]
    assert any(after["run"]["id"] in t for t in heads)
