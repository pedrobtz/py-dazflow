"""Word output, read back with python-docx (design §9, D22)."""

import datetime as dt
import io

import docx
import pytest
from docx.shared import RGBColor

from dazflow import Check, Column, Figure, Finding, Gate, Layer, Override, Source, Table
from dazflow.render import Document, Section, Style, load_style, render

FP = "0123456789abcdef" * 4
SOURCE = Source(run_id="run-7", fingerprint=FP, date=dt.date(2026, 11, 25), note="FRED")

GATE = Gate.evaluate(
    [
        Check(name="adf", observed=0.2, threshold=0.05, op="<"),
        Check(name="r2", observed=0.91234, threshold=0.5, op=">=", unit="share"),
        Check(name="dw", observed=0.4, threshold=1.5, op=">"),
    ],
    overrides=[Override(check="adf", reason="Short sample; see section 4.")],
)
GATE_TABLE = Table.from_gate(GATE, id="T8.1", caption="Diagnostic checks", source=SOURCE)
SUMMARY = Table(
    id="T3.1",
    caption="Series summary",
    columns=[
        Column(name="series", label="Series", kind="text"),
        Column(name="mean", label="Mean", unit="pp"),
        Column(name="n", label="Observations", kind="integer"),
    ],
    rows=[["Unemployment", 1234.5678, 1500], ["GDP growth", -0.25, 96]],
)
FIGURE = Figure(
    id="F3.1",
    caption="Unemployment rate",
    x=["2024Q1", "2024Q2", "2024Q3"],
    x_label="Quarter",
    y_label="Rate",
    y_unit="%",
    layers=[Layer(kind="line", y=[4.0, 4.2, 4.1], label="Rate")],
    source=SOURCE,
)


def read(obj, style=None):
    return docx.Document(io.BytesIO(render(obj, "docx", style=style)))


def texts(table):
    return [[c.text for c in row.cells] for row in table.rows]


def test_a_table_has_its_caption_column_labels_and_formatted_cells():
    doc = read(SUMMARY)

    (table,) = doc.tables
    caption = doc.paragraphs[0]
    assert caption.text == "Table T3.1. Series summary"
    assert caption.style.name == "Caption"
    assert texts(table) == [
        ["Series", "Mean (pp)", "Observations"],
        ["Unemployment", "1,234.57", "1,500"],
        ["GDP growth", "-0.25", "96"],
    ]
    assert all(r.bold for r in table.rows[0].cells[1].paragraphs[0].runs)
    assert table.rows[1].cells[1].paragraphs[0].alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.RIGHT
    assert table.rows[1].cells[0].paragraphs[0].alignment in (
        None,
        docx.enum.text.WD_ALIGN_PARAGRAPH.LEFT,
    )
    assert table.style.name == "Table Grid"


def test_pass_and_fail_are_marked_in_place_in_words_and_colour():
    doc = read(GATE_TABLE)

    (table,) = doc.tables
    rows = texts(table)
    assert rows[0] == ["Check", "Observed", "Rule", "Threshold", "Unit", "Result", "Reason"]
    assert rows[1:] == [
        ["adf", "0.20", "<", "0.05", "", "Overridden", "Short sample; see section 4."],
        ["r2", "0.91", ">=", "0.50", "share", "Pass", ""],
        ["dw", "0.40", ">", "1.50", "", "Fail", ""],
    ]
    house = Style().marks
    for row, mark in zip(
        table.rows[1:], (house.overridden, house.passed, house.failed), strict=True
    ):
        (run,) = row.cells[5].paragraphs[0].runs
        assert run.font.color.rgb == RGBColor.from_string(mark.colour.lstrip("#"))
        assert bool(run.bold) is mark.bold
    assert table.rows[3].cells[5].paragraphs[0].runs[0].bold is True


def test_the_source_note_carries_the_run_the_fingerprint_and_the_date():
    doc = read(GATE_TABLE)

    note = doc.paragraphs[-1].text
    assert note == "Source: FRED; run run-7, data 0123456789ab, 2026-11-25."


def test_a_table_without_a_source_has_no_source_note():
    doc = read(SUMMARY)

    assert [p.text for p in doc.paragraphs] == ["Table T3.1. Series summary"]


def test_a_document_has_its_title_headings_findings_tables_and_figures():
    finding = Finding(template="{{ n }} of 3 checks pass.", values={"n": 1})
    document = Document(
        title="Unemployment model",
        sections=[
            Section(
                title="Data",
                items=["The series are quarterly.", SUMMARY, FIGURE],
            ),
            Section(
                title="Diagnostics",
                items=[finding, GATE_TABLE, Section(title="Detail", items=["Nothing else."])],
            ),
        ],
    )

    doc = read(document)

    paras = [(p.style.name, p.text) for p in doc.paragraphs if p.text]
    assert paras == [
        ("Title", "Unemployment model"),
        ("Heading 1", "Data"),
        ("Normal", "The series are quarterly."),
        ("Caption", "Table T3.1. Series summary"),
        ("Caption", "Figure F3.1. Unemployment rate"),
        ("Normal", "Source: FRED; run run-7, data 0123456789ab, 2026-11-25."),
        ("Heading 1", "Diagnostics"),
        ("Normal", "1 of 3 checks pass."),
        ("Caption", "Table T8.1. Diagnostic checks"),
        ("Normal", "Source: FRED; run run-7, data 0123456789ab, 2026-11-25."),
        ("Heading 2", "Detail"),
        ("Normal", "Nothing else."),
    ]
    assert len(doc.tables) == 2
    assert len(doc.inline_shapes) == 1
    width_cm = doc.inline_shapes[0].width.cm
    assert width_cm == pytest.approx(Style().page.width_cm, abs=0.01)


def test_a_document_fills_a_missing_source_from_the_documents_source():
    document = Document(title="t", sections=[Section(title="s", items=[SUMMARY])], source=SOURCE)

    doc = read(document)

    assert doc.paragraphs[-1].text == "Source: FRED; run run-7, data 0123456789ab, 2026-11-25."


def test_the_organisations_reference_document_supplies_the_styles(tmp_path):
    reference = docx.Document()
    reference.styles["Normal"].font.name = "Georgia"
    reference.styles.add_style("House Table", docx.enum.style.WD_STYLE_TYPE.TABLE)
    reference.sections[0].header.paragraphs[0].text = "Internal"
    path = tmp_path / "reference.docx"
    reference.save(path)
    style = load_style({"word": {"reference": str(path), "table_style": "House Table"}})

    doc = read(Document(title="t", sections=[Section(title="s", items=[SUMMARY])]), style)

    assert doc.sections[0].header.paragraphs[0].text == "Internal"
    assert doc.tables[0].style.name == "House Table"
    assert doc.paragraphs[0].text == "t"


def test_a_reference_document_is_not_mutated_by_the_body_font(tmp_path):
    reference = docx.Document()
    reference.add_paragraph("old body")
    path = tmp_path / "reference.docx"
    reference.save(path)
    style = load_style({"word": {"reference": str(path)}})

    doc = read(SUMMARY, style)

    assert [p.text for p in doc.paragraphs] == ["Table T3.1. Series summary"]
    assert docx.Document(path).paragraphs[-1].text == "old body"


def test_a_table_style_the_document_lacks_is_refused():
    from dazflow import RenderError

    style = load_style({"word": {"table_style": "No Such Style"}})

    with pytest.raises(RenderError, match="No Such Style"):
        render(SUMMARY, "docx", style=style)


def test_the_body_font_is_the_style_font():
    doc = read(SUMMARY, load_style({"fonts": {"body": "Georgia", "size": 11}}))

    assert doc.styles["Normal"].font.name == "Georgia"
    assert doc.styles["Normal"].font.size.pt == 11
