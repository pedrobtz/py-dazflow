"""HTML and Markdown of tables, findings, sections and documents (design §9, D25)."""

import datetime as dt

import pytest
from snapshot_helpers import html_tree

from dazflow import Check, Column, Figure, Finding, Gate, Layer, Override, Source, Table
from dazflow.render import Document, Section, load_style, render, renderers

SOURCE = Source(
    run_id="run-7", fingerprint="0123456789abcdef" * 4, date=dt.date(2026, 11, 25), note="FRED"
)
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
    caption="Series summary <all>",
    columns=[
        Column(name="series", label="Series", kind="text"),
        Column(name="mean", label="Mean", unit="pp"),
        Column(name="share", label="Share", kind="percent"),
        Column(name="n", label="Observations", kind="integer"),
        Column(name="last", label="Last", kind="date"),
    ],
    rows=[
        ["Unemployment | total", 1234.5678, 0.125, 1500, "2024-03-31"],
        ["GDP growth & more", -0.25, None, 96, None],
    ],
    notes=["Means over the whole sample."],
)
FINDING = Finding(
    template="{{ n }} of {{ total }} checks pass; the p-value is {{ p }} <strong>.",
    values={"n": 1, "total": 3, "p": 0.04321},
)
FIGURE = Figure(
    id="F3.1",
    caption="Unemployment rate",
    x=["2024Q1", "2024Q2", "2024Q3"],
    x_label="Quarter",
    y_label="Rate",
    y_unit="%",
    layers=[Layer(kind="line", y=[4.0, 4.2, 4.1], label="Rate")],
)
DOCUMENT = Document(
    title="Unemployment model",
    sections=[
        Section(title="Data", items=["The series are quarterly.", SUMMARY, FIGURE]),
        Section(
            title="Diagnostics",
            items=[FINDING, GATE_TABLE, Section(title="Detail", items=["Nothing else."])],
        ),
    ],
)

CASES = {
    "summary": SUMMARY,
    "gate": GATE_TABLE,
    "finding": FINDING,
    "section": DOCUMENT.sections[1],
    "document": DOCUMENT,
}


@pytest.mark.parametrize("name", CASES)
def test_markdown_matches_its_snapshot(name, markdown_snapshot):
    assert render(CASES[name], "markdown") == markdown_snapshot


@pytest.mark.parametrize("name", CASES)
def test_html_matches_its_snapshot(name, html_snapshot):
    assert render(CASES[name], "html") == html_snapshot


def test_table_html_has_a_fixed_id_and_is_the_same_on_every_render():
    html = render(SUMMARY, "html")

    assert 'id="T3_1"' in html
    assert render(SUMMARY, "html") == html
    assert render(GATE_TABLE, "html") != html


def test_table_html_uses_the_style_font():
    html = render(SUMMARY, "html", style=load_style({"fonts": {"body": "Georgia"}}))

    assert "Georgia" in html


def test_markdown_escapes_pipes_and_html_escapes_markup():
    markdown = render(SUMMARY, "markdown")
    html = render(SUMMARY, "html")

    assert "Unemployment \\| total" in markdown
    assert "<all>" not in html
    assert "&lt;all&gt;" in html
    assert "<strong>" not in render(FINDING, "html")


def test_every_format_is_a_registered_renderer():
    assert renderers.keys() == ("docx", "html", "markdown", "png", "svg")


def test_an_unknown_format_is_refused_with_the_known_ones():
    from dazflow import UnknownKeyError

    with pytest.raises(UnknownKeyError, match="docx, html, markdown, png, svg"):
        render(SUMMARY, "pdf")


def test_something_that_is_not_an_artefact_is_refused():
    from dazflow import RenderError

    with pytest.raises(RenderError, match="dict"):
        render({"a": 1}, "markdown")


def test_write_picks_the_format_from_the_suffix_and_writes_figures_beside_markdown(tmp_path):
    from dazflow.render import write

    write(DOCUMENT, tmp_path / "model.md")
    write(DOCUMENT, tmp_path / "model.docx")
    write(FIGURE, tmp_path / "f.png")

    assert (tmp_path / "model.md").read_text(encoding="utf-8") == render(DOCUMENT, "markdown")
    assert "![Figure F3.1. Unemployment rate](F3.1.svg)" in (tmp_path / "model.md").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "F3.1.svg").read_text(encoding="utf-8") == render(FIGURE, "svg")
    assert (tmp_path / "model.docx").read_bytes()[:2] == b"PK"
    assert (tmp_path / "f.png").read_bytes() == render(FIGURE, "png")
    assert b"\r\n" not in (tmp_path / "model.md").read_bytes()


def test_write_refuses_a_suffix_with_no_format(tmp_path):
    from dazflow import RenderError
    from dazflow.render import write

    with pytest.raises(RenderError, match=r"\.pdf"):
        write(DOCUMENT, tmp_path / "model.pdf")


# The HTML comparison itself: a wrong normaliser would turn a failure into a pass.

GT_017 = """<div id="T3_1" style="padding:0"><style>#T3_1 table { font-size: 16px; }</style>
<table class="gt_table" data-quarto-disable-processing="false">
<tr><td class="gt_row gt_left">x</td>
<td style="color: #B00020;font-weight: bold;" class="gt_row gt_right">2.00</td></tr>
<tfoot class="gt_sourcenotes">
  <tr>
    <td class="gt_sourcenote" colspan="2">Source: x</td>
  </tr>
</tfoot></table></div>"""
GT_12 = """<div id="T3_1" style="padding:0"><style>#T3_1 table { font-size: 15px; }</style>
<table class="gt_table" aria-labelledby="T3_1-title" data-quarto-disable-processing="false">
<tr><td headers="T3_1-a" class="gt_row gt_left">x</td>
<td style="color: #B00020;font-weight: bold;" headers="T3_1-b" class="gt_row gt_right">2.00</td></tr>
<tfoot><tr class="gt_sourcenotes"><td class="gt_sourcenote" colspan="2"><span class="gt_from_md">Source: x</span></td></tr></tfoot></table></div>"""


def test_the_html_comparison_ignores_what_great_tables_releases_differ_in():
    assert html_tree(GT_017) == html_tree(GT_12)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (">2.00<", ">2.01<"),  # a cell's text
        ("color: #B00020;", "color: #000000;"),  # a pass/fail mark's style
        ('colspan="2"', 'colspan="3"'),  # the layout
        ('<td class="gt_row gt_left">x</td>', "<th>x</th>"),  # a cell became a header
        ("Source: x", "Source: y"),  # the source note
    ],
)
def test_the_html_comparison_still_sees_a_real_change(old, new):
    assert old in GT_017
    assert html_tree(GT_017) != html_tree(GT_017.replace(old, new, 1))
