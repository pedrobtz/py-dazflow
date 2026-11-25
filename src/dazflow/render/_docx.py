"""Word documents with python-docx (design §17 question 2, settled by D22)."""

from __future__ import annotations

import io
from typing import Any

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from dazflow.core.artefact import Figure, Table
from dazflow.core.errors import RenderError
from dazflow.core.finding import Finding
from dazflow.render._figures import figure_png
from dazflow.render._tables import NUMERIC_KINDS
from dazflow.render.document import Document, Section
from dazflow.render.format import caption_text, finding_text, format_cell, source_text
from dazflow.render.style import Style

__all__ = ["docx_bytes"]

_SOURCE_SIZE_STEP = 1.0  # the source note is one point smaller than the body


def _new(style: Style) -> Any:
    if style.word.reference is not None:
        doc = docx.Document(str(style.word.reference))
        body = doc.element.body
        for child in list(body):  # keep the styles, headers and page setup; drop the text
            if child.tag != qn("w:sectPr"):
                body.remove(child)
    else:
        doc = docx.Document()
        normal = doc.styles["Normal"]
        normal.font.name = style.fonts.body
        normal.font.size = Pt(style.fonts.size)
    return doc


def _need_style(doc: Any, name: str, kind: WD_STYLE_TYPE) -> str:
    if not any(s.name == name and s.type == kind for s in doc.styles):
        known = sorted(s.name for s in doc.styles if s.type == kind)
        raise RenderError(
            f"the Word document has no {kind.name.lower()} style {name!r}",
            observed=f"{kind.name.lower()} styles: {', '.join(known)}",
            change="Add the style to the reference document, or name one it has in the style file.",
        )
    return name


def _paragraph(doc: Any, text: str, style: str = "Normal") -> Any:
    return doc.add_paragraph(text, style=_need_style(doc, style, WD_STYLE_TYPE.PARAGRAPH))


def _notes(doc: Any, artefact: Table | Figure, style: Style) -> None:
    for note in artefact.notes:
        _paragraph(doc, note)
    if artefact.source is not None:
        p = _paragraph(doc, "")
        run = p.add_run(source_text(artefact.source, style))
        run.font.size = Pt(max(style.fonts.size - _SOURCE_SIZE_STEP, 1.0))


def _table(doc: Any, table: Table, style: Style) -> None:
    _paragraph(doc, caption_text(table, style), "Caption")
    table_style = _need_style(doc, style.word.table_style, WD_STYLE_TYPE.TABLE)
    out = doc.add_table(rows=1 + len(table.rows), cols=len(table.columns))
    out.style = table_style
    for j, col in enumerate(table.columns):
        p = out.rows[0].cells[j].paragraphs[0]
        p.add_run(col.heading).bold = True
        if col.kind in NUMERIC_KINDS:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for i, row in enumerate(table.rows, start=1):
        for j, (value, col) in enumerate(zip(row, table.columns, strict=True)):
            p = out.rows[i].cells[j].paragraphs[0]
            run = p.add_run(format_cell(value, col.kind, style))
            if col.kind in NUMERIC_KINDS:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if col.kind == "status":
                mark = style.marks[value]
                run.font.color.rgb = RGBColor.from_string(mark.colour.lstrip("#").upper())
                run.bold = mark.bold
    _notes(doc, table, style)


def _figure(doc: Any, figure: Figure, style: Style) -> None:
    doc.add_picture(io.BytesIO(figure_png(figure, style)), width=Cm(style.page.width_cm))
    _paragraph(doc, caption_text(figure, style), "Caption")
    _notes(doc, figure, style)


def _item(doc: Any, item: Any, style: Style, level: int) -> None:
    if isinstance(item, str):
        _paragraph(doc, item)
    elif isinstance(item, Finding):
        _paragraph(doc, finding_text(item, style))
    elif isinstance(item, Table):
        _table(doc, item, style)
    elif isinstance(item, Figure):
        _figure(doc, item, style)
    else:
        _section(doc, item, style, level)


def _section(doc: Any, section: Section, style: Style, level: int) -> None:
    _paragraph(doc, section.title, f"Heading {level}")
    for item in section.items:
        _item(doc, item, style, level + 1)


def docx_bytes(obj: Any, style: Style) -> bytes:
    """A Word document holding `obj`: a whole document, a section or one artefact."""
    doc = _new(style)
    if isinstance(obj, Document):
        _paragraph(doc, obj.title, "Title")
        for section in obj.sections:
            _section(doc, section, style, 1)
    else:
        _item(doc, obj, style, 1)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
