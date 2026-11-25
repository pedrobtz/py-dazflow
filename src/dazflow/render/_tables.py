"""Tables as HTML (great-tables) and Markdown."""

from __future__ import annotations

import re

import pandas as pd
from great_tables import GT, loc
from great_tables import style as gt_style

from dazflow.core.artefact import Table
from dazflow.render.format import caption_text, format_cell, source_text
from dazflow.render.style import Style

__all__ = [
    "NUMERIC_KINDS",
    "html_id",
    "markdown_text",
    "table_html",
    "table_markdown",
    "to_great_table",
]

NUMERIC_KINDS = frozenset({"number", "integer", "percent"})


def html_id(identifier: str) -> str:
    """An identifier usable as an HTML id and CSS selector: ``T3.1`` becomes ``T3_1``."""
    return identifier.replace(".", "_")


def _cells(table: Table, style: Style) -> list[list[str]]:
    return [
        [format_cell(v, c.kind, style) for v, c in zip(row, table.columns, strict=True)]
        for row in table.rows
    ]


def to_great_table(table: Table, style: Style) -> GT:
    """The table as a great-tables `GT`, its cells already written in the style.

    The id is the artefact's, so the HTML is the same on every render.
    """
    keys = [f"c{j}" for j in range(len(table.columns))]
    frame = pd.DataFrame(_cells(table, style), columns=keys, dtype=object)
    gt = GT(frame, id=html_id(table.id), auto_align=False)
    gt = gt.tab_header(title=caption_text(table, style))
    if keys:
        gt = gt.cols_label(**{k: c.heading for k, c in zip(keys, table.columns, strict=True)})
        numeric = [k for k, c in zip(keys, table.columns, strict=True) if c.kind in NUMERIC_KINDS]
        text = [k for k in keys if k not in numeric]
        if numeric:
            gt = gt.cols_align(align="right", columns=numeric)
        if text:
            gt = gt.cols_align(align="left", columns=text)
    for j, col in enumerate(table.columns):
        if col.kind != "status":
            continue
        for i, row in enumerate(table.rows):
            mark = style.marks[row[j]]
            gt = gt.tab_style(
                style=gt_style.text(color=mark.colour, weight="bold" if mark.bold else "normal"),
                locations=loc.body(columns=keys[j], rows=[i]),
            )
    for note in table.notes:
        gt = gt.tab_source_note(note)
    if table.source is not None:
        gt = gt.tab_source_note(source_text(table.source, style))
    gt = gt.opt_table_font(font=[style.fonts.body, "Arial", "sans-serif"])
    return gt.tab_options(table_font_size=f"{style.fonts.size:g}pt")


def table_html(table: Table, style: Style) -> str:
    """The table as an HTML fragment, from great-tables."""
    return to_great_table(table, style).as_raw_html()


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<|])")


def markdown_text(text: str) -> str:
    """`text` with the characters Markdown would read as markup escaped, on one line."""
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", " ")


def table_markdown(table: Table, style: Style) -> str:
    """The table as a pipe table, with its caption above and its notes and source below."""
    lines = [markdown_text(caption_text(table, style)), ""]
    if table.columns:
        lines.append("| " + " | ".join(markdown_text(c.heading) for c in table.columns) + " |")
        lines.append(
            "| "
            + " | ".join("---:" if c.kind in NUMERIC_KINDS else ":---" for c in table.columns)
            + " |"
        )
        for row, cells in zip(table.rows, _cells(table, style), strict=True):
            out = []
            for value, text, col in zip(row, cells, table.columns, strict=True):
                text = markdown_text(text)
                if col.kind == "status" and style.marks[value].bold:
                    text = f"**{text}**"
                out.append(text)
            lines.append("| " + " | ".join(out) + " |")
        lines.append("")
    for note in table.notes:
        lines += [markdown_text(note), ""]
    if table.source is not None:
        lines += [markdown_text(source_text(table.source, style)), ""]
    return "\n".join(lines)
