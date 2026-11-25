"""Numbers, dates, marks, captions, source notes and findings as text, in a style.

Every renderer writes text through these functions, so Word, HTML and
Markdown show the same characters for the same value.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from dazflow.core.artefact import Figure, Source, Table
from dazflow.core.finding import Finding, fill
from dazflow.render.style import Style

__all__ = [
    "caption_text",
    "finding_text",
    "format_cell",
    "format_date",
    "format_number",
    "source_text",
]


def _separators(text: str, style: Style) -> str:
    n = style.numbers
    return text.replace(",", "\0").replace(".", n.decimal_mark).replace("\0", n.thousands)


def format_number(value: float | int, style: Style, *, integer: bool = False) -> str:
    """`value` with the style's decimals and separators; a value that rounds to zero has no sign."""
    if integer:
        text = f"{abs(int(value)):,d}"
        negative = value < 0
    else:
        text = f"{abs(value):,.{style.numbers.decimals}f}"
        negative = value < 0 and any(ch not in "0,." for ch in text)
    return ("-" if negative else "") + _separators(text, style)


def format_date(value: str | dt.date, style: Style) -> str:
    """An ISO date (or a `date`) in the style's date format."""
    date = dt.date.fromisoformat(value) if isinstance(value, str) else value
    return date.strftime(style.dates.format)


def format_cell(value: Any, kind: str, style: Style) -> str:
    """One table cell as text: numbers, percentages, dates and statuses in the style."""
    if value is None:
        return style.numbers.missing
    if kind == "number":
        return format_number(value, style)
    if kind == "integer":
        return format_number(value, style, integer=True)
    if kind == "percent":
        return format_number(value * 100, style) + "%"
    if kind == "date":
        return format_date(value, style)
    if kind == "status":
        return style.marks[value].text
    return str(value)


class _Number(float):
    """A float that prints in the style; it still compares and computes as a float."""

    style: Style

    def __new__(cls, value: float, style: Style) -> _Number:
        obj = super().__new__(cls, value)
        obj.style = style
        return obj

    def __str__(self) -> str:
        return format_number(float(self), self.style)


def finding_text(finding: Finding, style: Style) -> str:
    """The finding's sentence, with the style's template for its key and numbers in the style.

    Floats are written with the style's decimals and separators wherever a
    template prints them as they are (``{{ x }}``); filters and comparisons
    see the plain number. Integers, booleans and text are printed as given.
    """
    template = (
        style.findings.get(finding.key, finding.template) if finding.key else finding.template
    )
    values = {
        k: _Number(v, style) if isinstance(v, float) else v for k, v in finding.values.items()
    }
    return fill(template, values)


def caption_text(artefact: Table | Figure, style: Style) -> str:
    """``Table T3.1. Caption``, from the style's caption template."""
    template = style.captions.table if isinstance(artefact, Table) else style.captions.figure
    return fill(template, {"id": artefact.id, "caption": artefact.caption})


def source_text(source: Source, style: Style) -> str:
    """The source note: the note, the run, the dataset fingerprint and the date."""
    return fill(
        style.captions.source,
        {
            "note": source.note or "",
            "run_id": source.run_id,
            "fingerprint": source.fingerprint[: style.captions.fingerprint_digits],
            "date": format_date(source.date, style),
        },
    )
