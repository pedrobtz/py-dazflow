"""Rendering: tables, figures, findings, sections and documents in the house style (design §9).

>>> from dazflow.render import render, write, load_style
>>> render(table, "docx", style=load_style("org-style.yaml"))  # doctest: +SKIP

Formats are renderers in `renderers`, keyed ``docx``, ``html``,
``markdown``, ``png`` and ``svg``; a plugin adds one with
``@renderers.register(key)`` or `renderers.add`.
"""

from dazflow.render._tables import to_great_table
from dazflow.render.document import Document, Section
from dazflow.render.format import (
    caption_text,
    finding_text,
    format_cell,
    format_date,
    format_number,
    source_text,
)
from dazflow.render.renderers import SUFFIXES, Renderable, Renderer, render, renderers, write
from dazflow.render.style import (
    HOUSE_MPLSTYLE,
    Captions,
    Dates,
    Figures,
    Fonts,
    Mark,
    Marks,
    Numbers,
    Page,
    Style,
    Word,
    load_style,
)

__all__ = [
    "HOUSE_MPLSTYLE",
    "SUFFIXES",
    "Captions",
    "Dates",
    "Document",
    "Figures",
    "Fonts",
    "Mark",
    "Marks",
    "Numbers",
    "Page",
    "Renderable",
    "Renderer",
    "Section",
    "Style",
    "Word",
    "caption_text",
    "finding_text",
    "format_cell",
    "format_date",
    "format_number",
    "load_style",
    "render",
    "renderers",
    "source_text",
    "to_great_table",
    "write",
]
