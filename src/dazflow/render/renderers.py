"""Renderers by format, in a registry like every other component (design §6), and `render`."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dazflow.core.artefact import Figure, Source, Table
from dazflow.core.errors import RenderError
from dazflow.core.finding import Finding
from dazflow.core.registry import Registry
from dazflow.render._docx import docx_bytes
from dazflow.render._figures import figure_png, figure_svg
from dazflow.render._tables import html_id, markdown_text, table_html, table_markdown
from dazflow.render.document import Document, Section, with_source
from dazflow.render.format import caption_text, finding_text, source_text
from dazflow.render.style import Style, load_style

__all__ = ["SUFFIXES", "Renderable", "Renderer", "render", "renderers", "write"]

Renderable = Table | Figure | Finding | Section | Document


@runtime_checkable
class Renderer(Protocol):
    """Writes artefacts in one format.

    `render` returns text, or bytes when `binary` is true, and raises
    `RenderError` for a kind of artefact the format cannot hold.
    """

    suffix: str
    binary: bool

    def render(self, obj: Renderable, style: Style) -> str | bytes: ...


renderers = Registry("renderer", protocol=Renderer)


def _refuse(fmt: str, obj: Any, kinds: str) -> RenderError:
    return RenderError(
        f"{fmt} holds {kinds}, not a {type(obj).__name__}",
        observed=f"a {type(obj).__name__} asked for as {fmt}",
        change="Render it as docx, html or markdown, or pick an artefact the format holds.",
    )


# ---------------------------------------------------------------------------
# HTML

_SVG_PROLOGUE = re.compile(r"\A(?:<\?xml[^>]*\?>\s*)?(?:<!DOCTYPE[^>]*>\s*)?", re.S)


def _figure_html(figure: Figure, style: Style) -> str:
    svg = _SVG_PROLOGUE.sub("", figure_svg(figure, style))
    parts = [f'<figure id="{html_id(figure.id)}">', svg.rstrip()]
    parts.append(f"<figcaption>{html.escape(caption_text(figure, style))}</figcaption>")
    parts.append("</figure>")
    for note in figure.notes:
        parts.append(f"<p>{html.escape(note)}</p>")
    if figure.source is not None:
        parts.append(f'<p class="dz-source">{html.escape(source_text(figure.source, style))}</p>')
    return "\n".join(parts) + "\n"


def _html_item(item: Any, style: Style, level: int) -> str:
    if isinstance(item, str):
        return f"<p>{html.escape(item)}</p>\n"
    if isinstance(item, Finding):
        return f'<p class="dz-finding">{html.escape(finding_text(item, style))}</p>\n'
    if isinstance(item, Table):
        return table_html(item, style).strip() + "\n"
    if isinstance(item, Figure):
        return _figure_html(item, style)
    tag = f"h{min(level, 6)}"
    body = "".join(_html_item(i, style, level + 1) for i in item.items)
    return f"<section>\n<{tag}>{html.escape(item.title)}</{tag}>\n{body}</section>\n"


def _html_document(doc: Document, style: Style) -> str:
    title = html.escape(doc.title)
    body = "".join(_html_item(s, style, 2) for s in doc.sections)
    font = html.escape(style.fonts.body, quote=True)
    css = (
        f"body {{ font-family: '{font}', Arial, sans-serif; font-size: {style.fonts.size:g}pt; "
        f"max-width: {style.page.width_cm:g}cm; margin: 2em auto; }}\n"
        "figure { margin: 1em 0; } figure svg { width: 100%; height: auto; }\n"
        ".dz-source { font-size: smaller; }"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>\n{css}\n</style>\n</head>\n<body>\n"
        f"<h1>{title}</h1>\n{body}</body>\n</html>\n"
    )


class HtmlRenderer:
    """HTML: tables from great-tables, figures as inline SVG; a document is a whole page."""

    suffix = ".html"
    binary = False

    def render(self, obj: Renderable, style: Style) -> str:
        if isinstance(obj, Document):
            return _html_document(obj, style)
        return _html_item(obj, style, 2)


# ---------------------------------------------------------------------------
# Markdown


def _figure_markdown(figure: Figure, style: Style) -> str:
    lines = [f"![{markdown_text(caption_text(figure, style))}]({figure.id}.svg)", ""]
    for note in figure.notes:
        lines += [markdown_text(note), ""]
    if figure.source is not None:
        lines += [markdown_text(source_text(figure.source, style)), ""]
    return "\n".join(lines)


def _markdown_item(item: Any, style: Style, level: int) -> str:
    if isinstance(item, str):
        return markdown_text(item) + "\n"
    if isinstance(item, Finding):
        return markdown_text(finding_text(item, style)) + "\n"
    if isinstance(item, Table):
        return table_markdown(item, style)
    if isinstance(item, Figure):
        return _figure_markdown(item, style)
    parts = [f"{'#' * min(level, 6)} {markdown_text(item.title)}\n"]
    parts += [_markdown_item(i, style, level + 1) for i in item.items]
    return "\n".join(parts)


class MarkdownRenderer:
    """Markdown: pipe tables, findings as paragraphs, figures as links to ``<id>.svg``."""

    suffix = ".md"
    binary = False

    def render(self, obj: Renderable, style: Style) -> str:
        if isinstance(obj, Document):
            parts = [f"# {markdown_text(obj.title)}\n"] + [
                _markdown_item(s, style, 2) for s in obj.sections
            ]
            return "\n".join(parts)
        return _markdown_item(obj, style, 2)


# ---------------------------------------------------------------------------
# Word, PNG, SVG


class DocxRenderer:
    """Word, with python-docx: a document, a section or one artefact as a ``.docx``."""

    suffix = ".docx"
    binary = True

    def render(self, obj: Renderable, style: Style) -> bytes:
        return docx_bytes(obj, style)


class PngRenderer:
    """A figure as PNG at the style's resolution."""

    suffix = ".png"
    binary = True

    def render(self, obj: Renderable, style: Style) -> bytes:
        if not isinstance(obj, Figure):
            raise _refuse("png", obj, "figures")
        return figure_png(obj, style)


class SvgRenderer:
    """A figure as SVG, its text drawn as paths."""

    suffix = ".svg"
    binary = False

    def render(self, obj: Renderable, style: Style) -> str:
        if not isinstance(obj, Figure):
            raise _refuse("svg", obj, "figures")
        return figure_svg(obj, style)


renderers.add("docx", DocxRenderer())
renderers.add("html", HtmlRenderer())
renderers.add("markdown", MarkdownRenderer())
renderers.add("png", PngRenderer())
renderers.add("svg", SvgRenderer())

SUFFIXES = {".docx": "docx", ".html": "html", ".md": "markdown", ".png": "png", ".svg": "svg"}


def render(
    obj: Renderable,
    format: str,
    *,
    style: Style | Any = None,
    source: Source | None = None,
) -> str | bytes:
    """`obj` in `format` (a renderer key: docx, html, markdown, png, svg), in the style.

    `style` is anything `load_style` takes. `source` fills the source note of
    every table and figure that has none; a `Document`'s own `source` does
    the same.
    """
    if not isinstance(obj, Table | Figure | Finding | Section | Document):
        raise RenderError(
            f"a {type(obj).__name__} is not an artefact",
            observed=f"{type(obj).__name__}: {obj!r:.80}",
            change="Render a Table, Figure, Finding, Section or Document.",
        )
    renderer = renderers.get(format)
    style = load_style(style)
    if isinstance(obj, Document):
        obj = with_source(obj, obj.source)
    obj = with_source(obj, source)
    return renderer.render(obj, style)


def _figures(obj: Any) -> list[Figure]:
    if isinstance(obj, Figure):
        return [obj]
    if isinstance(obj, Document):
        return [f for s in obj.sections for f in _figures(s)]
    if isinstance(obj, Section):
        return [f for i in obj.items for f in _figures(i)]
    return []


def write(
    obj: Renderable,
    path: str | Path,
    *,
    format: str | None = None,
    style: Style | Any = None,
    source: Source | None = None,
) -> Path:
    """Render `obj` to the file `path`, the format from its suffix unless `format` is given.

    Markdown links its figures as ``<id>.svg``; those files are written next to it.
    """
    path = Path(path)
    if format is None:
        format = SUFFIXES.get(path.suffix.lower())
        if format is None:
            raise RenderError(
                f"no format writes {path.suffix or 'a file without a suffix'!s}",
                observed=f"path {path}",
                change=f"Use one of the suffixes {', '.join(SUFFIXES)}, or pass format=.",
            )
    style = load_style(style)
    out = render(obj, format, style=style, source=source)
    if isinstance(out, bytes):
        path.write_bytes(out)
    else:
        path.write_text(out, encoding="utf-8", newline="\n")
    if format == "markdown" and not isinstance(obj, Figure):
        for figure in _figures(obj):
            (path.parent / f"{figure.id}.svg").write_text(
                figure_svg(figure, style), encoding="utf-8", newline="\n"
            )
    return path
