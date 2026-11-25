"""Snapshot extensions and comparisons for the rendering tests.

Every leg of the full profile compares against the same committed snapshot
files, from great-tables 0.17 and matplotlib 3.8.4 (the lowest-direct job) to
the latest releases. Neither library promises byte-identical output across
versions, so two snapshot kinds normalise what the versions are allowed to
differ in, and nothing else (design D25):

- HTML: great-tables' generated CSS, its accessibility attributes and its
  class names other than alignment are dropped, and an inline SVG figure is
  one line (the SVG snapshots cover it); the elements, the escaped text,
  inline styles, alignment and dazflow's own classes are kept.
- SVG: ids are renumbered in order of appearance, glyph offsets are written
  one way, and numbers are compared with a tolerance: one point for
  coordinates (numbers of 1 or more), 0.005 for scales and opacities.

Markdown is dazflow's own output and is compared exactly.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

# ---------------------------------------------------------------------------
# HTML


_VOID = {"br", "hr", "img", "meta", "link", "col", "input"}
_KEEP_ATTRS = {"style", "colspan", "rowspan", "href", "src", "alt", "lang", "charset"}
_KEEP_CLASS = re.compile(r"gt_(left|right|center)|dz-[\w-]+")
_SKIP = ("style", "script", "svg")


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.depth = 0
        self.skip = 0  # inside <style> or <script>
        self.unwrapped: list[bool] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            if tag == "svg" and not self.skip:
                self.lines.append("  " * self.depth + "<svg/>")
            self.skip += 1
            return
        if self.skip:
            return
        kept = [
            (k, v) for k, v in attrs if k in _KEEP_ATTRS or (k == "id" and tag in ("div", "figure"))
        ]
        classes = [c for c in dict(attrs).get("class", "").split() if _KEEP_CLASS.fullmatch(c)]
        if classes:
            kept.append(("class", " ".join(sorted(classes))))
        kept.sort()
        if tag == "span" and not kept:
            self.unwrapped.append(True)
            return
        if tag == "span":
            self.unwrapped.append(False)
        text = "".join(f' {k}="{v}"' for k, v in kept)
        self.lines.append("  " * self.depth + f"<{tag}{text}>")
        if tag not in _VOID:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in _SKIP:
            self.skip -= 1
        elif tag not in _VOID and not self.skip:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip -= 1
            return
        if self.skip or tag in _VOID:
            return
        if tag == "span" and self.unwrapped and self.unwrapped.pop():
            return
        self.depth -= 1
        self.lines.append("  " * self.depth + f"</{tag}>")

    def handle_data(self, data):
        if self.skip:
            return
        text = html.escape(" ".join(data.split()), quote=False)
        if text:
            self.lines.append("  " * self.depth + text)

    def handle_decl(self, decl):
        self.lines.append(f"<!{decl}>")


def html_tree(html: str) -> str:
    """The elements, text and inline styles of `html`, one per line."""
    parser = _Tree()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.lines) + "\n"


class HtmlSnapshot(SingleFileSnapshotExtension):
    _write_mode = WriteMode.TEXT
    file_extension = _file_extension = "html"

    def serialize(self, data, **kwargs):
        return html_tree(data)


# ---------------------------------------------------------------------------
# Markdown


class MarkdownSnapshot(SingleFileSnapshotExtension):
    _write_mode = WriteMode.TEXT
    file_extension = _file_extension = "markdown"


# ---------------------------------------------------------------------------
# SVG

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")
# matplotlib 3.8 and 3.11 measure text heights differently: axis labels move by
# up to 0.3 pt and a four-entry legend's last row by 0.5 pt. Glyph advances are
# rounded differently in the third decimal.
SVG_TOLERANCE = 1.0
SMALL_TOLERANCE = 0.005


def svg_normalise(svg: str) -> str:
    """`svg` with ids renumbered and glyph offsets written as transforms."""
    svg = re.sub(
        r'<use xlink:href="(#[^"]+)" x="(-?[\d.]+)"',
        r'<use xlink:href="\1" transform="translate(\2 0)"',
        svg,
    )
    ids: dict[str, str] = {}
    for m in re.finditer(r'\bid="([^"]+)"', svg):
        ids.setdefault(m.group(1), f"id{len(ids) + 1}")
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{ids[m.group(1)]}"', svg)
    svg = re.sub(r"#([A-Za-z_][\w.-]*)", lambda m: "#" + ids.get(m.group(1), m.group(1)), svg)
    return "".join(line.rstrip() + "\n" for line in svg.splitlines())


def _close(x: float, y: float) -> bool:
    tolerance = SVG_TOLERANCE if max(abs(x), abs(y)) >= 1 else SMALL_TOLERANCE
    return abs(x - y) <= tolerance


def svg_matches(a: str, b: str) -> bool:
    """Whether two normalised SVGs have the same text and numbers within tolerance."""
    if _NUMBER.split(a) != _NUMBER.split(b):
        return False
    xs, ys = _NUMBER.findall(a), _NUMBER.findall(b)
    return len(xs) == len(ys) and all(
        _close(float(x), float(y)) for x, y in zip(xs, ys, strict=True)
    )


class SvgSnapshot(SingleFileSnapshotExtension):
    _write_mode = WriteMode.TEXT
    file_extension = _file_extension = "svg"

    def serialize(self, data, **kwargs):
        return svg_normalise(data)

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        return snapshot_data is not None and svg_matches(serialized_data, snapshot_data)
