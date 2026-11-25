"""Renderers are components: registered by format, passing the contract suite (design §6)."""

import pytest

import dazflow as dz
from dazflow import Figure, Finding, Layer, RenderError
from dazflow.render import Renderer, renderers
from dazflow.testing import ComponentContract


class TestDocx(ComponentContract):
    registry = renderers
    key = "docx"


class TestHtml(ComponentContract):
    registry = renderers
    key = "html"


class TestMarkdown(ComponentContract):
    registry = renderers
    key = "markdown"


class TestPng(ComponentContract):
    registry = renderers
    key = "png"


class TestSvg(ComponentContract):
    registry = renderers
    key = "svg"


def test_the_registry_holds_renderers():
    assert renderers.kind == "renderer"
    assert renderers.protocol is Renderer


@pytest.mark.parametrize(
    ("key", "suffix", "binary"),
    [
        ("docx", ".docx", True),
        ("html", ".html", False),
        ("markdown", ".md", False),
        ("png", ".png", True),
        ("svg", ".svg", False),
    ],
)
def test_each_renderer_says_its_suffix_and_whether_it_writes_bytes(key, suffix, binary):
    r = renderers[key]
    figure = Figure(id="F1.1", caption="c", x=[1, 2], layers=[Layer(kind="line", y=[1.0, 2.0])])

    out = r.render(figure, dz.render.Style())

    assert r.suffix == suffix
    assert r.binary is binary
    assert isinstance(out, bytes if binary else str)


def test_a_new_format_registers_like_any_component():
    from dazflow.render import render

    class Plain:
        suffix = ".txt"
        binary = False

        def render(self, obj, style):
            if not isinstance(obj, Finding):
                raise RenderError("plain renders findings only", observed="x", change="y")
            return obj.text

    renderers.add("plain-test", Plain())
    try:
        assert render(Finding(template="Hi."), "plain-test") == "Hi."
    finally:
        renderers._items.pop("plain-test")


def test_render_is_reachable_from_the_top_level_package():
    import dazflow.render

    assert dz.render is dazflow.render
    assert "render" in dir(dz)
