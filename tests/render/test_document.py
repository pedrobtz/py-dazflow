"""Section and Document: arrangement, sources and the JSON round trip (design §9)."""

import datetime as dt

import pytest

from dazflow import ArtefactError, Column, Figure, Finding, Layer, Source, Table
from dazflow.render import Document, Section, render

OWN = Source(run_id="own", fingerprint="1" * 64, date=dt.date(2026, 1, 2))
RUN = Source(run_id="run-7", fingerprint="0" * 64, date=dt.date(2026, 11, 25))
TABLE = Table(id="T3.1", caption="t", columns=[Column(name="x", label="X")], rows=[[1.0]])
FIGURE = Figure(
    id="F3.1",
    caption="f",
    x=[1, 2],
    layers=[Layer(kind="line", y=[1.0, 2.0])],
    notes=["Shaded: 90% band."],
)
DOCUMENT = Document(
    title="Model",
    sections=[
        Section(
            title="One",
            items=[
                "Text.",
                Finding(template="{{ n }} checks.", values={"n": 2}, key="k"),
                TABLE,
                FIGURE,
                Section(title="Two", items=[TABLE.model_copy(update={"source": OWN})]),
            ],
        )
    ],
    source=RUN,
)


def test_a_document_round_trips_through_json():
    data = DOCUMENT.model_dump(mode="json")

    back = Document.model_validate(data)

    assert back == DOCUMENT
    kinds = [type(i).__name__ for i in back.sections[0].items]
    assert kinds == ["str", "Finding", "Table", "Figure", "Section"]


def test_the_documents_source_fills_only_the_artefacts_without_one():
    markdown = render(DOCUMENT, "markdown")

    assert markdown.count("run run-7") == 2
    assert markdown.count("run own") == 1


def test_a_render_source_fills_a_single_artefact_and_the_artefacts_own_wins():
    assert "run run-7" in render(TABLE, "markdown", source=RUN)
    assert "run own" in render(TABLE.model_copy(update={"source": OWN}), "markdown", source=RUN)


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_figure_notes_and_source_follow_the_caption(fmt):
    out = render(FIGURE, fmt, source=RUN)

    caption = out.index("Figure F3.1. f")
    assert caption < out.index("Shaded: 90% band.") < out.index("run run-7")


def test_a_section_item_that_is_not_text_or_an_artefact_is_refused():
    with pytest.raises(ArtefactError):
        Section(title="s", items=[3])
    with pytest.raises(ArtefactError):
        Section(title="", items=[])
