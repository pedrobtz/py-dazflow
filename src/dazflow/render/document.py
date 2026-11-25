"""Section and Document: artefacts arranged for a model document (design §9)."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from pydantic import Discriminator, Field, Tag

from dazflow.core._model import DazflowModel
from dazflow.core.artefact import Figure, Source, Table
from dazflow.core.errors import ArtefactError
from dazflow.core.finding import Finding

__all__ = ["Document", "Item", "Section"]


def _item_kind(value: Any) -> str:
    if isinstance(value, str):
        return "text"
    if isinstance(value, Table | Figure | Finding | Section):
        return type(value).__name__.lower()
    if isinstance(value, dict):
        if "kind" in value:
            return value["kind"]
        return "section" if "items" in value else "finding"
    return "text"


Item = Annotated[
    Annotated[str, Tag("text")]
    | Annotated[Finding, Tag("finding")]
    | Annotated[Table, Tag("table")]
    | Annotated[Figure, Tag("figure")]
    | Annotated["Section", Tag("section")],
    Discriminator(_item_kind),
]


class Section(DazflowModel):
    """A headed part of a document: paragraphs of text, findings, tables, figures, subsections."""

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = (
        "Give a section a title and items that are text, findings, tables, figures or sections."
    )

    title: str = Field(min_length=1)
    items: tuple[Item, ...] = ()


class Document(DazflowModel):
    """A model document: a title and sections.

    `source` fills the source note of every table and figure that has none.
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = "Give a document a title and sections."

    title: str = Field(min_length=1)
    sections: tuple[Section, ...] = ()
    source: Source | None = None


Section.model_rebuild()
Document.model_rebuild()


def with_source(item: Any, source: Source | None) -> Any:
    """`item` with `source` given to every table and figure in it that has none."""
    if source is None:
        return item
    if isinstance(item, Table | Figure):
        return item if item.source is not None else item.model_copy(update={"source": source})
    if isinstance(item, Section):
        return item.model_copy(update={"items": tuple(with_source(i, source) for i in item.items)})
    if isinstance(item, Document):
        return item.model_copy(
            update={"sections": tuple(with_source(s, source) for s in item.sections)}
        )
    return item
