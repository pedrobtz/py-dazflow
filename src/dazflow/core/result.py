"""StepResult: what every step returns (design §5.2)."""

from __future__ import annotations

import json
from typing import Annotated, Any, ClassVar, Self

from pydantic import Discriminator, Field, Tag, field_validator

from dazflow.core._model import DazflowModel
from dazflow.core.artefact import Figure, Table, id_sort_key, parse_id
from dazflow.core.config import Config
from dazflow.core.errors import ArtefactError, ResultError
from dazflow.core.finding import Finding
from dazflow.core.gate import Gate

__all__ = ["Artefact", "StepResult"]


def _artefact_kind(value: Any) -> str:
    if isinstance(value, Table | Figure | Finding):
        return type(value).__name__.lower()
    if isinstance(value, dict):
        return value.get("kind", "finding")
    return "finding"


Artefact = Annotated[
    Annotated[Table, Tag("table")]
    | Annotated[Figure, Tag("figure")]
    | Annotated[Finding, Tag("finding")],
    Discriminator(_artefact_kind),
]


def _letter(obj: Any) -> str:
    return "T" if isinstance(obj, Table) else "F" if isinstance(obj, Figure) else "S"


class StepResult(DazflowModel):
    """The shared fields of a step's result; each step subclasses it with typed values.

    A subclass narrows `config` to its step's `Config` subclass and adds its
    own fields. `to_dict()` gives strict JSON (non-finite floats become the
    strings ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``), and `from_dict()`
    on the same subclass rebuilds an equal result.
    """

    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "Correct the result fields listed in 'observed'."

    gate: Gate = Field(default_factory=Gate)
    finding: Finding | None = None
    # Identifier to artefact, sorted by identifier (design §9, D21).
    artefacts: dict[str, Artefact] = Field(default_factory=dict)
    config: Config | None = None

    @field_validator("artefacts")
    @classmethod
    def _keys_are_the_identifiers(cls, artefacts: dict[str, Any]) -> dict[str, Any]:
        for key, obj in artefacts.items():
            try:
                letter = parse_id(key)[0]
            except ArtefactError as err:
                raise ValueError(err.problem) from None
            if letter != _letter(obj):
                raise ValueError(
                    f"{key!r} holds a {type(obj).__name__}, whose identifier starts with "
                    f"{_letter(obj)}"
                )
            own = getattr(obj, "id", key)
            if own != key:
                raise ValueError(f"key {key!r} holds the artefact {own!r}; key it by {own!r}")
        return dict(sorted(artefacts.items(), key=lambda item: id_sort_key(item[0])))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dictionary, for the run record and ``--json``."""
        return json.loads(self.model_dump_json(serialize_as_any=True))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a result from `to_dict()` output."""
        return cls.model_validate(data)
