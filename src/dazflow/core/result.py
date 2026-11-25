"""StepResult: what every step returns (design §5.2)."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Self

from pydantic import Field

from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.errors import ResultError
from dazflow.core.finding import Finding
from dazflow.core.gate import Gate

__all__ = ["StepResult"]


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
    # Stage 2 widens the values to Table | Figure | Finding.
    artefacts: dict[str, Finding] = Field(default_factory=dict)
    config: Config | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dictionary, for the run record and ``--json``."""
        return json.loads(self.model_dump_json(serialize_as_any=True))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a result from `to_dict()` output."""
        return cls.model_validate(data)
