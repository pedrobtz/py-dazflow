"""Check, Gate and Override: the tests that end a step (design §5.2)."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field, computed_field, field_validator, model_validator

from dazflow.core._model import DazflowModel
from dazflow.core.errors import GateError

__all__ = ["Check", "Comparison", "Gate", "Override"]

Comparison = Literal["<", "<=", ">", ">="]

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


class Override(DazflowModel):
    """Accepts a failed check, with the reason written in configuration.

    A reason is required: an empty or blank one is refused with a `GateError`.
    """

    _error: ClassVar = GateError
    _hint: ClassVar[str] = (
        "Write why the failed check is accepted, e.g. "
        "overrides: [{check: adf, reason: 'Short sample; see section 4.'}]."
    )

    check: str = Field(min_length=1)
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an override needs a reason; it is empty or blank")
        return value


class Check(DazflowModel):
    """One test: `observed <op> threshold` passes.

    A missing (NaN) observed value fails. The threshold must be finite.
    """

    _error: ClassVar = GateError
    _hint: ClassVar[str] = (
        "Give a name, a numeric observed value, a finite threshold and an op of <, <=, > or >=."
    )

    name: str = Field(min_length=1)
    observed: float
    threshold: float = Field(allow_inf_nan=False)
    op: Comparison
    unit: str | None = None
    override: Override | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            data = {k: v for k, v in data.items() if k not in ("passed", "status")}
        return data

    @model_validator(mode="after")
    def _override_names_this_check(self) -> Check:
        if self.override is not None and self.override.check != self.name:
            raise ValueError(
                f"the override is for check {self.override.check!r}, not {self.name!r}"
            )
        return self

    @computed_field
    @property
    def passed(self) -> bool:
        """Whether the observed value is within the threshold, ignoring any override."""
        return not math.isnan(self.observed) and _OPS[self.op](self.observed, self.threshold)

    @computed_field
    @property
    def status(self) -> Literal["pass", "fail", "overridden"]:
        """``pass``, ``fail``, or ``overridden`` (failed, and accepted with a reason)."""
        if self.passed:
            return "pass"
        return "overridden" if self.override is not None else "fail"


class Gate(DazflowModel):
    """The checks that end a step. It passes when every check passes or is overridden."""

    _error: ClassVar = GateError
    _hint: ClassVar[str] = "Give each check in a gate its own name."

    checks: tuple[Check, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            data = {k: v for k, v in data.items() if k != "passed"}
        return data

    @field_validator("checks")
    @classmethod
    def _names_are_unique(cls, checks: tuple[Check, ...]) -> tuple[Check, ...]:
        names = [c.name for c in checks]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"check names must be unique; repeated: {', '.join(dupes)}")
        return checks

    @classmethod
    def evaluate(
        cls,
        checks: Iterable[Check],
        overrides: Iterable[Override | Mapping[str, Any]] = (),
    ) -> Gate:
        """Build a gate, attaching each override to the check it names.

        An override that names no check in the gate is refused.
        """
        checks = tuple(checks)
        by_name = {c.name: c for c in checks}
        attached: dict[str, Override] = {}
        for item in overrides:
            ov = item if isinstance(item, Override) else Override.model_validate(item)
            if ov.check not in by_name:
                raise GateError(
                    f"override for unknown check {ov.check!r}",
                    observed=f"checks in this gate: {', '.join(by_name) or 'none'}",
                    change="Name one of the gate's checks in the override, or remove it.",
                )
            attached[ov.check] = ov
        return cls(
            checks=tuple(
                c.model_copy(update={"override": attached[c.name]}) if c.name in attached else c
                for c in checks
            )
        )

    @computed_field
    @property
    def passed(self) -> bool:
        """Every check passed or was overridden with a reason."""
        return all(c.status != "fail" for c in self.checks)

    @property
    def failed(self) -> tuple[Check, ...]:
        """The checks that failed and were not overridden."""
        return tuple(c for c in self.checks if c.status == "fail")

    @property
    def overridden(self) -> tuple[Check, ...]:
        """The checks that failed and were accepted by an override."""
        return tuple(c for c in self.checks if c.status == "overridden")

    def __getitem__(self, name: str) -> Check:
        for c in self.checks:
            if c.name == name:
                return c
        raise GateError(
            f"no check named {name!r} in this gate",
            observed=f"checks: {', '.join(c.name for c in self.checks) or 'none'}",
            change="Use one of the check names listed.",
        )
