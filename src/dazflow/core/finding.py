"""Finding: a template sentence filled with numbers, never generated text (design §9)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import jinja2
from jinja2.sandbox import SandboxedEnvironment
from pydantic import Field, computed_field, field_validator, model_validator

from dazflow.core._model import DazflowModel
from dazflow.core.errors import FindingError

__all__ = ["Finding"]

Value = bool | int | float | str

# Sandboxed (no attribute access to Python internals), strict (a field without a
# value is an error, never an empty string), and no autoescaping: the text is
# plain, and each renderer escapes for its own format.
_ENV = SandboxedEnvironment(
    undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=True
)


class Finding(DazflowModel):
    """One or two sentences stating a step's result.

    `template` is jinja2 text; `values` fills its fields. The values are
    booleans, integers, finite floats or strings, so a finding is JSON-safe.
    The text is rendered, and checked, when the finding is made.

    Examples
    --------
    >>> Finding(template="{{ n }} of {{ total }} checks pass.", values={"n": 3, "total": 4}).text
    '3 of 4 checks pass.'
    """

    _error: ClassVar = FindingError
    _hint: ClassVar[str] = (
        "Give every template field a value that is a boolean, integer, finite float or string."
    )

    template: str
    values: dict[str, Value] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            data = {k: v for k, v in data.items() if k != "text"}
        return data

    @field_validator("values", mode="before")
    @classmethod
    def _values_are_scalars(cls, values: Any) -> Any:
        if isinstance(values, Mapping):
            for key, v in values.items():
                if not isinstance(v, bool | int | float | str):
                    raise ValueError(
                        f"value {key!r} is a {type(v).__name__}, not a number or string"
                    )
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError(f"value {key!r} is {v}, not a finite number")
        return values

    @model_validator(mode="after")
    def _template_renders(self) -> Finding:
        self._render()
        return self

    def _render(self) -> str:
        try:
            return _ENV.from_string(self.template).render(**self.values)
        except jinja2.TemplateSyntaxError as err:
            raise FindingError(
                "the finding template does not parse",
                observed=f"{err.message} at line {err.lineno} of {self.template!r}",
                change="Correct the jinja2 syntax of the template.",
            ) from err
        except jinja2.UndefinedError as err:
            raise FindingError(
                "a template field is missing its value",
                observed=f"{err.message}; values given: {', '.join(self.values) or 'none'}",
                change="Add the missing field to values, or remove it from the template.",
            ) from err
        except jinja2.exceptions.SecurityError as err:
            raise FindingError(
                "the finding template reaches outside its values",
                observed=str(err),
                change="Use only the values and jinja2 filters in a template.",
            ) from err

    @computed_field
    @property
    def text(self) -> str:
        """The template filled with the values."""
        return self._render()
