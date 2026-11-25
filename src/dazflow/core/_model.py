"""The pydantic base of the core objects: frozen, strict about extra fields,
and raising a `DazflowError` subclass instead of pydantic's `ValidationError`."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from dazflow.core.errors import DazflowError

# How many DazflowModel validations are running. Pydantic calls a nested model's
# __init__ while validating its parent; only the outermost converts the error,
# so the parent reports it with the full location.
_depth: ContextVar[int] = ContextVar("dazflow_validation_depth", default=0)


def _summary(err: ValidationError) -> str:
    parts = []
    for e in err.errors(include_url=False):
        loc = ".".join(str(p) for p in e["loc"]) or "(top level)"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)


class DazflowModel(BaseModel):
    """Frozen pydantic model whose validation errors are `_error` instances."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_inf_nan="strings",
    )

    _error: ClassVar[type[DazflowError]] = DazflowError
    _hint: ClassVar[str] = "Correct the fields listed in 'observed'."

    @classmethod
    def _raise(cls, err: ValidationError) -> None:
        raise cls._error(
            f"{cls.__name__} is not valid",
            observed=_summary(err),
            change=cls._hint,
        ) from err

    @classmethod
    def _guard(cls, call: Callable[[], Any]) -> Any:
        token = _depth.set(_depth.get() + 1)
        try:
            return call()
        except ValidationError as err:
            if _depth.get() > 1:
                raise
            cls._raise(err)
        finally:
            _depth.reset(token)

    def __init__(self, /, **data: Any) -> None:
        type(self)._guard(lambda: super(DazflowModel, self).__init__(**data))

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any):
        return cls._guard(lambda: super(DazflowModel, cls).model_validate(obj, **kwargs))

    def __setattr__(self, name: str, value: Any) -> None:
        type(self)._guard(lambda: super(DazflowModel, self).__setattr__(name, value))
