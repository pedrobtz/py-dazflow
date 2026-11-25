"""The check protocol, the registry, and the pointblank validation that evaluates a check.

A check names, for one series, the periods it tests (its test units) and
what each must satisfy. pointblank interrogates those units with the
configured warning, error and critical thresholds; dazflow turns the
counts into a gate `Check` and an `Outcome` (design §7.2, D12, D29).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

import pandas as pd
from pydantic import Field, field_validator

from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.errors import CheckError
from dazflow.core.gate import Check
from dazflow.core.registry import Registry

__all__ = [
    "CheckConfig",
    "Context",
    "Level",
    "Outcome",
    "QualityCheck",
    "Test",
    "Units",
    "evaluate",
    "get",
    "register",
    "registry",
]

Level = Literal["none", "warning", "error", "critical"]
Test = Literal["not_null", "between", "at_most", "above"]


class CheckConfig(Config):
    """What every check's configuration holds: whether it runs, where, and its thresholds.

    A threshold below 1 is a share of the check's test units, 1 or more a
    count of them (pointblank's rule); a level is reached when the failing
    units reach it. The gate check fails at `error`.
    """

    enabled: bool = True
    columns: list[str] | None = Field(
        None, description="The series checked; unset: every numeric series the check applies to."
    )
    warning: float | None = Field(None, description="Failing units that raise a warning.")
    error: float = Field(1, description="Failing units at which the gate check fails.")
    critical: float | None = Field(None, description="Failing units that are critical.")

    @field_validator("warning", "error", "critical")
    @classmethod
    def _share_or_count(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not value > 0:
            raise ValueError(f"a threshold must be above 0; got {value}")
        if value >= 1 and value != math.floor(value):
            raise ValueError(
                f"a threshold of 1 or more is a count of periods and must be whole; got {value}"
            )
        return value


@dataclass(frozen=True)
class Context:
    """What a check may need beyond the series itself.

    `frequency` is ``"M"``, ``"Q"`` or ``"Y"`` (read from the index when
    unset); `role` the series' role; `previous` the same series in the
    previous vintage; `excluded` the periods a treatment excluded, which no
    check counts against the series.
    """

    frequency: str | None = None
    role: str = "other"
    previous: pd.Series | None = None
    excluded: frozenset[pd.Period] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Units:
    """The periods a check tests (the index of `values`) and what each value must satisfy.

    `test` is ``not_null``; ``between`` (`bound` and `upper` inclusive; a
    ``None`` side is open); ``at_most`` (``value <= bound``); or ``above``
    (``value > bound``). `measure` says what `values` holds, for the report.
    When `reason` is set the check does not apply and nothing is tested.
    """

    values: pd.Series
    test: Test = "not_null"
    bound: float | None = None
    upper: float | None = None
    measure: str = "value"
    reason: str | None = None

    @classmethod
    def not_applicable(cls, reason: str) -> Units:
        return cls(values=pd.Series([], dtype="float64"), reason=reason)


class QualityCheck(Protocol):
    """A data-quality check: its `Config` and the units it tests on one series.

    `units` receives the series (a float `pd.Series` on a `PeriodIndex`,
    named after the column), its validated `Config` and a `Context`.
    """

    Config: type[CheckConfig]

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        """The periods tested and the condition each must meet; `Units.not_applicable` if none."""


registry = Registry("check", protocol=QualityCheck)
register = registry.register
get = registry.get


class Outcome(DazflowModel):
    """One check on one series: how many units failed, the level reached and the gate check.

    `check` is ``None`` when the check does not apply (`reason` says why).
    `flagged` lists the failing periods as text (``"2020Q1"``).
    """

    _error: ClassVar = CheckError
    _hint: ClassVar[str] = "Build outcomes with dazflow.checks.evaluate."

    key: str
    column: str
    measure: str = "value"
    units: int = 0
    failed: int = 0
    share: float = 0.0
    level: Level = "none"
    flagged: tuple[str, ...] = ()
    check: Check | None = None
    reason: str | None = None


def _validate(cls: type[CheckConfig], config: CheckConfig | Mapping[str, Any] | None) -> Any:
    if isinstance(config, cls):
        return config
    if isinstance(config, Config):
        config = config.model_dump()
    return cls.model_validate(dict(config or {}))


def _frequency(x: pd.Series) -> str | None:
    from dazflow.readers._base import frequency_of

    return frequency_of(x.index) if isinstance(x.index, pd.PeriodIndex) else None


def evaluate(
    check: str | QualityCheck,
    x: pd.Series,
    config: CheckConfig | Mapping[str, Any] | None = None,
    context: Context | None = None,
) -> Outcome:
    """Run one check on one series with pointblank; the series is not changed.

    `check` is a registry key or a check object. The gate check is named
    ``<key>:<series>`` and passes while the failing units stay below
    `error`, counted as periods or, for a threshold below 1, as a share.
    """
    key = check if isinstance(check, str) else getattr(check, "key", type(check).__name__)
    obj = registry.get(check)() if isinstance(check, str) else check
    cfg = _validate(obj.Config, config)
    ctx = context or Context()
    if ctx.frequency is None:
        ctx = Context(
            frequency=_frequency(x), role=ctx.role, previous=ctx.previous, excluded=ctx.excluded
        )
    column = str(x.name) if x.name is not None else "value"
    units = obj.units(x, cfg, ctx)
    if units.reason is not None:
        return Outcome(key=key, column=column, measure=units.measure, reason=units.reason)
    values = units.values
    if ctx.excluded:
        values = values[~values.index.isin(list(ctx.excluded))]
    n = len(values)
    if n == 0:
        # pointblank 0.13 cannot interrogate a table without rows; nothing failed.
        failed, share, level, flagged = 0, 0.0, "none", ()
    else:
        failed, share, level, flagged = _interrogate(values, units, cfg)
    by_share = cfg.error < 1
    gate = Check(
        name=f"{key}:{column}",
        observed=share if by_share else failed,
        threshold=cfg.error,
        op="<",
        unit="share of periods" if by_share else "periods",
    )
    return Outcome(
        key=key,
        column=column,
        measure=units.measure,
        units=n,
        failed=failed,
        share=share,
        level=level,
        flagged=flagged,
        check=gate,
    )


def _interrogate(
    values: pd.Series, units: Units, cfg: CheckConfig
) -> tuple[int, float, Level, tuple[str, ...]]:
    import pointblank as pb

    table = pd.DataFrame(
        {
            "period": [str(p) for p in values.index],
            "value": values.to_numpy(dtype="float64"),
        }
    )
    v = pb.Validate(
        table,
        thresholds=pb.Thresholds(warning=cfg.warning, error=cfg.error, critical=cfg.critical),
    )
    if units.test == "not_null":
        v = v.col_vals_not_null(columns="value")
    elif units.test == "at_most":
        v = v.col_vals_le(columns="value", value=units.bound)
    elif units.test == "above":
        v = v.col_vals_gt(columns="value", value=units.bound)
    elif units.bound is not None and units.upper is not None:
        v = v.col_vals_between(columns="value", left=units.bound, right=units.upper)
    elif units.bound is not None:
        v = v.col_vals_ge(columns="value", value=units.bound)
    else:
        v = v.col_vals_le(columns="value", value=units.upper)
    v = v.interrogate(extract_limit=len(table))
    failed = int(v.n_failed(i=1, scalar=True))
    share = float(v.f_failed(i=1, scalar=True))
    level: Level = "none"
    for name in ("warning", "error", "critical"):
        if getattr(v, name)(i=1, scalar=True):
            level = name  # type: ignore[assignment]
    flagged: tuple[str, ...] = ()
    if failed:
        extract = v.get_data_extracts(i=1, frame=True)
        flagged = tuple(str(p) for p in extract["period"])
    return failed, share, level, flagged
