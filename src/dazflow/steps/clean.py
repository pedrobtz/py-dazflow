"""clean: data-quality checks, configured treatments and their log (design §7.2).

>>> import pandas as pd
>>> import dazflow as dz
>>> frame = pd.DataFrame(
...     {"y": [1.0, None, 3.0, 4.0]}, index=pd.period_range("2024Q1", periods=4, freq="Q")
... )
>>> r = dz.clean.run(
...     dz.Dataset(frame, {"y": {"role": "target"}}),
...     {"treatments": [{"action": "interpolate", "reason": "One quarter lost."}],
...      "completeness": {"error": 2}},
... )
>>> r.dataset.frame["y"].tolist(), r.gate.passed
([1.0, 2.0, 3.0, 4.0], True)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from dazflow import checks
from dazflow.checks._builtin import (
    Completeness,
    Duplicates,
    Gaps,
    Outliers,
    Range,
    Representativeness,
    Revisions,
)
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset, LogEntry
from dazflow.core.errors import CleanError, ResultError
from dazflow.core.finding import Finding
from dazflow.core.gate import Gate, Override
from dazflow.core.result import StepResult
from dazflow.readers._base import frequency_of

__all__ = [
    "IDS",
    "CleanConfig",
    "CleanResult",
    "Config",
    "Exclude",
    "Fill",
    "Interpolate",
    "Replace",
    "Result",
    "Treatment",
    "TreatmentRecord",
    "Winsorise",
    "run",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "clean",
    tables={"quality": 1, "treatments": 2, "gate": 3},
    figures={"history": 1},
    findings={"summary": 1},
)

# The built-in checks, in the order they run and are reported.
BUILT_IN = (
    "completeness",
    "gaps",
    "duplicates",
    "range",
    "revisions",
    "outliers",
    "representativeness",
)

SUMMARY = (
    "{{ checks }} checks on {{ series }} series flagged {{ flagged }} periods before treatment. "
    "{{ treatments }} treatments changed {{ treated }} values. After treatment, "
    "{{ failing }} of {{ gated }} checks fail."
)


# --- treatments ----------------------------------------------------------------


class _Treatment(Config):
    """What every treatment holds: why, which series and which periods.

    The periods are `periods`, or the range `start` to `end` (either end
    open), or, when neither is given, every period.
    """

    reason: str = Field(description="Why the data are changed; required, never blank.")
    series: list[str] | None = Field(None, description="The series treated; unset: every one.")
    periods: list[str] | None = Field(None, description="The periods treated, as text.")
    start: str | None = None
    end: str | None = None

    _needs_periods: ClassVar[bool] = False

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a treatment needs a reason; it is empty or blank")
        return value

    @model_validator(mode="after")
    def _one_way_to_name_periods(self) -> _Treatment:
        ranged = self.start is not None or self.end is not None
        if self.periods is not None and ranged:
            raise ValueError("name the periods by periods or by start and end, not both")
        if self._needs_periods and self.periods is None and not ranged:
            raise ValueError(f"a {self.action} treatment needs periods, or a start or end")  # type: ignore[attr-defined]
        return self


class Fill(_Treatment):
    """Fill missing values from the previous or next value, or with a given value."""

    action: Literal["fill"]
    method: Literal["previous", "next", "value"] = "previous"
    value: float | None = None

    @model_validator(mode="after")
    def _value_with_method_value(self) -> Fill:
        if (self.method == "value") != (self.value is not None):
            raise ValueError("give a value exactly when method is value")
        return self


class Interpolate(_Treatment):
    """Fill missing values inside the history linearly between their neighbours."""

    action: Literal["interpolate"]


class Winsorise(_Treatment):
    """Clip values to the series' `lower` and `upper` quantiles."""

    action: Literal["winsorise"]
    lower: float = Field(0.01, ge=0, le=1, description="Quantile level of the lower bound.")
    upper: float = Field(0.99, ge=0, le=1, description="Quantile level of the upper bound.")

    @model_validator(mode="after")
    def _ordered(self) -> Winsorise:
        if not self.lower < self.upper:
            raise ValueError(f"lower ({self.lower}) must be below upper ({self.upper})")
        return self


class Replace(_Treatment):
    """Replace values at named periods by `value`, or linearly from their neighbours."""

    action: Literal["replace"]
    value: float | None = None

    _needs_periods: ClassVar[bool] = True


class Exclude(_Treatment):
    """Exclude periods: their values become missing and no check counts them as missing."""

    action: Literal["exclude"]

    _needs_periods: ClassVar[bool] = True


Treatment = Annotated[
    Fill | Interpolate | Winsorise | Replace | Exclude, Field(discriminator="action")
]


class TreatmentRecord(DazflowModel):
    """One value changed by a treatment, or a treatment of one series that changed none.

    `number` is the treatment's place in the configuration, from 1.
    `period`, `before` and `after` are ``None`` for a treatment that
    changed nothing in `series`; a missing value is ``None``.
    """

    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "Treatment records are made by the clean step."

    number: int
    action: str
    series: str
    period: str | None = None
    before: float | None = None
    after: float | None = None
    reason: str


# --- configuration and result ----------------------------------------------------


class CleanConfig(Config):
    """Configuration of the clean step: one section per check, the treatments, overrides."""

    completeness: Completeness.Config = Field(default_factory=Completeness.Config)
    gaps: Gaps.Config = Field(default_factory=Gaps.Config)
    duplicates: Duplicates.Config = Field(default_factory=Duplicates.Config)
    range: Range.Config = Field(default_factory=Range.Config)
    revisions: Revisions.Config = Field(default_factory=Revisions.Config)
    outliers: Outliers.Config = Field(default_factory=Outliers.Config)
    representativeness: Representativeness.Config = Field(default_factory=Representativeness.Config)
    custom: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Checks registered by plugins, by registry key, with their configuration.",
    )
    treatments: list[Treatment] = Field(default_factory=list)
    overrides: list[Override] = Field(default_factory=list)


class CleanResult(StepResult):
    """What clean returns. `dataset` is left out of `to_dict()`; `fingerprint` identifies it.

    `before` and `after` hold every check's outcome on every series, before
    and after treatment; the gate is built from `after`. `flagged` maps each
    series to the periods any check flagged before treatment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CleanConfig
    dataset: Dataset | None = Field(None, exclude=True)
    fingerprint: str | None = None
    input_fingerprint: str | None = None
    before: tuple[checks.Outcome, ...] = ()
    after: tuple[checks.Outcome, ...] = ()
    treatments: tuple[TreatmentRecord, ...] = ()
    flagged: dict[str, tuple[str, ...]] = Field(default_factory=dict)


Config = CleanConfig
Result = CleanResult


def _fail(problem: str, observed: str, change: str) -> CleanError:
    return CleanError(problem, observed=observed, change=change)


# --- running the checks ------------------------------------------------------------


def _numeric(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _plan(cfg: CleanConfig, ds: Dataset, numeric: list[str]) -> list[tuple[str, Any, Any, list]]:
    """(key, check object, its config, the series it runs on), for every enabled check."""
    plan = []
    sections = [(key, getattr(cfg, key)) for key in BUILT_IN]
    for key, raw in cfg.custom.items():
        obj = checks.get(key)  # UnknownKeyError lists the known keys
        sections.append((key, obj.Config.model_validate(raw)))
    for key, section in sections:
        if not section.enabled:
            continue
        impl = checks.get(key)
        if section.columns is not None:
            unknown = [c for c in section.columns if c not in numeric]
            if unknown:
                raise _fail(
                    f"the {key} check names series the dataset does not have as numbers: "
                    f"{', '.join(unknown)}",
                    f"numeric series: {', '.join(numeric) or 'none'}",
                    f"Name series of the dataset in {key}.columns.",
                )
            columns = list(section.columns)
        else:
            roles = getattr(impl, "roles", None)
            columns = [c for c in numeric if roles is None or ds.series[c].role in roles]
        obj = impl()
        obj.key = key
        plan.append((key, obj, section, columns))
    return plan


def _evaluate(
    plan: list[tuple[str, Any, Any, list]],
    frame: pd.DataFrame,
    ds: Dataset,
    freq: str,
    previous: Dataset | None,
    excluded: Mapping[str, frozenset[pd.Period]],
) -> tuple[checks.Outcome, ...]:
    out = []
    for _key, obj, section, columns in plan:
        for c in columns:
            prev = None
            if previous is not None and c in previous.columns:
                prev = previous.frame[c].astype("float64")
            ctx = checks.Context(
                frequency=freq,
                role=ds.series[c].role,
                previous=prev,
                excluded=excluded.get(c, frozenset()),
            )
            out.append(checks.evaluate(obj, frame[c].astype("float64"), section, ctx))
    return tuple(out)


# --- applying the treatments -------------------------------------------------------


def _period(text: str, index: pd.PeriodIndex, n: int) -> pd.Period:
    try:
        p = pd.Period(text, freq=index.freq)
    except (ValueError, TypeError) as err:
        raise _fail(
            f"treatment {n} names a period that cannot be read: {text!r}",
            f"the data are {index.freqstr}, from {index[0]} to {index[-1]}",
            "Write periods as the data's own: 2020Q1, 2020-03 or 2020.",
        ) from err
    if p not in index:
        raise _fail(
            f"treatment {n} names the period {text}, which the data do not have",
            f"the data run from {index[0]} to {index[-1]}",
            "Name periods inside the data.",
        )
    return p


def _selection(t: _Treatment, index: pd.PeriodIndex, n: int) -> np.ndarray:
    if t.periods is not None:
        wanted = {_period(p, index, n) for p in t.periods}
        return np.asarray([p in wanted for p in index])
    keep = np.ones(len(index), dtype=bool)
    if t.start is not None:
        keep &= index >= _period(t.start, index, n)
    if t.end is not None:
        keep &= index <= _period(t.end, index, n)
    return keep


def _treat(t: _Treatment, x: pd.Series, where: np.ndarray, n: int) -> pd.Series:
    """The values `t` gives the selected periods of `x` (the others are left alone)."""
    missing = x.isna().to_numpy()
    if isinstance(t, Fill):
        if t.method == "value":
            new = pd.Series(t.value, index=x.index)
        else:
            new = x.ffill() if t.method == "previous" else x.bfill()
        return x.where(~(where & missing), new)
    if isinstance(t, Interpolate):
        new = x.interpolate(method="linear", limit_area="inside")
        return x.where(~(where & missing), new)
    if isinstance(t, Winsorise):
        present = x.dropna().to_numpy()
        if not len(present):
            return x
        lo, hi = np.quantile(present, [t.lower, t.upper])
        return x.where(~where, x.clip(lo, hi))
    if isinstance(t, Replace):
        if t.value is not None:
            return x.where(~where, t.value)
        holed = x.where(~where)
        new = holed.interpolate(method="linear", limit_area="inside")
        stuck = where & new.isna().to_numpy()
        if stuck.any():
            raise _fail(
                f"treatment {n} replaces values of {x.name!r} by interpolation, "
                "but some have no value on both sides",
                f"periods without two neighbours: {', '.join(str(p) for p in x.index[stuck])}",
                "Give the replace treatment a value, or name periods inside the history.",
            )
        return x.where(~where, new)
    return x.where(~where)  # Exclude


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def _none(v: float) -> float | None:
    return None if np.isnan(v) else float(v)


def _apply(
    cfg: CleanConfig, ds: Dataset, numeric: list[str]
) -> tuple[pd.DataFrame, list[LogEntry], list[TreatmentRecord], dict[str, frozenset[pd.Period]]]:
    frame = ds.frame
    index = frame.index
    log: list[LogEntry] = []
    records: list[TreatmentRecord] = []
    excluded: dict[str, set[pd.Period]] = {}
    for n, t in enumerate(cfg.treatments, start=1):
        if t.series is not None:
            unknown = [s for s in t.series if s not in numeric]
            if unknown:
                raise _fail(
                    f"treatment {n} ({t.action}) names series the dataset does not have as "
                    f"numbers: {', '.join(unknown)}",
                    f"numeric series: {', '.join(numeric) or 'none'}",
                    "Name series of the dataset in the treatment's series.",
                )
        where = _selection(t, index, n)
        for name in t.series if t.series is not None else numeric:
            x = frame[name].astype("float64")
            keep = excluded.get(name, set())
            # An excluded period stays excluded: no later treatment fills it.
            selected = where & ~index.isin(list(keep))
            new = _treat(t, x, selected, n)
            changed = [
                i for i in range(len(x)) if not _same(x.iloc[i], new.iloc[i]) and selected[i]
            ]
            if isinstance(t, Exclude):
                excluded.setdefault(name, set()).update(index[selected])
            rows = tuple(str(index[i]) for i in changed)
            log.append(LogEntry(action=t.action, reason=t.reason, rows=rows, columns=(name,)))
            if changed:
                records.extend(
                    TreatmentRecord(
                        number=n,
                        action=t.action,
                        series=name,
                        period=str(index[i]),
                        before=_none(x.iloc[i]),
                        after=_none(new.iloc[i]),
                        reason=t.reason,
                    )
                    for i in changed
                )
            else:
                records.append(
                    TreatmentRecord(number=n, action=t.action, series=name, reason=t.reason)
                )
            frame[name] = new.to_numpy()
    return frame, log, records, {k: frozenset(v) for k, v in excluded.items()}


# --- the step ----------------------------------------------------------------------


def run(
    dataset: Dataset,
    config: CleanConfig | Mapping[str, Any] | None = None,
    *,
    previous: Dataset | None = None,
) -> CleanResult:
    """Check the data, apply the configured treatments, and check again.

    `previous` is the previous vintage of the data, for the revisions
    check. The caller's `dataset` is not changed. Prints nothing and
    writes no files.
    """
    cfg = config if isinstance(config, CleanConfig) else CleanConfig.model_validate(config or {})
    if dataset.is_panel:
        raise _fail(
            "clean does not treat panel datasets yet",
            f"a panel of {len(dataset.segments)} segments",
            "Clean each segment's dataset on its own.",
        )
    freq = frequency_of(dataset.frame.index)
    numeric = _numeric(dataset.frame)
    plan = _plan(cfg, dataset, numeric)
    raw = dataset.frame
    before = _evaluate(plan, raw, dataset, freq, previous, {})
    frame, log, records, excluded = _apply(cfg, dataset, numeric)
    cleaned = dataset.with_frame(frame)
    for entry in log:
        cleaned = cleaned.with_log(entry)
    after = _evaluate(plan, frame, dataset, freq, previous, excluded)

    gate = Gate.evaluate([o.check for o in after if o.check is not None], cfg.overrides)
    flagged: dict[str, tuple[str, ...]] = {}
    for c in numeric:
        found = {p for o in before if o.column == c for p in o.flagged}
        flagged[c] = tuple(str(p) for p in raw.index if str(p) in found)

    changed = [r for r in records if r.period is not None]
    gated = [o for o in after if o.check is not None]
    finding = Finding(
        template=SUMMARY,
        key="clean.summary",
        values={
            "checks": len({o.key for o in before}),
            "series": len(numeric),
            "flagged": len({(c, p) for c, ps in flagged.items() for p in ps}),
            "treatments": len(cfg.treatments),
            "treated": len(changed),
            "failing": len(gate.failed),
            "gated": len(gated),
        },
    )
    artefacts: dict[str, Any] = {
        IDS.table("quality"): _quality(before, after, gate),
        IDS.table("treatments"): _treatments(records),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Clean checks"),
        IDS.finding("summary"): finding,
    }
    targets = [c for c in numeric if dataset.series[c].role == "target"]
    if targets:
        artefacts[IDS.figure("history")] = _history(raw, frame, targets, flagged, dataset, freq)

    return CleanResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        dataset=cleaned,
        fingerprint=cleaned.fingerprint(),
        input_fingerprint=dataset.fingerprint(),
        before=before,
        after=after,
        treatments=tuple(records),
        flagged=flagged,
    )


def _quality(
    before: tuple[checks.Outcome, ...], after: tuple[checks.Outcome, ...], gate: Gate
) -> Table:
    columns = [
        Column(name="check", label="Check", kind="text"),
        Column(name="series", label="Series", kind="text"),
        Column(name="measure", label="Measure", kind="text"),
        Column(name="units", label="Periods tested", kind="integer"),
        Column(name="failed_before", label="Failing before", kind="integer"),
        Column(name="level_before", label="Level before", kind="text"),
        Column(name="flagged", label="Flagged before", kind="text"),
        Column(name="failed_after", label="Failing after", kind="integer"),
        Column(name="level_after", label="Level after", kind="text"),
        Column(name="threshold", label="Error threshold", kind="number"),
        Column(name="result", label="Result", kind="status"),
    ]
    rows = []
    skipped: dict[tuple[str, str], list[str]] = {}
    for b, a in zip(before, after, strict=True):
        if a.check is None or b.check is None:
            reason = a.reason or b.reason or "not applicable"
            skipped.setdefault((a.key, reason), []).append(a.column)
            continue
        rows.append(
            [
                b.key,
                b.column,
                b.measure,
                a.units,
                b.failed,
                b.level,
                ", ".join(b.flagged) or None,
                a.failed,
                a.level,
                a.check.threshold,
                gate[a.check.name].status,
            ]
        )
    notes = ["A threshold below 1 is a share of the periods tested, 1 or more a count of them."]
    notes += [
        f"{key} is not checked on {', '.join(cols)}: {reason}."
        for (key, reason), cols in skipped.items()
    ]
    return Table(
        id=IDS.table("quality"),
        caption="Data-quality report",
        columns=columns,
        rows=rows,
        notes=tuple(notes),
    )


def _treatments(records: list[TreatmentRecord]) -> Table:
    columns = [
        Column(name="number", label="Treatment", kind="integer"),
        Column(name="action", label="Action", kind="text"),
        Column(name="series", label="Series", kind="text"),
        Column(name="period", label="Period", kind="text"),
        Column(name="before", label="Before", kind="number"),
        Column(name="after", label="After", kind="number"),
        Column(name="reason", label="Reason", kind="text"),
    ]
    rows = [[r.number, r.action, r.series, r.period, r.before, r.after, r.reason] for r in records]
    return Table(
        id=IDS.table("treatments"),
        caption="Treatments and exclusions",
        columns=columns,
        rows=rows,
    )


_PERIOD_NAME = {"M": "Month", "Q": "Quarter", "Y": "Year"}


def _history(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    targets: list[str],
    flagged: Mapping[str, tuple[str, ...]],
    ds: Dataset,
    freq: str | None,
) -> Figure:
    layers = []
    labels = [str(p) for p in raw.index]
    for n in targets:
        layers.append(Layer(kind="line", y=[_none(v) for v in cleaned[n].to_numpy()], label=n))
        marked = set(flagged.get(n, ()))
        layers.append(
            Layer(
                kind="scatter",
                y=[
                    _none(v) if p in marked else None
                    for p, v in zip(labels, raw[n].to_numpy(), strict=True)
                ],
                label=f"{n}: flagged",
            )
        )
    units = {ds.series[n].unit for n in targets}
    return Figure(
        id=IDS.figure("history"),
        caption="History after cleaning, with the periods flagged before treatment",
        x=labels,
        layers=layers,
        x_label=_PERIOD_NAME.get(freq or "", "Period"),
        y_label="Value",
        y_unit=units.pop() if len(units) == 1 else None,
    )
