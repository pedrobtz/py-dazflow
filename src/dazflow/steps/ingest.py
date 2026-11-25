"""ingest: read the inputs and put every series on one calendar (design §7.1).

>>> import pandas as pd
>>> from dazflow.steps import ingest
>>> months = pd.DataFrame(
...     {"cpi": [1.0, 2.0, 3.0]}, index=pd.period_range("2024-01", periods=3, freq="M")
... )
>>> r = ingest.run(
...     {"frequency": "Q", "inputs": [{"name": "macro"}],
...      "series": [{"name": "cpi", "role": "target"}], "min_history": 1},
...     data={"macro": months},
... )
>>> r.dataset.frame["cpi"].tolist(), r.gate.passed
([2.0], True)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from dazflow import readers
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset, LogEntry, Role, SeriesMeta
from dazflow.core.errors import IngestError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import StepResult
from dazflow.readers._base import FREQUENCIES, Frequency, ReadSpec, frequency_of

__all__ = ["IDS", "Config", "IngestConfig", "IngestResult", "Input", "Result", "SeriesSpec", "run"]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "ingest",
    tables={"inventory": 1, "gate": 2},
    figures={"history": 1},
    findings={"summary": 1},
)

_PER_YEAR = {"M": 12, "Q": 4, "Y": 1}
_PERIOD = {"M": ("month", "months"), "Q": ("quarter", "quarters"), "Y": ("year", "years")}
_CONVERSION = {"average": "period-average", "end": "period-end"}

SUMMARY = (
    "{{ found }} of {{ configured }} configured series were found and put on a {{ frequency }} "
    "calendar from {{ first }} to {{ last }}. The targets and drivers overlap on {{ history }} "
    "{{ unit }}, against a minimum of {{ minimum }}."
)


class Input(ReadSpec):
    """One input: a file read by a reader, or a frame passed to `run` under the same name."""

    name: str = Field(min_length=1)


class SeriesSpec(Config):
    """One series to take from the inputs, with its metadata and how to align it."""

    name: str = Field(min_length=1, description="The series' name in the dataset.")
    input: str | None = Field(None, description="The input holding it; unset: the one that does.")
    column: str | None = Field(None, description="Its column in the input; unset: the name.")
    role: Role = "driver"
    unit: str | None = None
    source: str | None = Field(None, description="Where it comes from; unset: the input's path.")
    definition: str | None = None
    expected_sign: Literal[-1, 0, 1] | None = None
    aggregation: Literal["average", "end"] = Field(
        "average",
        description="To a lower frequency by the period-average or the period-end value.",
    )
    lag: int = Field(0, ge=0, description="Publication lag in periods of the target frequency.")


class IngestConfig(Config):
    """Configuration of the ingest step."""

    frequency: Frequency = Field(description="The calendar every series is put on: M, Q or Y.")
    inputs: list[Input] = Field(min_length=1)
    series: list[SeriesSpec] = Field(min_length=1)
    min_history: int = Field(
        40,
        ge=1,
        description="Fewest periods on which every target and driver has a value.",
    )
    overrides: list[Override] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_are_known_and_unique(self) -> IngestConfig:
        for what, names in (
            ("input", [i.name for i in self.inputs]),
            ("series", [s.name for s in self.series]),
        ):
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"{what} names must be unique; repeated: {', '.join(dupes)}")
        known = {i.name for i in self.inputs}
        for s in self.series:
            if s.input is not None and s.input not in known:
                raise ValueError(
                    f"series {s.name!r} names the input {s.input!r}; inputs: {', '.join(sorted(known))}"
                )
        return self


class IngestResult(StepResult):
    """What ingest returns. `dataset` is left out of `to_dict()`; `fingerprint` identifies it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: IngestConfig
    dataset: Dataset | None = Field(None, exclude=True)
    fingerprint: str | None = None
    frequency: Frequency
    missing: tuple[str, ...] = ()
    history: int = 0
    sample_start: str | None = None
    sample_end: str | None = None


Config = IngestConfig
Result = IngestResult


def _fail(problem: str, observed: str, change: str) -> IngestError:
    return IngestError(problem, observed=observed, change=change)


def _read(cfg: IngestConfig, data: Mapping[str, Any]) -> dict[str, Dataset]:
    names = {i.name for i in cfg.inputs}
    extra = sorted(set(data) - names)
    if extra:
        raise _fail(
            f"data was passed for inputs the configuration does not have: {', '.join(extra)}",
            f"inputs: {', '.join(sorted(names))}",
            "Pass data only under the names of configured inputs.",
        )
    out = {}
    for inp in cfg.inputs:
        if inp.name in data:
            if inp.path is not None:
                raise _fail(
                    f"the input {inp.name!r} has both a path and data",
                    f"path {inp.path!r}",
                    "Give the input a path or pass its data, not both.",
                )
            out[inp.name] = readers.from_frame(
                data[inp.name], date=inp.date, frequency=inp.frequency, columns=inp.columns
            )
        elif inp.path is None:
            raise _fail(
                f"the input {inp.name!r} has no path and no data",
                f"data passed for: {', '.join(sorted(data)) or 'none'}",
                f"Give the input a path, or pass run(config, data={{{inp.name!r}: frame}}).",
            )
        else:
            out[inp.name] = readers.read(inp)
    return out


def _locate(spec: SeriesSpec, inputs: Mapping[str, Dataset]) -> str | None:
    column = spec.column or spec.name
    if spec.input is not None:
        return spec.input if column in inputs[spec.input].columns else None
    found = [n for n, ds in inputs.items() if column in ds.columns]
    if len(found) > 1:
        raise _fail(
            f"the column {column!r} of series {spec.name!r} is in more than one input",
            f"inputs: {', '.join(found)}",
            "Name the series' input: input: <name>.",
        )
    return found[0] if found else None


def convert(s: pd.Series, target: str, how: str) -> pd.Series:
    """`s` at the lower frequency `target`, by period-average or period-end.

    A target period has a value only when the data cover it: every native
    period in it has a value (average), or its last one does (end).
    """
    native = frequency_of(s.index)
    if native == target:
        return s
    if _PER_YEAR[native] < _PER_YEAR[target]:
        raise _fail(
            f"the series {s.name!r} is {FREQUENCIES[native]}, so it cannot be put on a "
            f"{FREQUENCIES[target]} calendar",
            f"data {FREQUENCIES[native]}, calendar {FREQUENCIES[target]}",
            "Ingest converts to a lower frequency only: set frequency to the lowest one "
            "among the series.",
        )
    groups = s.index.asfreq(target, how="end")
    periods = pd.PeriodIndex(groups.unique(), freq=target)
    if how == "average":
        g = s.groupby(groups)
        complete = g.count() == _PER_YEAR[native] // _PER_YEAR[target]
        values = g.mean().where(complete).reindex(periods).to_numpy()
    else:
        values = s.reindex(periods.asfreq(native, how="end")).to_numpy()
    return pd.Series(values, index=periods, name=s.name, dtype="float64")


def _numeric(s: pd.Series, spec: SeriesSpec, input_name: str) -> pd.Series:
    try:
        if pd.api.types.is_bool_dtype(s.dtype):
            raise TypeError("booleans are not measurements")
        return pd.to_numeric(s, errors="raise").astype("float64")
    except (ValueError, TypeError) as err:
        raise _fail(
            f"the series {spec.name!r} is not numeric",
            f"column {s.name!r} of input {input_name!r}: dtype {s.dtype}, first values "
            f"{s.head(3).tolist()}",
            "Read a numeric column, or fix its values.",
        ) from err


def run(
    config: IngestConfig | Mapping[str, Any], data: Mapping[str, Any] | None = None
) -> IngestResult:
    """Read the inputs and put the configured series on one calendar.

    `data` maps input names to frames (pandas, polars, pyarrow) used
    instead of reading a path. Prints nothing and writes no files.
    """
    cfg = config if isinstance(config, IngestConfig) else IngestConfig.model_validate(config)
    target = cfg.frequency
    singular, plural = _PERIOD[target]
    inputs = _read(cfg, dict(data or {}))

    aligned: dict[str, pd.Series] = {}
    meta: dict[str, SeriesMeta] = {}
    log: list[LogEntry] = []
    native_of: dict[str, str] = {}
    missing: list[str] = []
    for spec in cfg.series:
        where = _locate(spec, inputs)
        if where is None:
            missing.append(spec.name)
            continue
        ds = inputs[where]
        column = spec.column or spec.name
        s = _numeric(ds.frame[column], spec, where).rename(spec.name)
        native = frequency_of(s.index)
        native_of[spec.name] = native
        s = convert(s, target, spec.aggregation)
        if native != target:
            log.append(
                LogEntry(
                    action="convert",
                    reason=f"{FREQUENCIES[native]} to {FREQUENCIES[target]} by "
                    f"{_CONVERSION[spec.aggregation]}",
                    columns=(spec.name,),
                )
            )
        if spec.lag:
            s.index = s.index + spec.lag
            log.append(
                LogEntry(
                    action="lag",
                    reason=f"published {spec.lag} {plural if spec.lag > 1 else singular} late: "
                    f"each value moved {spec.lag} later",
                    columns=(spec.name,),
                )
            )
        aligned[spec.name] = s
        meta[spec.name] = SeriesMeta(
            role=spec.role,
            unit=spec.unit,
            frequency=target,
            source=spec.source or ds.series[column].source,
            expected_sign=spec.expected_sign,
        )

    nonempty = [s.index for s in aligned.values() if len(s)]
    if nonempty:
        index = pd.period_range(
            min(i.min() for i in nonempty), max(i.max() for i in nonempty), freq=target
        )
    else:
        index = pd.PeriodIndex([], freq=target)
    frame = pd.DataFrame(
        {n: s.reindex(index).to_numpy(dtype="float64") for n, s in aligned.items()}, index=index
    )
    dataset = Dataset(frame, meta, log=log)

    modelled = [n for n in aligned if meta[n].role in ("target", "driver")] or list(aligned)
    complete = frame[modelled].notna().all(axis=1) if modelled else pd.Series([], dtype=bool)
    sample = frame.index[complete.to_numpy()] if modelled else frame.index[:0]
    history = len(sample)

    gate = Gate.evaluate(
        [
            Check(name="found", observed=len(missing), threshold=0, op="<=", unit="series"),
            Check(
                name="history", observed=history, threshold=cfg.min_history, op=">=", unit=plural
            ),
        ],
        cfg.overrides,
    )

    finding = Finding(
        template=SUMMARY,
        key="ingest.summary",
        values={
            "found": len(aligned),
            "configured": len(cfg.series),
            "frequency": FREQUENCIES[target],
            "first": str(index[0]) if len(index) else "no period",
            "last": str(index[-1]) if len(index) else "no period",
            "history": history,
            "unit": singular if history == 1 else plural,
            "minimum": cfg.min_history,
        },
    )
    artefacts: dict[str, Any] = {
        IDS.table("inventory"): _inventory(cfg, aligned, native_of, plural),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Ingest checks"),
        IDS.finding("summary"): finding,
    }
    targets = [n for n in aligned if meta[n].role == "target"]
    if targets:
        artefacts[IDS.figure("history")] = _history(frame, targets, meta, singular)

    return IngestResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        dataset=dataset,
        fingerprint=dataset.fingerprint(),
        frequency=target,
        missing=tuple(missing),
        history=history,
        sample_start=str(sample[0]) if history else None,
        sample_end=str(sample[-1]) if history else None,
    )


def _inventory(
    cfg: IngestConfig, aligned: Mapping[str, pd.Series], native_of: Mapping[str, str], plural: str
) -> Table:
    columns = [
        Column(name="series", label="Series", kind="text"),
        Column(name="role", label="Role", kind="text"),
        Column(name="unit", label="Unit", kind="text"),
        Column(name="source", label="Source", kind="text"),
        Column(name="definition", label="Definition", kind="text"),
        Column(name="native", label="Frequency read", kind="text"),
        Column(name="conversion", label="Conversion", kind="text"),
        Column(name="lag", label="Publication lag", unit=plural, kind="integer"),
        Column(name="first", label="First", kind="text"),
        Column(name="last", label="Last", kind="text"),
        Column(name="observations", label="Observations", kind="integer"),
        Column(name="found", label="Found", kind="status"),
    ]
    rows = []
    for spec in cfg.series:
        s = aligned.get(spec.name)
        if s is None:
            rows.append(
                [spec.name, spec.role, spec.unit, spec.source, spec.definition]
                + [None] * 6
                + ["fail"]
            )
            continue
        present = s.dropna()
        native = native_of[spec.name]
        rows.append(
            [
                spec.name,
                spec.role,
                spec.unit,
                spec.source,
                spec.definition,
                FREQUENCIES[native],
                "none" if native == cfg.frequency else _CONVERSION[spec.aggregation],
                spec.lag,
                str(present.index[0]) if len(present) else None,
                str(present.index[-1]) if len(present) else None,
                len(present),
                "pass",
            ]
        )
    return Table(
        id=IDS.table("inventory"),
        caption="Series inventory",
        columns=columns,
        rows=rows,
    )


def _history(
    frame: pd.DataFrame, targets: list[str], meta: Mapping[str, SeriesMeta], singular: str
) -> Figure:
    units = {meta[n].unit for n in targets}
    layers = [
        Layer(
            kind="line",
            y=[None if np.isnan(v) else float(v) for v in frame[n].to_numpy()],
            label=n,
        )
        for n in targets
    ]
    return Figure(
        id=IDS.figure("history"),
        caption="History of the target" if len(targets) == 1 else "History of the targets",
        x=[str(p) for p in frame.index],
        layers=layers,
        x_label=singular.capitalize(),
        y_label="Value",
        y_unit=units.pop() if len(units) == 1 else None,
    )
