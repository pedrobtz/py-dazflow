"""Dataset: the one object that passes between steps (design §5.1, D13)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pydantic import Field, field_validator

from dazflow.core._model import DazflowModel
from dazflow.core.errors import DatasetError

__all__ = ["FINGERPRINT_FORMAT", "Dataset", "LogEntry", "SeriesMeta"]

FINGERPRINT_FORMAT = "dazflow.dataset/1"

Role = Literal["target", "driver", "other"]


class SeriesMeta(DazflowModel):
    """Metadata of one series (one column of `Dataset.frame`).

    `transforms` is the transformation chain applied so far, oldest first.
    `expected_sign` is the sign a driver's coefficient is expected to have.
    """

    _error: ClassVar = DatasetError
    _hint: ClassVar[str] = "Use role target, driver or other, and expected_sign -1, 0, 1 or null."

    role: Role = "other"
    unit: str | None = None
    frequency: str | None = None
    source: str | None = None
    transforms: tuple[str, ...] = ()
    expected_sign: Literal[-1, 0, 1] | None = None


class LogEntry(DazflowModel):
    """One change made to the data, with its reason and what it touched.

    `rows` holds the labels of the rows affected, as text (``"2020Q1"``,
    ``"north/2020Q1"``); `columns` the series affected.
    """

    _error: ClassVar = DatasetError
    _hint: ClassVar[str] = "Give the change an action and a non-blank reason."

    action: str = Field(min_length=1)
    reason: str
    rows: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a log entry needs a reason; it is empty or blank")
        return value


# --- validation --------------------------------------------------------------


def _bad(problem: str, observed: str, change: str) -> DatasetError:
    return DatasetError(problem, observed=observed, change=change)


def _column_kind(name: str, s: pd.Series, where: str) -> str:
    """The canonical kind of a column: float, int, bool or string."""
    dtype = s.dtype
    if ptypes.is_bool_dtype(dtype):
        return "bool"
    if ptypes.is_integer_dtype(dtype):
        return "int"
    if ptypes.is_float_dtype(dtype):
        return "float"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    if ptypes.is_object_dtype(dtype) and all(isinstance(v, str) or _missing(v) for v in s):
        return "string"
    raise _bad(
        f"{where} column {name!r} has an unsupported kind",
        f"dtype {dtype}",
        "Convert the column to float, integer, boolean or string values.",
    )


def _missing(v: Any) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _check_columns(frame: pd.DataFrame, where: str) -> None:
    if isinstance(frame.columns, pd.MultiIndex):
        raise _bad(
            f"the {where} columns are a MultiIndex",
            f"{frame.columns.nlevels} column levels",
            "Use one column per series, named by a string.",
        )
    bad = [c for c in frame.columns if not isinstance(c, str) or not c]
    if bad:
        raise _bad(
            f"the {where} column names must be non-empty strings",
            f"column names {bad!r}",
            "Rename the columns to strings, e.g. frame.columns = frame.columns.astype(str).",
        )
    if frame.columns.has_duplicates:
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise _bad(
            f"the {where} column names are not unique",
            f"repeated: {', '.join(dupes)}",
            "Give every series its own column name.",
        )
    for name in frame.columns:
        _column_kind(name, frame[name], where)


def _check_labels(labels: Iterable[Any], what: str) -> None:
    bad = sorted(
        {
            repr(v)
            for v in labels
            if isinstance(v, bool) or not isinstance(v, str | int | np.integer)
        }
    )
    if bad:
        raise _bad(
            f"{what} labels must be strings or integers",
            f"labels {', '.join(bad)}",
            f"Relabel the {what}s with strings or integers.",
        )


def _normalise_index(frame: pd.DataFrame, where: str, outer: str) -> pd.DataFrame:
    """A copy of `frame` with its index named canonically, after validating it."""
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        if (
            index.nlevels != 2
            or not isinstance(index.levels[1], pd.PeriodIndex)
            or isinstance(index.levels[0], pd.PeriodIndex)
        ):
            raise _bad(
                f"the {where} MultiIndex must be ({outer}, period)",
                f"levels of type {', '.join(type(lv).__name__ for lv in index.levels)}",
                f"Index the frame by ({outer}, period), with pandas periods in the second level.",
            )
        _check_labels(index.get_level_values(0), outer)
        names = [outer, "period"]
    elif isinstance(index, pd.PeriodIndex):
        names = ["period"]
    else:
        raise _bad(
            f"the {where} index is not a PeriodIndex",
            f"{type(index).__name__} of {index.dtype}",
            "Index the frame by pandas periods (pd.PeriodIndex), or by a "
            f"({outer}, period) MultiIndex for {outer}s.",
        )
    if index.has_duplicates:
        raise _bad(
            f"the {where} index is not unique",
            f"{int(index.duplicated().sum())} repeated rows, first {index[index.duplicated()][0]!r}",
            "Keep one row per period (and segment); aggregate or drop the repeats.",
        )
    out = frame.copy(deep=True)
    out.index = out.index.set_names(names)
    return out


def _freq(index: pd.Index) -> Any:
    level = index.levels[1] if isinstance(index, pd.MultiIndex) else index
    return level.freq


# --- canonical form (design §5.1) --------------------------------------------


def _stamp(dti: pd.DatetimeIndex) -> list[str]:
    nanos = dti.microsecond * 1000 + dti.nanosecond
    return [
        f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}.{ns:09d}"
        for y, mo, d, h, mi, s, ns in zip(
            dti.year, dti.month, dti.day, dti.hour, dti.minute, dti.second, nanos, strict=True
        )
    ]


def _periods(index: pd.PeriodIndex) -> list[str]:
    starts = _stamp(pd.DatetimeIndex(index.start_time))
    ends = _stamp(pd.DatetimeIndex((index + 1).start_time))
    return [f"{a}/{b}" for a, b in zip(starts, ends, strict=True)]


def _label(v: Any) -> str | int:
    return int(v) if isinstance(v, int | np.integer) else str(v)


def _rows(index: pd.Index) -> list[Any]:
    if isinstance(index, pd.MultiIndex):
        outer = [_label(v) for v in index.get_level_values(0)]
        periods = _periods(pd.PeriodIndex(index.get_level_values(1)))
        return [[a, b] for a, b in zip(outer, periods, strict=True)]
    return _periods(index)


def _values(s: pd.Series, kind: str) -> list[Any]:
    missing = s.isna().to_numpy()
    if kind == "float":
        arr = s.to_numpy(dtype="float64", na_value=np.nan)
        arr = np.where(arr == 0.0, 0.0, arr)  # -0.0 and 0.0 are one value
        raw = arr.astype(">f8").tobytes().hex()
        return [None if m else raw[16 * i : 16 * i + 16] for i, m in enumerate(missing)]
    out = s.astype(object).tolist()
    if kind == "int":
        return [None if m else int(v) for v, m in zip(out, missing, strict=True)]
    if kind == "bool":
        return [None if m else bool(v) for v, m in zip(out, missing, strict=True)]
    return [None if m else str(v) for v, m in zip(out, missing, strict=True)]


def _meta(m: SeriesMeta) -> dict[str, Any]:
    return {
        "expected_sign": m.expected_sign,
        "frequency": m.frequency,
        "role": m.role,
        "source": m.source,
        "transforms": list(m.transforms),
        "unit": m.unit,
    }


def _columns(frame: pd.DataFrame, series: Mapping[str, SeriesMeta] | None) -> list[dict[str, Any]]:
    out = []
    for name in frame.columns:
        kind = _column_kind(name, frame[name], "")
        col: dict[str, Any] = {"kind": kind, "name": name, "values": _values(frame[name], kind)}
        if series is not None:
            col["meta"] = _meta(series[name])
        out.append(col)
    return out


# --- Dataset ------------------------------------------------------------------


class Dataset:
    """A frame of series with their metadata, scenarios and a log of changes.

    Immutable: the constructor copies `frame` and `scenarios`, the `frame` and
    `scenarios` properties return copies, and every method returns a new
    `Dataset`. Nothing changes the caller's objects.

    Parameters
    ----------
    frame
        One column per series (string names; float, integer, boolean or
        string values), indexed by a sorted, unique `pd.PeriodIndex`, or for
        panels by a ``(segment, period)`` MultiIndex.
    series
        `SeriesMeta` per column name; a column without one gets
        ``SeriesMeta()`` (role ``other``).
    scenarios
        Optional future paths: a frame indexed by ``(scenario, period)``
        whose columns are dataset columns, at the dataset's frequency.
    severity
        The scenario labels from mildest to most severe, each once. Defaults
        to their order of first appearance.
    log
        `LogEntry` records of the changes made so far, oldest first.
    """

    __slots__ = ("_frame", "_log", "_scenarios", "_series", "_severity")

    def __init__(
        self,
        frame: pd.DataFrame,
        series: Mapping[str, SeriesMeta | Mapping[str, Any]] | None = None,
        *,
        scenarios: pd.DataFrame | None = None,
        severity: Sequence[str | int] | None = None,
        log: Iterable[LogEntry] = (),
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise _bad(
                "the frame is not a pandas DataFrame",
                f"a {type(frame).__name__}",
                "Pass a pd.DataFrame.",
            )
        frame = _normalise_index(frame, "frame", "segment")
        if not frame.index.is_monotonic_increasing:
            raise _bad(
                "the frame index is not sorted",
                f"first rows {list(frame.index[:3])!r}",
                "Sort the frame by its index: frame.sort_index().",
            )
        _check_columns(frame, "frame")

        meta = dict(series or {})
        unknown = sorted(set(meta) - set(frame.columns))
        if unknown:
            raise _bad(
                "series metadata names columns the frame does not have",
                f"unknown: {', '.join(map(str, unknown))}; columns: {', '.join(frame.columns)}",
                "Name only frame columns in the metadata.",
            )
        resolved = {
            name: m if isinstance(m, SeriesMeta) else SeriesMeta.model_validate(m)
            for name, m in ((n, meta.get(n, SeriesMeta())) for n in frame.columns)
        }

        if scenarios is None:
            if severity is not None:
                raise _bad(
                    "a severity order was given without scenarios",
                    f"severity {list(severity)!r}",
                    "Pass the scenarios frame with its severity, or neither.",
                )
            sev: tuple[Any, ...] = ()
        else:
            scenarios, sev = _check_scenarios(scenarios, severity, frame)

        log = tuple(log)
        if not all(isinstance(e, LogEntry) for e in log):
            raise _bad(
                "the log holds something other than LogEntry records",
                f"{log!r}",
                "Pass LogEntry objects.",
            )

        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_series", MappingProxyType(resolved))
        object.__setattr__(self, "_scenarios", scenarios)
        object.__setattr__(self, "_severity", sev)
        object.__setattr__(self, "_log", log)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Dataset is immutable; use its with_* methods to make a changed copy ({name!r})"
        )

    __delattr__ = __setattr__

    # --- read --------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the data: changing it does not change the dataset."""
        return self._frame.copy(deep=True)

    @property
    def series(self) -> Mapping[str, SeriesMeta]:
        """Read-only mapping from column name to `SeriesMeta`, in column order."""
        return self._series

    @property
    def scenarios(self) -> pd.DataFrame | None:
        """A copy of the scenarios frame, or None."""
        return None if self._scenarios is None else self._scenarios.copy(deep=True)

    @property
    def severity(self) -> tuple[str | int, ...]:
        """Scenario labels from mildest to most severe; empty without scenarios."""
        return self._severity

    @property
    def log(self) -> tuple[LogEntry, ...]:
        """Every change made to the data, oldest first."""
        return self._log

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._frame.columns)

    @property
    def is_panel(self) -> bool:
        """True when indexed by ``(segment, period)``."""
        return isinstance(self._frame.index, pd.MultiIndex)

    @property
    def segments(self) -> tuple[str | int, ...]:
        """Segment labels in order of appearance; empty for a single-segment dataset."""
        if not self.is_panel:
            return ()
        return tuple(dict.fromkeys(_label(v) for v in self._frame.index.get_level_values(0)))

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(n for n, m in self._series.items() if m.role == "target")

    @property
    def drivers(self) -> tuple[str, ...]:
        return tuple(n for n, m in self._series.items() if m.role == "driver")

    def fingerprint(self) -> str:
        """SHA-256 (lowercase hex) of the canonical form of the frame, metadata and scenarios.

        The canonical form is fixed in design §5.1 (format
        ``dazflow.dataset/1``) and does not depend on the platform, the
        Python version or the pandas and numpy versions. The log is not part
        of it.
        """
        doc = {
            "columns": _columns(self._frame, self._series),
            "format": FINGERPRINT_FORMAT,
            "index": {
                "kind": "panel" if self.is_panel else "period",
                "rows": _rows(self._frame.index),
            },
            "scenarios": None
            if self._scenarios is None
            else {
                "columns": _columns(self._scenarios, None),
                "rows": _rows(self._scenarios.index),
                "severity": [_label(v) for v in self._severity],
            },
        }
        text = json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def __repr__(self) -> str:
        kind = "panel" if self.is_panel else "series"
        scen = f", {len(self._severity)} scenarios" if self._scenarios is not None else ""
        return f"<Dataset {kind}: {len(self._frame)} rows x {len(self.columns)} columns{scen}, {len(self._log)} log entries>"

    # --- changed copies ------------------------------------------------------

    def _replace(self, **changes: Any) -> Dataset:
        args = {
            "frame": self._frame,
            "series": self._series,
            "scenarios": self._scenarios,
            "severity": self._severity if self._scenarios is not None else None,
            "log": self._log,
        }
        args.update(changes)
        return Dataset(args.pop("frame"), args.pop("series"), **args)

    def with_frame(self, frame: pd.DataFrame, *, entry: LogEntry | None = None) -> Dataset:
        """A dataset with new data; metadata is kept for columns still present.

        New columns get ``SeriesMeta()``. `entry`, when given, is appended to the log.
        """
        series = {n: m for n, m in self._series.items() if n in frame.columns}
        log = self._log if entry is None else (*self._log, entry)
        return self._replace(frame=frame, series=series, log=log)

    def with_series(self, column: str, **changes: Any) -> Dataset:
        """A dataset with one column's metadata fields changed."""
        if column not in self._series:
            raise _bad(
                f"no column {column!r} in the dataset",
                f"columns: {', '.join(self.columns)}",
                "Name one of the dataset's columns.",
            )
        meta = dict(self._series)
        meta[column] = SeriesMeta.model_validate(self._series[column].model_dump() | changes)
        return self._replace(series=meta)

    def with_scenarios(
        self, scenarios: pd.DataFrame | None, severity: Sequence[str | int] | None = None
    ) -> Dataset:
        """A dataset with other scenarios (or none, when `scenarios` is None)."""
        return self._replace(scenarios=scenarios, severity=severity)

    def with_log(self, entry: LogEntry) -> Dataset:
        """A dataset with `entry` appended to the log."""
        return self._replace(log=(*self._log, entry))

    def select(self, columns: Sequence[str]) -> Dataset:
        """A dataset with only `columns`, in that order, and their metadata.

        Scenarios keep only the selected columns.
        """
        columns = list(columns)
        unknown = [c for c in columns if c not in self._series]
        if unknown:
            raise _bad(
                "select names columns the dataset does not have",
                f"unknown: {', '.join(map(str, unknown))}; columns: {', '.join(self.columns)}",
                "Name only the dataset's columns.",
            )
        scen = self._scenarios
        if scen is not None:
            scen = scen[[c for c in scen.columns if c in columns]]
        return self._replace(
            frame=self._frame[columns],
            series={c: self._series[c] for c in columns},
            scenarios=scen,
        )


def _check_scenarios(
    scenarios: pd.DataFrame, severity: Sequence[Any] | None, frame: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[Any, ...]]:
    if not isinstance(scenarios, pd.DataFrame):
        raise _bad(
            "the scenarios are not a pandas DataFrame",
            f"a {type(scenarios).__name__}",
            "Pass a pd.DataFrame.",
        )
    if not isinstance(scenarios.index, pd.MultiIndex):
        raise _bad(
            "the scenarios index must be (scenario, period)",
            f"{type(scenarios.index).__name__}",
            "Index the scenarios by a (scenario, period) MultiIndex.",
        )
    scenarios = _normalise_index(scenarios, "scenarios", "scenario")
    _check_columns(scenarios, "scenarios")
    unknown = [c for c in scenarios.columns if c not in frame.columns]
    if unknown:
        raise _bad(
            "the scenarios have columns the dataset does not have",
            f"unknown: {', '.join(unknown)}; columns: {', '.join(frame.columns)}",
            "Name the scenario columns after dataset series.",
        )
    if _freq(scenarios.index) != _freq(frame.index):
        raise _bad(
            "the scenarios frequency differs from the dataset's",
            f"scenarios {_freq(scenarios.index)}, dataset {_freq(frame.index)}",
            "Give the scenario periods the dataset's frequency.",
        )
    labels = tuple(dict.fromkeys(_label(v) for v in scenarios.index.get_level_values(0)))
    if severity is None:
        return scenarios, labels
    sev = tuple(_label(v) for v in severity)
    if len(set(sev)) != len(sev) or set(sev) != set(labels):
        raise _bad(
            "the severity order must list each scenario once",
            f"severity {list(sev)!r}; scenarios {list(labels)!r}",
            "List every scenario label once, mildest first.",
        )
    return scenarios, sev
