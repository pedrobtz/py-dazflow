"""Table, Figure, Source and artefact identifiers: what a step returns for a document (design §9).

These are plain data, JSON-safe so that a `StepResult` round-trips through
the run record. `dazflow.render` turns them into Word, HTML, Markdown, PNG
and SVG in the house style; no third-party object is held here.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import AllowInfNan, Field, field_validator, model_validator

from dazflow.core._model import DazflowModel
from dazflow.core.errors import ArtefactError

if TYPE_CHECKING:
    import pandas as pd

    from dazflow.core.gate import Gate

__all__ = [
    "STEP_NUMBERS",
    "ArtefactIds",
    "Column",
    "ColumnKind",
    "Figure",
    "Layer",
    "Source",
    "Table",
    "id_sort_key",
    "parse_id",
]

# The step numbers in identifiers: the steps' section numbers in design §7.
STEP_NUMBERS: Mapping[str, int] = {
    "ingest": 1,
    "clean": 2,
    "explore": 3,
    "build": 4,
    "segment": 5,
    "select": 6,
    "forecast": 7,
    "diagnose": 8,
    "adjust": 9,
    "report": 10,
}

# T: table, F: figure, S: finding (a sentence). Numbers start at 1, no leading zeros.
_ID = re.compile(r"(?P<kind>[TFS])(?P<step>[1-9]\d*)\.(?P<n>[1-9]\d*)")
_KIND_LETTER = {"table": "T", "figure": "F", "finding": "S"}
_LETTER_ORDER = {"T": 0, "F": 1, "S": 2}


def parse_id(identifier: str) -> tuple[str, int, int]:
    """Split ``"T3.2"`` into ``("T", 3, 2)``; anything else is an `ArtefactError`."""
    m = _ID.fullmatch(identifier) if isinstance(identifier, str) else None
    if m is None:
        raise ArtefactError(
            f"{identifier!r} is not an artefact identifier",
            observed=f"identifier {identifier!r}",
            change=(
                "Use T<step>.<n> for a table, F<step>.<n> for a figure or S<step>.<n> "
                "for a finding, numbers from 1 without leading zeros, e.g. T3.2."
            ),
        )
    return m["kind"], int(m["step"]), int(m["n"])


def id_sort_key(identifier: str) -> tuple[int, int, int]:
    """Tables, then figures, then findings; each by step, then by number."""
    letter, step, n = parse_id(identifier)
    return _LETTER_ORDER[letter], step, n


def _check_id(identifier: str, letter: str, what: str) -> str:
    try:
        kind = parse_id(identifier)[0]
    except ArtefactError as err:
        raise ValueError(err.problem) from None
    if kind != letter:
        raise ValueError(f"a {what} identifier starts with {letter}, not {identifier!r}")
    return identifier


class ArtefactIds:
    """The identifiers of one step's artefacts, fixed by name, not by creation order.

    Each artefact a step can make is declared once with its number, so
    ``T3.2`` stays the stationarity table on every rerun, whatever order the
    step builds its artefacts in and whichever of them it builds. Add a new
    artefact under a new number; never renumber one, or documents written
    earlier point at the wrong table.

    Parameters
    ----------
    step
        A step name from `STEP_NUMBERS` (``"explore"``), or a positive
        number for a step outside the built-in ones.
    tables, figures, findings
        Name to number, each number positive and used once per kind.

    Examples
    --------
    >>> ids = ArtefactIds("explore", tables={"summary": 1, "stationarity": 2})
    >>> ids.table("stationarity")
    'T3.2'
    """

    def __init__(
        self,
        step: str | int,
        *,
        tables: Mapping[str, int] | None = None,
        figures: Mapping[str, int] | None = None,
        findings: Mapping[str, int] | None = None,
    ) -> None:
        self._step = self._step_number(step)
        self._names = {
            "table": self._numbers("table", tables or {}),
            "figure": self._numbers("figure", figures or {}),
            "finding": self._numbers("finding", findings or {}),
        }

    @staticmethod
    def _step_number(step: str | int) -> int:
        if isinstance(step, str) and step in STEP_NUMBERS:
            return STEP_NUMBERS[step]
        if isinstance(step, int) and not isinstance(step, bool) and step >= 1:
            return step
        raise ArtefactError(
            f"{step!r} is not a step",
            observed=f"known steps: {', '.join(STEP_NUMBERS)}",
            change="Name a built-in step, or give a positive step number.",
        )

    @staticmethod
    def _numbers(kind: str, numbers: Mapping[str, int]) -> dict[str, int]:
        seen: dict[int, str] = {}
        for name, n in numbers.items():
            if not isinstance(name, str) or not name:
                raise ArtefactError(
                    f"a {kind} name must be a non-empty string",
                    observed=f"name {name!r}",
                    change=f"Name each {kind} with a non-empty string.",
                )
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ArtefactError(
                    f"the {kind} {name!r} needs a positive whole number",
                    observed=f"{name!r}: {n!r}",
                    change="Number artefacts 1, 2, 3, ...",
                )
            if n in seen:
                raise ArtefactError(
                    f"two {kind}s share number {n}",
                    observed=f"{seen[n]!r} and {name!r}",
                    change=f"Give each {kind} its own number; never reuse a retired one.",
                )
            seen[n] = name
        return dict(numbers)

    @property
    def step(self) -> int:
        return self._step

    def _id(self, kind: str, name: str) -> str:
        names = self._names[kind]
        if name not in names:
            raise ArtefactError(
                f"no {kind} named {name!r} is declared for step {self._step}",
                observed=f"declared {kind}s: {', '.join(sorted(names)) or 'none'}",
                change=f"Declare the {kind} in the step's ArtefactIds with its own number.",
            )
        return f"{_KIND_LETTER[kind]}{self._step}.{names[name]}"

    def table(self, name: str) -> str:
        """The identifier of the table declared as `name`: ``T<step>.<n>``."""
        return self._id("table", name)

    def figure(self, name: str) -> str:
        """The identifier of the figure declared as `name`: ``F<step>.<n>``."""
        return self._id("figure", name)

    def finding(self, name: str) -> str:
        """The identifier of the finding declared as `name`: ``S<step>.<n>``."""
        return self._id("finding", name)

    def __repr__(self) -> str:
        return f"ArtefactIds(step={self._step}, {self._names!r})"


class Source(DazflowModel):
    """Where an artefact's numbers come from: the run, the data and the date.

    Every source note in a document is written from one of these.
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = (
        "Give the run identifier, the dataset's fingerprint() (64 hexadecimal digits) and a date."
    )

    run_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    date: dt.date
    note: str | None = None


ColumnKind = Literal["number", "integer", "percent", "text", "date", "status"]
STATUSES = ("pass", "fail", "overridden")


class Column(DazflowModel):
    """One table column: its key, its plain-language label, its unit and the kind of value.

    Kinds: ``number`` (decimals from the style), ``integer``, ``percent`` (a
    fraction, shown times 100), ``text``, ``date`` (ISO ``YYYY-MM-DD``) and
    ``status`` (``pass``, ``fail`` or ``overridden``, marked in place).
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = "Give each column a name, a label and one of the column kinds."

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    unit: str | None = None
    kind: ColumnKind = "number"

    @property
    def heading(self) -> str:
        """The label with its unit: ``Mean (pp)``."""
        return f"{self.label} ({self.unit})" if self.unit else self.label


def _cell_problem(value: Any, kind: str) -> str | None:
    if value is None:
        return "a status is required" if kind == "status" else None
    if kind in ("number", "percent"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"{value!r} is not a number"
        if isinstance(value, float) and not math.isfinite(value):
            return f"{value!r} is not a finite number; leave the cell empty (None) instead"
        return None
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{value!r} is not a whole number"
        return None
    if kind == "text":
        return None if isinstance(value, str) else f"{value!r} is not text"
    if kind == "date":
        if isinstance(value, str):
            try:
                dt.date.fromisoformat(value)
            except ValueError:
                pass
            else:
                if len(value) == 10:
                    return None
        return f"{value!r} is not a date written YYYY-MM-DD"
    if value not in STATUSES:
        return f"{value!r} is not one of {', '.join(STATUSES)}"
    return None


def _plain(value: Any) -> Any:
    """A cell as JSON-safe Python: dates as ISO text."""
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time() else value
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


class Table(DazflowModel):
    """A table for a document: labelled columns, rows of plain values, a caption and a source.

    Values are stored unformatted; the style formats them when the table is
    rendered, so one style setting changes every table. A missing value is
    ``None``. A ``status`` column marks pass or fail in place.

    Examples
    --------
    >>> t = Table(id="T3.1", caption="Means",
    ...           columns=[Column(name="x", label="Mean", unit="pp")], rows=[[1.25]])
    >>> t.columns[0].heading
    'Mean (pp)'
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = "Correct the table fields listed in 'observed'."

    kind: Literal["table"] = "table"
    id: str
    caption: str
    columns: tuple[Column, ...]
    rows: tuple[tuple[Any, ...], ...] = ()
    notes: tuple[str, ...] = ()
    source: Source | None = None

    @field_validator("id")
    @classmethod
    def _id_is_a_table_id(cls, value: str) -> str:
        return _check_id(value, "T", "table")

    @field_validator("rows", mode="before")
    @classmethod
    def _dates_as_text(cls, rows: Any) -> Any:
        if isinstance(rows, Iterable) and not isinstance(rows, str | bytes):
            return [
                [_plain(v) for v in row]
                if isinstance(row, Iterable) and not isinstance(row, str)
                else row
                for row in rows
            ]
        return rows

    @model_validator(mode="after")
    def _cells_fit_their_columns(self) -> Table:
        names = [c.name for c in self.columns]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"column names must be unique; repeated: {', '.join(dupes)}")
        for i, row in enumerate(self.rows, start=1):
            if len(row) != len(self.columns):
                raise ValueError(f"row {i} has {len(row)} cells for {len(self.columns)} columns")
            for col, value in zip(self.columns, row, strict=True):
                problem = _cell_problem(value, col.kind)
                if problem:
                    raise ValueError(f"row {i}, column {col.name!r} ({col.kind}): {problem}")
        return self

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        id: str,
        caption: str,
        labels: Mapping[str, str] | None = None,
        units: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        index: str | None = None,
        notes: Iterable[str] = (),
        source: Source | None = None,
    ) -> Table:
        """A table from a pandas frame, the column kinds read from the dtypes.

        Integer columns become ``integer``, floats ``number``, datetimes
        ``date`` and everything else ``text``; `kinds` overrides that. Missing
        values become empty cells. `index`, when given, names a first column
        holding the index (periods as ``2024Q1``).
        """
        import pandas as pd

        labels, units, kinds = dict(labels or {}), dict(units or {}), dict(kinds or {})
        data = frame.copy()
        if index is not None:
            data.insert(0, index, data.index.astype(str))
            data = data.reset_index(drop=True)
        names = [str(c) for c in data.columns]
        unknown = sorted((set(labels) | set(units) | set(kinds)) - set(names))
        if unknown:
            raise ArtefactError(
                "labels, units or kinds name columns the frame is missing",
                observed=f"missing: {', '.join(unknown)}; frame columns: {', '.join(names)}",
                change="Name only columns of the frame.",
            )

        def infer(series: pd.Series) -> ColumnKind:
            if pd.api.types.is_bool_dtype(series):
                return "text"
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            if pd.api.types.is_float_dtype(series):
                return "number"
            if pd.api.types.is_datetime64_any_dtype(series):
                return "date"
            return "text"

        def cell(value: Any, kind: str) -> Any:
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                return None
            if kind == "integer":
                return int(value)
            if kind in ("number", "percent"):
                return float(value)
            if kind == "date":
                return pd.Timestamp(value).date().isoformat()
            if kind == "text":
                return str(value)
            return value

        columns = []
        for name, (_, series) in zip(names, data.items(), strict=True):
            kind = kinds.get(name, infer(series))
            columns.append(
                Column(name=name, label=labels.get(name, name), unit=units.get(name), kind=kind)
            )
        rows = [
            [cell(v, c.kind) for v, c in zip(row, columns, strict=True)]
            for row in data.itertuples(index=False, name=None)
        ]
        return cls(
            id=id, caption=caption, columns=columns, rows=rows, notes=tuple(notes), source=source
        )

    @classmethod
    def from_gate(
        cls,
        gate: Gate,
        *,
        id: str,
        caption: str,
        notes: Iterable[str] = (),
        source: Source | None = None,
    ) -> Table:
        """The gate's checks, one per row, with pass, fail or overridden marked in place."""
        columns = [
            Column(name="check", label="Check", kind="text"),
            Column(name="observed", label="Observed"),
            Column(name="op", label="Rule", kind="text"),
            Column(name="threshold", label="Threshold"),
            Column(name="unit", label="Unit", kind="text"),
            Column(name="status", label="Result", kind="status"),
            Column(name="reason", label="Reason", kind="text"),
        ]
        rows = [
            [
                c.name,
                None if math.isnan(c.observed) else c.observed,
                c.op,
                c.threshold,
                c.unit or "",
                c.status,
                c.override.reason if c.override is not None else "",
            ]
            for c in gate.checks
        ]
        return cls(
            id=id, caption=caption, columns=columns, rows=rows, notes=tuple(notes), source=source
        )


_Finite = Annotated[float, AllowInfNan(False)]
LayerKind = Literal["line", "scatter", "bar", "band", "hline"]


def _no_bools(values: Any, name: str) -> Any:
    if isinstance(values, Iterable) and not isinstance(values, str):
        values = list(values)
        if any(isinstance(v, bool) for v in values):
            raise ValueError(f"{name} holds a boolean, not a number")
    elif isinstance(values, bool):
        raise ValueError(f"{name} is a boolean, not a number")
    return values


class Layer(DazflowModel):
    """One thing drawn on a figure, against the figure's shared x values.

    ``line``, ``scatter`` and ``bar`` take `y`; ``band`` takes `y_low` and
    `y_high` (a shaded interval); ``hline`` takes `value` (a reference line).
    ``None`` in `y` leaves a gap.
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = (
        "Give line, scatter and bar layers y, band layers y_low and y_high, hline layers value."
    )

    kind: LayerKind
    y: tuple[_Finite | None, ...] = ()
    y_low: tuple[_Finite | None, ...] = ()
    y_high: tuple[_Finite | None, ...] = ()
    value: _Finite | None = None
    label: str | None = None

    @field_validator("y", "y_low", "y_high", "value", mode="before")
    @classmethod
    def _numbers_not_booleans(cls, values: Any, info: Any) -> Any:
        return _no_bools(values, info.field_name)

    @model_validator(mode="after")
    def _values_fit_the_kind(self) -> Layer:
        given = {f for f in ("y", "y_low", "y_high") if getattr(self, f)} | (
            {"value"} if self.value is not None else set()
        )
        need = {
            "line": {"y"},
            "scatter": {"y"},
            "bar": {"y"},
            "band": {"y_low", "y_high"},
            "hline": {"value"},
        }[self.kind]
        if given != need:
            raise ValueError(
                f"a {self.kind} layer takes {', '.join(sorted(need))}; "
                f"given: {', '.join(sorted(given)) or 'nothing'}"
            )
        if self.kind == "band" and len(self.y_low) != len(self.y_high):
            raise ValueError(f"y_low has {len(self.y_low)} values and y_high {len(self.y_high)}")
        return self

    @property
    def length(self) -> int | None:
        """How many x values the layer needs; ``None`` for a reference line."""
        if self.kind == "hline":
            return None
        return len(self.y_low) if self.kind == "band" else len(self.y)


class Figure(DazflowModel):
    """A chart for a document, as data: x values, layers, labelled axes, a caption and a source.

    `x` holds numbers or labels (periods as ``2024Q1``), shared by every
    layer. The style draws it at the page width; `aspect` is height over
    width.
    """

    _error: ClassVar = ArtefactError
    _hint: ClassVar[str] = "Correct the figure fields listed in 'observed'."

    kind: Literal["figure"] = "figure"
    id: str
    caption: str
    x: tuple[Any, ...] = ()
    layers: tuple[Layer, ...]
    x_label: str = ""
    y_label: str = ""
    x_unit: str | None = None
    y_unit: str | None = None
    aspect: float = Field(default=0.5, ge=0.2, le=2.0)
    notes: tuple[str, ...] = ()
    source: Source | None = None

    @field_validator("id")
    @classmethod
    def _id_is_a_figure_id(cls, value: str) -> str:
        return _check_id(value, "F", "figure")

    @field_validator("x", mode="before")
    @classmethod
    def _x_is_numbers_or_labels(cls, values: Any) -> Any:
        values = list(_no_bools(values, "x"))
        numbers = [isinstance(v, int | float) for v in values]
        labels = [isinstance(v, str) for v in values]
        if not (all(numbers) or all(labels)):
            raise ValueError("x holds numbers or labels, not both or other values")
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            raise ValueError("x holds a value that is not a finite number")
        return values

    @model_validator(mode="after")
    def _layers_fit_x(self) -> Figure:
        if not self.layers:
            raise ValueError("a figure needs at least one layer")
        for i, layer in enumerate(self.layers, start=1):
            if layer.length is not None and layer.length != len(self.x):
                raise ValueError(
                    f"layer {i} ({layer.kind}) has {layer.length} values for {len(self.x)} x values"
                )
        return self

    @property
    def x_heading(self) -> str:
        return f"{self.x_label} ({self.x_unit})" if self.x_unit else self.x_label

    @property
    def y_heading(self) -> str:
        return f"{self.y_label} ({self.y_unit})" if self.y_unit else self.y_label
