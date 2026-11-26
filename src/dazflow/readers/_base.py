"""The reader protocol, the registry, `read`, `from_frame` and the date rules."""

from __future__ import annotations

import datetime as dt
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pydantic import Field

from dazflow.core.config import Config
from dazflow.core.dataset import Dataset, SeriesMeta
from dazflow.core.errors import MissingExtraError, ReaderError
from dazflow.core.registry import Registry

__all__ = [
    "FREQUENCIES",
    "Frequency",
    "ReadSpec",
    "Reader",
    "from_frame",
    "get",
    "read",
    "register",
    "registry",
    "require",
    "to_periods",
]

Frequency = Literal["M", "Q", "Y"]

# Frequency code to its name, finest first.
FREQUENCIES: Mapping[str, str] = {"M": "monthly", "Q": "quarterly", "Y": "annual"}
_FINENESS = {f: i for i, f in enumerate(FREQUENCIES)}
# pandas' spelling of each period frequency (A-DEC before pandas 2.2).
_FREQSTR = {"M": "M", "Q-DEC": "Q", "Y-DEC": "Y", "A-DEC": "Y"}


def frequency_of(index: pd.PeriodIndex) -> str | None:
    """``"M"``, ``"Q"`` or ``"Y"`` for a period index at that frequency, else None."""
    return _FREQSTR.get(index.freqstr)


class ReadSpec(Config):
    """What to read and how to put its dates on a calendar.

    The first column holds the dates unless `date` names another (a frame
    indexed by periods or dates uses its index). Every other column is a
    series, or only `columns` when given. `options` go to the reader's own
    `Config` (``sep`` for csv, ``sheet`` for excel).
    """

    reader: str | None = Field(None, description="Reader key; unset picks it from the file suffix.")
    path: str | None = None
    date: str | None = Field(None, description="The column holding the dates; unset: the first.")
    frequency: Frequency | None = Field(
        None, description="The data's own frequency; unset reads it from the dates."
    )
    columns: list[str] | None = None
    options: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class Reader(Protocol):
    """A data source. `Config` holds its own options; `read` returns a `Dataset`."""

    Config: type[Config]

    def read(self, spec: ReadSpec) -> Dataset:
        """The data `spec` names; helpers `from_frame` and `require` do the common work."""


registry = Registry("reader", protocol=Reader)
register = registry.register
get = registry.get

SUFFIXES: Mapping[str, str] = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def read(spec: ReadSpec | Mapping[str, Any]) -> Dataset:
    """Read the data `spec` names into a `Dataset` with one column per series."""
    spec = spec if isinstance(spec, ReadSpec) else ReadSpec.model_validate(spec)
    key = spec.reader
    if key is None:
        suffix = Path(spec.path).suffix.lower() if spec.path else ""
        key = SUFFIXES.get(suffix)
        if key is None:
            raise ReaderError(
                f"no reader is known for {spec.path!r}",
                observed=f"suffix {suffix or 'none'}; known: {', '.join(SUFFIXES)}",
                change="Name the reader: reader: csv (or another registered reader key).",
            )
    return get(key)().read(spec)


def require(module: str, what: str, extra: str = "io") -> Any:
    """Import an optional `module`, or raise `MissingExtraError` naming ``dazflow[extra]``."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise MissingExtraError(what, module, extra) from err


def path_of(spec: ReadSpec, what: str) -> str:
    if not spec.path:
        raise ReaderError(
            f"{what} needs a path",
            observed="path is not set",
            change="Give the file's path: path: data/series.csv.",
        )
    return spec.path


# --- dates to periods -----------------------------------------------------------


def _bad(problem: str, observed: str, change: str) -> ReaderError:
    return ReaderError(problem, observed=observed, change=change)


def _unique(periods: pd.PeriodIndex, freq: str) -> pd.PeriodIndex:
    if periods.has_duplicates:
        first = periods[periods.duplicated()][0]
        raise _bad(
            f"two rows fall in the period {first}",
            f"{int(periods.duplicated().sum())} repeated {FREQUENCIES[freq]} periods",
            "Set frequency to the data's own frequency (ingest converts it), "
            "or aggregate the rows before reading.",
        )
    return periods


def _at(periods: pd.PeriodIndex, native: str, frequency: str | None) -> pd.PeriodIndex:
    if frequency is None or frequency == native:
        return _unique(periods, native)
    if _FINENESS[frequency] < _FINENESS[native]:
        raise _bad(
            f"the dates are {FREQUENCIES[native]}, coarser than the frequency {frequency!r}",
            f"first dates {[str(p) for p in periods[:3]]}",
            f"Set frequency: {native}, or leave it unset.",
        )
    return _unique(periods.asfreq(frequency, how="end"), frequency)


def _modal_step(periods: pd.PeriodIndex) -> int | None:
    ordinals = np.unique(periods.asi8)
    if len(ordinals) < 2:
        return None
    steps, counts = np.unique(np.diff(ordinals), return_counts=True)
    return int(steps[np.argmax(counts)])  # ties go to the smaller step


def _infer(dates: pd.DatetimeIndex, frequency: str | None) -> pd.PeriodIndex:
    if frequency is not None:
        return _unique(dates.to_period(frequency), frequency)
    for freq in FREQUENCIES:
        periods = dates.to_period(freq)
        if not periods.has_duplicates and _modal_step(periods) == 1:
            return periods
    raise _bad(
        "the frequency of the dates cannot be told",
        f"{len(dates)} dates from {dates.min().date()} to {dates.max().date()}",
        "Set frequency: M, Q or Y. Daily and weekly data are not read; "
        "aggregate them to months first.",
    )


def _parse(text: str) -> pd.Period:
    try:
        return pd.Period(text)
    except (ValueError, TypeError) as err:
        raise _bad(
            f"the date {text!r} cannot be read",
            f"{type(err).__name__}: {err}",
            "Write dates as 2024-03-31, 2024-03, 2024Q1 or 2024.",
        ) from err


def _timestamps(make: Any, values: pd.Index) -> pd.DatetimeIndex:
    """`make()` as a DatetimeIndex; a date pandas cannot hold as a timestamp is a ReaderError.

    pandas 2 keeps timestamps in nanoseconds, from 1677 to 2262 (issue #37).
    """
    try:
        return pd.DatetimeIndex(make())
    except (pd.errors.OutOfBoundsDatetime, OverflowError) as err:
        raise _bad(
            "a date is outside the range pandas can put on a calendar",
            f"dates from {min(map(str, values))} to {max(map(str, values))}: {err}",
            "Correct the dates; pandas 2 holds dates from 1677 to 2262.",
        ) from err


def to_periods(values: Sequence[Any] | pd.Index, frequency: str | None = None) -> pd.PeriodIndex:
    """The periods of `values` (periods, dates, years or date text) at `frequency` or the inferred one."""
    values = pd.Index(values)
    if len(values) == 0:
        raise _bad("there are no rows to read", "0 rows", "Read a file or frame with data.")
    if values.isna().any():
        i = int(np.flatnonzero(values.isna())[0])
        raise _bad(
            "a row has no date",
            f"row {i + 1} of {len(values)}",
            "Give every row a date, or drop the rows without one.",
        )
    if isinstance(values.dtype, pd.PeriodDtype):
        periods = pd.PeriodIndex(values)
        native = frequency_of(periods)
        if native is None:
            raise _bad(
                f"the periods are at an unsupported frequency {periods.freqstr}",
                f"first periods {[str(p) for p in periods[:3]]}",
                "Use monthly, quarterly (calendar quarters) or annual periods.",
            )
        return _at(periods, native, frequency)
    if ptypes.is_datetime64_any_dtype(values.dtype):
        dates = pd.DatetimeIndex(values)
        return _infer(dates.tz_localize(None) if dates.tz is not None else dates, frequency)
    if all(isinstance(v, dt.date) for v in values):
        return _infer(_timestamps(lambda: pd.to_datetime(list(values)), values), frequency)
    if ptypes.is_integer_dtype(values.dtype) or ptypes.is_float_dtype(values.dtype):
        if ptypes.is_float_dtype(values.dtype) and not all(float(v).is_integer() for v in values):
            raise _bad(
                "numeric dates must be whole years",
                f"first values {list(values[:3])}",
                "Write dates as 2024-03-31, 2024-03, 2024Q1 or 2024.",
            )
        values = pd.Index([str(int(v)) for v in values])
    parsed = [_parse(str(v)) for v in values]
    kinds = {_FREQSTR.get(p.freqstr) for p in parsed}
    if None in kinds:  # finer than a month: dates, read as days
        return _infer(_timestamps(lambda: [p.start_time for p in parsed], values), frequency)
    if len(kinds) > 1:
        raise _bad(
            "the dates are written at different frequencies",
            f"frequencies {sorted(FREQUENCIES[k] for k in kinds)}",
            "Write every date at one frequency.",
        )
    (native,) = kinds
    return _at(pd.PeriodIndex(parsed, freq=native), native, frequency)


# --- frames to datasets ----------------------------------------------------------


def _pandas(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    import narwhals as nw

    try:
        frame = nw.from_native(obj, eager_only=True)
    except TypeError as err:
        raise _bad(
            f"a {type(obj).__name__} is not a data frame",
            f"type {type(obj).__module__}.{type(obj).__qualname__}",
            "Pass a pandas or polars DataFrame, or a pyarrow Table.",
        ) from err
    return frame.to_pandas()


def from_frame(
    obj: Any,
    *,
    date: str | None = None,
    frequency: str | None = None,
    columns: Sequence[str] | None = None,
    source: str | None = None,
) -> Dataset:
    """A `Dataset` from a pandas or polars frame or a pyarrow table.

    Non-pandas input is converted through narwhals. The dates are the
    `date` column, else the index when it holds periods or dates, else the
    first column. `source`, when given, is recorded in each series' metadata.
    """
    frame = _pandas(obj)
    if date is None and isinstance(frame.index, pd.PeriodIndex | pd.DatetimeIndex):
        dates = frame.index
        data = frame.reset_index(drop=True)
    else:
        names = [str(c) for c in frame.columns]
        if date is None:
            if not names:
                raise _bad("the frame has no columns", "0 columns", "Read data with a date column.")
            date = names[0]
        if date not in names:
            raise _bad(
                f"the date column {date!r} is not in the data",
                f"columns: {', '.join(names)}",
                "Name the column that holds the dates: date: <column>.",
            )
        frame = frame.set_axis(names, axis=1)
        dates = frame[date]
        data = frame.drop(columns=date).reset_index(drop=True)
    data.columns = [str(c) for c in data.columns]
    if columns is not None:
        unknown = [c for c in columns if c not in data.columns]
        if unknown:
            raise _bad(
                f"the columns {', '.join(map(repr, unknown))} are not in the data",
                f"columns: {', '.join(data.columns)}",
                "Name only columns the data has.",
            )
        data = data[list(columns)]
    periods = to_periods(dates, frequency)
    data.index = periods
    data = data.sort_index()
    freq = frequency_of(periods)
    return Dataset(data, {c: SeriesMeta(frequency=freq, source=source) for c in data.columns})
