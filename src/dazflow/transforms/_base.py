"""The transform protocol, the registry, the text form and the chain helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from pydantic import Field

from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset, LogEntry
from dazflow.core.errors import ConfigError, TransformError
from dazflow.core.registry import Registry

__all__ = [
    "Transform",
    "TransformSpec",
    "apply",
    "consumed",
    "forward",
    "get",
    "inverse",
    "parse",
    "register",
    "registry",
]


class Transform(Protocol):
    """What a transform provides. Built-ins and plugins are classes registered by key.

    Each method receives the series (a float `pd.Series` on a `PeriodIndex`,
    named after the column) and the transform's validated `Config`.
    """

    Config: type[Config]

    def resolve(self, x: pd.Series, config: Any) -> Any:
        """The config with every parameter fixed from the data (a period, an estimate)."""

    def forward(self, x: pd.Series, config: Any) -> pd.Series:
        """The transformed series, on the same index; `TransformError` outside the domain."""

    def inverse(self, y: pd.Series, config: Any, anchor: pd.Series) -> pd.Series:
        """The level series on `y`'s index. `anchor` holds known levels for lagged values."""

    def consumed(self, config: Any) -> int:
        """How many leading periods the forward transform leaves without a value."""


registry = Registry("transform", protocol=Transform)
register = registry.register
get = registry.get


# --- text form ---------------------------------------------------------------

_TEXT = re.compile(r"\s*(?P<key>[A-Za-z_][\w.-]*)\s*(?:\((?P<args>[^()]*)\))?\s*")
_ARG = re.compile(r"\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^,=\s]+)\s*")


def _value(text: str) -> Any:
    if text in ("None", "null"):
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _bad_text(text: object, why: str) -> TransformError:
    return TransformError(
        f"{text!r} is not a transform",
        observed=why,
        change="Write a transform as key or key(param=value, ...), e.g. 'log' or 'diff(lag=4)'.",
    )


class TransformSpec(DazflowModel):
    """One transform of a chain: its registry key and its parameters.

    ``str(spec)`` is the text form recorded in `SeriesMeta.transforms`:
    the key, then the parameters in their declared order, leaving out any
    that are unset (``None``).
    """

    _error: ClassVar = TransformError
    _hint: ClassVar[str] = "Give a transform key and its parameters."

    key: str
    params: dict[str, int | float | None] = Field(default_factory=dict)

    @property
    def config(self) -> Config:
        return get(self.key).Config.model_validate(self.params)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.config.model_dump().items() if v is not None)
        return f"{self.key}({args})" if args else self.key


def parse(text: str | TransformSpec) -> TransformSpec:
    """A `TransformSpec` from its text form; its parameters validated by the transform's `Config`."""
    if isinstance(text, TransformSpec):
        key, params = text.key, dict(text.params)
    else:
        m = _TEXT.fullmatch(text) if isinstance(text, str) else None
        if m is None:
            raise _bad_text(text, "not of the form key or key(param=value, ...)")
        key, params = m["key"], {}
        if m["args"] is not None and m["args"].strip():
            for part in m["args"].split(","):
                a = _ARG.fullmatch(part)
                if a is None:
                    raise _bad_text(text, f"the parameter {part.strip()!r} is not name=value")
                params[a["name"]] = _value(a["value"])
    impl = get(key)  # UnknownKeyError lists the known keys
    try:  # the transform's own Config checks the raw values, so its error names the parameter
        config = impl.Config.model_validate(params)
    except ConfigError as err:
        raise ConfigError(
            f"the parameters of {key!r} are not valid",
            observed=err.observed,
            change=f"Give {key} only the parameters of {impl.Config.__name__}.",
        ) from err
    return TransformSpec(key=key, params=config.model_dump())


def _chain(chain: Iterable[str | TransformSpec] | str | TransformSpec) -> list[TransformSpec]:
    if isinstance(chain, str | TransformSpec):
        chain = [chain]
    return [parse(t) for t in chain]


# --- series checks -----------------------------------------------------------


def _name(x: pd.Series) -> str:
    return "series" if x.name is None else str(x.name)


def _series(x: Any, what: str = "the series") -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TransformError(
            f"{what} is not a pandas Series",
            observed=f"a {type(x).__name__}",
            change="Pass a pd.Series indexed by a pd.PeriodIndex.",
        )
    if not isinstance(x.index, pd.PeriodIndex):
        raise TransformError(
            f"{what} {_name(x)!r} is not indexed by a PeriodIndex",
            observed=f"{type(x.index).__name__} of {x.index.dtype}",
            change="Index the series by pandas periods (pd.PeriodIndex).",
            series=_name(x),
        )
    if x.index.has_duplicates or not x.index.is_monotonic_increasing:
        raise TransformError(
            f"{what} {_name(x)!r} has a repeated or unsorted period",
            observed=f"first periods {[str(p) for p in x.index[:3]]}",
            change="Keep one value per period, sorted: x.sort_index().",
            series=_name(x),
        )
    try:
        values = x.astype("float64")
    except (TypeError, ValueError) as err:
        raise TransformError(
            f"{what} {_name(x)!r} is not numeric",
            observed=f"dtype {x.dtype}",
            change="Transform numeric series only.",
            series=_name(x),
        ) from err
    return values


def domain_error(
    key: str, x: pd.Series, bad: np.ndarray, rule: str, *, inverse: bool = False
) -> None:
    """Raise `TransformError` at the first True of `bad`, naming the series, period and value."""
    if not bad.any():
        return
    i = int(np.flatnonzero(bad)[0])
    period = str(x.index[i])
    value = float(x.iloc[i])
    what = f"the inverse of {key}" if inverse else key
    raise TransformError(
        f"{what} is not defined for {value!r} in series {_name(x)!r} at {period}",
        observed=f"{int(bad.sum())} value(s) outside the domain ({rule}), first at {period}",
        change=f"Use {key} only on values where {rule}, or choose another transform.",
        series=_name(x),
        period=period,
    )


def shifted(x: pd.Series, lag: int) -> np.ndarray:
    """The value `lag` periods earlier for each period of `x` (NaN where there is none)."""
    return x.reindex(x.index - lag).to_numpy(dtype="float64")


def undo_lag(
    y: pd.Series, anchor: pd.Series, lag: int, combine: Callable[[float, float], float]
) -> pd.Series:
    """Rebuild levels period by period: ``x[t] = combine(y[t], x[t - lag])``.

    ``x[t - lag]`` is taken from `anchor` when it has a value there, else
    from the levels already rebuilt.
    """
    known = {p: float(v) for p, v in anchor.items() if not math.isnan(v)}
    out = np.full(len(y), np.nan)
    for i, (p, v) in enumerate(y.items()):
        base = known.get(p - lag)
        if base is not None and not math.isnan(v):
            out[i] = combine(float(v), base)
            known.setdefault(p, out[i])
    return pd.Series(out, index=y.index, name=y.name)


def _finite(x: pd.Series, key: str) -> None:
    v = x.to_numpy()
    domain_error(key, x, np.isinf(v), "values are finite")


# --- chains ------------------------------------------------------------------


def forward(
    x: pd.Series, chain: Sequence[str | TransformSpec] | str | TransformSpec
) -> tuple[pd.Series, tuple[TransformSpec, ...]]:
    """Apply `chain` to `x`, in order.

    Returns the transformed series and the chain with every parameter
    resolved: record that one, it is what `inverse` takes. Missing values
    stay missing; a value outside a transform's domain raises
    `TransformError` naming the series and the period.
    """
    cur = _series(x)
    resolved = []
    for spec in _chain(chain):
        impl = get(spec.key)()
        _finite(cur, spec.key)
        config = impl.resolve(cur, spec.config)
        cur = impl.forward(cur, config).rename(cur.name)
        resolved.append(TransformSpec(key=spec.key, params=config.model_dump()))
    return cur, tuple(resolved)


def _require_resolved(specs: list[TransformSpec]) -> None:
    for spec in specs:
        unset = [k for k, v in spec.params.items() if v is None]
        if unset:
            raise TransformError(
                f"the transform {str(spec)!r} has unset parameters, so it cannot be inverted",
                observed=f"unset: {', '.join(unset)}",
                change="Invert the resolved chain that forward() returned, as recorded in the series metadata.",
            )


def consumed(chain: Sequence[str | TransformSpec] | str | TransformSpec) -> int:
    """How many leading periods a resolved chain leaves without a value."""
    specs = _chain(chain)
    _require_resolved(specs)
    return sum(get(s.key)().consumed(s.config) for s in specs)


def inverse(
    y: pd.Series,
    chain: Sequence[str | TransformSpec] | str | TransformSpec,
    history: pd.Series | None = None,
) -> pd.Series:
    """Undo a resolved `chain`, giving levels on `y`'s periods.

    `history` is the untransformed series for the periods before (or
    overlapping) `y`: lagged transforms rebuild each level from the one
    `lag` periods earlier, taken from `history` where it has a value. For
    an in-sample round trip it is the first `consumed(chain)` values; for a
    forecast, the observed series.
    """
    y = _series(y)
    specs = _chain(chain)
    _require_resolved(specs)
    if history is None:
        history = pd.Series([], index=pd.PeriodIndex([], freq=y.index.freq), dtype="float64")
    history = _series(history, "the history").rename(y.name)
    if len(history) and history.index.freq != y.index.freq:
        raise TransformError(
            "the history and the series have different frequencies",
            observed=f"history {history.index.freqstr}, series {y.index.freqstr}",
            change="Pass the history at the series' frequency.",
            series=_name(y),
        )
    impls = [get(s.key)() for s in specs]
    levels = [history]
    for impl, spec in zip(impls[:-1], specs[:-1], strict=True):
        levels.append(impl.forward(levels[-1], spec.config).rename(y.name))
    cur = y
    for impl, spec, anchor in reversed(list(zip(impls, specs, levels, strict=True))):
        cur = impl.inverse(cur, spec.config, anchor).rename(y.name)
    return cur


def apply(
    dataset: Dataset, column: str, chain: Sequence[str | TransformSpec] | str | TransformSpec
) -> Dataset:
    """A dataset with `column` transformed by `chain`, the resolved chain appended to its metadata.

    The change is logged. Invert later with
    ``inverse(values, dataset.series[column].transforms, history=levels)``.
    """
    if column not in dataset.series:
        raise TransformError(
            f"no column {column!r} in the dataset",
            observed=f"columns: {', '.join(dataset.columns)}",
            change="Name one of the dataset's columns.",
        )
    if dataset.is_panel:
        raise TransformError(
            "transforms are applied to single-segment datasets",
            observed=f"a panel with segments {list(dataset.segments)}",
            change="Select one segment's rows and transform them.",
            series=column,
        )
    frame = dataset.frame
    y, resolved = forward(frame[column].rename(column), chain)
    frame[column] = y
    text = tuple(str(s) for s in resolved)
    entry = LogEntry(action="transform", reason=f"applied {', '.join(text)}", columns=(column,))
    out = dataset.with_frame(frame, entry=entry)
    return out.with_series(column, transforms=(*dataset.series[column].transforms, *text))
