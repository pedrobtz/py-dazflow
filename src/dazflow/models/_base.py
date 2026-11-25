"""The model protocols, the registry, `Forecast`, the stored specification and refitting.

A model form is a class with a `Config` (its parameters) and
``fit(y, X=None) -> Fitted``, built from its configuration:
``models.get("arimax")(order=(1, 0, 0)).fit(y, X)``. A fitted model is
stored as a `ModelRecord`: specification, parameters, covariance and the
fingerprint of the data it was fitted on, never a pickled object (design
§5.3, D8). `refit` rebuilds a fitted model from a record and the same data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Literal, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import ConfigError, ModelError
from dazflow.core.registry import Registry

__all__ = [
    "DEFAULT_LEVELS",
    "RECORD_FORMAT",
    "Fitted",
    "FittedModel",
    "Forecast",
    "Model",
    "ModelForm",
    "ModelRecord",
    "ModelSpec",
    "Sample",
    "fingerprint",
    "get",
    "refit",
    "register",
    "registry",
    "season_length",
]

DEFAULT_LEVELS: tuple[float, ...] = (0.8, 0.95)
RECORD_FORMAT = 1

T = TypeVar("T")


# --- protocols and registry ----------------------------------------------------


@runtime_checkable
class Fitted(Protocol):
    """What a fitted model provides (design §6)."""

    @property
    def spec(self) -> ModelSpec: ...

    @property
    def params(self) -> pd.Series: ...

    @property
    def cov_params(self) -> pd.DataFrame: ...

    @property
    def residuals(self) -> pd.Series: ...

    def predict(
        self, h: int, X_future: pd.DataFrame | None = None, levels: Sequence[float] = ...
    ) -> Forecast: ...

    def record(self) -> ModelRecord: ...


class Model(Protocol):
    """What a model form provides: its `Config` and ``fit(y, X) -> Fitted``.

    A form is a class built from its configuration, ``Form(config)`` or
    ``Form(**params)``; `ModelForm` is a base class that does this.
    """

    Config: type[Config]

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> Fitted: ...


registry = Registry("model", protocol=Model)
_MODELS = registry  # for functions whose `registry` argument shadows it
get = registry.get


def register(key: str) -> Callable[[T], T]:
    """Decorator: register a model form under `key`, which its specifications name it by."""

    def decorate(obj: T) -> T:
        registry.add(key, obj)
        if getattr(obj, "key", None) is None:
            obj.key = key  # type: ignore[attr-defined]
        return obj

    return decorate


def season_length(index: pd.PeriodIndex) -> int:
    """Periods per year of a monthly (12), quarterly (4) or annual (1) index."""
    code = index.freqstr[0]
    lengths = {"M": 12, "Q": 4, "Y": 1, "A": 1}
    if code not in lengths:
        raise ModelError(
            f"no seasonal period is known for the frequency {index.freqstr}",
            observed=f"frequency {index.freqstr}; known: monthly (12), quarterly (4), annual (1)",
            change="Give the seasonal period in the form's configuration.",
        )
    return lengths[code]


# --- stored objects --------------------------------------------------------------


class ModelSpec(DazflowModel):
    """What a fit needs besides the data: the form, its configuration, the series' names.

    `config` holds the form's `Config` with every default filled in, as
    `model_dump(mode="json")` writes it, so it rebuilds the same form.
    """

    _error: ClassVar = ModelError
    _hint: ClassVar[str] = "Give the form key, its configuration, the target and the drivers."

    form: str
    config: dict[str, Any] = Field(default_factory=dict)
    target: str
    drivers: tuple[str, ...] = ()

    def model(self, registry: Registry | None = None) -> Any:
        """The form built from this configuration, looked up in `registry` (the models')."""
        return (registry if registry is not None else _MODELS).get(self.form)(self.config)


class Sample(DazflowModel):
    """The periods a model was fitted on."""

    _error: ClassVar = ModelError

    start: str
    end: str
    observations: int
    frequency: str


class ModelRecord(DazflowModel):
    """A fitted model as stored: specification, parameters, covariance and data fingerprint.

    The format (``kind: dazflow.model``, ``format: 1``) is fixed in design
    §5.3. `params` keeps the form's order; `cov_params` is the covariance
    matrix in that order, row by row. `data` is the `Dataset` fingerprint of
    the target and drivers fitted on (design §5.1.1), checked by `refit`.
    """

    _error: ClassVar = ModelError
    _hint: ClassVar[str] = "Rebuild the record from a ModelRecord.to_dict() dictionary."

    kind: Literal["dazflow.model"] = "dazflow.model"
    format: Literal[1] = RECORD_FORMAT
    spec: ModelSpec
    sample: Sample
    data: str
    params: dict[str, float]
    cov_params: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def _square(self) -> Self:
        k = len(self.params)
        if len(self.cov_params) != k or any(len(row) != k for row in self.cov_params):
            raise ValueError(f"cov_params must be {k} x {k}, the number of parameters")
        return self

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dictionary; non-finite floats are the strings ``"NaN"``, ``"Infinity"``."""
        return json.loads(self.model_dump_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a record from `to_dict()` output."""
        return cls.model_validate(data)

    def difference(self, fitted: Fitted) -> float:
        """The largest absolute difference between these parameters and `fitted`'s."""
        other = fitted.params
        if list(other.index) != list(self.params):
            raise ModelError(
                "the fitted model has other parameters than the record",
                observed=f"record {list(self.params)}; fitted {list(other.index)}",
                change="Compare a model refitted from this record's specification.",
            )
        a = np.array(list(self.params.values()), dtype=float)
        b = other.to_numpy(dtype=float)
        return float(np.max(np.abs(a - b), initial=0.0))


# --- forecasts -------------------------------------------------------------------


class Forecast(DazflowModel):
    """A forecast path: the mean and the prediction interval at each level, by period.

    `lower[i]` and `upper[i]` are the bounds at `levels[i]`, a central
    interval holding that probability. Periods are written as text
    (``"2024Q1"``) at `frequency`; `frame()` gives the pandas form.
    """

    _error: ClassVar = ModelError

    form: str
    frequency: str
    periods: tuple[str, ...]
    mean: tuple[float, ...]
    levels: tuple[float, ...] = ()
    lower: tuple[tuple[float, ...], ...] = ()
    upper: tuple[tuple[float, ...], ...] = ()

    @model_validator(mode="after")
    def _aligned(self) -> Self:
        h = len(self.periods)
        if len(self.mean) != h:
            raise ValueError(f"mean has {len(self.mean)} values for {h} periods")
        if len(self.lower) != len(self.levels) or len(self.upper) != len(self.levels):
            raise ValueError("lower and upper need one path per level")
        if any(len(p) != h for p in (*self.lower, *self.upper)):
            raise ValueError(f"every interval path needs {h} values")
        return self

    @property
    def h(self) -> int:
        return len(self.periods)

    @property
    def index(self) -> pd.PeriodIndex:
        return pd.PeriodIndex([pd.Period(p, freq=self.frequency) for p in self.periods])

    def interval(self, level: float) -> tuple[pd.Series, pd.Series]:
        """The lower and upper bounds at `level`."""
        i = _level_position(self.levels, level)
        return (
            pd.Series(self.lower[i], index=self.index, name=f"lower_{level:g}"),
            pd.Series(self.upper[i], index=self.index, name=f"upper_{level:g}"),
        )

    def frame(self) -> pd.DataFrame:
        """Columns ``mean``, then ``lower_<level>`` and ``upper_<level>`` per level."""
        out = {"mean": list(self.mean)}
        for level, lo, up in zip(self.levels, self.lower, self.upper, strict=True):
            out[f"lower_{level:g}"] = list(lo)
            out[f"upper_{level:g}"] = list(up)
        return pd.DataFrame(out, index=self.index, dtype=float)


def _level_position(levels: Sequence[float], level: float) -> int:
    for i, lv in enumerate(levels):
        if math.isclose(lv, level, rel_tol=0, abs_tol=1e-12):
            return i
    raise ModelError(
        f"the forecast has no interval at level {level:g}",
        observed=f"levels {', '.join(f'{v:g}' for v in levels) or 'none'}",
        change="Ask predict() for that level.",
    )


def _levels(levels: Iterable[float]) -> tuple[float, ...]:
    out = tuple(float(v) for v in levels)
    if any(not (0 < v < 1) for v in out) or len(set(out)) != len(out):
        raise ModelError(
            "interval levels must be distinct probabilities strictly between 0 and 1",
            observed=f"levels {list(out)!r}",
            change="Give levels such as (0.8, 0.95).",
        )
    return tuple(sorted(out))


# --- the form base class and the fitted base class ---------------------------------


def _bad_data(problem: str, observed: str) -> ModelError:
    return ModelError(problem, observed=observed, change=_DATA_HINT)


_DATA_HINT = (
    "Pass y as a float pd.Series on a PeriodIndex and X as a pd.DataFrame on the same "
    "index, with no missing values."
)


def _check_y(y: Any) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise _bad_data("the target is not a pandas Series", f"a {type(y).__name__}")
    if not isinstance(y.index, pd.PeriodIndex):
        raise _bad_data("the target is not on a PeriodIndex", f"{type(y.index).__name__}")
    if not y.index.is_monotonic_increasing or not y.index.is_unique:
        raise _bad_data("the target's periods are not sorted and unique", f"{list(y.index[:3])}")
    try:
        y = y.astype(float)
    except (TypeError, ValueError):
        raise _bad_data("the target is not numeric", f"dtype {y.dtype}") from None
    y = y.rename("y" if y.name is None else str(y.name))
    _no_missing(y.to_frame(), "target")
    return y


def _no_missing(frame: pd.DataFrame, what: str) -> None:
    bad = ~np.isfinite(frame.to_numpy(dtype=float))
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise _bad_data(
            f"missing or non-finite values in the {what}",
            f"{frame.columns[col]!r} at {frame.index[row]}",
        )


def _check_X(X: Any, index: pd.PeriodIndex, target: str, what: str = "drivers") -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise _bad_data(f"the {what} are not a pandas DataFrame", f"a {type(X).__name__}")
    if not X.index.equals(index):
        raise _bad_data(
            f"the {what} are not on the target's periods",
            f"target {index[0]}..{index[-1]} ({len(index)}); "
            f"{what} {X.index[0] if len(X) else '-'}..{X.index[-1] if len(X) else '-'} ({len(X)})",
        )
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    if target in X.columns:
        raise _bad_data("a driver has the target's name", f"column {target!r}")
    try:
        X = X.astype(float)
    except (TypeError, ValueError):
        raise _bad_data(f"the {what} are not numeric", f"dtypes {dict(X.dtypes)}") from None
    _no_missing(X, what)
    return X


def fingerprint(y: pd.Series, X: pd.DataFrame | None) -> str:
    """The `Dataset` fingerprint of the target and drivers, as a record stores it."""
    frame = y.to_frame() if X is None else pd.concat([y.to_frame(), X], axis=1)
    return Dataset(frame).fingerprint()


class ModelForm:
    """Base class of the built-in forms: holds the validated `Config`, checks the data.

    A subclass sets `Config`, `drivers` (whether it takes driver series) and
    `_fit(y, X, spec) -> FittedModel`.
    """

    Config: ClassVar[type[Config]] = Config
    key: ClassVar[str | None] = None
    drivers: ClassVar[bool] = False

    def __init__(self, config: Config | dict[str, Any] | None = None, /, **params: Any) -> None:
        if config is not None and params:
            raise ConfigError(
                f"{type(self).__name__} was given a config and parameters",
                observed=f"parameters {sorted(params)}",
                change="Pass either a Config or keyword parameters.",
            )
        if isinstance(config, self.Config):
            self.config = config
        else:
            self.config = self.Config.model_validate(params if config is None else config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"

    def spec(self, target: str, drivers: Sequence[str] = ()) -> ModelSpec:
        return ModelSpec(
            form=self.key or type(self).__name__,
            config=self.config.model_dump(mode="json"),
            target=target,
            drivers=tuple(drivers),
        )

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> FittedModel:
        """Fit to `y` (a float series on a `PeriodIndex`) and drivers `X` on the same periods."""
        y = _check_y(y)
        if X is not None and not self.drivers and len(X.columns):
            raise ModelError(
                f"the {self.key} form takes no drivers",
                observed=f"drivers {[str(c) for c in X.columns]}",
                change="Fit it with X=None, or choose a form with drivers (ols, glm, arimax).",
            )
        X = None if X is None or not len(X.columns) else _check_X(X, y.index, str(y.name))
        if len(y) < self.min_observations(X):
            raise _bad_data(
                f"the {self.key} form needs at least {self.min_observations(X)} observations",
                f"{len(y)} observations",
            )
        drivers = () if X is None else tuple(X.columns)
        return self._fit(y, X, self.spec(str(y.name), drivers))

    def min_observations(self, X: pd.DataFrame | None) -> int:
        return 2 + (0 if X is None else len(X.columns))

    def _fit(self, y: pd.Series, X: pd.DataFrame | None, spec: ModelSpec) -> FittedModel:
        raise NotImplementedError


class FittedModel:
    """Base class of the built-in fitted models.

    Holds the specification, the parameters, their covariance, the
    in-sample fitted values and residuals (``y - fitted``, on the target's
    scale) and the data fingerprint. A subclass implements `_predict`.
    """

    def __init__(
        self,
        spec: ModelSpec,
        y: pd.Series,
        X: pd.DataFrame | None,
        *,
        params: pd.Series,
        cov_params: pd.DataFrame,
        fitted_values: pd.Series,
        notes: Sequence[str] = (),
    ) -> None:
        self._spec = spec
        self._y = y
        self._X = X
        self._params = params.astype(float)
        self._cov = cov_params.astype(float)
        self._fitted = fitted_values.astype(float).rename("fitted")
        self._data = fingerprint(y, X)
        self.notes = tuple(notes)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self._spec.form} on {self.sample.start}..{self.sample.end}>"
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def params(self) -> pd.Series:
        return self._params.copy()

    @property
    def cov_params(self) -> pd.DataFrame:
        return self._cov.copy()

    @property
    def fitted_values(self) -> pd.Series:
        return self._fitted.copy()

    @property
    def residuals(self) -> pd.Series:
        y = self._y.reindex(self._fitted.index)
        return (y - self._fitted).rename("residual")

    @property
    def data_fingerprint(self) -> str:
        return self._data

    @property
    def sample(self) -> Sample:
        idx = self._y.index
        return Sample(
            start=str(idx[0]), end=str(idx[-1]), observations=len(idx), frequency=idx.freqstr
        )

    def record(self) -> ModelRecord:
        """The stored form: specification, parameters, covariance and data fingerprint."""
        return ModelRecord(
            spec=self._spec,
            sample=self.sample,
            data=self._data,
            params={str(k): float(v) for k, v in self._params.items()},
            cov_params=tuple(tuple(float(v) for v in row) for row in self._cov.to_numpy()),
        )

    def future_index(self, h: int) -> pd.PeriodIndex:
        last = self._y.index[-1]
        return pd.period_range(last + 1, periods=h, freq=self._y.index.freq)

    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
        levels: Sequence[float] = DEFAULT_LEVELS,
    ) -> Forecast:
        """The forecast `h` periods past the sample, with intervals at `levels`.

        A form with drivers needs `X_future`: their values for the `h`
        periods after the sample, in rows, on that period index or a plain one.
        """
        if not isinstance(h, int | np.integer) or isinstance(h, bool) or h < 1:
            raise ModelError(
                "the horizon must be a positive integer",
                observed=f"h={h!r}",
                change="Ask for h >= 1 periods.",
            )
        h = int(h)
        lv = _levels(levels)
        index = self.future_index(h)
        Xf = self._future_drivers(X_future, index)
        mean, lower, upper = self._predict(h, Xf, lv)
        return Forecast(
            form=self._spec.form,
            frequency=index.freqstr,
            periods=tuple(str(p) for p in index),
            mean=tuple(float(v) for v in mean),
            levels=lv,
            lower=tuple(tuple(float(v) for v in p) for p in lower),
            upper=tuple(tuple(float(v) for v in p) for p in upper),
        )

    def _future_drivers(
        self, X_future: pd.DataFrame | None, index: pd.PeriodIndex
    ) -> pd.DataFrame | None:
        drivers = list(self._spec.drivers)
        if not drivers:
            if X_future is not None and len(X_future.columns):
                raise ModelError(
                    "future drivers were given to a model fitted without drivers",
                    observed=f"columns {[str(c) for c in X_future.columns]}",
                    change="Call predict(h) without X_future.",
                )
            return None
        if X_future is None:
            raise ModelError(
                "the model has drivers but no future values were given",
                observed=f"drivers {drivers}",
                change=f"Pass X_future with {len(index)} rows of {', '.join(drivers)}.",
            )
        if not isinstance(X_future, pd.DataFrame) or len(X_future) != len(index):
            raise ModelError(
                "the future drivers do not cover the horizon",
                observed=f"{len(X_future) if hasattr(X_future, '__len__') else X_future!r} rows "
                f"for h={len(index)}",
                change=f"Pass X_future with one row per forecast period ({len(index)}).",
            )
        if isinstance(X_future.index, pd.PeriodIndex) and not X_future.index.equals(index):
            raise ModelError(
                "the future drivers are not on the forecast periods",
                observed=f"{X_future.index[0]}..{X_future.index[-1]}; forecast {index[0]}..{index[-1]}",
                change="Index X_future by the periods following the sample, or by position.",
            )
        missing = [d for d in drivers if d not in X_future.columns]
        if missing:
            raise ModelError(
                "the future drivers lack a fitted driver",
                observed=f"missing {missing}; given {[str(c) for c in X_future.columns]}",
                change="Give every driver the model was fitted with.",
            )
        out = pd.DataFrame(X_future[drivers].to_numpy(dtype=float), index=index, columns=drivers)
        _no_missing(out, "future drivers")
        return out

    def _predict(
        self, h: int, X_future: pd.DataFrame | None, levels: tuple[float, ...]
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        raise NotImplementedError


# --- refit -------------------------------------------------------------------------


def refit(
    record: ModelRecord | dict[str, Any],
    y: pd.Series,
    X: pd.DataFrame | None = None,
    *,
    tolerance: float | None = None,
    registry: Registry | None = None,
) -> Fitted:
    """Fit a model again from its stored specification, on the data it was fitted on.

    The data must have the record's fingerprint (`ModelError` otherwise).
    With `tolerance`, the refitted parameters must also be within it of the
    stored ones, in absolute value. `registry` is where the form is looked
    up, `dazflow.models.registry` by default.
    """
    if not isinstance(record, ModelRecord):
        record = ModelRecord.from_dict(record)
    spec = record.spec
    yy = _check_y(y)
    if str(yy.name) != spec.target:
        yy = yy.rename(spec.target)
    XX = None
    if spec.drivers:
        if X is None:
            raise _bad_data("the record has drivers but none were given", f"drivers {spec.drivers}")
        missing = [d for d in spec.drivers if d not in X.columns]
        if missing:
            raise _bad_data("the drivers lack one the record names", f"missing {missing}")
        XX = _check_X(X[list(spec.drivers)], yy.index, spec.target)
    found = fingerprint(yy, XX)
    if found != record.data:
        raise ModelError(
            "the data are not those the model was fitted on",
            observed=f"fingerprint {found[:12]}..., record {record.data[:12]}...; "
            f"periods {yy.index[0]}..{yy.index[-1]}, record {record.sample.start}..{record.sample.end}",
            change="Refit on the dataset the record names (same periods, values and drivers).",
        )
    fitted = spec.model(registry).fit(yy, XX)
    if tolerance is not None:
        diff = record.difference(fitted)
        if not diff <= tolerance:
            raise ModelError(
                "the refitted parameters differ from the record",
                observed=f"largest absolute difference {diff:.3g}; tolerance {tolerance:g}",
                change="Check the package versions against the record's environment.",
            )
    return fitted


def _round_start(values: Any) -> np.ndarray:
    """Start values rounded to 10 significant digits (D36).

    statsmodels computes start values by least squares, whose last bit can
    depend on memory alignment; an optimiser started a bit apart stops up to
    1e-9 apart. Rounding makes the start, and so the fit, repeatable.
    """
    return np.array([float(f"{v:.10g}") for v in np.asarray(values, dtype=float)])
