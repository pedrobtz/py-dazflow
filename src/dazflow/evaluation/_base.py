"""Splitters and metrics: the protocols, the registries and the built-ins (design §6)."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from pydantic import Field

from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.errors import ConfigError, EvaluationError
from dazflow.core.registry import Registry

__all__ = [
    "MAE",
    "MASE",
    "RMSE",
    "Coverage",
    "Expanding",
    "Fold",
    "Metric",
    "Rolling",
    "Splitter",
    "metrics",
    "splitters",
]


class _Component:
    """Holds the validated `Config`: ``Component(config)`` or ``Component(**params)``."""

    Config: ClassVar[type[Config]] = Config
    key: ClassVar[str | None] = None

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


def _keyed(registry: Registry, key: str):
    def decorate(cls):
        registry.add(key, cls)
        cls.key = key
        return cls

    return decorate


# --- splitters ---------------------------------------------------------------------


class Fold(DazflowModel):
    """One origin: the training and test positions, each ``(start, stop)``, stop excluded.

    `origin` is the last training position, the period a forecast is made from.
    """

    _error: ClassVar = EvaluationError

    number: int
    train: tuple[int, int]
    test: tuple[int, int]

    @property
    def origin(self) -> int:
        return self.train[1] - 1

    @property
    def horizon(self) -> int:
        return self.test[1] - self.test[0]


class Splitter(Protocol):
    """What a splitter provides: its `Config` and ``split(index) -> folds``."""

    Config: type[Config]

    def split(self, index: pd.PeriodIndex) -> tuple[Fold, ...]: ...


splitters = Registry("splitter", protocol=Splitter)


class _FoldConfig(Config):
    horizon: int = Field(1, ge=1, description="Test periods after each origin.")
    step: int = Field(1, ge=1, description="Periods between successive origins.")
    folds: int | None = Field(None, ge=1, description="Keep only the last n origins; unset: all.")


def _folds(index: Any, first: int, config: _FoldConfig, start: Any, what: str) -> tuple[Fold, ...]:
    if not isinstance(index, pd.PeriodIndex):
        raise EvaluationError(
            "a splitter needs a PeriodIndex",
            observed=f"{type(index).__name__}",
            change="Split the periods of the series (series.index).",
        )
    n, h = len(index), config.horizon
    origins = list(range(first, n - h + 1, config.step))
    if config.folds is not None:
        origins = origins[-config.folds :]
    if not origins:
        raise EvaluationError(
            f"the {what} splitter leaves no fold",
            observed=f"{n} periods; first test period at position {first}, horizon {h}",
            change="Use a shorter training window or horizon, or a longer series.",
        )
    return tuple(
        Fold(number=i + 1, train=(start(t), t), test=(t, t + h)) for i, t in enumerate(origins)
    )


class ExpandingConfig(_FoldConfig):
    initial: int | None = Field(
        None, ge=1, description="Training periods at the first origin; unset: half the series."
    )


@_keyed(splitters, "expanding")
class Expanding(_Component):
    """Origins that train on every period up to them, from `initial` periods on.

    Every fold tests `horizon` periods; the last test period is the last
    period of the series, and origins are `step` apart.
    """

    Config = ExpandingConfig

    def split(self, index: pd.PeriodIndex) -> tuple[Fold, ...]:
        first = self.config.initial if self.config.initial is not None else len(index) // 2
        return _folds(index, max(first, 1), self.config, lambda t: 0, "expanding")


class RollingConfig(_FoldConfig):
    window: int = Field(..., ge=1, description="Training periods at every origin.")


@_keyed(splitters, "rolling")
class Rolling(_Component):
    """Origins that train on the `window` periods before them."""

    Config = RollingConfig

    def split(self, index: pd.PeriodIndex) -> tuple[Fold, ...]:
        w = self.config.window
        return _folds(index, w, self.config, lambda t: t - w, "rolling")


# --- metrics -----------------------------------------------------------------------


class Metric(Protocol):
    """What a metric provides: its `Config` and ``score(actual, forecast, ...) -> float``.

    Keywords a metric does not use are ignored: `scale` (one value, or one
    per point) or `train` for scaled errors, `lower` and `upper` for interval
    metrics.
    """

    Config: type[Config]

    def score(self, actual: Any, forecast: Any, **kwargs: Any) -> float: ...


metrics = Registry("metric", protocol=Metric)


def _arrays(actual: Any, forecast: Any, **others: Any) -> list[np.ndarray]:
    arrays = [np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)]
    arrays += [np.asarray(v, dtype=float) for v in others.values()]
    lengths = {a.shape for a in arrays}
    if len(lengths) != 1 or arrays[0].ndim != 1:
        raise EvaluationError(
            "the values to score have different lengths",
            observed=f"shapes {[a.shape for a in arrays]}",
            change="Pass one actual, forecast (and bound) value per period.",
        )
    if not len(arrays[0]):
        raise EvaluationError(
            "there is no value to score",
            observed="0 periods",
            change="Score at least one period.",
        )
    if not all(np.isfinite(a).all() for a in arrays):
        raise EvaluationError(
            "a value to score is missing or not finite",
            observed=f"{sum(int((~np.isfinite(a)).sum()) for a in arrays)} non-finite values",
            change="Score only periods with an actual and a forecast value.",
        )
    return arrays


class _NoParams(Config):
    """No parameters."""


@_keyed(metrics, "mae")
class MAE(_Component):
    """Mean absolute error, ``mean |actual - forecast|``."""

    Config = _NoParams

    def score(self, actual: Any, forecast: Any, **kwargs: Any) -> float:
        a, f = _arrays(actual, forecast)
        return float(np.mean(np.abs(a - f)))


@_keyed(metrics, "rmse")
class RMSE(_Component):
    """Root mean squared error, ``sqrt(mean (actual - forecast)^2)``."""

    Config = _NoParams

    def score(self, actual: Any, forecast: Any, **kwargs: Any) -> float:
        a, f = _arrays(actual, forecast)
        return float(np.sqrt(np.mean((a - f) ** 2)))


class MASEConfig(Config):
    season: int = Field(
        1, ge=1, description="Lag of the naive forecast that scales the errors (1: naive)."
    )


@_keyed(metrics, "mase")
class MASE(_Component):
    """Mean absolute scaled error (Hyndman and Koehler, 2006).

    Each error is divided by the in-sample mean absolute error of the
    (seasonal) naive forecast on the training series,
    ``mean |train[t] - train[t - season]|``. `scale` gives that value, one
    for all points or one per point (each fold its own); `train` computes it.
    """

    Config = MASEConfig

    def scale(self, train: Any) -> float:
        x = np.asarray(train, dtype=float)
        m = self.config.season
        if x.ndim != 1 or len(x) <= m or not np.isfinite(x).all():
            raise EvaluationError(
                "the MASE scale needs a training series longer than its season",
                observed=f"{x.size} values, season {m}",
                change="Pass the complete training series.",
            )
        value = float(np.mean(np.abs(x[m:] - x[:-m])))
        if value == 0:
            raise EvaluationError(
                "the MASE scale is zero: the training series does not change",
                observed=f"{len(x)} values, season {m}",
                change="Use another metric for a constant series.",
            )
        return value

    def score(
        self, actual: Any, forecast: Any, *, scale: Any = None, train: Any = None, **kwargs: Any
    ) -> float:
        if scale is None:
            if train is None:
                raise EvaluationError(
                    "MASE needs a scale or the training series",
                    observed="neither scale nor train given",
                    change="Pass scale=... or train=....",
                )
            scale = self.scale(train)
        a, f = _arrays(actual, forecast)
        s = np.broadcast_to(np.asarray(scale, dtype=float), a.shape)
        if not (np.isfinite(s).all() and (s > 0).all()):
            raise EvaluationError(
                "the MASE scale must be positive",
                observed=f"scale {np.unique(s)[:3].tolist()}",
                change="Scale by a training series that changes.",
            )
        return float(np.mean(np.abs(a - f) / s))


class CoverageConfig(Config):
    level: float = Field(
        0.95, gt=0, lt=1, description="The interval whose coverage is scored, as a probability."
    )


@_keyed(metrics, "coverage")
class Coverage(_Component):
    """Share of actual values within the interval, bounds included.

    Compare it with the interval's `level`: a well-calibrated 95% interval
    covers about 0.95 of the actual values.
    """

    Config = CoverageConfig

    def score(
        self, actual: Any, forecast: Any, *, lower: Any = None, upper: Any = None, **kwargs: Any
    ) -> float:
        if lower is None or upper is None:
            raise EvaluationError(
                "coverage needs the interval bounds",
                observed=f"lower {'given' if lower is not None else 'missing'}, "
                f"upper {'given' if upper is not None else 'missing'}",
                change="Pass lower=... and upper=....",
            )
        a, _, lo, up = _arrays(actual, forecast, lower=lower, upper=upper)
        return float(np.mean((lo <= a) & (a <= up)))
