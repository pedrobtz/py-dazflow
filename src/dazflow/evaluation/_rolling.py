"""Rolling-origin evaluation: refit at successive origins and score the h-step forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from dazflow.core._model import DazflowModel
from dazflow.core.errors import EvaluationError
from dazflow.evaluation._base import Expanding, Fold, splitters
from dazflow.evaluation._base import metrics as _metrics
from dazflow.models._base import _MODELS as _models
from dazflow.models._base import DEFAULT_LEVELS, ModelSpec, _check_X, _check_y, _levels

__all__ = ["DEFAULT_METRICS", "Evaluation", "EvaluationRow", "Score", "evaluate"]

DEFAULT_METRICS: tuple[str, ...] = ("mae", "rmse", "mase", "coverage")


class EvaluationRow(DazflowModel):
    """One forecast value: its fold, origin, period and horizon, the actual and the forecast.

    `lower[i]` and `upper[i]` are the bounds at the evaluation's `levels[i]`.
    """

    _error: ClassVar = EvaluationError

    fold: int
    origin: str
    period: str
    horizon: int
    actual: float
    forecast: float
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()


class Score(DazflowModel):
    """A metric's value over every forecast (`horizon` None) or over one horizon."""

    _error: ClassVar = EvaluationError

    metric: str
    config: dict[str, Any]
    horizon: int | None
    value: float


class Evaluation(DazflowModel):
    """What `evaluate` returns: the folds, every forecast value and the scores.

    `scores` holds each metric over all rows first (``horizon: null``), then
    by horizon 1, 2, ... Every value is data; `frame()` gives the rows as a
    pandas frame.
    """

    _error: ClassVar = EvaluationError

    spec: ModelSpec
    splitter: str
    splitter_config: dict[str, Any]
    levels: tuple[float, ...]
    folds: tuple[Fold, ...]
    rows: tuple[EvaluationRow, ...]
    scores: tuple[Score, ...]

    def score(self, metric: str, horizon: int | None = None) -> float:
        """The first score of `metric` (a key), over all rows or at one `horizon`."""
        for s in self.scores:
            if s.metric == metric and s.horizon == horizon:
                return s.value
        raise EvaluationError(
            f"the evaluation has no {metric} score"
            + (f" at horizon {horizon}" if horizon is not None else ""),
            observed=f"metrics {sorted({s.metric for s in self.scores})}",
            change="Evaluate with that metric, or ask for a horizon that was forecast.",
        )

    def frame(self) -> pd.DataFrame:
        """One row per forecast value, with ``lower_<level>`` and ``upper_<level>`` columns."""
        records = []
        for r in self.rows:
            rec = r.model_dump(exclude={"lower", "upper"})
            for lv, lo, up in zip(self.levels, r.lower, r.upper, strict=True):
                rec[f"lower_{lv:g}"] = lo
                rec[f"upper_{lv:g}"] = up
            records.append(rec)
        return pd.DataFrame.from_records(records)


def _key(component: Any) -> str:
    return getattr(component, "key", None) or type(component).__name__


def _config(component: Any) -> dict[str, Any]:
    config = getattr(component, "config", None)
    return {} if config is None else config.model_dump(mode="json")


def _build(registry: Any, item: Any) -> Any:
    return registry.get(item)() if isinstance(item, str) else item


def evaluate(
    model: Any,
    y: pd.Series,
    X: pd.DataFrame | None = None,
    *,
    splitter: Any = None,
    metrics: Sequence[Any] = DEFAULT_METRICS,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> Evaluation:
    """Refit `model` at every origin of `splitter` and score its forecasts.

    Parameters
    ----------
    model
        A model form (``models.get("ols")()``), its key, or a `ModelSpec`.
    y, X
        The target and the drivers, as `fit` takes them. The forecast at
        each origin uses the drivers' actual values over the test periods.
    splitter
        A splitter, or its key; default ``expanding`` with one-step folds.
    metrics
        Metrics or their keys. MASE divides each fold's errors by that
        fold's training scale; coverage scores the interval at its `level`,
        which must be among `levels`.
    levels
        The interval levels forecast at every origin.
    """
    if isinstance(model, ModelSpec):
        model = model.model()
    elif isinstance(model, str):
        model = _models.get(model)()
    y = _check_y(y)
    X = _check_X(X, y.index, str(y.name)) if X is not None and len(X.columns) else None
    split = Expanding() if splitter is None else _build(splitters, splitter)
    scorers = [_build(_metrics, m) for m in metrics]
    lv = _levels(levels)
    for m in scorers:
        level = getattr(m.config, "level", None)
        if level is not None and not any(abs(level - v) < 1e-12 for v in lv):
            raise EvaluationError(
                f"the {_key(m)} metric scores the {level:g} interval, which is not forecast",
                observed=f"levels {list(lv)}",
                change=f"Add {level:g} to levels, or set the metric's level to one of them.",
            )
    folds = split.split(y.index)

    spec: ModelSpec | None = None
    rows: list[EvaluationRow] = []
    scales: dict[int, list[float]] = {i: [] for i, m in enumerate(scorers) if hasattr(m, "scale")}
    for fold in folds:
        tr = slice(*fold.train)
        te = slice(*fold.test)
        fitted = model.fit(y.iloc[tr], None if X is None else X.iloc[tr])
        spec = spec or fitted.spec
        fc = fitted.predict(fold.horizon, None if X is None else X.iloc[te], levels=lv)
        actual = y.iloc[te].to_numpy()
        for k in range(fold.horizon):
            rows.append(
                EvaluationRow(
                    fold=fold.number,
                    origin=str(y.index[fold.origin]),
                    period=fc.periods[k],
                    horizon=k + 1,
                    actual=float(actual[k]),
                    forecast=fc.mean[k],
                    lower=tuple(p[k] for p in fc.lower),
                    upper=tuple(p[k] for p in fc.upper),
                )
            )
        for i in scales:
            scales[i] += [scorers[i].scale(y.iloc[tr].to_numpy())] * fold.horizon

    scores: list[Score] = []
    horizons = sorted({r.horizon for r in rows})
    for i, m in enumerate(scorers):
        for h in (None, *horizons):
            keep = [j for j, r in enumerate(rows) if h is None or r.horizon == h]
            kwargs: dict[str, Any] = {}
            if i in scales:
                kwargs["scale"] = np.asarray(scales[i])[keep]
            level = getattr(m.config, "level", None)
            if level is not None:
                pos = next(p for p, v in enumerate(lv) if abs(level - v) < 1e-12)
                kwargs["lower"] = [rows[j].lower[pos] for j in keep]
                kwargs["upper"] = [rows[j].upper[pos] for j in keep]
            value = m.score(
                [rows[j].actual for j in keep], [rows[j].forecast for j in keep], **kwargs
            )
            scores.append(Score(metric=_key(m), config=_config(m), horizon=h, value=value))

    return Evaluation(
        spec=spec,
        splitter=_key(split),
        splitter_config=_config(split),
        levels=lv,
        folds=folds,
        rows=tuple(rows),
        scores=tuple(scores),
    )
