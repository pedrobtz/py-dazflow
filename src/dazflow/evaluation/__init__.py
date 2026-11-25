"""Splitters, metrics and rolling-origin evaluation, shared by select and segment (design §6, §7.6).

>>> import pandas as pd
>>> folds = splitters.get("expanding")(initial=4, horizon=2).split(
...     pd.period_range("2020Q1", periods=8, freq="Q"))
>>> [(f.train, f.test) for f in folds]
[((0, 4), (4, 6)), ((0, 5), (5, 7)), ((0, 6), (6, 8))]
>>> metrics.get("mase")().score([5, 6], [4, 6], train=[2, 4, 3])
0.3333333333333333

Splitter keys: ``expanding``, ``rolling``. Metric keys: ``mae``, ``rmse``,
``mase``, ``coverage``. `evaluate(model, y, X, splitter=..., metrics=...)`
refits a model form at every origin and returns an `Evaluation`.
"""

from dazflow.evaluation._base import (
    MAE,
    MASE,
    RMSE,
    Coverage,
    Expanding,
    Fold,
    Metric,
    Rolling,
    Splitter,
    metrics,
    splitters,
)
from dazflow.evaluation._rolling import (
    DEFAULT_METRICS,
    Evaluation,
    EvaluationRow,
    Score,
    evaluate,
)

__all__ = [
    "DEFAULT_METRICS",
    "MAE",
    "MASE",
    "RMSE",
    "Coverage",
    "Evaluation",
    "EvaluationRow",
    "Expanding",
    "Fold",
    "Metric",
    "Rolling",
    "Score",
    "Splitter",
    "evaluate",
    "metrics",
    "splitters",
]
