"""Data-quality checks, evaluated by pointblank (design §7.2, D12, D29).

A check is a class with a `Config` (a `CheckConfig`: its thresholds and
parameters) and `units(x, config, context) -> Units`, the periods it tests
on one series and what each must satisfy. `evaluate` runs it through a
pointblank validation and returns an `Outcome` holding the gate `Check`.

>>> import pandas as pd
>>> from dazflow import checks
>>> x = pd.Series([1.0, None, 3.0], index=pd.period_range("2024Q1", periods=3, freq="Q"), name="y")
>>> out = checks.evaluate("gaps", x)
>>> out.flagged, out.check.passed
(('2024Q2',), False)
"""

from dazflow.checks import _builtin
from dazflow.checks._base import (
    CheckConfig,
    Context,
    Level,
    Outcome,
    QualityCheck,
    Test,
    Units,
    evaluate,
    get,
    register,
    registry,
)
from dazflow.checks._builtin import Downturn, outlier_scores

__all__ = [
    "CheckConfig",
    "Context",
    "Downturn",
    "Level",
    "Outcome",
    "QualityCheck",
    "Test",
    "Units",
    "evaluate",
    "get",
    "outlier_scores",
    "register",
    "registry",
]

del _builtin
