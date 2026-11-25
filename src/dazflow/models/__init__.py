"""Model forms and benchmarks behind one interface (design §6).

A form is registered by key and built from its configuration; fitting it
returns a `Fitted` model, whose `predict` returns a `Forecast`::

    >>> import numpy as np, pandas as pd
    >>> y = pd.Series([1.0, 2.0, 4.0, 3.0], index=pd.period_range("2020Q1", periods=4, freq="Q"))
    >>> fitted = get("naive")().fit(y)
    >>> fitted.predict(2).mean
    (3.0, 3.0)

Built-in keys: ``naive``, ``seasonal_naive``, ``mean``, ``ols``, ``glm``,
``arimax`` and ``ets``. A fitted model is stored as its `record()`
(specification, parameters, covariance and the data fingerprint) and is
rebuilt by `refit(record, y, X)`; it is never pickled (design §5.3).
"""

from dazflow.models import _builtin  # noqa: F401 - registers the built-ins
from dazflow.models._base import (
    DEFAULT_LEVELS,
    RECORD_FORMAT,
    Fitted,
    FittedModel,
    Forecast,
    Model,
    ModelForm,
    ModelRecord,
    ModelSpec,
    Sample,
    fingerprint,
    get,
    refit,
    register,
    registry,
    season_length,
)
from dazflow.models._builtin import newey_west_lags

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
    "newey_west_lags",
    "refit",
    "register",
    "registry",
    "season_length",
]
