"""Transforms: differences, logs, growth, logit and Box-Cox, each with its inverse (design §6).

A transform is named in configuration by its text form, ``key`` or
``key(param=value, ...)``: ``"log"``, ``"diff(lag=4)"``,
``"boxcox(lmbda=0.5)"``. A chain is a list of them, applied in order. The
chain applied to a series is recorded in its `SeriesMeta.transforms`, with
every parameter resolved (a seasonal period read from the frequency, a
Box-Cox lambda estimated), so the same text inverts it later.

>>> import pandas as pd
>>> x = pd.Series([1.0, 2.0, 4.0], index=pd.period_range("2020Q1", periods=3, freq="Q"))
>>> y, chain = forward(x, ["log", "diff"])
>>> [str(s) for s in chain]
['log', 'diff(lag=1)']
>>> inverse(y.iloc[1:], chain, history=x.iloc[:1]).tolist()
[2.0, 4.0]
"""

from dazflow.transforms import _builtin  # noqa: F401 - registers the built-ins
from dazflow.transforms._base import (
    Transform,
    TransformSpec,
    apply,
    consumed,
    forward,
    get,
    inverse,
    parse,
    register,
    registry,
)

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
