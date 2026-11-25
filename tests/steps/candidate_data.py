"""Data shared by the build and select tests: a target in levels and drivers with known effects."""

import numpy as np
import pandas as pd
import pytest

import dazflow as dz


def levels(n=100, seed=0, start="1995Q1"):
    """A stationary target driven by `a` (+0.8) and `b` (-0.5), with `c` unrelated.

    The series are already stationary, so build needs no transform; signs
    are declared in the metadata, as ingest records them.
    """
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, periods=n, freq="Q")
    X = rng.normal(size=(n, 3))
    y = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1], "c": X[:, 2]}, index=index)
    meta = {
        "y": {"role": "target", "unit": "%"},
        "a": {"role": "driver", "expected_sign": 1},
        "b": {"role": "driver", "expected_sign": -1},
        "c": {"role": "driver", "expected_sign": 1},
    }
    return dz.Dataset(frame, meta)


def scenarios(dataset, h=8, shifts=(0.0, 1.0, 2.0)):
    """Driver paths after the last period, one per shift: `a` rises, `b` falls with the shift.

    Labels are ``s0``, ``s1``, ... in the order given, so with rising
    shifts the target's projection rises with the label: mildest first.
    """
    last = dataset.frame.index[-1]
    periods = pd.period_range(last + 1, periods=h, freq=last.freq)
    parts = []
    for i, s in enumerate(shifts):
        part = pd.DataFrame(
            {"a": np.full(h, 0.1 + s), "b": np.full(h, -0.1 - s), "c": np.zeros(h)},
            index=pd.MultiIndex.from_product([[f"s{i}"], periods], names=["scenario", "period"]),
        )
        parts.append(part)
    return dataset.with_scenarios(pd.concat(parts), [f"s{i}" for i in range(len(shifts))])


def close(a, b, path="result"):
    """Equal, floats to 1e-12 relative: statsmodels' intervals move by an ulp between runs."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-14), path
    else:
        assert a == b, path
