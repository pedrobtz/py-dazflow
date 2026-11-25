"""Refit: a model rebuilt from its stored specification gives the same parameters to 1e-10.

The record goes through JSON text on the way, as the run record stores it.
Repeated refits run between allocations of different sizes, because
statsmodels' least-squares start values can differ in the last bit with
memory alignment (D36).
"""

import json

import numpy as np
import pytest

from dazflow import ModelError, models

TOLERANCE = 1e-10

FORMS = [
    ("naive", {}),
    ("seasonal_naive", {}),
    ("mean", {}),
    ("ols", {}),
    ("ols", {"cov_type": "HAC"}),
    ("glm", {}),
    ("glm", {"family": "gaussian", "cov_type": "HC1"}),
    ("arimax", {"order": (1, 0, 0)}),
    ("arimax", {"order": (1, 1, 1), "trend": "n"}),
    ("arimax", {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 4)}),
    ("ets", {}),
    ("ets", {"trend": "add", "damped_trend": True, "seasonal": "add"}),
]


def fit(key, config, y, X):
    form = models.get(key)(**config)
    return form.fit(y, X if form.drivers else None)


def stored(fitted):
    return json.loads(json.dumps(fitted.record().to_dict(), allow_nan=False))


@pytest.mark.parametrize(("key", "config"), FORMS)
def test_refit_from_the_stored_record_gives_the_same_parameters(data, key, config):
    y, X = data
    fitted = fit(key, config, y, X)
    text = stored(fitted)

    junk = []
    for size in (1, 7, 129, 1023):
        junk.append(np.empty(size))  # move later allocations to other alignments
        again = models.refit(text, y, X)
        assert list(again.params.index) == list(fitted.params.index)
        np.testing.assert_allclose(again.params, fitted.params, rtol=0, atol=TOLERANCE)
        assert models.ModelRecord.from_dict(text).difference(again) <= TOLERANCE
        assert again.spec == fitted.spec


@pytest.mark.parametrize(("key", "config"), FORMS)
def test_the_record_holds_spec_params_covariance_and_fingerprint(data, key, config):
    y, X = data
    fitted = fit(key, config, y, X)
    text = stored(fitted)

    assert list(text) == ["kind", "format", "spec", "sample", "data", "params", "cov_params"]
    assert (text["kind"], text["format"]) == ("dazflow.model", 1)
    assert text["spec"]["form"] == key
    assert text["spec"]["config"] == models.get(key).Config(**config).model_dump(mode="json")
    assert text["sample"] == {
        "start": "2000Q1",
        "end": "2019Q4",
        "observations": 80,
        "frequency": "Q-DEC",
    }
    assert text["data"] == models.fingerprint(y, X if fitted.spec.drivers else None)
    assert list(text["params"]) == list(fitted.params.index)
    k = len(text["params"])
    assert len(text["cov_params"]) == k and all(len(r) == k for r in text["cov_params"])


def test_refit_refuses_other_data(data):
    y, X = data
    text = stored(models.get("ols")().fit(y, X))
    changed = y.copy()
    changed.iloc[5] += 1e-9

    with pytest.raises(ModelError, match="not those the model was fitted on"):
        models.refit(text, changed, X)
    with pytest.raises(ModelError, match="not those the model was fitted on"):
        models.refit(text, y.iloc[1:], X.iloc[1:])
    with pytest.raises(ModelError, match="drivers"):
        models.refit(text, y)


def test_refit_with_a_tolerance_refuses_parameters_that_moved(data):
    y, X = data
    text = stored(models.get("ols")().fit(y, X))
    text["params"]["gdp"] += 2e-10

    assert models.refit(text, y, X, tolerance=3e-10)
    with pytest.raises(ModelError, match="differ from the record"):
        models.refit(text, y, X, tolerance=1e-10)


def test_a_record_of_another_format_is_refused(data):
    y, _X = data
    text = stored(models.get("mean")().fit(y))
    with pytest.raises(ModelError):
        models.ModelRecord.from_dict({**text, "format": 2})
    with pytest.raises(ModelError):
        models.ModelRecord.from_dict({**text, "cov_params": [[1.0, 2.0]]})
