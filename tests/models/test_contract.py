"""The model contract (`dazflow.testing.ModelContract`): every built-in form passes it,
and forms broken in one way each fail the test that guards that promise."""

import numpy as np
import pandas as pd
import pytest

from dazflow import Registry, models
from dazflow.testing import ComponentContract, ModelContract

BUILT_INS = [
    ("naive", {}),
    ("seasonal_naive", {}),
    ("mean", {}),
    ("ols", {}),
    ("glm", {}),
    ("glm", {"family": "gaussian"}),
    ("arimax", {}),
    ("arimax", {"order": (0, 1, 1), "trend": "n"}),
    ("ets", {}),
    ("ets", {"trend": "add", "seasonal": "add"}),
]


def contract_tests():
    return sorted(n for n in dir(ModelContract) if n.startswith("test_"))


def run_contract(key, config=None, registry=models.registry):
    probe = type(
        "Probe", (ModelContract,), {"registry": registry, "key": key, "config": config or {}}
    )()
    failures = {}
    for name in contract_tests():
        try:
            getattr(probe, name)()
        except AssertionError as err:
            failures[name] = str(err)
    return failures


def test_the_model_contract_extends_the_component_contract():
    assert issubclass(ModelContract, ComponentContract)
    assert set(contract_tests()) > {n for n in dir(ComponentContract) if n.startswith("test_")}


def test_every_built_in_form_is_in_the_contract_run():
    assert {k for k, _ in BUILT_INS} == set(models.registry.keys())


@pytest.mark.parametrize(("key", "config"), BUILT_INS)
def test_built_in_form_passes_the_model_contract(key, config):
    assert run_contract(key, config) == {}


class TestOLS(ModelContract):
    """How a plugin author uses the suite: subclass it and name the key."""

    key = "ols"


# --- deliberately broken forms -----------------------------------------------------

broken = Registry("model", protocol=models.Model)


class _Fitted(models.FittedModel):
    def _predict(self, h, X_future, levels):
        mean = np.full(h, self._params["const"])
        return mean, [mean - lv for lv in levels], [mean + lv for lv in levels]


class _Mean(models.ModelForm):
    Config = models.get("mean").Config

    def _fit(self, y, X, spec):
        c = float(y.mean())
        return _Fitted(
            spec,
            y,
            X,
            params=pd.Series({"const": c}),
            cov_params=pd.DataFrame([[1.0]], index=["const"], columns=["const"]),
            fitted_values=pd.Series(c, index=y.index),
        )


def register(key):
    def decorate(cls):
        broken.add(key, cls)
        cls.key = key
        return cls

    return decorate


@register("good")
class Good(_Mean):
    pass


@register("short")
class ShortForecast(_Mean):
    """Forecasts one period too few."""

    def _fit(self, y, X, spec):
        fitted = super()._fit(y, X, spec)
        predict = fitted.predict
        fitted.predict = lambda h, X_future=None, levels=(0.8, 0.95): predict(
            h - 1, X_future, levels
        )
        return fitted


@register("wobbly")
class Wobbly(_Mean):
    """Fits a different constant every time: a refit cannot repeat it."""

    calls = 0

    def _fit(self, y, X, spec):
        type(self).calls += 1
        fitted = super()._fit(y, X, spec)
        fitted._params = fitted._params + 1e-6 * type(self).calls
        return fitted


@register("inverted")
class Inverted(_Mean):
    """Its wider level gives the narrower interval."""

    def _fit(self, y, X, spec):
        fitted = super()._fit(y, X, spec)
        fitted._predict = lambda h, Xf, levels: (
            np.zeros(h),
            [np.full(h, -1 / lv) for lv in levels],
            [np.full(h, 1 / lv) for lv in levels],
        )
        return fitted


@register("offcentre")
class OffCentre(_Mean):
    """Its interval lies above its mean."""

    def _fit(self, y, X, spec):
        fitted = super()._fit(y, X, spec)
        fitted._predict = lambda h, Xf, levels: (
            np.zeros(h),
            [np.full(h, 1.0 - lv) for lv in levels],
            [np.full(h, 1.0 + lv) for lv in levels],
        )
        return fitted


@register("greedy")
class Greedy(_Mean):
    """Accepts drivers it does not use, and changes the caller's data."""

    def fit(self, y, X=None):
        y.iloc[0] = 0.0
        return self._fit(y.copy(), None, self.spec(str(y.name)))


@register("misnamed")
class Misnamed(_Mean):
    """Its specification names another form."""

    def _fit(self, y, X, spec):
        return super()._fit(y, X, spec.model_copy(update={"form": "mean"}))


@pytest.mark.parametrize(
    ("key", "fails"),
    [
        ("short", {"test_predict_covers_the_horizon_with_nested_intervals"}),
        ("wobbly", {"test_refit_from_the_stored_record_repeats_the_fit"}),
        ("inverted", {"test_predict_covers_the_horizon_with_nested_intervals"}),
        ("offcentre", {"test_predict_covers_the_horizon_with_nested_intervals"}),
        ("greedy", {"test_fit_leaves_its_inputs_unchanged", "test_drivers_are_used_or_refused"}),
        (
            "misnamed",
            {
                "test_fit_returns_a_fitted_model",
                "test_refit_from_the_stored_record_repeats_the_fit",
            },
        ),
        ("missing", {"test_key_is_registered"}),
    ],
)
def test_a_broken_form_fails_the_contract(key, fails):
    failures = run_contract(key, registry=broken)
    if key == "missing":
        assert fails <= set(failures)
    else:
        assert set(failures) == fails, failures


def test_the_working_form_beside_them_passes():
    assert run_contract("good", registry=broken) == {}
