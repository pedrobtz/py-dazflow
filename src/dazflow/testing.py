"""The public contract suite every component passes (design §13).

Plugin authors subclass `ComponentContract` in their own tests, naming the
registry and the key; pytest then runs the contract tests on that component::

    from dazflow.testing import ComponentContract
    import dazflow_theta

    class TestTheta(ComponentContract):
        registry = dazflow_theta.models
        key = "theta"

The suite grows with the component kinds: each kind's contract subclasses
`ComponentContract` and adds the tests its protocol promises. `ModelContract`
is the one for model forms::

    from dazflow.testing import ModelContract

    class TestTheta(ModelContract):
        key = "theta"                  # in dazflow.models.registry by default
        config = {"alpha": 0.5}        # optional: the configuration under test
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from dazflow import models
from dazflow.core.config import Config
from dazflow.core.errors import DazflowError, ModelError
from dazflow.core.registry import Registry, _protocol_members

__all__ = ["ComponentContract", "ModelContract", "sample_series"]


class ComponentContract:
    """Tests every registered component passes, whatever its kind.

    Set `registry` and `key` in a subclass whose name starts with ``Test``.
    """

    registry: ClassVar[Registry]
    key: ClassVar[str]

    def component(self) -> Any:
        """The component under test, looked up the way configuration looks it up."""
        return self.registry.get(self.key)

    def test_key_is_registered(self) -> None:
        keys = self.registry.keys()
        assert self.key in keys, (
            f"{self.key!r} is not registered; known keys: {', '.join(keys) or 'none'}"
        )

    def test_lookup_returns_the_registered_component(self) -> None:
        assert self.key in self.registry, f"{self.key!r} is not registered"
        assert self.registry.get(self.key) is self.registry[self.key]

    def test_component_follows_the_protocol(self) -> None:
        protocol = self.registry.protocol
        if protocol is None:
            return
        assert self.key in self.registry, f"{self.key!r} is not registered"
        obj = self.component()
        missing = [m for m in _protocol_members(protocol) if not hasattr(obj, m)]
        assert not missing, f"{obj!r} lacks {protocol.__name__} members: {', '.join(missing)}"

    def test_config_is_a_dazflow_config(self) -> None:
        assert self.key in self.registry, f"{self.key!r} is not registered"
        config = getattr(self.component(), "Config", None)
        if config is None:
            return
        assert isinstance(config, type) and issubclass(config, Config), (
            f"{self.key!r}.Config is {config!r}, not a dazflow.Config subclass"
        )
        schema = config.json_schema()
        assert schema.get("type") == "object", (
            f"{self.key!r}.Config does not export an object schema"
        )


def sample_series(n: int = 60, seed: int = 20261125) -> tuple[pd.Series, pd.DataFrame]:
    """A quarterly share between 0 and 1 (``loss``) and two drivers, for the model contract.

    The share is positive and below one, so forms with a domain (a binomial
    GLM, a log) can fit it; its noise is AR(1), and it has a mild season.
    """
    rng = np.random.default_rng(seed)
    index = pd.period_range("2005Q1", periods=n, freq="Q")
    X = pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["gdp", "rate"])
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + rng.normal(scale=0.02)
    season = np.resize([0.01, -0.005, 0.0, -0.005], n)
    y = 0.3 + 0.03 * X["gdp"] - 0.02 * X["rate"] + season + e
    return y.rename("loss"), X


class ModelContract(ComponentContract):
    """Tests every model form passes (design §6, §13).

    Set `key` (and `registry`, when the form is not in `dazflow.models.registry`)
    in a subclass whose name starts with ``Test``. Optional: `config`, the
    form's parameters under test; `sample()`, the data (`sample_series` by
    default); `refit_tolerance`, how close a refit from the stored record
    must come (absolute, per parameter).
    """

    registry: ClassVar[Registry] = models.registry
    config: ClassVar[dict[str, Any]] = {}
    horizon: ClassVar[int] = 5
    levels: ClassVar[tuple[float, ...]] = (0.8, 0.95)
    refit_tolerance: ClassVar[float] = 1e-10

    def sample(self) -> tuple[pd.Series, pd.DataFrame]:
        return sample_series()

    def component(self) -> Any:
        assert self.key in self.registry, f"{self.key!r} is not registered"
        return super().component()

    def uses_drivers(self) -> bool:
        return bool(getattr(self.component(), "drivers", False))

    def form(self) -> Any:
        return self.component()(dict(self.config))

    def fit(self) -> tuple[Any, pd.Series, pd.DataFrame | None]:
        y, X = self.sample()
        X = X if self.uses_drivers() else None
        return self.form().fit(y, X), y, X

    def future(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        if X is None:
            return None
        index = pd.period_range(X.index[-1] + 1, periods=self.horizon, freq=X.index.freq)
        values = np.random.default_rng(1).normal(size=(self.horizon, X.shape[1]))
        return pd.DataFrame(values, index=index, columns=X.columns)

    # --- the tests -----------------------------------------------------------------

    def test_form_is_built_from_its_config(self) -> None:
        cls = self.component()
        config = cls.Config.model_validate(dict(self.config))
        form = cls(config)
        assert getattr(form, "config", config) == config, (
            f"{self.key!r} built from {config!r} holds another configuration"
        )

    def test_fit_returns_a_fitted_model(self) -> None:
        fitted, y, X = self.fit()
        missing = [m for m in _protocol_members(models.Fitted) if not hasattr(fitted, m)]
        assert not missing, f"{self.key!r}.fit() result lacks {', '.join(missing)}"
        params = fitted.params
        assert isinstance(params, pd.Series) and len(params), "params is not a non-empty Series"
        assert all(isinstance(k, str) for k in params.index), "parameter names are not text"
        assert params.index.is_unique, "parameter names repeat"
        assert np.isfinite(params.to_numpy(dtype=float)).all(), f"params not finite: {params}"
        cov = fitted.cov_params
        assert isinstance(cov, pd.DataFrame), "cov_params is not a DataFrame"
        assert list(cov.index) == list(params.index) == list(cov.columns), (
            "cov_params is not indexed by the parameters on both axes"
        )
        resid = fitted.residuals
        assert isinstance(resid, pd.Series) and isinstance(resid.index, pd.PeriodIndex), (
            "residuals are not a Series on a PeriodIndex"
        )
        assert len(resid) and resid.index.isin(y.index).all(), (
            "residuals are not on the sample's periods"
        )
        assert np.isfinite(resid.to_numpy(dtype=float)).all(), "residuals are not finite"
        spec = fitted.spec
        assert isinstance(spec, models.ModelSpec), "spec is not a ModelSpec"
        assert spec.form == self.key, f"spec names the form {spec.form!r}, not {self.key!r}"
        assert spec.target == y.name, f"spec names the target {spec.target!r}, not {y.name!r}"
        drivers = () if X is None else tuple(X.columns)
        assert spec.drivers == drivers, f"spec names drivers {spec.drivers}, not {drivers}"

    def test_predict_covers_the_horizon_with_nested_intervals(self) -> None:
        fitted, y, X = self.fit()
        fc = fitted.predict(self.horizon, self.future(X), levels=self.levels)
        assert isinstance(fc, models.Forecast), "predict() did not return a Forecast"
        expected = tuple(str(p) for p in pd.period_range(y.index[-1] + 1, periods=self.horizon))
        assert fc.periods == expected, f"forecast periods {fc.periods}, not {expected}"
        assert fc.frequency == y.index.freqstr, f"forecast frequency {fc.frequency}"
        assert fc.levels == tuple(sorted(self.levels)), f"forecast levels {fc.levels}"
        mean = np.asarray(fc.mean)
        assert np.isfinite(mean).all(), f"forecast mean not finite: {mean}"
        for i, level in enumerate(fc.levels):
            lo, up = np.asarray(fc.lower[i]), np.asarray(fc.upper[i])
            assert (lo <= mean).all() and (mean <= up).all(), (
                f"the {level:g} interval does not hold the mean"
            )
            if i:
                assert (np.asarray(fc.lower[i - 1]) >= lo).all(), "intervals are not nested"
                assert (np.asarray(fc.upper[i - 1]) <= up).all(), "intervals are not nested"

    def test_fit_leaves_its_inputs_unchanged(self) -> None:
        y, X = self.sample()
        y0, X0 = y.copy(), X.copy()
        self.form().fit(y, X if self.uses_drivers() else None)
        pd.testing.assert_series_equal(y, y0)
        pd.testing.assert_frame_equal(X, X0)

    def test_refit_from_the_stored_record_repeats_the_fit(self) -> None:
        fitted, y, X = self.fit()
        text = json.loads(json.dumps(fitted.record().to_dict(), allow_nan=False))
        try:
            again = models.refit(text, y, X, registry=self.registry)
        except DazflowError as err:
            raise AssertionError(f"a refit from the stored record failed: {err}") from err
        diff = models.ModelRecord.from_dict(text).difference(again)
        assert diff <= self.refit_tolerance, (
            f"a refit from the stored record moved a parameter by {diff:.3g} "
            f"(tolerance {self.refit_tolerance:g})"
        )

    def test_drivers_are_used_or_refused(self) -> None:
        y, X = self.sample()
        form = self.form()
        if self.uses_drivers():
            fitted = form.fit(y, X)
            try:
                fitted.predict(self.horizon)
            except ModelError:
                pass
            else:
                raise AssertionError("predict() without future drivers did not raise ModelError")
        else:
            try:
                form.fit(y, X)
            except ModelError:
                pass
            else:
                raise AssertionError(f"{self.key!r} takes no drivers but accepted them")
