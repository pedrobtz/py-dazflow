"""The built-in forms: naive, seasonal naive, historic mean, OLS, GLM, ARIMAX and ETS.

The three benchmarks are a few lines of numpy, checked against the
statsmodels models they equal (a random walk, a seasonal random walk,
regression on a constant). The other forms wrap statsmodels, which is
imported when a form is fitted so that ``import dazflow`` stays light.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from dazflow.core.config import Config
from dazflow.core.errors import ModelError
from dazflow.models._base import (
    FittedModel,
    ModelForm,
    ModelSpec,
    _round_start,
    register,
    season_length,
)

__all__ = [
    "ARIMAX",
    "ETS",
    "GLM",
    "OLS",
    "HistoricMean",
    "Naive",
    "SeasonalNaive",
]

Paths = tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]


def _z(level: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2))


@contextmanager
def _notes(into: list[str]) -> Iterator[None]:
    """Collect statsmodels' fit warnings (convergence, start values) as notes.

    User and runtime warnings raised while fitting are model findings, kept in
    `FittedModel.notes`; any other warning is passed on as raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("always", RuntimeWarning)
        yield
    for w in caught:
        if issubclass(w.category, UserWarning | RuntimeWarning):
            into.append(f"{w.category.__name__}: {w.message}")
        else:  # pragma: no cover - depends on the installed versions
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _flat(levels: tuple[float, ...], mean: np.ndarray, half: list[np.ndarray]) -> Paths:
    return mean, [mean - w for w in half], [mean + w for w in half]


# --- benchmarks ------------------------------------------------------------------


class _NoParams(Config):
    """No parameters."""


class _RandomWalk(FittedModel):
    """A fitted (seasonal) random walk: the last value of the season carried forward."""

    def __init__(self, spec: ModelSpec, y: pd.Series, season: int) -> None:
        values = y.to_numpy()
        resid = values[season:] - values[:-season]
        sigma2 = float(np.mean(resid**2))
        m = len(resid)
        super().__init__(
            spec,
            y,
            None,
            params=pd.Series({"sigma2": sigma2}),
            cov_params=pd.DataFrame([[2 * sigma2**2 / m]], index=["sigma2"], columns=["sigma2"]),
            fitted_values=pd.Series(values[:-season], index=y.index[season:]),
        )
        self._season = season

    def _predict(self, h: int, X_future: Any, levels: tuple[float, ...]) -> Paths:
        values = self._y.to_numpy()
        m = self._season
        steps = np.arange(h)
        mean = values[len(values) - m + steps % m]
        sd = np.sqrt(self._params["sigma2"] * (steps // m + 1))
        return _flat(levels, mean, [_z(lv) * sd for lv in levels])


@register("naive")
class Naive(ModelForm):
    """The last value carried forward: a random walk, ``y[t] = y[t-1] + e[t]``.

    `sigma2` is the mean squared one-step change (the maximum-likelihood
    estimate), and the interval at horizon h is ``± z sigma sqrt(h)``.
    """

    Config = _NoParams

    def _fit(self, y: pd.Series, X: Any, spec: ModelSpec) -> FittedModel:
        return _RandomWalk(spec, y, 1)


class SeasonalNaiveConfig(Config):
    season: int | None = Field(
        None, ge=1, description="Periods per season; unset reads 12 monthly, 4 quarterly, 1 annual."
    )


@register("seasonal_naive")
class SeasonalNaive(ModelForm):
    """The value one season earlier carried forward: ``y[t] = y[t-m] + e[t]``.

    The interval at horizon h is ``± z sigma sqrt(k + 1)`` with ``k = (h-1) // m``.
    """

    Config = SeasonalNaiveConfig

    def min_observations(self, X: Any) -> int:
        return (self.config.season or 1) + 2

    def _fit(self, y: pd.Series, X: Any, spec: ModelSpec) -> FittedModel:
        m = self.config.season or season_length(y.index)
        if len(y) < m + 2:
            raise ModelError(
                f"the seasonal_naive form needs at least {m + 2} observations",
                observed=f"{len(y)} observations, season {m}",
                change="Fit it on a longer series.",
            )
        return _RandomWalk(spec, y, m)


class _Mean(FittedModel):
    def _predict(self, h: int, X_future: Any, levels: tuple[float, ...]) -> Paths:
        from scipy.stats import t

        n = len(self._y)
        s = np.sqrt(float(np.sum(self.residuals**2)) / (n - 1))
        mean = np.full(h, self._params["const"])
        half = [float(t.ppf(0.5 + lv / 2, n - 1)) * s * np.sqrt(1 + 1 / n) for lv in levels]
        return _flat(levels, mean, [np.full(h, w) for w in half])


@register("mean")
class HistoricMean(ModelForm):
    """The historic mean, ``y[t] = c + e[t]``: regression on a constant.

    The interval is the regression's prediction interval,
    ``± t(n-1) s sqrt(1 + 1/n)``.
    """

    Config = _NoParams

    def _fit(self, y: pd.Series, X: Any, spec: ModelSpec) -> FittedModel:
        values = y.to_numpy()
        n = len(values)
        c = float(values.mean())
        s2 = float(np.sum((values - c) ** 2)) / (n - 1)
        return _Mean(
            spec,
            y,
            None,
            params=pd.Series({"const": c}),
            cov_params=pd.DataFrame([[s2 / n]], index=["const"], columns=["const"]),
            fitted_values=pd.Series(c, index=y.index),
        )


# --- regression --------------------------------------------------------------------


def _design(X: pd.DataFrame | None, index: pd.PeriodIndex, constant: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=index) if X is None else X.copy()
    if constant:
        out.insert(0, "const", 1.0)
    return out


def newey_west_lags(n: int) -> int:
    """The Newey-West rule of thumb, ``floor(4 (n / 100) ** (2 / 9))``."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


class OLSConfig(Config):
    constant: bool = Field(True, description="Include an intercept, named 'const'.")
    cov_type: Literal["nonrobust", "HC0", "HC1", "HC3", "HAC"] = Field(
        "nonrobust", description="Covariance of the coefficients, as statsmodels names it."
    )
    hac_maxlags: int | None = Field(
        None, ge=0, description="Lags for HAC; unset: floor(4 (n/100)^(2/9))."
    )


class _OLSFit(FittedModel):
    def _predict(self, h: int, X_future: pd.DataFrame | None, levels: tuple[float, ...]) -> Paths:
        exog = _design(X_future, self.future_index(h), self._constant)
        pred = self._result.get_prediction(exog)
        mean = pred.predicted_mean
        lower, upper = [], []
        for lv in levels:
            frame = pred.summary_frame(alpha=1 - lv)
            lower.append(frame["obs_ci_lower"].to_numpy())
            upper.append(frame["obs_ci_upper"].to_numpy())
        return np.asarray(mean), lower, upper


@register("ols")
class OLS(ModelForm):
    """Linear regression on the drivers, statsmodels `OLS`.

    The interval is statsmodels' prediction interval for a new observation
    (``obs_ci_lower``, ``obs_ci_upper``).
    """

    Config = OLSConfig
    drivers = True

    def _fit(self, y: pd.Series, X: pd.DataFrame | None, spec: ModelSpec) -> FittedModel:
        import statsmodels.api as sm

        cfg = self.config
        exog = _design(X, y.index, cfg.constant)
        if not len(exog.columns):
            raise ModelError(
                "the ols form has nothing to regress on",
                observed="no constant and no drivers",
                change="Set constant: true or give drivers.",
            )
        kwargs: dict[str, Any] = {"cov_type": cfg.cov_type}
        if cfg.cov_type == "HAC":
            lags = cfg.hac_maxlags if cfg.hac_maxlags is not None else newey_west_lags(len(y))
            kwargs["cov_kwds"] = {"maxlags": lags}
        notes: list[str] = []
        with _notes(notes):
            result = sm.OLS(y, exog).fit(**kwargs)
        fitted = _OLSFit(
            spec,
            y,
            X,
            params=result.params,
            cov_params=result.cov_params(),
            fitted_values=result.fittedvalues,
            notes=notes,
        )
        fitted._result = result
        fitted._constant = cfg.constant
        return fitted


class GLMConfig(Config):
    family: Literal["binomial", "gaussian"] = Field(
        "binomial",
        description="binomial (logit link) for a share between 0 and 1; gaussian (identity).",
    )
    constant: bool = Field(True, description="Include an intercept, named 'const'.")
    scale: Literal["X2", "dev", "fixed"] = Field(
        "X2",
        description=(
            "Dispersion: Pearson chi-squared (X2) or deviance (dev) over the residual degrees "
            "of freedom, the quasi-likelihood fit of a fractional target; fixed keeps the "
            "family's own (1 for binomial)."
        ),
    )
    cov_type: Literal["nonrobust", "HC0", "HC1", "HC3"] = Field(
        "nonrobust", description="Covariance of the coefficients, as statsmodels names it."
    )


class _GLMFit(FittedModel):
    def _predict(self, h: int, X_future: pd.DataFrame | None, levels: tuple[float, ...]) -> Paths:
        exog = _design(X_future, self.future_index(h), self._constant)
        frame = self._result.get_prediction(exog).summary_frame()
        mean = frame["mean"].to_numpy()
        variance = self._result.scale * self._result.family.variance(mean) + frame["mean_se"] ** 2
        sd = np.sqrt(np.asarray(variance))
        lower, upper = [], []
        for lv in levels:
            lo, up = mean - _z(lv) * sd, mean + _z(lv) * sd
            if self._bounded:
                lo, up = np.clip(lo, 0.0, 1.0), np.clip(up, 0.0, 1.0)
            lower.append(lo)
            upper.append(up)
        return mean, lower, upper


@register("glm")
class GLM(ModelForm):
    """A generalised linear model, statsmodels `GLM`; binomial for fractional targets.

    The binomial family takes a target in [0, 1] (a rate or share) with the
    logit link and, by default, the Pearson dispersion: the quasi-likelihood
    fractional logit. The interval is ``mean ± z sqrt(scale V(mean) + se²)``,
    the normal approximation for a new observation with the variance function
    ``V`` of the family and the standard error of the fitted mean, clipped to
    [0, 1] for binomial.
    """

    Config = GLMConfig
    drivers = True

    def _fit(self, y: pd.Series, X: pd.DataFrame | None, spec: ModelSpec) -> FittedModel:
        import statsmodels.api as sm

        cfg = self.config
        if cfg.family == "binomial" and (y.min() < 0 or y.max() > 1):
            raise ModelError(
                "the binomial glm needs a target between 0 and 1",
                observed=f"{y.name!r} ranges {y.min():g} to {y.max():g}",
                change="Model a share or rate in [0, 1], or use family: gaussian.",
            )
        exog = _design(X, y.index, cfg.constant)
        if not len(exog.columns):
            raise ModelError(
                "the glm form has nothing to regress on",
                observed="no constant and no drivers",
                change="Set constant: true or give drivers.",
            )
        family = sm.families.Binomial() if cfg.family == "binomial" else sm.families.Gaussian()
        notes: list[str] = []
        with _notes(notes):
            result = sm.GLM(y, exog, family=family).fit(
                scale=None if cfg.scale == "fixed" else cfg.scale, cov_type=cfg.cov_type
            )
        fitted = _GLMFit(
            spec,
            y,
            X,
            params=result.params,
            cov_params=result.cov_params(),
            fitted_values=result.fittedvalues,
            notes=notes,
        )
        fitted._result = result
        fitted._constant = cfg.constant
        fitted._bounded = cfg.family == "binomial"
        return fitted


# --- state-space forms ----------------------------------------------------------------


class ARIMAXConfig(Config):
    order: tuple[int, int, int] = Field(
        (1, 0, 0), description="(p, d, q): autoregressive, difference and moving-average orders."
    )
    seasonal_order: tuple[int, int, int, int] = Field(
        (0, 0, 0, 0), description="(P, D, Q, s): the seasonal orders and the season length."
    )
    trend: Literal["n", "c", "t", "ct"] = Field(
        "c", description="Deterministic terms: none, constant, time trend, or both."
    )
    maxiter: int = Field(200, ge=1, description="Most iterations of the L-BFGS optimiser.")

    @model_validator(mode="after")
    def _orders(self) -> Self:
        if min(self.order) < 0 or min(self.seasonal_order) < 0:
            raise ValueError("orders cannot be negative")
        if any(self.seasonal_order[:3]) and self.seasonal_order[3] < 2:
            raise ValueError("a seasonal order needs a season length s of at least 2")
        return self


class _StateSpaceFit(FittedModel):
    def _predict(self, h: int, X_future: pd.DataFrame | None, levels: tuple[float, ...]) -> Paths:
        pred = self._result.get_forecast(steps=h, exog=X_future)
        mean = np.asarray(pred.predicted_mean)
        lower, upper = [], []
        for lv in levels:
            frame = pred.summary_frame(alpha=1 - lv)
            lower.append(frame["mean_ci_lower"].to_numpy())
            upper.append(frame["mean_ci_upper"].to_numpy())
        return mean, lower, upper


@register("arimax")
class ARIMAX(ModelForm):
    """ARIMA with drivers, statsmodels `SARIMAX`, fitted by maximum likelihood (L-BFGS).

    The optimiser starts at statsmodels' start values rounded to 10
    significant digits, so that refitting the same data repeats the fit
    exactly (D36). Residuals leave out the diffuse burn-in periods. The
    interval is the state-space forecast interval.
    """

    Config = ARIMAXConfig
    drivers = True

    def min_observations(self, X: Any) -> int:
        p, d, q = self.config.order
        P, D, Q, s = self.config.seasonal_order
        return p + d + q + s * (P + D + Q) + 3 + (0 if X is None else len(X.columns))

    def _fit(self, y: pd.Series, X: pd.DataFrame | None, spec: ModelSpec) -> FittedModel:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        cfg = self.config
        notes: list[str] = []
        with _notes(notes):
            model = SARIMAX(
                y, exog=X, order=cfg.order, seasonal_order=cfg.seasonal_order, trend=cfg.trend
            )
            result = model.fit(
                start_params=_round_start(model.start_params), disp=False, maxiter=cfg.maxiter
            )
            cov = result.cov_params()
        burn = max(int(result.loglikelihood_burn), int(result.nobs_diffuse))
        fitted = _StateSpaceFit(
            spec,
            y,
            X,
            params=result.params,
            cov_params=cov,
            fitted_values=result.fittedvalues.iloc[burn:],
            notes=notes,
        )
        fitted._result = result
        return fitted


class ETSConfig(Config):
    error: Literal["add"] = Field("add", description="Additive errors.")
    trend: Literal["add"] | None = Field(None, description="An additive trend, or none.")
    damped_trend: bool = Field(False, description="Damp the trend.")
    seasonal: Literal["add"] | None = Field(None, description="An additive season, or none.")
    seasonal_periods: int | None = Field(
        None, ge=2, description="Season length; unset reads 12 monthly and 4 quarterly."
    )
    maxiter: int = Field(1000, ge=1, description="Most iterations of the L-BFGS optimiser.")

    @model_validator(mode="after")
    def _damped(self) -> Self:
        if self.damped_trend and self.trend is None:
            raise ValueError("damped_trend needs a trend")
        return self


class _ETSFit(FittedModel):
    def _predict(self, h: int, X_future: Any, levels: tuple[float, ...]) -> Paths:
        n = len(self._y)
        pred = self._result.get_prediction(start=n, end=n + h - 1)
        mean = np.asarray(pred.predicted_mean)
        lower, upper = [], []
        for lv in levels:
            frame = pred.summary_frame(alpha=1 - lv)
            lower.append(frame["pi_lower"].to_numpy())
            upper.append(frame["pi_upper"].to_numpy())
        return mean, lower, upper


@register("ets")
class ETS(ModelForm):
    """Exponential smoothing, statsmodels `ETSModel`, with additive errors.

    Trend and season are additive or absent, the models for which
    statsmodels computes the forecast interval exactly; multiplicative
    components need simulated intervals and are left for later (D36). The
    optimiser starts at statsmodels' start values rounded to 10 significant
    digits. Takes no drivers.
    """

    Config = ETSConfig

    def min_observations(self, X: Any) -> int:
        return 2 * (self.config.seasonal_periods or 1) + 4

    def _fit(self, y: pd.Series, X: Any, spec: ModelSpec) -> FittedModel:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        cfg = self.config
        periods = None
        if cfg.seasonal is not None:
            periods = cfg.seasonal_periods or season_length(y.index)
            if periods < 2:
                raise ModelError(
                    "a seasonal ets needs a season of at least 2 periods",
                    observed=f"frequency {y.index.freqstr}, season {periods}",
                    change="Set seasonal: null for annual data, or give seasonal_periods.",
                )
            if len(y) < 2 * periods + 4:
                raise ModelError(
                    f"a seasonal ets needs at least {2 * periods + 4} observations",
                    observed=f"{len(y)} observations, season {periods}",
                    change="Fit it on a longer series, or without a season.",
                )
        notes: list[str] = []
        with _notes(notes):
            model = ETSModel(
                y,
                error=cfg.error,
                trend=cfg.trend,
                damped_trend=cfg.damped_trend,
                seasonal=cfg.seasonal,
                seasonal_periods=periods,
            )
            result = model.fit(
                start_params=_round_start(model.start_params), disp=False, maxiter=cfg.maxiter
            )
            names = list(model.param_names)
            cov = pd.DataFrame(np.asarray(result.cov_params()), index=names, columns=names)
        fitted = _ETSFit(
            spec,
            y,
            None,
            params=pd.Series(np.asarray(result.params), index=names),
            cov_params=cov,
            fitted_values=pd.Series(np.asarray(result.fittedvalues), index=y.index),
            notes=notes,
        )
        fitted._result = result
        return fitted
