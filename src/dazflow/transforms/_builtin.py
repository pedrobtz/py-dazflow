"""The built-in transforms: diff, seasonal_diff, log, logdiff, growth, logit, boxcox."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from dazflow.core.config import Config
from dazflow.core.errors import TransformError
from dazflow.transforms._base import domain_error, register, shifted, undo_lag

__all__ = ["BoxCox", "Diff", "Growth", "Log", "LogDiff", "Logit", "SeasonalDiff"]


def _like(x: pd.Series, values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=x.index, name=x.name)


class _Stateless:
    """A transform of each value on its own: nothing to resolve, nothing consumed."""

    def resolve(self, x: pd.Series, config: Config) -> Config:
        return config

    def consumed(self, config: Config) -> int:
        return 0


class _NoParams(Config):
    """No parameters."""


class LagConfig(Config):
    lag: int = Field(
        1, ge=1, description="Periods between the value and the one it is compared with."
    )


class _Lagged:
    """A transform of each value against the value `lag` periods earlier."""

    def resolve(self, x: pd.Series, config: LagConfig) -> LagConfig:
        return config

    def consumed(self, config: LagConfig) -> int:
        return config.lag


@register("diff")
class Diff(_Lagged):
    """``y[t] = x[t] - x[t - lag]``."""

    Config = LagConfig

    def forward(self, x: pd.Series, config: LagConfig) -> pd.Series:
        return _like(x, x.to_numpy() - shifted(x, config.lag))

    def inverse(self, y: pd.Series, config: LagConfig, anchor: pd.Series) -> pd.Series:
        return undo_lag(y, anchor, config.lag, lambda v, base: v + base)


class SeasonalConfig(Config):
    period: int | None = Field(
        None, ge=2, description="Periods per season; unset reads 12 for monthly, 4 for quarterly."
    )


_SEASON = {"M": 12, "Q-DEC": 4}


@register("seasonal_diff")
class SeasonalDiff:
    """``y[t] = x[t] - x[t - period]``, the period read from the frequency when unset."""

    Config = SeasonalConfig

    def resolve(self, x: pd.Series, config: SeasonalConfig) -> SeasonalConfig:
        if config.period is not None:
            return config
        period = _SEASON.get(x.index.freqstr)
        if period is None:
            raise TransformError(
                f"seasonal_diff has no seasonal period for the {x.index.freqstr} series {str(x.name)!r}",
                observed=f"frequency {x.index.freqstr}; known: monthly (12), quarterly (4)",
                change="Give the period: seasonal_diff(period=n).",
                series=str(x.name),
            )
        return SeasonalConfig(period=period)

    def consumed(self, config: SeasonalConfig) -> int:
        return config.period

    def forward(self, x: pd.Series, config: SeasonalConfig) -> pd.Series:
        return _like(x, x.to_numpy() - shifted(x, config.period))

    def inverse(self, y: pd.Series, config: SeasonalConfig, anchor: pd.Series) -> pd.Series:
        return undo_lag(y, anchor, config.period, lambda v, base: v + base)


def _positive(key: str, x: pd.Series) -> None:
    v = x.to_numpy()
    domain_error(key, x, ~np.isnan(v) & (v <= 0), "values are greater than 0")


@register("log")
class Log(_Stateless):
    """``y = log(x)``, for x > 0."""

    Config = _NoParams

    def forward(self, x: pd.Series, config: Config) -> pd.Series:
        _positive("log", x)
        return _like(x, np.log(x.to_numpy()))

    def inverse(self, y: pd.Series, config: Config, anchor: pd.Series) -> pd.Series:
        return _like(y, np.exp(y.to_numpy()))


@register("logdiff")
class LogDiff(_Lagged):
    """``y[t] = log(x[t]) - log(x[t - lag])``, for x > 0."""

    Config = LagConfig

    def forward(self, x: pd.Series, config: LagConfig) -> pd.Series:
        _positive("logdiff", x)
        return _like(x, np.log(x.to_numpy()) - np.log(shifted(x, config.lag)))

    def inverse(self, y: pd.Series, config: LagConfig, anchor: pd.Series) -> pd.Series:
        return undo_lag(y, anchor, config.lag, lambda v, base: base * float(np.exp(v)))


@register("growth")
class Growth(_Lagged):
    """``y[t] = x[t] / x[t - lag] - 1``, for x[t - lag] other than 0."""

    Config = LagConfig

    def forward(self, x: pd.Series, config: LagConfig) -> pd.Series:
        v = x.to_numpy()
        # A value is a denominator when the period `lag` later is in the series.
        is_base = np.asarray((x.index + config.lag).isin(x.index))
        domain_error("growth", x, is_base & (v == 0), "the earlier value is not 0")
        return _like(x, v / shifted(x, config.lag) - 1.0)

    def inverse(self, y: pd.Series, config: LagConfig, anchor: pd.Series) -> pd.Series:
        return undo_lag(y, anchor, config.lag, lambda v, base: base * (1.0 + v))


@register("logit")
class Logit(_Stateless):
    """``y = log(x / (1 - x))``, for 0 < x < 1."""

    Config = _NoParams

    def forward(self, x: pd.Series, config: Config) -> pd.Series:
        from scipy.special import logit

        v = x.to_numpy()
        domain_error("logit", x, ~np.isnan(v) & ((v <= 0) | (v >= 1)), "0 < value < 1")
        return _like(x, logit(v))

    def inverse(self, y: pd.Series, config: Config, anchor: pd.Series) -> pd.Series:
        from scipy.special import expit

        return _like(y, expit(y.to_numpy()))


class BoxCoxConfig(Config):
    lmbda: float | None = Field(
        None, description="The Box-Cox lambda; unset estimates it by maximum likelihood."
    )
    bound: float = Field(
        2.0,
        gt=0,
        description=(
            "An estimated lmbda lies in [-bound, bound]: past it, x**lmbda loses the digits "
            "the inverse needs."
        ),
    )


# Below this |lmbda|, the second-order series in lmbda is exact to double precision.
_SMALL_LAMBDA = 1e-12


@register("boxcox")
class BoxCox:
    """``y = (x**lmbda - 1) / lmbda`` (``log(x)`` at 0), for x > 0; unset lmbda is estimated."""

    Config = BoxCoxConfig

    def resolve(self, x: pd.Series, config: BoxCoxConfig) -> BoxCoxConfig:
        if config.lmbda is not None:
            return config
        _positive("boxcox", x)
        v = x.dropna().to_numpy()
        if len(v) < 2 or np.ptp(v) == 0:
            raise TransformError(
                f"boxcox cannot estimate lmbda for series {str(x.name)!r}",
                observed=f"{len(v)} values, {len(np.unique(v))} distinct",
                change="Give lmbda: boxcox(lmbda=...), or use a series with at least two distinct values.",
                series=str(x.name),
            )
        from scipy.optimize import minimize_scalar
        from scipy.stats import boxcox_llf

        with np.errstate(all="ignore"):
            best = minimize_scalar(
                lambda lmbda: -boxcox_llf(lmbda, v),
                bounds=(-config.bound, config.bound),
                method="bounded",
            )
        return config.model_copy(update={"lmbda": float(best.x)})

    def consumed(self, config: BoxCoxConfig) -> int:
        return 0

    def forward(self, x: pd.Series, config: BoxCoxConfig) -> pd.Series:
        _positive("boxcox", x)
        lmbda, log_x = config.lmbda, np.log(x.to_numpy())
        if abs(lmbda) < _SMALL_LAMBDA:
            # (x**l - 1) / l = log x (1 + l log x / 2 + O(l**2)): exact to double precision here,
            # where l log x could underflow (scipy 1.13's inv_boxcox fails at l = 5e-324).
            return _like(x, log_x * (1 + lmbda * log_x / 2))
        return _like(x, np.expm1(lmbda * log_x) / lmbda)

    def inverse(self, y: pd.Series, config: BoxCoxConfig, anchor: pd.Series) -> pd.Series:
        v, lmbda = y.to_numpy(), config.lmbda
        if lmbda != 0:
            rule = f"{lmbda!r} * value + 1 > 0"
            domain_error("boxcox", y, ~np.isnan(v) & (lmbda * v + 1 <= 0), rule, inverse=True)
        if abs(lmbda) < _SMALL_LAMBDA:
            return _like(y, np.exp(v * (1 - lmbda * v / 2)))
        with np.errstate(invalid="ignore"):  # NaN stays NaN
            return _like(y, np.exp(np.log1p(lmbda * v) / lmbda))
