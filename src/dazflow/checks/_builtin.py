"""The seven built-in checks and the STL outlier scores (design §7.2, D29)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dazflow.checks._base import CheckConfig, Context, Units, register
from dazflow.core.config import Config
from dazflow.core.errors import CheckError

__all__ = ["Downturn", "outlier_scores"]

_PERIOD = {"M": 12, "Q": 4}


def _span(x: pd.Series) -> pd.Series:
    """`x` from its first to its last value (empty when it has none)."""
    first, last = x.first_valid_index(), x.last_valid_index()
    if first is None:
        return x.iloc[:0]
    return x.loc[first:last]


def _missing(x: pd.Series, context: Context) -> pd.Series:
    """True where a value is missing and was not excluded by a treatment."""
    return x.isna() & ~x.index.isin(list(context.excluded))


@register("completeness")
class Completeness:
    """Every period of the calendar has a value. A unit fails when its value is missing."""

    class Config(CheckConfig):
        warning: float | None = 1
        error: float = 0.1

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        return Units(values=x, test="not_null", measure="value")


@register("gaps")
class Gaps:
    """No run of missing values inside the history is longer than `max_length` periods.

    The units are the periods from the first to the last value; each holds
    the length of the run of missing values it is in (0 when it has a value).
    """

    class Config(CheckConfig):
        max_length: int = Field(0, ge=0, description="Longest accepted run of missing periods.")

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        span = _span(x)
        gap = _missing(span, context).to_numpy()
        run = np.zeros(len(span))
        i = 0
        while i < len(gap):
            if gap[i]:
                j = i
                while j < len(gap) and gap[j]:
                    j += 1
                run[i:j] = j - i
                i = j
            else:
                i += 1
        return Units(
            values=pd.Series(run, index=span.index),
            test="at_most",
            bound=float(config.max_length),
            measure="length of the gap",
        )


@register("duplicates")
class Duplicates:
    """No value repeats the previous period's value, as a value carried forward would.

    A `Dataset` has one row per period (§5.1), so a repeated period cannot
    reach `clean`; a repeated value is the duplicate left to find. The
    units are the periods with a value whose previous period has one; each
    holds the absolute change, which must be above 0.
    """

    class Config(CheckConfig):
        warning: float | None = 1
        error: float = 0.1

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        change = (x - x.shift(1)).abs()
        return Units(values=change.dropna(), test="above", bound=0.0, measure="absolute change")


@register("range")
class Range:
    """Values stay within configured bounds, inclusive. Series without bounds are not checked."""

    class Config(CheckConfig):
        bounds: dict[str, tuple[float | None, float | None]] = Field(
            default_factory=dict,
            description="[lower, upper] per series; null leaves that side open.",
        )

        @field_validator("bounds")
        @classmethod
        def _ordered(cls, bounds: dict[str, tuple[float | None, float | None]]) -> Any:
            for name, (lo, hi) in bounds.items():
                if lo is None and hi is None:
                    raise ValueError(f"the bounds of {name!r} are both open")
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError(f"the lower bound of {name!r} is above its upper bound")
            return bounds

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        if x.name not in config.bounds:
            return Units.not_applicable("no bounds are configured for the series")
        lo, hi = config.bounds[x.name]
        return Units(values=x.dropna(), test="between", bound=lo, upper=hi, measure="value")


@register("revisions")
class Revisions:
    """Values already published change by at most `tolerance` of the previous vintage's value.

    The units are the periods with a value in both vintages; each holds
    ``|new - old| / |old|`` (0 when both are 0, infinite when only old is).
    """

    class Config(CheckConfig):
        tolerance: float = Field(0.01, ge=0, description="Largest accepted relative revision.")

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        if context.previous is None:
            return Units.not_applicable("no previous vintage of the series was given")
        old = context.previous.reindex(x.index)
        both = x.notna() & old.notna()
        new, old = x[both].to_numpy(dtype=float), old[both].to_numpy(dtype=float)
        delta = np.abs(new - old)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(old == 0, np.where(delta == 0, 0.0, np.inf), delta / np.abs(old))
        return Units(
            values=pd.Series(rel, index=x.index[both.to_numpy()]),
            test="at_most",
            bound=config.tolerance,
            measure="relative revision",
        )


def outlier_scores(
    x: pd.Series,
    frequency: str | None = None,
    *,
    z: float = 4.0,
    seasonal: int = 13,
    trend: int | None = None,
    max_iter: int = 20,
) -> pd.Series:
    """Robust z-scores of `x`'s residuals from an STL decomposition, on its present values.

    Monthly and quarterly series are decomposed by statsmodels' `STL`
    (seasonal smoother of `seasonal` years, local-constant); annual series,
    which have no season, by a `lowess` trend over `trend` years (default 21).
    The residuals are standardised by their median and MAD (1.4826 MAD,
    the standard deviation for normal residuals). A first, robust fit finds
    the gross outliers; then each flagged period is replaced by its fitted
    value and the series decomposed again without robustness weights, until
    the flagged set is stable, so a spike does not bend the fit around
    itself (D29). Missing values inside the history are interpolated for
    the fit and get no score. Raises `CheckError` when the series has fewer
    than two years from its first to its last value (seven for annual data).
    """
    from dazflow.readers._base import frequency_of

    freq = frequency or frequency_of(x.index)
    span = _span(x)
    period = _PERIOD.get(freq or "")
    minimum = _minimum(period)
    if len(span) < minimum:
        raise CheckError(
            f"the series {x.name!r} is too short for outlier scores",
            observed=f"{len(span)} periods from its first to its last value",
            change=f"Score a series with at least {minimum} periods (two years).",
        )
    missing = span.isna().to_numpy()
    y = span.interpolate(method="linear").to_numpy(dtype=float)
    # First pass: residuals from a robust fit, which a gross spike cannot bend,
    # scaled by the plain fit's residuals, which a robust fit on a short
    # series understates; only gross outliers are flagged.
    plain = _fit(y, period, seasonal, trend, robust=False)
    robust = _fit(y, period, seasonal, trend, robust=True)
    scores = _scores(y - robust, missing, scale_of=y - plain)
    mask = (scores > z) & ~missing
    fitted = robust
    for _ in range(max_iter):
        fitted, scores = _refit(y, mask, fitted, missing, period, seasonal, trend)
        # A period flagged by the robust pass stays flagged only if it still
        # scores above z when the fit sees its own value.
        for p in sorted(np.flatnonzero(mask), key=lambda i: scores[i]):
            alone = mask.copy()
            alone[p] = False
            f, s = _refit(y, alone, fitted, missing, period, seasonal, trend)
            if s[p] <= z:
                mask, fitted, scores = alone, f, s
        new = (scores > z) & ~missing
        if np.array_equal(new, mask):
            break
        mask = new
    return pd.Series(scores[~missing], index=span.index[~missing], name=x.name)


def _refit(
    y: np.ndarray,
    mask: np.ndarray,
    fitted: np.ndarray,
    missing: np.ndarray,
    period: int | None,
    seasonal: int,
    trend: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The plain fit with the flagged periods replaced by `fitted`, and the scores it gives."""
    yy = y.copy()
    yy[mask] = fitted[mask]
    out = _fit(yy, period, seasonal, trend, robust=False)
    return out, _scores(y - out, missing)


def _scores(r: np.ndarray, missing: np.ndarray, scale_of: np.ndarray | None = None) -> np.ndarray:
    """|r - median| / (1.4826 MAD), the median and MAD taken over the present values."""
    base = r[~missing]
    med = np.median(base)
    ref = base if scale_of is None else scale_of[~missing]
    scale = 1.4826 * np.median(np.abs(ref - np.median(ref)))
    dev = np.abs(r - med)
    if scale > 0:
        return dev / scale
    return np.where(dev > 0, np.inf, 0.0)  # a perfect fit: any departure is unusual


def _minimum(period: int | None) -> int:
    return 2 * period if period else 7


def _fit(
    y: np.ndarray, period: int | None, seasonal: int, trend: int | None, *, robust: bool
) -> np.ndarray:
    if period:
        from statsmodels.tsa.seasonal import STL

        stl = STL(y, period=period, seasonal=seasonal, trend=trend, seasonal_deg=0, robust=robust)
        res = stl.fit()
        return np.asarray(res.trend) + np.asarray(res.seasonal)
    from statsmodels.nonparametric.smoothers_lowess import lowess

    frac = min(1.0, (trend or 21) / len(y))
    t = np.arange(len(y), dtype=float)
    return lowess(y, t, frac=frac, it=3 if robust else 0, return_sorted=False)


@register("outliers")
class Outliers:
    """No value is an outlier: its robust z-score on STL residuals is at most `z`.

    See `outlier_scores`. The units are the present values of a series
    with at least two years of history.
    """

    class Config(CheckConfig):
        warning: float | None = 1
        error: float = 0.05
        z: float = Field(4.0, gt=0, description="Largest accepted robust z-score.")
        seasonal: int = Field(13, ge=3, description="STL seasonal smoother, in years; odd.")
        trend: int | None = Field(
            None, ge=3, description="Trend smoother in periods; odd; unset: STL's default."
        )

        @model_validator(mode="after")
        def _odd(self) -> Outliers.Config:
            for name in ("seasonal", "trend"):
                value = getattr(self, name)
                if value is not None and value % 2 == 0:
                    raise ValueError(f"{name} must be odd; got {value}")
            return self

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        minimum = _minimum(_PERIOD.get(context.frequency or ""))
        if len(_span(x)) < minimum:
            return Units.not_applicable(
                f"the series is too short: it needs at least {minimum} periods "
                "from its first to its last value"
            )
        z = outlier_scores(
            x, context.frequency, z=config.z, seasonal=config.seasonal, trend=config.trend
        )
        return Units(values=z, test="at_most", bound=config.z, measure="robust z-score")


class Downturn(Config):
    """A period the history must cover, such as a recession, named in configuration."""

    label: str = Field(min_length=1)
    start: str = Field(min_length=1, description="First period, as text: 2008Q1, 2008-03, 2008.")
    end: str = Field(min_length=1, description="Last period, as text.")


@register("representativeness")
class Representativeness:
    """The history covers every configured downturn: each of its periods has a value.

    `clean` applies it to targets and drivers (`roles`) unless `columns`
    names other series. The units are the downturn periods, whether or not the calendar
    reaches them.
    """

    roles = ("target", "driver")

    class Config(CheckConfig):
        downturns: list[Downturn] = Field(default_factory=list)

    def units(self, x: pd.Series, config: Any, context: Context) -> Units:
        if not config.downturns:
            return Units.not_applicable("no downturn is configured")
        freq = x.index.freq
        periods: list[pd.Period] = []
        for d in config.downturns:
            try:
                start, end = pd.Period(d.start, freq=freq), pd.Period(d.end, freq=freq)
            except (ValueError, TypeError) as err:
                raise CheckError(
                    f"the downturn {d.label!r} has a period that cannot be read",
                    observed=f"start {d.start!r}, end {d.end!r}, data {x.index.freqstr}",
                    change="Write the periods as the data's own: 2008Q1, 2008-03 or 2008.",
                ) from err
            if start > end:
                raise CheckError(
                    f"the downturn {d.label!r} ends before it starts",
                    observed=f"start {d.start}, end {d.end}",
                    change="Swap the start and the end.",
                )
            periods.extend(pd.period_range(start, end, freq=freq))
        index = pd.PeriodIndex(sorted(set(periods)), freq=freq)
        return Units(values=x.reindex(index), test="not_null", measure="value")
