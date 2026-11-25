"""explore: each series alone, each driver against the target (design §7.3).

Every numeric series is profiled (trend and seasonal strength from STL,
autocorrelation, volatility, intermittency) and tested for stationarity
with ADF, DFGLS, Phillips-Perron and KPSS from arch, read together into an
order of integration. Each series is then transformed by its configured
chain and the differences that order calls for, and each driver is
searched for its lag against the target and screened by a single-driver
OLS regression with HAC standard errors. The gate keeps a driver with the
expected sign and a p-value at most the significance threshold.

>>> import numpy as np, pandas as pd
>>> import dazflow as dz
>>> rng = np.random.default_rng(0)
>>> x = rng.normal(size=120)
>>> y = 0.8 * np.roll(x, 1) + 0.3 * rng.normal(size=120)
>>> frame = pd.DataFrame({"y": y, "x": x},
...                      index=pd.period_range("1990Q1", periods=120, freq="Q")).iloc[1:]
>>> ds = dz.Dataset(frame, {"y": {"role": "target"},
...                         "x": {"role": "driver", "expected_sign": 1}})
>>> r = dz.explore.run(ds, {"max_lag": 4})
>>> r.lags, r.kept, r.gate.passed
({'x': 1}, ('x',), True)
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field

from dazflow import transforms
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import ExploreError, ResultError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import StepResult
from dazflow.readers._base import frequency_of

__all__ = [
    "IDS",
    "Config",
    "ExploreConfig",
    "ExploreResult",
    "Integration",
    "LagProfile",
    "Level",
    "Profile",
    "ProfileConfig",
    "Result",
    "Screen",
    "StationarityConfig",
    "UnitRoot",
    "driver_checks",
    "integration_order",
    "lag_search",
    "newey_west_lags",
    "profile",
    "run",
    "says_stationary",
    "screen",
    "unit_root",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "explore",
    tables={"profile": 1, "stationarity": 2, "screen": 3, "gate": 4},
    figures={"lags": 1},
    findings={"summary": 1},
)

SUMMARY = (
    "{{ series }} series explored; {{ differenced }} were differenced to stationarity. "
    "{{ kept }} of {{ drivers }} drivers are kept, {{ overridden }} of them by override: "
    "each has its expected sign and a p-value of at most {{ significance }}."
)

_PERIOD = {"M": 12, "Q": 4}
_PERIOD_NAME = {"M": "Month", "Q": "Quarter", "Y": "Year"}
_DIFF = "diff(lag=1)"
_TESTS = ("adf", "dfgls", "pp", "kpss")
_TEST_NAMES = {"adf": "ADF", "dfgls": "DFGLS", "pp": "Phillips-Perron", "kpss": "KPSS"}


# --- configuration ---------------------------------------------------------------


class ProfileConfig(Config):
    """How each series is profiled."""

    seasonal: int = Field(
        7, ge=3, description="STL seasonal smoother length, in periods (odd, at least 3)."
    )
    trend_window: int = Field(
        21, ge=3, description="Annual series: the lowess trend's window, in years."
    )


class StationarityConfig(Config):
    """How the four tests are run and read together into an order of integration.

    A level of differencing is stationary when at least `votes` of the
    four tests say so at significance `alpha`: ADF, DFGLS and
    Phillips-Perron by rejecting their unit-root null, KPSS by not
    rejecting its stationarity null. Levels are tried in order: the series
    around a constant, then around a linear trend, then its first, second
    ... differences around a constant, up to `max_order`.
    """

    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level of every test.")
    votes: int = Field(3, ge=1, le=4, description="Tests of four that must say stationary.")
    max_order: int = Field(2, ge=0, le=3, description="Most differences tried.")
    lags: int | None = Field(
        None,
        ge=0,
        description="Lags (ADF, DFGLS) and bandwidth (PP, KPSS) of every test; unset: chosen "
        "by `method` for ADF and DFGLS, arch's default bandwidth for PP and KPSS.",
    )
    method: Literal["aic", "bic", "t-stat"] = "aic"
    trend_stationary: Literal["difference", "keep"] = Field(
        "difference",
        description="A trend-stationary series is differenced once, or kept in levels.",
    )
    min_observations: int = Field(
        20, ge=8, description="Fewest values a series needs to be tested."
    )


class ExploreConfig(Config):
    """Configuration of the explore step."""

    target: str | None = Field(None, description="The target; unset: the dataset's one target.")
    drivers: list[str] | None = Field(
        None, description="The drivers screened, in this order; unset: the dataset's drivers."
    )
    expected_signs: dict[str, Literal[-1, 0, 1]] = Field(
        default_factory=dict,
        description="Expected sign by driver; replaces the series metadata. 0: no sign check.",
    )
    transforms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="A chain applied before the stationarity tests, by series (e.g. log).",
    )
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    stationarity: StationarityConfig = Field(default_factory=StationarityConfig)
    max_lag: int = Field(4, ge=0, description="Lags searched: 0 to max_lag periods.")
    significance: float = Field(
        0.05, gt=0, lt=1, description="A driver is kept with a HAC p-value at most this."
    )
    hac_maxlags: int | None = Field(
        None, ge=0, description="HAC lags; unset: floor(4 (n/100)^(2/9)) (Newey-West)."
    )
    overrides: list[Override] = Field(default_factory=list)


# --- result objects --------------------------------------------------------------


class _Record(DazflowModel):
    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "These records are made by the explore step."


class Profile(_Record):
    """One series' profile, on its values from the first to the last present one.

    Strengths are ``max(0, 1 - Var(R) / Var(T + R))`` for the trend and
    ``max(0, 1 - Var(R) / Var(S + R))`` for the season, from an STL fit
    (Wang, Smith and Hyndman 2006); annual series have a lowess trend and
    no seasonal measure. `acf1` and `acf_seasonal` are sample
    autocorrelations at lag 1 and at a year; `volatility` is the standard
    deviation of period-on-period changes; `intermittency` the share of
    present values that are zero. Missing values inside the history are
    interpolated for the fit and counted in `missing`.
    """

    series: str
    observations: int
    missing: int
    trend_strength: float | None = None
    seasonal_strength: float | None = None
    acf1: float | None = None
    acf_seasonal: float | None = None
    volatility: float | None = None
    intermittency: float | None = None


class UnitRoot(_Record):
    """One test on one level of differencing. `stationary` reads it at `alpha`.

    `statistic`, `pvalue` and `lags` are ``None`` when arch could not run
    the test (`note` says why); such a test does not say stationary.
    """

    test: Literal["adf", "dfgls", "pp", "kpss"]
    trend: Literal["c", "ct"]
    statistic: float | None = None
    pvalue: float | None = None
    lags: int | None = None
    stationary: bool = False
    note: str | None = None


class Level(_Record):
    """The four tests on the series differenced `differences` times, around `trend`."""

    differences: int
    trend: Literal["c", "ct"]
    tests: tuple[UnitRoot, ...]
    votes: int
    stationary: bool


class Integration(_Record):
    """A series' order of integration and the levels tested to reach it.

    `kind` is ``stationary`` (order 0), ``trend-stationary`` (order 0
    around a trend), ``integrated`` (order 1 or more) or ``undetermined``
    (not stationary within `max_order` differences; `order` is ``None``).
    `chain` holds the differences chosen, as transform text.
    """

    series: str | None = None
    order: int | None
    kind: Literal["stationary", "trend-stationary", "integrated", "undetermined"]
    steps: tuple[Level, ...]
    chain: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """``I(0)``, ``I(1)``, ``trend-stationary`` or ``undetermined``."""
        if self.kind in ("trend-stationary", "undetermined"):
            return self.kind
        return f"I({self.order})"


class LagProfile(_Record):
    """Correlations of the target with a driver lagged 0 to `max_lag` periods.

    All lags are measured on one sample, the periods where the target and
    every lag of the driver are present, so they compare. `lag` has the
    largest absolute correlation (the smallest such lag on a tie).
    """

    driver: str | None = None
    lag: int
    correlations: tuple[float | None, ...]
    observations: int


class Screen(_Record):
    """A single-driver regression, target on a constant and the driver at `lag`.

    Standard errors are HAC (Newey-West, Bartlett kernel, `maxlags`), as
    statsmodels' ``OLS(...).fit(cov_type="HAC")``. `expected_sign`,
    `observed_sign` and `kept` are filled by `run`.
    """

    driver: str | None = None
    lag: int
    coefficient: float | None = None
    std_error: float | None = None
    t: float | None = None
    pvalue: float | None = None
    observations: int
    maxlags: int
    expected_sign: int | None = None
    observed_sign: int | None = None
    kept: bool = False


class ExploreResult(StepResult):
    """What explore returns. `dataset` (left out of `to_dict()`) holds the transformed series.

    `chains` is the chain explore applied to each series (configured
    transforms, then differences), also appended to its
    `SeriesMeta.transforms`; `orders` its order of integration; `lags`
    each driver's lag; `kept` the drivers whose gate checks pass or are
    overridden, in the configured order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ExploreConfig
    dataset: Dataset | None = Field(None, exclude=True)
    fingerprint: str | None = None
    input_fingerprint: str | None = None
    target: str | None = None
    drivers: tuple[str, ...] = ()
    profiles: tuple[Profile, ...] = ()
    integration: tuple[Integration, ...] = ()
    orders: dict[str, int | None] = Field(default_factory=dict)
    chains: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    lag_profiles: tuple[LagProfile, ...] = ()
    lags: dict[str, int] = Field(default_factory=dict)
    screens: tuple[Screen, ...] = ()
    kept: tuple[str, ...] = ()


Config = ExploreConfig
Result = ExploreResult


def _fail(problem: str, observed: str, change: str) -> ExploreError:
    return ExploreError(problem, observed=observed, change=change)


def _finite(v: Any) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _span(x: pd.Series) -> pd.Series:
    """`x` from its first to its last value, as floats (empty when it has none)."""
    x = pd.Series(x, dtype="float64")
    first, last = x.first_valid_index(), x.last_valid_index()
    if first is None:
        return x.iloc[:0]
    return x.loc[first:last]


# --- profile ---------------------------------------------------------------------


def _strength(part: np.ndarray, resid: np.ndarray) -> float | None:
    total = np.var(part + resid)
    if not total > 0:
        return None
    return max(0.0, 1.0 - float(np.var(resid) / total))


def _acf(d: np.ndarray, k: int) -> float | None:
    denom = float(np.sum(d * d))
    if k >= len(d) or not denom > 0:
        return None
    return float(np.sum(d[k:] * d[:-k]) / denom)


def profile(
    x: pd.Series,
    frequency: str | None = None,
    config: ProfileConfig | Mapping[str, Any] | None = None,
) -> Profile:
    """Profile one series: trend and seasonal strength, autocorrelation, volatility, intermittency.

    `frequency` is ``"M"``, ``"Q"`` or ``"Y"`` (unset: read from the index).
    A measure the series is too short for is ``None``.
    """
    cfg = (
        config if isinstance(config, ProfileConfig) else ProfileConfig.model_validate(config or {})
    )
    if frequency is None and isinstance(getattr(x, "index", None), pd.PeriodIndex):
        frequency = frequency_of(x.index)
    span = _span(x)
    present = span.dropna().to_numpy()
    out: dict[str, Any] = {
        "series": str(x.name) if getattr(x, "name", None) is not None else "x",
        "observations": len(present),
        "missing": int(span.isna().sum()),
    }
    if len(present) == 0:
        return Profile(**out)
    y = span.interpolate(method="linear").to_numpy(dtype=float)
    period = _PERIOD.get(frequency or "")
    varies = bool(np.ptp(y) > 0)  # a fit to a constant leaves only rounding noise
    if varies and period and len(y) > 2 * period:
        from statsmodels.tsa.seasonal import STL

        fit = STL(y, period=period, seasonal=cfg.seasonal).fit()
        trend, seasonal, resid = (np.asarray(a) for a in (fit.trend, fit.seasonal, fit.resid))
        out["trend_strength"] = _strength(trend, resid)
        out["seasonal_strength"] = _strength(seasonal, resid)
    elif varies and not period and len(y) >= 3:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        t = np.arange(len(y), dtype=float)
        trend = lowess(y, t, frac=min(1.0, cfg.trend_window / len(y)), return_sorted=False)
        out["trend_strength"] = _strength(trend, y - trend)
    d = y - y.mean()
    out["acf1"] = _acf(d, 1)
    if period:
        out["acf_seasonal"] = _acf(d, period)
    if len(y) > 2:
        out["volatility"] = _finite(np.std(np.diff(y), ddof=1))
    out["intermittency"] = float(np.mean(present == 0))
    return Profile(**out)


# --- stationarity ----------------------------------------------------------------


def unit_root(
    x: pd.Series | Sequence[float],
    *,
    trend: Literal["c", "ct"] = "c",
    lags: int | None = None,
    method: Literal["aic", "bic", "t-stat"] = "aic",
    alpha: float = 0.05,
) -> tuple[UnitRoot, ...]:
    """ADF, DFGLS, Phillips-Perron and KPSS (arch) on `x`, around a constant or a trend.

    `lags` fixes the lags of ADF and DFGLS and the bandwidth of PP and
    KPSS; unset, ADF and DFGLS choose by `method` and PP and KPSS take
    arch's default bandwidth. The lags used are recorded. A unit-root test
    says stationary when its p-value is below `alpha`; KPSS when its
    p-value is at least `alpha`.
    """
    from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron
    from arch.utility.exceptions import InfeasibleTestException

    y = _span(pd.Series(x)).interpolate(method="linear").to_numpy(dtype=float)
    if not len(y) or not np.ptp(y) > 0:
        # No test means anything on a constant series; arch 7.0's KPSS still
        # returns a statistic, later releases raise.
        return tuple(UnitRoot(test=k, trend=trend, note="the series is constant") for k in _TESTS)
    classes = {"adf": ADF, "dfgls": DFGLS, "pp": PhillipsPerron, "kpss": KPSS}
    out = []
    for key in _TESTS:
        kwargs: dict[str, Any] = {"lags": lags, "trend": trend}
        if key in ("adf", "dfgls") and lags is None:
            kwargs["method"] = method
        try:
            with warnings.catch_warnings():
                # arch 7.0 announces that KPSS's automatic bandwidth became data
                # dependent (the method used here); later releases do not warn.
                warnings.filterwarnings("ignore", message="Lag selection has changed")
                res = classes[key](y, **kwargs)
                stat, p, used = float(res.stat), float(res.pvalue), int(res.lags)
        except (InfeasibleTestException, ValueError, np.linalg.LinAlgError) as err:
            # Constant or too short input (InfeasibleTestException is a ValueError
            # from arch 7.1 on, a RuntimeError only before).
            out.append(
                UnitRoot(test=key, trend=trend, note=str(err).splitlines()[0] or type(err).__name__)
            )
            continue
        finite = math.isfinite(stat) and math.isfinite(p)
        says = finite and says_stationary(key, p, alpha)
        out.append(
            UnitRoot(
                test=key,
                trend=trend,
                statistic=_finite(stat),
                pvalue=_finite(p),
                lags=used,
                stationary=bool(says),
                note=None if finite else "the test gave no finite statistic or p-value",
            )
        )
    return tuple(out)


def says_stationary(test: str, pvalue: float, alpha: float) -> bool:
    """Whether a test's p-value says stationary at `alpha`.

    ADF, DFGLS and Phillips-Perron test a unit-root null, so they say
    stationary when ``pvalue < alpha``; KPSS tests a stationarity null, so
    it says stationary when ``pvalue >= alpha``. A NaN p-value says nothing.
    """
    if not math.isfinite(pvalue):
        return False
    return pvalue >= alpha if test == "kpss" else pvalue < alpha


def _level(y: pd.Series, d: int, trend: Literal["c", "ct"], cfg: StationarityConfig) -> Level:
    tests = unit_root(y, trend=trend, lags=cfg.lags, method=cfg.method, alpha=cfg.alpha)
    votes = sum(t.stationary for t in tests)
    return Level(
        differences=d, trend=trend, tests=tests, votes=votes, stationary=votes >= cfg.votes
    )


def integration_order(
    x: pd.Series, config: StationarityConfig | Mapping[str, Any] | None = None
) -> Integration:
    """The order of integration of `x`, from the four tests read together (`StationarityConfig`).

    Raises `ExploreError` when `x` has fewer than `min_observations`
    values from its first to its last.
    """
    cfg = (
        config
        if isinstance(config, StationarityConfig)
        else StationarityConfig.model_validate(config or {})
    )
    name = str(x.name) if getattr(x, "name", None) is not None else None
    cur = _span(pd.Series(x))
    if int(cur.notna().sum()) < cfg.min_observations:
        raise _fail(
            f"the series {name or 'x'!r} is too short to test for stationarity",
            f"{int(cur.notna().sum())} values",
            f"Test a series with at least {cfg.min_observations} values, or lower "
            "stationarity.min_observations.",
        )
    steps: list[Level] = []
    for d in range(cfg.max_order + 1):
        if d:
            cur = cur.diff().iloc[1:]
        level = _level(cur, d, "c", cfg)
        steps.append(level)
        if level.stationary:
            kind = "stationary" if d == 0 else "integrated"
            return Integration(
                series=name, order=d, kind=kind, steps=tuple(steps), chain=(_DIFF,) * d
            )
        if d == 0:
            level = _level(cur, 0, "ct", cfg)
            steps.append(level)
            if level.stationary:
                chain = (_DIFF,) if cfg.trend_stationary == "difference" else ()
                return Integration(
                    series=name, order=0, kind="trend-stationary", steps=tuple(steps), chain=chain
                )
    return Integration(
        series=name,
        order=None,
        kind="undetermined",
        steps=tuple(steps),
        chain=(_DIFF,) * cfg.max_order,
    )


# --- lags and screen ---------------------------------------------------------------


def _name(x: Any) -> str | None:
    return str(x.name) if getattr(x, "name", None) is not None else None


def lag_search(y: pd.Series, x: pd.Series, max_lag: int) -> LagProfile:
    """Correlate `y` with `x` lagged 0 to `max_lag` periods and pick the strongest lag.

    `y` and `x` share an index (consecutive periods). Raises `ExploreError`
    when fewer than three periods have the target and every lag present.
    """
    if max_lag < 0:
        raise _fail(
            "max_lag must be 0 or more", f"max_lag={max_lag}", "Search lags from 0 upwards."
        )
    cols = {"y": pd.Series(y, dtype="float64")}
    x = pd.Series(x, dtype="float64")
    for lag in range(max_lag + 1):
        cols[lag] = x.shift(lag)
    data = pd.DataFrame(cols).dropna()
    if len(data) < 3:
        raise _fail(
            f"too few periods to search the lags of {_name(x) or 'the driver'!r}",
            f"{len(data)} periods with the target and lags 0 to {max_lag} present",
            "Lower max_lag, or give the driver and target a longer common history.",
        )
    corr = [_finite(data["y"].corr(data[lag])) for lag in range(max_lag + 1)]
    strength = [abs(c) if c is not None else -1.0 for c in corr]
    return LagProfile(
        driver=_name(x),
        lag=int(np.argmax(strength)),
        correlations=tuple(corr),
        observations=len(data),
    )


def newey_west_lags(n: int) -> int:
    """The Newey-West (1994) rule of thumb, floor(4 (n / 100)^(2/9))."""
    return math.floor(4 * (n / 100) ** (2 / 9))


def screen(y: pd.Series, x: pd.Series, lag: int = 0, maxlags: int | None = None) -> Screen:
    """Regress `y` on a constant and `x` lagged `lag` periods, with HAC standard errors.

    The sample is every period with both values present; `maxlags`
    defaults to `newey_west_lags` of its size. A driver with no variation
    gets no coefficient.
    """
    import statsmodels.api as sm

    data = pd.DataFrame(
        {"y": pd.Series(y, dtype="float64"), "x": pd.Series(x, dtype="float64").shift(lag)}
    ).dropna()
    n = len(data)
    if n < 3:
        raise _fail(
            f"too few periods to screen {_name(x) or 'the driver'!r}",
            f"{n} periods with the target and the driver at lag {lag} present",
            "Give the driver and target a longer common history.",
        )
    m = newey_west_lags(n) if maxlags is None else maxlags
    out: dict[str, Any] = {"driver": _name(x), "lag": lag, "observations": n, "maxlags": m}
    xv = data["x"].to_numpy()
    if not np.ptp(xv) > 0:
        return Screen(**out)
    design = np.column_stack([np.ones(n), xv])
    fit = sm.OLS(data["y"].to_numpy(), design).fit(cov_type="HAC", cov_kwds={"maxlags": m})
    out.update(
        coefficient=_finite(fit.params[1]),
        std_error=_finite(fit.bse[1]),
        t=_finite(fit.tvalues[1]),
        pvalue=_finite(fit.pvalues[1]),
    )
    return Screen(**out)


# --- the step ----------------------------------------------------------------------


def _numeric(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _roles(cfg: ExploreConfig, ds: Dataset, numeric: list[str]) -> tuple[str | None, list[str]]:
    listed = f"numeric series: {', '.join(numeric) or 'none'}"
    if cfg.target is not None:
        if cfg.target not in numeric:
            raise _fail(
                f"the target {cfg.target!r} is not a numeric series of the dataset",
                listed,
                "Name a numeric series in target.",
            )
        target = cfg.target
    else:
        targets = [c for c in ds.targets if c in numeric]
        if len(targets) > 1:
            raise _fail(
                "the dataset has several targets and the configuration names none",
                f"targets: {', '.join(targets)}",
                "Name the target to screen the drivers against in target.",
            )
        target = targets[0] if targets else None
    if cfg.drivers is not None:
        unknown = [d for d in cfg.drivers if d not in numeric or d == target]
        if unknown or len(set(cfg.drivers)) != len(cfg.drivers):
            raise _fail(
                "drivers names series that are not numeric, are the target or repeat: "
                f"{', '.join(unknown) or 'a repeated name'}",
                listed,
                "Name each driver once, from the dataset's numeric series other than the target.",
            )
        drivers = list(cfg.drivers)
    else:
        drivers = [c for c in ds.drivers if c in numeric and c != target]
    if drivers and target is None:
        raise _fail(
            "drivers cannot be screened without a target",
            f"drivers: {', '.join(drivers)}; no series has role target",
            "Give one series role target, or name it in target.",
        )
    stray = [d for d in cfg.expected_signs if d not in drivers]
    if stray:
        raise _fail(
            f"an expected sign is given for series that are not drivers: {', '.join(stray)}",
            f"drivers: {', '.join(drivers) or 'none'}",
            "Give expected signs for the drivers only.",
        )
    unknown = [c for c in cfg.transforms if c not in numeric]
    if unknown:
        raise _fail(
            f"transforms names series the dataset does not have as numbers: {', '.join(unknown)}",
            listed,
            "Name numeric series of the dataset in transforms.",
        )
    return target, drivers


def _sign(v: float | None) -> int | None:
    return None if v is None else int(np.sign(v))


def run(dataset: Dataset, config: ExploreConfig | Mapping[str, Any] | None = None) -> ExploreResult:
    """Profile and test every numeric series, transform each, and screen each driver.

    The caller's `dataset` is not changed; the result's `dataset` holds
    the transformed series with their chains recorded. Prints nothing and
    writes no files.
    """
    cfg = (
        config if isinstance(config, ExploreConfig) else ExploreConfig.model_validate(config or {})
    )
    if dataset.is_panel:
        raise _fail(
            "explore does not take panel datasets yet",
            f"a panel of {len(dataset.segments)} segments",
            "Explore each segment's dataset on its own.",
        )
    frame = dataset.frame
    freq = frequency_of(frame.index)
    numeric = _numeric(frame)
    target, drivers = _roles(cfg, dataset, numeric)

    profiles, integration, chains, orders = [], [], {}, {}
    out = dataset
    for name in numeric:
        level = frame[name].astype("float64").rename(name)
        profiles.append(profile(level, freq, cfg.profile))
        base = cfg.transforms.get(name, [])
        tested = transforms.forward(level, base)[0] if base else level
        integ = integration_order(tested.rename(name), cfg.stationarity)
        integration.append(integ)
        orders[name] = integ.order
        chain = [*base, *integ.chain]
        before = len(out.series[name].transforms)
        if chain:
            out = transforms.apply(out, name, chain)
        chains[name] = tuple(out.series[name].transforms[before:])

    lag_profiles, screens, checks = [], [], []
    moved = out.frame
    for d in drivers:
        y, x = moved[target], moved[d].rename(d)
        lp = lag_search(y, x, cfg.max_lag)
        lag_profiles.append(lp)
        s = screen(y, x, lp.lag, cfg.hac_maxlags)
        sign = cfg.expected_signs.get(d, dataset.series[d].expected_sign)
        s = s.model_copy(update={"expected_sign": sign, "observed_sign": _sign(s.coefficient)})
        screens.append(s)
        checks.extend(driver_checks(s, cfg.significance))
    gate = Gate.evaluate(checks, cfg.overrides)
    status = {d: _status(gate, d) for d in drivers}
    kept = tuple(d for d in drivers if status[d] != "fail")
    screens = [s.model_copy(update={"kept": s.driver in kept}) for s in screens]

    finding = Finding(
        template=SUMMARY,
        key="explore.summary",
        values={
            "series": len(numeric),
            "differenced": sum(1 for c in chains.values() if _DIFF in c),
            "kept": len(kept),
            "drivers": len(drivers),
            "overridden": sum(1 for d in kept if status[d] == "overridden"),
            "significance": cfg.significance,
        },
    )
    artefacts: dict[str, Any] = {
        IDS.table("profile"): _profile_table(profiles, integration, chains, dataset),
        IDS.table("stationarity"): _stationarity_table(integration, cfg.stationarity),
        IDS.table("screen"): _screen_table(screens, lag_profiles, status, cfg),
        IDS.table("gate"): Table.from_gate(
            gate, id=IDS.table("gate"), caption="Driver screen checks"
        ),
        IDS.finding("summary"): finding,
    }
    if drivers:
        artefacts[IDS.figure("lags")] = _lag_figure(lag_profiles, target, freq)

    return ExploreResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        dataset=out,
        fingerprint=out.fingerprint(),
        input_fingerprint=dataset.fingerprint(),
        target=target,
        drivers=tuple(drivers),
        profiles=tuple(profiles),
        integration=tuple(integration),
        orders=orders,
        chains=chains,
        lag_profiles=tuple(lag_profiles),
        lags={p.driver: p.lag for p in lag_profiles if p.driver is not None},
        screens=tuple(screens),
        kept=kept,
    )


def driver_checks(screen: Screen, significance: float) -> list[Check]:
    """The gate checks of one screened driver, before overrides.

    ``sign:<driver>`` (only when `expected_sign` is -1 or 1) passes when
    the coefficient times the expected sign is above 0;
    ``significance:<driver>`` passes when the HAC p-value is at most
    `significance`. A missing coefficient or p-value fails.
    """
    d = screen.driver
    checks = []
    if screen.expected_sign in (-1, 1):
        coef = screen.coefficient
        observed = coef * screen.expected_sign if coef is not None else math.nan
        checks.append(Check(name=f"sign:{d}", observed=observed, threshold=0.0, op=">"))
    pvalue = screen.pvalue if screen.pvalue is not None else math.nan
    checks.append(Check(name=f"significance:{d}", observed=pvalue, threshold=significance, op="<="))
    return checks


def _status(gate: Gate, driver: str) -> str:
    mine = [c for c in gate.checks if c.name in (f"sign:{driver}", f"significance:{driver}")]
    if any(c.status == "fail" for c in mine):
        return "fail"
    return "overridden" if any(c.status == "overridden" for c in mine) else "pass"


# --- artefacts ---------------------------------------------------------------------


def _profile_table(
    profiles: list[Profile],
    integration: list[Integration],
    chains: Mapping[str, tuple[str, ...]],
    ds: Dataset,
) -> Table:
    columns = [
        Column(name="series", label="Series", kind="text"),
        Column(name="role", label="Role", kind="text"),
        Column(name="observations", label="Observations", kind="integer"),
        Column(name="missing", label="Missing inside", kind="integer"),
        Column(name="trend", label="Trend strength", kind="number"),
        Column(name="seasonal", label="Seasonal strength", kind="number"),
        Column(name="acf1", label="Autocorrelation, lag 1", kind="number"),
        Column(name="acf_seasonal", label="Autocorrelation, a year", kind="number"),
        Column(name="volatility", label="Volatility", kind="number"),
        Column(name="intermittency", label="Zero values", kind="percent"),
        Column(name="order", label="Integration", kind="text"),
        Column(name="chain", label="Transforms", kind="text"),
    ]
    rows = [
        [
            p.series,
            ds.series[p.series].role,
            p.observations,
            p.missing,
            p.trend_strength,
            p.seasonal_strength,
            p.acf1,
            p.acf_seasonal,
            p.volatility,
            p.intermittency,
            i.label,
            ", ".join(chains[p.series]) or "none",
        ]
        for p, i in zip(profiles, integration, strict=True)
    ]
    return Table(
        id=IDS.table("profile"),
        caption="Series profile",
        columns=columns,
        rows=rows,
        notes=(
            "Strengths are 1 - Var(remainder) / Var(component + remainder) of an STL fit, "
            "floored at 0; annual series have a lowess trend and no season.",
            "Volatility is the standard deviation of period-on-period changes, in the "
            "series' unit.",
        ),
    )


def _stationarity_table(integration: list[Integration], cfg: StationarityConfig) -> Table:
    columns = [
        Column(name="series", label="Series", kind="text"),
        Column(name="differences", label="Differences", kind="integer"),
        Column(name="around", label="Around", kind="text"),
    ]
    for key in _TESTS:
        columns += [
            Column(name=f"{key}_stat", label=f"{_TEST_NAMES[key]} statistic", kind="number"),
            Column(name=f"{key}_p", label=f"{_TEST_NAMES[key]} p-value", kind="number"),
        ]
    columns += [
        Column(name="votes", label="Tests saying stationary", kind="integer"),
        Column(name="stationary", label="Stationary", kind="text"),
    ]
    rows, notes = [], []
    for i in integration:
        for level in i.steps:
            row: list[Any] = [
                i.series,
                level.differences,
                "trend" if level.trend == "ct" else "constant",
            ]
            for t in level.tests:
                row += [t.statistic, t.pvalue]
                if t.note:
                    notes.append(
                        f"{_TEST_NAMES[t.test]} could not be run on {i.series} "
                        f"({level.differences} differences): {t.note}"
                    )
            rows.append([*row, level.votes, "yes" if level.stationary else "no"])
    rule = (
        f"A level is stationary when at least {cfg.votes} of the four tests say so at the "
        f"{cfg.alpha:g} level: ADF, DFGLS and Phillips-Perron by rejecting a unit root, KPSS "
        "by not rejecting stationarity."
    )
    return Table(
        id=IDS.table("stationarity"),
        caption="Stationarity tests",
        columns=columns,
        rows=rows,
        notes=(rule, *notes),
    )


_SIGN_TEXT = {-1: "negative", 0: "none", 1: "positive", None: "none"}


def _screen_table(
    screens: list[Screen],
    lag_profiles: list[LagProfile],
    status: Mapping[str, str],
    cfg: ExploreConfig,
) -> Table:
    columns = [
        Column(name="driver", label="Driver", kind="text"),
        Column(name="lag", label="Lag", kind="integer"),
        Column(name="correlation", label="Correlation", kind="number"),
        Column(name="coefficient", label="Coefficient", kind="number"),
        Column(name="std_error", label="Standard error (HAC)", kind="number"),
        Column(name="t", label="t", kind="number"),
        Column(name="pvalue", label="p-value", kind="number"),
        Column(name="observations", label="Observations", kind="integer"),
        Column(name="expected", label="Expected sign", kind="text"),
        Column(name="observed", label="Observed sign", kind="text"),
        Column(name="result", label="Result", kind="status"),
    ]
    rows = [
        [
            s.driver,
            s.lag,
            p.correlations[s.lag],
            s.coefficient,
            s.std_error,
            s.t,
            s.pvalue,
            s.observations,
            _SIGN_TEXT[s.expected_sign],
            _SIGN_TEXT[s.observed_sign],
            status[s.driver],
        ]
        for s, p in zip(screens, lag_profiles, strict=True)
    ]
    return Table(
        id=IDS.table("screen"),
        caption="Driver screen",
        columns=columns,
        rows=rows,
        notes=(
            f"Each driver is regressed alone on the transformed target at the lag of strongest "
            f"correlation (0 to {cfg.max_lag}); it is kept with its expected sign and a p-value "
            f"of at most {cfg.significance:g}.",
        ),
    )


def _lag_figure(lag_profiles: list[LagProfile], target: str | None, freq: str | None) -> Figure:
    k = len(lag_profiles[0].correlations)
    layers = [Layer(kind="line", y=p.correlations, label=p.driver) for p in lag_profiles]
    layers.append(Layer(kind="hline", value=0.0))
    unit = _PERIOD_NAME.get(freq or "", "Period").lower() + "s"
    return Figure(
        id=IDS.figure("lags"),
        caption=f"Lag profiles: correlation of each driver with {target}, by lag",
        x=list(range(k)),
        layers=layers,
        x_label="Lag",
        x_unit=unit,
        y_label="Correlation",
    )
