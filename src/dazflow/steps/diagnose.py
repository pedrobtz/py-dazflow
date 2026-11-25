"""diagnose: the chosen model tested against its assumptions and against history (design §7.8).

Select's chosen model is refitted from its record on the series in levels
and tested five ways:

- **residuals**: Ljung-Box, Breusch-Godfrey, Breusch-Pagan, White, ARCH,
  Jarque-Bera and RESET, each a p-value against a configured ``alpha``;
- **stability**: the OLS-residual CUSUM test, the recursive CUSUM against
  Brown, Durbin and Evans' bounds, rolling coefficients and Chow tests at
  configured or candidate break dates;
- **back-testing**: multi-step forecasts from past origins against the
  actual values, through `dz.evaluation.evaluate`, with the interval's
  coverage against a configured minimum;
- **sensitivity**: one at a time, the forecast's response to a shock in
  each driver, and to each driver's scenario path alone;
- **anomalies**: observations outside the model's interval at a configured
  level, their share against a configured maximum.

A test that does not apply to the chosen form (a benchmark, or a
regression-based test of a form that is not OLS) is recorded as not
applicable, with its reason, and is not in the gate.

Gate: every applicable diagnostic within its threshold.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from dazflow import evaluation, models, readers, transforms
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import (
    DiagnoseError,
    EvaluationError,
    ModelError,
    ReaderError,
    ResultError,
)
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Comparison, Gate, Override
from dazflow.core.result import StepResult
from dazflow.evaluation import Evaluation
from dazflow.models import Design, ModelRecord

__all__ = [
    "IDS",
    "Anomaly",
    "AnomalyConfig",
    "BacktestConfig",
    "ChowConfig",
    "ChowTest",
    "Config",
    "Cusum",
    "CusumConfig",
    "DiagnoseConfig",
    "DiagnoseResult",
    "Diagnostic",
    "LagTest",
    "PValueTest",
    "ResetConfig",
    "Result",
    "Rolling",
    "RollingConfig",
    "Sensitivity",
    "SensitivityConfig",
    "anomalies",
    "chow",
    "default_lags",
    "residual_tests",
    "run",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "diagnose",
    tables={"diagnostics": 1, "sensitivity": 2, "anomalies": 3, "backtest": 4, "gate": 5},
    figures={"residuals": 1, "cusum": 2, "rolling": 3},
    findings={"summary": 1},
)

SUMMARY = (
    "{{ chosen }} passes {{ passed }} of {{ checks }} diagnostic checks"
    "{% if failed %} and fails {{ failed }}{% endif %}"
    "{% if overridden %}; {{ overridden }} failed and {{ 'is' if overridden_count == 1 else 'are' }} "
    "accepted by override{% endif %}"
    "{% if inapplicable %}; {{ inapplicable }} tests do not apply to the {{ form }} form{% endif %}. "
    "Back-tested from {{ origins }} origins up to {{ horizon }} periods ahead, its {{ level }} "
    "interval covered {{ coverage }} of the actual values, and {{ anomalies }} of "
    "{{ observations }} observations lie outside its {{ anomaly_level }} interval."
)

BENCHMARKS = frozenset({"naive", "seasonal_naive", "mean"})
# Tests of an OLS regression: they need its design matrix or its statsmodels result.
OLS_ONLY = frozenset(
    {"breusch-godfrey", "breusch-pagan", "white", "reset", "cusum", "cusum-recursive", "chow"}
)
NORMAL_ERRORS = frozenset({"arch", "jarque-bera"})
NEEDS_DRIVERS = frozenset({"breusch-pagan", "white", "reset"})
# Brown, Durbin and Evans (1975), Table 1: the CUSUM bound's constant by significance level.
BDE = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}


# --- configuration ---------------------------------------------------------------


class PValueTest(Config):
    """A test read by its p-value: the check passes when the p-value is at least `alpha`."""

    alpha: float = Field(
        0.05, gt=0, lt=1, description="The check passes when the p-value is at least alpha."
    )


class LagTest(PValueTest):
    """A p-value test over a number of lags."""

    lags: int | None = Field(
        None,
        ge=1,
        description="Lags tested; unset: twice the seasonal period (10 for annual data), at "
        "most a fifth of the sample.",
    )


class ResetConfig(PValueTest):
    """Ramsey's RESET: powers of the fitted values added to the regression."""

    power: int = Field(3, ge=2, description="The highest power of the fitted values added.")


class CusumConfig(Config):
    """The CUSUM tests: the OLS-residual test (a p-value) and the recursive CUSUM's bounds."""

    alpha: float = Field(
        0.05,
        description="Level of the OLS-residual CUSUM test and of the recursive CUSUM's bounds: "
        "0.01, 0.05 or 0.10, the levels Brown, Durbin and Evans tabulate.",
    )
    max_crossings: int = Field(
        0, ge=0, description="Most periods the recursive CUSUM may lie outside its bounds."
    )

    @field_validator("alpha")
    @classmethod
    def _tabulated(cls, value: float) -> float:
        # A plain float with a check, not a Literal: pydantic 2.8 writes a float
        # Literal's JSON Schema type as "numeric", which is not a JSON Schema type.
        for level in BDE:
            if abs(value - level) < 1e-12:
                return level
        raise ValueError(f"alpha must be one of {', '.join(f'{v:g}' for v in BDE)}")


Fraction = Annotated[float, Field(gt=0, lt=1)]
Level = Annotated[float, Field(gt=0, lt=1)]


class ChowConfig(PValueTest):
    """Chow tests at given break dates, or at fixed fractions of the sample."""

    breaks: list[str] = Field(
        default_factory=list, description="Break dates, as periods (2008Q1); unset: candidates."
    )
    candidates: list[Fraction] = Field(
        default_factory=lambda: [0.5],
        description="Without breaks: candidate dates as fractions of the sample (0.5: the "
        "middle), fixed before the data are seen.",
    )


class RollingConfig(Config):
    """Rolling-window coefficients (statsmodels `RollingOLS`)."""

    window: int | None = Field(
        None, ge=2, description="Window in periods; unset: a third of the sample, at least 2k."
    )


class BacktestConfig(Config):
    """Multi-step forecasts from past origins, through `dz.evaluation.evaluate`."""

    splitter: str = Field("expanding", description="Rolling-origin splitter, by registry key.")
    splitter_config: dict[str, Any] = Field(
        default_factory=lambda: {"horizon": 4},
        description="The splitter's settings; by default four periods ahead from every origin "
        "after the first half of the sample.",
    )
    level: Level = Field(0.95, description="The interval whose coverage is checked.")
    min_coverage: float = Field(
        0.8, ge=0, le=1, description="Least share of actual values inside that interval."
    )


class SensitivityConfig(Config):
    """One-at-a-time sensitivity of the forecast to each driver."""

    shock: float = Field(1.0, description="The shock added to a driver at every horizon.")
    scale: Literal["sd", "unit"] = Field(
        "sd",
        description="sd: the shock is in standard deviations of the driver as the model takes "
        "it; unit: in its units.",
    )
    horizon: int | None = Field(
        None, ge=1, description="Periods; unset: forecast's horizon, else the scenarios' coverage."
    )
    base: str | int | None = Field(
        None, description="The scenario shocks and paths are measured from; unset: the mildest."
    )


class AnomalyConfig(Config):
    """Observations outside the model's interval."""

    level: Level = Field(0.99, description="Level of the interval around the fitted values.")
    max_share: float = Field(
        0.05, ge=0, le=1, description="Largest share of observations outside it."
    )


class DiagnoseConfig(Config):
    """Configuration of the diagnose step."""

    ljung_box: LagTest = Field(default_factory=LagTest)
    breusch_godfrey: LagTest = Field(default_factory=LagTest)
    breusch_pagan: PValueTest = Field(default_factory=PValueTest)
    white: PValueTest = Field(default_factory=PValueTest)
    arch: LagTest = Field(default_factory=LagTest)
    jarque_bera: PValueTest = Field(default_factory=PValueTest)
    reset: ResetConfig = Field(default_factory=ResetConfig)
    cusum: CusumConfig = Field(default_factory=CusumConfig)
    chow: ChowConfig = Field(default_factory=ChowConfig)
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig)
    overrides: list[Override] = Field(default_factory=list)


# --- result objects --------------------------------------------------------------


class _Record(DazflowModel):
    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "These records are made by the diagnose step."


class Diagnostic(_Record):
    """One diagnostic: its statistic and p-value, what the gate compares, or why it does not apply.

    `observed op threshold` passes. A diagnostic that is not `applicable`
    has a `reason` and no check.
    """

    name: str
    group: Literal["residuals", "stability", "backtest", "anomalies"]
    label: str
    statistic: float | None = None
    pvalue: float | None = None
    observed: float | None = None
    threshold: float | None = None
    op: Comparison | None = None
    applicable: bool = True
    reason: str | None = None

    def check(self) -> Check | None:
        """The gate's check, or ``None`` when the diagnostic does not apply."""
        if not self.applicable or self.observed is None or self.threshold is None:
            return None
        assert self.op is not None
        return Check(name=self.name, observed=self.observed, threshold=self.threshold, op=self.op)


class ChowTest(_Record):
    """A Chow test at `period`: F with `df_num` and `df_denom` degrees of freedom."""

    period: str
    statistic: float
    pvalue: float
    df_num: int
    df_denom: int


class Cusum(_Record):
    """The recursive CUSUM (statsmodels `RecursiveLS`) and its bounds, by period."""

    alpha: float
    periods: tuple[str, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]


class Rolling(_Record):
    """Rolling-window coefficients (statsmodels `RollingOLS`), by the window's last period."""

    window: int
    periods: tuple[str, ...]
    params: dict[str, tuple[float, ...]]


class Sensitivity(_Record):
    """One response: to a shock in a driver, or to one driver's scenario path (``all``: every one).

    `response` is on the model's scale, horizon by horizon; `level_response`
    on the target's, through its transform chain without a bias correction.
    `size` is the shock added to the driver's column.
    """

    kind: Literal["shock", "scenario"]
    driver: str
    scenario: str | int | None = None
    size: float | None = None
    response: tuple[float, ...]
    level_response: tuple[float, ...]


class Anomaly(_Record):
    """An observation outside the interval ``fitted ± z scale``, on the model's scale."""

    period: str
    actual: float
    fitted: float
    residual: float
    lower: float
    upper: float
    z: float


class DiagnoseResult(StepResult):
    """What diagnose returns: every diagnostic, the stability paths, the back-test and the rest.

    `diagnostics` lists every test in a fixed order, applicable or not;
    `diagnostic(name)` finds one. `scale` is the standard error behind the
    anomaly interval, on the model's scale.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: DiagnoseConfig
    fingerprint: str | None = None
    target: str | None = None
    chosen: str | None = None
    form: str | None = None
    record: ModelRecord | None = None
    design: Design | None = None
    lags: int = 0
    diagnostics: tuple[Diagnostic, ...] = ()
    cusum: Cusum | None = None
    rolling: Rolling | None = None
    backtest: Evaluation | None = None
    base: str | int | None = None
    horizon: int = 0
    sensitivity: tuple[Sensitivity, ...] = ()
    scale: float | None = None
    anomalies: tuple[Anomaly, ...] = ()

    def diagnostic(self, name: str) -> Diagnostic:
        """The diagnostic named `name` (``ljung-box``, ``chow:2008Q1``, ...)."""
        for d in self.diagnostics:
            if d.name == name:
                return d
        raise DiagnoseError(
            f"no diagnostic {name!r}",
            observed=f"diagnostics: {', '.join(d.name for d in self.diagnostics)}",
            change="Name one of the diagnostics listed.",
        )


Config = DiagnoseConfig
Result = DiagnoseResult


# --- the pieces ------------------------------------------------------------------


def _fail(problem: str, observed: str, change: str) -> DiagnoseError:
    return DiagnoseError(problem, observed=observed, change=change)


def _plain(f: Any, *args: Any, **kwargs: Any) -> Any:
    # statsmodels 0.15 warns unless asked for the tuple it returns; 0.14 has no such argument.
    if "result_object" in inspect.signature(f).parameters:
        kwargs["result_object"] = False
    return f(*args, **kwargs)


def default_lags(n: int, index: pd.PeriodIndex) -> int:
    """Twice the seasonal period (10 for annual data), at most a fifth of `n` (Hyndman)."""
    try:
        m = models.season_length(index)
    except ModelError:
        m = 1
    return max(1, min(2 * m if m > 1 else 10, n // 5))


def _reason(test: str, form: str, drivers: bool) -> str | None:
    if form in BENCHMARKS:
        return f"the {form} benchmark has no estimated dynamics or drivers to test"
    if test in OLS_ONLY and form != "ols":
        return f"a test of an OLS regression; the model is {form}, not ols"
    if test in NORMAL_ERRORS and form == "glm":
        return "a GLM does not assume normal errors of constant variance"
    if test in NEEDS_DRIVERS and not drivers:
        return "the model has no drivers"
    return None


def _arma_order(form: str, config: Mapping[str, Any]) -> int:
    if form != "arimax":
        return 0
    p, _, q = config.get("order", (0, 0, 0))
    P, _, Q, _ = config.get("seasonal_order", (0, 0, 0, 0))
    return int(p + q + P + Q)


def _exog(X: pd.DataFrame | None, index: pd.PeriodIndex, constant: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=index) if X is None else X.astype("float64").copy()
    if constant:
        out.insert(0, "const", 1.0)
    return out


def _ols(y: pd.Series, X: pd.DataFrame | None, constant: bool) -> Any:
    import statsmodels.api as sm

    return sm.OLS(y.astype("float64"), _exog(X, y.index, constant)).fit()


def _pvalue_diag(
    name: str, group: Any, label: str, statistic: Any, pvalue: Any, alpha: float
) -> Diagnostic:
    p = float(pvalue)
    return Diagnostic(
        name=name,
        group=group,
        label=label,
        statistic=float(np.squeeze(statistic)),
        pvalue=p,
        observed=p,
        threshold=alpha,
        op=">=",
    )


def _not_applicable(name: str, group: Any, label: str, reason: str) -> Diagnostic:
    return Diagnostic(name=name, group=group, label=label, applicable=False, reason=reason)


def residual_tests(
    fitted: Any,
    y: pd.Series,
    X: pd.DataFrame | None,
    config: DiagnoseConfig | Mapping[str, Any] | None = None,
) -> tuple[Diagnostic, ...]:
    """The residual tests of a fitted model, in a fixed order, each applicable or with its reason.

    `fitted` is a `dz.models.Fitted`; `y` and `X` the data it was fitted on.
    Ljung-Box (with the ARMA order as `model_df` for an ARIMAX), ARCH and
    Jarque-Bera take the model's residuals; Breusch-Godfrey, Breusch-Pagan,
    White and RESET are tests of an OLS regression and are run on the
    statsmodels OLS fit of the same data.
    """
    from statsmodels.stats import diagnostic as smd
    from statsmodels.stats.stattools import jarque_bera

    cfg = (
        config
        if isinstance(config, DiagnoseConfig)
        else DiagnoseConfig.model_validate(config or {})
    )
    form = fitted.spec.form
    drivers = X is not None and len(X.columns) > 0
    resid = fitted.residuals
    n = len(resid)
    lags = default_lags(n, resid.index)
    lb_lags = cfg.ljung_box.lags or lags
    bg_lags = cfg.breusch_godfrey.lags or lags
    arch_lags = cfg.arch.lags or lags
    labels = {
        "ljung-box": f"Ljung-Box ({lb_lags} lags)",
        "breusch-godfrey": f"Breusch-Godfrey ({bg_lags} lags)",
        "breusch-pagan": "Breusch-Pagan",
        "white": "White",
        "arch": f"ARCH LM ({arch_lags} lags)",
        "jarque-bera": "Jarque-Bera",
        "reset": f"RESET (power {cfg.reset.power})",
    }
    reasons = {name: _reason(name, form, drivers) for name in labels}
    res = None
    if form == "ols" and any(reasons[t] is None for t in OLS_ONLY & set(labels)):
        res = _ols(y, X, bool(fitted.spec.config.get("constant", True)))
        e = np.asarray(res.resid)
        exog_het = _exog(X, y.index, True).to_numpy()
    else:
        e = resid.to_numpy(dtype=float)
    df = _arma_order(form, fitted.spec.config)
    if reasons["ljung-box"] is None and lb_lags <= df:
        reasons["ljung-box"] = f"{lb_lags} lags do not exceed the ARMA order, {df}"

    out = []
    for name, label in labels.items():
        if reasons[name] is not None:
            out.append(_not_applicable(name, "residuals", label, reasons[name]))
            continue
        if name == "ljung-box":
            t = smd.acorr_ljungbox(e, lags=[lb_lags], model_df=df, return_df=True)
            stat, p, alpha = t["lb_stat"].iloc[0], t["lb_pvalue"].iloc[0], cfg.ljung_box.alpha
        elif name == "breusch-godfrey":
            stat, p = _plain(smd.acorr_breusch_godfrey, res, nlags=bg_lags)[:2]
            alpha = cfg.breusch_godfrey.alpha
        elif name == "breusch-pagan":
            stat, p = smd.het_breuschpagan(e, exog_het)[:2]
            alpha = cfg.breusch_pagan.alpha
        elif name == "white":
            stat, p = smd.het_white(e, exog_het)[:2]
            alpha = cfg.white.alpha
        elif name == "arch":
            stat, p = _plain(smd.het_arch, e, nlags=arch_lags)[:2]
            alpha = cfg.arch.alpha
        elif name == "jarque-bera":
            stat, p = jarque_bera(e)[:2]
            alpha = cfg.jarque_bera.alpha
        else:
            t = smd.linear_reset(res, power=cfg.reset.power)
            stat, p, alpha = t.statistic, t.pvalue, cfg.reset.alpha
        out.append(_pvalue_diag(name, "residuals", label, stat, p, alpha))
    return tuple(out)


def _matrix(X: pd.DataFrame | None, n: int, constant: bool) -> np.ndarray:
    cols = [] if X is None else [X.to_numpy(dtype=float)]
    if constant:
        cols.insert(0, np.ones((n, 1)))
    if not cols:
        raise _fail("the regression has no coefficients", "no constant and no drivers", "Give X.")
    return np.hstack(cols)


def _ssr(A: np.ndarray, b: np.ndarray) -> float:
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    e = b - A @ coef
    return float(e @ e)


def chow(y: pd.Series, X: pd.DataFrame | None, period: Any, *, constant: bool = True) -> ChowTest:
    """Chow's F test of a break at `period`: every coefficient may change from `period` on.

    F = ((RSS_pooled - RSS_1 - RSS_2) / k) / ((RSS_1 + RSS_2) / (n - 2k)),
    with k coefficients (the drivers and, with `constant`, an intercept),
    on F(k, n - 2k). Each part needs more than k observations; otherwise a
    `DiagnoseError`.
    """
    from scipy import stats

    at = pd.Period(period, freq=y.index.freq)
    b = y.to_numpy(dtype=float)
    A = _matrix(X, len(b), constant)
    n, k = A.shape
    after = np.asarray(y.index >= at)
    n1, n2 = int((~after).sum()), int(after.sum())
    if n1 <= k or n2 <= k:
        raise _fail(
            "the break leaves too few observations on one side for its own fit",
            f"break {at}: {n1} observations before it and {n2} from it; {k} coefficients",
            f"Choose a break with more than {k} observations on each side.",
        )
    pooled = _ssr(A, b)
    parts = _ssr(A[~after], b[~after]) + _ssr(A[after], b[after])
    F = ((pooled - parts) / k) / (parts / (n - 2 * k))
    return ChowTest(
        period=str(at),
        statistic=float(F),
        pvalue=float(stats.f.sf(F, k, n - 2 * k)),
        df_num=k,
        df_denom=n - 2 * k,
    )


def anomalies(
    actual: pd.Series, fitted: pd.Series, scale: float, level: float
) -> tuple[Anomaly, ...]:
    """The periods where `actual` lies outside ``fitted ± z scale``, z the normal `level` quantile.

    A value on a bound is inside.
    """
    z = NormalDist().inv_cdf((1 + level) / 2)
    half = z * float(scale)
    fit = fitted.astype("float64")
    act = actual.astype("float64").reindex(fit.index)
    out = []
    for p, a, f in zip(fit.index, act.to_numpy(), fit.to_numpy(), strict=True):
        e = a - f
        if abs(e) > half:
            out.append(
                Anomaly(
                    period=str(p),
                    actual=float(a),
                    fitted=float(f),
                    residual=float(e),
                    lower=float(f - half),
                    upper=float(f + half),
                    z=float(e / scale),
                )
            )
    return tuple(out)


def _scale(fitted: Any) -> float:
    params = fitted.params
    if "sigma2" in params.index:
        return float(np.sqrt(params["sigma2"]))
    e = fitted.residuals.to_numpy(dtype=float)
    return float(np.sqrt(e @ e / (len(e) - len(params))))


def _stability(
    y: pd.Series, X: pd.DataFrame | None, constant: bool, cfg: DiagnoseConfig, reason: str | None
) -> tuple[list[Diagnostic], Cusum | None, Rolling | None]:
    from statsmodels.regression.recursive_ls import RecursiveLS
    from statsmodels.regression.rolling import RollingOLS
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    n = len(y)
    index = y.index
    if cfg.chow.breaks:
        dates = [pd.Period(b, freq=index.freq) for b in cfg.chow.breaks]
    else:
        dates = [index[min(n - 1, int(f * n))] for f in cfg.chow.candidates]
    dates = list(dict.fromkeys(dates))
    a = BDE[cfg.cusum.alpha]
    if reason is not None:
        out = [
            _not_applicable("cusum", "stability", "CUSUM of OLS residuals", reason),
            _not_applicable("cusum-recursive", "stability", "Recursive CUSUM", reason),
        ]
        out += [_not_applicable(f"chow:{d}", "stability", f"Chow at {d}", reason) for d in dates]
        return out, None, None

    exog = _exog(X, index, constant)
    k = exog.shape[1]
    res = _ols(y, X, constant)
    sup, p, _ = breaks_cusumolsresid(np.asarray(res.resid), ddof=k)
    out = [
        _pvalue_diag("cusum", "stability", "CUSUM of OLS residuals", sup, p, cfg.cusum.alpha),
    ]
    rls = RecursiveLS(y.to_numpy(dtype=float), exog.to_numpy()).fit()
    d = max(int(rls.nobs_diffuse), int(rls.loglikelihood_burn))
    values = np.asarray(rls.cusum, dtype=float)
    root = math.sqrt(n - d)
    upper = a * root + 2 * a * (np.arange(d, n) - d) / root
    crossings = int(np.sum(np.abs(values) > upper))
    out.append(
        Diagnostic(
            name="cusum-recursive",
            group="stability",
            label=f"Recursive CUSUM, periods outside the {cfg.cusum.alpha:.0%} bounds",
            statistic=float(np.max(np.abs(values) / upper)),
            observed=float(crossings),
            threshold=float(cfg.cusum.max_crossings),
            op="<=",
        )
    )
    cusum = Cusum(
        alpha=cfg.cusum.alpha,
        periods=tuple(str(q) for q in index[d:]),
        values=tuple(float(v) for v in values),
        lower=tuple(float(-v) for v in upper),
        upper=tuple(float(v) for v in upper),
    )
    for date in dates:
        label = f"Chow at {date}"
        try:
            t = chow(y, X, date, constant=constant)
        except DiagnoseError as err:
            out.append(_not_applicable(f"chow:{date}", "stability", label, err.problem))
            continue
        out.append(
            _pvalue_diag(f"chow:{date}", "stability", label, t.statistic, t.pvalue, cfg.chow.alpha)
        )

    window = cfg.rolling.window if cfg.rolling.window is not None else max(2 * k, n // 3)
    if not k < window <= n:
        raise _fail(
            "the rolling window does not fit the sample",
            f"window {window}; {n} observations and {k} coefficients",
            f"Set rolling.window between {k + 1} and {n}.",
        )
    params = RollingOLS(y.to_numpy(dtype=float), exog.to_numpy(), window=window).fit().params
    rolling = Rolling(
        window=window,
        periods=tuple(str(q) for q in index[window - 1 :]),
        params={
            str(c): tuple(float(v) for v in np.asarray(params)[window - 1 :, j])
            for j, c in enumerate(exog.columns)
        },
    )
    return out, cusum, rolling


def _backtest(
    record: ModelRecord, y: pd.Series, X: pd.DataFrame | None, cfg: BacktestConfig
) -> Evaluation:
    split = evaluation.splitters.get(cfg.splitter)(cfg.splitter_config)
    coverage = evaluation.metrics.get("coverage")(level=cfg.level)
    try:
        return evaluation.evaluate(
            record.spec,
            y,
            X,
            splitter=split,
            metrics=("mae", "rmse", "mase", coverage),
            levels=[cfg.level],
        )
    except (EvaluationError, ModelError) as err:
        raise _fail(
            "the back-test cannot run as configured", err.problem, err.change or ""
        ) from err


def _covered(paths: pd.DataFrame, end: pd.Period) -> int:
    have = set(paths.index)
    n = 0
    while end + n + 1 in have:
        n += 1
    return n


def _scenarios(dataset: Dataset, plain: Dataset, select: Any) -> tuple[pd.DataFrame | None, list]:
    if dataset.scenarios is not None:
        return dataset.scenarios, list(dataset.severity)
    spec = getattr(getattr(select, "config", None), "scenarios", None)
    if spec is None:
        return None, []
    try:
        frame, severity = readers.read_scenarios(spec)
    except ReaderError as err:
        raise _fail(err.problem, err.observed, err.change) from err
    with_scen = plain.with_scenarios(frame, severity)
    return with_scen.scenarios, list(with_scen.severity)


def _sensitivity(
    fitted: Any,
    design: Design,
    levels: pd.DataFrame,
    X: pd.DataFrame,
    scen: pd.DataFrame | None,
    severity: Sequence[Any],
    forecast: Any,
    cfg: SensitivityConfig,
) -> tuple[tuple[Sensitivity, ...], Any, int]:
    freq = levels.index.freq
    end = pd.Period(design.end, freq=freq)
    target = design.target
    chain = list(design.chains.get(target, ()))
    history = levels[target].astype("float64").loc[:end].rename(target)
    drivers = list(design.drivers)

    if scen is not None:
        base = cfg.base if cfg.base is not None else severity[0]
        if base not in severity:
            raise _fail(
                f"no base scenario {base!r}",
                f"scenarios: {', '.join(map(str, severity))}",
                "Set sensitivity.base to one of the scenarios, or leave it unset.",
            )
        cover = min(_covered(scen.xs(s, level="scenario"), end) for s in severity)
    else:
        base, cover = None, None
    if cfg.horizon is not None:
        h = cfg.horizon
    elif forecast is not None and getattr(forecast, "horizon", 0):
        h = int(forecast.horizon)
    elif cover:
        h = cover
    else:
        raise _fail(
            "without scenarios or a forecast, the sensitivity horizon must be given",
            "sensitivity.horizon unset",
            "Set sensitivity.horizon in config/diagnose.yaml.",
        )
    if cover is not None and h > cover:
        raise _fail(
            f"the scenarios do not cover the sensitivity horizon of {h} periods",
            f"every scenario covers {cover} periods after {end}",
            "Shorten sensitivity.horizon, or extend the scenario paths.",
        )
    periods = pd.period_range(end + 1, periods=h, freq=freq)
    if scen is not None:
        base_paths = scen.xs(base, level="scenario").reindex(periods)
    else:
        last = levels[drivers].astype("float64").loc[:end].iloc[-1]
        base_paths = pd.DataFrame({d: np.full(h, last[d]) for d in drivers}, index=periods)

    def mean(paths: pd.DataFrame | None, columns: pd.DataFrame | None = None) -> np.ndarray:
        try:
            future = columns if columns is not None else design.future(levels, paths, h)
            return np.asarray(fitted.predict(h, future, levels=(0.95,)).mean, dtype=float)
        except ModelError as err:
            raise _fail(
                "the scenarios cannot drive the chosen model", err.problem, err.change
            ) from err

    def level(m: np.ndarray) -> np.ndarray:
        if not chain:
            return m
        series = pd.Series(m, index=periods, name=target)
        return transforms.inverse(series, chain, history).to_numpy(dtype=float)

    try:
        X_base = design.future(levels, base_paths, h)
    except ModelError as err:
        raise _fail("the scenarios cannot drive the chosen model", err.problem, err.change) from err
    m0 = mean(None, X_base)
    l0 = level(m0)
    out: list[Sensitivity] = []
    for d in drivers:
        size = cfg.shock * (float(X[d].std(ddof=1)) if cfg.scale == "sd" else 1.0)
        shocked = X_base.copy()
        shocked[d] = shocked[d] + size
        m = mean(None, shocked)
        out.append(
            Sensitivity(
                kind="shock",
                driver=d,
                scenario=base,
                size=size,
                response=tuple(float(v) for v in m - m0),
                level_response=tuple(float(v) for v in level(m) - l0),
            )
        )
    if scen is not None:
        for s in severity:
            if s == base:
                continue
            other = scen.xs(s, level="scenario").reindex(periods)
            for d in [*drivers, "all"]:
                paths = other if d == "all" else base_paths.assign(**{d: other[d]})
                m = mean(paths)
                out.append(
                    Sensitivity(
                        kind="scenario",
                        driver=d,
                        scenario=s,
                        response=tuple(float(v) for v in m - m0),
                        level_response=tuple(float(v) for v in level(m) - l0),
                    )
                )
    return tuple(out), base, h


# --- the step --------------------------------------------------------------------


def run(
    dataset: Dataset,
    config: DiagnoseConfig | Mapping[str, Any] | None = None,
    *,
    select: Any,
    forecast: Any = None,
) -> DiagnoseResult:
    """Test select's chosen model: residuals, stability, back-test, sensitivity and anomalies.

    `dataset` holds the series in levels select ran on, with any scenarios
    in `dataset.scenarios`; without them, select's `config.scenarios` file
    is read (relative to the working directory). `forecast`, the forecast
    step's result for the same model, sets the sensitivity horizon. The
    caller's `dataset` is not changed. Prints nothing and writes no files.
    """
    cfg = (
        config
        if isinstance(config, DiagnoseConfig)
        else DiagnoseConfig.model_validate(config or {})
    )
    plain = dataset.with_scenarios(None) if dataset.scenarios is not None else dataset
    if plain.fingerprint() != getattr(select, "fingerprint", None):
        raise _fail(
            "the dataset is not the one select ran on",
            f"fingerprint {plain.fingerprint()[:12]}...; select's "
            f"{str(getattr(select, 'fingerprint', None))[:12]}...",
            "Pass diagnose the dataset select received (in levels), with select's result.",
        )
    if select.chosen is None or select.record is None or select.design is None:
        raise _fail(
            "select chose no model, so there is no model to diagnose",
            f"select's decision: {getattr(select, 'decision', 'none')}",
            "Choose a candidate in config/select.yaml: choice, with a reason.",
        )
    if forecast is not None and (
        forecast.chosen != select.chosen or forecast.fingerprint != select.fingerprint
    ):
        raise _fail(
            "the forecast is of another model or dataset than select's",
            f"forecast: {forecast.chosen}; select: {select.chosen}",
            "Pass the forecast made from this select result, or none.",
        )
    design: Design = select.design
    record: ModelRecord = select.record
    form = record.spec.form
    levels = plain.frame
    try:
        y, X = design.frames(levels)
        fitted = models.refit(record, y, X)
    except ModelError as err:
        raise _fail(
            "the chosen model cannot be refitted", err.problem, "Pass the data select ran on."
        ) from err
    constant = bool(record.spec.config.get("constant", True))
    drivers = X is not None and len(X.columns) > 0

    diagnostics = list(residual_tests(fitted, y, X, cfg))
    reason = _reason("cusum", form, drivers)
    stab, cusum, rolling = _stability(y, X, constant, cfg, reason)
    diagnostics += stab

    ev = _backtest(record, y, X, cfg.backtest)
    coverage = ev.score("coverage")
    diagnostics.append(
        Diagnostic(
            name="backtest-coverage",
            group="backtest",
            label=f"Back-test coverage of the {cfg.backtest.level:.0%} interval",
            observed=coverage,
            threshold=cfg.backtest.min_coverage,
            op=">=",
        )
    )

    scale = _scale(fitted)
    found = anomalies(y, fitted.fitted_values, scale, cfg.anomalies.level)
    n = len(fitted.fitted_values)
    share = len(found) / n
    diagnostics.append(
        Diagnostic(
            name="anomalies",
            group="anomalies",
            label=f"Share of observations outside the {cfg.anomalies.level:.0%} interval",
            statistic=float(len(found)),
            observed=share,
            threshold=cfg.anomalies.max_share,
            op="<=",
        )
    )

    if drivers:
        scen, severity = _scenarios(dataset, plain, select)
        sens, base, h = _sensitivity(
            fitted, design, levels, X, scen, severity, forecast, cfg.sensitivity
        )
    else:
        sens, base, h = (), None, 0

    checks = [c for c in (d.check() for d in diagnostics) if c is not None]
    gate = Gate.evaluate(checks, cfg.overrides)
    inapplicable = sum(1 for d in diagnostics if not d.applicable)
    failed = [c.name for c in gate.checks if c.status == "fail"]
    finding = Finding(
        template=SUMMARY,
        key="diagnose.summary",
        values={
            "chosen": select.chosen,
            "form": form,
            "passed": sum(1 for c in gate.checks if c.passed),
            "checks": len(gate.checks),
            "failed": ", ".join(failed),
            "overridden": ", ".join(c.name for c in gate.overridden),
            "overridden_count": len(gate.overridden),
            "inapplicable": inapplicable,
            "origins": len(ev.folds),
            "horizon": max(r.horizon for r in ev.rows),
            "level": f"{cfg.backtest.level:.0%}",
            "coverage": coverage,
            "anomalies": len(found),
            "observations": n,
            "anomaly_level": f"{cfg.anomalies.level:.0%}",
        },
    )
    residuals = fitted.residuals
    artefacts: dict[str, Any] = {
        IDS.table("diagnostics"): _diagnostics_table(diagnostics, gate, select.chosen),
        IDS.table("sensitivity"): _sensitivity_table(sens, design.target, h),
        IDS.table("anomalies"): _anomaly_table(found, cfg.anomalies.level, design),
        IDS.table("backtest"): _backtest_table(ev, cfg.backtest.level),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Diagnose checks"),
        IDS.figure("residuals"): _residual_figure(
            residuals, found, scale, cfg.anomalies.level, select.chosen
        ),
        IDS.finding("summary"): finding,
    }
    if cusum is not None:
        artefacts[IDS.figure("cusum")] = _cusum_figure(cusum, select.chosen)
    if rolling is not None and design.drivers:
        artefacts[IDS.figure("rolling")] = _rolling_figure(rolling, design.drivers, select.chosen)
    return DiagnoseResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        fingerprint=plain.fingerprint(),
        target=design.target,
        chosen=select.chosen,
        form=form,
        record=record,
        design=design,
        lags=default_lags(len(residuals), residuals.index),
        diagnostics=tuple(diagnostics),
        cusum=cusum,
        rolling=rolling,
        backtest=ev,
        base=base,
        horizon=h,
        sensitivity=sens,
        scale=scale,
        anomalies=found,
    )


# --- artefacts ---------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _diagnostics_table(diagnostics: Sequence[Diagnostic], gate: Gate, chosen: str) -> Table:
    status = {c.name: c.status for c in gate.checks}
    rows = [
        [
            d.label,
            d.group,
            _finite(d.statistic),
            _finite(d.pvalue),
            _finite(d.observed),
            d.op,
            _finite(d.threshold),
            status[d.name],
        ]
        for d in diagnostics
        if d.applicable
    ]
    notes = [
        "A check passes when the compared value (a p-value, a coverage, a count or a share) "
        "stands in its relation to the threshold."
    ]
    notes += [f"Not applicable: {d.label}, as {d.reason}." for d in diagnostics if not d.applicable]
    return Table(
        id=IDS.table("diagnostics"),
        caption=f"Diagnostics of {chosen}",
        columns=[
            Column(name="test", label="Test", kind="text"),
            Column(name="group", label="Group", kind="text"),
            Column(name="statistic", label="Statistic", kind="number"),
            Column(name="pvalue", label="p-value", kind="number"),
            Column(name="observed", label="Compared", kind="number"),
            Column(name="op", label="Passes when", kind="text"),
            Column(name="threshold", label="Threshold", kind="number"),
            Column(name="status", label="Status", kind="status"),
        ],
        rows=rows,
        notes=tuple(notes),
    )


def _sensitivity_table(rows: Sequence[Sensitivity], target: str, h: int) -> Table:
    body = [
        [
            r.kind,
            r.driver,
            None if r.scenario is None else str(r.scenario),
            _finite(r.size),
            _finite(r.response[0]),
            _finite(r.response[-1]),
            _finite(r.level_response[-1]),
        ]
        for r in rows
    ]
    return Table(
        id=IDS.table("sensitivity"),
        caption=f"Sensitivity of the forecast of {target}, one driver at a time",
        columns=[
            Column(name="kind", label="Kind", kind="text"),
            Column(name="driver", label="Driver", kind="text"),
            Column(name="scenario", label="Scenario", kind="text"),
            Column(name="size", label="Shock", kind="number"),
            Column(name="first", label="Response at h = 1 (model scale)", kind="number"),
            Column(name="last", label=f"Response at h = {h} (model scale)", kind="number"),
            Column(name="level", label=f"Response at h = {h} ({target})", kind="number"),
        ],
        rows=body,
        notes=(
            "A shock is added to the driver as the model takes it (after its transform and lag) "
            "at every horizon, from the base scenario; a scenario row replaces the base path of "
            "one driver (all: every driver) by that scenario's. Responses on the target's scale "
            "go through its transform chain without a bias correction.",
        ),
    )


def _anomaly_table(found: Sequence[Anomaly], level: float, design: Design) -> Table:
    chain = ", ".join(design.chains.get(design.target, ())) or "levels"
    return Table(
        id=IDS.table("anomalies"),
        caption=f"Observations outside the model's {level:.0%} interval",
        columns=[
            Column(name="period", label="Period", kind="text"),
            Column(name="actual", label="Actual", kind="number"),
            Column(name="fitted", label="Fitted", kind="number"),
            Column(name="residual", label="Residual", kind="number"),
            Column(name="lower", label="Lower", kind="number"),
            Column(name="upper", label="Upper", kind="number"),
            Column(name="z", label="Standardised residual", kind="number"),
        ],
        rows=[[a.period, a.actual, a.fitted, a.residual, a.lower, a.upper, a.z] for a in found],
        notes=(f"On the model's scale ({design.target}: {chain}).",),
    )


def _backtest_table(ev: Evaluation, level: float) -> Table:
    horizons = sorted({r.horizon for r in ev.rows})
    rows = []
    for h in (None, *horizons):
        rows.append(
            [
                "all" if h is None else str(h),
                _finite(ev.score("mae", h)),
                _finite(ev.score("rmse", h)),
                _finite(ev.score("mase", h)),
                _finite(ev.score("coverage", h)),
            ]
        )
    return Table(
        id=IDS.table("backtest"),
        caption="Back-test: forecasts from past origins against the actual values",
        columns=[
            Column(name="horizon", label="Horizon", kind="text"),
            Column(name="mae", label="MAE", kind="number"),
            Column(name="rmse", label="RMSE", kind="number"),
            Column(name="mase", label="MASE", kind="number"),
            Column(name="coverage", label=f"Coverage of the {level:.0%} interval", kind="percent"),
        ],
        rows=rows,
        notes=(
            f"{len(ev.folds)} origins ({ev.splitter}); the model is refitted at every origin and "
            "forecast with the drivers' actual values, on the model's scale.",
        ),
    )


def _residual_figure(
    residuals: pd.Series, found: Sequence[Anomaly], scale: float, level: float, chosen: str
) -> Figure:
    z = NormalDist().inv_cdf((1 + level) / 2)
    x = [str(p) for p in residuals.index]
    flagged = {a.period for a in found}
    half = z * scale
    layers = [
        Layer(
            kind="band",
            y_low=[-half] * len(x),
            y_high=[half] * len(x),
            label=f"{level:.0%} interval",
        ),
        Layer(kind="line", y=[float(v) for v in residuals], label="residual"),
        Layer(kind="hline", value=0.0),
    ]
    if flagged:
        layers.append(
            Layer(
                kind="scatter",
                y=[float(v) if p in flagged else None for p, v in zip(x, residuals, strict=True)],
                label="outside the interval",
            )
        )
    return Figure(
        id=IDS.figure("residuals"),
        caption=f"Residuals of {chosen}",
        x=x,
        layers=layers,
        x_label="Period",
        y_label="Residual (model scale)",
    )


def _cusum_figure(c: Cusum, chosen: str) -> Figure:
    return Figure(
        id=IDS.figure("cusum"),
        caption=f"Recursive CUSUM of {chosen}",
        x=list(c.periods),
        layers=[
            Layer(kind="line", y=list(c.values), label="CUSUM"),
            Layer(kind="line", y=list(c.upper), label=f"{c.alpha:.0%} bound"),
            Layer(kind="line", y=list(c.lower), label=None),
        ],
        x_label="Period",
        y_label="Cumulated standardised recursive residuals",
        notes=("Bounds of Brown, Durbin and Evans (1975).",),
    )


def _rolling_figure(r: Rolling, drivers: Sequence[str], chosen: str) -> Figure:
    return Figure(
        id=IDS.figure("rolling"),
        caption=f"Rolling coefficients of {chosen}, {r.window}-period window",
        x=list(r.periods),
        layers=[Layer(kind="line", y=list(r.params[d]), label=d) for d in drivers],
        x_label="Last period of the window",
        y_label="Coefficient",
    )
