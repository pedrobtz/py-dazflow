"""build: candidate models within the declared signs and exclusions (design §7.4).

The drivers (by default those explore kept) are checked for correlation
and variance inflation. Driver combinations up to `max_size` are searched,
exhaustively or stepwise, skipping excluded pairs. For each combination
the forms are fitted in the configured order of complexity, moving to the
next form only when the simpler one fails its gate: every driver
coefficient has its expected sign and a p-value at most `significance`,
and no driver's VIF in the candidate exceeds `max_vif`. The benchmarks are
always fitted. Every candidate is fitted on one common sample, so select
compares them on the same periods.

>>> import numpy as np, pandas as pd
>>> import dazflow as dz
>>> rng = np.random.default_rng(0)
>>> x = rng.normal(size=(80, 2))
>>> y = 1 + 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * rng.normal(size=80)
>>> index = pd.period_range("2000Q1", periods=80, freq="Q")
>>> ds = dz.Dataset(pd.DataFrame({"y": y, "a": x[:, 0], "b": x[:, 1]}, index=index),
...                 {"y": {"role": "target"}, "a": {"role": "driver", "expected_sign": 1},
...                  "b": {"role": "driver", "expected_sign": -1}})
>>> r = dz.build.run(ds, {"forms": ["ols"]})
>>> r.passing, r.gate.passed
(('ols:a', 'ols:b', 'ols:a+b'), True)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from dazflow import models
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import BuildError, GateError, ModelError, ResultError, UnknownKeyError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import StepResult
from dazflow.models import TREND, Design, ModelRecord, Sample

__all__ = [
    "IDS",
    "BuildConfig",
    "BuildResult",
    "Candidate",
    "Coefficient",
    "Config",
    "FormSpec",
    "Result",
    "combinations",
    "run",
    "stepwise",
    "vif",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "build",
    tables={"correlation": 1, "candidates": 2, "coefficients": 3, "gate": 4},
    figures={"coefficients": 1},
    findings={"summary": 1},
)

SUMMARY = (
    "{{ candidates }} candidates were fitted on {{ combinations }} driver combinations of up to "
    "{{ size }} drivers, beside {{ benchmarks }} benchmarks; {{ passing }} pass every check: "
    "each driver has its expected sign and a p-value of at most {{ significance }}, and no "
    "VIF exceeds {{ max_vif }}."
)

_DIFF = "diff("


# --- configuration ---------------------------------------------------------------


class FormSpec(Config):
    """A model form by registry key, with its configuration (``ols`` or ``{form: ols, config: ...}``)."""

    form: str = Field(min_length=1, description="Model form, by registry key.")
    config: dict[str, Any] = Field(default_factory=dict, description="The form's settings.")

    @model_validator(mode="before")
    @classmethod
    def _key_alone(cls, data: Any) -> Any:
        return {"form": data} if isinstance(data, str) else data


def _forms() -> list[FormSpec]:
    return [
        FormSpec(form="ols", config={"cov_type": "HAC"}),
        FormSpec(form="arimax", config={"order": [1, 0, 0]}),
    ]


def _benchmarks() -> list[FormSpec]:
    return [FormSpec(form="naive"), FormSpec(form="mean")]


class BuildConfig(Config):
    """Configuration of the build step."""

    target: str | None = Field(
        None, description="The target; unset: explore's target, else the dataset's one target."
    )
    drivers: list[str] | None = Field(
        None,
        description="The drivers searched, in this order; unset: those explore kept, else the "
        "dataset's drivers.",
    )
    expected_signs: dict[str, Literal[-1, 0, 1]] = Field(
        default_factory=dict,
        description="Expected sign by driver; replaces explore's and the metadata. 0: no check.",
    )
    lags: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict,
        description="Lag by driver, in periods (0 or more); replaces explore's lags.",
    )
    search: Literal["exhaustive", "stepwise"] = Field(
        "exhaustive",
        description="Every combination up to max_size, or forward stepwise by the criterion.",
    )
    max_size: int = Field(2, ge=1, description="Most drivers in a candidate.")
    criterion: Literal["aic", "bic"] = Field(
        "aic", description="Stepwise: the information criterion of an OLS fit that adds a driver."
    )
    exclusions: list[tuple[str, str]] = Field(
        default_factory=list, description="Pairs of drivers never in one candidate: [[a, b]]."
    )
    forms: list[FormSpec] = Field(
        default_factory=_forms,
        description="Forms with drivers, simplest first; the next is fitted only when the "
        "simpler one fails a check.",
    )
    benchmarks: list[FormSpec] = Field(
        default_factory=_benchmarks, description="Forms without drivers, always fitted."
    )
    significance: float = Field(
        0.05, gt=0, lt=1, description="A driver's coefficient needs a p-value at most this."
    )
    max_vif: float = Field(10.0, gt=1, description="Largest accepted VIF of a candidate's drivers.")
    trend: Literal["auto", "always", "never"] = Field(
        "auto",
        description="A linear trend term: auto adds one when the target or a driver is "
        "trend-stationary and kept in levels (explore's trend_stationary: keep).",
    )
    min_candidates: int = Field(
        1, ge=1, description="Fewest candidates that must pass every check."
    )
    overrides: list[Override] = Field(
        default_factory=list,
        description="Accepted failed checks, with reasons; a candidate's check is "
        "'<candidate>/<check>', as in ols:gdp/significance:gdp.",
    )

    @field_validator("benchmarks")
    @classmethod
    def _one_of_each(cls, value: list[FormSpec]) -> list[FormSpec]:
        keys = [b.form for b in value]
        if len(set(keys)) != len(keys):
            raise ValueError(f"each benchmark form is listed once; given {keys}")
        return value

    @field_validator("forms")
    @classmethod
    def _each_form_once(cls, value: list[FormSpec]) -> list[FormSpec]:
        # A candidate's id is <form>:<drivers>, so a form listed twice would
        # give two fits one id (#49).
        keys = [f.form for f in value]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"each form is listed once, since a candidate's id is <form>:<drivers>; "
                f"given {keys}. Register the second settings under another key"
            )
        return value


# --- result objects --------------------------------------------------------------


class _Record(DazflowModel):
    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "These records are made by the build step."


class Coefficient(_Record):
    """One parameter: estimate, standard error, z and two-sided normal p-value.

    `gated` marks a driver's coefficient, which has the sign and
    significance checks; the constant, the trend and a form's own
    parameters (``ar.L1``, ``sigma2``) are reported, not gated.
    """

    term: str
    estimate: float
    std_error: float | None = None
    z: float | None = None
    pvalue: float | None = None
    expected_sign: int | None = None
    gated: bool = False


class Candidate(_Record):
    """One fitted model: its form, drivers, design, stored record and checks.

    `id` is ``<form>:<driver>+<driver>`` (the form key alone for a
    benchmark). `record` is ``None`` when the fit failed (`error` says
    why). `gate` holds the candidate's own checks: ``sign:<driver>``,
    ``significance:<driver>`` and ``vif``. `passed` is whether they all
    pass or are overridden.
    """

    id: str
    role: Literal["benchmark", "candidate"]
    form: str
    drivers: tuple[str, ...] = ()
    design: Design
    record: ModelRecord | None = None
    error: str | None = None
    coefficients: tuple[Coefficient, ...] = ()
    max_vif: float | None = None
    gate: Gate = Field(default_factory=Gate)
    passed: bool = False


class BuildResult(StepResult):
    """What build returns: every candidate fitted, and the passing ones.

    `fingerprint` is the dataset build ran on (in levels, scenarios left
    out); `sample` the common periods every candidate was fitted on;
    `chains` and `lags` how each series was transformed and each driver
    lagged; `trend_series` the series that call for a trend term.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BuildConfig
    fingerprint: str | None = None
    explore_fingerprint: str | None = None
    target: str | None = None
    drivers: tuple[str, ...] = ()
    expected_signs: dict[str, int] = Field(default_factory=dict)
    lags: dict[str, int] = Field(default_factory=dict)
    chains: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    trend_series: tuple[str, ...] = ()
    sample: Sample | None = None
    correlation: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    vif: dict[str, float | None] = Field(default_factory=dict)
    combinations: tuple[tuple[str, ...], ...] = ()
    candidates: tuple[Candidate, ...] = ()
    passing: tuple[str, ...] = ()

    def candidate(self, id: str) -> Candidate:
        """The candidate with identifier `id`."""
        for c in self.candidates:
            if c.id == id:
                return c
        raise BuildError(
            f"no candidate {id!r}",
            observed=f"candidates: {', '.join(c.id for c in self.candidates) or 'none'}",
            change="Name one of the candidates build fitted.",
        )


Config = BuildConfig
Result = BuildResult


# --- search and multicollinearity ------------------------------------------------


def _fail(problem: str, observed: str, change: str) -> BuildError:
    return BuildError(problem, observed=observed, change=change)


def _check_exclusions(drivers: Sequence[str], exclusions: Sequence[Sequence[str]]) -> set:
    pairs = set()
    for pair in exclusions:
        unknown = [d for d in pair if d not in drivers]
        if unknown or len(pair) != 2:
            raise _fail(
                f"an exclusion names drivers that are not searched: {', '.join(unknown) or pair}",
                f"drivers: {', '.join(drivers) or 'none'}",
                "Name two of the searched drivers in each exclusion.",
            )
        pairs.add(frozenset(pair))
    return pairs


def _allowed(combo: Sequence[str], pairs: set) -> bool:
    return not any(frozenset(p) in pairs for p in itertools.combinations(combo, 2))


def combinations(
    drivers: Sequence[str], max_size: int, exclusions: Sequence[Sequence[str]] = ()
) -> tuple[tuple[str, ...], ...]:
    """Every set of 1 to `max_size` drivers that holds no excluded pair.

    Sets are ordered by size, then as `itertools.combinations` gives them
    in the order of `drivers`. Without exclusions there are
    ``sum(comb(n, k) for k in 1..max_size)`` of them.
    """
    pairs = _check_exclusions(drivers, exclusions)
    out = []
    for k in range(1, min(max_size, len(drivers)) + 1):
        out += [c for c in itertools.combinations(drivers, k) if _allowed(c, pairs)]
    return tuple(out)


def _criterion(y: np.ndarray, X: np.ndarray, kind: str) -> float:
    import statsmodels.api as sm

    fit = sm.OLS(y, np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))).fit()
    return float(fit.aic if kind == "aic" else fit.bic)


def stepwise(
    y: pd.Series,
    X: pd.DataFrame,
    max_size: int,
    exclusions: Sequence[Sequence[str]] = (),
    criterion: Literal["aic", "bic"] = "aic",
) -> tuple[tuple[str, ...], ...]:
    """Forward stepwise search: the sets met adding, each step, the driver that lowers the criterion most.

    Each step fits OLS on a constant and the current drivers plus one more,
    skipping a driver that would join an excluded pair, and keeps the one
    with the lowest AIC (or BIC). The search stops at `max_size` drivers or
    when no addition lowers the criterion. Drivers are in the order added.
    """
    drivers = [str(c) for c in X.columns]
    pairs = _check_exclusions(drivers, exclusions)
    yv = y.to_numpy(dtype=float)
    current: list[str] = []
    best = _criterion(yv, np.empty((len(yv), 0)), criterion)
    path = []
    while len(current) < max_size:
        scores = {
            d: _criterion(yv, X[[*current, d]].to_numpy(dtype=float), criterion)
            for d in drivers
            if d not in current and _allowed([*current, d], pairs)
        }
        if not scores:
            break
        d = min(scores, key=lambda k: (scores[k], drivers.index(k)))
        if not scores[d] < best:
            break
        best = scores[d]
        current.append(d)
        path.append(tuple(current))
    return tuple(path)


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors: ``1 / (1 - R²)`` of each column on a constant and the others.

    R² is centred, as in statsmodels' `variance_inflation_factor` on a
    design with a constant. A column the others explain exactly has an
    infinite VIF.
    """
    values = X.to_numpy(dtype=float)
    n, k = values.shape
    out = []
    for j in range(k):
        x = values[:, j]
        others = np.column_stack([np.ones(n), np.delete(values, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, x, rcond=None)
        ssr = float(np.sum((x - others @ beta) ** 2))
        sst = float(np.sum((x - x.mean()) ** 2))
        out.append(math.inf if ssr <= 0 or sst <= 0 else sst / ssr)
    return pd.Series(out, index=list(X.columns), dtype=float, name="vif")


# --- the step ----------------------------------------------------------------------


def _numeric(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _roles(cfg: BuildConfig, ds: Dataset, explore: Any) -> tuple[str, list[str]]:
    numeric = _numeric(ds.frame)
    listed = f"numeric series: {', '.join(numeric) or 'none'}"
    target = cfg.target or getattr(explore, "target", None)
    if target is None:
        targets = [c for c in ds.targets if c in numeric]
        if len(targets) != 1:
            raise _fail(
                "build needs one target and the configuration names none",
                f"targets: {', '.join(targets) or 'none'}",
                "Give one series role target, or name it in target.",
            )
        target = targets[0]
    if target not in numeric:
        raise _fail(f"the target {target!r} is not a numeric series", listed, "Name the target.")
    if cfg.drivers is not None:
        drivers = list(cfg.drivers)
    elif explore is not None:
        drivers = list(explore.kept)
    else:
        drivers = [c for c in ds.drivers if c in numeric and c != target]
    bad = [d for d in drivers if d not in numeric or d in (target, TREND)]
    if bad or len(set(drivers)) != len(drivers):
        raise _fail(
            f"drivers names series that are not numeric, are the target, repeat or are called "
            f"{TREND!r}: {', '.join(bad) or 'a repeated name'}",
            listed,
            "Name each driver once, from the dataset's numeric series other than the target.",
        )
    stray = [d for d in [*cfg.expected_signs, *cfg.lags] if d not in drivers]
    if stray:
        raise _fail(
            f"a sign or lag is given for series that are not drivers: {', '.join(stray)}",
            f"drivers: {', '.join(drivers) or 'none'}",
            "Give signs and lags for the searched drivers only.",
        )
    return target, drivers


def _form(spec: FormSpec, want_drivers: bool) -> Any:
    try:
        cls = models.get(spec.form)
    except UnknownKeyError as err:
        raise _fail(
            f"no model form {spec.form!r} is registered",
            f"forms: {', '.join(err.known)}",
            "Name registered forms in forms and benchmarks.",
        ) from None
    takes = bool(getattr(cls, "drivers", False))
    if takes != want_drivers:
        where = "forms" if want_drivers else "benchmarks"
        raise _fail(
            f"the {spec.form} form {'takes no' if want_drivers else 'takes'} drivers, so it is "
            f"not one of the {where}",
            f"{where}: {spec.form}",
            "List forms with drivers in forms and forms without them in benchmarks.",
        )
    return cls(spec.config)


def _sample(frame: pd.DataFrame) -> pd.PeriodIndex:
    full = frame.notna().all(axis=1)
    if not full.any():
        raise _fail(
            "no period has the target and every driver present",
            f"{len(frame)} periods",
            "Give the target and drivers a common history.",
        )
    first, last = full.idxmax(), full[::-1].idxmax()
    inside = full.loc[first:last]
    if not inside.all():
        gap = inside.index[~inside.to_numpy()][0]
        raise _fail(
            "a value is missing inside the common sample of the target and drivers",
            f"first gap at {gap}; sample {first}..{last}",
            "Fill or exclude the gap in clean, or drop the driver.",
        )
    return frame.loc[first:last].index


def _coefficients(fitted: Any, drivers: Sequence[str], signs: Mapping[str, int]) -> list:
    from scipy import stats

    params = fitted.params
    cov = fitted.cov_params
    out = []
    for term in params.index:
        est = float(params[term])
        var = float(cov.loc[term, term])
        se = math.sqrt(var) if var >= 0 and math.isfinite(var) else math.nan
        z = est / se if se > 0 else math.nan
        p = float(2 * stats.norm.sf(abs(z))) if math.isfinite(z) else math.nan
        gated = term in drivers
        out.append(
            Coefficient(
                term=str(term),
                estimate=est,
                std_error=se if math.isfinite(se) else None,
                z=z if math.isfinite(z) else None,
                pvalue=p if math.isfinite(p) else None,
                expected_sign=signs.get(term) if gated else None,
                gated=gated,
            )
        )
    return out


def _checks(coefs: Sequence[Coefficient], max_vif: float, cfg: BuildConfig) -> list[Check]:
    checks = []
    for c in coefs:
        if not c.gated:
            continue
        if c.expected_sign in (-1, 1):
            checks.append(
                Check(
                    name=f"sign:{c.term}",
                    observed=c.estimate * c.expected_sign,
                    threshold=0.0,
                    op=">",
                )
            )
        checks.append(
            Check(
                name=f"significance:{c.term}",
                observed=c.pvalue if c.pvalue is not None else math.nan,
                threshold=cfg.significance,
                op="<=",
            )
        )
    checks.append(Check(name="vif", observed=max_vif, threshold=cfg.max_vif, op="<="))
    return checks


def run(
    dataset: Dataset,
    config: BuildConfig | Mapping[str, Any] | None = None,
    *,
    explore: Any = None,
) -> BuildResult:
    """Search driver combinations and fit the forms and benchmarks on the series in levels.

    `dataset` holds the series in levels, as explore received them;
    `explore` (its result, optional) gives the target, the kept drivers,
    their lags and signs, each series' transform chain and which series are
    trend-stationary. Without it the series are used as they are, at lag 0.
    The caller's `dataset` is not changed. Prints nothing and writes no files.
    """
    cfg = config if isinstance(config, BuildConfig) else BuildConfig.model_validate(config or {})
    if dataset.is_panel:
        raise _fail(
            "build does not take panel datasets yet",
            f"a panel of {len(dataset.segments)} segments",
            "Build each segment's dataset on its own.",
        )
    plain = dataset.with_scenarios(None) if dataset.scenarios is not None else dataset
    fp = plain.fingerprint()
    if explore is not None and explore.input_fingerprint not in (fp, dataset.fingerprint()):
        raise _fail(
            "the dataset is not the one explore ran on",
            f"fingerprint {fp[:12]}...; explore's input {str(explore.input_fingerprint)[:12]}...",
            "Pass build the dataset explore received (in levels), with explore's result.",
        )
    target, drivers = _roles(cfg, dataset, explore)
    series = [target, *drivers]
    chains = {s: tuple(getattr(explore, "chains", {}).get(s, ())) for s in series}
    lags = {d: int(getattr(explore, "lags", {}).get(d, 0)) for d in drivers} | dict(cfg.lags)
    screened = {s.driver: s.expected_sign for s in getattr(explore, "screens", ())}
    signs: dict[str, int] = {}
    for d in drivers:
        sign = cfg.expected_signs.get(d, screened.get(d, dataset.series[d].expected_sign))
        if sign in (-1, 1):
            signs[d] = int(sign)
    trendy = tuple(
        i.series
        for i in getattr(explore, "integration", ())
        if i.series in series
        and i.kind == "trend-stationary"
        and not any(t.startswith(_DIFF) for t in chains.get(i.series, ()))
    )

    levels = dataset.frame
    first = str(levels.index[0])
    probe = Design(
        target=target, drivers=tuple(drivers), lags=lags, chains=chains, start=first, end=first
    )
    moved = pd.DataFrame(
        {target: probe._series(levels, target)}
        | {d: probe._series(levels, d).shift(lags[d]) for d in drivers}
    )
    sample = _sample(moved)
    start, end = str(sample[0]), str(sample[-1])
    on_sample = moved.loc[sample]

    X_all = on_sample[drivers]
    corr = X_all.corr() if drivers else pd.DataFrame()
    vifs = vif(X_all) if drivers else pd.Series(dtype=float)

    if cfg.search == "exhaustive":
        combos = combinations(drivers, cfg.max_size, cfg.exclusions)
    else:
        combos = stepwise(on_sample[target], X_all, cfg.max_size, cfg.exclusions, cfg.criterion)

    own: dict[str, list[Override]] = {}
    rest: list[Override] = []
    for ov in cfg.overrides:
        cid, sep, check = ov.check.partition("/")
        if sep:
            own.setdefault(cid, []).append(ov.model_copy(update={"check": check}))
        else:
            rest.append(ov)

    def design(combo: Sequence[str], with_trend: bool) -> Design:
        return Design(
            target=target,
            drivers=tuple(combo),
            lags={d: lags[d] for d in combo},
            chains={s: chains[s] for s in (target, *combo) if chains[s]},
            trend=with_trend,
            trend_start=first if with_trend else None,
            start=start,
            end=end,
        )

    candidates: list[Candidate] = []
    fitted_ids: set[str] = set()
    for spec in cfg.benchmarks:
        form = _form(spec, want_drivers=False)
        d = design((), False)
        candidates.append(_fit(spec.form, "benchmark", form, d, levels, signs, cfg, []))
        fitted_ids.add(spec.form)

    forms = [(spec, _form(spec, want_drivers=True)) for spec in cfg.forms]
    for combo in combos:
        needs_trend = cfg.trend == "always" or (
            cfg.trend == "auto" and any(s in trendy for s in (target, *combo))
        )
        for spec, form in forms:
            cid = f"{spec.form}:{'+'.join(combo)}"
            d = design(combo, needs_trend)
            cand = _fit(cid, "candidate", form, d, levels, signs, cfg, own.get(cid, []))
            candidates.append(cand)
            fitted_ids.add(cid)
            if cand.passed:
                break

    stray = sorted(set(own) - fitted_ids)
    if stray:
        raise GateError(
            f"override for checks of candidates that were not fitted: {', '.join(stray)}",
            observed=f"candidates fitted: {', '.join(sorted(fitted_ids)) or 'none'}",
            change="Name a fitted candidate's check, as in ols:gdp/significance:gdp.",
        )
    passing = tuple(c.id for c in candidates if c.role == "candidate" and c.passed)
    gate = Gate.evaluate(
        [
            Check(
                name="candidates",
                observed=float(len(passing)),
                threshold=float(cfg.min_candidates),
                op=">=",
            )
        ],
        rest,
    )

    n_bench = sum(1 for c in candidates if c.role == "benchmark")
    finding = Finding(
        template=SUMMARY,
        key="build.summary",
        values={
            "candidates": len(candidates) - n_bench,
            "combinations": len(combos),
            "size": cfg.max_size,
            "benchmarks": n_bench,
            "passing": len(passing),
            "significance": cfg.significance,
            "max_vif": cfg.max_vif,
        },
    )
    artefacts: dict[str, Any] = {
        IDS.table("correlation"): _correlation_table(drivers, corr, vifs, cfg),
        IDS.table("candidates"): _candidate_table(candidates),
        IDS.table("coefficients"): _coefficient_table(candidates),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Build checks"),
        IDS.finding("summary"): finding,
    }
    figure = _coefficient_figure(candidates, cfg.significance)
    if figure is not None:
        artefacts[IDS.figure("coefficients")] = figure

    def _f(v: Any) -> float | None:
        return float(v) if v is not None and math.isfinite(v) else None

    return BuildResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        fingerprint=fp,
        explore_fingerprint=getattr(explore, "fingerprint", None),
        target=target,
        drivers=tuple(drivers),
        expected_signs=signs,
        lags=lags,
        chains=chains,
        trend_series=trendy,
        sample=Sample(start=start, end=end, observations=len(sample), frequency=sample.freqstr),
        correlation={a: {b: _f(corr.loc[a, b]) for b in drivers} for a in drivers},
        vif={d: float(vifs[d]) for d in drivers},
        combinations=tuple(combos),
        candidates=tuple(candidates),
        passing=passing,
    )


def _fit(
    cid: str,
    role: str,
    form: Any,
    design: Design,
    levels: pd.DataFrame,
    signs: Mapping[str, int],
    cfg: BuildConfig,
    overrides: Sequence[Override],
) -> Candidate:
    y, X = design.frames(levels)
    base: dict[str, Any] = {
        "id": cid,
        "role": role,
        "form": form.key or type(form).__name__,
        "drivers": design.drivers,
        "design": design,
    }
    try:
        fitted = form.fit(y, X)
    except ModelError as err:
        gate = Gate.evaluate([Check(name="fit", observed=math.nan, threshold=0.0, op="<=")])
        return Candidate(**base, error=err.problem, gate=gate, passed=False)
    if role == "benchmark":
        return Candidate(
            **base,
            record=fitted.record(),
            coefficients=tuple(_coefficients(fitted, (), {})),
            passed=True,
        )
    coefs = _coefficients(fitted, design.drivers, signs)
    worst = float(vif(X[list(design.drivers)]).max())
    gate = Gate.evaluate(_checks(coefs, worst, cfg), overrides)
    return Candidate(
        **base,
        record=fitted.record(),
        coefficients=tuple(coefs),
        max_vif=worst,
        gate=gate,
        passed=gate.passed,
    )


# --- artefacts ---------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _correlation_table(
    drivers: list[str], corr: pd.DataFrame, vifs: pd.Series, cfg: BuildConfig
) -> Table:
    columns = [Column(name="driver", label="Driver", kind="text")]
    columns += [Column(name=f"r_{d}", label=d, kind="number") for d in drivers]
    columns.append(Column(name="vif", label="VIF", kind="number"))
    rows = [[a, *[_finite(corr.loc[a, b]) for b in drivers], _finite(vifs[a])] for a in drivers]
    return Table(
        id=IDS.table("correlation"),
        caption="Correlation and variance inflation of the drivers",
        columns=columns,
        rows=rows,
        notes=(
            "Correlations of the transformed, lagged drivers on the common sample; the VIF is "
            f"1 / (1 - R²) of each driver on a constant and the others (a candidate fails above "
            f"{cfg.max_vif:g}).",
        ),
    )


def _status(c: Candidate) -> str:
    if not c.passed:
        return "fail"
    return "overridden" if c.gate.overridden else "pass"


def _candidate_table(candidates: Sequence[Candidate]) -> Table:
    columns = [
        Column(name="candidate", label="Candidate", kind="text"),
        Column(name="role", label="Role", kind="text"),
        Column(name="form", label="Form", kind="text"),
        Column(name="drivers", label="Drivers", kind="text"),
        Column(name="size", label="Size", kind="integer"),
        Column(name="lags", label="Lags", kind="text"),
        Column(name="trend", label="Trend", kind="text"),
        Column(name="max_vif", label="Largest VIF", kind="number"),
        Column(name="failed", label="Failed checks", kind="text"),
        Column(name="result", label="Result", kind="status"),
    ]
    rows = []
    for c in candidates:
        failed = c.error or ", ".join(k.name for k in c.gate.failed)
        rows.append(
            [
                c.id,
                c.role,
                c.form,
                ", ".join(c.drivers) or "none",
                len(c.drivers),
                ", ".join(str(c.design.lags.get(d, 0)) for d in c.drivers) or "",
                "yes" if c.design.trend else "no",
                _finite(c.max_vif),
                failed,
                _status(c),
            ]
        )
    return Table(
        id=IDS.table("candidates"),
        caption="Candidate models",
        columns=columns,
        rows=rows,
        notes=(
            "Forms are fitted simplest first for each set of drivers; the next form is fitted "
            "only when the simpler one fails a check. Benchmarks are always fitted.",
        ),
    )


_SIGN_TEXT = {-1: "negative", 1: "positive", None: ""}


def _coefficient_table(candidates: Sequence[Candidate]) -> Table:
    columns = [
        Column(name="candidate", label="Candidate", kind="text"),
        Column(name="term", label="Term", kind="text"),
        Column(name="estimate", label="Estimate", kind="number"),
        Column(name="std_error", label="Standard error", kind="number"),
        Column(name="z", label="z", kind="number"),
        Column(name="pvalue", label="p-value", kind="number"),
        Column(name="expected", label="Expected sign", kind="text"),
        Column(name="checked", label="Checked", kind="text"),
    ]
    rows = []
    for c in candidates:
        for k in c.coefficients:
            if k.gated:
                mine = [x for x in c.gate.checks if x.name.endswith(f":{k.term}")]
                state = (
                    "fail"
                    if any(x.status == "fail" for x in mine)
                    else "overridden"
                    if any(x.status == "overridden" for x in mine)
                    else "pass"
                )
            else:
                state = "not checked"
            rows.append(
                [
                    c.id,
                    k.term,
                    _finite(k.estimate),
                    k.std_error,
                    k.z,
                    k.pvalue,
                    _SIGN_TEXT.get(k.expected_sign, ""),
                    state,
                ]
            )
    return Table(
        id=IDS.table("coefficients"),
        caption="Coefficients of the candidate models",
        columns=columns,
        rows=rows,
        notes=(
            "p-values are two-sided, from the estimate over its standard error read on the "
            "normal distribution; only the drivers' coefficients are checked.",
        ),
    )


def _coefficient_figure(candidates: Sequence[Candidate], significance: float) -> Figure | None:
    from scipy import stats

    labels, z = [], []
    for c in candidates:
        for k in c.coefficients:
            if k.gated:
                labels.append(f"{c.id} {k.term}")
                z.append(k.z)
    if not labels:
        return None
    crit = float(stats.norm.ppf(1 - significance / 2))
    return Figure(
        id=IDS.figure("coefficients"),
        caption="Driver coefficients of every candidate, as z statistics",
        x=labels,
        layers=(
            Layer(kind="bar", y=z, label="z"),
            Layer(kind="hline", value=crit, label=f"±{crit:.2f}"),
            Layer(kind="hline", value=-crit),
        ),
        x_label="Candidate and driver",
        y_label="z statistic",
    )
