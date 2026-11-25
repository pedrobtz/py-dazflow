"""select: rank the candidates out of sample and propose a default (design §7.6).

Every candidate build fitted, and every benchmark, is refitted at each
origin of a rolling-origin evaluation and scored on its forecasts; its
error is scaled by the configured benchmark's. Arch's model confidence set
(`arch.bootstrap.MCS`) on the per-origin losses says which candidates are
not distinguishable from the best, from a seeded bootstrap. Each candidate
is then forecast under every scenario of `Dataset.scenarios`, mapped back
to the target's levels, and checked for ordering by severity and for the
configured bounds.

The default rule (design D38): among the candidates that pass every check
(build's and these), take those in the confidence set; choose the one with
the fewest drivers, then the lowest scaled error. With none, no model is
proposed and the ``confidence-set`` check fails. The modeller confirms the
proposal or overrides it in configuration (`choice`), with a reason.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from dazflow import evaluation as _evaluation
from dazflow import models, readers, transforms
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import ModelError, ResultError, SelectError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import StepResult
from dazflow.evaluation import Evaluation
from dazflow.models import Design, Forecast, ModelRecord
from dazflow.readers._base import Frequency

__all__ = [
    "IDS",
    "CandidateEvaluation",
    "Choice",
    "ConfidenceSet",
    "Config",
    "MCSConfig",
    "OrderingConfig",
    "Ranked",
    "Result",
    "RuleRow",
    "ScenarioForecast",
    "ScenarioInput",
    "SelectConfig",
    "SelectResult",
    "confidence_set",
    "default_rule",
    "order_check",
    "out_of_order",
    "read_scenarios",
    "run",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "select",
    tables={"ranking": 1, "evaluation": 2, "scenarios": 3, "selection": 4, "gate": 5},
    figures={"backtest": 1, "scenarios": 2},
    findings={"summary": 1},
)

SUMMARY = (
    "{{ evaluated }} models were scored at {{ origins }} origins, their errors scaled by the "
    "{{ benchmark }} benchmark's; the model confidence set at {{ size }} holds {{ in_set }} of "
    "them, {{ eligible }} passing every check. {{ chosen }} is {{ decision }}: {{ drivers }} "
    "drivers and a scaled error of {{ error }}."
)
EMPTY = (
    "{{ evaluated }} models were scored at {{ origins }} origins, their errors scaled by the "
    "{{ benchmark }} benchmark's; the model confidence set at {{ size }} holds {{ in_set }} of "
    "them, and no candidate that passes every check is in it, so no model is proposed."
)


# --- configuration ---------------------------------------------------------------


class MCSConfig(Config):
    """The model confidence set: arch's `MCS` on the per-origin losses."""

    size: float = Field(
        0.10, gt=0, lt=1, description="Size of the set's test: the set holds the best at 1 - size."
    )
    reps: int = Field(1000, ge=10, description="Bootstrap replications.")
    block_size: int | None = Field(
        None, ge=1, description="Bootstrap block length; unset: arch's default."
    )
    method: Literal["R", "max"] = Field("R", description="arch's elimination statistic.")
    bootstrap: Literal["stationary", "circular", "moving block"] = "stationary"
    seed: int | None = Field(
        None, description="Bootstrap seed; unset: project.yaml's seed (0 outside a project)."
    )


class ScenarioInput(Config):
    """A CSV of driver paths in levels: a scenario label column, dates and one column per driver."""

    path: str = Field(min_length=1, description="The file, relative to the project folder.")
    scenario: str = Field("scenario", description="The column holding the scenario labels.")
    date: str | None = Field(
        None, description="The column holding the dates; unset: the first other column."
    )
    frequency: Frequency | None = Field(None, description="Unset: read from the dates.")
    severity: list[str] | None = Field(
        None, description="Scenario labels, mildest first; unset: their order in the file."
    )


class OrderingConfig(Config):
    """How projections are compared across scenarios."""

    statistic: Literal["mean", "last", "max", "min"] = Field(
        "mean", description="What of each projection, in levels, is compared."
    )
    direction: Literal[-1, 1] = Field(
        1, description="1: a more severe scenario projects a higher target; -1: a lower one."
    )


class Choice(Config):
    """The modeller's decision: the proposed candidate confirmed, or another, with a reason."""

    candidate: str = Field(min_length=1, description="The chosen candidate's id.")
    reason: str = Field(description="Why; required, whether it confirms or overrides.")

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a choice needs a reason; it is empty or blank")
        return value


class SelectConfig(Config):
    """Configuration of the select step."""

    benchmark: str = Field(
        "mean", description="The benchmark (one build fitted) whose errors scale every other's."
    )
    splitter: str = Field("expanding", description="Rolling-origin splitter, by registry key.")
    splitter_config: dict[str, Any] = Field(
        default_factory=dict,
        description="The splitter's settings; unset: half the sample first, one-step folds.",
    )
    loss: Literal["squared", "absolute"] = Field(
        "squared", description="Per-origin loss: squared (RMSE ratios) or absolute (MAE ratios)."
    )
    levels: list[float] = Field(
        default_factory=lambda: [0.8, 0.95], description="Interval levels forecast."
    )
    benchmark_ratio: float = Field(
        1.0, gt=0, description="A candidate beats the benchmark with a scaled error below this."
    )
    mcs: MCSConfig = Field(default_factory=MCSConfig)
    scenarios: ScenarioInput | None = Field(
        None, description="Scenario paths read from a file; unset: the dataset's own scenarios."
    )
    ordering: OrderingConfig = Field(default_factory=OrderingConfig)
    bounds: tuple[float | None, float | None] | None = Field(
        None,
        description="[lower, upper] for every projection in the target's levels; null leaves "
        "a side open; unset: no bounds check.",
    )
    choice: Choice | None = Field(
        None, description="Confirm the proposal or choose another: {candidate: ..., reason: ...}."
    )
    overrides: list[Override] = Field(default_factory=list)


# --- result objects --------------------------------------------------------------


class _Record(DazflowModel):
    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "These records are made by the select step."


class ConfidenceSet(_Record):
    """The candidates arch's MCS keeps at `size`, in the losses' column order, and every p-value."""

    included: tuple[str, ...]
    pvalues: dict[str, float]
    size: float
    reps: int
    block_size: int | None = None
    method: str = "R"
    bootstrap: str = "stationary"
    seed: int = 0


class RuleRow(_Record):
    """One candidate as the default rule sees it."""

    candidate: str
    drivers: int
    error: float
    passed: bool
    in_set: bool


class CandidateEvaluation(_Record):
    """A candidate's rolling-origin evaluation, or why it has none (`note`)."""

    candidate: str
    evaluation: Evaluation | None = None
    note: str | None = None


class ScenarioForecast(_Record):
    """A candidate's forecast under one scenario, and its mean path in the target's levels."""

    candidate: str
    scenario: str | int
    forecast: Forecast
    levels: tuple[float, ...]


class Ranked(_Record):
    """One row of the ranking: scores, set membership and every check select makes."""

    rank: int | None = None
    candidate: str
    role: Literal["benchmark", "candidate"]
    form: str
    drivers: tuple[str, ...] = ()
    size: int = 0
    scaled_error: float | None = None
    in_set: bool = False
    pvalue: float | None = None
    build_passed: bool = False
    beats_benchmark: bool | None = None
    statistics: dict[str, float] = Field(default_factory=dict)
    out_of_order: int | None = None
    out_of_bounds: int | None = None
    eligible: bool = False
    note: str | None = None


class SelectResult(StepResult):
    """What select returns: the ranking, the confidence set, the scenarios and the choice.

    `record` is the chosen candidate's `ModelRecord` (its full-sample fit),
    `design` how its target and drivers are built from the levels,
    `drivers` its drivers, and `scenario_forecasts` every candidate under
    every scenario, which forecast (Stage 9) projects from. `decision` is
    ``proposed`` (by the default rule, not yet confirmed), ``confirmed``,
    ``overridden`` or ``none``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SelectConfig
    fingerprint: str | None = None
    target: str | None = None
    benchmark: str | None = None
    seed: int = 0
    origins: tuple[str, ...] = ()
    losses: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    evaluations: tuple[CandidateEvaluation, ...] = ()
    confidence: ConfidenceSet | None = None
    ranking: tuple[Ranked, ...] = ()
    severity: tuple[str | int, ...] = ()
    scenario_forecasts: tuple[ScenarioForecast, ...] = ()
    proposal: str | None = None
    chosen: str | None = None
    decision: Literal["proposed", "confirmed", "overridden", "none"] = "none"
    reason: str | None = None
    record: ModelRecord | None = None
    design: Design | None = None
    drivers: tuple[str, ...] = ()

    def evaluation(self, candidate: str) -> Evaluation | None:
        """The rolling-origin evaluation of `candidate`."""
        for e in self.evaluations:
            if e.candidate == candidate:
                return e.evaluation
        raise SelectError(
            f"no evaluation of {candidate!r}",
            observed=f"candidates: {', '.join(e.candidate for e in self.evaluations)}",
            change="Name a candidate build fitted.",
        )


Config = SelectConfig
Result = SelectResult


# --- the pieces --------------------------------------------------------------------


def _fail(problem: str, observed: str, change: str) -> SelectError:
    return SelectError(problem, observed=observed, change=change)


def confidence_set(
    losses: pd.DataFrame,
    *,
    size: float = 0.10,
    reps: int = 1000,
    block_size: int | None = None,
    method: Literal["R", "max"] = "R",
    bootstrap: str = "stationary",
    seed: int = 0,
) -> ConfidenceSet:
    """Arch's model confidence set on `losses` (one row per origin, one column per model).

    The call is ``arch.bootstrap.MCS(losses, size, reps, block_size,
    method, bootstrap, seed=seed).compute()``. A single model is its own set.
    """
    from arch.bootstrap import MCS

    names = [str(c) for c in losses.columns]
    if len(names) < 2:
        return ConfidenceSet(
            included=tuple(names),
            pvalues=dict.fromkeys(names, 1.0),
            size=size,
            reps=reps,
            block_size=block_size,
            method=method,
            bootstrap=bootstrap,
            seed=seed,
        )
    mcs = MCS(
        losses,
        size,
        reps=reps,
        block_size=block_size,
        method=method,
        bootstrap=bootstrap,
        seed=seed,
    )
    mcs.compute()
    kept = {str(c) for c in mcs.included}
    pv = mcs.pvalues["Pvalue"]
    return ConfidenceSet(
        included=tuple(n for n in names if n in kept),
        pvalues={n: float(pv[n]) for n in names},
        size=size,
        reps=reps,
        block_size=block_size,
        method=method,
        bootstrap=bootstrap,
        seed=seed,
    )


def default_rule(rows: Sequence[RuleRow]) -> str | None:
    """The default choice (design D38), or ``None`` when no passing candidate is in the set.

    Among the rows that pass every check and are in the confidence set:
    the fewest drivers, then the lowest scaled error, then the first listed.
    """
    pool = [
        (r.drivers, r.error, i, r.candidate) for i, r in enumerate(rows) if r.passed and r.in_set
    ]
    return min(pool)[3] if pool else None


def out_of_order(values: Mapping[Any, float], severity: Sequence[Any], direction: int = 1) -> int:
    """How many successive scenarios, mildest first, project a less severe target than the one before.

    A more severe scenario should give a higher value (`direction` 1) or a
    lower one (-1). Equal values are not out of order.
    """
    v = [values[s] for s in severity]
    return sum(1 for a, b in itertools.pairwise(v) if direction * (b - a) < 0)


def order_check(values: Mapping[Any, float], severity: Sequence[Any], direction: int = 1) -> Check:
    """The ``scenario-order`` check: no pair of successive scenarios out of order."""
    return Check(
        name="scenario-order",
        observed=float(out_of_order(values, severity, direction)),
        threshold=0.0,
        op="<=",
    )


def read_scenarios(
    spec: ScenarioInput, root: str | Path | None = None
) -> tuple[pd.DataFrame, list]:
    """The scenario frame, indexed by ``(scenario, period)``, and the severity order, from a CSV.

    A relative path is read from `root` (the working directory by default).
    """
    path = Path(spec.path)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise _fail(
            f"the scenarios file {spec.path!r} does not exist",
            f"looked for {path}",
            "Give the path of a CSV of scenarios, relative to the project folder.",
        )
    raw = pd.read_csv(path, dtype={spec.scenario: str})
    if spec.scenario not in raw.columns:
        raise _fail(
            f"the scenarios file has no column {spec.scenario!r}",
            f"columns: {', '.join(map(str, raw.columns))}",
            "Name the column holding the scenario labels in scenarios.scenario.",
        )
    parts = []
    for label in dict.fromkeys(raw[spec.scenario]):
        part = raw.loc[raw[spec.scenario] == label].drop(columns=spec.scenario)
        ds = readers.from_frame(
            part.reset_index(drop=True), date=spec.date, frequency=spec.frequency
        )
        frame = ds.frame
        frame.index = pd.MultiIndex.from_arrays(
            [[label] * len(frame), frame.index], names=["scenario", "period"]
        )
        parts.append(frame)
    if not parts:
        raise _fail("the scenarios file has no rows", f"{path}", "Give at least one scenario.")
    severity = (
        spec.severity if spec.severity is not None else list(dict.fromkeys(raw[spec.scenario]))
    )
    return pd.concat(parts), severity


def _stat(path: Sequence[float], statistic: str) -> float:
    a = np.asarray(path, dtype=float)
    return float({"mean": a.mean, "last": lambda: a[-1], "max": a.max, "min": a.min}[statistic]())


def _per_fold(ev: Evaluation, loss: str) -> list[float]:
    by: dict[int, list[float]] = {}
    for r in ev.rows:
        e = r.actual - r.forecast
        by.setdefault(r.fold, []).append(e * e if loss == "squared" else abs(e))
    return [float(np.mean(by[f.number])) for f in ev.folds]


def _overall(ev: Evaluation, loss: str) -> float:
    e = np.array([r.actual - r.forecast for r in ev.rows])
    return float(np.sqrt(np.mean(e * e)) if loss == "squared" else np.mean(np.abs(e)))


# --- the step ----------------------------------------------------------------------


def run(
    dataset: Dataset,
    config: SelectConfig | Mapping[str, Any] | None = None,
    *,
    build: Any,
    seed: int | None = None,
) -> SelectResult:
    """Score, rank and check every candidate of `build`, and propose a default.

    `dataset` holds the series in levels that build ran on, with any
    scenarios in `dataset.scenarios` (or read from `config.scenarios`);
    `build` is build's result. The bootstrap seed is `config.mcs.seed`, else
    `seed` (project.yaml's), else 0. The caller's `dataset` is not changed.
    Prints nothing and writes no files.
    """
    cfg = config if isinstance(config, SelectConfig) else SelectConfig.model_validate(config or {})
    plain = dataset.with_scenarios(None) if dataset.scenarios is not None else dataset
    if plain.fingerprint() != build.fingerprint:
        raise _fail(
            "the dataset is not the one build ran on",
            f"fingerprint {plain.fingerprint()[:12]}...; build's {str(build.fingerprint)[:12]}...",
            "Pass select the dataset build received (in levels), with build's result.",
        )
    if dataset.scenarios is not None:
        scen, severity = dataset.scenarios, list(dataset.severity)
    elif cfg.scenarios is not None:
        frame, sev = read_scenarios(cfg.scenarios)
        scen_ds = plain.with_scenarios(frame, sev)
        scen, severity = scen_ds.scenarios, list(scen_ds.severity)
    else:
        scen, severity = None, []
    use_seed = cfg.mcs.seed if cfg.mcs.seed is not None else (seed if seed is not None else 0)
    levels = plain.frame
    target = build.target
    by_id = {c.id: c for c in build.candidates}
    bench = by_id.get(cfg.benchmark)
    if bench is None or bench.role != "benchmark" or bench.record is None:
        known = [c.id for c in build.candidates if c.role == "benchmark" and c.record is not None]
        raise _fail(
            f"the benchmark {cfg.benchmark!r} is not one build fitted",
            f"benchmarks: {', '.join(known) or 'none'}",
            "Name one of build's benchmarks in benchmark, or add it to build's benchmarks.",
        )

    split = _evaluation.splitters.get(cfg.splitter)(cfg.splitter_config)
    scorers = [
        _evaluation.metrics.get("mae")(),
        _evaluation.metrics.get("rmse")(),
        _evaluation.metrics.get("mase")(),
        _evaluation.metrics.get("coverage")(level=max(cfg.levels)),
    ]

    evaluations: list[CandidateEvaluation] = []
    data: dict[str, tuple[pd.Series, pd.DataFrame | None]] = {}
    for c in build.candidates:
        if c.record is None:
            evaluations.append(CandidateEvaluation(candidate=c.id, note=c.error))
            continue
        y, X = c.design.frames(levels)
        data[c.id] = (y, X)
        try:
            ev = _evaluation.evaluate(
                c.record.spec, y, X, splitter=split, metrics=scorers, levels=cfg.levels
            )
        except ModelError as err:
            if c.id == cfg.benchmark:
                raise _fail(
                    f"the benchmark {c.id} cannot be evaluated",
                    err.problem,
                    "Give the splitter a longer first training window (splitter_config.initial).",
                ) from err
            evaluations.append(CandidateEvaluation(candidate=c.id, note=err.problem))
            continue
        evaluations.append(CandidateEvaluation(candidate=c.id, evaluation=ev))

    scored = {e.candidate: e.evaluation for e in evaluations if e.evaluation is not None}
    ref = scored[cfg.benchmark]
    origins = tuple(dict.fromkeys(r.origin for r in ref.rows))
    losses = {cid: tuple(_per_fold(ev, cfg.loss)) for cid, ev in scored.items()}
    base = _overall(ref, cfg.loss)
    scaled = {
        cid: _overall(ev, cfg.loss) / base if base > 0 else math.nan for cid, ev in scored.items()
    }

    conf = confidence_set(
        pd.DataFrame(losses, index=list(origins)),
        size=cfg.mcs.size,
        reps=cfg.mcs.reps,
        block_size=cfg.mcs.block_size,
        method=cfg.mcs.method,
        bootstrap=cfg.mcs.bootstrap,
        seed=use_seed,
    )

    forecasts: list[ScenarioForecast] = []
    stats: dict[str, dict[str, float]] = {}
    if scen is not None:
        for cid in scored:
            c = by_id[cid]
            y, X = data[cid]
            fitted = models.refit(c.record, y, X)
            stats[cid] = {}
            for label in severity:
                paths = scen.xs(label, level="scenario")
                ahead = paths.index[paths.index > pd.Period(c.design.end, freq=levels.index.freq)]
                if not len(ahead):
                    raise _fail(
                        f"the scenario {label!r} has no period after the sample",
                        f"sample ends {c.design.end}; scenario {paths.index[0]}..{paths.index[-1]}",
                        "Give scenario paths for the periods after the last observation.",
                    )
                h = len(ahead)
                fc = fitted.predict(h, c.design.future(levels, paths, h), levels=cfg.levels)
                path = _to_levels(fc, c.design, levels, target)
                forecasts.append(
                    ScenarioForecast(candidate=cid, scenario=label, forecast=fc, levels=path)
                )
                stats[cid][str(label)] = _stat(path, cfg.ordering.statistic)

    ranking = _rank(build, cfg, evaluations, scaled, conf, stats, severity, forecasts)
    rows = [
        RuleRow(
            candidate=r.candidate,
            drivers=r.size,
            error=r.scaled_error,
            passed=r.eligible,
            in_set=r.in_set,
        )
        for r in ranking
        if r.role == "candidate" and r.scaled_error is not None
    ]
    proposal = default_rule(rows)

    if cfg.choice is not None:
        chosen = cfg.choice.candidate
        if chosen not in by_id or by_id[chosen].record is None:
            raise _fail(
                f"the chosen candidate {chosen!r} is not one build fitted",
                f"candidates: {', '.join(c.id for c in build.candidates if c.record is not None)}",
                "Name a fitted candidate in choice.candidate.",
            )
        decision = "confirmed" if chosen == proposal else "overridden"
        reason = cfg.choice.reason
    else:
        chosen, reason = proposal, None
        decision = "proposed" if proposal is not None else "none"

    eligible_in_set = sum(1 for r in ranking if r.eligible and r.in_set)
    checks = [
        Check(
            name="confidence-set",
            observed=float(eligible_in_set),
            threshold=1.0,
            op=">=",
        )
    ]
    pick = next((r for r in ranking if r.candidate == chosen), None)
    if pick is not None:
        cand = by_id[chosen]
        checks.append(
            Check(name="build", observed=float(len(cand.gate.failed)), threshold=0.0, op="<=")
        )
        checks.append(
            Check(
                name="benchmark",
                observed=pick.scaled_error if pick.scaled_error is not None else math.nan,
                threshold=cfg.benchmark_ratio,
                op="<",
            )
        )
        if scen is not None and len(severity) > 1:
            checks.append(
                order_check(pick.statistics, [str(s) for s in severity], cfg.ordering.direction)
            )
        if scen is not None and cfg.bounds is not None:
            checks.append(
                Check(
                    name="scenario-bounds",
                    observed=float(pick.out_of_bounds or 0),
                    threshold=0.0,
                    op="<=",
                )
            )
    gate = Gate.evaluate(checks, cfg.overrides)

    values = {
        "evaluated": len(scored),
        "origins": len(origins),
        "benchmark": cfg.benchmark,
        "size": cfg.mcs.size,
        "in_set": len(conf.included),
        "eligible": eligible_in_set,
    }
    if pick is not None:
        finding = Finding(
            template=SUMMARY,
            key="select.summary",
            values={
                **values,
                "chosen": chosen,
                "decision": decision,
                "drivers": pick.size,
                "error": pick.scaled_error if pick.scaled_error is not None else "none",
            },
        )
    else:
        finding = Finding(template=EMPTY, key="select.empty", values=values)

    record = by_id[chosen].record if chosen is not None else None
    design = by_id[chosen].design if chosen is not None else None
    artefacts: dict[str, Any] = {
        IDS.table("ranking"): _ranking_table(ranking, proposal, chosen, cfg),
        IDS.table("evaluation"): _evaluation_table(evaluations, scaled),
        IDS.table("scenarios"): _scenario_table(ranking, severity, cfg),
        IDS.table("selection"): _selection_table(
            cfg, conf, proposal, chosen, decision, reason, len(origins)
        ),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Selection checks"),
        IDS.finding("summary"): finding,
    }
    artefacts[IDS.figure("backtest")] = _backtest_figure(scored, target)
    if chosen is not None and forecasts:
        fig = _scenario_figure(forecasts, chosen, levels[target], severity, target)
        if fig is not None:
            artefacts[IDS.figure("scenarios")] = fig

    return SelectResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        fingerprint=plain.fingerprint(),
        target=target,
        benchmark=cfg.benchmark,
        seed=use_seed,
        origins=origins,
        losses=losses,
        evaluations=tuple(evaluations),
        confidence=conf,
        ranking=tuple(ranking),
        severity=tuple(severity),
        scenario_forecasts=tuple(forecasts),
        proposal=proposal,
        chosen=chosen,
        decision=decision,
        reason=reason,
        record=record,
        design=design,
        drivers=by_id[chosen].drivers if chosen is not None else (),
    )


def _to_levels(
    fc: Forecast, design: Design, levels: pd.DataFrame, target: str
) -> tuple[float, ...]:
    chain = design.chains.get(target, ())
    mean = pd.Series(fc.mean, index=fc.index, name=target)
    if not chain:
        return tuple(float(v) for v in mean)
    end = pd.Period(design.end, freq=levels.index.freq)
    history = levels[target].astype("float64").loc[:end]
    back = transforms.inverse(mean, list(chain), history=history)
    return tuple(float(v) for v in back.loc[fc.index])


def _rank(
    build: Any,
    cfg: SelectConfig,
    evaluations: Sequence[CandidateEvaluation],
    scaled: Mapping[str, float],
    conf: ConfidenceSet,
    stats: Mapping[str, Mapping[str, float]],
    severity: Sequence[Any],
    forecasts: Sequence[ScenarioForecast],
) -> list[Ranked]:
    notes = {e.candidate: e.note for e in evaluations}
    out = []
    for c in build.candidates:
        err = scaled.get(c.id)
        err = err if err is not None and math.isfinite(err) else None
        st = dict(stats.get(c.id, {}))
        order = (
            out_of_order(st, [str(s) for s in severity], cfg.ordering.direction)
            if st and len(severity) > 1
            else None
        )
        bounds = None
        if cfg.bounds is not None and st:
            lo, hi = cfg.bounds
            vals = [v for f in forecasts if f.candidate == c.id for v in f.levels]
            bounds = sum(
                1 for v in vals if (lo is not None and v < lo) or (hi is not None and v > hi)
            )
        beats = None if err is None else err < cfg.benchmark_ratio
        eligible = (
            c.role == "candidate"
            and err is not None
            and c.passed
            and bool(beats)
            and order in (None, 0)
            and bounds in (None, 0)
        )
        out.append(
            Ranked(
                candidate=c.id,
                role=c.role,
                form=c.form,
                drivers=c.drivers,
                size=len(c.drivers),
                scaled_error=err,
                in_set=c.id in conf.included,
                pvalue=conf.pvalues.get(c.id),
                build_passed=c.passed,
                beats_benchmark=beats,
                statistics=st,
                out_of_order=order,
                out_of_bounds=bounds,
                eligible=eligible,
                note=notes.get(c.id),
            )
        )
    order_key = sorted(
        range(len(out)),
        key=lambda i: (out[i].scaled_error is None, out[i].scaled_error or 0.0, i),
    )
    ranked = []
    for rank, i in enumerate(order_key, start=1):
        r = out[i]
        ranked.append(r.model_copy(update={"rank": rank if r.scaled_error is not None else None}))
    return ranked


# --- artefacts ---------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _yes(v: bool | None) -> str:
    return "" if v is None else "yes" if v else "no"


def _ranking_table(
    ranking: Sequence[Ranked], proposal: str | None, chosen: str | None, cfg: SelectConfig
) -> Table:
    columns = [
        Column(name="rank", label="Rank", kind="integer"),
        Column(name="candidate", label="Candidate", kind="text"),
        Column(name="role", label="Role", kind="text"),
        Column(name="size", label="Drivers", kind="integer"),
        Column(name="scaled_error", label="Scaled error", kind="number"),
        Column(name="in_set", label="In the confidence set", kind="text"),
        Column(name="pvalue", label="MCS p-value", kind="number"),
        Column(name="build", label="Build checks", kind="text"),
        Column(name="beats", label="Beats the benchmark", kind="text"),
        Column(name="order", label="Scenarios out of order", kind="integer"),
        Column(name="bounds", label="Projections out of bounds", kind="integer"),
        Column(name="choice", label="Choice", kind="text"),
        Column(name="eligible", label="Passes every check", kind="status"),
    ]
    rows = []
    for r in ranking:
        mark = "chosen" if r.candidate == chosen else "proposed" if r.candidate == proposal else ""
        rows.append(
            [
                r.rank,
                r.candidate,
                r.role,
                r.size,
                r.scaled_error,
                _yes(r.in_set),
                _finite(r.pvalue),
                "pass" if r.build_passed else "fail",
                _yes(r.beats_benchmark),
                r.out_of_order,
                r.out_of_bounds,
                mark,
                "pass" if r.eligible else "fail",
            ]
        )
    loss = "RMSE" if cfg.loss == "squared" else "MAE"
    return Table(
        id=IDS.table("ranking"),
        caption="Ranking of the candidates",
        columns=columns,
        rows=rows,
        notes=(
            f"Scaled error: the candidate's rolling-origin {loss} over the {cfg.benchmark} "
            f"benchmark's; it beats the benchmark below {cfg.benchmark_ratio:g}.",
            f"Confidence set: arch's MCS at size {cfg.mcs.size:g} on the per-origin losses.",
            "Default rule: among the candidates that pass every check and are in the set, the "
            "fewest drivers, then the lowest scaled error. Benchmarks are compared, not chosen.",
        ),
    )


def _evaluation_table(
    evaluations: Sequence[CandidateEvaluation], scaled: Mapping[str, float]
) -> Table:
    columns = [
        Column(name="candidate", label="Candidate", kind="text"),
        Column(name="folds", label="Origins", kind="integer"),
        Column(name="mae", label="MAE", kind="number"),
        Column(name="rmse", label="RMSE", kind="number"),
        Column(name="mase", label="MASE", kind="number"),
        Column(name="coverage", label="Interval coverage", kind="percent"),
        Column(name="scaled", label="Scaled error", kind="number"),
        Column(name="note", label="Note", kind="text"),
    ]
    rows = []
    for e in evaluations:
        ev = e.evaluation
        if ev is None:
            rows.append([e.candidate, None, None, None, None, None, None, e.note or ""])
            continue
        rows.append(
            [
                e.candidate,
                len(ev.folds),
                _finite(ev.score("mae")),
                _finite(ev.score("rmse")),
                _finite(ev.score("mase")),
                _finite(ev.score("coverage")),
                _finite(scaled.get(e.candidate)),
                "",
            ]
        )
    return Table(
        id=IDS.table("evaluation"),
        caption="Rolling-origin evaluation",
        columns=columns,
        rows=rows,
        notes=(
            "Each model is refitted at every origin and forecast with the drivers' actual "
            "values; a model that cannot be fitted at an origin is not scored (its note says "
            "why) and is not chosen.",
        ),
    )


def _scenario_table(ranking: Sequence[Ranked], severity: Sequence[Any], cfg: SelectConfig) -> Table:
    columns = [Column(name="candidate", label="Candidate", kind="text")]
    columns += [Column(name=f"s_{s}", label=str(s), kind="number") for s in severity]
    columns.append(Column(name="order", label="Out of order", kind="integer"))
    rows = [
        [r.candidate, *[_finite(r.statistics.get(str(s))) for s in severity], r.out_of_order]
        for r in ranking
        if r.statistics
    ]
    return Table(
        id=IDS.table("scenarios"),
        caption="Projections by scenario",
        columns=columns,
        rows=rows,
        notes=(
            f"The {cfg.ordering.statistic} of each projection in the target's levels, scenarios "
            "mildest first; a more severe scenario should project a "
            f"{'higher' if cfg.ordering.direction == 1 else 'lower'} value.",
        )
        if severity
        else ("No scenarios were given.",),
    )


def _selection_table(
    cfg: SelectConfig,
    conf: ConfidenceSet,
    proposal: str | None,
    chosen: str | None,
    decision: str,
    reason: str | None,
    origins: int,
) -> Table:
    items = [
        ("Benchmark", cfg.benchmark),
        ("Loss", cfg.loss),
        ("Origins", str(origins)),
        ("Confidence set size", f"{conf.size:g}"),
        ("Bootstrap", f"{conf.bootstrap}, {conf.reps} replications, method {conf.method}"),
        ("Seed", str(conf.seed)),
        ("In the set", ", ".join(conf.included) or "none"),
        ("Proposed by the default rule", proposal or "none"),
        ("Decision", decision),
        ("Chosen", chosen or "none"),
        ("Reason", reason or ""),
    ]
    return Table(
        id=IDS.table("selection"),
        caption="Selection record",
        columns=[
            Column(name="item", label="Item", kind="text"),
            Column(name="value", label="Value", kind="text"),
        ],
        rows=[list(i) for i in items],
    )


def _backtest_figure(scored: Mapping[str, Evaluation], target: str | None) -> Figure:
    first = next(iter(scored.values()))
    periods = [r.period for r in first.rows if r.horizon == 1]
    layers = [
        Layer(kind="line", y=[r.actual for r in first.rows if r.horizon == 1], label="actual")
    ]
    for cid, ev in scored.items():
        layers.append(
            Layer(kind="line", y=[r.forecast for r in ev.rows if r.horizon == 1], label=cid)
        )
    return Figure(
        id=IDS.figure("backtest"),
        caption=f"Back-test: one-step forecasts of {target} at every origin, by candidate",
        x=periods,
        layers=layers,
        x_label="Period",
        y_label=str(target),
    )


def _scenario_figure(
    forecasts: Sequence[ScenarioForecast],
    chosen: str,
    history: pd.Series,
    severity: Sequence[Any],
    target: str | None,
) -> Figure | None:
    mine = [f for f in forecasts if f.candidate == chosen]
    if not mine:
        return None
    periods = list(mine[0].forecast.periods)
    past = history.dropna().iloc[-12:]
    x = [str(p) for p in past.index] + periods
    pad = [None] * len(periods)
    layers = [Layer(kind="line", y=[float(v) for v in past] + pad, label="history")]
    for f in sorted(mine, key=lambda f: [str(s) for s in severity].index(str(f.scenario))):
        layers.append(
            Layer(kind="line", y=[None] * len(past) + list(f.levels), label=str(f.scenario))
        )
    return Figure(
        id=IDS.figure("scenarios"),
        caption=f"Scenario projections of {target} by the chosen model, {chosen}",
        x=x,
        layers=layers,
        x_label="Period",
        y_label=str(target),
    )
