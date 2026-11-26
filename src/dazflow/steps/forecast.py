"""forecast: the chosen model projected under every scenario (design §7.7).

Select's chosen model is refitted from its record on the series in levels
and forecast from the origin, the last period of its sample, under every
scenario, with intervals. On the model's scale the gap between the
observed and the fitted value at the origin is added by the anchoring
rule (`carry`, `decay` or `none`). The path is then taken back to the
target's scale through the target's recorded transform chain
(`dazflow.transforms.inverse`, with the observed history), with a bias
correction suited to the chain: smearing (Duan, 1983) for a log or Box-Cox,
simulation for a logit or a nonlinear chain with a lag, none for a linear
chain. Last, an optional linear reversion moves the path towards a
long-run level over the horizon.

Gate: ``reconcile`` (the forecast taken back to the origin is within
`tolerance` of the observed value) and ``forecast-bounds`` (no forecast
mean outside the bounds, select's unless configured).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from dazflow import models, readers, transforms
from dazflow.core._model import DazflowModel
from dazflow.core.artefact import ArtefactIds, Column, Figure, Layer, Table
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import ForecastError, ModelError, ReaderError, ResultError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import StepResult
from dazflow.models import Design, Forecast, ModelRecord

__all__ = [
    "IDS",
    "AnchorConfig",
    "BackTransform",
    "Config",
    "ForecastConfig",
    "ForecastResult",
    "Result",
    "ReversionConfig",
    "ScenarioPath",
    "SimulationConfig",
    "adjustment",
    "back_transform",
    "revert",
    "run",
]

# Never renumber: documents cite these (design §9, D21).
IDS = ArtefactIds(
    "forecast",
    tables={"forecasts": 1, "projection": 2, "gate": 3},
    figures={"fan": 1},
    findings={"summary": 1},
)

SUMMARY = (
    "{{ chosen }} forecasts {{ target }} from {{ origin }} for {{ horizon }} periods under "
    "{{ scenarios }} scenarios. The jump-off gap of {{ gap }} is anchored by the {{ rule }} "
    "rule, and the forecasts are back on the target's scale (bias correction: "
    "{{ correction }}). At the last period the mean runs from {{ low }} under {{ low_scenario }} to "
    "{{ high }} under {{ high_scenario }}."
)

# Transforms whose inverse is affine in the transformed values.
LINEAR = frozenset({"diff", "seasonal_diff"})
UNCONDITIONAL = "unconditional"


# --- configuration ---------------------------------------------------------------


class AnchorConfig(Config):
    """How the gap between observed and fitted at the origin enters the forecast."""

    rule: Literal["carry", "decay", "none"] = Field(
        "decay",
        description="carry: the gap is added at every horizon; decay: it shrinks by `decay` "
        "each period; none: it is not added.",
    )
    decay: float = Field(
        0.5,
        gt=0,
        le=1,
        description="decay: the share of the gap removed each period; the gap at horizon h is "
        "gap x (1 - decay)^h.",
    )


class ReversionConfig(Config):
    """A linear move towards a long-run level, on the target's scale."""

    level: float = Field(description="The long-run level, in the target's units.")
    periods: int | None = Field(
        None, ge=1, description="Periods to reach it, linearly; unset: the horizon."
    )


class SimulationConfig(Config):
    """The draws of a simulated correction or interval."""

    draws: int = Field(2000, ge=100, le=100_000, description="Simulated paths per scenario.")
    seed: int | None = Field(
        None, description="Seed of the draws; unset: project.yaml's seed (0 outside a project)."
    )


Level = Annotated[float, Field(gt=0, lt=1)]


class ForecastConfig(Config):
    """Configuration of the forecast step."""

    horizon: int | None = Field(
        None, ge=1, description="Periods forecast; unset: as many as every scenario covers."
    )
    levels: list[Level] = Field(
        default_factory=lambda: [0.8, 0.95], min_length=1, description="Interval levels."
    )
    anchor: AnchorConfig = Field(default_factory=AnchorConfig)
    reversion: ReversionConfig | None = Field(
        None, description="Revert towards a long-run level: {level: ..., periods: ...}; unset: no."
    )
    correction: Literal["auto", "none", "smearing", "simulation"] = Field(
        "auto",
        description="Bias correction back on the target's scale; auto: smearing for a log or "
        "Box-Cox, simulation for a logit or a nonlinear chain with a lag, none for a linear chain.",
    )
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    tolerance: float = Field(
        1e-6, ge=0, description="Largest jump-off gap at the origin, in the target's units."
    )
    bounds: tuple[float | None, float | None] | None = Field(
        None,
        description="[lower, upper] for every forecast mean in the target's units; null leaves "
        "a side open; unset: select's bounds.",
    )
    overrides: list[Override] = Field(default_factory=list)

    @field_validator("levels")
    @classmethod
    def _distinct(cls, value: list[float]) -> list[float]:
        if len(set(value)) != len(value):
            raise ValueError("interval levels must be distinct")
        return value


# --- result objects --------------------------------------------------------------


class _Record(DazflowModel):
    _error: ClassVar = ResultError
    _hint: ClassVar[str] = "These records are made by the forecast step."


class BackTransform(_Record):
    """A forecast on the target's scale and how it got there.

    `correction` is the bias correction of the mean; `intervals` how the
    bounds were mapped: ``model`` (no chain), ``inverse`` (the bounds
    inverted, exact for a monotone chain without lags), ``linear`` (a
    linear chain: errors independent across horizons, variances added) or
    ``simulated`` (quantiles of simulated paths). `draws` and `seed` are set
    when anything was simulated.
    """

    forecast: Forecast
    correction: Literal["none", "smearing", "simulation"]
    intervals: Literal["model", "inverse", "linear", "simulated"]
    draws: int | None = None
    seed: int | None = None


class ScenarioPath(_Record):
    """One scenario: the model's forecast, anchored, and the forecast on the target's scale.

    `raw` is the chosen model's own forecast (model scale), `anchored` it
    with the anchoring adjustments, `forecast` the result: back-transformed,
    corrected and reverted. `reversion` is the shift reversion added.
    """

    scenario: str | int
    raw: Forecast
    anchored: Forecast
    forecast: Forecast
    reversion: tuple[float, ...] = ()


class ForecastResult(StepResult):
    """What forecast returns: every scenario's path on the target's scale, and how it was made.

    `paths[i].forecast` is the forecast of scenario `severity[i]`; `record`
    and `chosen` the model used. `gap` is observed minus fitted at the
    origin on the model's scale, `origin_gap` the forecast taken back to the
    origin minus the observed value, in the target's units (the gate's).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ForecastConfig
    fingerprint: str | None = None
    target: str | None = None
    chosen: str | None = None
    form: str | None = None
    record: ModelRecord | None = None
    design: Design | None = None
    origin: str | None = None
    observed: float | None = None
    fitted: float | None = None
    gap: float | None = None
    origin_gap: float | None = None
    horizon: int = 0
    severity: tuple[str | int, ...] = ()
    chain: tuple[str, ...] = ()
    correction: Literal["none", "smearing", "simulation"] = "none"
    intervals: Literal["model", "inverse", "linear", "simulated"] = "model"
    draws: int | None = None
    seed: int = 0
    adjustments: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    bounds: tuple[float | None, float | None] | None = None
    paths: tuple[ScenarioPath, ...] = ()

    def path(self, scenario: str | int) -> Forecast:
        """The forecast of `scenario` on the target's scale."""
        for p in self.paths:
            if p.scenario == scenario:
                return p.forecast
        raise ForecastError(
            f"no scenario {scenario!r}",
            observed=f"scenarios: {', '.join(str(p.scenario) for p in self.paths)}",
            change="Name a scenario the forecast was made for.",
        )

    def frame(self) -> pd.DataFrame:
        """One row per scenario and horizon: ``scenario``, ``horizon``, ``period``, the forecast."""
        parts = []
        for p in self.paths:
            f = p.forecast.frame().reset_index(drop=True)
            f.insert(0, "period", list(p.forecast.periods))
            f.insert(0, "horizon", range(1, p.forecast.h + 1))
            f.insert(0, "scenario", p.scenario)
            parts.append(f)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


Config = ForecastConfig
Result = ForecastResult


# --- the pieces ------------------------------------------------------------------


def _fail(problem: str, observed: str, change: str) -> ForecastError:
    return ForecastError(problem, observed=observed, change=change)


def adjustment(gap: float, h: int, rule: str = "decay", decay: float = 0.5) -> np.ndarray:
    """What anchoring adds at horizons 1..h: the gap (carry), ``gap (1 - decay)^h`` or nothing."""
    if rule == "carry":
        return np.full(h, float(gap))
    if rule == "decay":
        return float(gap) * (1.0 - decay) ** np.arange(1, h + 1)
    if rule == "none":
        return np.zeros(h)
    raise _fail(f"no anchoring rule {rule!r}", f"rule {rule!r}", "Use carry, decay or none.")


def _weights(h: int, periods: int | None) -> np.ndarray:
    return np.minimum(np.arange(1, h + 1) / (h if periods is None else periods), 1.0)


def revert(path: Sequence[float], level: float, periods: int | None = None) -> np.ndarray:
    """The shift moving `path` linearly towards `level`: weight h/periods, full from `periods` on."""
    m = np.asarray(path, dtype=float)
    return _weights(len(m), periods) * (float(level) - m)


def _shifted(fc: Forecast, shift: np.ndarray) -> Forecast:
    return fc.model_copy(
        update={
            "mean": tuple(float(v) for v in np.asarray(fc.mean) + shift),
            "lower": tuple(tuple(float(v) for v in np.asarray(lo) + shift) for lo in fc.lower),
            "upper": tuple(tuple(float(v) for v in np.asarray(up) + shift) for up in fc.upper),
        }
    )


def _rebuilt(fc: Forecast, mean: Any, lower: Sequence[Any], upper: Sequence[Any]) -> Forecast:
    return fc.model_copy(
        update={
            "mean": tuple(float(v) for v in mean),
            "lower": tuple(tuple(float(v) for v in lo) for lo in lower),
            "upper": tuple(tuple(float(v) for v in up) for up in upper),
        }
    )


def _stateless_inverse(values: np.ndarray, specs: Sequence[Any]) -> np.ndarray:
    # A chain without lags maps value by value; the periods are placeholders.
    flat = np.asarray(values, dtype=float).ravel()
    index = pd.period_range("1970-01-01", periods=len(flat), freq="D")
    out = transforms.inverse(pd.Series(flat, index=index), list(specs))
    return out.to_numpy(dtype=float).reshape(np.shape(values))


def _linear_matrix(specs: Sequence[Any], index: pd.PeriodIndex, history: pd.Series) -> np.ndarray:
    # A linear chain's inverse is affine in the values: its linear part, column by column.
    zero = history * 0.0
    cols = []
    for j in range(len(index)):
        e = np.zeros(len(index))
        e[j] = 1.0
        cols.append(transforms.inverse(pd.Series(e, index=index), list(specs), zero).to_numpy())
    return np.column_stack(cols)


def _sigma(fc: Forecast) -> np.ndarray:
    i = int(np.argmax(fc.levels))
    q = NormalDist().inv_cdf((1 + fc.levels[i]) / 2)
    return (np.asarray(fc.upper[i]) - np.asarray(fc.lower[i])) / (2 * q)


def back_transform(
    forecast: Forecast,
    chain: Sequence[str],
    history: pd.Series | None = None,
    *,
    residuals: Sequence[float] | pd.Series | None = None,
    correction: str = "auto",
    draws: int = 2000,
    seed: int = 0,
) -> BackTransform:
    """`forecast` (on the model's scale) on the target's scale, through the resolved `chain`.

    `history` is the target in levels up to the origin (needed by a chain
    with a lag). A linear chain (differences) needs no correction: the mean
    is the inverse of the mean, and the bounds add the horizons' variances,
    errors taken independent. A chain without lags inverts the bounds;
    its mean is corrected by ``smearing`` (the inverse averaged over the
    model's `residuals`; Duan's factor for a log) or ``simulation`` (averaged
    over `draws` normal errors with the forecast's standard error, from the
    widest interval, seeded by `seed`). A nonlinear chain with a lag is
    simulated, its bounds the simulated quantiles. ``auto`` picks none,
    smearing for a log or Box-Cox, simulation for a logit or a lag.
    """
    specs = [transforms.parse(c) for c in chain]
    lagged = any(transforms.consumed([s]) > 0 for s in specs)
    nonlinear = [s.key for s in specs if s.key not in LINEAR]
    text = ", ".join(str(s) for s in specs) or "none"
    if not nonlinear:
        if correction not in ("auto", "none"):
            raise _fail(
                f"the chain is linear, so no {correction} correction applies",
                f"chain {text}",
                "Set correction to auto or none.",
            )
        if not specs:
            return BackTransform(forecast=forecast, correction="none", intervals="model")
        return _linear(forecast, specs, history)
    if correction == "auto":
        correction = "simulation" if ("logit" in nonlinear or lagged) else "smearing"
    if correction == "smearing":
        if lagged:
            raise _fail(
                "smearing applies to a chain without lags",
                f"chain {text}",
                "Set correction to simulation (or auto) for this chain.",
            )
        if residuals is None or len(residuals) == 0:
            raise _fail(
                "smearing needs the model's residuals",
                "no residuals given",
                "Pass the fitted model's residuals.",
            )
    if lagged and (history is None or len(history) == 0):
        raise _fail(
            "a chain with a lag needs the target's history",
            f"chain {text}",
            "Pass the target in levels up to the origin.",
        )
    z = np.asarray(forecast.mean, dtype=float)
    if not lagged:
        lower = [_stateless_inverse(np.asarray(lo), specs) for lo in forecast.lower]
        upper = [_stateless_inverse(np.asarray(up), specs) for up in forecast.upper]
        if correction == "smearing":
            e = np.asarray(residuals, dtype=float)
            mean = _stateless_inverse(z[None, :] + e[:, None], specs).mean(axis=0)
            return BackTransform(
                forecast=_rebuilt(forecast, mean, lower, upper),
                correction="smearing",
                intervals="inverse",
            )
        if correction == "none":
            mean = _stateless_inverse(z, specs)
            return BackTransform(
                forecast=_rebuilt(forecast, mean, lower, upper),
                correction="none",
                intervals="inverse",
            )
        eps = np.random.default_rng(seed).standard_normal((draws, len(z)))
        mean = _stateless_inverse(z + _sigma(forecast) * eps, specs).mean(axis=0)
        return BackTransform(
            forecast=_rebuilt(forecast, mean, lower, upper),
            correction="simulation",
            intervals="inverse",
            draws=draws,
            seed=seed,
        )
    sims = _simulate(forecast, specs, history, draws, seed)
    if correction == "simulation":
        mean = sims.mean(axis=0)
    else:
        mean = _full_inverse(z, forecast, specs, history)
    lower, upper = [], []
    for lv in forecast.levels:
        lo, up = np.quantile(sims, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        lower.append(lo)
        upper.append(up)
    return BackTransform(
        forecast=_rebuilt(forecast, mean, lower, upper),
        correction="simulation" if correction == "simulation" else "none",
        intervals="simulated",
        draws=draws,
        seed=seed,
    )


def _full_inverse(
    values: np.ndarray, fc: Forecast, specs: Sequence[Any], history: pd.Series | None
) -> np.ndarray:
    series = pd.Series(values, index=fc.index, name=None if history is None else history.name)
    return transforms.inverse(series, list(specs), history).to_numpy(dtype=float)


def _linear(fc: Forecast, specs: Sequence[Any], history: pd.Series | None) -> BackTransform:
    if history is None or len(history) == 0:
        raise _fail(
            "a chain with a lag needs the target's history",
            f"chain {', '.join(str(s) for s in specs)}",
            "Pass the target in levels up to the origin.",
        )
    z = np.asarray(fc.mean, dtype=float)
    mean = _full_inverse(z, fc, specs, history)
    L2 = _linear_matrix(specs, fc.index, history) ** 2
    lower = [mean - np.sqrt(L2 @ (z - np.asarray(lo)) ** 2) for lo in fc.lower]
    upper = [mean + np.sqrt(L2 @ (np.asarray(up) - z) ** 2) for up in fc.upper]
    return BackTransform(
        forecast=_rebuilt(fc, mean, lower, upper), correction="none", intervals="linear"
    )


def _simulate(
    fc: Forecast, specs: Sequence[Any], history: pd.Series, draws: int, seed: int
) -> np.ndarray:
    """Level paths from `draws` normal model-scale paths, errors independent across horizons."""
    z = np.asarray(fc.mean, dtype=float)
    eps = np.random.default_rng(seed).standard_normal((draws, len(z)))
    paths = z + _sigma(fc) * eps
    # Stateless transforms first, then linear lags: invert the lags as an affine map.
    split = next((i for i, s in enumerate(specs) if transforms.consumed([s]) > 0), len(specs))
    head, tail = specs[:split], specs[split:]
    if all(s.key in LINEAR for s in tail):
        base = transforms.forward(history, list(head))[0] if head else history
        centre = _full_inverse(z, fc, tail, base)
        L = _linear_matrix(tail, fc.index, base)
        inner = centre + (paths - z) @ L.T
        return _stateless_inverse(inner, head) if head else inner
    return np.vstack([_full_inverse(p, fc, specs, history) for p in paths])


# --- the step --------------------------------------------------------------------


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


def run(
    dataset: Dataset,
    config: ForecastConfig | Mapping[str, Any] | None = None,
    *,
    select: Any,
    seed: int | None = None,
) -> ForecastResult:
    """Forecast select's chosen model under every scenario, anchored, on the target's scale.

    `dataset` holds the series in levels select ran on, with any scenarios
    in `dataset.scenarios`; without them, select's `config.scenarios` file
    is read (relative to the working directory), so scenarios are declared
    once. The simulation seed is `config.simulation.seed`, else `seed`
    (project.yaml's), else 0. The caller's `dataset` is not changed. Prints
    nothing and writes no files.
    """
    cfg = (
        config
        if isinstance(config, ForecastConfig)
        else ForecastConfig.model_validate(config or {})
    )
    plain = dataset.with_scenarios(None) if dataset.scenarios is not None else dataset
    if plain.fingerprint() != getattr(select, "fingerprint", None):
        raise _fail(
            "the dataset is not the one select ran on",
            f"fingerprint {plain.fingerprint()[:12]}...; select's "
            f"{str(getattr(select, 'fingerprint', None))[:12]}...",
            "Pass forecast the dataset select received (in levels), with select's result.",
        )
    if select.chosen is None or select.record is None or select.design is None:
        raise _fail(
            "select chose no model, so there is no model to forecast",
            f"select's decision: {getattr(select, 'decision', 'none')}",
            "Choose a candidate in config/select.yaml: choice, with a reason.",
        )
    design: Design = select.design
    record: ModelRecord = select.record
    target = design.target
    levels = plain.frame
    freq = levels.index.freq
    end = pd.Period(design.end, freq=freq)
    scen, severity = _scenarios(dataset, plain, select)
    use_seed = (
        cfg.simulation.seed
        if cfg.simulation.seed is not None
        else (seed if seed is not None else 0)
    )

    if scen is None:
        if design.columns:
            raise _fail(
                "the chosen model has drivers and there are no scenarios",
                f"drivers: {', '.join(design.columns)}",
                "Give scenarios in config/select.yaml: scenarios, or in dataset.scenarios.",
            )
        if cfg.horizon is None:
            raise _fail(
                "without scenarios the horizon must be given",
                "horizon unset",
                "Set horizon in config/forecast.yaml.",
            )
        labelled: list[tuple[Any, pd.DataFrame | None]] = [(UNCONDITIONAL, None)]
        severity, h = [UNCONDITIONAL], cfg.horizon
    else:
        labelled = [(label, scen.xs(label, level="scenario")) for label in severity]
        cover = min(readers.scenario_cover(paths, end) for _, paths in labelled)
        if cover == 0:
            raise _fail(
                "a scenario has no period after the origin",
                f"origin {end}",
                "Give scenario paths for the periods after the last observation.",
            )
        h = cfg.horizon if cfg.horizon is not None else cover
        if h > cover:
            raise _fail(
                f"the scenarios do not cover the horizon of {h} periods",
                f"every scenario covers {cover} periods after {end}",
                "Shorten the horizon, or extend the scenario paths.",
            )

    try:
        y, X = design.frames(levels)
        fitted = models.refit(record, y, X)
    except ModelError as err:
        raise _fail(
            "the chosen model cannot be refitted", err.problem, "Pass the data select ran on."
        ) from err
    fitted_values = fitted.fitted_values
    fitted_origin = float(fitted_values.loc[end])
    gap = float(y.loc[end]) - fitted_origin
    adj = adjustment(gap, h, cfg.anchor.rule, cfg.anchor.decay)
    at_origin = 0.0 if cfg.anchor.rule == "none" else gap
    chain = tuple(design.chains.get(target, ()))
    observed = float(levels[target].astype("float64").loc[end])
    history = levels[target].astype("float64").loc[:end].rename(target)
    z0 = pd.Series([fitted_origin + at_origin], index=pd.PeriodIndex([end]), name=target)
    origin_value = (
        float(transforms.inverse(z0, list(chain), history.loc[: end - 1]).iloc[0])
        if chain
        else float(z0.iloc[0])
    )
    origin_gap = origin_value - observed

    paths_out: list[ScenarioPath] = []
    first: BackTransform | None = None
    weights = _weights(h, cfg.reversion.periods) if cfg.reversion is not None else np.zeros(h)
    for label, paths in labelled:
        future = design.future(levels, paths, h) if paths is not None else None
        try:
            raw = fitted.predict(h, future, levels=cfg.levels)
        except ModelError as err:
            raise _fail(
                f"the chosen model cannot forecast scenario {label!r}",
                err.problem,
                err.change,
            ) from err
        anchored = _shifted(raw, adj)
        bt = back_transform(
            anchored,
            chain,
            history,
            residuals=fitted.residuals,
            correction=cfg.correction,
            draws=cfg.simulation.draws,
            seed=use_seed,
        )
        first = first or bt
        shift = (
            revert(bt.forecast.mean, cfg.reversion.level, cfg.reversion.periods)
            if cfg.reversion is not None
            else np.zeros(h)
        )
        final = _shifted(bt.forecast, shift)
        paths_out.append(
            ScenarioPath(
                scenario=label,
                raw=raw,
                anchored=anchored,
                forecast=final,
                reversion=tuple(float(v) for v in shift),
            )
        )
    assert first is not None

    bounds = cfg.bounds if cfg.bounds is not None else getattr(select.config, "bounds", None)
    checks = [
        Check(
            name="reconcile",
            observed=abs(origin_gap),
            threshold=cfg.tolerance,
            op="<=",
        )
    ]
    if bounds is not None and any(b is not None for b in bounds):
        lo, hi = bounds
        out = sum(
            1
            for p in paths_out
            for v in p.forecast.mean
            if (lo is not None and v < lo) or (hi is not None and v > hi)
        )
        checks.append(Check(name="forecast-bounds", observed=float(out), threshold=0.0, op="<="))
    gate = Gate.evaluate(checks, cfg.overrides)

    last = [(p.forecast.mean[-1], str(p.scenario)) for p in paths_out]
    low, high = min(last), max(last)
    finding = Finding(
        template=SUMMARY,
        key="forecast.summary",
        values={
            "chosen": select.chosen,
            "target": target,
            "origin": str(end),
            "horizon": h,
            "scenarios": len(paths_out),
            "gap": gap,
            "rule": cfg.anchor.rule,
            "correction": first.correction,
            "low": low[0],
            "low_scenario": low[1],
            "high": high[0],
            "high_scenario": high[1],
        },
    )
    simulated = first.correction == "simulation" or first.intervals == "simulated"
    artefacts: dict[str, Any] = {
        IDS.table("forecasts"): _forecast_table(paths_out, target),
        IDS.table("projection"): _projection_table(
            cfg,
            select.chosen,
            record,
            end,
            observed,
            fitted_origin,
            gap,
            origin_gap,
            chain,
            first,
            use_seed if simulated else None,
            bounds,
        ),
        IDS.table("gate"): Table.from_gate(gate, id=IDS.table("gate"), caption="Forecast checks"),
        IDS.figure("fan"): _fan_figure(paths_out, history, target, select.chosen),
        IDS.finding("summary"): finding,
    }
    return ForecastResult(
        gate=gate,
        finding=finding,
        artefacts=artefacts,
        config=cfg,
        fingerprint=plain.fingerprint(),
        target=target,
        chosen=select.chosen,
        form=record.spec.form,
        record=record,
        design=design,
        origin=str(end),
        observed=observed,
        fitted=fitted_origin,
        gap=gap,
        origin_gap=origin_gap,
        horizon=h,
        severity=tuple(severity),
        chain=chain,
        correction=first.correction,
        intervals=first.intervals,
        draws=cfg.simulation.draws if simulated else None,
        seed=use_seed,
        adjustments=tuple(float(v) for v in adj),
        weights=tuple(float(v) for v in weights),
        bounds=bounds,
        paths=tuple(paths_out),
    )


# --- artefacts ---------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _forecast_table(paths: Sequence[ScenarioPath], target: str) -> Table:
    lv = paths[0].forecast.levels
    columns = [
        Column(name="scenario", label="Scenario", kind="text"),
        Column(name="horizon", label="Horizon", kind="integer"),
        Column(name="period", label="Period", kind="text"),
        Column(name="mean", label="Mean", kind="number"),
    ]
    for level in lv:
        columns.append(Column(name=f"lower_{level:g}", label=f"Lower {level:.0%}", kind="number"))
        columns.append(Column(name=f"upper_{level:g}", label=f"Upper {level:.0%}", kind="number"))
    rows = []
    for p in paths:
        f = p.forecast
        for i in range(f.h):
            row: list[Any] = [str(p.scenario), i + 1, f.periods[i], _finite(f.mean[i])]
            for j in range(len(lv)):
                row += [_finite(f.lower[j][i]), _finite(f.upper[j][i])]
            rows.append(row)
    return Table(
        id=IDS.table("forecasts"),
        caption=f"Forecasts of {target} by scenario and horizon",
        columns=columns,
        rows=rows,
        notes=(
            "Means and prediction intervals on the target's scale, after anchoring, the "
            "bias correction and any reversion.",
        ),
    )


def _projection_table(
    cfg: ForecastConfig,
    chosen: str,
    record: ModelRecord,
    origin: pd.Period,
    observed: float,
    fitted: float,
    gap: float,
    origin_gap: float,
    chain: tuple[str, ...],
    bt: BackTransform,
    seed: int | None,
    bounds: tuple[float | None, float | None] | None,
) -> Table:
    a = cfg.anchor
    rule = f"decay ({a.decay:g} of the gap removed each period)" if a.rule == "decay" else a.rule
    rev = cfg.reversion
    items = [
        ("Model", chosen),
        ("Form", record.spec.form),
        ("Origin", str(origin)),
        ("Observed at the origin", f"{observed:.6g}"),
        ("Fitted at the origin (model scale)", f"{fitted:.6g}"),
        ("Jump-off gap (model scale)", f"{gap:.6g}"),
        ("Anchoring", rule),
        ("Gap at the origin after anchoring (target scale)", f"{origin_gap:.3g}"),
        (
            "Mean reversion",
            "none"
            if rev is None
            else f"to {rev.level:g} over {rev.periods or 'the horizon'} periods, linearly",
        ),
        ("Transform chain", ", ".join(chain) or "none"),
        ("Bias correction", bt.correction),
        ("Intervals", bt.intervals),
        ("Simulated draws", "" if seed is None else str(cfg.simulation.draws)),
        ("Seed", "" if seed is None else str(seed)),
        ("Bounds", "none" if bounds is None else f"[{bounds[0]}, {bounds[1]}]"),
    ]
    return Table(
        id=IDS.table("projection"),
        caption="Projection record",
        columns=[
            Column(name="item", label="Item", kind="text"),
            Column(name="value", label="Value", kind="text"),
        ],
        rows=[list(i) for i in items],
    )


def _fan_figure(
    paths: Sequence[ScenarioPath], history: pd.Series, target: str, chosen: str
) -> Figure:
    periods = list(paths[0].forecast.periods)
    past = history.dropna().iloc[-12:]
    x = [str(p) for p in past.index] + periods
    pad_past = [None] * len(past)
    layers = []
    for p in paths:
        f = p.forecast
        i = int(np.argmax(f.levels))
        layers.append(
            Layer(
                kind="band",
                y_low=pad_past + [_finite(v) for v in f.lower[i]],
                y_high=pad_past + [_finite(v) for v in f.upper[i]],
                label=f"{p.scenario}, {f.levels[i]:.0%}",
            )
        )
    layers.append(
        Layer(kind="line", y=[float(v) for v in past] + [None] * len(periods), label="history")
    )
    for p in paths:
        layers.append(
            Layer(
                kind="line",
                y=pad_past + [_finite(v) for v in p.forecast.mean],
                label=str(p.scenario),
            )
        )
    return Figure(
        id=IDS.figure("fan"),
        caption=f"Fan chart of {target} by scenario, from {chosen}",
        x=x,
        layers=layers,
        x_label="Period",
        y_label=str(target),
    )
