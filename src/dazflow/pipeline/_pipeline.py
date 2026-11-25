"""Pipeline: steps by name in configured order; Run: what one pass through them returned."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dazflow import readers
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import PipelineError
from dazflow.core.registry import Registry
from dazflow.core.result import StepResult
from dazflow.pipeline._adapters import (
    InputRecord,
    StepInputs,
    adapters,
    default_adapter,
    file_sha256,
)
from dazflow.pipeline._record import DatasetRecord, RunInfo, RunRecord, environment
from dazflow.readers._base import ReadSpec

__all__ = ["DEFAULT_STEPS", "Pipeline", "Run"]

# The 0.1.0 steps built so far, in design §7 order.
DEFAULT_STEPS: tuple[str, ...] = ("ingest", "clean", "explore", "build", "select", "forecast")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Run:
    """One pass through a pipeline. Nothing in it has been written anywhere.

    `results` and `configs` hold every step up to the last run, in pipeline
    order: those run now (`steps`) and those carried from an earlier record
    (`carried`). The run stops after a step whose gate fails; that step is
    `stopped_at`, and the steps after it are not run.
    """

    pipeline: tuple[str, ...]
    steps: tuple[str, ...]
    results: Mapping[str, StepResult]
    configs: Mapping[str, Config]
    started: str
    finished: str
    carried: tuple[str, ...] = ()
    start: str | None = None
    stop: str | None = None
    stopped_at: str | None = None
    inputs: tuple[InputRecord, ...] = ()
    carried_record: RunRecord | None = field(default=None, repr=False)

    @property
    def passed(self) -> bool:
        """Every step run has a gate that passed (failed checks overridden count as passed)."""
        return self.stopped_at is None

    @property
    def id(self) -> str:
        """``<finished time>-<8 hex digits>`` from the configuration and the datasets."""
        h = hashlib.sha256()
        for name in self.results:
            h.update(name.encode())
            h.update(
                json.dumps(self.configs[name].model_dump(mode="json"), sort_keys=True).encode()
            )
            h.update(str(getattr(self.results[name], "fingerprint", None)).encode())
        stamp = self.finished.replace("-", "").replace(":", "")
        return f"{stamp}-{h.hexdigest()[:8]}"

    def record(
        self,
        project: Config | Mapping[str, Any] | None = None,
        datasets: Mapping[str, str] | None = None,
    ) -> RunRecord:
        """The run record. `project` is the project configuration; `datasets` the paths written."""
        old = self.carried_record
        paths = dict(datasets or {})
        config: dict[str, dict[str, Any]] = {}
        results: dict[str, dict[str, Any]] = {}
        checks: list[dict[str, Any]] = []
        overrides: list[dict[str, Any]] = []
        records: dict[str, DatasetRecord] = {}
        for name, result in self.results.items():
            data = result.to_dict()
            config[name] = self.configs[name].model_dump(mode="json")
            results[name] = data
            checks.extend({"step": name, **c} for c in data["gate"]["checks"])
            overrides.extend({"step": name, **o} for o in config[name].get("overrides", []) or [])
            fp = getattr(result, "fingerprint", None)
            if fp is not None:
                path = paths.get(name)
                if path is None and old is not None and name in self.carried:
                    path = old.datasets[name].path if name in old.datasets else None
                records[name] = DatasetRecord(fingerprint=fp, path=path)
        inputs = list(self.inputs)
        if old is not None:
            inputs = [i for i in old.inputs if i.step in self.carried] + inputs
        if isinstance(project, Config):
            project = project.model_dump(mode="json")
        return RunRecord(
            run=RunInfo(
                id=self.id,
                started=self.started,
                finished=self.finished,
                start=self.start,
                stop=self.stop,
                steps=self.steps,
                carried=self.carried,
                status="passed" if self.passed else "gate_failed",
                stopped_at=self.stopped_at,
            ),
            project=dict(project) if project is not None else None,
            pipeline=self.pipeline,
            config=config,
            results=results,
            checks=tuple(checks),
            overrides=tuple(overrides),
            environment=environment(),
            inputs=tuple(inputs),
            datasets=records,
        )


class Pipeline:
    """Steps placed by name, run in order, each through its adapter.

    Parameters
    ----------
    steps
        Step names in the order they run, each registered in `registry`.
        Defaults to `DEFAULT_STEPS`.
    registry
        Where steps are looked up; `dazflow.steps.registry` by default.
    adapters
        How each step is called; `dazflow.pipeline.adapters` by default. A
        step with no adapter is called as ``run(dataset, config)`` with the
        latest dataset.

    Steps never see each other: the pipeline passes an earlier result to a
    later step through the later step's adapter.
    """

    def __init__(
        self,
        steps: Sequence[str] = DEFAULT_STEPS,
        *,
        registry: Registry | None = None,
        adapters: Registry | None = None,
    ) -> None:
        if registry is None:
            from dazflow.steps import registry
        self._registry = registry
        self._adapters = adapters if adapters is not None else _default_adapters()
        steps = tuple(steps)
        if not steps:
            raise PipelineError(
                "the pipeline has no steps",
                observed="pipeline: []",
                change="List the steps in project.yaml: pipeline: [ingest, clean, explore].",
            )
        repeated = sorted({s for s in steps if steps.count(s) > 1})
        if repeated:
            raise PipelineError(
                "a step appears more than once in the pipeline",
                observed=f"repeated: {', '.join(repeated)}",
                change="List each step once.",
            )
        unknown = [s for s in steps if s not in registry]
        if unknown:
            raise PipelineError(
                f"the pipeline names steps that are not registered: {', '.join(unknown)}",
                observed=f"registered steps: {', '.join(registry.keys())}",
                change="Use registered step names, or install the plugin that registers the step.",
            )
        self._steps = steps

    @property
    def steps(self) -> tuple[str, ...]:
        return self._steps

    def __repr__(self) -> str:
        return f"Pipeline({list(self._steps)!r})"

    def step(self, name: str) -> Any:
        """The step module registered under `name`."""
        return self._registry.get(name)

    def select(self, start: str | None = None, stop: str | None = None) -> tuple[str, ...]:
        """The steps from `start` to `stop`, both included, in pipeline order."""
        for what, name in (("--from", start), ("--to", stop)):
            if name is not None and name not in self._steps:
                raise PipelineError(
                    f"{what} names {name!r}, which is not in the pipeline",
                    observed=f"pipeline: {', '.join(self._steps)}",
                    change="Name a step of the pipeline.",
                )
        i = self._steps.index(start) if start is not None else 0
        j = self._steps.index(stop) if stop is not None else len(self._steps) - 1
        if i > j:
            raise PipelineError(
                f"--from {start} comes after --to {stop} in the pipeline",
                observed=f"pipeline: {', '.join(self._steps)}",
                change="Name the earlier step in --from and the later one in --to.",
            )
        return self._steps[i : j + 1]

    def config(self, name: str, value: Config | Mapping[str, Any] | None) -> Config:
        """`value` validated as the step's `Config`, every default filled in."""
        cls = self.step(name).Config
        if isinstance(value, cls):
            return value
        return cls.model_validate(dict(value or {}))

    def run(
        self,
        configs: Mapping[str, Config | Mapping[str, Any] | None] | None = None,
        *,
        root: str | Path | None = None,
        data: Mapping[str, Any] | None = None,
        previous: Dataset | ReadSpec | Mapping[str, Any] | None = None,
        start: str | None = None,
        stop: str | None = None,
        earlier: Mapping[str, StepResult] | None = None,
        earlier_configs: Mapping[str, Config | Mapping[str, Any]] | None = None,
        earlier_record: RunRecord | None = None,
        seed: int | None = None,
    ) -> Run:
        """Run the steps from `start` to `stop`. Prints nothing and writes no files.

        `configs` maps step names to configurations (a missing one takes the
        step's defaults). `root` is the folder relative input paths are read
        from. `data` holds frames for ingest's inputs by name. `previous` is
        the previous data vintage for clean's revisions check: a `Dataset`, or
        a `ReadSpec` read from `root`. A run that starts after the first step
        needs the earlier steps' results in `earlier` (with their datasets),
        and their configurations in `earlier_configs`. `seed` is passed to
        every step's adapter for the step's random draws (``project.yaml: seed``).
        """
        configs = dict(configs or {})
        unknown = sorted(set(configs) - set(self._steps))
        if unknown:
            raise PipelineError(
                f"configuration was given for steps not in the pipeline: {', '.join(unknown)}",
                observed=f"pipeline: {', '.join(self._steps)}",
                change="Add the step to the pipeline, or remove its configuration.",
            )
        chosen = self.select(start, stop)
        before = self._steps[: self._steps.index(chosen[0])]
        earlier = dict(earlier or {})
        missing = [s for s in before if s not in earlier]
        if missing:
            raise PipelineError(
                f"the run starts at {chosen[0]!r} but has no result for {', '.join(missing)}",
                observed=f"earlier results: {', '.join(earlier) or 'none'}",
                change=f"Run the pipeline from its first step, or run {missing[0]} first.",
            )
        root = Path(root) if root is not None else None
        started = _now()
        inputs: list[InputRecord] = []
        vintage = self._previous(previous, root, inputs)

        results: dict[str, StepResult] = {s: earlier[s] for s in before}
        cfgs: dict[str, Config] = {
            s: self.config(s, (earlier_configs or {}).get(s, getattr(earlier[s], "config", None)))
            for s in before
        }
        done: list[str] = []
        stopped = None
        for name in chosen:
            cfg = self.config(name, configs.get(name))
            ctx = StepInputs(
                step=name,
                results=dict(results),
                dataset=_latest_dataset(results),
                root=root,
                data=data,
                previous=vintage,
                seed=seed,
            )
            adapter = self._adapters.get(name) if name in self._adapters else default_adapter
            result = adapter(self.step(name), cfg, ctx)
            if not isinstance(result, StepResult):
                raise PipelineError(
                    f"the step {name!r} did not return a StepResult",
                    observed=f"a {type(result).__name__}",
                    change="Return the step's Result from its run().",
                )
            results[name] = result
            cfgs[name] = cfg
            inputs.extend(ctx.inputs)
            done.append(name)
            if not result.gate.passed:
                stopped = name
                break
        return Run(
            pipeline=self._steps,
            steps=tuple(done),
            results=results,
            configs=cfgs,
            started=started,
            finished=_now(),
            carried=before,
            start=start,
            stop=stop,
            stopped_at=stopped,
            inputs=tuple(inputs),
            carried_record=earlier_record,
        )

    @staticmethod
    def _previous(
        previous: Dataset | ReadSpec | Mapping[str, Any] | None,
        root: Path | None,
        inputs: list[InputRecord],
    ) -> Dataset | None:
        if previous is None:
            return None
        if isinstance(previous, Dataset):
            inputs.append(
                InputRecord(
                    step="clean",
                    name="previous",
                    kind="dataset",
                    fingerprint=previous.fingerprint(),
                )
            )
            return previous
        spec = previous if isinstance(previous, ReadSpec) else ReadSpec.model_validate(previous)
        if spec.path is None:
            raise PipelineError(
                "the previous vintage has no path",
                observed="previous: {path: null}",
                change="Give the previous vintage's file: previous: {path: data/raw/previous.csv}.",
            )
        path = Path(spec.path)
        if root is not None and not path.is_absolute():
            path = root / path
        ds = readers.read(spec.model_copy(update={"path": str(path)}))
        inputs.append(
            InputRecord(
                step="clean",
                name="previous",
                path=spec.path,
                kind="file-sha256",
                fingerprint=file_sha256(path),
            )
        )
        return ds


def _latest_dataset(results: Mapping[str, StepResult]) -> Dataset | None:
    for result in reversed(list(results.values())):
        ds = getattr(result, "dataset", None)
        if isinstance(ds, Dataset):
            return ds
    return None


def _default_adapters() -> Registry:
    return adapters
