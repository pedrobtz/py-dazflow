"""How the pipeline calls each step: the inputs a step receives and the adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from dazflow import readers
from dazflow.core._model import DazflowModel
from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import PipelineError, RecordError
from dazflow.core.registry import Registry
from dazflow.core.result import StepResult

__all__ = [
    "Adapter",
    "InputRecord",
    "StepInputs",
    "adapters",
    "default_adapter",
    "file_sha256",
    "register_adapter",
]


class InputRecord(DazflowModel):
    """One input a run read: a file by the SHA-256 of its bytes, or a frame by its fingerprint.

    `path` is the path as configured, relative to the project folder when it was.
    """

    _error: ClassVar = RecordError
    _hint: ClassVar[str] = "Give the step, the input's name, its kind and a 64-digit fingerprint."

    step: str = Field(min_length=1)
    name: str = Field(min_length=1)
    path: str | None = None
    kind: str = Field(pattern=r"^(file-sha256|dataset)$")
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of a file's bytes; a folder hashes its files' relative paths and bytes."""
    path = Path(path)
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(f.relative_to(path).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class StepInputs:
    """What a step's adapter has to call the step with.

    `results` holds the earlier steps' results in pipeline order (run now,
    or carried from an earlier run's record); `dataset` is the dataset of
    the latest of them that holds one. `root` is the project folder that
    relative paths are read from (``None``: as given). `data` holds frames
    passed in memory, by input name; `previous` the previous data vintage;
    `seed` the project's seed (``project.yaml: seed``) for a step's random
    draws. An adapter appends what it read to `inputs`. `record()` gives the
    run record of the steps before this one, for a step that reports on them.
    """

    step: str
    results: Mapping[str, StepResult]
    dataset: Dataset | None = None
    root: Path | None = None
    data: Mapping[str, Any] | None = None
    previous: Dataset | None = None
    seed: int | None = None
    inputs: list[InputRecord] = field(default_factory=list)
    # The run so far (a `Run` of the steps before this one) and the project configuration.
    recorder: Callable[[], Any] | None = field(default=None, repr=False)
    project: Any = field(default=None, repr=False)

    def record(self) -> Any:
        """The `RunRecord` of the run so far: every step before this one, run or carried."""
        if self.recorder is None:
            raise PipelineError(
                f"the step {self.step!r} needs the run record and was not called by a pipeline",
                observed="StepInputs.recorder is None",
                change="Run the step through dazflow.pipeline.Pipeline, or pass it a record.",
            )
        return self.recorder().record(self.project)

    def resolve(self, path: str | Path) -> Path:
        """`path` read from the project folder when it is relative and there is one."""
        path = Path(path)
        return path if self.root is None or path.is_absolute() else self.root / path

    def require_dataset(self) -> Dataset:
        """The latest dataset, or `PipelineError` naming the step that needs one."""
        if self.dataset is None:
            earlier = ", ".join(self.results) or "none"
            raise PipelineError(
                f"the step {self.step!r} needs a dataset and no earlier step returned one",
                observed=f"earlier steps: {earlier}",
                change="Place a step that returns a dataset, such as ingest, before it.",
            )
        return self.dataset

    def dataset_with(self, fingerprint: str | None) -> Dataset:
        """The dataset of an earlier result with this fingerprint (the data a step ran on)."""
        for result in self.results.values():
            ds = getattr(result, "dataset", None)
            if isinstance(ds, Dataset) and getattr(result, "fingerprint", None) == fingerprint:
                return ds
        raise PipelineError(
            f"the step {self.step!r} needs a dataset no earlier step holds",
            observed=f"fingerprint {str(fingerprint)[:12]}...; earlier steps: "
            f"{', '.join(self.results) or 'none'}",
            change="Run the pipeline from the step that returned it (dazflow run).",
        )

    def require(self, name: str) -> StepResult:
        """The earlier result of the step `name`, or `PipelineError` saying it must come first."""
        if name not in self.results:
            raise PipelineError(
                f"the step {self.step!r} needs the result of {name!r}",
                observed=f"earlier steps: {', '.join(self.results) or 'none'}",
                change=f"Place {name} before {self.step} in the pipeline.",
            )
        return self.results[name]


Adapter = Callable[[Any, Config, StepInputs], StepResult]

# Step name -> adapter. A step with no adapter is called as run(dataset, config).
adapters = Registry("step adapter")


def register_adapter(step: str) -> Callable[[Adapter], Adapter]:
    """Decorator: the function the pipeline calls the step named `step` through."""
    return adapters.register(step)


def default_adapter(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    """``step.run(dataset, config)`` with the latest dataset: the common case."""
    return step.run(inputs.require_dataset(), config)


@register_adapter("ingest")
def _ingest(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Paths are read from the project folder and kept as written, in the result and in the
    # dataset's series sources, so neither depends on where the folder is (design D35).
    data = dict(inputs.data or {})
    result = step.run(config, data=data or None, root=inputs.root)
    for inp in config.inputs:
        if inp.name in data:
            frame = readers.from_frame(
                data[inp.name], date=inp.date, frequency=inp.frequency, columns=inp.columns
            )
            inputs.inputs.append(
                InputRecord(
                    step=inputs.step, name=inp.name, kind="dataset", fingerprint=frame.fingerprint()
                )
            )
        else:
            inputs.inputs.append(
                InputRecord(
                    step=inputs.step,
                    name=inp.name,
                    path=inp.path,
                    kind="file-sha256",
                    fingerprint=file_sha256(inputs.resolve(inp.path)),
                )
            )
    return result


@register_adapter("clean")
def _clean(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    return step.run(inputs.require_dataset(), config, previous=inputs.previous)


@register_adapter("build")
def _build(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Build takes the series in levels, as explore received them, and explore's result.
    explore = inputs.results.get("explore")
    if explore is None:
        return step.run(inputs.require_dataset(), config)
    return step.run(inputs.dataset_with(explore.input_fingerprint), config, explore=explore)


@register_adapter("select")
def _select(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Select takes the levels build ran on and build's result; a scenarios file is read
    # from the project folder, recorded by its hash, and kept in the result as written.
    build = inputs.require("build")
    levels = inputs.dataset_with(build.fingerprint)
    spec = getattr(config, "scenarios", None)
    if spec is None or levels.scenarios is not None:
        return step.run(levels, config, build=build, seed=inputs.seed)
    path = inputs.resolve(spec.path)
    run_config = config.model_copy(
        update={"scenarios": spec.model_copy(update={"path": str(path)})}
    )
    result = step.run(levels, run_config, build=build, seed=inputs.seed)
    inputs.inputs.append(
        InputRecord(
            step=inputs.step,
            name="scenarios",
            path=spec.path,
            kind="file-sha256",
            fingerprint=file_sha256(path),
        )
    )
    return result.model_copy(update={"config": config})


@register_adapter("forecast")
def _forecast(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Forecast takes the levels select ran on and select's result. Scenarios are declared
    # once, in select's configuration (design D33, D40): the file is read from the project
    # folder and recorded by its hash under this step too.
    select = inputs.require("select")
    levels = inputs.dataset_with(select.fingerprint)
    spec = getattr(getattr(select, "config", None), "scenarios", None)
    if spec is None or levels.scenarios is not None:
        return step.run(levels, config, select=select, seed=inputs.seed)
    path = inputs.resolve(spec.path)
    frame, severity = readers.read_scenarios(spec, inputs.root)
    result = step.run(
        levels.with_scenarios(frame, severity), config, select=select, seed=inputs.seed
    )
    inputs.inputs.append(
        InputRecord(
            step=inputs.step,
            name="scenarios",
            path=spec.path,
            kind="file-sha256",
            fingerprint=file_sha256(path),
        )
    )
    return result


@register_adapter("diagnose")
def _diagnose(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Diagnose takes the levels select ran on, select's result and, when it ran, forecast's
    # (for the sensitivity horizon). Scenarios are select's, read and recorded as forecast's.
    select = inputs.require("select")
    forecast = inputs.results.get("forecast")
    levels = inputs.dataset_with(select.fingerprint)
    spec = getattr(getattr(select, "config", None), "scenarios", None)
    if spec is None or levels.scenarios is not None:
        return step.run(levels, config, select=select, forecast=forecast)
    path = inputs.resolve(spec.path)
    frame, severity = readers.read_scenarios(spec, inputs.root)
    result = step.run(
        levels.with_scenarios(frame, severity), config, select=select, forecast=forecast
    )
    inputs.inputs.append(
        InputRecord(
            step=inputs.step,
            name="scenarios",
            path=spec.path,
            kind="file-sha256",
            fingerprint=file_sha256(path),
        )
    )
    return result


@register_adapter("report")
def _report(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    # Report is built from the run record alone (design §7.10): the record of every step
    # before it, as it would be written now.
    return step.run(inputs.record(), config)
