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
    passed in memory, by input name; `previous` the previous data vintage.
    An adapter appends what it read to `inputs`.
    """

    step: str
    results: Mapping[str, StepResult]
    dataset: Dataset | None = None
    root: Path | None = None
    data: Mapping[str, Any] | None = None
    previous: Dataset | None = None
    inputs: list[InputRecord] = field(default_factory=list)

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
    # Paths are read from the project folder; the result keeps them as written,
    # so the record does not depend on where the folder is.
    data = dict(inputs.data or {})
    resolved = []
    for inp in config.inputs:
        if inp.path is not None and inp.name not in data:
            path = inputs.resolve(inp.path)
            resolved.append(inp.model_copy(update={"path": str(path)}))
        else:
            resolved.append(inp)
    result = step.run(config.model_copy(update={"inputs": resolved}), data=data or None)
    for inp, res in zip(config.inputs, resolved, strict=True):
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
                    fingerprint=file_sha256(res.path),
                )
            )
    return result.model_copy(update={"config": config})


@register_adapter("clean")
def _clean(step: Any, config: Config, inputs: StepInputs) -> StepResult:
    return step.run(inputs.require_dataset(), config, previous=inputs.previous)
