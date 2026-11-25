"""RunRecord: what a run did, enough to replay it (design §10.1)."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import Field

from dazflow.core._model import DazflowModel
from dazflow.core.errors import RecordError
from dazflow.core.result import StepResult
from dazflow.pipeline._adapters import InputRecord

__all__ = [
    "RECORD_KIND",
    "SCHEMA_VERSION",
    "DatasetRecord",
    "Environment",
    "RunInfo",
    "RunRecord",
    "environment",
]

# Raise when a field changes meaning or is removed; Stage 11 replay reads by it.
SCHEMA_VERSION = 1
RECORD_KIND = "dazflow.record"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class _Part(DazflowModel):
    _error: ClassVar = RecordError
    _hint: ClassVar[str] = "Correct the record fields listed in 'observed'."


class RunInfo(_Part):
    """When the run happened, what it ran and how it ended.

    `steps` were run; `carried` were taken from an earlier record unchanged.
    `status` is ``passed``, or ``gate_failed`` when the run stopped at
    `stopped_at` because that step's gate failed.
    """

    id: str = Field(min_length=1)
    started: str
    finished: str
    start: str | None = None
    stop: str | None = None
    steps: tuple[str, ...] = ()
    carried: tuple[str, ...] = ()
    status: Literal["passed", "gate_failed"]
    stopped_at: str | None = None


class Environment(_Part):
    """Where the run ran: Python, platform and the versions of dazflow and its dependencies."""

    python: str
    implementation: str
    platform: str
    machine: str
    packages: dict[str, str]


class DatasetRecord(_Part):
    """A step's dataset: its fingerprint and, when the project folder holds it, its path."""

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str | None = None


def environment() -> Environment:
    """The running environment, with dazflow's installed version and its requirements'."""
    names = {"dazflow"}
    try:
        requires = importlib.metadata.requires("dazflow") or []
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - a source tree
        requires = []
    for req in requires:
        m = _NAME.match(req)
        if m:
            names.add(m.group(0))
    packages = {}
    for name in sorted(names, key=str.lower):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue  # an extra that is not installed
    return Environment(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        packages=packages,
    )


class RunRecord(_Part):
    """The run record, written as ``reports/record.json``.

    Fields:

    - `schema_version` and `kind`: the format; `load` refuses another version.
    - `run`: identifier, times, steps run and carried, status.
    - `project`: the project configuration as run, defaults filled in (or None).
    - `pipeline`: the step names in order.
    - `config`: each step's configuration as run, every default filled in.
    - `results`: each step's `StepResult.to_dict()`.
    - `checks`: every gate check of every step, with its step.
    - `overrides`: every configured gate override, with its step.
    - `environment`: Python, platform and package versions.
    - `inputs`: a fingerprint of every input read.
    - `datasets`: each step's dataset fingerprint and path.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    kind: Literal["dazflow.record"] = RECORD_KIND
    run: RunInfo
    project: dict[str, Any] | None = None
    pipeline: tuple[str, ...]
    config: dict[str, dict[str, Any]]
    results: dict[str, dict[str, Any]]
    checks: tuple[dict[str, Any], ...] = ()
    overrides: tuple[dict[str, Any], ...] = ()
    environment: Environment
    inputs: tuple[InputRecord, ...] = ()
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The record as strict JSON-safe data."""
        return json.loads(self.model_dump_json())

    @classmethod
    def from_dict(cls, data: Any) -> Self:
        """A record from `to_dict()` output; another schema version is a `RecordError`."""
        if not isinstance(data, dict):
            raise RecordError(
                "the run record is not a JSON object",
                observed=f"a {type(data).__name__}",
                change="Read a record.json written by dazflow.",
            )
        version = data.get("schema_version")
        if version != SCHEMA_VERSION or data.get("kind") != RECORD_KIND:
            raise RecordError(
                f"the run record is not a {RECORD_KIND} of schema version {SCHEMA_VERSION}",
                observed=f"kind {data.get('kind')!r}, schema_version {version!r}",
                change="Read it with the dazflow version that wrote it, or run the project again.",
            )
        return cls.model_validate(data)

    def dumps(self) -> str:
        """The record as JSON text, two-space indented, with a final newline."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def dump(self, path: str | Path) -> Path:
        """Write the record to `path` as UTF-8 JSON."""
        path = Path(path)
        path.write_text(self.dumps(), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a record written by `dump`."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as err:
            raise RecordError(
                "the run record cannot be read",
                observed=f"{path}: {err.strerror or err}",
                change="Run the project first (dazflow run), or check the path.",
            ) from err
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise RecordError(
                "the run record is not valid JSON",
                observed=f"{path}: {err}",
                change="Run the project again to rewrite reports/record.json.",
            ) from err
        return cls.from_dict(data)

    def result(self, step: str) -> StepResult:
        """The step's result rebuilt as its own `Result` type, with ``dataset=None``."""
        from dazflow.steps import registry

        if step not in self.results:
            raise RecordError(
                f"the run record holds no result for {step!r}",
                observed=f"results: {', '.join(self.results) or 'none'}",
                change=f"Run the step first: dazflow {step}.",
            )
        return registry.get(step).Result.from_dict(self.results[step])
