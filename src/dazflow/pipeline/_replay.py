"""Replay: rerun every step from a run record and compare what it recorded (design §7.10, D43).

Floats match within a relative tolerance, `RTOL`, with an absolute floor,
`ATOL`, for values near zero; everything else (integers, choices, ids,
statuses, text, fingerprints) must match exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from dazflow.core._model import DazflowModel
from dazflow.core.errors import RecordError
from dazflow.pipeline._record import RunRecord

__all__ = ["ATOL", "RTOL", "Difference", "Replay", "compare", "compare_records", "replay"]

# Design D43: a rerun on one machine moves recorded floats by at most about 3e-14
# relative (measured on the example), a refit of an optimised form agrees to 1e-10
# (D36), and one input value changed in its last printed digit moves most results
# by far more (median 2e-4). 1e-9 lies between them, with room either side.
RTOL = 1e-9
# The floor for values that are zero up to rounding (a residual mean, a sum of
# signed errors), whose relative difference means nothing.
ATOL = 1e-12


class Difference(DazflowModel):
    """One recorded value the replay did not reproduce: where it is, and both values.

    `path` names it in the record (``results.select.ranking[0].mase``);
    `recorded` is the record's value and `found` the replay's (or the
    project's current value, for a configuration that changed since the run).
    """

    _error: ClassVar = RecordError
    _hint: ClassVar[str] = "Give the path and both values."

    path: str
    recorded: Any = None
    found: Any = None
    note: str | None = None

    def __str__(self) -> str:
        text = f"{self.path}: recorded {self.recorded!r}, found {self.found!r}"
        return f"{text} ({self.note})" if self.note else text


class Replay(DazflowModel):
    """What a replay found: the steps rerun, the tolerance and every difference.

    `passed` when there is no difference. `notes` say what differs without
    counting as a difference, such as another Python or package version.
    `record` is the replayed run's record, kept in memory only.
    """

    _error: ClassVar = RecordError
    _hint: ClassVar[str] = "Build a replay with replay()."

    record_id: str
    steps: tuple[str, ...] = ()
    rtol: float = RTOL
    atol: float = ATOL
    differences: tuple[Difference, ...] = ()
    notes: tuple[str, ...] = ()
    record: RunRecord | None = Field(None, exclude=True, repr=False)

    @property
    def passed(self) -> bool:
        return not self.differences

    def summary(self) -> str:
        """One line: the run, the steps and the outcome."""
        head = f"replay of run {self.record_id} ({len(self.steps)} steps rerun)"
        tol = f"rtol {self.rtol:g}, atol {self.atol:g}"
        if self.passed:
            return f"{head}: matches its record ({tol})"
        n = len(self.differences)
        return f"{head}: {n} difference{'s' if n != 1 else ''} from its record ({tol})"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def compare(
    recorded: Any,
    found: Any,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    path: str = "",
) -> list[Difference]:
    """Every difference between two JSON values, by path.

    Two numbers of which one is a float match when
    ``|a - b| <= max(rtol * max(|a|, |b|), atol)`` (`math.isclose`); two
    integers, and everything that is not a number, must be equal and of the
    same type (``True`` is not ``1``). Dictionaries are compared key by key,
    lists position by position; a missing key, an extra key or another length
    is a difference.
    """
    out: list[Difference] = []
    _walk(recorded, found, rtol, atol, path, out)
    return out


def _walk(a: Any, b: Any, rtol: float, atol: float, path: str, out: list[Difference]) -> None:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key, value in a.items():
            sub = f"{path}.{key}" if path else str(key)
            if key in b:
                _walk(value, b[key], rtol, atol, sub, out)
            else:
                out.append(Difference(path=sub, recorded=value, note="missing on replay"))
        for key, value in b.items():
            if key not in a:
                sub = f"{path}.{key}" if path else str(key)
                out.append(Difference(path=sub, found=value, note="not in the record"))
        return
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            out.append(Difference(path=path, recorded=len(a), found=len(b), note="lengths differ"))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _walk(x, y, rtol, atol, f"{path}[{i}]", out)
        return
    if _is_number(a) and _is_number(b):
        if isinstance(a, int) and isinstance(b, int):
            same = a == b
        else:
            same = _close(float(a), float(b), rtol, atol)
    else:
        same = type(a) is type(b) and a == b
    if not same:
        out.append(Difference(path=path, recorded=a, found=b))


def _inputs(record: RunRecord) -> dict[str, dict[str, Any]]:
    return {
        f"{i.step}.{i.name}": {"path": i.path, "kind": i.kind, "fingerprint": i.fingerprint}
        for i in record.inputs
    }


def compare_records(
    recorded: RunRecord, replayed: RunRecord, *, rtol: float = RTOL, atol: float = ATOL
) -> tuple[Difference, ...]:
    """The differences between a record and its replay, inputs first, then in pipeline order.

    Compared: every input's fingerprint, the steps run and the status, each
    step's dataset fingerprint, and each step's result in full. The
    configuration is the replay's own input, and the environment is reported
    by `environment_notes`, not compared.
    """
    diffs = compare(_inputs(recorded), _inputs(replayed), rtol=rtol, atol=atol, path="inputs")
    steps_a, steps_b = list(recorded.results), list(replayed.results)
    diffs += compare(steps_a, steps_b, path="run.steps")
    diffs += compare(recorded.run.status, replayed.run.status, path="run.status")
    diffs += compare(
        {k: v.fingerprint for k, v in recorded.datasets.items()},
        {k: v.fingerprint for k, v in replayed.datasets.items()},
        path="datasets",
    )
    for step in steps_a:
        if step in replayed.results:
            diffs += compare(
                recorded.results[step],
                replayed.results[step],
                rtol=rtol,
                atol=atol,
                path=f"results.{step}",
            )
    return tuple(diffs)


def environment_notes(recorded: RunRecord, now: Any) -> tuple[str, ...]:
    """What differs between the recorded environment and `now`: notes, not differences."""
    a, b = recorded.environment.model_dump(), now.model_dump()
    notes = [
        f"{key}: recorded {a[key]}, now {b[key]}"
        for key in ("python", "implementation", "platform", "machine")
        if a[key] != b[key]
    ]
    for name in sorted(set(a["packages"]) | set(b["packages"]), key=str.lower):
        old, new = a["packages"].get(name), b["packages"].get(name)
        if old != new:
            notes.append(f"{name}: recorded {old or 'not installed'}, now {new or 'not installed'}")
    return tuple(notes)


def replay(
    record: RunRecord | str | Path,
    *,
    root: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    previous: Any = None,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Replay:
    """Rerun every step of `record` from its recorded configuration and inputs, and compare.

    `record` is a `RunRecord` or the path of a ``record.json``. `root` is the
    folder its relative input paths are read from. A run that read frames in
    memory needs them again in `data`, by input name; the previous vintage
    is read from the record's ``project: previous`` unless given here.
    Prints nothing and writes no files.
    """
    from dazflow.pipeline._pipeline import Pipeline

    if not isinstance(record, RunRecord):
        record = RunRecord.load(record)
    frames = [i for i in record.inputs if i.kind == "dataset" and i.name != "previous"]
    missing = [i.name for i in frames if i.name not in (data or {})]
    if missing:
        raise RecordError(
            f"the run read frames in memory that the replay does not have: {', '.join(missing)}",
            observed="inputs of kind 'dataset' are recorded by fingerprint only",
            change="Pass the same frames again: replay(record, data={name: frame}).",
        )
    project = record.project or {}
    if previous is None:
        previous = project.get("previous")
    steps = list(record.results)
    pipeline = Pipeline(record.pipeline)
    run = pipeline.run(
        {s: record.config[s] for s in steps},
        root=root,
        data=data,
        previous=previous if "clean" in steps else None,
        stop=steps[-1] if steps else None,
        seed=project.get("seed"),
        project=record.project,
        environment=record.environment,
    )
    again = run.record(record.project)
    from dazflow.pipeline._record import environment

    return Replay(
        record_id=record.run.id,
        steps=run.steps,
        rtol=rtol,
        atol=atol,
        differences=compare_records(record, again, rtol=rtol, atol=atol),
        notes=environment_notes(record, environment()),
        record=again,
    )
