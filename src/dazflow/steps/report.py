"""report: the run as a model document, built from the run record alone (design §7.10).

The step reads nothing but the record of the steps before it: their
configuration as run, their results with every table, figure and finding,
their gates, the inputs read and the environment. From it, it builds:

- a **document** (`dazflow.render.Document`): a contents list, a summary,
  one section per step in pipeline order holding that step's finding,
  tables and figures, and two appendices, the configuration of every step
  and the environment and inputs;
- the report's own tables: the run summary (`T10.1`), the environment
  (`T10.2`), the inputs (`T10.3`), the configuration (`T10.4`) and its gate
  (`T10.5`), with a finding (`S10.1`);
- a **replay script**: plain Python that reruns the project from its
  ``reports/record.json`` and compares every recorded number.

`Project.write` writes the document as Word and the script to
``reports/document/``; the step itself writes nothing.

Gate: every step of the pipeline before report is in the record
(``complete``), and every one of their gates passed (``gates``). Whether a
replay reproduces the record is `dazflow replay`'s exit status (design D44).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, TypeAdapter

from dazflow.core.artefact import ArtefactIds, Column, Figure, Table
from dazflow.core.config import Config
from dazflow.core.errors import ReportError
from dazflow.core.finding import Finding
from dazflow.core.gate import Check, Gate, Override
from dazflow.core.result import Artefact, StepResult

__all__ = [
    "IDS",
    "TITLES",
    "Config",
    "ReplayTolerance",
    "ReportConfig",
    "ReportResult",
    "Result",
    "replay_script",
    "run",
]

# Never renumber: documents cite these (design §9, D21). Report is design §7.10.
IDS = ArtefactIds(
    "report",
    tables={"summary": 1, "environment": 2, "inputs": 3, "configuration": 4, "gate": 5},
    findings={"summary": 1},
)

# Section titles of the built-in steps: their headings in design §7. A step
# without one (a plugin's) is titled by its name.
TITLES: Mapping[str, str] = {
    "ingest": "Ingest: import and align",
    "clean": "Clean: data quality and treatment",
    "explore": "Explore: each series alone, each driver against the target",
    "build": "Build: candidate models",
    "segment": "Segment: one model or several",
    "select": "Select: rank and choose",
    "forecast": "Forecast: projection and calibration",
    "diagnose": "Diagnose: test the chosen model",
    "adjust": "Adjust: adjustments and margins",
}
CONTENTS = "Contents"
SUMMARY_TITLE = "Summary"
CONFIGURATION_TITLE = "Appendix A. Configuration"
ENVIRONMENT_TITLE = "Appendix B. Environment and inputs"

SUMMARY = (
    "The document reports {{ steps }} steps, {{ first }} to {{ last }}, in pipeline order, "
    "with {{ tables }} tables, {{ figures }} figures and {{ findings }} findings. "
    "{% if failed %}The gates of {{ failed }} failed{% else %}Every step's gate passed"
    "{% endif %}{% if overridden %}, {{ overridden }} {{ 'check' if overridden == 1 else 'checks' }} "
    "by override with a reason"
    "{% endif %}{% if missing %}; {{ missing }} are not in the record{% endif %}."
)

_ARTEFACTS = TypeAdapter(dict[str, Artefact])


# --- configuration ---------------------------------------------------------------


class ReplayTolerance(Config):
    """How closely a replay must reproduce each recorded float (design D43)."""

    rtol: float = Field(
        1e-9, gt=0, lt=1, description="Relative tolerance for floats; all else matches exactly."
    )
    atol: float = Field(
        1e-12, ge=0, description="Absolute tolerance for floats near zero, in their own units."
    )


class ReportConfig(Config):
    """``config/report.yaml``: the document's title and file, and the replay's tolerance."""

    title: str | None = Field(None, description="The document's title; unset: the project name.")
    file: str = Field(
        "document.docx",
        pattern=r"^[^/\\]+\.docx$",
        description="The Word file written to reports/document/.",
    )
    replay: ReplayTolerance = Field(
        default_factory=ReplayTolerance,
        description="The tolerance of `dazflow replay` and of the replay script.",
    )
    overrides: list[Override] = Field(default_factory=list)


class ReportResult(StepResult):
    """What report returns: the document and replay script (in memory only) and its tables.

    `steps` are the steps documented, in pipeline order; `sections` the
    document's section titles; `missing` the pipeline steps before report
    that the record lacks; `failed` the documented steps whose gate failed.
    `document` and `script` are not in `to_dict()`: the record they come
    from holds everything in them.
    """

    config: ReportConfig
    title: str
    steps: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    document: Any = Field(None, exclude=True, repr=False)
    script: str | None = Field(None, exclude=True, repr=False)


Config = ReportConfig
Result = ReportResult


# --- the record -------------------------------------------------------------------


def _record_dict(record: Any) -> dict[str, Any]:
    data = record.to_dict() if hasattr(record, "to_dict") else record
    if (
        not isinstance(data, Mapping)
        or data.get("kind") != "dazflow.record"
        or not isinstance(data.get("results"), Mapping)
    ):
        kind = data.get("kind") if isinstance(data, Mapping) else type(data).__name__
        raise ReportError(
            "the report needs a run record, and this is not a dazflow run record",
            observed=f"kind {kind!r}",
            change="Pass a dazflow.pipeline.RunRecord, or the dictionary of a reports/record.json.",
        )
    return dict(data)


def _documented(data: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The steps to document in pipeline order, and the pipeline steps before report missing."""
    pipeline = list(data.get("pipeline") or data["results"])
    before = pipeline[: pipeline.index("report")] if "report" in pipeline else pipeline
    steps = [s for s in before if s in data["results"]]
    steps += [s for s in data["results"] if s not in before and s != "report"]
    missing = [s for s in before if s not in data["results"]]
    return steps, missing


def _title(step: str) -> str:
    return TITLES.get(step, step.replace("_", " ").capitalize())


# --- tables -----------------------------------------------------------------------


def _summary_table(steps: Sequence[str], gates: Mapping[str, Gate], results: Mapping) -> Table:
    rows = []
    for step in steps:
        gate = gates[step]
        failed = sum(c.status == "fail" for c in gate.checks)
        overridden = sum(c.status == "overridden" for c in gate.checks)
        status = "fail" if failed else "overridden" if overridden else "pass"
        rows.append(
            [step, len(gate.checks), failed, overridden, len(results[step]["artefacts"]), status]
        )
    return Table(
        id=IDS.table("summary"),
        caption="The run, step by step",
        columns=[
            Column(name="step", label="Step", kind="text"),
            Column(name="checks", label="Checks", kind="integer"),
            Column(name="failed", label="Failed", kind="integer"),
            Column(name="overridden", label="Overridden", kind="integer"),
            Column(name="artefacts", label="Artefacts", kind="integer"),
            Column(name="gate", label="Gate", kind="status"),
        ],
        rows=rows,
        notes=("Overridden checks failed and were accepted in configuration, with a reason.",),
    )


def _environment_table(env: Mapping[str, Any]) -> Table:
    rows = [
        ["Python", env["python"]],
        ["Implementation", env["implementation"]],
        ["Platform", env["platform"]],
        ["Machine", env["machine"]],
    ]
    rows += [[name, v] for name, v in env["packages"].items()]
    return Table(
        id=IDS.table("environment"),
        caption="Environment of the run",
        columns=[
            Column(name="item", label="Item", kind="text"),
            Column(name="version", label="Version or value", kind="text"),
        ],
        rows=rows,
        notes=("Packages: dazflow and its installed requirements.",),
    )


def _inputs_table(inputs: Sequence[Mapping[str, Any]]) -> Table:
    return Table(
        id=IDS.table("inputs"),
        caption="Inputs read by the run",
        columns=[
            Column(name="step", label="Step", kind="text"),
            Column(name="input", label="Input", kind="text"),
            Column(name="path", label="Path", kind="text"),
            Column(name="kind", label="Kind", kind="text"),
            Column(name="fingerprint", label="Fingerprint", kind="text"),
        ],
        rows=[[i["step"], i["name"], i.get("path"), i["kind"], i["fingerprint"]] for i in inputs],
        notes=(
            "file-sha256: the SHA-256 of the file's bytes; dataset: the fingerprint of a frame "
            "passed in memory.",
        ),
    )


def _leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(node, Mapping) and node:
        out = []
        for key, value in node.items():
            out += _leaves(value, f"{prefix}.{key}" if prefix else str(key))
        return out
    return [(prefix, node)]


def _setting_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _configuration_table(steps: Sequence[str], config: Mapping[str, Any]) -> Table:
    rows = [
        [step, key, _setting_text(value)]
        for step in steps
        for key, value in _leaves(config.get(step, {}))
    ]
    return Table(
        id=IDS.table("configuration"),
        caption="Configuration of every step as run, every default filled in",
        columns=[
            Column(name="step", label="Step", kind="text"),
            Column(name="setting", label="Setting", kind="text"),
            Column(name="value", label="Value", kind="text"),
        ],
        rows=rows,
        notes=("Values other than text are written as JSON.",),
    )


# --- the document -------------------------------------------------------------------


def _step_items(artefacts: Mapping[str, Any]) -> list[Any]:
    findings = [(k, a) for k, a in artefacts.items() if isinstance(a, Finding)]
    items: list[Any] = []
    for key, finding in findings:
        items += [f"Finding {key}.", finding]
    items += [a for a in artefacts.values() if isinstance(a, Table)]
    items += [a for a in artefacts.values() if isinstance(a, Figure)]
    return items


def replay_script(config: ReportConfig) -> str:
    """The replay script: plain Python that reruns the project from its record and compares."""
    tol = config.replay
    return f'''"""Replay this project's last run: rerun every step from reports/record.json and compare.

Written by dazflow's report step. Every step is rerun from the configuration
and inputs the record holds; every recorded number must be reproduced, floats
within the relative tolerance RTOL (ATOL near zero), everything else exactly.
The project's current configuration must also still be the recorded one.
Exit status: 0 when the replay matches the record, 2 when it does not.

    python reports/document/replay.py
"""

import sys
from pathlib import Path

from dazflow import project

ROOT = Path(__file__).resolve().parents[2]  # the project folder
RTOL = {tol.rtol!r}
ATOL = {tol.atol!r}


def main() -> int:
    replay = project.load(ROOT).replay(rtol=RTOL, atol=ATOL)
    for difference in replay.differences:
        print(difference)
    for note in replay.notes:
        print(f"note: {{note}}")
    print(replay.summary())
    return 0 if replay.passed else 2


if __name__ == "__main__":
    sys.exit(main())
'''


def run(record: Any, config: ReportConfig | Mapping[str, Any] | None = None) -> ReportResult:
    """Build the document, its tables and the replay script from a run record.

    `record` is a `dazflow.pipeline.RunRecord` or its dictionary (a
    ``reports/record.json`` read as JSON), holding the steps to report. A
    result of ``report`` in it is not reported on. Prints nothing and writes
    no files.
    """
    from dazflow.render.document import Document, Section

    cfg = config if isinstance(config, ReportConfig) else ReportConfig.model_validate(config or {})
    data = _record_dict(record)
    results = data["results"]
    steps, missing = _documented(data)
    gates = {s: Gate.model_validate(results[s].get("gate") or {}) for s in steps}
    artefacts = {s: _ARTEFACTS.validate_python(results[s].get("artefacts") or {}) for s in steps}
    failed = [s for s in steps if not gates[s].passed]
    project = data.get("project") or {}
    title = cfg.title or project.get("name") or "dazflow run"

    checks = [
        Check(
            name="complete",
            observed=len(missing),
            threshold=0,
            op="<=",
            unit="steps",
        ),
        Check(
            name="gates",
            observed=len(failed),
            threshold=0,
            op="<=",
            unit="steps",
        ),
    ]
    gate = Gate.evaluate(checks, cfg.overrides)
    every = [a for s in steps for a in artefacts[s].values()]
    finding = Finding(
        template=SUMMARY,
        key="report.summary",
        values={
            "steps": len(steps),
            "first": steps[0] if steps else "none",
            "last": steps[-1] if steps else "none",
            "tables": sum(isinstance(a, Table) for a in every),
            "figures": sum(isinstance(a, Figure) for a in every),
            "findings": sum(isinstance(a, Finding) for a in every),
            "failed": ", ".join(failed),
            "overridden": sum(len(gates[s].overridden) for s in steps),
            "missing": ", ".join(missing),
        },
    )
    tables = {
        "summary": _summary_table(steps, gates, results),
        "environment": _environment_table(data["environment"]),
        "inputs": _inputs_table(data.get("inputs") or []),
        "configuration": _configuration_table(steps, data.get("config") or {}),
        "gate": Table.from_gate(gate, id=IDS.table("gate"), caption="Report checks"),
    }
    own = {t.id: t for t in tables.values()}
    own[IDS.finding("summary")] = finding

    summary_items: list[Any] = []
    if project.get("description"):
        summary_items.append(project["description"])
    summary_items += [f"Finding {IDS.finding('summary')}.", finding]
    summary_items += [tables["summary"], tables["gate"]]
    body = [
        Section(title=SUMMARY_TITLE, items=tuple(summary_items)),
        *(Section(title=_title(s), items=tuple(_step_items(artefacts[s]))) for s in steps),
        Section(title=CONFIGURATION_TITLE, items=(tables["configuration"],)),
        Section(title=ENVIRONMENT_TITLE, items=(tables["environment"], tables["inputs"])),
    ]
    contents = Section(
        title=CONTENTS,
        items=tuple(f"{i}. {s.title}" for i, s in enumerate(body, start=1)),
    )
    document = Document(title=title, sections=(contents, *body))
    return ReportResult(
        gate=gate,
        finding=finding,
        artefacts=own,
        config=cfg,
        title=title,
        steps=tuple(steps),
        sections=tuple(s.title for s in document.sections),
        missing=tuple(missing),
        failed=tuple(failed),
        document=document,
        script=replay_script(cfg),
    )
