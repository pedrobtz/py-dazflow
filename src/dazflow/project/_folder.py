"""The project folder (design §10): reading its configuration, running it, writing its outputs."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import Field

from dazflow import readers
from dazflow.core.config import Config
from dazflow.core.errors import ProjectError
from dazflow.core.result import StepResult
from dazflow.pipeline import DEFAULT_STEPS, Pipeline, Run, RunRecord
from dazflow.project._commented import commented_yaml
from dazflow.project._datasets import read_dataset, write_dataset
from dazflow.readers._base import ReadSpec

__all__ = ["PROJECT_FILE", "RECORD_PATH", "Project", "ProjectConfig", "init", "load"]

PROJECT_FILE = "project.yaml"
RECORD_PATH = "reports/record.json"
# Folders of design §10 that `init` creates.
LAYOUT = (
    "config",
    "data/raw",
    "data/interim",
    "data/processed",
    "models/work",
    "reports",
)
# Where a step's dataset is written: ingest and clean to interim, later steps to processed.
INTERIM_STEPS = ("ingest", "clean")


class ProjectConfig(Config):
    """``project.yaml``: the project's name, its pipeline and the settings every step shares."""

    name: str = Field(min_length=1, description="The project's name, used in reports.")
    description: str | None = Field(None, description="One line on what the model is for.")
    pipeline: list[str] = Field(
        default_factory=lambda: list(DEFAULT_STEPS),
        description="The steps `dazflow run` runs, in order; each reads config/<step>.yaml.",
    )
    seed: int = Field(0, description="Seed for any random draw a step makes.")
    style: str | None = Field(
        None,
        description="A style YAML file merged onto the house style, relative to this folder.",
    )
    previous: ReadSpec | None = Field(
        None,
        description=(
            "The previous vintage of the data, for clean's revisions check: a file of the same\n"
            "series, such as an earlier run's data/interim/ingest.csv kept under data/raw/.\n"
            "Unset: the revisions check does not apply. Example: previous: {path: data/raw/v1.csv}"
        ),
    )


def _folder(step: str) -> str:
    return "data/interim" if step in INTERIM_STEPS else "data/processed"


class Project:
    """A project folder: ``project.yaml``, ``config/<step>.yaml``, data and reports.

    `run()` reads the folder and writes nothing; `write(run)` writes the
    datasets, each step's artefacts under ``reports/<step>/`` and
    ``reports/record.json``.
    """

    def __init__(self, root: str | Path, config: ProjectConfig) -> None:
        self.root = Path(root)
        self.config = config

    def __repr__(self) -> str:
        return f"Project({str(self.root)!r}, name={self.config.name!r})"

    @classmethod
    def load(cls, root: str | Path = ".") -> Project:
        """Read ``project.yaml`` from `root`."""
        root = Path(root)
        path = root / PROJECT_FILE
        if not path.is_file():
            raise ProjectError(
                f"{root} is not a dazflow project",
                observed=f"no {PROJECT_FILE} in {root.resolve()}",
                change="Run the command in a project folder, pass --project, or create one: dazflow init <dir>.",
            )
        return cls(root, ProjectConfig.load(path))

    # --- configuration ---------------------------------------------------------

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self.config.pipeline)

    def config_path(self, step: str) -> Path:
        return self.root / "config" / f"{step}.yaml"

    def step_config(self, step: str) -> Config:
        """The step's configuration from ``config/<step>.yaml``; no file: the step's defaults."""
        cls = self.pipeline.step(step).Config
        path = self.config_path(step)
        return cls.load(path) if path.is_file() else cls.model_validate({})

    def configs(self) -> dict[str, Config]:
        """Every pipeline step's configuration."""
        return {s: self.step_config(s) for s in self.pipeline.steps}

    def style_path(self) -> Path | None:
        return None if self.config.style is None else self.root / self.config.style

    def dataset_path(self, step: str) -> Path:
        return self.root / _folder(step) / f"{step}.csv"

    @property
    def record_path(self) -> Path:
        return self.root / RECORD_PATH

    def record(self) -> RunRecord | None:
        """The last run's record, or None when the project has not run."""
        return RunRecord.load(self.record_path) if self.record_path.is_file() else None

    # --- validate --------------------------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Check the configuration and that every input reads, without running a step.

        Raises the first `DazflowError` found; returns what was checked.
        """
        pipeline = self.pipeline
        known = set(pipeline.steps)
        config_dir = self.root / "config"
        stray = (
            sorted(p.name for p in config_dir.glob("*.yaml") if p.stem not in known)
            if config_dir.is_dir()
            else []
        )
        if stray:
            raise ProjectError(
                f"config/ holds files for steps not in the pipeline: {', '.join(stray)}",
                observed=f"pipeline: {', '.join(pipeline.steps)}",
                change="Rename the file after its step, or add the step to the pipeline.",
            )
        configs = self.configs()
        inputs = []
        for inp in getattr(configs.get("ingest"), "inputs", []) or []:
            if inp.path is None:
                raise ProjectError(
                    f"the input {inp.name!r} has no path",
                    observed="a project reads every input from a file",
                    change=f"Give the input a path under data/raw/: path: data/raw/{inp.name}.csv.",
                )
            readers.read(inp.model_copy(update={"path": str(self.root / inp.path)}))
            inputs.append({"name": inp.name, "path": inp.path})
        spec = getattr(configs.get("select"), "scenarios", None)
        if spec is not None:
            from dazflow.steps import select

            select.read_scenarios(spec, self.root)
            inputs.append({"name": "scenarios", "path": spec.path})
        if self.config.previous is not None:
            spec = self.config.previous
            readers.read(spec.model_copy(update={"path": str(self.root / (spec.path or ""))}))
            inputs.append({"name": "previous", "path": spec.path})
        if self.style_path() is not None:
            from dazflow.render import load_style

            load_style(self.style_path())
        return {"project": self.config.name, "steps": list(pipeline.steps), "inputs": inputs}

    # --- run -------------------------------------------------------------------

    def _earlier(
        self, pipeline: Pipeline, start: str | None
    ) -> tuple[dict[str, StepResult], dict[str, Config], RunRecord | None]:
        if start is None or start == pipeline.steps[0]:
            return {}, {}, None
        before = pipeline.steps[: pipeline.steps.index(start)]
        record = self.record()
        missing = [s for s in before if record is None or s not in record.results]
        if missing:
            raise ProjectError(
                f"{start} needs the results of {', '.join(missing)}, which the project has not run",
                observed=f"{RECORD_PATH}: "
                + ("missing" if record is None else f"results for {', '.join(record.results)}"),
                change=f"Run them first: dazflow run --to {before[-1]}.",
            )
        results, configs = {}, {}
        for s in before:
            result = record.result(s)
            entry = record.datasets.get(s)
            if entry is not None and entry.path is not None:
                ds = read_dataset(self.root / entry.path, entry.fingerprint)
                result = result.model_copy(update={"dataset": ds})
            results[s] = result
            configs[s] = record.config[s]
        return results, configs, record

    def run(
        self,
        start: str | None = None,
        stop: str | None = None,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> Run:
        """Run the pipeline from `start` to `stop` on the folder's data. Writes nothing.

        Steps before `start` are carried from the last record, their datasets
        read from the folder and checked against their recorded fingerprints.
        """
        pipeline = self.pipeline
        chosen = pipeline.select(start, stop)
        earlier, earlier_configs, record = self._earlier(pipeline, chosen[0])
        return pipeline.run(
            {s: self.step_config(s) for s in chosen},
            root=self.root,
            data=data,
            previous=self.config.previous if "clean" in chosen else None,
            start=start,
            stop=stop,
            earlier=earlier,
            earlier_configs=earlier_configs,
            earlier_record=record,
            seed=self.config.seed,
        )

    # --- write -----------------------------------------------------------------

    def write(self, run: Run) -> RunRecord:
        """Write the run's datasets, each step's artefacts and ``reports/record.json``."""
        datasets: dict[str, str] = {}
        for name in run.steps:
            ds = getattr(run.results[name], "dataset", None)
            if ds is not None:
                path = self.dataset_path(name)
                write_dataset(ds, path)
                datasets[name] = path.relative_to(self.root).as_posix()
        record = run.record(self.config, datasets)
        for name in run.steps:
            self._write_reports(name, run.results[name], record)
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        record.dump(self.record_path)
        return record

    def _write_reports(self, name: str, result: StepResult, record: RunRecord) -> None:
        from dazflow.core.artefact import Figure, Source, Table
        from dazflow.render import load_style, write

        folder = self.root / "reports" / name
        if folder.exists():
            shutil.rmtree(folder)  # the step's folder holds only what its last run wrote
        folder.mkdir(parents=True)
        style = load_style(self.style_path())
        fp = getattr(result, "fingerprint", None)
        source = (
            Source(
                run_id=record.run.id,
                fingerprint=fp,
                date=dt.date.fromisoformat(record.run.finished[:10]),
            )
            if fp is not None
            else None
        )
        for key, artefact in result.artefacts.items():
            if isinstance(artefact, Table):
                write(artefact, folder / f"{key}.md", style=style, source=source)
                write(artefact, folder / f"{key}.html", style=style, source=source)
            elif isinstance(artefact, Figure):
                write(artefact, folder / f"{key}.svg", style=style, source=source)
            else:
                write(artefact, folder / f"{key}.md", style=style)
        (folder / "result.json").write_text(
            json.dumps(record.results[name], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def load(root: str | Path = ".") -> Project:
    """The project in folder `root`."""
    return Project.load(root)


# --- init ------------------------------------------------------------------------

_HEADER = {
    "project": "dazflow project. Every setting is written with its default; edit what you need.\n"
    "Each step reads config/<step>.yaml. Run `dazflow validate`, then `dazflow run`.",
    "ingest": "ingest: the inputs and the series taken from them (design §7.1).\n"
    "Paths are relative to the project folder.",
    "clean": "clean: data-quality checks and treatments (design §7.2). Every treatment needs a reason.",
    "explore": "explore: profiles, unit-root tests, lag search and the driver screen (design §7.3).",
    "build": "build: driver combinations and forms, simplest first, within the declared signs\n"
    "(design §7.4). A candidate's check is overridden as '<candidate>/<check>'.",
    "select": "select: rolling-origin scores, the model confidence set, scenarios and the default\n"
    "rule (design §7.6). Confirm or override the proposal in choice, with a reason.\n"
    "data/raw/scenarios.csv holds illustrative driver paths written for the example data,\n"
    "not a published scenario.",
    "diagnose": "diagnose: the chosen model's residual tests, stability, back-test, sensitivity\n"
    "and anomalies (design §7.8). Each test is a threshold check; tests that do not apply to\n"
    "the model's form are listed as such.",
    "forecast": "forecast: the chosen model under every scenario of config/select.yaml, anchored at\n"
    "the origin and back on the target's scale with intervals (design §7.7).",
}


def _example_ingest() -> Config:
    from dazflow.steps import ingest

    return ingest.Config.model_validate(
        {
            "frequency": "Q",
            "inputs": [{"name": "macro", "path": "data/raw/series.csv"}],
            "series": [
                {
                    "name": "loss_rate",
                    "role": "target",
                    "unit": "%",
                    "definition": "Annual loss rate of the portfolio (example data)",
                },
                {
                    "name": "gdp_growth",
                    "unit": "%",
                    "definition": "Real GDP growth (example data)",
                    "expected_sign": -1,
                },
                {
                    "name": "unemployment",
                    "unit": "%",
                    "definition": "Unemployment rate (example data)",
                    "expected_sign": 1,
                },
            ],
        }
    )


def _example_select() -> Config:
    from dazflow.steps import select

    return select.Config.model_validate(
        {
            "scenarios": {
                "path": "data/raw/scenarios.csv",
                "severity": ["baseline", "adverse", "severe"],
            },
            "bounds": [0.0, 100.0],
        }
    )


def _example_diagnose() -> Config:
    from dazflow.steps import diagnose

    return diagnose.Config.model_validate(
        {
            "overrides": [
                {
                    "check": "reset",
                    "reason": "Example data: RESET rejects the linear form at 5% (p = 0.03). "
                    "The series were made up for the example; a real model would revisit "
                    "its functional form before use.",
                }
            ]
        }
    )


_EXAMPLES = {"ingest": _example_ingest, "select": _example_select, "diagnose": _example_diagnose}
_EXAMPLE_DATA = ("series.csv", "scenarios.csv")


def init(root: str | Path, name: str | None = None) -> Project:
    """Create a project in `root` (new or empty) with commented defaults and example data.

    The example runs as it is: ``dazflow validate`` and ``dazflow run`` pass.
    """
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ProjectError(
            f"{root} already exists and is not an empty folder",
            observed=f"{root.resolve()}",
            change="Name a new folder: dazflow init <new-dir>.",
        )
    from dazflow.steps import registry

    config = ProjectConfig(name=name or root.resolve().name)
    for folder in LAYOUT:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / PROJECT_FILE).write_text(
        commented_yaml(config, _HEADER["project"]), encoding="utf-8", newline="\n"
    )
    for step in config.pipeline:
        cfg = _EXAMPLES[step]() if step in _EXAMPLES else registry.get(step).Config()
        (root / "config" / f"{step}.yaml").write_text(
            commented_yaml(cfg, _HEADER.get(step)), encoding="utf-8", newline="\n"
        )
    for file in _EXAMPLE_DATA:
        sample = resources.files("dazflow.project").joinpath("example", file)
        (root / "data" / "raw" / file).write_bytes(sample.read_bytes())
    return Project(root, config)
