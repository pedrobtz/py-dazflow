"""The command line (design §11): ``dazflow init | validate | <step> | run | show | plugins``.

Every command calls the library: `dazflow.project` reads and writes the
folder, `dazflow.pipeline` runs the steps. Exit codes: 0 success, 1 error
(any `DazflowError`), 2 a step completed with a failed gate. ``--json``
prints the result as JSON on standard output.

A step registered in `dazflow.steps.registry` (built in, or by a plugin)
gets a command of the same name.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from dazflow.core.errors import DazflowError, PluginError, ProjectError

__all__ = ["EXIT_ERROR", "EXIT_GATE", "EXIT_OK", "app", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GATE = 2
# Command names that a step cannot take.
RESERVED = frozenset({"init", "validate", "run", "show", "plugins", "replay", "model"})

ProjectOpt = Annotated[
    Path,
    typer.Option("--project", "-p", help="The project folder.", file_okay=False),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print the result as JSON.")]


def _echo_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def _guarded(fn: Callable[..., int | None]) -> Callable[..., None]:
    """Run a command; a `DazflowError` is printed to standard error and exits 1."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            code = fn(*args, **kwargs)
        except DazflowError as err:
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(EXIT_ERROR) from None
        raise typer.Exit(code or EXIT_OK)

    return wrapper


def _version(value: bool) -> None:
    if value:
        from dazflow import __version__

        typer.echo(f"dazflow {__version__}")
        raise typer.Exit(EXIT_OK)


app = typer.Typer(
    name="dazflow",
    help="A time-series modelling workflow whose every step returns documented artefacts.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
) -> None:
    """dazflow: ingest, clean, explore and the later steps, from a project folder."""


def _summary(record: Any, steps: tuple[str, ...]) -> None:
    for name in steps:
        result = record.results[name]
        checks = result["gate"]["checks"]
        failed = [c["name"] for c in checks if c["status"] == "fail"]
        state = "passed" if not failed else f"FAILED ({', '.join(failed)})"
        typer.echo(f"{name}: gate {state}; {len(result['artefacts'])} artefacts in reports/{name}/")
        finding = result.get("finding")
        if finding:
            typer.echo(f"  {finding['text']}")
    if record.run.stopped_at is not None:
        typer.echo(f"stopped after {record.run.stopped_at}: its gate failed", err=True)


def _run(
    project_dir: Path, start: str | None, stop: str | None, json_out: bool, *, step: bool
) -> int:
    from dazflow.project import Project

    project = Project.load(project_dir)
    run = project.run(start, stop)
    record = project.write(run)
    if json_out:
        _echo_json(record.results[start] if step else record.to_dict())
    else:
        _summary(record, run.steps)
    return EXIT_OK if run.passed else EXIT_GATE


@app.command()
@_guarded
def init(
    directory: Annotated[Path, typer.Argument(help="The new project folder.")],
    name: Annotated[str | None, typer.Option(help="Project name; default the folder name.")] = None,
    json_out: JsonOpt = False,
) -> int:
    """Create a project folder with commented defaults and example data."""
    from dazflow.project import init as make

    project = make(directory, name)
    files = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    if json_out:
        _echo_json({"project": project.config.name, "path": str(directory), "files": files})
    else:
        typer.echo(
            f"created {directory} ({len(files)} files); next: dazflow validate -p {directory}"
        )
    return EXIT_OK


@app.command()
@_guarded
def validate(project_dir: ProjectOpt = Path("."), json_out: JsonOpt = False) -> int:
    """Check the configuration and that every input reads, without running a step."""
    from dazflow.project import Project

    summary = Project.load(project_dir).validate()
    if json_out:
        _echo_json({"valid": True, **summary})
    else:
        typer.echo(
            f"{summary['project']}: valid; steps {', '.join(summary['steps'])}; "
            f"{len(summary['inputs'])} inputs read"
        )
    return EXIT_OK


@app.command("run")
@_guarded
def run_(
    project_dir: ProjectOpt = Path("."),
    start: Annotated[str | None, typer.Option("--from", help="First step to run.")] = None,
    stop: Annotated[str | None, typer.Option("--to", help="Last step to run.")] = None,
    json_out: JsonOpt = False,
) -> int:
    """Run the pipeline's steps in order; stop after a step whose gate fails (exit 2)."""
    return _run(project_dir, start, stop, json_out, step=False)


@app.command()
@_guarded
def show(
    artefact_id: Annotated[str, typer.Argument(help="An artefact identifier: T1.1, F3.1, S2.1.")],
    project_dir: ProjectOpt = Path("."),
    json_out: JsonOpt = False,
) -> int:
    """Print one table or finding of the last run (a figure: its file)."""
    from dazflow.core.artefact import Figure, Table
    from dazflow.project import Project

    project = Project.load(project_dir)
    record = project.record()
    if record is None:
        raise ProjectError(
            "the project has not run yet",
            observed=f"no reports/record.json in {project.root}",
            change="Run it first: dazflow run.",
        )
    for step, data in record.results.items():
        if artefact_id in data["artefacts"]:
            if json_out:
                _echo_json(data["artefacts"][artefact_id])
                return EXIT_OK
            obj = record.result(step).artefacts[artefact_id]
            if isinstance(obj, Table):
                from dazflow.render import render

                typer.echo(render(obj, "markdown", style=project.style_path()))
            elif isinstance(obj, Figure):
                path = project.root / "reports" / step / f"{artefact_id}.svg"
                typer.echo(f"{artefact_id}: {obj.caption or ''}\n{path}")
            else:
                typer.echo(obj.text)
            return EXIT_OK
    known = [k for d in record.results.values() for k in d["artefacts"]]
    raise ProjectError(
        f"the last run has no artefact {artefact_id!r}",
        observed=f"artefacts: {', '.join(known) or 'none'}",
        change="Name one of the identifiers listed.",
    )


def _registries() -> dict[str, Any]:
    from dazflow import checks, pipeline, readers, transforms
    from dazflow.render import renderers
    from dazflow.steps import registry as steps

    return {
        "step": steps,
        "step adapter": pipeline.adapters,
        "reader": readers.registry,
        "check": checks.registry,
        "transform": transforms.registry,
        "renderer": renderers,
    }


@app.command()
@_guarded
def plugins(json_out: JsonOpt = False) -> int:
    """List the registered components of every kind, built in and from plugins."""
    found = {kind: list(reg.keys()) for kind, reg in _registries().items()}
    if json_out:
        _echo_json(found)
    else:
        for kind, keys in found.items():
            typer.echo(f"{kind}: {', '.join(keys)}")
    return EXIT_OK


def _step_command(name: str) -> Callable[..., None]:
    @_guarded
    def command(project_dir: ProjectOpt = Path("."), json_out: JsonOpt = False) -> int:
        return _run(project_dir, name, name, json_out, step=True)

    command.__name__ = name
    command.__doc__ = (
        f"Run the {name} step on the project, from the last run's earlier results; "
        f"write reports/{name}/."
    )
    return command


def _step_names() -> tuple[str, ...]:
    from dazflow.steps import registry

    try:
        return registry.keys()
    except PluginError as err:  # a broken plugin: the built-in commands still work
        print(f"warning: {err}", file=sys.stderr)
        return ("ingest", "clean", "explore")


for _name in _step_names():
    if _name not in RESERVED:
        app.command(_name)(_step_command(_name))


def main() -> None:
    """The ``dazflow`` console script."""
    app()
