"""init: the project it generates passes validate unchanged, runs, and is the example project.

Oracles: `dazflow validate` itself, the folder's bytes before and after,
the layout of design §10, and the committed `examples/loss-rate`.
"""

import yaml
from example_helpers import EXAMPLE, tree
from typer.testing import CliRunner

from dazflow.cli import app
from dazflow.steps import registry

runner = CliRunner()


def test_init_then_validate_passes_and_changes_nothing(tmp_path):
    root = tmp_path / "model"
    assert runner.invoke(app, ["init", str(root)]).exit_code == 0
    before = tree(root)
    result = runner.invoke(app, ["validate", "-p", str(root)])
    assert result.exit_code == 0, result.output
    assert tree(root) == before


def test_init_creates_the_design_layout(tmp_path):
    root = tmp_path / "model"
    assert runner.invoke(app, ["init", str(root)]).exit_code == 0
    for folder in (
        "config",
        "data/raw",
        "data/interim",
        "data/processed",
        "models/work",
        "reports",
    ):
        assert (root / folder).is_dir(), folder
    assert sorted(tree(root)) == [
        "config/clean.yaml",
        "config/explore.yaml",
        "config/ingest.yaml",
        "data/raw/series.csv",
        "project.yaml",
    ]


def test_init_writes_every_default_with_comments(tmp_path):
    root = tmp_path / "model"
    runner.invoke(app, ["init", str(root)])
    for step in ("clean", "explore"):
        text = (root / "config" / f"{step}.yaml").read_text()
        assert yaml.safe_load(text) == registry.get(step).Config().model_dump(mode="json")
        assert text.startswith(f"# {step}:")
    assert "# The calendar every series is put on" in (root / "config" / "ingest.yaml").read_text()


def test_the_example_project_is_what_init_generates(tmp_path):
    root = tmp_path / EXAMPLE.name
    assert runner.invoke(app, ["init", str(root)]).exit_code == 0
    assert tree(root) == tree(EXAMPLE)


def test_init_then_run_passes(tmp_path):
    root = tmp_path / "model"
    runner.invoke(app, ["init", str(root)])
    result = runner.invoke(app, ["run", "-p", str(root)])
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "record.json").is_file()


def test_init_names_the_project_after_its_folder_or_the_option(tmp_path):
    runner.invoke(app, ["init", str(tmp_path / "a")])
    runner.invoke(app, ["init", str(tmp_path / "b"), "--name", "Loss model"])
    assert yaml.safe_load((tmp_path / "a" / "project.yaml").read_text())["name"] == "a"
    assert yaml.safe_load((tmp_path / "b" / "project.yaml").read_text())["name"] == "Loss model"


def test_init_into_an_empty_folder_is_allowed(tmp_path):
    (tmp_path / "empty").mkdir()
    assert runner.invoke(app, ["init", str(tmp_path / "empty")]).exit_code == 0
