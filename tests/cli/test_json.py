"""--json: what the command line prints parses and equals the library's own results.

Oracles: `StepResult.to_dict()` from calling the steps directly
(`dz.ingest.run`, `dz.clean.run`, `dz.explore.run`) on the same project, and
the record file the same command wrote.
"""

import json
import math

import pytest
from typer.testing import CliRunner

import dazflow as dz
from dazflow.cli import app
from dazflow.steps import build, clean, explore, ingest, select

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def close(a, b, path="$"):
    """Equal JSON, floats within 1e-9 relative: arch and statsmodels reruns differ by an ulp."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), path
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def library(root):
    """The steps called directly, the input paths resolved by hand."""
    proj = dz.project.load(root)
    cfg = proj.step_config("ingest")
    inputs = [i.model_copy(update={"path": str(root / i.path)}) for i in cfg.inputs]
    ing = ingest.run(cfg.model_copy(update={"inputs": inputs}))
    cl = clean.run(ing.dataset, proj.step_config("clean"))
    ex = explore.run(cl.dataset, proj.step_config("explore"))
    return cfg, ing, cl, ex


def library_select(root, cl, ex):
    """build and select called directly, the scenarios path resolved by hand."""
    proj = dz.project.load(root)
    b = build.run(cl.dataset, proj.step_config("build"), explore=ex)
    cfg = proj.step_config("select")
    spec = cfg.scenarios.model_copy(update={"path": str(root / cfg.scenarios.path)})
    sel = select.run(cl.dataset, cfg.model_copy(update={"scenarios": spec}), build=b, seed=0)
    return b, sel.model_copy(update={"config": cfg})


def test_step_commands_print_the_libraries_results(example):
    cfg, ing, cl, ex = library(example)

    got = invoke("ingest", "-p", example, "--json")
    want = ing.to_dict()
    # The only difference: the result keeps the path as configured, relative to the folder.
    assert got["config"]["inputs"][0]["path"] == "data/raw/series.csv"
    want["config"] = cfg.model_dump(mode="json")
    assert got == want

    assert invoke("clean", "-p", example, "--json") == cl.to_dict()
    close(invoke("explore", "-p", example, "--json"), ex.to_dict())
    b, sel = library_select(example, cl, ex)
    close(invoke("build", "-p", example, "--json"), b.to_dict())
    close(invoke("select", "-p", example, "--json"), sel.to_dict())


def test_run_json_is_the_record_it_wrote(example):
    got = invoke("run", "-p", example, "--json")
    assert got == json.loads((example / "reports" / "record.json").read_text())
    _, _, cl, ex = library(example)
    assert got["results"]["clean"] == cl.to_dict()
    close(got["results"]["explore"], ex.to_dict())


def test_the_record_holds_the_same_results_as_the_step_folders(ran_example):
    record = json.loads((ran_example / "reports" / "record.json").read_text())
    for step in ("ingest", "clean", "explore", "build", "select", "forecast"):
        folder = json.loads((ran_example / "reports" / step / "result.json").read_text())
        assert folder == record["results"][step]


@pytest.mark.parametrize("artefact", ["T1.1", "F1.1", "S2.1", "T3.3"])
def test_show_json_is_the_artefact_in_the_record(ran_example, artefact):
    record = json.loads((ran_example / "reports" / "record.json").read_text())
    step = {"1": "ingest", "2": "clean", "3": "explore"}[artefact[1]]
    assert (
        invoke("show", artefact, "-p", ran_example, "--json")
        == (record["results"][step]["artefacts"][artefact])
    )


def test_show_prints_a_finding_as_its_text(ran_example):
    record = dz.pipeline.RunRecord.load(ran_example / "reports" / "record.json")
    result = runner.invoke(app, ["show", "S3.1", "-p", str(ran_example)])
    assert result.exit_code == 0
    assert result.stdout.strip() == record.result("explore").artefacts["S3.1"].text


def test_show_prints_a_table_as_markdown(ran_example):
    record = dz.pipeline.RunRecord.load(ran_example / "reports" / "record.json")
    result = runner.invoke(app, ["show", "T3.3", "-p", str(ran_example)])
    assert result.exit_code == 0
    table = record.result("explore").artefacts["T3.3"]
    assert result.stdout.strip() == dz.render.render(table, "markdown").strip()


def test_plugins_json_lists_every_registry():
    from dazflow.steps import registry

    got = invoke("plugins", "--json")
    assert got["step"] == list(registry.keys())
    assert got["reader"] == list(dz.readers.registry.keys())
    assert got["check"] == list(dz.checks.registry.keys())
    assert got["transform"] == list(dz.transforms.registry.keys())
    assert set(got) == {"step", "step adapter", "reader", "check", "transform", "renderer"}


def test_validate_json(example):
    got = invoke("validate", "-p", example, "--json")
    assert got == {
        "valid": True,
        "project": "loss-rate",
        "steps": ["ingest", "clean", "explore", "build", "select", "forecast"],
        "inputs": [
            {"name": "macro", "path": "data/raw/series.csv"},
            {"name": "scenarios", "path": "data/raw/scenarios.csv"},
        ],
    }
