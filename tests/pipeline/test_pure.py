"""Pure steps: running steps through the library writes no file (design §4).

Oracle: the operating system. The project folder and the working
directory are made read-only, so any write raises, and their bytes are
compared before and after.
"""

import os
import stat

import pytest
from example_helpers import tree

import dazflow as dz
from dazflow.pipeline import Pipeline


@pytest.fixture
def frozen(example, tmp_path, monkeypatch):
    """The example project, read-only, as the working directory."""
    folders = [p for p in [example, *example.rglob("*")] if p.is_dir()]
    files = [p for p in example.rglob("*") if p.is_file()]
    for f in files:
        f.chmod(stat.S_IRUSR)
    for d in folders:
        d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.chdir(example)
    yield example
    for d in folders:
        d.chmod(stat.S_IRWXU)
    for f in files:
        f.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.skipif(os.name == "nt", reason="Windows ignores read-only bits on folders")
def test_the_folder_is_really_read_only(frozen):
    with pytest.raises(PermissionError):
        (frozen / "probe.txt").write_text("x")


def test_each_step_called_directly_writes_nothing(frozen):
    before = tree(frozen)
    proj = dz.project.load(".")
    cfg = proj.step_config("ingest")
    ing = dz.ingest.run(cfg)  # the path is relative to the working directory, the project
    cl = dz.clean.run(ing.dataset, proj.step_config("clean"))
    ex = dz.explore.run(cl.dataset, proj.step_config("explore"))
    b = dz.build.run(cl.dataset, proj.step_config("build"), explore=ex)
    sel = dz.select.run(cl.dataset, proj.step_config("select"), build=b)  # reads the scenarios
    assert ex.gate.passed and b.gate.passed and sel.gate.passed
    for result in (ing, cl, ex, b, sel):
        result.to_dict()
    assert tree(frozen) == before


def test_a_pipeline_run_writes_nothing(frozen):
    before = tree(frozen)
    proj = dz.project.load(frozen)
    run = Pipeline().run(proj.configs(), root=frozen)
    assert run.passed
    run.record(proj.config)
    assert tree(frozen) == before


def test_a_project_run_writes_nothing(frozen):
    before = tree(frozen)
    run = dz.project.load(frozen).run()
    assert run.steps == (
        "ingest",
        "clean",
        "explore",
        "build",
        "select",
        "forecast",
        "diagnose",
        "report",
    )
    assert tree(frozen) == before


def test_validate_writes_nothing(frozen):
    before = tree(frozen)
    dz.project.load(frozen).validate()
    assert tree(frozen) == before
