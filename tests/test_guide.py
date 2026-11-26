"""The guide's code runs: every Python block of a page, in order, on the example project.

`user_guide/get-started.qmd` runs in a folder holding a copy of
`examples/loss-rate` (what `dazflow init loss-rate` writes). In
`user_guide/writing-a-plugin.qmd` the code is a contract: a block whose first
line is ``# <file>.py`` is a file the reader writes, the ``toml`` block is the
entry point that installs it, the ``yaml`` block is merged into the
project's configuration, and the contract class it shows must pass.
"""

import importlib
import importlib.metadata
import inspect
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from example_helpers import EXAMPLE

from dazflow import models
from dazflow.core import ENTRY_POINT_GROUP

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "user_guide"
FENCE = re.compile(r"^```(\w*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)
FILE = re.compile(r"^# ([\w/]+\.py)\n")


def blocks(page: str, folder: Path = GUIDE) -> list[tuple[str, str]]:
    """The fenced blocks of a guide page as ``(language, code)``, in page order."""
    return [(m[1], m[2]) for m in FENCE.finditer((folder / page).read_text(encoding="utf-8"))]


def run_python(page: str, code: list[str], namespace: dict) -> None:
    for i, source in enumerate(code, 1):
        exec(compile(source, f"{page}, Python block {i}", "exec"), namespace)


def test_blocks_reads_every_fence_in_order():
    found = blocks("get-started.qmd")
    assert [lang for lang, _ in found][:3] == ["sh", "", "python"]
    assert found[2][1].startswith("import dazflow as dz\n")


def test_get_started_runs_on_the_example_project(tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE, tmp_path / "loss-rate")
    monkeypatch.chdir(tmp_path)
    code = [c for lang, c in blocks("get-started.qmd") if lang == "python"]
    assert len(code) >= 10

    namespace: dict = {}
    run_python("get-started.qmd", code, namespace)

    # The page's last claims, checked here so a block that silently does less fails.
    assert namespace["run"].passed
    assert namespace["replay"].passed
    assert (tmp_path / "loss-rate" / "reports" / "document" / "document.docx").is_file()


def test_readme_python_runs_on_the_example_project(tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE, tmp_path / "loss-rate")
    monkeypatch.chdir(tmp_path)
    (code,) = [c for lang, c in blocks("README.md", ROOT) if lang == "python"]
    namespace: dict = {}
    run_python("README.md", [code], namespace)
    assert namespace["record"].run.status == "passed"
    assert namespace["project"].replay().passed


@pytest.fixture
def plugin_site(tmp_path, monkeypatch):
    """A directory on sys.path for the plugin, and a models registry restored afterwards."""
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(models.registry, "_items", dict(models.registry._items))
    monkeypatch.setattr(models.registry, "_plugins_loaded", False)
    before = set(sys.modules)
    yield site
    for name in set(sys.modules) - before:
        if not name.startswith(("dazflow", "_pytest")):
            del sys.modules[name]
    importlib.invalidate_caches()


def test_writing_a_plugin_registers_through_the_entry_point_and_passes_the_contract(
    plugin_site, tmp_path, monkeypatch
):
    page = "writing-a-plugin.qmd"
    found = blocks(page)
    files = {FILE.match(c)[1]: c for lang, c in found if lang == "python" and FILE.match(c)}
    code = [c for lang, c in found if lang == "python" and not FILE.match(c)]
    (toml,) = [c for lang, c in found if lang == "toml"]
    (config,) = [c for lang, c in found if lang == "yaml"]
    assert set(files) == {"dazflow_drift.py", "tests/test_drift.py"}

    # Install the plugin as a distribution with the page's entry point.
    (plugin_site / "dazflow_drift.py").write_text(files["dazflow_drift.py"], encoding="utf-8")
    meta = tomllib.loads(toml)
    entry_points = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    assert entry_points == {"drift": "dazflow_drift"}
    info = plugin_site / "dazflow_drift-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {meta['project']['name']}\nVersion: 0.1\n", encoding="utf-8"
    )
    lines = "".join(f"{k} = {v}\n" for k, v in entry_points.items())
    (info / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\n{lines}", encoding="utf-8")
    importlib.invalidate_caches()
    assert "dazflow_drift" not in sys.modules

    # The page's configuration, merged into the example project's.
    shutil.copytree(EXAMPLE, tmp_path / "loss-rate")
    build = tmp_path / "loss-rate" / "config" / "build.yaml"
    settings = yaml.safe_load(build.read_text(encoding="utf-8"))
    settings.update(yaml.safe_load(config))
    build.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    namespace: dict = {}
    run_python(page, code, namespace)

    # The form came in through the entry point: nothing on the page imports it.
    assert models.registry["drift"].__module__ == "dazflow_drift"
    assert "drift" in [c.id for c in namespace["run"].results["build"].candidates]

    # The page's contract class passes, test by test.
    contract: dict = {}
    exec(compile(files["tests/test_drift.py"], f"{page}, tests/test_drift.py", "exec"), contract)
    (cls,) = [v for k, v in contract.items() if k.startswith("Test") and inspect.isclass(v)]
    assert issubclass(cls, importlib.import_module("dazflow.testing").ModelContract)
    tests = [name for name in dir(cls) if name.startswith("test_")]
    assert len(tests) >= 10
    for name in tests:
        getattr(cls(), name)()
