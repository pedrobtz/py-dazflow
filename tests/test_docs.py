"""The site's API reference (great-docs.yml) lists every public name, and only real ones.

The oracle is each public module's ``__all__``. A name is covered when the
reference lists it, or lists the same class or function under another name
(``ingest.Config`` is ``ingest.IngestConfig``; ``select.ScenarioInput`` is
``readers.ScenarioInput``). Modules themselves are not entries: their
members are.
"""

import importlib
import importlib.metadata
import inspect
import types
from pathlib import Path

import pytest
import yaml

import dazflow

ROOT = Path(__file__).resolve().parents[1]

PUBLIC = [
    "dazflow",
    "dazflow.core",
    "dazflow.readers",
    "dazflow.checks",
    "dazflow.transforms",
    "dazflow.models",
    "dazflow.evaluation",
    "dazflow.render",
    "dazflow.steps",
    *(f"dazflow.steps.{s}" for s in importlib.import_module("dazflow.steps").registry),
    "dazflow.pipeline",
    "dazflow.project",
    "dazflow.testing",
]


def reference() -> list[str]:
    config = yaml.safe_load((ROOT / "great-docs.yml").read_text(encoding="utf-8"))
    return [name for group in config["reference"] for name in group["contents"]]


def resolve(name: str):
    """The object a reference entry names, relative to the package (as great-docs reads it)."""
    obj, path = dazflow, "dazflow"
    for part in name.split("."):
        path = f"{path}.{part}"
        if not hasattr(obj, part) and isinstance(obj, types.ModuleType):
            importlib.import_module(path)  # a submodule `import dazflow` does not load
        obj = getattr(obj, part)
    return obj


def public_names() -> list[tuple[str, str, object]]:
    found = []
    for modname in PUBLIC:
        mod = importlib.import_module(modname)
        for name in mod.__all__:
            obj = getattr(mod, name)
            if not isinstance(obj, types.ModuleType) and name != "__version__":
                found.append((modname, name, obj))
    return found


def covered(name: str, obj: object, entries: dict[str, object]) -> bool:
    if inspect.isclass(obj) or inspect.isfunction(obj):
        return any(o is obj for o in entries.values())
    # A constant or an instance (a registry): the same object under the same name.
    return any(e.rsplit(".", 1)[-1] == name and o is obj for e, o in entries.items())


def test_public_modules_are_every_step():
    steps = [m for m in PUBLIC if m.startswith("dazflow.steps.")]
    assert [m.rsplit(".", 1)[1] for m in steps] == [
        "build", "clean", "diagnose", "explore", "forecast", "ingest", "report", "select"
    ]  # fmt: skip


def test_every_reference_entry_is_listed_once_and_resolves():
    names = reference()
    assert len(names) == len(set(names)), "a name is listed twice"
    for name in names:
        resolve(name)  # AttributeError names the stale entry


@pytest.mark.parametrize("modname", PUBLIC)
def test_every_public_name_is_in_a_reference_group(modname):
    entries = {name: resolve(name) for name in reference()}
    missing = [
        f"{m}.{n}" for m, n, obj in public_names() if m == modname and not covered(n, obj, entries)
    ]
    assert not missing, f"not in great-docs.yml: {', '.join(missing)}"


def test_the_reference_has_a_group_per_layer():
    config = yaml.safe_load((ROOT / "great-docs.yml").read_text(encoding="utf-8"))
    titles = [g["title"] for g in config["reference"]]
    for layer in ["Core", "Readers", "Model forms", "Evaluation", "Rendering", "Step: ingest"]:
        assert any(t.startswith(layer) for t in titles), layer
    assert titles[-3:] == ["Pipeline, record and replay", "Project folder", "Testing"]


def test_the_documentation_url_is_the_site_and_the_readme_links_it():
    site = yaml.safe_load((ROOT / "great-docs.yml").read_text(encoding="utf-8"))["site_url"]
    urls = dict(
        u.split(", ", 1) for u in importlib.metadata.metadata("dazflow").get_all("Project-URL")
    )
    assert urls["Documentation"] == site == "https://pedrobtz.github.io/py-dazflow/"
    assert f"<{site}>" in (ROOT / "README.md").read_text(encoding="utf-8")
