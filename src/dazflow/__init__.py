"""dazflow: a documented time-series modelling workflow."""

from importlib.metadata import PackageNotFoundError, version

from dazflow import checks, core, evaluation, models, pipeline, project, readers, transforms
from dazflow.core import *  # noqa: F403 - the core objects are the top-level API
from dazflow.core import __all__ as _core_all
from dazflow.steps import clean, explore, ingest

try:  # one source: the version in pyproject.toml, read from the installed metadata
    __version__ = version("dazflow")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "checks",
    "clean",
    "core",
    "evaluation",
    "explore",
    "ingest",
    "models",
    "pipeline",
    "project",
    "readers",
    "transforms",
    *_core_all,
]


def __getattr__(name: str):
    # `dz.render` imports matplotlib, great-tables and python-docx, so it loads
    # on first use rather than with `import dazflow`.
    if name == "render":
        import importlib

        return importlib.import_module("dazflow.render")
    raise AttributeError(f"module 'dazflow' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "render"])
