"""dazflow: a documented time-series modelling workflow."""

from importlib.metadata import PackageNotFoundError, version

from dazflow import core
from dazflow.core import *  # noqa: F403 - the core objects are the top-level API
from dazflow.core import __all__ as _core_all

try:  # one source: the version in pyproject.toml, read from the installed metadata
    __version__ = version("dazflow")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = ["__version__", "core", *_core_all]
