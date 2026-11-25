"""dazflow: a documented time-series modelling workflow."""

from importlib.metadata import PackageNotFoundError, version

try:  # one source: the version in pyproject.toml, read from the installed metadata
    __version__ = version("dazflow")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
