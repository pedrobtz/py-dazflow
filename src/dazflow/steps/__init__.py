"""The workflow steps (design §7). Each is a module with a `Config`, a `Result` and `run()`.

Steps are registered by name in `registry`, as any component is, so a
pipeline (Stage 6) places them by the name configuration gives.
"""

from typing import Any, Protocol

from dazflow.core.registry import Registry

__all__ = [
    "Step",
    "build",
    "clean",
    "diagnose",
    "explore",
    "forecast",
    "ingest",
    "registry",
    "report",
    "select",
]


class Step(Protocol):
    """A step module: its configuration, its result type and the function that runs it."""

    Config: type
    Result: type

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the step; it prints nothing and writes no files."""


registry = Registry("step", protocol=Step)

from dazflow.steps import build, clean, diagnose, explore, forecast, ingest, report, select  # noqa: E402 - the registry exists before the steps register

# The built-in steps in pipeline order; the CLI keeps their commands when a plugin is broken.
_BUILT_IN = {
    "ingest": ingest,
    "clean": clean,
    "explore": explore,
    "build": build,
    "select": select,
    "forecast": forecast,
    "diagnose": diagnose,
    "report": report,
}
for _key, _module in _BUILT_IN.items():
    registry.add(_key, _module)
