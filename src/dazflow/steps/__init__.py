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
    "select",
]


class Step(Protocol):
    """A step module: its configuration, its result type and the function that runs it."""

    Config: type
    Result: type

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the step; it prints nothing and writes no files."""


registry = Registry("step", protocol=Step)

from dazflow.steps import build, clean, diagnose, explore, forecast, ingest, select  # noqa: E402 - the registry exists before the steps register

registry.add("ingest", ingest)
registry.add("clean", clean)
registry.add("explore", explore)
registry.add("build", build)
registry.add("select", select)
registry.add("forecast", forecast)
registry.add("diagnose", diagnose)
