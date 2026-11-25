"""Error classes.

Every error `dazflow` raises is a `DazflowError`. Its message has three parts,
in this order: what went wrong, the state that was observed, and what to
change::

    the frame index is not a PeriodIndex
      observed: Index of int64
      change: Index the frame by pandas periods (pd.PeriodIndex) ...
"""

from __future__ import annotations

__all__ = [
    "ArtefactError",
    "BuildError",
    "CheckError",
    "CleanError",
    "ConfigError",
    "DatasetError",
    "DazflowError",
    "DuplicateKeyError",
    "EvaluationError",
    "ExploreError",
    "FindingError",
    "GateError",
    "IngestError",
    "MissingExtraError",
    "ModelError",
    "PipelineError",
    "PluginError",
    "ProjectError",
    "ReaderError",
    "RecordError",
    "RegistryError",
    "RenderError",
    "ResultError",
    "SelectError",
    "TransformError",
    "UnknownKeyError",
]


class DazflowError(Exception):
    """Base class of every error raised by dazflow.

    Parameters
    ----------
    problem
        What went wrong, as a lower-case clause.
    observed
        The state that was found: the value, the shape, the keys present.
    change
        What the caller should change, as an instruction.
    """

    def __init__(self, problem: str, *, observed: str, change: str) -> None:
        self.problem = problem
        self.observed = observed
        self.change = change
        super().__init__(f"{problem}\n  observed: {observed}\n  change: {change}")


class ConfigError(DazflowError):
    """A configuration could not be read, parsed or validated."""


class DatasetError(DazflowError):
    """A `Dataset` or its metadata is not valid."""


class GateError(DazflowError):
    """A `Check`, `Gate` or `Override` is not valid, such as an override without a reason."""


class FindingError(DazflowError):
    """A `Finding` template could not be filled with its values."""


class ArtefactError(DazflowError):
    """A `Table`, `Figure`, `Source`, `Section`, `Document` or artefact identifier is not valid."""


class RenderError(DazflowError):
    """An artefact could not be rendered in the format asked for."""


class TransformError(DazflowError):
    """A transform could not be applied or inverted, such as the log of a value that is not positive.

    `series` and `period` name the first value outside the transform's
    domain, when there is one.
    """

    def __init__(
        self,
        problem: str,
        *,
        observed: str,
        change: str,
        series: str | None = None,
        period: str | None = None,
    ) -> None:
        self.series = series
        self.period = period
        super().__init__(problem, observed=observed, change=change)


class ReaderError(DazflowError):
    """Data could not be read, or its dates could not be put on a calendar."""


class MissingExtraError(ReaderError):
    """An optional package is not installed; `extra` names the ``dazflow`` extra that has it."""

    def __init__(self, what: str, module: str, extra: str) -> None:
        self.module = module
        self.extra = extra
        super().__init__(
            f"{what} needs {module}, which is not installed",
            observed=f"import {module} failed; it is in the dazflow[{extra}] extra",
            change=f"Install the extra: pip install 'dazflow[{extra}]' (or uv add 'dazflow[{extra}]').",
        )


class IngestError(DazflowError):
    """The ingest step cannot put its inputs on one calendar as configured."""


class CheckError(DazflowError):
    """A data-quality check cannot be evaluated as configured, such as a downturn that cannot be read."""


class CleanError(DazflowError):
    """The clean step cannot treat the data as configured, such as a treatment naming an unknown period."""


class ExploreError(DazflowError):
    """The explore step cannot profile, test or screen the series as configured."""


class ModelError(DazflowError):
    """A model form cannot be fitted, predicted or refitted as asked, such as missing values or drivers."""


class EvaluationError(DazflowError):
    """A splitter, metric or rolling-origin evaluation cannot run as configured."""


class BuildError(DazflowError):
    """Candidate models cannot be built as configured (unknown drivers or forms, no data)."""


class SelectError(DazflowError):
    """Candidates cannot be ranked or chosen as configured (benchmark, choice or scenarios)."""


class PipelineError(DazflowError):
    """A pipeline cannot be built or run as configured, such as an unknown step or order."""


class ProjectError(DazflowError):
    """A project folder cannot be created, read or written, or its run record is missing."""


class RecordError(DazflowError):
    """A run record cannot be read, is of another schema version, or does not match its data."""


class ResultError(DazflowError):
    """A `StepResult` is not valid or could not be rebuilt from its dictionary."""


class RegistryError(DazflowError):
    """A component could not be registered or looked up."""


class UnknownKeyError(RegistryError):
    """No component is registered under the key; `known` lists the keys that are."""

    def __init__(self, kind: str, key: object, known: tuple[str, ...]) -> None:
        self.kind = kind
        self.key = key
        self.known = known
        super().__init__(
            f"no {kind} is registered as {key!r}",
            observed=f"known {kind} keys: {', '.join(known) if known else 'none'}",
            change=(
                "Use one of the known keys, or register the component with "
                f"@<registry>.register({key!r}) or a 'dazflow.plugins' entry point."
            ),
        )


class DuplicateKeyError(RegistryError):
    """A key is already taken in a registry."""


class PluginError(RegistryError):
    """A `dazflow.plugins` entry point failed to load."""
