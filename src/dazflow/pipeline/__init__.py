"""The pipeline layer: steps in configured order, the run and its record (design §4, §10.1).

>>> from dazflow import pipeline
>>> pipeline.Pipeline().steps
('ingest', 'clean', 'explore', 'build', 'select', 'forecast')

A step joins a pipeline by name once it is in `dazflow.steps.registry`. The
pipeline calls it through its adapter in `adapters` (``@register_adapter(name)``),
which turns the earlier results in `StepInputs` into the step's own `run()`
arguments; a step without one is called as ``run(dataset, config)``.
"""

from dazflow.pipeline._adapters import (
    Adapter,
    InputRecord,
    StepInputs,
    adapters,
    default_adapter,
    file_sha256,
    register_adapter,
)
from dazflow.pipeline._pipeline import DEFAULT_STEPS, Pipeline, Run
from dazflow.pipeline._record import (
    RECORD_KIND,
    SCHEMA_VERSION,
    DatasetRecord,
    Environment,
    RunInfo,
    RunRecord,
    environment,
)

__all__ = [
    "DEFAULT_STEPS",
    "RECORD_KIND",
    "SCHEMA_VERSION",
    "Adapter",
    "DatasetRecord",
    "Environment",
    "InputRecord",
    "Pipeline",
    "Run",
    "RunInfo",
    "RunRecord",
    "StepInputs",
    "adapters",
    "default_adapter",
    "environment",
    "file_sha256",
    "register_adapter",
]
