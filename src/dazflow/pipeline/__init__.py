"""The pipeline layer: steps in configured order, the run and its record (design §4, §10.1).

>>> from dazflow import pipeline
>>> pipeline.Pipeline().steps
('ingest', 'clean', 'explore', 'build', 'select', 'forecast', 'diagnose', 'report')

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
from dazflow.pipeline._replay import (
    ATOL,
    RTOL,
    Difference,
    Replay,
    compare,
    compare_records,
    replay,
)

__all__ = [
    "ATOL",
    "DEFAULT_STEPS",
    "RECORD_KIND",
    "RTOL",
    "SCHEMA_VERSION",
    "Adapter",
    "DatasetRecord",
    "Difference",
    "Environment",
    "InputRecord",
    "Pipeline",
    "Replay",
    "Run",
    "RunInfo",
    "RunRecord",
    "StepInputs",
    "adapters",
    "compare",
    "compare_records",
    "default_adapter",
    "environment",
    "file_sha256",
    "register_adapter",
    "replay",
]
