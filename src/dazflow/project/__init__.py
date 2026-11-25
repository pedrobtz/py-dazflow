"""The project folder (design §10): `init`, `load`, `Project.run` and `Project.write`.

``Project.run()`` reads the folder and returns a `dazflow.pipeline.Run`
without writing; ``Project.write(run)`` writes its datasets to
``data/interim/`` and ``data/processed/``, each step's artefacts to
``reports/<step>/`` and the record to ``reports/record.json``.
"""

from dazflow.project._commented import commented_yaml
from dazflow.project._datasets import DATASET_FORMAT, read_dataset, schema_path, write_dataset
from dazflow.project._folder import (
    PROJECT_FILE,
    RECORD_PATH,
    Project,
    ProjectConfig,
    init,
    load,
)

__all__ = [
    "DATASET_FORMAT",
    "PROJECT_FILE",
    "RECORD_PATH",
    "Project",
    "ProjectConfig",
    "commented_yaml",
    "init",
    "load",
    "read_dataset",
    "schema_path",
    "write_dataset",
]
