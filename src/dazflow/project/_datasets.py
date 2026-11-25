"""A dataset on disk: a CSV of its series and a JSON file declaring its schema (design §10.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dazflow.core.dataset import Dataset, LogEntry, SeriesMeta
from dazflow.core.errors import ProjectError, RecordError
from dazflow.readers._base import frequency_of

__all__ = ["DATASET_FORMAT", "read_dataset", "schema_path", "write_dataset"]

DATASET_FORMAT = "dazflow.dataset-csv/1"


def schema_path(path: str | Path) -> Path:
    """The schema file next to a dataset CSV: ``ingest.csv`` -> ``ingest.schema.json``."""
    path = Path(path)
    return path.with_name(f"{path.stem}.schema.json")


def write_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Write `dataset` as ``<path>`` (CSV, first column ``period``) and its schema file.

    The CSV is readable by the csv reader as it is; the schema keeps the
    series metadata, the log and the fingerprint, which `read_dataset` checks.
    Panels and scenarios are not written yet.
    """
    if dataset.is_panel or dataset.scenarios is not None:
        raise ProjectError(
            "only a single-segment dataset without scenarios is written to the project folder",
            observed="a panel" if dataset.is_panel else "a dataset with scenarios",
            change="Keep the dataset in memory; panels and scenarios are stored after 0.1.0.",
        )
    path = Path(path)
    frame = dataset.frame
    out = frame.copy()
    out.index = pd.Index([str(p) for p in frame.index], name="period")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, lineterminator="\n", encoding="utf-8")
    schema = {
        "format": DATASET_FORMAT,
        "fingerprint": dataset.fingerprint(),
        "frequency": frequency_of(frame.index),
        "columns": [{"name": c, "dtype": str(frame[c].dtype)} for c in frame.columns],
        "series": {c: m.model_dump(mode="json") for c, m in dataset.series.items()},
        "log": [e.model_dump(mode="json") for e in dataset.log],
    }
    schema_path(path).write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _parse(values: pd.Series, dtype: str) -> pd.Series:
    if dtype.startswith("float"):
        return pd.Series([float(v) if v != "" else np.nan for v in values], dtype=dtype)
    if dtype.startswith(("int", "Int")):
        return pd.Series([int(v) for v in values], dtype=dtype)
    if dtype == "bool":
        return pd.Series([{"True": True, "False": False}[v] for v in values], dtype=bool)
    return pd.Series([v if v != "" else None for v in values], dtype=object)


def read_dataset(path: str | Path, fingerprint: str | None = None) -> Dataset:
    """The dataset `write_dataset` wrote to `path`, checked against its fingerprint.

    `fingerprint`, when given, must also match: the one the run record holds.
    """
    path = Path(path)
    spath = schema_path(path)
    try:
        schema = json.loads(spath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise RecordError(
            "the dataset's schema file cannot be read",
            observed=f"{spath}: {err}",
            change="Run the step that writes it again.",
        ) from err
    if schema.get("format") != DATASET_FORMAT:
        raise RecordError(
            f"the dataset's schema is not {DATASET_FORMAT}",
            observed=f"{spath}: format {schema.get('format')!r}",
            change="Run the step that writes it again.",
        )
    dtypes = {c["name"]: c["dtype"] for c in schema["columns"]}
    try:
        # Read as text and parse each value with Python: float() inverts repr() exactly,
        # on every pandas version, so the fingerprint survives the round trip.
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        index = pd.PeriodIndex([pd.Period(p, freq=schema["frequency"]) for p in raw.iloc[:, 0]])
        frame = pd.DataFrame(
            {c: _parse(raw[c], d).to_numpy() for c, d in dtypes.items()}, index=index
        )
    except (OSError, ValueError, KeyError) as err:
        raise RecordError(
            "the dataset CSV cannot be read as its schema declares",
            observed=f"{path}: {type(err).__name__}: {err}",
            change="Run the step that writes it again.",
        ) from err
    ds = Dataset(
        frame,
        {c: SeriesMeta.model_validate(m) for c, m in schema["series"].items()},
        log=[LogEntry.model_validate(e) for e in schema["log"]],
    )
    got = ds.fingerprint()
    for want in (schema["fingerprint"], fingerprint):
        if want is not None and got != want:
            raise RecordError(
                "the dataset on disk does not match its recorded fingerprint",
                observed=f"{path}: fingerprint {got[:12]}..., recorded {want[:12]}...",
                change="Do not edit files under data/interim or data/processed; run the step again.",
            )
    return ds
