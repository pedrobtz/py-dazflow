"""Scenario paths read from a CSV: one row per scenario and period, one column per driver.

Select and forecast both read them (design §7.6, §7.7), from one setting,
`config/select.yaml: scenarios` (D33, D40), so the reader is a component
rather than a part of either step.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from dazflow.core.config import Config
from dazflow.core.errors import ReaderError
from dazflow.readers._base import Frequency, from_frame

__all__ = ["ScenarioInput", "read_scenarios"]


class ScenarioInput(Config):
    """A CSV of driver paths in levels: a scenario label column, dates and one column per driver."""

    path: str = Field(min_length=1, description="The file, relative to the project folder.")
    scenario: str = Field("scenario", description="The column holding the scenario labels.")
    date: str | None = Field(
        None, description="The column holding the dates; unset: the first other column."
    )
    frequency: Frequency | None = Field(None, description="Unset: read from the dates.")
    severity: list[str] | None = Field(
        None, description="Scenario labels, mildest first; unset: their order in the file."
    )


def read_scenarios(
    spec: ScenarioInput, root: str | Path | None = None
) -> tuple[pd.DataFrame, list]:
    """The scenario frame, indexed by ``(scenario, period)``, and the severity order, from a CSV.

    A relative path is read from `root` (the working directory by default).
    A missing file, label column or row is a `ReaderError`.
    """
    path = Path(spec.path)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise ReaderError(
            f"the scenarios file {spec.path!r} does not exist",
            observed=f"looked for {path}",
            change="Give the path of a CSV of scenarios, relative to the project folder.",
        )
    raw = pd.read_csv(path, dtype={spec.scenario: str})
    if spec.scenario not in raw.columns:
        raise ReaderError(
            f"the scenarios file has no column {spec.scenario!r}",
            observed=f"columns: {', '.join(map(str, raw.columns))}",
            change="Name the column holding the scenario labels in scenarios.scenario.",
        )
    parts = []
    for label in dict.fromkeys(raw[spec.scenario]):
        part = raw.loc[raw[spec.scenario] == label].drop(columns=spec.scenario)
        ds = from_frame(part.reset_index(drop=True), date=spec.date, frequency=spec.frequency)
        frame = ds.frame
        frame.index = pd.MultiIndex.from_arrays(
            [[label] * len(frame), frame.index], names=["scenario", "period"]
        )
        parts.append(frame)
    if not parts:
        raise ReaderError(
            "the scenarios file has no rows",
            observed=f"{path}",
            change="Give at least one scenario.",
        )
    severity = (
        spec.severity if spec.severity is not None else list(dict.fromkeys(raw[spec.scenario]))
    )
    return pd.concat(parts), severity
