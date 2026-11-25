"""The scenario reader shared by select and forecast: frames by (scenario, period), errors.

Oracle: the CSV written by hand.
"""

import pandas as pd
import pytest

from dazflow import ReaderError, readers


def test_a_scenarios_file_reads_by_scenario_and_period_in_file_order(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("scenario,date,a\nup,2025Q1,1\nup,2025Q2,2\ndown,2025Q1,3\ndown,2025Q2,4\n")
    frame, severity = readers.read_scenarios(readers.ScenarioInput(path="s.csv"), tmp_path)
    assert severity == ["up", "down"]
    assert frame.loc[("down", pd.Period("2025Q2", "Q")), "a"] == 4
    spec = readers.ScenarioInput(path=str(path), severity=["down", "up"])
    assert readers.read_scenarios(spec)[1] == ["down", "up"]


def test_a_missing_file_column_or_row_is_a_reader_error(tmp_path):
    with pytest.raises(ReaderError, match="does not exist"):
        readers.read_scenarios(readers.ScenarioInput(path="none.csv"), tmp_path)
    (tmp_path / "s.csv").write_text("date,a\n2025Q1,1\n")
    with pytest.raises(ReaderError, match="no column 'scenario'"):
        readers.read_scenarios(readers.ScenarioInput(path="s.csv"), tmp_path)
    (tmp_path / "e.csv").write_text("scenario,date,a\n")
    with pytest.raises(ReaderError, match="no rows"):
        readers.read_scenarios(readers.ScenarioInput(path="e.csv"), tmp_path)
