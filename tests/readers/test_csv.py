"""The csv reader and the date rules every reader shares."""

import datetime as dt

import pandas as pd
import pytest

from dazflow import ConfigError, ReaderError
from dazflow import readers as rd


def write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def labels(ds):
    return [str(p) for p in ds.frame.index]


def test_csv_with_month_end_dates_is_monthly(tmp_path):
    path = write(tmp_path, "date,cpi,rate\n2020-01-31,1.0,5\n2020-02-29,2.0,6\n2020-03-31,3.0,7\n")
    ds = rd.read({"path": path})
    assert labels(ds) == ["2020-01", "2020-02", "2020-03"]
    assert ds.frame["cpi"].tolist() == [1.0, 2.0, 3.0]
    assert {m.frequency for m in ds.series.values()} == {"M"}
    assert ds.series["cpi"].source == path


@pytest.mark.parametrize(
    ("dates", "expected", "freq"),
    [
        (["2020Q1", "2020Q2"], ["2020Q1", "2020Q2"], "Q"),
        (["2020-01", "2020-02"], ["2020-01", "2020-02"], "M"),
        (["2019", "2020"], ["2019", "2020"], "Y"),
        (["2020-01-01", "2020-04-01", "2020-07-01"], ["2020Q1", "2020Q2", "2020Q3"], "Q"),
        (["2020-03-31", "2020-06-30", "2020-12-31"], ["2020Q1", "2020Q2", "2020Q4"], "Q"),
        (["2018-12-31", "2019-12-31", "2020-12-31"], ["2018", "2019", "2020"], "Y"),
    ],
)
def test_frequency_is_inferred_from_the_dates(tmp_path, dates, expected, freq):
    text = "date,x\n" + "".join(f"{d},{i}\n" for i, d in enumerate(dates))
    ds = rd.read({"path": write(tmp_path, text)})
    assert labels(ds) == expected
    assert ds.series["x"].frequency == freq


def test_integer_years_are_annual(tmp_path):
    ds = rd.read({"path": write(tmp_path, "year,x\n2019,1\n2020,2\n")})
    assert labels(ds) == ["2019", "2020"]


def test_a_given_frequency_reads_finer_dates_into_its_periods(tmp_path):
    ds = rd.read({"path": write(tmp_path, "d,x\n2020-02-15,1\n2020-05-15,2\n"), "frequency": "Q"})
    assert labels(ds) == ["2020Q1", "2020Q2"]


def test_a_frequency_finer_than_the_dates_is_refused(tmp_path):
    with pytest.raises(ReaderError, match="coarser"):
        rd.read({"path": write(tmp_path, "d,x\n2020Q1,1\n2020Q2,2\n"), "frequency": "M"})


def test_two_rows_in_one_period_are_refused(tmp_path):
    with pytest.raises(ReaderError, match="2020Q1"):
        rd.read({"path": write(tmp_path, "d,x\n2020-01-31,1\n2020-02-29,2\n"), "frequency": "Q"})


def test_daily_data_has_no_frequency_to_infer(tmp_path):
    text = "d,x\n" + "".join(f"2020-01-{i:02d},{i}\n" for i in range(1, 10))
    with pytest.raises(ReaderError, match="frequency"):
        rd.read({"path": write(tmp_path, text)})


def test_rows_are_sorted_by_period(tmp_path):
    ds = rd.read({"path": write(tmp_path, "d,x\n2020Q2,2\n2020Q1,1\n")})
    assert labels(ds) == ["2020Q1", "2020Q2"]
    assert ds.frame["x"].tolist() == [1, 2]


def test_date_column_and_columns_are_chosen(tmp_path):
    path = write(tmp_path, "a,when,b\n1,2020Q1,3\n2,2020Q2,4\n")
    ds = rd.read({"path": path, "date": "when", "columns": ["b"]})
    assert ds.columns == ("b",)
    with pytest.raises(ReaderError, match="nope"):
        rd.read({"path": path, "date": "nope"})
    with pytest.raises(ReaderError, match="zz"):
        rd.read({"path": path, "date": "when", "columns": ["zz"]})


def test_csv_options_are_passed_and_unknown_ones_refused(tmp_path):
    path = write(tmp_path, "d;x\n2020Q1;1,5\n2020Q2;2,5\n")
    ds = rd.read({"path": path, "options": {"sep": ";", "decimal": ","}})
    assert ds.frame["x"].tolist() == [1.5, 2.5]
    with pytest.raises(ConfigError, match="separator"):
        rd.read({"path": path, "options": {"separator": ";"}})


def test_unparseable_dates_name_the_value(tmp_path):
    with pytest.raises(ReaderError, match="soon"):
        rd.read({"path": write(tmp_path, "d,x\n2020Q1,1\nsoon,2\n")})


def test_a_missing_file_is_a_reader_error(tmp_path):
    with pytest.raises(ReaderError, match=r"missing\.csv"):
        rd.read({"path": str(tmp_path / "missing.csv")})


def test_the_reader_is_chosen_by_suffix_or_named(tmp_path):
    path = write(tmp_path, "d,x\n2020Q1,1\n", name="data.txt")
    with pytest.raises(ReaderError, match="reader"):
        rd.read({"path": path})
    assert rd.read({"path": path, "reader": "csv"}).columns == ("x",)
    assert set(rd.registry.keys()) >= {"csv", "excel", "parquet"}


def test_from_frame_uses_a_period_or_datetime_index():
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    ds = rd.from_frame(pd.DataFrame({"x": [1.0, 2.0]}, index=idx))
    assert labels(ds) == ["2020Q1", "2020Q2"]
    dti = pd.DatetimeIndex(["2020-01-31", "2020-02-29"])
    assert labels(rd.from_frame(pd.DataFrame({"x": [1.0, 2.0]}, index=dti))) == [
        "2020-01",
        "2020-02",
    ]


def test_from_frame_refuses_what_is_not_a_frame():
    with pytest.raises(ReaderError, match="list"):
        rd.from_frame([1, 2, 3])


# --- refusals and the remaining date forms -------------------------------------------


def periods_frame(index, **cols):
    return pd.DataFrame(cols or {"x": list(range(len(index)))}, index=index)


def test_periods_are_read_at_a_coarser_frequency_when_asked():
    ds = rd.from_frame(periods_frame(pd.PeriodIndex(["2019Q4", "2020Q4"], freq="Q")), frequency="Y")
    assert labels(ds) == ["2019", "2020"]


def test_periods_at_an_unsupported_frequency_are_refused():
    with pytest.raises(ReaderError, match="unsupported frequency"):
        rd.from_frame(periods_frame(pd.period_range("2020-01-06", periods=2, freq="W")))


def test_date_objects_are_read_as_dates():
    import datetime as dt

    frame = pd.DataFrame({"d": [dt.date(2020, 3, 31), dt.date(2020, 6, 30)], "x": [1, 2]})
    assert labels(rd.from_frame(frame, date="d")) == ["2020Q1", "2020Q2"]


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (pd.DataFrame({"d": [], "x": []}), "no rows"),
        (pd.DataFrame({"d": ["2020Q1", None], "x": [1, 2]}), "row 2"),
        (pd.DataFrame({"d": [2020.5, 2021.0], "x": [1, 2]}), "whole years"),
        (pd.DataFrame({"d": ["2020Q1", "2020-05"], "x": [1, 2]}), "different frequencies"),
        (pd.DataFrame({"d": pd.to_datetime(["2020-01-31"]), "x": [1]}), "cannot be told"),
        (pd.DataFrame(index=pd.RangeIndex(2)), "no columns"),
    ],
)
def test_dates_that_cannot_be_put_on_a_calendar_are_refused(frame, match):
    with pytest.raises(ReaderError, match=match):
        rd.from_frame(frame)


def test_a_file_without_a_known_suffix_needs_a_reader(tmp_path):
    with pytest.raises(ReaderError, match="no reader"):
        rd.read({"path": str(tmp_path / "data")})
    with pytest.raises(ReaderError, match="needs a path"):
        rd.read({"reader": "csv"})


def test_a_file_that_does_not_parse_is_a_reader_error(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("d,x\n2020Q1,caf\xe9\n".encode("latin-1"))
    with pytest.raises(ReaderError, match="cannot be parsed"):
        rd.read({"path": str(path)})
    assert (
        rd.read({"path": str(path), "options": {"encoding": "latin-1"}}).frame["x"].iloc[0]
        == "café"
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"d": ["1000", "2000-01-01"], "x": [1.0, 2.0]}),
        pd.DataFrame({"d": ["3000", "2000-01-01"], "x": [1.0, 2.0]}),
        pd.DataFrame({"d": [dt.date(1000, 1, 1), dt.date(2000, 1, 1)], "x": [1.0, 2.0]}),
    ],
)
def test_37_dates_outside_pandas_nanosecond_range_are_a_reader_error(frame):
    # On pandas 2, these dates overflow its nanosecond timestamps (issue #37).
    with pytest.raises(ReaderError):
        rd.from_frame(frame, date="d")
