"""Readers behind the io extra: a classed error naming dazflow[io] when it is missing."""

import sys

import pandas as pd
import pytest

from dazflow import MissingExtraError, ReaderError
from dazflow import readers as rd


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"date": ["2020-01-31", "2020-02-29", "2020-03-31"], "cpi": [1.0, 2.0, 3.5]}
    )


def test_excel_without_openpyxl_names_the_io_extra(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not read: the import fails first")
    monkeypatch.setitem(sys.modules, "openpyxl", None)  # `import openpyxl` raises ImportError
    with pytest.raises(MissingExtraError) as info:
        rd.read({"path": str(path)})
    err = info.value
    assert isinstance(err, ReaderError)
    assert err.module == "openpyxl" and err.extra == "io"
    assert "dazflow[io]" in str(err)


def test_parquet_without_pyarrow_names_the_io_extra(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(MissingExtraError, match=r"dazflow\[io\]") as info:
        rd.read({"path": str(path), "reader": "parquet"})
    assert info.value.module == "pyarrow"


def test_polars_input_without_polars_installed_is_not_needed_for_pandas(frame):
    # pandas input never touches narwhals' optional backends.
    ds = rd.from_frame(frame, date="date")
    assert list(ds.columns) == ["cpi"]


@pytest.mark.extra("openpyxl")
def test_excel_reads_when_the_extra_is_installed(tmp_path, frame):
    path = tmp_path / "data.xlsx"
    frame.assign(date=pd.to_datetime(frame["date"])).to_excel(path, index=False, engine="openpyxl")
    ds = rd.read({"path": str(path)})
    assert ds.frame.index.freqstr == "M"
    assert ds.frame["cpi"].tolist() == [1.0, 2.0, 3.5]
    assert ds.series["cpi"].frequency == "M"


@pytest.mark.extra("openpyxl")
def test_excel_reads_a_named_sheet(tmp_path, frame):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="other", index=False)
        frame.to_excel(w, sheet_name="macro", index=False)
    ds = rd.read({"path": str(path), "options": {"sheet": "macro"}})
    assert ds.frame["cpi"].tolist() == [1.0, 2.0, 3.5]


@pytest.mark.extra("pyarrow")
def test_parquet_reads_when_the_extra_is_installed(tmp_path, frame):
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, engine="pyarrow")
    ds = rd.read({"path": str(path)})
    assert [str(p) for p in ds.frame.index] == ["2020-01", "2020-02", "2020-03"]


@pytest.mark.extra("polars", "pyarrow")
def test_polars_input_goes_through_narwhals(frame):
    import polars as pl

    ds = rd.from_frame(pl.from_pandas(frame), date="date")
    pd.testing.assert_frame_equal(ds.frame, rd.from_frame(frame, date="date").frame)


@pytest.mark.extra("pyarrow")
def test_pyarrow_table_input_goes_through_narwhals(frame):
    import pyarrow as pa

    ds = rd.from_frame(pa.Table.from_pandas(frame), date="date")
    assert ds.frame["cpi"].tolist() == [1.0, 2.0, 3.5]
