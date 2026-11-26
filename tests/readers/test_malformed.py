"""Malformed CSV: every input ends in a dataset or a classed `DazflowError` (roadmap Stage 13).

Hypothesis writes random bytes, ragged rows, unreadable dates and columns of
mixed types to a file and reads it with the csv reader. The oracle is the
contract of `readers.read`: it returns a `Dataset` whose index is sorted,
unique periods at one supported frequency, or it raises a `DazflowError`
subclass. Any other exception (a pandas `TypeError`, a numpy error) fails.
The number of examples comes from the hypothesis profile (`tests/conftest.py`).
"""

import string

import pandas as pd
from hypothesis import example, given
from hypothesis import strategies as st

from dazflow import Dataset, DazflowError
from dazflow import readers as rd
from dazflow.readers._base import frequency_of


def outcome(path):
    """The dataset read from `path`, or the `DazflowError` raised; anything else propagates."""
    try:
        ds = rd.read({"path": str(path), "reader": "csv"})
    except DazflowError as err:
        assert str(err)
        return err
    assert isinstance(ds, Dataset)
    index = ds.frame.index
    assert isinstance(index, pd.PeriodIndex)
    assert index.is_monotonic_increasing
    assert index.is_unique
    assert frequency_of(index) in {"M", "Q", "Y"}
    # Every series carries the index's frequency (a file with only dates has none).
    assert {m.frequency for m in ds.series.values()} <= {frequency_of(index)}
    return ds


def csv_file(tmp_path_factory, data: bytes):
    path = tmp_path_factory.mktemp("malformed") / "data.csv"
    path.write_bytes(data)
    return path


DATES = st.one_of(
    st.dates(
        min_value=pd.Timestamp("1900-01-01").date(), max_value=pd.Timestamp("2100-12-31").date()
    ).map(str),
    st.builds(lambda y, q: f"{y}Q{q}", st.integers(1900, 2100), st.integers(0, 6)),
    st.builds(lambda y, m: f"{y}-{m:02d}", st.integers(1900, 2100), st.integers(0, 14)),
    st.integers(-5, 3000).map(str),
    st.floats(allow_nan=True, allow_infinity=True).map(str),
    st.text(string.printable.replace(",", "").replace("\n", "").replace("\r", ""), max_size=12),
    st.just(""),
)
CELLS = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True).map(repr),
    st.integers(-(10**20), 10**20).map(str),
    st.sampled_from(["", "NA", "nan", "inf", "-inf", "true", "1e400", "1,5", '"x"', "N/A"]),
    st.text(string.ascii_letters + string.digits + " .-+", max_size=8),
)


@given(st.binary(max_size=400))
@example(b"")
@example(b"\xff\xfe\x00d\x00a\x00t\x00e")
@example(b"date,x\n2020Q1,1\n\x00\x00")
def test_random_bytes(tmp_path_factory, data):
    outcome(csv_file(tmp_path_factory, data))


@given(
    st.lists(
        st.lists(st.one_of(DATES, CELLS), min_size=0, max_size=6),
        min_size=0,
        max_size=12,
    )
)
@example([["date", "x"], ["2020Q1", "1"], ["2020Q2", "2", "3"]])
@example([["date", "x", "y"], ["2020Q1"], ["2020Q2", "1", "2"]])
def test_ragged_rows(tmp_path_factory, rows):
    text = "".join(",".join(r) + "\n" for r in rows)
    outcome(csv_file(tmp_path_factory, text.encode()))


@given(st.lists(DATES, min_size=1, max_size=16), st.data())
@example(["2020Q1", "2020-13"], None)
@example(["2020Q5", "2020Q6"], None)
@example(["1e400", "nan"], None)
@example(["1000", "2000-01-01"], None)  # issue #37: past pandas 2's nanosecond range
def test_bad_dates(tmp_path_factory, dates, data):
    values = [str(i) for i in range(len(dates))]
    text = "date,x\n" + "".join(f"{d},{v}\n" for d, v in zip(dates, values, strict=True))
    outcome(csv_file(tmp_path_factory, text.encode()))


@given(
    st.integers(1, 12), st.lists(st.lists(CELLS, min_size=2, max_size=2), min_size=1, max_size=12)
)
@example(1, [["1", "x"]])
@example(2, [["1.5", "true"], ["", "2"]])
def test_mixed_types(tmp_path_factory, start, cells):
    periods = pd.period_range(f"2000Q{(start - 1) % 4 + 1}", periods=len(cells), freq="Q")
    text = "date,x,y\n" + "".join(
        f"{p},{a},{b}\n" for p, (a, b) in zip(periods, cells, strict=True)
    )
    path = csv_file(tmp_path_factory, text.encode())
    result = outcome(path)
    if isinstance(result, Dataset):
        # Differential: the values kept are pandas' own reading of the file, cell by cell.
        expected = pd.read_csv(path).set_index("date")
        assert list(result.frame.columns) == ["x", "y"]
        assert [str(p) for p in result.frame.index] == [str(p) for p in periods]
        for column in ("x", "y"):
            got, want = result.frame[column].tolist(), expected[column].tolist()
            assert len(got) == len(want)
            for g, w in zip(got, want, strict=True):
                assert (pd.isna(g) and pd.isna(w)) or g == w, (column, got, want)
