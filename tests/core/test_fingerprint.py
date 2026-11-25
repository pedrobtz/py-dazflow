"""Dataset.fingerprint(): the canonical form of design §5.1 and its stored hashes.

The stored SHA-256 values below must come out the same on every leg of the
full CI profile (Ubuntu, Windows, macOS; Python 3.12-3.14; latest and lowest
direct dependencies). If one changes, the canonical form changed: that is a
new format version in the design, never an edit of the expected value.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dazflow import Dataset, LogEntry, SeriesMeta

FIXTURE_SHA256 = "330b72109a38de8b39ed696ced86701de411e0d03e8dfb70e345c00e4c819055"
PANEL_SHA256 = "b25abc62ac8608412acc8576ea8fc55275a7cff9f97e506c44a9324ab8339b71"


def fixture():
    """Every value kind, every special float, every metadata field and scenarios."""
    index = pd.period_range("2019Q3", periods=5, freq="Q")
    frame = pd.DataFrame(
        {
            "loss_rate": [0.1, -0.0, np.nan, np.inf, 5e-324],
            "gdp": [1.0, -2.5, 1e300, -np.inf, 0.30000000000000004],
            "count": pd.array([1, 2, None, -(2**53) - 1, 2**62], dtype="Int64"),
            "flag": pd.array([True, False, None, True, False], dtype="boolean"),
            "note": ["a", "é ü", None, "日本", '"quoted"\n'],
        },
        index=index,
    )
    series = {
        "loss_rate": SeriesMeta(
            role="target",
            unit="%",
            frequency="Q",
            source="internal ledger",
            transforms=("logit", "diff"),
        ),
        "gdp": SeriesMeta(role="driver", unit="% y/y", source="ONS", expected_sign=-1),
    }
    scen_index = pd.MultiIndex.from_product(
        [["adverse", "baseline"], pd.period_range("2020Q4", periods=2, freq="Q")]
    )
    scenarios = pd.DataFrame({"gdp": [-3.0, -1.0, 1.5, 2.0]}, index=scen_index)
    return Dataset(frame, series, scenarios=scenarios, severity=["baseline", "adverse"])


def panel_fixture():
    index = pd.MultiIndex.from_product(
        [["north", "south", "west"], pd.period_range("2023-11", periods=3, freq="M")]
    )
    frame = pd.DataFrame({"balance": np.arange(9, dtype="float64") / 3}, index=index)
    return Dataset(frame, {"balance": SeriesMeta(role="target", unit="GBP m")})


# --- the oracle: stored hashes --------------------------------------------


def test_fixture_has_its_stored_fingerprint():
    assert fixture().fingerprint() == FIXTURE_SHA256


def test_panel_fixture_has_its_stored_fingerprint():
    assert panel_fixture().fingerprint() == PANEL_SHA256


# --- the oracle: the canonical bytes written out by hand -------------------


def test_fingerprint_is_the_sha256_of_the_documented_canonical_form():
    frame = pd.DataFrame(
        {"y": [1.5, np.nan], "n": [3, -4]},
        index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"),
    )
    ds = Dataset(frame, {"y": SeriesMeta(role="target", unit="%", expected_sign=1)})
    # Written from design §5.1, not from the implementation.
    expected = (
        '{"columns":['
        '{"kind":"float","meta":{"expected_sign":1,"frequency":null,"role":"target",'
        '"source":null,"transforms":[],"unit":"%"},"name":"y","values":["3ff8000000000000",null]},'
        '{"kind":"int","meta":{"expected_sign":null,"frequency":null,"role":"other",'
        '"source":null,"transforms":[],"unit":null},"name":"n","values":[3,-4]}'
        "],"
        '"format":"dazflow.dataset/1",'
        '"index":{"kind":"period","rows":['
        '"2020-01-01T00:00:00.000000000/2020-04-01T00:00:00.000000000",'
        '"2020-04-01T00:00:00.000000000/2020-07-01T00:00:00.000000000"]},'
        '"scenarios":null}'
    ).encode("ascii")

    assert ds.fingerprint() == hashlib.sha256(expected).hexdigest()


def test_panel_and_scenario_rows_follow_the_documented_canonical_form():
    index = pd.MultiIndex.from_tuples(
        [("a", pd.Period("2021", freq="Y")), ("b", pd.Period("2021", freq="Y"))]
    )
    frame = pd.DataFrame({"s": ["x", None]}, index=index)
    scen = pd.DataFrame(
        {"s": ["ü"]},
        index=pd.MultiIndex.from_tuples([("up", pd.Period("2022", freq="Y"))]),
    )
    ds = Dataset(frame, scenarios=scen)
    meta = '{"expected_sign":null,"frequency":null,"role":"other","source":null,"transforms":[],"unit":null}'
    year = "2021-01-01T00:00:00.000000000/2022-01-01T00:00:00.000000000"
    expected = (
        '{"columns":[{"kind":"string","meta":' + meta + ',"name":"s","values":["x",null]}],'
        '"format":"dazflow.dataset/1",'
        '"index":{"kind":"panel","rows":[["a","' + year + '"],["b","' + year + '"]]},'
        '"scenarios":{"columns":[{"kind":"string","name":"s","values":["\\u00fc"]}],'
        '"rows":[["up","2022-01-01T00:00:00.000000000/2023-01-01T00:00:00.000000000"]],'
        '"severity":["up"]}}'
    ).encode("ascii")

    assert ds.fingerprint() == hashlib.sha256(expected).hexdigest()


def test_fingerprint_is_lowercase_hex_sha256():
    fp = fixture().fingerprint()

    assert len(fp) == 64
    assert fp == fp.lower()
    int(fp, 16)


# --- what changes it --------------------------------------------------------


def float_frames():
    return st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        min_size=1,
        max_size=12,
    )


@settings(max_examples=100, deadline=None)
@given(values=float_frames(), data=st.data())
def test_changing_one_value_changes_the_fingerprint(values, data):
    i = data.draw(st.integers(min_value=0, max_value=len(values) - 1))
    new = data.draw(st.floats(allow_nan=True, allow_infinity=True).filter(lambda v: v != values[i]))
    index = pd.period_range("2001-01", periods=len(values), freq="M")
    changed = list(values)
    changed[i] = new

    a = Dataset(pd.DataFrame({"x": values}, index=index))
    b = Dataset(pd.DataFrame({"x": changed}, index=index))

    assert a.fingerprint() != b.fingerprint()


def test_changing_the_smallest_possible_amount_changes_the_fingerprint():
    ds = fixture()
    frame = ds.frame
    frame.iloc[1, 1] = np.nextafter(frame.iloc[1, 1], np.inf)

    assert ds.with_frame(frame).fingerprint() != ds.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "other"),
        ("unit", "pp"),
        ("frequency", "M"),
        ("source", "elsewhere"),
        ("transforms", ("logit",)),
        ("expected_sign", 1),
    ],
)
def test_changing_one_metadata_field_changes_the_fingerprint(field, value):
    ds = fixture()

    assert ds.with_series("loss_rate", **{field: value}).fingerprint() != ds.fingerprint()


def test_changing_the_column_order_changes_the_fingerprint():
    ds = fixture()
    reordered = ds.with_frame(ds.frame[list(reversed(ds.columns))])

    assert set(reordered.columns) == set(ds.columns)
    assert dict(reordered.series) == dict(ds.series)
    assert reordered.fingerprint() != ds.fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        lambda ds: ds.with_scenarios(None),
        lambda ds: ds.with_scenarios(ds.scenarios, severity=["adverse", "baseline"]),
        lambda ds: ds.with_scenarios(ds.scenarios * 2),
        lambda ds: ds.with_frame(ds.frame.set_axis(ds.frame.index + 1)),
        lambda ds: ds.with_frame(ds.frame.astype({"loss_rate": "float32"})),
    ],
    ids=[
        "no scenarios",
        "severity order",
        "scenario values",
        "periods shifted",
        "float32 rounding",
    ],
)
def test_other_changes_change_the_fingerprint(change):
    ds = fixture()

    assert change(ds).fingerprint() != ds.fingerprint()


def test_int_and_float_columns_with_equal_values_differ():
    index = pd.period_range("2020", periods=2, freq="Y")

    a = Dataset(pd.DataFrame({"x": [1, 2]}, index=index))
    b = Dataset(pd.DataFrame({"x": [1.0, 2.0]}, index=index))

    assert a.fingerprint() != b.fingerprint()


def test_same_label_different_frequency_differs():
    a = Dataset(pd.DataFrame({"x": [1.0]}, index=pd.PeriodIndex(["2020Q1"], freq="Q-DEC")))
    b = Dataset(pd.DataFrame({"x": [1.0]}, index=pd.PeriodIndex(["2020Q1"], freq="Q-MAR")))

    assert a.fingerprint() != b.fingerprint()


# --- what does not change it ------------------------------------------------


def test_the_log_is_not_part_of_the_fingerprint():
    ds = fixture()

    assert ds.with_log(LogEntry(action="fill", reason="gap")).fingerprint() == ds.fingerprint()


def test_representation_details_do_not_change_the_fingerprint():
    ds = fixture()
    frame = ds.frame
    frame.index.name = "quarter"
    frame["note"] = frame["note"].astype(object)
    frame["loss_rate"] = np.asfortranarray(frame[["loss_rate"]].to_numpy())[:, 0]
    frame.loc[frame.index[1], "loss_rate"] = 0.0  # -0.0 and 0.0 are one value
    frame.loc[frame.index[2], "loss_rate"] = np.float64(
        np.frombuffer(b"\x01\x00\x00\x00\x00\x00\xf8\x7f")[0]
    )

    assert np.isnan(frame["loss_rate"].iloc[2])
    assert ds.with_frame(frame).fingerprint() == ds.fingerprint()


def test_string_dtype_and_object_strings_give_one_fingerprint():
    index = pd.period_range("2020", periods=2, freq="Y")

    a = Dataset(pd.DataFrame({"x": pd.array(["a", None], dtype="string")}, index=index))
    b = Dataset(pd.DataFrame({"x": pd.Series(["a", None], dtype=object, index=index)}))

    assert a.fingerprint() == b.fingerprint()
