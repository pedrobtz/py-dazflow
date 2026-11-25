"""Dataset: validation, metadata and immutability (design §5.1, D13)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dazflow as dz
from dazflow import Dataset, DatasetError, LogEntry, SeriesMeta


def quarterly(n=4):
    return pd.period_range("2020Q1", periods=n, freq="Q")


def single_frame():
    return pd.DataFrame(
        {"rate": [1.0, 2.0, 3.0, 4.0], "gdp": [0.5, 0.25, -0.5, 1.5]},
        index=quarterly(),
    )


def single(**kwargs):
    series = {
        "rate": SeriesMeta(role="target", unit="%"),
        "gdp": SeriesMeta(role="driver", unit="%", expected_sign=-1),
    }
    return Dataset(single_frame(), series, **kwargs)


def scenarios_frame():
    index = pd.MultiIndex.from_product(
        [["base", "severe"], pd.period_range("2021Q1", periods=2, freq="Q")]
    )
    return pd.DataFrame({"gdp": [1.0, 1.2, -2.0, -3.0]}, index=index)


def snapshot(frame):
    return frame.copy(deep=True), list(frame.index.names)


def assert_unchanged(frame, before):
    copy, names = before
    pd.testing.assert_frame_equal(frame, copy)
    assert list(frame.index.names) == names


# --- construction -----------------------------------------------------------


def test_names_are_public():
    assert dz.Dataset is Dataset
    assert dz.core.Dataset is Dataset


def test_single_segment_dataset_has_a_period_index():
    ds = single()

    assert isinstance(ds.frame.index, pd.PeriodIndex)
    assert ds.frame.index.name == "period"
    assert ds.columns == ("rate", "gdp")
    assert not ds.is_panel
    assert ds.targets == ("rate",)
    assert ds.drivers == ("gdp",)
    assert ds.series["gdp"].expected_sign == -1


def test_panel_dataset_has_a_segment_period_multiindex():
    index = pd.MultiIndex.from_product([["north", "south"], quarterly(2)])
    ds = Dataset(pd.DataFrame({"rate": [1.0, 2.0, 3.0, 4.0]}, index=index))

    assert ds.is_panel
    assert list(ds.frame.index.names) == ["segment", "period"]
    assert ds.segments == ("north", "south")


def test_missing_metadata_defaults_to_role_other():
    ds = Dataset(single_frame())

    assert ds.series["rate"] == SeriesMeta()
    assert ds.series["rate"].role == "other"


@pytest.mark.parametrize(
    ("frame", "fragment"),
    [
        (pd.DataFrame({"x": [1.0]}, index=pd.Index([0])), "PeriodIndex"),
        (
            pd.DataFrame({"x": [1.0, 2.0]}, index=pd.PeriodIndex(["2020Q1", "2020Q1"], freq="Q")),
            "unique",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0]}, index=pd.PeriodIndex(["2020Q2", "2020Q1"], freq="Q")),
            "sorted",
        ),
        (
            pd.DataFrame({0: [1.0]}, index=pd.period_range("2020Q1", periods=1, freq="Q")),
            "column names",
        ),
        (
            pd.DataFrame(
                {"x": pd.to_datetime(["2020-01-01"])},
                index=pd.period_range("2020Q1", periods=1, freq="Q"),
            ),
            "kind",
        ),
        (
            pd.DataFrame(
                {"x": [1.0, 2.0]},
                index=pd.MultiIndex.from_product([quarterly(1), ["a", "b"]]),
            ),
            "(segment, period)",
        ),
    ],
)
def test_invalid_frames_raise_a_dataset_error(frame, fragment):
    with pytest.raises(DatasetError, match=r"(?s)observed:.*change:") as info:
        Dataset(frame)

    assert fragment in str(info.value)


def test_metadata_for_an_unknown_column_is_refused():
    with pytest.raises(DatasetError, match="nope"):
        Dataset(single_frame(), {"nope": SeriesMeta()})


def test_invalid_metadata_is_a_dataset_error():
    with pytest.raises(DatasetError, match="role"):
        SeriesMeta(role="regressor")
    with pytest.raises(DatasetError, match="expected_sign"):
        SeriesMeta(expected_sign=2)


def test_scenarios_take_a_severity_order():
    ds = single(scenarios=scenarios_frame(), severity=["base", "severe"])

    assert ds.severity == ("base", "severe")
    assert list(ds.scenarios.index.names) == ["scenario", "period"]


def test_severity_defaults_to_order_of_appearance():
    assert single(scenarios=scenarios_frame()).severity == ("base", "severe")


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"severity": ["base"]}, "severity"),
        ({"severity": ["base", "severe", "extra"]}, "severity"),
        ({"severity": ["base", "base"]}, "severity"),
    ],
)
def test_severity_must_list_each_scenario_once(kwargs, fragment):
    with pytest.raises(DatasetError, match=fragment):
        single(scenarios=scenarios_frame(), **kwargs)


def test_scenario_columns_must_be_dataset_columns():
    frame = scenarios_frame().rename(columns={"gdp": "unemployment"})

    with pytest.raises(DatasetError, match="unemployment"):
        single(scenarios=frame)


def test_scenario_frequency_must_match():
    frame = scenarios_frame()
    frame.index = pd.MultiIndex.from_product(
        [["base", "severe"], pd.period_range("2021-01", periods=2, freq="M")]
    )

    with pytest.raises(DatasetError, match="frequency"):
        single(scenarios=frame)


def test_severity_without_scenarios_is_refused():
    with pytest.raises(DatasetError, match="severity"):
        single(severity=["base"])


# --- immutability -----------------------------------------------------------


def test_construction_leaves_the_callers_frame_unchanged():
    frame = single_frame()
    frame.index.name = "quarter"
    before = snapshot(frame)

    Dataset(frame)

    assert_unchanged(frame, before)


def test_changing_the_callers_frame_later_does_not_change_the_dataset():
    frame = single_frame()
    ds = Dataset(frame)
    fp = ds.fingerprint()

    frame.iloc[0, 0] = 99.0
    frame["new"] = 1.0

    assert ds.frame.iloc[0, 0] == 1.0
    assert "new" not in ds.frame
    assert ds.fingerprint() == fp


def test_changing_the_returned_frame_does_not_change_the_dataset():
    ds = single(scenarios=scenarios_frame())
    fp = ds.fingerprint()

    view = ds.frame
    view.iloc[0, 0] = 99.0
    view.iloc[1, 1] = 99.0
    scen = ds.scenarios
    scen.iloc[0, 0] = 99.0

    assert ds.frame.iloc[0, 0] == 1.0
    assert ds.frame.iloc[1, 1] == 0.25
    assert ds.scenarios.iloc[0, 0] == 1.0
    assert ds.fingerprint() == fp


def test_attributes_cannot_be_reassigned():
    ds = single()

    with pytest.raises(AttributeError):
        ds.frame = single_frame()
    with pytest.raises(AttributeError):
        ds.anything = 1
    with pytest.raises(TypeError):
        ds.series["rate"] = SeriesMeta()


def methods(ds):
    entry = LogEntry(action="exclude", reason="test", rows=("2020Q1",), columns=("rate",))
    other = single_frame() * 2
    return {
        "with_frame": lambda: ds.with_frame(other),
        "with_frame_logged": lambda: ds.with_frame(other, entry=entry),
        "with_series": lambda: ds.with_series("gdp", unit="pp"),
        "with_scenarios": lambda: ds.with_scenarios(scenarios_frame(), severity=["severe", "base"]),
        "without_scenarios": lambda: ds.with_scenarios(None),
        "with_log": lambda: ds.with_log(entry),
        "select": lambda: ds.select(["gdp"]),
    }


@pytest.mark.parametrize("name", list(methods(None)))
def test_every_method_returns_a_new_dataset_and_leaves_the_original(name):
    frame = single_frame()
    scen = scenarios_frame()
    ds = Dataset(frame, {"rate": SeriesMeta(role="target")}, scenarios=scen)
    frame_before, scen_before = snapshot(frame), snapshot(scen)
    state = (ds.frame, dict(ds.series), ds.scenarios, ds.severity, ds.log, ds.fingerprint())

    out = methods(ds)[name]()

    assert isinstance(out, Dataset)
    assert out is not ds
    assert_unchanged(frame, frame_before)
    assert_unchanged(scen, scen_before)
    pd.testing.assert_frame_equal(ds.frame, state[0])
    assert dict(ds.series) == state[1]
    pd.testing.assert_frame_equal(ds.scenarios, state[2])
    assert (ds.severity, ds.log, ds.fingerprint()) == state[3:]


def test_methods_do_what_they_say():
    ds = single()
    entry = LogEntry(action="fill", reason="gap", rows=("2020Q2",))

    assert ds.with_series("gdp", unit="pp").series["gdp"].unit == "pp"
    assert ds.with_series("gdp", unit="pp").series["gdp"].expected_sign == -1
    assert ds.with_log(entry).log == (entry,)
    assert ds.with_frame(single_frame(), entry=entry).log == (entry,)
    assert ds.select(["gdp"]).columns == ("gdp",)
    assert ds.select(["gdp"]).series["gdp"] == ds.series["gdp"]
    assert ds.with_scenarios(scenarios_frame()).severity == ("base", "severe")
    assert ds.with_scenarios(scenarios_frame()).with_scenarios(None).scenarios is None


def test_with_frame_keeps_metadata_of_kept_columns_and_defaults_new_ones():
    frame = single_frame().drop(columns="rate").assign(new=1.0)

    out = single().with_frame(frame)

    assert out.series["gdp"].role == "driver"
    assert out.series["new"] == SeriesMeta()
    assert "rate" not in out.series


def test_with_series_on_an_unknown_column_is_refused():
    with pytest.raises(DatasetError, match="nope"):
        single().with_series("nope", unit="pp")


def test_select_of_an_unknown_column_is_refused():
    with pytest.raises(DatasetError, match="nope"):
        single().select(["nope"])


def test_log_entries_are_kept_in_order():
    a = LogEntry(action="exclude", reason="a")
    b = LogEntry(action="fill", reason="b")

    assert single().with_log(a).with_log(b).log == (a, b)


def test_log_entry_needs_a_reason():
    with pytest.raises(DatasetError, match="reason"):
        LogEntry(action="exclude", reason=" ")


@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1, max_size=8),
    i=st.integers(min_value=0, max_value=7),
)
def test_mutating_any_returned_frame_never_reaches_the_dataset(values, i):
    frame = pd.DataFrame(
        {"x": values}, index=pd.period_range("2000-01", periods=len(values), freq="M")
    )
    ds = Dataset(frame)
    fp = ds.fingerprint()

    out = ds.frame
    out.iloc[i % len(values), 0] = (
        np.float64(12345.0) if values[i % len(values)] != 12345.0 else 0.0
    )
    frame.iloc[i % len(values), 0] = 54321.0

    assert ds.fingerprint() == fp


@pytest.mark.parametrize(
    ("make", "fragment"),
    [
        (lambda: Dataset(single_frame().to_numpy()), "DataFrame"),
        (
            lambda: Dataset(
                pd.DataFrame(
                    [[1.0, 2.0]],
                    columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")]),
                    index=quarterly(1),
                )
            ),
            "MultiIndex",
        ),
        (
            lambda: Dataset(pd.DataFrame([[1.0, 2.0]], columns=["a", "a"], index=quarterly(1))),
            "repeated: a",
        ),
        (
            lambda: Dataset(
                pd.DataFrame(
                    {"x": [1.0, 2.0]}, index=pd.MultiIndex.from_product([[1.5, 2.5], quarterly(1)])
                )
            ),
            "segment labels",
        ),
        (lambda: Dataset(single_frame(), log=["not an entry"]), "LogEntry"),
        (lambda: single(scenarios=scenarios_frame().to_numpy()), "DataFrame"),
        (lambda: single(scenarios=scenarios_frame().droplevel(0)), "(scenario, period)"),
    ],
)
def test_other_invalid_inputs_raise_a_dataset_error(make, fragment):
    with pytest.raises(DatasetError) as info:
        make()

    assert fragment in str(info.value)


def test_metadata_may_be_given_as_mappings():
    ds = Dataset(single_frame(), {"rate": {"role": "target", "unit": "%"}})

    assert ds.series["rate"] == SeriesMeta(role="target", unit="%")


def test_single_segment_dataset_has_no_segments_and_a_short_repr():
    ds = single(scenarios=scenarios_frame())

    assert ds.segments == ()
    assert repr(ds) == "<Dataset series: 4 rows x 2 columns, 2 scenarios, 0 log entries>"
