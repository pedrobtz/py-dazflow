"""Table, Figure and Source: the artefact objects a step returns (design §9)."""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from dazflow import (
    ArtefactError,
    Check,
    Column,
    Figure,
    Gate,
    Layer,
    Override,
    Source,
    Table,
)

FP = "ab" * 32


def table(**kwargs):
    base = dict(
        id="T2.1",
        caption="Checks",
        columns=[
            Column(name="series", label="Series", kind="text"),
            Column(name="value", label="Value", unit="pp"),
        ],
        rows=[["a", 1.5], ["b", None]],
    )
    base.update(kwargs)
    return Table(**base)


def test_a_table_keeps_its_rows_and_columns():
    t = table()

    assert t.kind == "table"
    assert [c.name for c in t.columns] == ["series", "value"]
    assert t.rows == (("a", 1.5), ("b", None))


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([["a"]], "row 1 has 1 cells"),
        ([["a", 1.0, 2.0]], "row 1 has 3 cells"),
        ([["a", "1.5"]], "value"),
        ([[1, 1.5]], "series"),
        ([["a", math.nan]], "finite"),
        ([["a", math.inf]], "finite"),
        ([["a", True]], "value"),
    ],
)
def test_a_cell_of_the_wrong_kind_or_a_row_of_the_wrong_length_is_refused(rows, match):
    with pytest.raises(ArtefactError, match=match):
        table(rows=rows)


def test_repeated_column_names_are_refused():
    with pytest.raises(ArtefactError, match="repeated"):
        table(columns=[Column(name="a", label="A"), Column(name="a", label="B")], rows=[])


@pytest.mark.parametrize(
    ("kind", "good", "bad"),
    [
        ("integer", 3, 3.5),
        ("integer", None, True),
        ("percent", 0.25, "25%"),
        ("text", "x", 1),
        ("date", "2024-03-31", "31/03/2024"),
        ("status", "pass", "ok"),
        ("status", "overridden", None),
    ],
)
def test_each_column_kind_takes_its_own_values(kind, good, bad):
    cols = [Column(name="c", label="C", kind=kind)]

    assert Table(id="T1.1", caption="c", columns=cols, rows=[[good]]).rows == ((good,),)
    with pytest.raises(ArtefactError):
        Table(id="T1.1", caption="c", columns=cols, rows=[[bad]])


def test_a_date_cell_accepts_a_date_and_stores_it_as_iso_text():
    cols = [Column(name="d", label="Date", kind="date")]

    t = Table(id="T1.1", caption="c", columns=cols, rows=[[dt.date(2024, 3, 31)]])

    assert t.rows == (("2024-03-31",),)


def test_from_frame_infers_the_kinds_and_turns_missing_values_into_none():
    frame = pd.DataFrame(
        {
            "name": ["a", "b"],
            "n": np.array([1, 2], dtype="int64"),
            "x": [0.5, np.nan],
            "when": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        },
        index=pd.period_range("2024Q1", periods=2, freq="Q"),
    )

    t = Table.from_frame(
        frame,
        id="T3.1",
        caption="Frame",
        labels={"x": "Level"},
        units={"x": "%"},
        index="Quarter",
    )

    assert [(c.name, c.label, c.kind, c.unit) for c in t.columns] == [
        ("Quarter", "Quarter", "text", None),
        ("name", "name", "text", None),
        ("n", "n", "integer", None),
        ("x", "Level", "number", "%"),
        ("when", "when", "date", None),
    ]
    assert t.rows == (
        ("2024Q1", "a", 1, 0.5, "2024-01-31"),
        ("2024Q2", "b", 2, None, "2024-02-29"),
    )


def test_from_frame_refuses_labels_for_columns_it_does_not_have():
    with pytest.raises(ArtefactError, match="missing"):
        Table.from_frame(pd.DataFrame({"a": [1.0]}), id="T1.1", caption="c", labels={"b": "B"})


def test_from_gate_marks_each_check_with_its_status_and_reason():
    gate = Gate.evaluate(
        [
            Check(name="adf", observed=0.2, threshold=0.05, op="<"),
            Check(name="r2", observed=0.9, threshold=0.5, op=">=", unit="share"),
            Check(name="dw", observed=0.4, threshold=1.5, op=">"),
        ],
        overrides=[Override(check="adf", reason="Short sample.")],
    )

    t = Table.from_gate(gate, id="T8.1", caption="Gate")

    assert [c.kind for c in t.columns] == [
        "text",
        "number",
        "text",
        "number",
        "text",
        "status",
        "text",
    ]
    assert t.rows == (
        ("adf", 0.2, "<", 0.05, "", "overridden", "Short sample."),
        ("r2", 0.9, ">=", 0.5, "share", "pass", ""),
        ("dw", 0.4, ">", 1.5, "", "fail", ""),
    )


def test_from_gate_writes_a_missing_observed_value_as_an_empty_cell():
    gate = Gate.evaluate([Check(name="adf", observed=math.nan, threshold=0.05, op="<")])

    t = Table.from_gate(gate, id="T8.1", caption="Gate")

    assert t.rows[0][1] is None
    assert t.rows[0][5] == "fail"


def test_a_figure_checks_every_layer_against_its_x_values():
    with pytest.raises(ArtefactError, match="layer 1"):
        Figure(id="F1.1", caption="c", x=[1, 2, 3], layers=[Layer(kind="line", y=[1.0, 2.0])])


@pytest.mark.parametrize(
    "layer",
    [
        {"kind": "band", "y": [1.0, 2.0]},
        {"kind": "hline", "y": [1.0, 2.0]},
        {"kind": "line"},
        {"kind": "line", "y": [1.0, math.nan]},
        {"kind": "pie", "y": [1.0, 2.0]},
    ],
)
def test_a_layer_without_the_values_its_kind_needs_is_refused(layer):
    with pytest.raises(ArtefactError):
        Layer(**layer)


def test_a_figure_needs_at_least_one_layer_and_mixed_x_is_refused():
    with pytest.raises(ArtefactError, match="layer"):
        Figure(id="F1.1", caption="c", x=[1, 2], layers=[])
    with pytest.raises(ArtefactError, match="x"):
        Figure(id="F1.1", caption="c", x=[1, "b"], layers=[Layer(kind="line", y=[1.0, 2.0])])


@pytest.mark.parametrize("bad", ["F1", "T1.1", "F1.0"])
def test_a_figure_identifier_has_the_f_step_n_form(bad):
    with pytest.raises(ArtefactError):
        Figure(id=bad, caption="c", x=[1], layers=[Layer(kind="line", y=[1.0])])


def test_a_source_needs_a_run_id_and_a_sha256_fingerprint():
    s = Source(run_id="r-1", fingerprint=FP, date=dt.date(2026, 11, 25))

    assert s.note is None
    with pytest.raises(ArtefactError):
        Source(run_id="", fingerprint=FP, date=dt.date(2026, 11, 25))
    with pytest.raises(ArtefactError):
        Source(run_id="r-1", fingerprint="abc", date=dt.date(2026, 11, 25))


def test_artefacts_are_frozen():
    t = table()

    with pytest.raises(ArtefactError):
        t.caption = "other"


def test_a_band_needs_as_many_lows_as_highs():
    with pytest.raises(ArtefactError, match="y_low has 2 values and y_high 1"):
        Layer(kind="band", y_low=[1.0, 2.0], y_high=[3.0])


@pytest.mark.parametrize(
    "layer",
    [{"kind": "line", "y": [1.0, True]}, {"kind": "hline", "value": False}],
)
def test_a_boolean_is_not_a_number_in_a_layer(layer):
    with pytest.raises(ArtefactError, match="boolean"):
        Layer(**layer)


@pytest.mark.parametrize("x", [[1.0, math.inf], [True, 2]])
def test_x_must_be_finite_numbers_or_labels(x):
    with pytest.raises(ArtefactError, match="x"):
        Figure(id="F1.1", caption="c", x=x, layers=[Layer(kind="line", y=[1.0, 2.0])])


def test_a_timestamp_at_midnight_is_a_date_and_one_with_a_time_is_refused():
    cols = [Column(name="d", label="Date", kind="date")]

    t = Table(id="T1.1", caption="c", columns=cols, rows=[[dt.datetime(2024, 3, 31)]])

    assert t.rows == (("2024-03-31",),)
    with pytest.raises(ArtefactError, match="YYYY-MM-DD"):
        Table(id="T1.1", caption="c", columns=cols, rows=[[dt.datetime(2024, 3, 31, 12)]])


def test_from_frame_writes_booleans_as_text_unless_told_otherwise():
    frame = pd.DataFrame({"ok": [True, False], "share": [0.25, 0.5]})

    t = Table.from_frame(frame, id="T1.1", caption="c", kinds={"share": "percent"})

    assert [c.kind for c in t.columns] == ["text", "percent"]
    assert t.rows == (("True", 0.25), ("False", 0.5))


def test_the_catalogue_reports_its_step():
    from dazflow import ArtefactIds

    ids = ArtefactIds(12, tables={"a": 1})

    assert ids.step == 12
    assert "step=12" in repr(ids)
