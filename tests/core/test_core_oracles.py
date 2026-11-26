"""Oracles the mutation run found missing in `core/` (roadmap Stage 13, design D46).

Each test names, in its comment, the mutants it kills: a mutant mutmut made
that the earlier tests let through. The scheduled `Mutation` workflow is the
check that they stay killed.
"""

import datetime as dt
import decimal
import math
import sys
import textwrap

import pandas as pd
import pytest
from pydantic import Field

from dazflow import (
    ArtefactError,
    ArtefactIds,
    Check,
    Dataset,
    DatasetError,
    Gate,
    GateError,
    PluginError,
    Registry,
    SeriesMeta,
    Source,
    Table,
)
from dazflow.core.artefact import parse_id
from dazflow.core.config import Config

SOURCE = Source(run_id="run-1", fingerprint="ab" * 32, date=dt.date(2026, 11, 26))


# --- the gate arithmetic (core/gate.py) --------------------------------------------


@pytest.mark.parametrize(
    ("op", "below", "at", "above"),
    [
        ("<", True, False, False),
        ("<=", True, True, False),
        (">", False, False, True),
        (">=", False, True, True),
    ],
)
def test_each_operator_straddles_its_threshold_by_one_ulp(op, below, at, above):
    # _passes: every comparison flipped or widened by one (< to <=) is a different row here.
    t = 0.05
    for observed, passes in ((math.nextafter(t, 0), below), (t, at), (math.nextafter(t, 1), above)):
        assert Check(name="c", observed=observed, threshold=t, op=op).passed is passes


def test_the_status_of_a_check_is_pass_fail_or_overridden():
    # _status and _with_status: the three outcomes, and the gate's lists of them.
    passing = Check(name="a", observed=1.0, threshold=2.0, op="<")
    failing = Check(name="b", observed=3.0, threshold=2.0, op="<")
    accepted = Check(name="c", observed=3.0, threshold=2.0, op="<")
    gate = Gate.evaluate(
        [passing, failing, accepted], [{"check": "c", "reason": "Accepted in review."}]
    )

    assert [c.status for c in gate.checks] == ["pass", "fail", "overridden"]
    assert [c.name for c in gate.failed] == ["b"]
    assert [c.name for c in gate.overridden] == ["c"]
    assert gate.passed is False
    assert Gate.evaluate([passing, accepted], [{"check": "c", "reason": "Ok."}]).passed is True
    assert Gate.evaluate([]).passed is True


def test_an_override_given_as_a_mapping_is_attached_with_its_reason():
    # Gate.evaluate mutants 6 and 8: an override from configuration is a mapping.
    gate = Gate.evaluate(
        [Check(name="adf", observed=0.2, threshold=0.05, op="<")],
        [{"check": "adf", "reason": "Short sample."}],
    )
    assert gate["adf"].override.reason == "Short sample."
    with pytest.raises(GateError, match="reason"):
        Gate.evaluate([Check(name="adf", observed=0.2, threshold=0.05, op="<")], [{"check": "adf"}])


# --- artefact identifiers (core/artefact.py) -----------------------------------------


@pytest.mark.parametrize("step", [0, -1, True, False, "ingestion", 1.0])
def test_a_step_that_is_not_a_built_in_or_a_positive_whole_number_is_refused(step):
    # ArtefactIds._step_number mutants 1-7: the bounds 0 and 1, booleans and unknown names.
    with pytest.raises(ArtefactError, match="is not a step"):
        ArtefactIds(step)


def test_step_one_and_a_named_step_are_accepted():
    assert ArtefactIds(1).step == 1
    assert ArtefactIds("explore").step == 3
    assert ArtefactIds(12, tables={"t": 1}).table("t") == "T12.1"


@pytest.mark.parametrize("name", ["", 3, None])
def test_an_artefact_name_must_be_a_non_empty_string(name):
    # ArtefactIds._numbers mutant 2.
    with pytest.raises(ArtefactError, match="non-empty string"):
        ArtefactIds(1, tables={name: 1})


@pytest.mark.parametrize("n", [0, -1, True, 1.0])
def test_an_artefact_number_must_be_a_positive_whole_number(n):
    # ArtefactIds._numbers mutants 11 and 12: 0 straddles the bound, a bool is not a number.
    with pytest.raises(ArtefactError, match="positive whole number"):
        ArtefactIds(1, figures={"f": n})


def test_two_artefacts_sharing_a_number_are_named_in_the_refusal():
    # ArtefactIds._numbers mutant 32: the first holder of the number is named.
    with pytest.raises(ArtefactError, match="'first' and 'second'"):
        ArtefactIds(1, findings={"first": 2, "second": 2})


@pytest.mark.parametrize("identifier", [3, None, ("T", 1, 1)])
def test_an_identifier_that_is_not_text_is_an_artefact_error(identifier):
    # parse_id mutant 3: a non-string would reach the regular expression as a TypeError.
    with pytest.raises(ArtefactError, match="not an artefact identifier"):
        parse_id(identifier)


# --- tables built from a gate and from a frame ----------------------------------------


def test_a_gate_table_has_the_designs_columns_rows_notes_and_source():
    # Table.from_gate mutants 8-106: every column name and label, the notes and the source.
    gate = Gate.evaluate(
        [
            Check(name="adf", observed=0.01, threshold=0.05, op="<", unit="p-value"),
            Check(name="kpss", observed=float("nan"), threshold=0.05, op=">"),
        ],
        [{"check": "kpss", "reason": "Not computed on 8 periods."}],
    )
    table = Table.from_gate(gate, id="T3.9", caption="Gate", notes=["n1"], source=SOURCE)

    assert [(c.name, c.label, c.kind) for c in table.columns] == [
        ("check", "Check", "text"),
        ("observed", "Observed", "number"),
        ("op", "Rule", "text"),
        ("threshold", "Threshold", "number"),
        ("unit", "Unit", "text"),
        ("status", "Result", "status"),
        ("reason", "Reason", "text"),
    ]
    assert [list(r) for r in table.rows] == [
        ["adf", 0.01, "<", 0.05, "p-value", "pass", ""],
        ["kpss", None, ">", 0.05, "", "overridden", "Not computed on 8 periods."],
    ]
    assert table.notes == ("n1",)
    assert table.source == SOURCE


def test_a_frame_table_reads_kinds_from_dtypes_and_converts_cells_by_kind():
    # Table.from_frame mutants 26-139: bool is text, the kind overrides, notes and source.
    frame = pd.DataFrame(
        {
            "n": [1, 2],
            "x": [0.5, float("nan")],
            "flag": [True, False],
            "when": pd.to_datetime(["2024-03-31", "2024-06-30"]),
            "share": ["0.25", "0.5"],
            "count": ["3", "4"],
            "day": ["2024-01-02", "2024-02-03"],
        }
    )
    table = Table.from_frame(
        frame,
        id="T1.9",
        caption="c",
        kinds={"share": "percent", "count": "integer", "day": "date"},
        units={"x": "%"},
        notes=["n1"],
        source=SOURCE,
    )

    assert [(c.name, c.kind, c.unit) for c in table.columns] == [
        ("n", "integer", None),
        ("x", "number", "%"),
        ("flag", "text", None),
        ("when", "date", None),
        ("share", "percent", None),
        ("count", "integer", None),
        ("day", "date", None),
    ]
    assert [list(r) for r in table.rows] == [
        [1, 0.5, "True", "2024-03-31", 0.25, 3, "2024-01-02"],
        [2, None, "False", "2024-06-30", 0.5, 4, "2024-02-03"],
    ]
    assert table.notes == ("n1",)
    assert table.source == SOURCE


@pytest.mark.parametrize("given", ["labels", "units", "kinds"])
def test_labels_units_or_kinds_for_a_missing_column_are_refused(given):
    # Table.from_frame mutant 26: each of the three mappings alone.
    value = "number" if given == "kinds" else "text"
    with pytest.raises(ArtefactError, match="missing: ghost"):
        Table.from_frame(
            pd.DataFrame({"a": [1]}), id="T1.1", caption="c", **{given: {"ghost": value}}
        )


# --- datasets (core/dataset.py) ------------------------------------------------------


def quarters(n, start="2020Q1"):
    return pd.period_range(start, periods=n, freq="Q")


def two_series_with_scenarios():
    frame = pd.DataFrame(
        {"y": [1.0, 2.0], "gdp": [0.5, 0.6], "rate": [3.0, 3.1]}, index=quarters(2)
    )
    index = pd.MultiIndex.from_product([["base", "severe"], quarters(2, "2020Q3")])
    scen = pd.DataFrame({"gdp": [1.0, 1.1, -2.0, -3.0], "rate": [3.0, 3.0, 5.0, 6.0]}, index=index)
    meta = {"y": SeriesMeta(role="target", unit="%"), "gdp": SeriesMeta(role="driver")}
    return Dataset(frame, meta, scenarios=scen, severity=["base", "severe"])


def test_select_keeps_the_selected_columns_of_the_scenarios_only():
    # Dataset.select mutants 27 and 33.
    ds = two_series_with_scenarios().select(["y", "gdp"])

    assert list(ds.columns) == ["y", "gdp"]
    assert list(ds.scenarios.columns) == ["gdp"]
    assert ds.scenarios["gdp"].tolist() == [1.0, 1.1, -2.0, -3.0]
    assert ds.severity == ("base", "severe")


def test_with_scenarios_takes_the_severity_given_and_keeps_the_metadata():
    # Dataset.with_scenarios mutant 2, Dataset._replace mutant 19.
    ds = two_series_with_scenarios()
    again = ds.with_scenarios(ds.scenarios, severity=["severe", "base"])

    assert again.severity == ("severe", "base")
    assert dict(again.series) == dict(ds.series)
    assert ds.with_scenarios(None).scenarios is None
    assert dict(ds.with_scenarios(None).series) == dict(ds.series)


@pytest.mark.parametrize(
    "values",
    [
        [decimal.Decimal("1.5"), decimal.Decimal("2.5")],
        [1, "two"],
        [{"a": 1}, {"b": 2}],
    ],
)
def test_an_object_column_that_is_not_text_is_refused(values):
    # _column_kind mutant 13 and _missing mutant 4: only strings (or missing) make a text column.
    frame = pd.DataFrame({"x": pd.Series(values, index=quarters(2), dtype=object)})
    with pytest.raises(DatasetError, match="Convert the column"):
        Dataset(frame)


def test_an_object_column_of_text_and_missing_values_is_text():
    frame = pd.DataFrame({"x": pd.Series(["a", None], index=quarters(2), dtype=object)})
    values = Dataset(frame).frame["x"].tolist()
    assert values[0] == "a"
    assert pd.isna(values[1])


@pytest.mark.parametrize("name", [0, 1.5, ""])
def test_a_column_name_that_is_not_a_non_empty_string_is_refused(name):
    # _check_columns mutant 11.
    frame = pd.DataFrame({name: [1.0, 2.0]}, index=quarters(2))
    with pytest.raises(DatasetError, match="name"):
        Dataset(frame)


# --- configuration (core/config.py) --------------------------------------------------


class Note(Config):
    title: str = "Taux de défaut à 12 mois"
    tags: list[str] = Field(default_factory=lambda: ["un", "deux"])
    text: str = "word " * 40


def test_to_yaml_keeps_unicode_order_block_style_and_long_lines():
    # Config.to_yaml mutants 2-19: allow_unicode, sort_keys, default_flow_style, width.
    assert Note().to_yaml() == (
        f"title: Taux de défaut à 12 mois\ntags:\n- un\n- deux\ntext: '{'word ' * 40}'\n"
    )


def test_dump_writes_utf8_with_unix_newlines_and_load_reads_it_back(tmp_path):
    # Config.dump mutants 2-9 and Config.load mutant 4.
    path = tmp_path / "note.yaml"
    Note().dump(path)

    data = path.read_bytes()
    assert b"\r\n" not in data
    assert "défaut".encode() in data
    assert Note.load(path) == Note()


def test_a_duplicate_key_in_a_nested_mapping_is_refused():
    # _construct_mapping mutants: duplicates are found at every depth, with their position.
    class Outer(Config):
        inner: dict[str, int] = Field(default_factory=dict)

    with pytest.raises(Exception, match=r"duplicate key 'a'.*line 3"):
        Outer.from_yaml("inner:\n  a: 1\n  a: 2\n")


# --- the plugin registry (core/registry.py) ------------------------------------------


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [m for m in sys.modules if m.startswith("dzo_")]:
        del sys.modules[name]


def install(site, dist, entry_points, modules):
    import importlib

    info = site / f"{dist}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n", "utf-8")
    lines = "".join(f"{k} = {v}\n" for k, v in entry_points.items())
    (info / "entry_points.txt").write_text(f"[dazflow.plugins]\n{lines}", encoding="utf-8")
    for name, source in modules.items():
        (site / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    importlib.invalidate_caches()


def test_a_broken_plugin_is_reported_on_every_lookup_not_only_the_first(site):
    # Registry._load_plugins mutant 5: after a failure, the next lookup tries again.
    install(site, "dzo-broken", {"broken": "dzo_broken"}, {"dzo_broken": "raise ValueError('x')\n"})
    widgets = Registry("widget")

    for _ in range(2):
        with pytest.raises(PluginError, match="dzo_broken"):
            widgets.keys()


def test_a_plugin_that_looks_up_its_registry_while_it_loads_does_not_recurse(site):
    # Registry._load_plugins mutants 1 and 2: the flag is set before the plugins load.
    host = """
        from dazflow import Registry
        widgets = Registry("widget")
    """
    plugin = """
        import dzo_host
        assert "fancy" not in dzo_host.widgets  # a lookup during the load
        dzo_host.widgets.add("fancy", 1)
    """
    install(site, "dzo-plugin", {"fancy": "dzo_plugin"}, {"dzo_host": host, "dzo_plugin": plugin})
    import dzo_host

    assert dzo_host.widgets.get("fancy") == 1
