"""The ingest step: calendar alignment, publication lags, the history gate, artefacts.

Expected frames are worked by hand from the inputs written in each test.
"""

import numpy as np
import pandas as pd
import pytest

import dazflow as dz
from dazflow import ConfigError, IngestError
from dazflow.steps import ingest

NAN = np.nan


def monthly(values, start="2020-01", name="cpi"):
    index = pd.period_range(start, periods=len(values), freq="M")
    return pd.DataFrame({name: values}, index=index)


def quarterly(values, start="2020Q1", name="gdp"):
    index = pd.period_range(start, periods=len(values), freq="Q")
    return pd.DataFrame({name: values}, index=index)


def q(*labels):
    return pd.PeriodIndex(list(labels), freq="Q", name="period")


def config(series, *, inputs=("m", "q"), min_history=1, frequency="Q", **extra):
    return {
        "frequency": frequency,
        "inputs": [{"name": n} for n in inputs],
        "series": series,
        "min_history": min_history,
        **extra,
    }


def run(series, data, **kw):
    return ingest.run(config(series, inputs=tuple(data), **kw), data=data)


# --- alignment ---------------------------------------------------------------


def test_monthly_to_quarterly_by_period_end():
    data = {"m": monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
    r = run([{"name": "cpi", "aggregation": "end"}], data)
    expected = pd.DataFrame({"cpi": [3.0, 6.0]}, index=q("2020Q1", "2020Q2"))
    pd.testing.assert_frame_equal(r.dataset.frame, expected)
    assert r.dataset.series["cpi"].frequency == "Q"


def test_monthly_to_quarterly_by_period_average():
    data = {"m": monthly([1.0, 2.0, 3.0, 4.0, 5.0, 9.0])}
    r = run([{"name": "cpi", "aggregation": "average"}], data)
    expected = pd.DataFrame({"cpi": [2.0, 6.0]}, index=q("2020Q1", "2020Q2"))
    pd.testing.assert_frame_equal(r.dataset.frame, expected)


def test_an_incomplete_quarter_has_no_value():
    # Feb missing in Q1; May is the last month read, so Q2 lacks June.
    data = {"m": monthly([1.0, NAN, 3.0, 4.0, 5.0])}
    avg = run([{"name": "cpi", "aggregation": "average"}], data).dataset.frame
    end = run([{"name": "cpi", "aggregation": "end"}], data).dataset.frame
    pd.testing.assert_frame_equal(
        avg, pd.DataFrame({"cpi": [NAN, NAN]}, index=q("2020Q1", "2020Q2"))
    )
    pd.testing.assert_frame_equal(
        end, pd.DataFrame({"cpi": [3.0, NAN]}, index=q("2020Q1", "2020Q2"))
    )


def test_monthly_to_annual_and_quarterly_to_annual():
    data = {
        "m": monthly([float(i) for i in range(1, 25)], start="2019-01"),
        "q": quarterly([1.0, 2.0, 3.0, 4.0], start="2019Q1"),
    }
    r = run(
        [{"name": "cpi", "aggregation": "average"}, {"name": "gdp", "aggregation": "end"}],
        data,
        frequency="Y",
    )
    years = pd.PeriodIndex(["2019", "2020"], freq="Y", name="period")
    expected = pd.DataFrame({"cpi": [6.5, 18.5], "gdp": [4.0, NAN]}, index=years)
    pd.testing.assert_frame_equal(r.dataset.frame, expected)


def test_a_one_period_publication_lag():
    data = {"m": monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "q": quarterly([10.0, 20.0, 30.0])}
    r = run(
        [
            {"name": "gdp", "role": "target"},
            {"name": "cpi", "aggregation": "end", "lag": 1},
        ],
        data,
    )
    expected = pd.DataFrame(
        {"gdp": [10.0, 20.0, 30.0], "cpi": [NAN, 3.0, 6.0]},
        index=q("2020Q1", "2020Q2", "2020Q3"),
    )
    pd.testing.assert_frame_equal(r.dataset.frame, expected)
    entries = [(e.action, e.columns) for e in r.dataset.log]
    assert ("lag", ("cpi",)) in entries and ("convert", ("cpi",)) in entries


def test_a_lag_past_the_last_period_extends_the_calendar():
    r = run([{"name": "gdp", "lag": 2}], {"q": quarterly([1.0, 2.0])})
    expected = pd.DataFrame({"gdp": [1.0, 2.0]}, index=q("2020Q3", "2020Q4"))
    pd.testing.assert_frame_equal(r.dataset.frame, expected)


def test_series_on_one_calendar_from_the_first_to_the_last_period():
    data = {
        "q": quarterly([1.0, 2.0], start="2020Q1"),
        "r": quarterly([5.0], start="2020Q4", name="u"),
    }
    r = run([{"name": "gdp"}, {"name": "u"}], data)
    expected = pd.DataFrame(
        {"gdp": [1.0, 2.0, NAN, NAN], "u": [NAN, NAN, NAN, 5.0]},
        index=q("2020Q1", "2020Q2", "2020Q3", "2020Q4"),
    )
    pd.testing.assert_frame_equal(r.dataset.frame, expected)


def test_a_finer_target_than_the_input_is_refused():
    with pytest.raises(IngestError, match="lower"):
        run([{"name": "gdp"}], {"q": quarterly([1.0, 2.0])}, frequency="M")


def test_metadata_comes_from_the_configuration():
    data = {"q": quarterly([1.0, 2.0]), "r": quarterly([3.0, 4.0], name="raw_u")}
    r = run(
        [
            {"name": "gdp", "role": "target", "unit": "EUR bn", "source": "ONS"},
            {"name": "u", "input": "r", "column": "raw_u", "expected_sign": -1},
        ],
        data,
    )
    meta = r.dataset.series
    assert meta["gdp"].role == "target" and meta["gdp"].unit == "EUR bn"
    assert meta["gdp"].source == "ONS"
    assert meta["u"].role == "driver" and meta["u"].expected_sign == -1
    assert r.dataset.columns == ("gdp", "u")
    assert r.dataset.frame["u"].tolist() == [3.0, 4.0]


def test_inputs_are_read_through_the_readers(tmp_path):
    path = tmp_path / "macro.csv"
    path.write_text("date,cpi\n2020-01-31,1\n2020-02-29,2\n2020-03-31,6\n", encoding="utf-8")
    cfg = {
        "frequency": "Q",
        "inputs": [{"name": "macro", "path": str(path)}],
        "series": [{"name": "cpi"}],
        "min_history": 1,
    }
    r = ingest.run(cfg)
    assert r.dataset.frame["cpi"].tolist() == [3.0]
    assert r.dataset.series["cpi"].source == str(path)


# --- gate --------------------------------------------------------------------


@pytest.mark.parametrize(("minimum", "passed"), [(3, True), (4, True), (5, False)])
def test_history_gate_at_one_below_and_one_above_the_minimum(minimum, passed):
    # Four common quarters: a minimum of 4 is exactly met, 3 is one below it, 5 one above.
    data = {
        "q": quarterly([1.0, 2.0, 3.0, 4.0, 5.0]),
        "r": quarterly([NAN, 1.0, 1.0, 1.0, 1.0], name="u"),
    }
    r = run([{"name": "gdp", "role": "target"}, {"name": "u"}], data, min_history=minimum)
    check = r.gate["history"]
    assert check.observed == 4 and check.threshold == minimum and check.op == ">="
    assert check.passed is passed and r.gate.passed is passed
    assert r.history == 4 and (r.sample_start, r.sample_end) == ("2020Q2", "2021Q1")


def test_role_other_series_do_not_shorten_the_common_sample():
    data = {"q": quarterly([1.0, 2.0, 3.0]), "r": quarterly([NAN, NAN, 1.0], name="note")}
    r = run([{"name": "gdp", "role": "target"}, {"name": "note", "role": "other"}], data)
    assert r.history == 3


def test_a_failed_history_gate_can_be_overridden_with_a_reason():
    r = run(
        [{"name": "gdp"}],
        {"q": quarterly([1.0, 2.0])},
        min_history=3,
        overrides=[{"check": "history", "reason": "Short series; see section 2."}],
    )
    assert r.gate["history"].status == "overridden" and r.gate.passed


def test_a_configured_series_that_is_not_found_fails_the_gate():
    r = run([{"name": "gdp"}, {"name": "absent"}], {"q": quarterly([1.0, 2.0])})
    assert r.gate["found"].observed == 1 and not r.gate["found"].passed
    assert r.missing == ("absent",)
    assert r.dataset.columns == ("gdp",)


# --- configuration -------------------------------------------------------------


def test_configuration_errors_are_config_errors():
    with pytest.raises(ConfigError, match="input"):
        ingest.Config.model_validate(config([{"name": "a", "input": "zz"}]))
    with pytest.raises(ConfigError, match="repeated"):
        ingest.Config.model_validate(config([{"name": "a"}, {"name": "a"}]))
    with pytest.raises(ConfigError):
        ingest.Config.model_validate(config([{"name": "a", "lag": -1}]))
    with pytest.raises(ConfigError):
        ingest.Config.model_validate(config([{"name": "a"}], min_history=0))


def test_an_input_needs_a_path_or_data():
    with pytest.raises(IngestError, match="m"):
        ingest.run(config([{"name": "cpi"}], inputs=("m",)))


def test_a_column_in_two_inputs_needs_its_input_named():
    data = {"q": quarterly([1.0]), "r": quarterly([2.0])}
    with pytest.raises(IngestError, match="input"):
        run([{"name": "gdp"}], data)
    assert run([{"name": "gdp", "input": "r"}], data).dataset.frame["gdp"].tolist() == [2.0]


def test_a_non_numeric_series_is_refused():
    frame = pd.DataFrame({"gdp": ["a", "b"]}, index=pd.period_range("2020Q1", periods=2, freq="Q"))
    with pytest.raises(IngestError, match="gdp"):
        run([{"name": "gdp"}], {"q": frame})


def test_the_config_round_trips_through_yaml():
    cfg = ingest.Config.model_validate(config([{"name": "gdp", "role": "target"}]))
    assert ingest.Config.from_yaml(cfg.to_yaml()) == cfg
    assert ingest.Config.json_schema()["type"] == "object"


# --- result and artefacts --------------------------------------------------------


@pytest.fixture
def result():
    data = {
        "m": monthly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "q": quarterly([10.0, 20.0]),
    }
    return run(
        [
            {"name": "gdp", "role": "target", "unit": "EUR bn", "source": "ONS"},
            {"name": "cpi", "aggregation": "end", "unit": "index"},
            {"name": "absent"},
        ],
        data,
        min_history=2,
    )


def test_inventory_table_has_one_row_per_configured_series(result):
    table = result.artefacts["T1.1"]
    names = [c.name for c in table.columns]
    rows = {r[0]: dict(zip(names, r, strict=True)) for r in table.rows}
    assert list(rows) == ["gdp", "cpi", "absent"]
    assert rows["gdp"]["role"] == "target" and rows["gdp"]["unit"] == "EUR bn"
    assert rows["cpi"]["native"] == "monthly" and rows["cpi"]["conversion"] == "period-end"
    assert rows["gdp"]["conversion"] == "none"
    assert (rows["cpi"]["first"], rows["cpi"]["last"], rows["cpi"]["observations"]) == (
        "2020Q1",
        "2020Q2",
        2,
    )
    assert rows["gdp"]["found"] == "pass" and rows["absent"]["found"] == "fail"


def test_history_chart_holds_every_target(result):
    fig = result.artefacts["F1.1"]
    assert fig.x == ("2020Q1", "2020Q2")
    assert [(layer.label, layer.y) for layer in fig.layers] == [("gdp", (10.0, 20.0))]
    assert fig.y_unit == "EUR bn"


def test_gate_table_and_finding(result):
    assert [r[0] for r in result.artefacts["T1.2"].rows] == ["found", "history"]
    finding = result.artefacts["S1.1"]
    assert finding == result.finding
    assert finding.text == (
        "2 of 3 configured series were found and put on a quarterly calendar from 2020Q1 "
        "to 2020Q2. The targets and drivers overlap on 2 quarters, against a minimum of 2."
    )


def test_identifiers_are_fixed_by_declaration(result):
    assert list(result.artefacts) == ["T1.1", "T1.2", "F1.1", "S1.1"]
    assert ingest.IDS.table("inventory") == "T1.1"
    assert ingest.IDS.figure("history") == "F1.1"


def test_no_target_means_no_history_chart():
    r = run([{"name": "gdp"}], {"q": quarterly([1.0])})
    assert "F1.1" not in r.artefacts


def test_result_round_trips_without_the_dataset(result):
    d = result.to_dict()
    assert "dataset" not in d and d["fingerprint"] == result.dataset.fingerprint()
    back = ingest.Result.from_dict(d)
    assert back.dataset is None
    assert back.to_dict() == d


def test_run_is_pure(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run([{"name": "gdp"}], {"q": quarterly([1.0])})
    assert capsys.readouterr() == ("", "")
    assert list(tmp_path.iterdir()) == []


def test_ingest_is_registered_and_on_the_top_level():
    from dazflow.steps import registry

    assert registry.get("ingest") is ingest is dz.ingest
    assert isinstance(
        dz.ingest.run(
            config([{"name": "x"}], inputs=("q",)), data={"q": quarterly([1.0], name="x")}
        ),
        dz.StepResult,
    )


def test_data_must_match_the_inputs():
    with pytest.raises(IngestError, match="zz"):
        ingest.run(config([{"name": "gdp"}], inputs=("q",)), data={"q": quarterly([1.0]), "zz": 1})
    cfg = config([{"name": "gdp"}], inputs=())
    cfg["inputs"] = [{"name": "q", "path": "data.csv"}]
    with pytest.raises(IngestError, match="both"):
        ingest.run(cfg, data={"q": quarterly([1.0])})


def test_a_boolean_series_is_refused():
    frame = quarterly([True, False])
    with pytest.raises(IngestError, match="not numeric"):
        run([{"name": "gdp"}], {"q": frame})


def test_no_series_found_gives_an_empty_dataset_and_a_failed_gate():
    r = run([{"name": "absent"}], {"q": quarterly([1.0])})
    assert len(r.dataset.frame) == 0 and r.dataset.columns == ()
    assert not r.gate["found"].passed and not r.gate["history"].passed
    assert "from no period to no period" in r.finding.text
