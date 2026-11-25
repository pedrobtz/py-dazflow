"""Stable artefact identifiers: T<step>.<n>, F<step>.<n>, S<step>.<n> (design §9, D21)."""

import itertools

import pytest

from dazflow import (
    STEP_NUMBERS,
    ArtefactError,
    ArtefactIds,
    Column,
    Figure,
    Finding,
    Layer,
    ResultError,
    StepResult,
    Table,
)

IDS = ArtefactIds(
    "explore",
    tables={"summary": 1, "stationarity": 2},
    figures={"series": 1, "acf": 3},
    findings={"overview": 1},
)


def make(name):
    """Each artefact a step might make, built by name and numbered by the catalogue."""
    if name == "summary":
        return Table(
            id=IDS.table("summary"),
            caption="Summary",
            columns=[Column(name="x", label="X")],
            rows=[[1.0]],
        )
    if name == "stationarity":
        return Table(
            id=IDS.table("stationarity"),
            caption="Stationarity",
            columns=[Column(name="p", label="p-value")],
            rows=[[0.01]],
        )
    if name in ("series", "acf"):
        return Figure(
            id=IDS.figure(name),
            caption=name,
            x=[1, 2],
            layers=[Layer(kind="line", y=[1.0, 2.0])],
        )
    return Finding(template="{{ n }} series.", values={"n": 2})


NAMES = ["summary", "stationarity", "series", "acf", "overview"]


def result_in_order(order):
    artefacts = {}
    for name in order:
        obj = make(name)
        key = IDS.finding(name) if isinstance(obj, Finding) else obj.id
        artefacts[key] = obj
    return StepResult(artefacts=artefacts)


def test_identifiers_come_from_the_step_number_and_the_declared_number():
    assert STEP_NUMBERS["explore"] == 3
    assert IDS.table("stationarity") == "T3.2"
    assert IDS.figure("acf") == "F3.3"
    assert IDS.finding("overview") == "S3.1"


@pytest.mark.parametrize("order", list(itertools.permutations(NAMES))[::17])
def test_creation_order_changes_no_identifier_and_no_serialised_result(order):
    reference = result_in_order(NAMES)

    result = result_in_order(order)

    assert {k: getattr(v, "id", k) for k, v in result.artefacts.items()} == {
        "T3.1": "T3.1",
        "T3.2": "T3.2",
        "F3.1": "F3.1",
        "F3.3": "F3.3",
        "S3.1": "S3.1",
    }
    assert list(result.artefacts) == ["T3.1", "T3.2", "F3.1", "F3.3", "S3.1"]
    assert result.to_dict() == reference.to_dict()


def test_a_rerun_gives_the_same_identifiers():
    first = result_in_order(NAMES)
    again = result_in_order(NAMES)

    assert first == again
    assert list(first.artefacts) == list(again.artefacts)


def test_identifiers_sort_by_number_not_by_text():
    ids = ArtefactIds(2, tables={"a": 10, "b": 2})
    tables = [
        Table(id=ids.table(n), caption=n, columns=[Column(name="x", label="x")], rows=[])
        for n in ("a", "b")
    ]

    result = StepResult(artefacts={t.id: t for t in tables})

    assert list(result.artefacts) == ["T2.2", "T2.10"]


def test_a_result_round_trips_every_artefact_kind():
    result = result_in_order(NAMES)

    back = StepResult.from_dict(result.to_dict())

    assert back == result
    assert type(back.artefacts["T3.1"]) is Table
    assert type(back.artefacts["F3.1"]) is Figure
    assert type(back.artefacts["S3.1"]) is Finding


def test_an_unknown_name_is_refused_with_the_known_names():
    with pytest.raises(ArtefactError, match="stationarity, summary"):
        IDS.table("trend")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tables": {"a": 1, "b": 1}},
        {"tables": {"a": 0}},
        {"figures": {"a": -1}},
        {"findings": {"a": True}},
        {"tables": {"": 1}},
    ],
)
def test_a_catalogue_with_repeated_or_non_positive_numbers_is_refused(kwargs):
    with pytest.raises(ArtefactError):
        ArtefactIds("clean", **kwargs)


@pytest.mark.parametrize("step", ["nonsense", 0, -1, True, 1.5])
def test_an_unknown_step_is_refused(step):
    with pytest.raises(ArtefactError):
        ArtefactIds(step)


def test_a_key_that_is_not_the_artefacts_own_identifier_is_refused():
    table = make("summary")

    with pytest.raises(ResultError, match=r"T3\.1"):
        StepResult(artefacts={"T3.9": table})


@pytest.mark.parametrize("key", ["X1.1", "T3.1", "F3.1", "S3", "S3.1.1", "s3.1", "S03.1"])
def test_a_finding_needs_an_s_identifier(key):
    with pytest.raises(ResultError):
        StepResult(artefacts={key: make("overview")})


@pytest.mark.parametrize("bad", ["T3", "F3.1", "T3.0", "T0.1", "T3.1a", "t3.1", "T03.1"])
def test_a_table_identifier_has_the_t_step_n_form(bad):
    with pytest.raises(ArtefactError):
        Table(id=bad, caption="c", columns=[Column(name="x", label="x")], rows=[])
