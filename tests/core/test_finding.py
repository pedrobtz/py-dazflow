"""Finding: template text filled with numbers, nothing generated (design §9)."""

import math

import pytest

from dazflow import Finding, FindingError


def test_finding_fills_its_template_with_its_values():
    finding = Finding(
        template="{{ name }} rose by {{ '%.1f' | format(change) }} pp in {{ n }} quarters.",
        values={"name": "Unemployment", "change": 1.25, "n": 4},
    )

    assert finding.text == "Unemployment rose by 1.2 pp in 4 quarters."


def test_text_is_exactly_the_template_when_it_has_no_fields():
    assert Finding(template="No check failed.").text == "No check failed."


def test_a_field_without_a_value_is_refused():
    with pytest.raises(FindingError, match="missing") as info:
        Finding(template="{{ a }} and {{ b }}", values={"a": 1})

    assert "change:" in str(info.value)


def test_a_template_that_does_not_parse_is_refused():
    with pytest.raises(FindingError, match="template"):
        Finding(template="{{ a ", values={"a": 1})


@pytest.mark.parametrize("value", [math.nan, math.inf, [1, 2], {"a": 1}, object()])
def test_values_are_finite_numbers_strings_or_booleans(value):
    with pytest.raises(FindingError, match="x"):
        Finding(template="{{ x }}", values={"x": value})


def test_templates_cannot_reach_python_internals():
    with pytest.raises(FindingError):
        Finding(template="{{ x.__class__.__mro__ }}", values={"x": "s"})


def test_findings_are_immutable_and_equal_by_content():
    a = Finding(template="{{ x }}", values={"x": 1})

    assert a == Finding(template="{{ x }}", values={"x": 1})
    with pytest.raises(FindingError):
        a.template = "other"
