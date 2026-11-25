"""Replay's comparison: floats within a relative tolerance, everything else exactly (design D43).

Oracle: `math.isclose`'s definition, |a - b| <= max(rtol max(|a|, |b|), atol),
evaluated by hand at values just inside and just outside the tolerance,
and the JSON values themselves for everything that is not a float.
"""

import pytest

from dazflow.pipeline import ATOL, RTOL, Difference, compare


def test_the_default_tolerance_is_the_designs():
    assert (RTOL, ATOL) == (1e-9, 1e-12)


@pytest.mark.parametrize("scale", [1.0, 0.01, 123456.0, -0.0371])
def test_a_float_just_inside_the_relative_tolerance_matches(scale):
    assert compare(scale, scale * (1 + 0.5 * RTOL)) == []


@pytest.mark.parametrize("scale", [1.0, 0.01, 123456.0, -0.0371])
def test_a_float_just_outside_the_relative_tolerance_is_a_difference(scale):
    found = scale * (1 + 2 * RTOL)
    assert compare(scale, found, path="x") == [Difference(path="x", recorded=scale, found=found)]


def test_below_atol_over_rtol_the_absolute_floor_is_the_wider_one():
    # At |x| = ATOL / RTOL = 1e-3 the two bounds meet; below it ATOL governs.
    small = 1e-6
    assert compare(small, small * (1 + 2 * RTOL)) == []
    assert compare(small, small + 0.5 * ATOL) == []
    assert compare(small, small + 2 * ATOL) != []


def test_near_zero_the_absolute_tolerance_applies():
    assert compare(0.0, 0.5 * ATOL) == []
    assert compare(0.0, -0.5 * ATOL) == []
    assert compare(0.0, 2 * ATOL) != []
    assert compare(1e-17, 3e-17) == []  # rounding noise of a sum that is zero


def test_the_tolerance_is_an_argument():
    assert compare(1.0, 1.001) != []
    assert compare(1.0, 1.001, rtol=1e-2) == []
    assert compare(0.0, 1e-6, atol=1e-5) == []


def test_an_int_and_a_float_compare_as_numbers():
    assert compare(1, 1.0) == []
    assert compare(2, 2.0 * (1 + 2 * RTOL)) != []


@pytest.mark.parametrize(
    ("recorded", "found"),
    [
        (3, 4),  # integers: counts, sizes, positions
        (10**17, 10**17 + 1),  # past a float's 53 bits, still exact
        ("ols:gdp", "ols:gdp+rate"),  # choices and ids
        ("pass", "fail"),  # statuses
        (True, False),
        (True, 1),  # a boolean is not a number
        (None, 0.0),
        ("NaN", 0.0),  # non-finite floats are stored as text
        ("Infinity", "-Infinity"),
    ],
)
def test_everything_but_floats_must_match_exactly(recorded, found):
    assert compare(recorded, found, path="v") == [
        Difference(path="v", recorded=recorded, found=found)
    ]


def test_equal_text_and_non_finite_values_match():
    assert compare("NaN", "NaN") == []
    assert compare({"a": [1, "x", None, True]}, {"a": [1, "x", None, True]}) == []


def test_paths_name_keys_and_positions():
    recorded = {"ranking": [{"id": "ols", "mase": 0.9}, {"id": "mean", "mase": 1.0}]}
    found = {"ranking": [{"id": "ols", "mase": 0.9}, {"id": "naive", "mase": 1.0}]}
    assert compare(recorded, found, path="results.select") == [
        Difference(path="results.select.ranking[1].id", recorded="mean", found="naive")
    ]


def test_a_missing_or_extra_key_is_a_difference():
    diffs = compare({"a": 1, "b": 2}, {"a": 1, "c": 3}, path="r")
    assert [(d.path, d.recorded, d.found, d.note) for d in diffs] == [
        ("r.b", 2, None, "missing on replay"),
        ("r.c", None, 3, "not in the record"),
    ]


def test_lists_of_another_length_differ_and_their_common_part_is_compared():
    diffs = compare([1, 2, 3], [1, 5], path="r")
    assert [(d.path, d.recorded, d.found) for d in diffs] == [("r", 3, 2), ("r[1]", 2, 5)]
    assert diffs[0].note == "lengths differ"


def test_a_difference_reads_as_its_path_and_both_values():
    d = Difference(path="results.diagnose.scale", recorded=0.25, found=0.5)
    assert str(d) == "results.diagnose.scale: recorded 0.25, found 0.5"
    assert "missing" in str(Difference(path="p", recorded=1, found=None, note="missing on replay"))
