"""Check and Gate: threshold arithmetic and overrides (design §5.2, §7)."""

import math
import operator

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dazflow import Check, Gate, GateError, Override

THRESHOLD = 0.05
AT = THRESHOLD
BELOW = float(np.nextafter(THRESHOLD, -np.inf))
ABOVE = float(np.nextafter(THRESHOLD, np.inf))

# The comparison each operator states, written out: (observed, passes?)
STRADDLE = {
    "<": [(BELOW, True), (AT, False), (ABOVE, False)],
    "<=": [(BELOW, True), (AT, True), (ABOVE, False)],
    ">": [(BELOW, False), (AT, False), (ABOVE, True)],
    ">=": [(BELOW, False), (AT, True), (ABOVE, True)],
}
CASES = [(op, obs, ok) for op, rows in STRADDLE.items() for obs, ok in rows]


@pytest.mark.parametrize(("op", "observed", "passes"), CASES)
def test_check_straddles_its_threshold(op, observed, passes):
    check = Check(name="p_value", observed=observed, threshold=THRESHOLD, op=op)

    assert check.passed is passes
    assert check.status == ("pass" if passes else "fail")
    assert Gate(checks=[check]).passed is passes


PY = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


@given(
    op=st.sampled_from(list(PY)),
    observed=st.floats(allow_nan=False),
    threshold=st.floats(allow_nan=False, allow_infinity=False),
)
def test_check_agrees_with_the_python_comparison(op, observed, threshold):
    check = Check(name="c", observed=observed, threshold=threshold, op=op)

    assert check.passed is PY[op](observed, threshold)


@pytest.mark.parametrize("op", list(PY))
def test_a_missing_observed_value_fails(op):
    check = Check(name="c", observed=math.nan, threshold=1.0, op=op)

    assert check.passed is False
    assert check.status == "fail"


def test_threshold_must_be_finite():
    with pytest.raises(GateError, match="threshold"):
        Check(name="c", observed=1.0, threshold=math.inf, op="<")


def test_unknown_operator_is_refused():
    with pytest.raises(GateError, match="op"):
        Check(name="c", observed=1.0, threshold=1.0, op="==")


def test_gate_passes_only_when_every_check_passes():
    ok = Check(name="a", observed=0.0, threshold=1.0, op="<")
    bad = Check(name="b", observed=2.0, threshold=1.0, op="<")

    assert Gate(checks=[ok]).passed
    assert not Gate(checks=[ok, bad]).passed
    assert Gate(checks=[ok, bad]).failed == (bad,)
    assert Gate(checks=[]).passed


def test_gate_checks_are_found_by_name_and_names_are_unique():
    a = Check(name="a", observed=0.0, threshold=1.0, op="<")

    assert Gate(checks=[a])["a"] == a
    with pytest.raises(GateError, match="'z'"):
        Gate(checks=[a])["z"]
    with pytest.raises(GateError, match="unique"):
        Gate(checks=[a, a])


# --- overrides ------------------------------------------------------------


def failing():
    return Check(name="adf", observed=0.2, threshold=0.05, op="<")


def test_override_with_a_reason_lets_a_failed_check_through():
    gate = Gate.evaluate(
        [failing()], overrides=[Override(check="adf", reason="Short sample; see note 4.")]
    )

    check = gate["adf"]
    assert check.passed is False
    assert check.status == "overridden"
    assert check.override.reason == "Short sample; see note 4."
    assert gate.passed is True
    assert gate.failed == ()
    assert gate.overridden == (check,)


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_override_without_a_reason_is_refused(reason):
    with pytest.raises(GateError, match="reason") as info:
        Override(check="adf", reason=reason)

    assert "change:" in str(info.value)


def test_override_with_no_reason_field_is_refused():
    with pytest.raises(GateError, match="reason"):
        Override(check="adf")


def test_override_from_config_data_without_a_reason_is_refused():
    with pytest.raises(GateError, match="reason"):
        Gate.evaluate([failing()], overrides=[{"check": "adf", "reason": ""}])


def test_override_of_an_unknown_check_is_refused_naming_the_checks():
    with pytest.raises(GateError, match="adf") as info:
        Gate.evaluate([failing()], overrides=[Override(check="adff", reason="typo")])

    assert "adff" in str(info.value)


def test_override_of_a_passing_check_is_kept_and_changes_nothing():
    ok = Check(name="adf", observed=0.01, threshold=0.05, op="<")

    gate = Gate.evaluate([ok], overrides=[Override(check="adf", reason="belt and braces")])

    assert gate["adf"].status == "pass"
    assert gate["adf"].override.reason == "belt and braces"
    assert gate.overridden == ()


def test_check_override_must_name_that_check():
    with pytest.raises(GateError, match="other"):
        Check(
            name="adf",
            observed=0.2,
            threshold=0.05,
            op="<",
            override=Override(check="other", reason="x"),
        )


def test_objects_are_immutable():
    check = failing()

    with pytest.raises(GateError):
        check.observed = 0.0
