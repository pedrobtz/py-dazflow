"""StepResult.to_dict(): JSON-safe and round-tripping (design §5.2)."""

import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import Field

from dazflow import Check, Config, Finding, Gate, Override, ResultError, StepResult


class DemoConfig(Config):
    alpha: float = 0.05
    lags: list[int] = Field(default_factory=lambda: [1, 2])


class DemoResult(StepResult):
    """What a step's result subclass looks like: typed values plus the shared fields."""

    config: DemoConfig
    mean: float
    labels: tuple[str, ...] = ()


def build(observed=0.2, mean=1.25):
    checks = [
        Check(name="adf", observed=observed, threshold=0.05, op="<"),
        Check(name="r2", observed=0.9, threshold=0.5, op=">=", unit="share"),
    ]
    gate = Gate.evaluate(checks, overrides=[Override(check="adf", reason="Short sample.")])
    finding = Finding(
        template="The mean is {{ mean }} and {{ n }} of {{ total }} checks pass.",
        values={"mean": mean, "n": 1, "total": 2},
    )
    return DemoResult(
        gate=gate,
        finding=finding,
        artefacts={"S1.1": finding},
        config=DemoConfig(alpha=0.1),
        mean=mean,
        labels=("a", "é"),
    )


def strict_json(data):
    return json.loads(json.dumps(data, allow_nan=False))


def test_to_dict_is_strict_json_and_round_trips():
    result = build()

    data = result.to_dict()
    back = DemoResult.from_dict(strict_json(data))

    assert back == result
    assert back.to_dict() == data


def test_to_dict_carries_the_shared_fields_and_the_config_as_run():
    data = build().to_dict()

    assert data["config"] == {"alpha": 0.1, "lags": [1, 2]}
    assert data["gate"]["passed"] is True
    assert [c["status"] for c in data["gate"]["checks"]] == ["overridden", "pass"]
    assert data["gate"]["checks"][0]["override"] == {"check": "adf", "reason": "Short sample."}
    assert data["finding"]["text"] == "The mean is 1.25 and 1 of 2 checks pass."
    assert data["artefacts"]["S1.1"]["text"] == data["finding"]["text"]
    assert data["mean"] == 1.25
    assert data["labels"] == ["a", "é"]


def test_base_result_without_a_config_round_trips():
    result = StepResult(gate=Gate(checks=[]))

    assert StepResult.from_dict(strict_json(result.to_dict())) == result


@pytest.mark.parametrize("observed", [math.nan, math.inf, -math.inf])
def test_non_finite_observed_values_survive_as_strict_json(observed):
    result = build(observed=observed)

    back = DemoResult.from_dict(strict_json(result.to_dict()))

    got = back.gate["adf"].observed
    assert (math.isnan(got) and math.isnan(observed)) or got == observed
    assert back.to_dict() == result.to_dict()


@settings(max_examples=60, deadline=None)
@given(
    observed=st.floats(),
    mean=st.floats(allow_nan=False, allow_infinity=False),
)
def test_any_result_round_trips(observed, mean):
    result = build(observed=observed, mean=mean)

    data = result.to_dict()

    assert DemoResult.from_dict(strict_json(data)).to_dict() == data


def test_from_dict_of_bad_data_is_a_result_error():
    data = build().to_dict()
    data["mean"] = "high"

    with pytest.raises(ResultError, match="mean"):
        DemoResult.from_dict(data)


def test_base_result_records_a_subclass_config_in_full():
    result = StepResult(config=DemoConfig(alpha=0.1))

    assert result.to_dict()["config"] == {"alpha": 0.1, "lags": [1, 2]}
