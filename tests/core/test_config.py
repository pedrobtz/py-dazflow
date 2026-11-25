"""Config: YAML round trip and JSON Schema export (design §8)."""

from pathlib import Path
from typing import Literal

import jsonschema
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import Field

from dazflow import Config, ConfigError, Override

EXAMPLE = Path(__file__).parent / "data" / "example-config.yaml"


class Window(Config):
    start: str
    end: str | None = None


class ExampleConfig(Config):
    """A configuration shaped like a step's: nested models, lists, literals, overrides."""

    target: str
    drivers: list[str] = Field(default_factory=list)
    transform: Literal["none", "log", "logit", "diff"] = "none"
    max_lag: int = 4
    alpha: float = 0.05
    seasonal: bool = False
    window: Window | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    overrides: list[Override] = Field(default_factory=list)


def test_example_yaml_round_trips_byte_for_byte():
    text = EXAMPLE.read_text(encoding="utf-8")

    cfg = ExampleConfig.from_yaml(text)

    assert cfg.to_yaml() == text


def test_example_loads_from_a_file_and_dumps_to_one(tmp_path):
    cfg = ExampleConfig.load(EXAMPLE)
    out = tmp_path / "out.yaml"

    cfg.dump(out)

    assert out.read_text(encoding="utf-8") == EXAMPLE.read_text(encoding="utf-8")
    assert ExampleConfig.load(out) == cfg


def test_defaults_are_filled_in_when_dumped():
    cfg = ExampleConfig.from_yaml("target: loss\n")

    assert yaml.safe_load(cfg.to_yaml()) == {
        "target": "loss",
        "drivers": [],
        "transform": "none",
        "max_lag": 4,
        "alpha": 0.05,
        "seasonal": False,
        "window": None,
        "weights": {},
        "overrides": [],
    }


def test_exported_schema_validates_the_example():
    schema = ExampleConfig.json_schema()
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(data)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize(
    "change",
    [
        {"max_lag": "four"},
        {"transform": "sqrt"},
        {"typo_field": 1},
        {"overrides": [{"check": "adf"}]},
    ],
)
def test_schema_and_model_both_refuse_an_invalid_config(change):
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")) | change

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, ExampleConfig.json_schema())
    with pytest.raises(ConfigError):
        ExampleConfig.from_yaml(yaml.safe_dump(data))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("target: [unclosed\n", "YAML"),
        ("- a\n- b\n", "mapping"),
        ("drivers: [x]\n", "target"),
        ("target: loss\nalpha: .inf\n", "alpha"),
        ("target: loss\nmax_lag: 2\nmax_lag: 3\n", "max_lag"),
    ],
)
def test_bad_yaml_raises_a_config_error_naming_the_problem(text, fragment):
    with pytest.raises(ConfigError, match=r"(?s)observed:.*change:") as info:
        ExampleConfig.from_yaml(text)

    assert fragment in str(info.value)


def test_load_of_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"missing\.yaml"):
        ExampleConfig.load(tmp_path / "missing.yaml")


def test_configs_are_immutable():
    cfg = ExampleConfig(target="loss")

    with pytest.raises(ConfigError):
        cfg.target = "other"


names = st.text(alphabet=st.characters(codec="utf-8", exclude_categories=["Cs", "Cc"]), max_size=12)


@settings(max_examples=60, deadline=None)
@given(
    target=names,
    drivers=st.lists(names, max_size=4),
    max_lag=st.integers(min_value=-(2**63), max_value=2**63),
    alpha=st.floats(allow_nan=False, allow_infinity=False),
    weights=st.dictionaries(names, st.floats(allow_nan=False, allow_infinity=False), max_size=3),
    start=st.none() | names,
)
def test_any_config_round_trips_through_yaml(target, drivers, max_lag, alpha, weights, start):
    cfg = ExampleConfig(
        target=target,
        drivers=drivers,
        max_lag=max_lag,
        alpha=alpha,
        weights=weights,
        window=None if start is None else Window(start=start),
    )

    text = cfg.to_yaml()

    assert ExampleConfig.from_yaml(text) == cfg
    assert ExampleConfig.from_yaml(text).to_yaml() == text


def test_empty_yaml_gives_the_defaults():
    assert Window.model_fields["end"].default is None
    assert ExampleConfig.from_yaml("target: x\n") == ExampleConfig.from_yaml("target: x\n\n# end\n")

    class Defaults(Config):
        level: float = 0.95

    assert Defaults.from_yaml("") == Defaults()
