"""dazflow.testing: the component contract suite passes good components and fails bad ones."""

from typing import ClassVar, Protocol, runtime_checkable

import pytest

from dazflow import Config, Registry
from dazflow.testing import ComponentContract


@runtime_checkable
class Widget(Protocol):
    def spin(self, x: float) -> float: ...


widgets = Registry("widget", protocol=Widget)


class SpinConfig(Config):
    speed: float = 1.0


@widgets.register("good")
class Good:
    Config = SpinConfig

    def spin(self, x):
        return x


class TestGoodWidget(ComponentContract):
    """How a plugin author uses the suite: subclass it and name the key."""

    registry: ClassVar[Registry] = widgets
    key: ClassVar[str] = "good"


def contract_tests():
    return sorted(name for name in dir(ComponentContract) if name.startswith("test_"))


def run_contract(registry, key):
    class Probe(ComponentContract):
        pass

    Probe.registry = registry
    Probe.key = key
    failures = []
    for name in contract_tests():
        try:
            getattr(Probe(), name)()
        except AssertionError as err:
            failures.append((name, str(err)))
    return failures


def test_the_suite_has_tests():
    assert contract_tests()


def test_a_good_component_passes_every_contract_test():
    assert run_contract(widgets, "good") == []


def test_a_key_that_is_not_registered_fails():
    failures = dict(run_contract(widgets, "missing"))

    assert "test_key_is_registered" in failures
    assert "missing" in failures["test_key_is_registered"]


def test_a_component_that_breaks_its_protocol_fails():
    # Registration refuses such a component, so the protocol is set afterwards:
    # the contract must still catch one that got in another way.
    loose = Registry("widget")
    loose.add("bad", type("Bad", (), {"turn": lambda self, x: x}))
    loose._protocol = Widget

    failures = dict(run_contract(loose, "bad"))

    assert "test_component_follows_the_protocol" in failures
    assert "spin" in failures["test_component_follows_the_protocol"]


@pytest.mark.parametrize(
    "config",
    [dict, type("NotDazflow", (), {})],
    ids=["builtin type", "plain class"],
)
def test_a_component_whose_config_is_not_a_dazflow_config_fails(config):
    reg = Registry("widget", protocol=Widget)
    reg.add("odd", type("Odd", (), {"Config": config, "spin": lambda self, x: x}))

    failures = dict(run_contract(reg, "odd"))

    assert "test_config_is_a_dazflow_config" in failures


def test_a_registry_without_a_protocol_passes_any_registered_object():
    plain = Registry("thing")
    plain.add("x", object())

    assert run_contract(plain, "x") == []
