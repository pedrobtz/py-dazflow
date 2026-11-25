"""The public contract suite every component passes (design §13).

Plugin authors subclass `ComponentContract` in their own tests, naming the
registry and the key; pytest then runs the contract tests on that component::

    from dazflow.testing import ComponentContract
    import dazflow_theta

    class TestTheta(ComponentContract):
        registry = dazflow_theta.models
        key = "theta"

The suite grows with the component kinds: each kind's contract (model forms
from Stage 7, transforms, readers, ...) subclasses `ComponentContract` and
adds the tests its protocol promises.
"""

from __future__ import annotations

from typing import Any, ClassVar

from dazflow.core.config import Config
from dazflow.core.registry import Registry, _protocol_members

__all__ = ["ComponentContract"]


class ComponentContract:
    """Tests every registered component passes, whatever its kind.

    Set `registry` and `key` in a subclass whose name starts with ``Test``.
    """

    registry: ClassVar[Registry]
    key: ClassVar[str]

    def component(self) -> Any:
        """The component under test, looked up the way configuration looks it up."""
        return self.registry.get(self.key)

    def test_key_is_registered(self) -> None:
        keys = self.registry.keys()
        assert self.key in keys, (
            f"{self.key!r} is not registered; known keys: {', '.join(keys) or 'none'}"
        )

    def test_lookup_returns_the_registered_component(self) -> None:
        assert self.key in self.registry, f"{self.key!r} is not registered"
        assert self.registry.get(self.key) is self.registry[self.key]

    def test_component_follows_the_protocol(self) -> None:
        protocol = self.registry.protocol
        if protocol is None:
            return
        assert self.key in self.registry, f"{self.key!r} is not registered"
        obj = self.component()
        missing = [m for m in _protocol_members(protocol) if not hasattr(obj, m)]
        assert not missing, f"{obj!r} lacks {protocol.__name__} members: {', '.join(missing)}"

    def test_config_is_a_dazflow_config(self) -> None:
        assert self.key in self.registry, f"{self.key!r} is not registered"
        config = getattr(self.component(), "Config", None)
        if config is None:
            return
        assert isinstance(config, type) and issubclass(config, Config), (
            f"{self.key!r}.Config is {config!r}, not a dazflow.Config subclass"
        )
        schema = config.json_schema()
        assert schema.get("type") == "object", (
            f"{self.key!r}.Config does not export an object schema"
        )
