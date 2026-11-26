"""Registry: components by key, from decorators and `dazflow.plugins` entry points (design §6)."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from dazflow.core.errors import DuplicateKeyError, PluginError, RegistryError, UnknownKeyError

__all__ = ["ENTRY_POINT_GROUP", "Registry"]

ENTRY_POINT_GROUP = "dazflow.plugins"

T = TypeVar("T")


def _protocol_members(protocol: type) -> tuple[str, ...]:
    members = getattr(protocol, "__protocol_attrs__", None)  # Python 3.12+
    return tuple(sorted(members)) if members is not None else ()


class Registry:
    """Components of one kind, looked up by the key configuration names them by.

    Components are added with the `register` decorator, or by a third-party
    package through the ``dazflow.plugins`` entry-point group. Each entry
    point names a module (``name = package.module``, or ``package.module:attr``);
    loading it imports the module, whose decorators register its components.
    Entry points are loaded on the registry's first lookup, once.

    Parameters
    ----------
    kind
        What the registry holds, for messages: ``"model"``, ``"reader"``.
    protocol
        Optional `typing.Protocol` every component must have the members of.

    Examples
    --------
    >>> widgets = Registry("widget")
    >>> @widgets.register("plain")
    ... class Plain: ...
    >>> widgets["plain"] is Plain
    True
    """

    def __init__(self, kind: str, protocol: type | None = None) -> None:
        self._kind = kind
        self._protocol = protocol
        self._items: dict[str, Any] = {}
        self._plugins_loaded = False

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def protocol(self) -> type | None:
        return self._protocol

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator: register the decorated class or function under `key`."""

        def decorate(obj: T) -> T:
            self.add(key, obj)
            return obj

        return decorate

    def add(self, key: str, obj: Any) -> None:
        """Register `obj` under `key`; a taken key is refused."""
        if not isinstance(key, str) or not key:
            raise RegistryError(
                f"a {self._kind} key must be a non-empty string",
                observed=f"key {key!r}",
                change="Register the component under a non-empty string key.",
            )
        if key in self._items:
            raise DuplicateKeyError(
                f"the {self._kind} key {key!r} is already registered",
                observed=f"{key!r} holds {self._items[key]!r}; refused {obj!r}",
                change="Register the component under another key.",
            )
        missing = self._missing_members(obj)
        if missing:
            raise RegistryError(
                f"{obj!r} does not follow the {self._kind} protocol",
                observed=f"missing members: {', '.join(missing)}",
                change=f"Give the component the members of {self._protocol.__name__}.",
            )
        self._items[key] = obj

    def _missing_members(self, obj: Any) -> tuple[str, ...]:
        if self._protocol is None:
            return ()
        return tuple(m for m in _protocol_members(self._protocol) if not hasattr(obj, m))

    def get(self, key: str) -> Any:
        """The component registered under `key`; `UnknownKeyError` lists the known keys."""
        self._load_plugins()
        try:
            return self._items[key]
        except (KeyError, TypeError):
            raise UnknownKeyError(self._kind, key, self.keys()) from None

    __getitem__ = get

    def keys(self) -> tuple[str, ...]:
        """Every registered key, sorted."""
        self._load_plugins()
        return tuple(sorted(self._items))

    def __contains__(self, key: object) -> bool:
        self._load_plugins()
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, keys={list(self._items)!r})"

    def _load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True  # set first: a plugin may look up while it loads
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                ep.load()
            except Exception as err:
                self._plugins_loaded = False  # the next lookup reports it again
                dist = ep.dist.name if ep.dist is not None else "unknown distribution"
                raise PluginError(
                    f"the dazflow plugin {ep.name!r} failed to load",
                    observed=f"entry point {ep.name} = {ep.value} from {dist}: {type(err).__name__}: {err}",
                    change=f"Fix or uninstall {dist}; its import raised the error shown.",
                ) from err
