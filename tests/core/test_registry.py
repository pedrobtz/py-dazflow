"""Registry: decorator registration, `dazflow.plugins` entry points, lookup (design §6).

The plugin test installs a real distribution: a `*.dist-info` directory with
`METADATA` and `entry_points.txt`, written to a temporary directory that is
put on `sys.path`. `importlib.metadata` finds it the way it finds any
installed package; nothing is monkeypatched in the metadata machinery.
"""

import importlib.metadata
import sys
import textwrap
from typing import Protocol, runtime_checkable

import pytest

import dazflow as dz
from dazflow import (
    DuplicateKeyError,
    PluginError,
    Registry,
    RegistryError,
    UnknownKeyError,
)


@runtime_checkable
class Widget(Protocol):
    def spin(self, x: float) -> float: ...


def test_decorator_registers_and_returns_the_class_unchanged():
    widgets = Registry("widget", protocol=Widget)

    @widgets.register("double")
    class Double:
        def spin(self, x):
            return 2 * x

    assert widgets.get("double") is Double
    assert widgets["double"] is Double
    assert "double" in widgets
    assert widgets.keys() == ("double",)
    assert Double().spin(2) == 4


def test_unknown_key_raises_the_classed_error_listing_known_keys():
    widgets = Registry("widget")
    widgets.add("zeta", object())
    widgets.add("alpha", object())

    with pytest.raises(UnknownKeyError) as info:
        widgets.get("gamma")

    err = info.value
    assert isinstance(err, RegistryError)
    assert isinstance(err, dz.DazflowError)
    assert err.key == "gamma"
    assert err.known == ("alpha", "zeta")
    assert "'gamma'" in str(err)
    assert "alpha, zeta" in str(err)
    assert "change:" in str(err)


def test_unknown_key_in_an_empty_registry_says_none_are_known():
    with pytest.raises(UnknownKeyError, match="none"):
        Registry("widget")["x"]


def test_a_key_is_registered_once():
    widgets = Registry("widget")
    widgets.add("a", 1)

    with pytest.raises(DuplicateKeyError, match="'a'"):
        widgets.add("a", 2)
    assert widgets["a"] == 1


@pytest.mark.parametrize("key", ["", 3, None])
def test_keys_are_non_empty_strings(key):
    with pytest.raises(RegistryError, match="key"):
        Registry("widget").add(key, object())


def test_a_component_missing_protocol_members_is_refused():
    widgets = Registry("widget", protocol=Widget)

    with pytest.raises(RegistryError, match="spin"):

        @widgets.register("broken")
        class Broken:
            def turn(self, x):
                return x


# --- entry points ---------------------------------------------------------

GROUP = "dazflow.plugins"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A directory on sys.path into which a test writes distributions."""
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [m for m in sys.modules if m.startswith(("dzt_host", "dzt_plugin"))]:
        del sys.modules[name]
    importlib.invalidate_caches()


def install(site, dist, entry_points, modules):
    info = site / f"{dist}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n", encoding="utf-8"
    )
    lines = "".join(f"{name} = {value}\n" for name, value in entry_points.items())
    (info / "entry_points.txt").write_text(f"[{GROUP}]\n{lines}", encoding="utf-8")
    for name, source in modules.items():
        (site / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    importlib.invalidate_caches()


HOST = """
    from dazflow import Registry
    widgets = Registry("widget")
"""

PLUGIN = """
    import dzt_host

    @dzt_host.widgets.register("fancy")
    class Fancy:
        def spin(self, x):
            return x + 1
"""


def test_plugin_from_an_installed_distribution_is_found_by_key(site):
    install(site, "dzt-plugin", {"fancy": "dzt_plugin"}, {"dzt_host": HOST, "dzt_plugin": PLUGIN})
    assert [
        ep.name for ep in importlib.metadata.entry_points(group=GROUP) if ep.name == "fancy"
    ] == ["fancy"]
    import dzt_host

    assert "dzt_plugin" not in sys.modules  # loaded on first use, not before

    fancy = dzt_host.widgets.get("fancy")

    assert fancy.__name__ == "Fancy"
    assert fancy().spin(1) == 2
    assert "fancy" in dzt_host.widgets


def test_entry_point_naming_an_attribute_loads_its_module(site):
    plugin = PLUGIN + "\n    hook = Fancy\n"
    install(
        site, "dzt-plugin", {"fancy": "dzt_plugin:hook"}, {"dzt_host": HOST, "dzt_plugin": plugin}
    )
    import dzt_host

    assert dzt_host.widgets["fancy"].__name__ == "Fancy"


def test_a_plugin_that_fails_to_load_raises_a_plugin_error_naming_it(site):
    install(
        site,
        "dzt-broken",
        {"broken": "dzt_plugin_broken"},
        {"dzt_host": HOST, "dzt_plugin_broken": "raise RuntimeError('boom')\n"},
    )
    import dzt_host

    with pytest.raises(PluginError, match="dzt_plugin_broken") as info:
        dzt_host.widgets.get("anything")

    assert "dzt-broken" in str(info.value)
    assert "boom" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_unknown_key_after_plugins_load_lists_plugin_keys(site):
    install(site, "dzt-plugin", {"fancy": "dzt_plugin"}, {"dzt_host": HOST, "dzt_plugin": PLUGIN})
    import dzt_host

    with pytest.raises(UnknownKeyError, match="fancy"):
        dzt_host.widgets.get("plain")


def test_registry_describes_itself():
    widgets = Registry("widget")
    widgets.add("b", 1)
    widgets.add("a", 2)

    assert widgets.kind == "widget"
    assert widgets.protocol is None
    assert Registry("widget", protocol=Widget).protocol is Widget
    assert list(widgets) == ["a", "b"]
    assert len(widgets) == 2
    assert repr(widgets) == "Registry('widget', keys=['b', 'a'])"
