"""The skip helper behind the `extra` marker: a wrong one would turn failures into skips."""

import sys

import pytest


def test_installed_reads_distribution_metadata(distribution_installed):
    assert distribution_installed("pytest") is True
    assert distribution_installed("dazflow") is True
    assert distribution_installed("no-such-distribution-dazflow-test") is False


def test_installed_ignores_whether_the_module_imports(distribution_installed, monkeypatch):
    # A hidden (or broken) module must not look uninstalled, or the tests guarded by
    # the marker would skip on exactly the failure they exist to catch.
    monkeypatch.setitem(sys.modules, "pytest_randomly", None)
    with pytest.raises(ImportError):
        __import__("pytest_randomly")
    assert distribution_installed("pytest-randomly") is True
