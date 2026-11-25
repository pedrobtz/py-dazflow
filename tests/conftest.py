"""Shared test settings: the `extra` marker skips a test whose optional distribution is absent.

The skip asks whether the distribution is *installed* (its metadata), not
whether it imports, so an import that breaks inside an installed extra fails
the test instead of skipping it. The quick and matrix jobs install every
extra, so these tests run there; the core job installs none, so they skip.
"""

import importlib.metadata

import pytest


def installed(distribution: str) -> bool:
    """True when `distribution` is installed in this environment."""
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "extra(*distributions): skip unless these optional distributions are installed"
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers("extra"):
        missing = [d for d in marker.args if not installed(d)]
        if missing:
            pytest.skip(f"needs the dazflow[io] extra; not installed: {', '.join(missing)}")


@pytest.fixture
def distribution_installed():
    return installed
