"""Shared test settings: the `extra` marker skips a test whose optional distribution is absent.

The skip asks whether the distribution is *installed* (its metadata), not
whether it imports, so an import that breaks inside an installed extra fails
the test instead of skipping it. The quick and matrix jobs install every
extra, so these tests run there; the core job installs none, so they skip.
"""

import importlib.metadata
import os
import shutil
from pathlib import Path

import pytest
from example_helpers import EXAMPLE
from hypothesis import HealthCheck, settings

# Hypothesis profiles (design D46), chosen by HYPOTHESIS_PROFILE: `quick` (the
# default, and the quick CI job), `full` (the full profile's jobs) and
# `mutation` (the scheduled mutation run, where a test must give the same
# verdict on every run). A test that sets its own max_examples keeps it.
settings.register_profile("quick", max_examples=100, deadline=None)
settings.register_profile(
    "full", max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("mutation", max_examples=100, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


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


# --- the example project (examples/loss-rate) ------------------------------------


@pytest.fixture
def example(tmp_path) -> Path:
    """A fresh copy of the example project, safe to change."""
    dest = tmp_path / "loss-rate"
    shutil.copytree(EXAMPLE, dest)
    return dest


@pytest.fixture(scope="session")
def ran_example(tmp_path_factory) -> Path:
    """A copy of the example project after `dazflow run`. Read it; do not change it."""
    from dazflow import project

    dest = tmp_path_factory.mktemp("ran") / "loss-rate"
    shutil.copytree(EXAMPLE, dest)
    proj = project.load(dest)
    proj.write(proj.run())
    return dest
