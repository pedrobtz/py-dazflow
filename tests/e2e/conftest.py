"""End-to-end fixtures: the example project run from ingest to report, and its replay."""

import shutil

import pytest


@pytest.fixture
def ran_copy(ran_example, tmp_path):
    """A copy of the example project after `dazflow run`, safe to change."""
    dest = tmp_path / "loss-rate"
    shutil.copytree(ran_example, dest)
    return dest


@pytest.fixture(scope="session")
def replayed(ran_example):
    """The example's record replayed: every step rerun from it and compared."""
    from dazflow import project

    return project.load(ran_example).replay()
