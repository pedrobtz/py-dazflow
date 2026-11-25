"""The package imports and reports the version it was installed with."""

import importlib
import importlib.metadata


def test_package_imports():
    module = importlib.import_module("dazflow")

    assert module.__name__ == "dazflow"


def test_version_is_the_installed_metadata_version():
    import dazflow

    assert dazflow.__version__ == importlib.metadata.version("dazflow")
