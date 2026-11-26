"""The package imports and reports the version it was installed with."""

import importlib
import importlib.metadata


def test_package_imports():
    module = importlib.import_module("dazflow")

    assert module.__name__ == "dazflow"


def test_version_is_the_installed_metadata_version():
    import dazflow

    assert dazflow.__version__ == importlib.metadata.version("dazflow")


def test_the_package_is_marked_as_typed():
    """PEP 561: type checkers read the annotations only of a package carrying py.typed."""
    from importlib.resources import files

    assert (files("dazflow") / "py.typed").is_file()
    classifiers = importlib.metadata.metadata("dazflow").get_all("Classifier")
    assert "Typing :: Typed" in classifiers
