"""DazflowError: what went wrong, the observed state and what to change (design §4)."""

import pytest

import dazflow as dz
from dazflow import errors

CLASSES = [
    errors.ConfigError,
    errors.DatasetError,
    errors.GateError,
    errors.FindingError,
    errors.ResultError,
    errors.RegistryError,
    errors.UnknownKeyError,
    errors.DuplicateKeyError,
    errors.PluginError,
    errors.PipelineError,
    errors.ProjectError,
    errors.RecordError,
    errors.ModelError,
    errors.EvaluationError,
    errors.BuildError,
    errors.SelectError,
    errors.ForecastError,
    errors.DiagnoseError,
]


@pytest.mark.parametrize("cls", CLASSES)
def test_every_error_is_a_dazflow_error_exported_at_top_level(cls):
    assert issubclass(cls, dz.DazflowError)
    assert getattr(dz, cls.__name__) is cls


def test_message_has_three_parts_in_order():
    err = errors.DatasetError(
        "the frame has no rows", observed="0 rows, 2 columns", change="Pass a frame with data."
    )

    assert (
        str(err)
        == "the frame has no rows\n  observed: 0 rows, 2 columns\n  change: Pass a frame with data."
    )
    assert (err.problem, err.observed, err.change) == (
        "the frame has no rows",
        "0 rows, 2 columns",
        "Pass a frame with data.",
    )


def test_observed_and_change_are_required():
    with pytest.raises(TypeError):
        errors.DatasetError("x")
