"""Every built-in check passes the public component contract (`dazflow.testing`)."""

import pytest

from dazflow import checks
from dazflow.testing import ComponentContract

KEYS = (
    "completeness",
    "gaps",
    "duplicates",
    "range",
    "revisions",
    "outliers",
    "representativeness",
)


@pytest.mark.parametrize("key", KEYS)
def test_built_in_check_contract(key):
    contract = type(
        f"Test_{key}", (ComponentContract,), {"registry": checks.registry, "key": key}
    )()
    for name in dir(contract):
        if name.startswith("test_"):
            getattr(contract, name)()


@pytest.mark.parametrize("key", KEYS)
def test_every_built_in_config_is_a_check_config(key):
    assert issubclass(checks.get(key).Config, checks.CheckConfig)
