"""Readers: data from files or frames into a `Dataset` on a period calendar (design §6).

``read(spec)`` picks the reader by ``spec.reader`` or the file suffix:
``csv`` (core), ``excel`` (openpyxl) and ``parquet`` (pyarrow), the last two
in the ``dazflow[io]`` extra. ``from_frame(obj)`` takes a pandas frame, or a
polars frame or pyarrow table through narwhals.

Dates become periods at monthly (``M``), quarterly (``Q``) or annual (``Y``)
frequency: given in ``spec.frequency``, or read from the dates, the finest
of the three on which the dates fall in consecutive periods.
"""

from dazflow.readers import _builtin  # noqa: F401 - registers the built-ins
from dazflow.readers._base import (
    FREQUENCIES,
    Reader,
    ReadSpec,
    from_frame,
    get,
    read,
    register,
    registry,
    require,
)
from dazflow.readers._scenarios import ScenarioInput, read_scenarios

__all__ = [
    "FREQUENCIES",
    "ReadSpec",
    "Reader",
    "ScenarioInput",
    "from_frame",
    "get",
    "read",
    "read_scenarios",
    "register",
    "registry",
    "require",
]
