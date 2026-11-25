"""The built-in readers: csv (core), excel and parquet (the io extra)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from pydantic import Field

from dazflow.core.config import Config
from dazflow.core.dataset import Dataset
from dazflow.core.errors import ReaderError
from dazflow.readers._base import ReadSpec, from_frame, path_of, register, require

__all__ = ["Csv", "Excel", "Parquet"]


def _load(path: str, what: str, call: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    try:
        return call()
    except OSError as err:
        raise ReaderError(
            f"the {what} file {path!r} cannot be read",
            observed=f"{type(err).__name__}: {err.strerror or err}",
            change="Check the path and that the file exists.",
        ) from err
    except (ValueError, KeyError, pd.errors.ParserError) as err:
        raise ReaderError(
            f"the {what} file {path!r} cannot be parsed",
            observed=f"{type(err).__name__}: {err}",
            change=f"Check that the file is {what} and the reader options match it.",
        ) from err


class CsvConfig(Config):
    sep: str = Field(",", min_length=1, description="The field separator.")
    decimal: str = Field(".", min_length=1, max_length=1, description="The decimal mark.")
    encoding: str = "utf-8"


@register("csv")
class Csv:
    """Comma-separated text, read with pandas."""

    Config = CsvConfig

    def read(self, spec: ReadSpec) -> Dataset:
        opts = CsvConfig.model_validate(spec.options)
        path = path_of(spec, "the csv reader")
        frame = _load(
            path,
            "csv",
            lambda: pd.read_csv(path, sep=opts.sep, decimal=opts.decimal, encoding=opts.encoding),
        )
        return _read(frame, spec, path)


class ExcelConfig(Config):
    sheet: str | int = Field(0, description="Sheet name, or its position from 0.")


@register("excel")
class Excel:
    """An Excel workbook (.xlsx), read with openpyxl from the io extra."""

    Config = ExcelConfig

    def read(self, spec: ReadSpec) -> Dataset:
        opts = ExcelConfig.model_validate(spec.options)
        require("openpyxl", "reading Excel")
        path = path_of(spec, "the excel reader")
        frame = _load(
            path,
            "Excel",
            lambda: pd.read_excel(path, sheet_name=opts.sheet, engine="openpyxl"),
        )
        return _read(frame, spec, path)


class ParquetConfig(Config):
    """No options."""


@register("parquet")
class Parquet:
    """A Parquet file, read with pyarrow from the io extra."""

    Config = ParquetConfig

    def read(self, spec: ReadSpec) -> Dataset:
        ParquetConfig.model_validate(spec.options)
        require("pyarrow", "reading Parquet")
        path = path_of(spec, "the parquet reader")
        frame = _load(path, "Parquet", lambda: pd.read_parquet(path, engine="pyarrow"))
        return _read(frame, spec, path)


def _read(frame: Any, spec: ReadSpec, path: str) -> Dataset:
    return from_frame(
        frame, date=spec.date, frequency=spec.frequency, columns=spec.columns, source=path
    )
