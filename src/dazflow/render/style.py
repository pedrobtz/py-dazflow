"""The house style and an organisation's override of it (design §9, D24)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field, field_validator

from dazflow.core.config import Config, _UniqueKeyLoader
from dazflow.core.errors import ConfigError

__all__ = [
    "HOUSE_MPLSTYLE",
    "Captions",
    "Dates",
    "Figures",
    "Fonts",
    "Mark",
    "Marks",
    "Numbers",
    "Page",
    "Style",
    "Word",
    "load_style",
]

HOUSE_MPLSTYLE = str(Path(__file__).with_name("dazflow.mplstyle"))

_HEX = r"^#[0-9A-Fa-f]{6}$"


def _existing_file(path: Path | None) -> Path | None:
    if path is not None and not Path(path).is_file():
        raise ValueError(f"{path} is not a file")
    return path


class Fonts(Config):
    """Text in Word and HTML. Word takes its fonts from `Word.reference` when one is given."""

    body: str = "Calibri"
    size: float = Field(default=10.0, gt=0)


class Numbers(Config):
    """How numbers are written: in tables, and in findings' numeric values."""

    decimals: int = Field(default=2, ge=0, le=12)
    thousands: str = ","
    decimal_mark: str = Field(default=".", min_length=1)
    missing: str = "\u2013"  # an en dash


class Dates(Config):
    """How date cells and the source note's date are written (a `strftime` format)."""

    format: str = Field(default="%Y-%m-%d", min_length=1)


class Page(Config):
    """The text width of the page: tables and figures are laid out to it."""

    width_cm: float = Field(default=16.0, gt=0)


class Mark(Config):
    """How one check status is marked in a table."""

    text: str = Field(min_length=1)
    colour: str = Field(pattern=_HEX)
    bold: bool = False


class Marks(Config):
    """Pass, fail and overridden, marked in place in words and colour."""

    passed: Mark = Mark(text="Pass", colour="#1B7F3B")
    failed: Mark = Mark(text="Fail", colour="#B00020", bold=True)
    overridden: Mark = Mark(text="Overridden", colour="#8A5A00", bold=True)

    def __getitem__(self, status: str) -> Mark:
        return {"pass": self.passed, "fail": self.failed, "overridden": self.overridden}[status]


class Captions(Config):
    """Caption and source-note templates (jinja2, as findings are)."""

    table: str = "Table {{ id }}. {{ caption }}"
    figure: str = "Figure {{ id }}. {{ caption }}"
    source: str = (
        "Source: {% if note %}{{ note }}; {% endif %}"
        "run {{ run_id }}, data {{ fingerprint }}, {{ date }}."
    )
    fingerprint_digits: int = Field(default=12, ge=4, le=64)


class Figures(Config):
    """Figures: the font matplotlib draws with, the colours, PNG resolution, an extra mplstyle.

    The default font is DejaVu Sans, which matplotlib ships, so a figure is
    drawn the same on every platform.
    """

    font: str = "DejaVu Sans"
    font_size: float = Field(default=9.0, gt=0)
    dpi: int = Field(default=200, ge=50, le=1200)
    palette: tuple[str, ...] = Field(
        default=("#1F4E79", "#C55A11", "#548235", "#7F6000", "#7030A0", "#404040"),
        min_length=1,
    )
    reference_colour: str = Field(default="#7F7F7F", pattern=_HEX)
    mplstyle: Path | None = None

    @field_validator("palette")
    @classmethod
    def _colours_are_hex(cls, palette: tuple[str, ...]) -> tuple[str, ...]:
        bad = [c for c in palette if not re.fullmatch(_HEX, c)]
        if bad:
            raise ValueError(f"colours are written #RRGGBB; not: {', '.join(bad)}")
        return palette

    @field_validator("mplstyle")
    @classmethod
    def _mplstyle_exists(cls, path: Path | None) -> Path | None:
        return _existing_file(path)


class Word(Config):
    """Word output: the organisation's reference document and its table style."""

    reference: Path | None = None
    table_style: str = Field(default="Table Grid", min_length=1)

    @field_validator("reference")
    @classmethod
    def _reference_exists(cls, path: Path | None) -> Path | None:
        return _existing_file(path)


class Style(Config):
    """Everything that decides how an artefact looks, in one file per organisation.

    `load_style` merges an organisation's partial override onto these
    defaults. `findings` maps a finding's `key` to a template that replaces
    the step's own, so rewording a finding needs no code.
    """

    _hint: ClassVar[str] = "Correct the style settings listed in 'observed'."

    fonts: Fonts = Fonts()
    numbers: Numbers = Numbers()
    dates: Dates = Dates()
    page: Page = Page()
    marks: Marks = Marks()
    captions: Captions = Captions()
    figures: Figures = Figures()
    word: Word = Word()
    findings: dict[str, str] = Field(default_factory=dict)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(data: dict[str, Any], folder: Path) -> dict[str, Any]:
    for section, field in (("word", "reference"), ("figures", "mplstyle")):
        part = data.get(section)
        if isinstance(part, dict) and isinstance(part.get(field), str):
            path = Path(part[field])
            part[field] = path if path.is_absolute() else folder / path
    return data


def load_style(source: Style | Mapping[str, Any] | str | Path | None = None) -> Style:
    """The style to render with.

    `source` is ``None`` (the house style), a `Style` (used as it is), a
    mapping, or the path of a YAML file. A mapping or file is a partial
    override, merged onto the house style setting by setting; paths in a
    file are relative to the file. An unknown or invalid setting is a
    `ConfigError`.
    """
    if source is None:
        return Style()
    if isinstance(source, Style):
        return source
    if isinstance(source, Mapping):
        override = dict(source)
    elif isinstance(source, str | Path):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as err:
            raise ConfigError(
                "the style file cannot be read",
                observed=f"{path}: {err.strerror or err}",
                change="Check the path of the style file and that it exists.",
            ) from err
        try:
            loaded = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
        except yaml.YAMLError as err:
            raise ConfigError(
                "the style file does not parse",
                observed=f"{path}: {str(err).replace(chr(10), ' ')}",
                change="Correct the YAML syntax at the position given.",
            ) from err
        loaded = {} if loaded is None else loaded
        if not isinstance(loaded, dict):
            raise ConfigError(
                "the style file is not a mapping",
                observed=f"{path}: a {type(loaded).__name__} at the top level",
                change="Write the style as 'section: {setting: value}' lines.",
            )
        override = _resolve_paths(loaded, path.parent)
    else:
        raise ConfigError(
            "a style is a Style, a mapping or the path of a YAML file",
            observed=f"a {type(source).__name__}",
            change="Pass a Style, a dict of settings or a path.",
        )
    return Style.model_validate(_merge(Style().model_dump(), override))
