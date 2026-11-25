"""The house style and an organisation's override (design §9, D24)."""

import datetime as dt
import io
import re
from pathlib import Path

import docx
import pytest

from dazflow import Column, ConfigError, Finding, Source, Table
from dazflow.render import Style, format_cell, load_style, render

FP = "0123456789abcdef" * 4
SOURCE = Source(run_id="run-7", fingerprint=FP, date=dt.date(2026, 11, 25), note="FRED")

COLUMNS = [
    Column(name="series", label="Series", kind="text"),
    Column(name="n", label="Observations", kind="integer"),
    Column(name="level", label="Level", unit="pp"),
    Column(name="share", label="Share", kind="percent"),
    Column(name="date", label="Last", kind="date"),
    Column(name="result", label="Result", kind="status"),
]
ROWS = [
    ["Unemployment", 120, 1234.5678, 0.12345, "2024-03-31", "pass"],
    ["GDP growth", 96, -0.0001, 0.5, "2023-12-31", "fail"],
    ["Inflation", 1500, 12.0, None, "2024-06-30", "overridden"],
]
NUMERIC = {2, 3}  # columns whose cells carry decimals
TABLE = Table(id="T3.1", caption="Series summary", columns=COLUMNS, rows=ROWS, source=SOURCE)


def docx_cells(style):
    doc = docx.Document(io.BytesIO(render(TABLE, "docx", style=style)))
    (table,) = doc.tables
    body = [[c.text for c in row.cells] for row in table.rows]
    rest = [p.text for p in doc.paragraphs]
    return body, rest


def markdown_cells(style):
    text = render(TABLE, "markdown", style=style)
    rows = [line for line in text.splitlines() if line.startswith("|")]
    cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
    other = [line for line in text.splitlines() if not line.startswith("|")]
    return [cells[0], *cells[2:]], other


def html_cells(style):
    text = render(TABLE, "html", style=style)
    text = re.sub(r"<style>.*?</style>", "", text, flags=re.S)
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, flags=re.S)
    return [re.sub(r"<[^>]+>", "", c).strip() for c in cells], []


@pytest.mark.parametrize("reader", [docx_cells, markdown_cells, html_cells])
def test_one_changed_decimal_setting_changes_every_decimal_number_and_nothing_else(reader):
    house = Style()
    three = load_style({"numbers": {"decimals": 3}})

    before, before_rest = reader(house)
    after, after_rest = reader(three)

    assert before_rest == after_rest
    flat_before = [c for row in before for c in (row if isinstance(row, list) else [row])]
    flat_after = [c for row in after for c in (row if isinstance(row, list) else [row])]
    assert len(flat_before) == len(flat_after)
    changed = {(a, b) for a, b in zip(flat_before, flat_after, strict=True) if a != b}
    assert changed == {
        ("1,234.57", "1,234.568"),
        ("12.35%", "12.345%"),
        ("0.00", "0.000"),
        ("50.00%", "50.000%"),
        ("12.00", "12.000"),
    }


def test_the_decimal_setting_reaches_every_numeric_cell_and_no_other():
    three = load_style({"numbers": {"decimals": 3}})

    for row in ROWS:
        for j, (col, value) in enumerate(zip(COLUMNS, row, strict=True)):
            same = format_cell(value, col.kind, Style()) == format_cell(value, col.kind, three)
            assert same == (j not in NUMERIC or value is None), (col.name, value)


@pytest.mark.parametrize(
    ("value", "kind", "settings", "expected"),
    [
        (1234567.891, "number", {}, "1,234,567.89"),
        (1234567.891, "number", {"thousands": ".", "decimal_mark": ","}, "1.234.567,89"),
        (1234567.891, "number", {"thousands": "", "decimals": 0}, "1234568"),
        (-0.001, "number", {}, "0.00"),
        (-1.5, "number", {}, "-1.50"),
        (None, "number", {}, "\u2013"),
        (None, "number", {"missing": "n/a"}, "n/a"),
        (0.0525, "percent", {"decimals": 1}, "5.2%"),
        (1234567, "integer", {}, "1,234,567"),
        (-12, "integer", {"thousands": " "}, "-12"),
        ("a|b", "text", {}, "a|b"),
    ],
)
def test_numbers_follow_the_number_settings(value, kind, settings, expected):
    style = load_style({"numbers": settings})

    assert format_cell(value, kind, style) == expected


def test_dates_and_marks_follow_their_settings():
    style = load_style({"dates": {"format": "%d.%m.%Y"}, "marks": {"failed": {"text": "FAIL"}}})

    assert format_cell("2024-03-31", "date", style) == "31.03.2024"
    assert format_cell("fail", "status", style) == "FAIL"
    assert format_cell("pass", "status", style) == "Pass"


def test_an_override_file_is_merged_onto_the_house_style(tmp_path):
    (tmp_path / "org.docx").write_bytes(b"")
    (tmp_path / "org.mplstyle").write_text("axes.grid: True\n")
    path = tmp_path / "style.yaml"
    path.write_text(
        "numbers:\n  decimals: 1\n"
        "word:\n  reference: org.docx\n"
        "figures:\n  mplstyle: org.mplstyle\n"
        "findings:\n  explore.overview: 'There are {{ n }} series.'\n",
        encoding="utf-8",
    )

    style = load_style(path)

    assert style.numbers.decimals == 1
    assert style.numbers.thousands == Style().numbers.thousands
    assert style.fonts == Style().fonts
    assert style.word.reference == tmp_path / "org.docx"
    assert style.figures.mplstyle == tmp_path / "org.mplstyle"
    assert style.findings == {"explore.overview": "There are {{ n }} series."}


def test_a_style_object_or_none_is_taken_as_it_is():
    style = load_style({"numbers": {"decimals": 4}})

    assert load_style(style) is style
    assert load_style(None) == Style()


@pytest.mark.parametrize(
    "override",
    [
        {"numbers": {"decimal": 3}},
        {"colour": "red"},
        {"numbers": {"decimals": -1}},
        {"numbers": {"decimals": "two"}},
        {"marks": {"failed": {"colour": "red"}}},
        {"page": {"width_cm": 0}},
    ],
)
def test_an_override_with_an_unknown_or_invalid_setting_is_refused(override):
    with pytest.raises(ConfigError):
        load_style(override)


def test_a_missing_override_file_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="style"):
        load_style(tmp_path / "nope.yaml")


def test_a_style_file_whose_paths_do_not_exist_is_refused(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text("word:\n  reference: missing.docx\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"missing\.docx"):
        load_style(path)


def test_a_finding_template_is_overridden_by_its_key_without_code():
    finding = Finding(key="explore.overview", template="{{ n }} series.", values={"n": 3})
    style = load_style({"findings": {"explore.overview": "The data hold {{ n }} series."}})

    assert render(finding, "markdown") == "3 series.\n"
    assert render(finding, "markdown", style=style) == "The data hold 3 series.\n"


def test_an_override_template_that_misses_a_value_is_refused():
    from dazflow import FindingError

    finding = Finding(key="k", template="{{ n }} series.", values={"n": 3})
    style = load_style({"findings": {"k": "{{ n }} series in {{ m }}."}})

    with pytest.raises(FindingError, match="missing"):
        render(finding, "markdown", style=style)


def test_numbers_in_a_finding_follow_the_number_settings():
    finding = Finding(
        template="The mean is {{ mean }}{% if p < 0.05 %}, significant{% endif %} ({{ n }} obs).",
        values={"mean": 1234.5678, "p": 0.01, "n": 1500},
    )

    assert render(finding, "markdown") == "The mean is 1,234.57, significant (1500 obs).\n"
    assert (
        render(finding, "markdown", style=load_style({"numbers": {"decimals": 1}}))
        == "The mean is 1,234.6, significant (1500 obs).\n"
    )
    assert finding.text == "The mean is 1234.5678, significant (1500 obs)."


def test_the_house_style_round_trips_through_yaml(tmp_path):
    path = tmp_path / "house.yaml"
    Style().dump(path)

    assert load_style(path) == Style()


def test_the_house_mplstyle_ships_with_the_package():
    from dazflow.render import HOUSE_MPLSTYLE

    assert Path(HOUSE_MPLSTYLE).is_file()
    assert "svg.fonttype" not in Path(HOUSE_MPLSTYLE).read_text()


def test_a_palette_colour_must_be_hex():
    with pytest.raises(ConfigError, match="RRGGBB"):
        load_style({"figures": {"palette": ["#1F4E79", "blue"]}})


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("numbers: [1\n", "does not parse"),
        ("- a\n- b\n", "not a mapping"),
        ("a: 1\na: 2\n", "parse"),
    ],
)
def test_a_style_file_that_is_not_a_mapping_of_settings_is_refused(tmp_path, text, match):
    path = tmp_path / "style.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=match):
        load_style(path)


def test_an_empty_style_file_is_the_house_style(tmp_path):
    path = tmp_path / "style.yaml"
    path.write_text("", encoding="utf-8")

    assert load_style(path) == Style()


def test_a_style_of_another_type_is_refused():
    with pytest.raises(ConfigError, match="int"):
        load_style(3)
