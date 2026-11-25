"""Figures: SVG snapshots that match on every platform, and PNG (design §9, D25)."""

import datetime as dt
import re
import struct

import matplotlib as mpl
import pytest
from snapshot_helpers import svg_matches, svg_normalise

from dazflow import Figure, Layer, RenderError, Source, Table
from dazflow.render import Style, load_style, render

SOURCE = Source(run_id="run-7", fingerprint="0123456789abcdef" * 4, date=dt.date(2026, 11, 25))
QUARTERS = [f"{y}Q{q}" for y in range(2019, 2025) for q in range(1, 5)]
LEVEL = [4.0 + 0.1 * i - (0.8 if 4 <= i < 8 else 0.0) for i in range(len(QUARTERS))]

SERIES = Figure(
    id="F3.1",
    caption="Unemployment rate with its forecast band",
    x=QUARTERS,
    x_label="Quarter",
    y_label="Unemployment rate",
    y_unit="%",
    layers=[
        Layer(kind="band", y_low=[v - 0.3 for v in LEVEL], y_high=[v + 0.3 for v in LEVEL]),
        Layer(kind="line", y=LEVEL, label="Observed"),
        Layer(kind="line", y=[None] * 20 + LEVEL[20:], label="Forecast"),
        Layer(kind="hline", value=5.0, label="Threshold"),
    ],
    source=SOURCE,
)
BARS = Figure(
    id="F3.2",
    caption="Autocorrelation of the first difference",
    x=[1, 2, 3, 4, 5, 6],
    x_label="Lag",
    y_label="Autocorrelation",
    layers=[
        Layer(kind="bar", y=[0.62, 0.31, 0.12, -0.05, -0.1, 0.02]),
        Layer(kind="hline", value=0.0),
    ],
)
SCATTER = Figure(
    id="F3.3",
    caption="Unemployment against GDP growth",
    x=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0],
    x_label="GDP growth",
    x_unit="%",
    y_label="Change in unemployment",
    y_unit="pp",
    layers=[
        Layer(kind="scatter", y=[0.9, 0.6, 0.2, 0.1, -0.1, -0.3, -0.5], label="Quarters"),
        Layer(kind="line", y=[0.8, 0.55, 0.3, 0.17, 0.05, -0.2, -0.45], label="Fit"),
    ],
    aspect=0.75,
)


@pytest.mark.parametrize("figure", [SERIES, BARS, SCATTER], ids=["series", "bars", "scatter"])
def test_figure_svg_matches_its_snapshot(figure, svg_snapshot):
    assert render(figure, "svg") == svg_snapshot


def test_svg_text_is_drawn_as_paths_without_dates_or_random_ids():
    svg = render(SERIES, "svg")

    assert "<text" not in svg
    assert "<dc:date>" not in svg
    assert "Matplotlib v" not in svg
    assert render(SERIES, "svg") == svg


def test_svg_has_the_page_width_and_the_figures_aspect():
    svg = render(SCATTER, "svg")

    width, height = (
        float(v) for v in re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', svg).groups()
    )
    assert width == pytest.approx(Style().page.width_cm / 2.54 * 72, abs=0.01)
    assert height == pytest.approx(width * 0.75, abs=0.01)


def test_png_has_the_page_width_at_the_style_dpi():
    style = load_style({"figures": {"dpi": 100}})

    png = render(BARS, "png", style=style)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert abs(width - Style().page.width_cm / 2.54 * 100) < 1
    assert abs(height - width * BARS.aspect) < 1
    assert b"Matplotlib" not in png


def test_rendering_leaves_matplotlibs_global_settings_alone():
    before = dict(mpl.rcParams)

    render(SERIES, "svg", style=load_style({"figures": {"font_size": 12}}))
    render(SERIES, "png")

    assert dict(mpl.rcParams) == before


def test_the_style_font_size_and_an_organisation_mplstyle_change_the_figure(tmp_path):
    org = tmp_path / "org.mplstyle"
    org.write_text("axes.facecolor: EEEEEE\n", encoding="utf-8")

    house = render(BARS, "svg")
    bigger = render(BARS, "svg", style=load_style({"figures": {"font_size": 12}}))
    grey = render(BARS, "svg", style=load_style({"figures": {"mplstyle": str(org)}}))

    assert "scale(0.09 -0.09)" in house
    assert "scale(0.12 -0.12)" in bigger
    assert "#eeeeee" in grey
    assert "#eeeeee" not in house


def test_the_figure_palette_comes_from_the_style():
    red = render(BARS, "svg", style=load_style({"figures": {"palette": ["#AA0000"]}}))

    assert "#aa0000" in red


@pytest.mark.parametrize("fmt", ["png", "svg"])
def test_a_table_has_no_image_format(fmt):
    table = Table(id="T1.1", caption="c", columns=[], rows=[])

    with pytest.raises(RenderError, match=fmt):
        render(table, fmt)


# The snapshot comparison itself: a wrong normaliser would turn a failure into a pass.

OLD = """<g id="PolyCollection_1"><path clip-path="url(#p85a6e9cc87)" d="M 54.82 218.894594"/>
<path id="DejaVuSans-32" d="M 1228 531"/><use xlink:href="#DejaVuSans-32"/>
<use xlink:href="#DejaVuSans-32" x="63.623047"/></g>
<clipPath id="p85a6e9cc87"/>"""
NEW = """<g id="FillBetweenPolyCollection_1"><path clip-path="url(#p2a162ef4a6)" d="M 54.82 218.893891"/>
<path id="DejaVuSans-15" d="M 1228 531"/><use xlink:href="#DejaVuSans-15"/>
<use xlink:href="#DejaVuSans-15" transform="translate(63.625 0)"/></g>
<clipPath id="p2a162ef4a6"/>"""


def test_the_svg_comparison_ignores_what_matplotlib_releases_differ_in():
    assert svg_matches(svg_normalise(OLD), svg_normalise(NEW))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("218.894594", "220.0"),  # a point moved by more than a point
        ("M 1228 531", "M 1228 731"),  # a glyph outline changed
        ('x="63.623047"', 'x="65.623047"'),  # a glyph moved
        ('clip-path="url(#p85a6e9cc87)" ', ""),  # an element lost its clip path
        ('<use xlink:href="#DejaVuSans-32"/>\n', ""),  # a glyph went missing
    ],
)
def test_the_svg_comparison_still_sees_a_real_change(old, new):
    assert old in OLD
    assert not svg_matches(svg_normalise(OLD), svg_normalise(OLD.replace(old, new, 1)))


def test_the_svg_comparison_sees_a_changed_scale():
    a = '<g transform="scale(0.09 -0.09)"/>'
    b = '<g transform="scale(0.1 -0.1)"/>'

    assert not svg_matches(svg_normalise(a), svg_normalise(b))
