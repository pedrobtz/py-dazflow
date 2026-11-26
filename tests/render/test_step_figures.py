"""Every figure a step draws, as SVG, against a stored snapshot (design §13, D25).

`test_figures.py` snapshots three hand-built figures; this file snapshots the
figures the steps themselves build, from the example project's run: ingest's
history, clean's flagged periods, explore's lag profiles, build's z
statistics, select's back-test and scenario projections, forecast's fan
chart, and diagnose's residuals, CUSUM and rolling coefficients. A change to
a step's figure, or to the renderer, is then a reviewed snapshot change.

The figures come from `reports/record.json`, so they are what the document
gets. The example's numbers repeat between runs to about 1e-14 (design D43),
far inside the SVG tolerance of `snapshot_helpers`, and they are the same to
1e-9 on the lowest direct dependencies (statsmodels 0.14.2).

Four figures are drawn differently by matplotlib 3.8.4, the lowest-direct
job's release, than by the 3.11 the snapshots come from: the legend of
select's six-line back-test and of forecast's fan chart moves by about
1.3 pt, clean's scatter and diagnose's rolling coefficients have other
ticks. Their snapshots are compared on every leg with matplotlib 3.11 or
later and skipped on the lowest-direct job only; the difference is tracked
as a gap in `.agents/assurance.md`.
"""

import matplotlib
import pytest

import dazflow as dz
from dazflow.render import render

FIGURES = {
    "F1.1": "ingest",
    "F2.1": "clean",
    "F3.1": "explore",
    "F4.1": "build",
    "F6.1": "select",
    "F6.2": "select",
    "F7.1": "forecast",
    "F8.1": "diagnose",
    "F8.2": "diagnose",
    "F8.3": "diagnose",
}


@pytest.fixture(scope="module")
def record(ran_example):
    return dz.pipeline.RunRecord.load(ran_example / "reports" / "record.json")


def test_every_step_figure_has_a_snapshot(record):
    drawn = {
        (step, key)
        for step in record.run.steps
        for key, artefact in record.result(step).artefacts.items()
        if isinstance(artefact, dz.Figure)
    }
    assert drawn == {(step, key) for key, step in FIGURES.items()}


# Drawn differently by matplotlib before 3.11 (see the module docstring).
NEWER_MATPLOTLIB = {"F2.1", "F6.1", "F7.1", "F8.3"}
MATPLOTLIB = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            k,
            marks=pytest.mark.skipif(
                k in NEWER_MATPLOTLIB and MATPLOTLIB < (3, 11),
                reason=f"matplotlib {matplotlib.__version__} draws {k} differently; gap #42",
            ),
        )
        for k in FIGURES
    ],
)
def test_step_figure_svg_matches_its_snapshot(record, key, svg_snapshot):
    figure = record.result(FIGURES[key]).artefacts[key]
    assert render(figure, "svg") == svg_snapshot
