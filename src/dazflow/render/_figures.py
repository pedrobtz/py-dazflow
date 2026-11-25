"""Figures drawn with matplotlib in the house style, as SVG and PNG."""

from __future__ import annotations

import io
import math

import matplotlib.style
from matplotlib import cycler
from matplotlib.figure import Figure as MplFigure

from dazflow.core.artefact import Figure
from dazflow.render.style import HOUSE_MPLSTYLE, Style

__all__ = ["draw", "figure_png", "figure_size", "figure_svg"]

CM_PER_INCH = 2.54
MAX_TICK_LABELS = 8
# Margins around the axes, in inches. Fixed rather than computed by a layout
# engine, whose results move by more than half a point between matplotlib
# releases; room for y tick labels of about ten characters at 9 pt.
MARGINS = {"left": 0.8, "right": 0.15, "top": 0.15, "bottom": 0.55}

# Fixed in code, after any organisation mplstyle, so that the same figure gives
# the same SVG on every platform: text as paths (no font lookup by the viewer),
# ids from a fixed salt, and no date in the metadata.
_REPRODUCIBLE = {"svg.fonttype": "path", "svg.hashsalt": "dazflow", "path.simplify": False}
_SVG_METADATA = {"Date": None, "Creator": None, "Format": None, "Type": None}
_PNG_METADATA = {"Software": None}


def figure_size(figure: Figure, style: Style) -> tuple[float, float]:
    """Width and height in inches: the page width, and the figure's aspect."""
    width = style.page.width_cm / CM_PER_INCH
    return width, width * figure.aspect


def _rc(style: Style) -> list:
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [style.figures.font],
        "font.size": style.figures.font_size,
        "axes.prop_cycle": cycler(color=list(style.figures.palette)),
    }
    styles: list = [HOUSE_MPLSTYLE, rc]
    if style.figures.mplstyle is not None:
        styles.append(str(style.figures.mplstyle))
    return [*styles, _REPRODUCIBLE]


def _positions(figure: Figure) -> tuple[list[float], list[float] | None, list[str] | None]:
    """Where the x values go; for labels, which of them get a tick."""
    if figure.x and all(isinstance(v, str) for v in figure.x):
        n = len(figure.x)
        step = max(1, math.ceil(n / MAX_TICK_LABELS))
        ticks = list(range(0, n, step))
        return list(range(n)), [float(t) for t in ticks], [figure.x[t] for t in ticks]
    return [float(v) for v in figure.x], None, None


def _values(values: tuple) -> list[float]:
    return [math.nan if v is None else float(v) for v in values]


def draw(figure: Figure, style: Style) -> MplFigure:
    """The figure as a matplotlib `Figure`; call inside `matplotlib.style.context(_rc(style))`."""
    fig = MplFigure(figsize=figure_size(figure, style))
    ax = fig.subplots()
    x, ticks, labels = _positions(figure)
    palette = list(style.figures.palette)
    colour = 0
    for layer in figure.layers:
        kwargs = {"label": layer.label} if layer.label else {}
        if layer.kind == "hline":
            ax.axhline(
                layer.value,
                color=style.figures.reference_colour,
                linestyle="--",
                linewidth=1.0,
                **kwargs,
            )
            continue
        c = palette[colour % len(palette)]
        if layer.kind == "band":
            ax.fill_between(
                x, _values(layer.y_low), _values(layer.y_high), color=c, alpha=0.2, **kwargs
            )
            continue
        colour += 1
        if layer.kind == "line":
            ax.plot(x, _values(layer.y), color=c, **kwargs)
        elif layer.kind == "scatter":
            ax.scatter(x, _values(layer.y), color=c, **kwargs)
        else:
            ax.bar(x, _values(layer.y), color=c, **kwargs)
    if ticks is not None:
        ax.set_xticks(ticks, labels)
    ax.set_xlabel(figure.x_heading)
    ax.set_ylabel(figure.y_heading)
    if any(layer.label for layer in figure.layers):
        ax.legend()
    width, height = fig.get_size_inches()
    fig.subplots_adjust(
        left=MARGINS["left"] / width,
        right=1 - MARGINS["right"] / width,
        bottom=MARGINS["bottom"] / height,
        top=1 - MARGINS["top"] / height,
    )
    return fig


def figure_svg(figure: Figure, style: Style) -> str:
    """The figure as SVG text, the same bytes on every platform for one matplotlib release."""
    buf = io.StringIO()
    with matplotlib.style.context(_rc(style)):
        draw(figure, style).savefig(buf, format="svg", metadata=_SVG_METADATA)
    return buf.getvalue()


def figure_png(figure: Figure, style: Style) -> bytes:
    """The figure as PNG at the style's resolution."""
    buf = io.BytesIO()
    with matplotlib.style.context(_rc(style)):
        draw(figure, style).savefig(
            buf, format="png", dpi=style.figures.dpi, metadata=_PNG_METADATA
        )
    return buf.getvalue()
