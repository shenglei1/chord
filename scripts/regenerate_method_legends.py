"""Regenerate the method legends used by the main comparison figures.

This script changes presentation only: it contains no experimental values and
does not read or write result CSV files.  The exported dimensions match the
existing legend panels (522 x 39.6 pt), while the labels use the manuscript's
formal baseline names and remain legible at the final single-column width.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


COMPARISON_OUTPUT_NAMES = ("Figure_3L.pdf", "Figure_4L.pdf")
LEARNING_OUTPUT_NAME = "Figure_B_1L.pdf"
PAGE_SIZE_PT = (522.0, 39.6)
FIGURE_DIR = Path("figs")
ARTWORK_FONT = font_manager.findfont("Arial", fallback_to_default=True)

ENTRIES = (
    ("CHORD-CAHR", "#0072B2", "circle", 0, 0),
    ("Entity-MAPPO", "#009E73", "triangle", 0, 1),
    ("PLATO-faithful", "#E66101", "square", 1, 0),
    ("Graph-MAPPO", "#E69F00", "diamond", 1, 1),
    ("MOHITO-512", "#CC79A7", "plus", 2, 0),
    ("Rolling-Horizon MILP", "#4D4D4D", "cross", 2, 1),
)


def draw_marker(pdf: canvas.Canvas, x: float, y: float, colour: str, shape: str) -> None:
    pdf.setStrokeColor(colour)
    pdf.setFillColor(colour)
    pdf.setLineWidth(0.8)
    if shape != "cross":
        pdf.line(x, y - 5.0, x, y + 5.0)
        pdf.line(x - 2.4, y - 5.0, x + 2.4, y - 5.0)
        pdf.line(x - 2.4, y + 5.0, x + 2.4, y + 5.0)
    if shape == "circle":
        pdf.circle(x, y, 2.6, stroke=1, fill=1)
    elif shape == "triangle":
        path = pdf.beginPath()
        path.moveTo(x, y + 3.2)
        path.lineTo(x - 3.2, y - 2.8)
        path.lineTo(x + 3.2, y - 2.8)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif shape == "square":
        pdf.rect(x - 2.7, y - 2.7, 5.4, 5.4, stroke=1, fill=1)
    elif shape == "diamond":
        path = pdf.beginPath()
        path.moveTo(x, y + 3.4)
        path.lineTo(x - 3.4, y)
        path.lineTo(x, y - 3.4)
        path.lineTo(x + 3.4, y)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif shape == "plus":
        pdf.setLineWidth(2.0)
        pdf.line(x - 3.0, y, x + 3.0, y)
        pdf.line(x, y - 3.0, x, y + 3.0)
    else:
        pdf.setLineWidth(2.2)
        pdf.line(x - 2.8, y - 2.8, x + 2.8, y + 2.8)
        pdf.line(x - 2.8, y + 2.8, x + 2.8, y - 2.8)


def make_legend(path: Path) -> None:
    pdfmetrics.registerFont(TTFont("ArialEmbedded", ARTWORK_FONT))
    pdf = canvas.Canvas(
        str(path),
        pagesize=PAGE_SIZE_PT,
        pageCompression=1,
        initialFontName="ArialEmbedded",
        initialFontSize=9.5,
    )
    pdf.setTitle("Method legend")
    pdf.setFont("ArialEmbedded", 9.5)
    icon_x = (120.0, 216.0, 311.0)
    text_x = (134.0, 230.0, 325.0)
    row_y = (25.0, 10.5)
    for label, colour, shape, column, row in ENTRIES:
        draw_marker(pdf, icon_x[column], row_y[row], colour, shape)
        pdf.setFillColor("#111111")
        pdf.drawString(text_x[column], row_y[row] - 3.2, label)
    pdf.showPage()
    pdf.save()


def make_learning_legend(path: Path) -> None:
    pdfmetrics.registerFont(TTFont("ArialEmbedded", ARTWORK_FONT))
    pdf = canvas.Canvas(
        str(path),
        pagesize=PAGE_SIZE_PT,
        pageCompression=1,
        initialFontName="ArialEmbedded",
        initialFontSize=9.5,
    )
    pdf.setTitle("Learning-curve method legend")
    pdf.setFont("ArialEmbedded", 9.5)
    entries = (
        ("CHORD-CAHR", "#0072B2", (), 0, 0),
        ("Entity-MAPPO", "#009E73", (1.2, 1.8), 0, 1),
        ("PLATO-faithful", "#E66101", (4.0, 2.0), 1, 0),
        ("Graph-MAPPO", "#E69F00", (4.0, 2.0), 1, 1),
        ("MOHITO-512", "#CC79A7", (6.0, 2.0, 1.2, 2.0), 2, 0),
    )
    line_start = (121.0, 217.0, 312.0)
    text_x = (145.0, 241.0, 336.0)
    row_y = (25.0, 10.5)
    for label, colour, dash, column, row in entries:
        pdf.setStrokeColor(colour)
        pdf.setLineWidth(1.25)
        pdf.setDash(dash)
        pdf.line(line_start[column], row_y[row], line_start[column] + 19.0, row_y[row])
        pdf.setDash()
        pdf.setFillColor("#111111")
        pdf.drawString(text_x[column], row_y[row] - 3.2, label)
    pdf.showPage()
    pdf.save()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in COMPARISON_OUTPUT_NAMES:
        output = FIGURE_DIR / name
        make_legend(output)
        print(f"regenerated {output}")
    learning_output = FIGURE_DIR / LEARNING_OUTPUT_NAME
    make_learning_legend(learning_output)
    print(f"regenerated {learning_output}")


if __name__ == "__main__":
    main()
