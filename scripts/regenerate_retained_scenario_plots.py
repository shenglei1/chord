"""Regenerate the scenario-dependent plots used by the current manuscript.

The script reads archived aggregate CSV files without modifying them and
rebuilds the application-comparison and matched-encoder effect panels for the
six-scenario evaluation set reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from matplotlib import font_manager
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


RESULTS_ROOT = Path("results")
PAPER_ARTIFACTS = Path("figs/paper_artifacts")
FIGURES = Path("figs")

RETAINED_SCENARIOS = (
    "in_distribution",
    "unseen_small",
    "unseen_large",
    "agv_heavy_ratio",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
)
APPLICATION_SCENARIOS = (
    "in_distribution",
    "agv_heavy_ratio",
    "fixed_team_dynamic_orders",
    "robot_breakdown",
)
SCENARIO_LABELS = {
    "in_distribution": "In-distribution",
    "unseen_small": "Unseen small",
    "unseen_large": "Unseen large",
    "agv_heavy_ratio": "AGV-heavy",
    "fixed_team_dynamic_orders": "Dynamic orders",
    "robot_breakdown": "Robot breakdown",
}
METHODS = (
    "chord_mappo",
    "plato_faithful",
    "mohito_official",
    "entity_mappo",
    "graph_mappo",
    "rolling_horizon",
)
STYLES = {
    "chord_mappo": ("#0072B2", "circle"),
    "plato_faithful": ("#D55E00", "square"),
    "mohito_official": ("#CC79A7", "plus"),
    "entity_mappo": ("#009E73", "triangle"),
    "graph_mappo": ("#E69F00", "diamond"),
    "rolling_horizon": ("#4D4D4D", "cross"),
}

pdfmetrics.registerFont(
    TTFont("ArtworkArial", font_manager.findfont("Arial", fallback_to_default=True))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=("all", "figure3", "appendix-j"), default="all",
        help="Generate both retained plot groups or one selected group.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def marker(pdf: canvas.Canvas, x: float, y: float, kind: str, colour: str, size: float = 3.0) -> None:
    pdf.setStrokeColor(HexColor(colour))
    pdf.setFillColor(HexColor(colour))
    pdf.setLineWidth(1.0)
    if kind == "circle":
        pdf.circle(x, y, size, stroke=1, fill=1)
    elif kind == "square":
        pdf.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    elif kind == "triangle":
        path = pdf.beginPath()
        path.moveTo(x, y + size)
        path.lineTo(x - size, y - size)
        path.lineTo(x + size, y - size)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif kind == "left_triangle":
        path = pdf.beginPath()
        path.moveTo(x - size, y)
        path.lineTo(x + size, y + size)
        path.lineTo(x + size, y - size)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif kind == "diamond":
        path = pdf.beginPath()
        path.moveTo(x, y + size)
        path.lineTo(x - size, y)
        path.lineTo(x, y - size)
        path.lineTo(x + size, y)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif kind == "plus":
        pdf.setLineWidth(1.6)
        pdf.line(x - size, y, x + size, y)
        pdf.line(x, y - size, x, y + size)
    else:
        pdf.setLineWidth(1.4)
        pdf.line(x - size, y - size, x + size, y + size)
        pdf.line(x - size, y + size, x + size, y - size)


def rotated_text(pdf: canvas.Canvas, x: float, y: float, text: str, angle: float, size: float, align: str = "right") -> None:
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(angle)
    pdf.setFillColor(HexColor("#111111"))
    pdf.setFont("ArtworkArial", size)
    if align == "center":
        pdf.drawCentredString(0, 0, text)
    elif align == "left":
        pdf.drawString(0, 0, text)
    else:
        pdf.drawRightString(0, 0, text)
    pdf.restoreState()


def error_bar(pdf: canvas.Canvas, x: float, low: float, high: float, colour: str) -> None:
    pdf.setStrokeColor(HexColor(colour))
    pdf.setLineWidth(0.75)
    pdf.line(x, low, x, high)
    pdf.line(x - 2.0, low, x + 2.0, low)
    pdf.line(x - 2.0, high, x + 2.0, high)


def draw_axes(
    pdf: canvas.Canvas,
    width: float,
    height: float,
    labels: list[str],
    y_ticks: list[float],
    y_min: float,
    y_max: float,
    y_label: str,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    x_rotation: float,
    x_axis_label: str | None = None,
):
    plot_width = width - left - right
    plot_height = height - bottom - top
    x_values = [left + plot_width * (index + 0.5) / len(labels) for index in range(len(labels))]

    def map_y(value: float) -> float:
        return bottom + (value - y_min) * plot_height / (y_max - y_min)

    pdf.setFont("ArtworkArial", 8.0)
    for tick in y_ticks:
        y = map_y(tick)
        pdf.setStrokeColor(HexColor("#D8D8D8"))
        pdf.setLineWidth(0.45)
        pdf.line(left, y, width - right, y)
        pdf.setFillColor(HexColor("#111111"))
        label = f"{tick:.2f}" if max(abs(v) for v in y_ticks) < 1 else f"{tick:g}"
        pdf.drawRightString(left - 4, y - 2.5, label)
    pdf.setStrokeColor(HexColor("#111111"))
    pdf.setLineWidth(0.8)
    pdf.line(left, bottom, width - right, bottom)
    pdf.line(left, bottom, left, height - top)
    for x, label in zip(x_values, labels):
        pdf.line(x, bottom, x, bottom - 3)
        rotated_text(pdf, x - 1, bottom - 7, label, x_rotation, 8.0)
    rotated_text(pdf, 12, bottom + plot_height / 2, y_label, 90, 9.0, "center")
    if x_axis_label:
        pdf.setFont("ArtworkArial", 8.5)
        pdf.setFillColor(HexColor("#111111"))
        pdf.drawCentredString(left + plot_width / 2, 5, x_axis_label)
    return x_values, map_y


def application_panels() -> None:
    rows = read_rows(PAPER_ARTIFACTS / "table_main_results_seed_level.csv")
    indexed = {(row["scenario"], row["method"]): row for row in rows}
    panels = (
        ("service_level", "Service level", "Figure_3A.pdf", 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("throughput_per_100_steps", "Logged throughput", "Figure_3B.pdf", 0.0, 25.0, [0, 5, 10, 15, 20, 25]),
    )
    width, height = 264.0, 201.5
    for metric, ylabel, filename, y_min, y_max, ticks in panels:
        pdf = canvas.Canvas(str(FIGURES / filename), pagesize=(width, height), pageCompression=1)
        labels = [SCENARIO_LABELS[scenario] for scenario in APPLICATION_SCENARIOS]
        xs, map_y = draw_axes(
            pdf, width, height, labels, ticks, y_min, y_max, ylabel,
            left=47, right=7, bottom=51, top=7, x_rotation=30,
        )
        group_width = (width - 47 - 7) / len(labels)
        offsets = [(-0.25 + index * 0.10) * group_width for index in range(len(METHODS))]
        for method, offset in zip(METHODS, offsets):
            colour, shape = STYLES[method]
            for x, scenario in zip(xs, APPLICATION_SCENARIOS):
                row = indexed[(scenario, method)]
                mean = float(row[f"{metric}_mean"])
                if method != "rolling_horizon":
                    low = float(row[f"{metric}_low"])
                    high = float(row[f"{metric}_high"])
                    error_bar(pdf, x + offset, map_y(low), map_y(high), colour)
                marker(pdf, x + offset, map_y(mean), shape, colour, 2.6)
        pdf.showPage()
        pdf.save()


def nice_limits(values: list[float]) -> tuple[float, float, list[float]]:
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    span = max(high - low, 1e-6)
    rough = span / 5
    magnitude = 10 ** math.floor(math.log10(rough))
    fraction = rough / magnitude
    step = (1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10) * magnitude
    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step
    if low == high:
        high = low + step
    ticks = []
    value = low
    while value <= high + step * 0.1:
        ticks.append(round(value, 12))
        value += step
    return low, high, ticks


def encoder_effect_panels() -> None:
    rows = read_rows(RESULTS_ROOT / "mechanism_controls" / "aggregate" / "paired_differences.csv")
    indexed = {(row["baseline"], row["scenario"], row["metric"]): row for row in rows}
    panels = (
        ("service_level", "service-level", "Figure_J_1A.pdf", 1.0),
        ("throughput_per_100_steps", "logged-throughput", "Figure_J_1B.pdf", 1.0),
        ("deadline_miss_rate", "deadline-miss", "Figure_J_2A.pdf", -1.0),
        ("mean_flow_time", "flow-time", "Figure_J_2B.pdf", -1.0),
    )
    baselines = (
        ("entity_cahr", "Entity+CAHR", "#56B4E9", "triangle", -4.0),
        ("graph_cahr", "Graph+CAHR", "#B79F00", "left_triangle", 4.0),
    )
    width, height = 491.5, 203.5
    for metric, short_name, filename, sign in panels:
        oriented: dict[tuple[str, str], tuple[float, float, float]] = {}
        bounds = [0.0]
        for baseline, _, _, _, _ in baselines:
            for scenario in RETAINED_SCENARIOS:
                row = indexed[(baseline, scenario, metric)]
                mean = sign * float(row["mean_difference"])
                if sign > 0:
                    low, high = float(row["ci95_low"]), float(row["ci95_high"])
                else:
                    low, high = -float(row["ci95_high"]), -float(row["ci95_low"])
                oriented[(baseline, scenario)] = (mean, low, high)
                bounds.extend((low, high))
        y_min, y_max, ticks = nice_limits(bounds)
        pdf = canvas.Canvas(str(FIGURES / filename), pagesize=(width, height), pageCompression=1)
        labels = [SCENARIO_LABELS[scenario] for scenario in RETAINED_SCENARIOS]
        xs, map_y = draw_axes(
            pdf, width, height, labels, ticks, y_min, y_max,
            f"Oriented {short_name} difference",
            left=64, right=8, bottom=55, top=25, x_rotation=27,
        )
        rotated_text(pdf, 24, 55 + (height - 55 - 25) / 2, "(positive favours CHORD)", 90, 8.0, "center")
        pdf.setStrokeColor(HexColor("#333333"))
        pdf.setLineWidth(0.65)
        pdf.line(64, map_y(0.0), width - 8, map_y(0.0))
        for baseline, _, colour, shape, offset in baselines:
            for x, scenario in zip(xs, RETAINED_SCENARIOS):
                mean, low, high = oriented[(baseline, scenario)]
                error_bar(pdf, x + offset, map_y(low), map_y(high), colour)
                marker(pdf, x + offset, map_y(mean), shape, colour, 3.2)
        legend_y = height - 11
        legend_positions = (width / 2 - 70, width / 2 + 70)
        for x, (_, label, colour, shape, _) in zip(legend_positions, baselines):
            marker(pdf, x - 34, legend_y, shape, colour, 3.2)
            pdf.setFillColor(HexColor("#111111"))
            pdf.setFont("ArtworkArial", 8.5)
            pdf.drawString(x - 25, legend_y - 3, label)
        pdf.showPage()
        pdf.save()


if __name__ == "__main__":
    arguments = parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    if arguments.group in ("all", "figure3"):
        application_panels()
    if arguments.group in ("all", "appendix-j"):
        encoder_effect_panels()
