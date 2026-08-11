from __future__ import annotations

import csv
import math
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "results" / "tables"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"
SUPPORTED_CLASS_SUMMARY = ANALYSIS_DIR / "real_uav_finetune_supported_class_summary.csv"
STABLE_CLASS_SUMMARY = ANALYSIS_DIR / "real_uav_finetune_stable_class_summary.csv"
PER_CLASS_SUPPORTED_METRICS = ANALYSIS_DIR / "real_uav_finetune_per_class_supported_metrics.csv"

SYN_COLOR = colors.HexColor("#7A869A")
FINE_COLOR = colors.HexColor("#2878B5")
GRID_COLOR = colors.HexColor("#D7DCE2")
TEXT_COLOR = colors.HexColor("#20242A")
MUTED_COLOR = colors.HexColor("#5F6B7A")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row[key]
    if value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


def add_centered(d: Drawing | Group, x: float, y: float, text: str, size: int = 8, color=TEXT_COLOR) -> None:
    d.add(String(x, y, text, fontName="Helvetica", fontSize=size, fillColor=color, textAnchor="middle"))


def add_right(d: Drawing | Group, x: float, y: float, text: str, size: int = 8, color=TEXT_COLOR) -> None:
    d.add(String(x, y, text, fontName="Helvetica", fontSize=size, fillColor=color, textAnchor="end"))


def add_left(d: Drawing | Group, x: float, y: float, text: str, size: int = 8, color=TEXT_COLOR) -> None:
    d.add(String(x, y, text, fontName="Helvetica", fontSize=size, fillColor=color))


def short_label(text: str, max_width: float, size: int) -> str:
    if stringWidth(text, "Helvetica", size) <= max_width:
        return text
    out = text
    while out and stringWidth(out + "...", "Helvetica", size) > max_width:
        out = out[:-1]
    return out + "..."


def draw_axis(d: Drawing | Group, x: float, y: float, width: float, height: float, ticks: list[float]) -> None:
    d.add(Line(x, y, x + width, y, strokeColor=TEXT_COLOR, strokeWidth=0.7))
    for tick in ticks:
        tx = x + tick * width
        d.add(Line(tx, y, tx, y + height, strokeColor=GRID_COLOR, strokeWidth=0.45))
        d.add(Line(tx, y, tx, y - 3, strokeColor=TEXT_COLOR, strokeWidth=0.7))
        add_centered(d, tx, y - 14, f"{tick:.1f}", 7, MUTED_COLOR)


def make_overall_plot(summary_rows: list[dict[str, str]], stable_rows: list[dict[str, str]]) -> None:
    width, height = 510, 300
    d = Drawing(width, height)

    add_left(d, 30, 277, "Real UAV transfer: synthetic-only vs real-image fine-tuned", 12)
    add_left(d, 30, 262, "Metrics computed on the same 39-image evaluation subset; confidence >= 0.25, IoU >= 0.50.", 8, MUTED_COLOR)
    add_left(d, 30, 248, "Stable classes (whitevan, suv, male): mAP50 0.008 -> 0.432; F1 0.014 -> 0.563.", 8, MUTED_COLOR)

    plot_x, plot_y = 60, 58
    plot_w, plot_h = 405, 172
    metrics = [
        ("Precision", "precision_at_0.25_iou50"),
        ("Recall", "recall_at_0.25_iou50"),
        ("F1", "f1_at_0.25_iou50"),
        ("mAP50", "mAP50_supported_classes"),
    ]
    runs = {row["run"]: row for row in summary_rows}
    group_w = plot_w / len(metrics)
    bar_w = 19

    draw_axis(d, plot_x, plot_y, plot_w, plot_h, [0, 0.25, 0.5, 0.75, 1.0])
    d.add(Line(plot_x, plot_y, plot_x, plot_y + plot_h, strokeColor=TEXT_COLOR, strokeWidth=0.7))
    add_centered(d, plot_x + plot_w / 2, 24, "score", 8, MUTED_COLOR)

    for i, (label, key) in enumerate(metrics):
        cx = plot_x + i * group_w + group_w / 2
        values = [
            ("synthetic-only", as_float(runs["synthetic_only"], key), SYN_COLOR, -bar_w / 2 - 2),
            ("fine-tuned", as_float(runs["finetuned"], key), FINE_COLOR, bar_w / 2 + 2),
        ]
        for _, value, color, offset in values:
            bar_h = max(0, min(value, 1)) * plot_h
            x = cx + offset - bar_w / 2
            d.add(Rect(x, plot_y, bar_w, bar_h, fillColor=color, strokeColor=None))
            add_centered(d, x + bar_w / 2, plot_y + bar_h + 5, f"{value:.2f}", 7, TEXT_COLOR)
        add_centered(d, cx, plot_y - 30, label, 8, TEXT_COLOR)

    legend_y = 230
    d.add(Rect(338, legend_y, 10, 10, fillColor=SYN_COLOR, strokeColor=None))
    add_left(d, 353, legend_y + 2, "synthetic-only", 8, TEXT_COLOR)
    d.add(Rect(430, legend_y, 10, 10, fillColor=FINE_COLOR, strokeColor=None))
    add_left(d, 445, legend_y + 2, "fine-tuned", 8, TEXT_COLOR)

    renderPDF.drawToFile(d, str(FIGURE_DIR / "real_uav_finetune_overall_comparison.pdf"))


def make_per_class_plot(per_class_rows: list[dict[str, str]]) -> None:
    width, height = 510, 360
    d = Drawing(width, height)

    add_left(d, 30, 337, "Per-class AP50 after real-image fine-tuning", 12)
    add_left(d, 30, 322, "Classes with one to three objects are shown but should be interpreted as small-sample diagnostics.", 8, MUTED_COLOR)

    rows_by_class: dict[str, dict[str, dict[str, str]]] = {}
    support_by_class: dict[str, int] = {}
    for row in per_class_rows:
        cls = row["class_name"]
        rows_by_class.setdefault(cls, {})[row["run"]] = row
        support_by_class[cls] = int(row["gt_support"])

    class_order = ["suv", "whitevan", "male", "rock", "tower", "tree", "barrel", "tank", "container", "tent"]
    plot_x, plot_y = 132, 42
    plot_w = 330
    row_h = 24
    max_y = plot_y + row_h * len(class_order)

    draw_axis(d, plot_x, plot_y, plot_w, row_h * len(class_order), [0, 0.25, 0.5, 0.75, 1.0])
    d.add(Line(plot_x, plot_y, plot_x, max_y, strokeColor=TEXT_COLOR, strokeWidth=0.7))
    add_centered(d, plot_x + plot_w / 2, 12, "AP50", 8, MUTED_COLOR)

    for i, cls in enumerate(class_order):
        y = max_y - (i + 1) * row_h + 5
        add_right(d, plot_x - 12, y + 3, short_label(cls, 74, 8), 8, TEXT_COLOR)
        add_left(d, 30, y + 3, f"n={support_by_class[cls]}", 7, MUTED_COLOR)
        syn = as_float(rows_by_class[cls]["synthetic_only"], "ap50")
        fine = as_float(rows_by_class[cls]["finetuned"], "ap50")
        for value, color, offset in [(syn, SYN_COLOR, 0), (fine, FINE_COLOR, 9)]:
            bar_h = 7
            bar_w = max(0, min(value, 1)) * plot_w
            d.add(Rect(plot_x, y + offset, bar_w, bar_h, fillColor=color, strokeColor=None))
            if value >= 0.12:
                add_left(d, plot_x + bar_w + 4, y + offset - 1, f"{value:.2f}", 7, TEXT_COLOR)
            elif value > 0:
                add_left(d, plot_x + 3, y + offset - 1, f"{value:.2f}", 7, TEXT_COLOR)

    legend_y = 298
    d.add(Rect(318, legend_y, 10, 10, fillColor=SYN_COLOR, strokeColor=None))
    add_left(d, 333, legend_y + 2, "synthetic-only", 8, TEXT_COLOR)
    d.add(Rect(410, legend_y, 10, 10, fillColor=FINE_COLOR, strokeColor=None))
    add_left(d, 425, legend_y + 2, "fine-tuned", 8, TEXT_COLOR)

    renderPDF.drawToFile(d, str(FIGURE_DIR / "real_uav_finetune_per_class_ap50.pdf"))


def make_confusion_comparison() -> None:
    width, height = 510, 330
    d = Drawing(width, height)
    add_left(d, 30, 307, "Error balance at confidence >= 0.25 and IoU >= 0.50", 12)
    add_left(d, 30, 292, "Fine-tuning converts many misses into true positives, but false negatives remain substantial.", 8, MUTED_COLOR)

    stable_rows = read_rows(STABLE_CLASS_SUMMARY)
    summary_rows = read_rows(SUPPORTED_CLASS_SUMMARY)
    panels = [
        ("All supported classes", summary_rows, "gt_objects_supported_classes"),
        ("Stable classes", stable_rows, "gt_objects"),
    ]
    colors_by_part = {
        "tp": colors.HexColor("#2E8B57"),
        "fp": colors.HexColor("#D88935"),
        "fn": colors.HexColor("#B54A4A"),
    }

    panel_w = 155
    plot_y = 72
    for p_idx, (title, rows, gt_key) in enumerate(panels):
        x0 = 105 + p_idx * 225
        add_centered(d, x0 + panel_w / 2, 252, title, 9, TEXT_COLOR)
        max_total = max(
            int(row["tp_at_0.25_iou50"]) + int(row["fp_at_0.25_iou50"]) + int(row["fn_at_0.25_iou50"])
            for row in rows
        )
        for r_idx, row in enumerate(rows):
            run_label = "synthetic-only" if row["run"] == "synthetic_only" else "fine-tuned"
            y = plot_y + (1 - r_idx) * 62
            add_right(d, x0 - 10, y + 15, run_label, 8, TEXT_COLOR)
            cursor = x0
            for part in ["tp", "fp", "fn"]:
                value = int(row[f"{part}_at_0.25_iou50"])
                w = value / max_total * panel_w
                d.add(Rect(cursor, y, w, 24, fillColor=colors_by_part[part], strokeColor=None))
                if w > 18:
                    add_centered(d, cursor + w / 2, y + 8, str(value), 7, colors.white)
                cursor += w
        d.add(Line(x0, plot_y - 8, x0 + panel_w, plot_y - 8, strokeColor=TEXT_COLOR, strokeWidth=0.7))
        add_centered(d, x0 + panel_w / 2, plot_y - 24, "count", 8, MUTED_COLOR)

    legend_y = 35
    legend_x = 165
    for label, part in [("TP", "tp"), ("FP", "fp"), ("FN", "fn")]:
        d.add(Rect(legend_x, legend_y, 10, 10, fillColor=colors_by_part[part], strokeColor=None))
        add_left(d, legend_x + 15, legend_y + 2, label, 8, TEXT_COLOR)
        legend_x += 55

    renderPDF.drawToFile(d, str(FIGURE_DIR / "real_uav_finetune_error_balance.pdf"))


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    summary_rows = read_rows(SUPPORTED_CLASS_SUMMARY)
    stable_rows = read_rows(STABLE_CLASS_SUMMARY)
    per_class_rows = read_rows(PER_CLASS_SUPPORTED_METRICS)
    make_overall_plot(summary_rows, stable_rows)
    make_per_class_plot(per_class_rows)
    make_confusion_comparison()


if __name__ == "__main__":
    main()
