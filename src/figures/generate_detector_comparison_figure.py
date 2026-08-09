from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "recomputed" / "detector_family_comparison" / "standardized_test_eval"
GT_JSON = EVAL_DIR / "ground_truth" / "M4_test_gt.json"
FRCNN_JSON = EVAL_DIR / "predictions" / "Faster_R-CNN_M4_predictions.json"
YOLO_JSON = EVAL_DIR / "predictions" / "YOLOv8l_M4_test_predictions.json"
IMAGE_DIR = ROOT / "data_collection" / "raw_data" / "synthetic_subset" / "images" / "test"
OUT_PATH = ROOT / "results" / "figures" / "figure_3_4_yolo_frcnn_comparison_recomputed.png"

SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MIN_GROUND_TRUTH_OBJECTS = 10

TP_COLOR = (27, 158, 119)
FP_COLOR = (217, 95, 2)
FN_COLOR = (31, 120, 180)
TEXT_COLOR = (24, 24, 24)
MUTED_TEXT = (88, 88, 88)
BOX_EDGE = (185, 185, 185)
PANEL_BG = (252, 252, 250)
WHITE = (255, 255, 255)


@dataclass
class MatchResult:
    matched_predictions: list[dict]
    unmatched_predictions: list[dict]
    missed_ground_truth: list[dict]

    @property
    def tp(self) -> int:
        return len(self.matched_predictions)

    @property
    def fp(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def fn(self) -> int:
        return len(self.missed_ground_truth)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


@dataclass
class ImageComparison:
    image_id: int
    file_name: str
    ground_truth_count: int
    contrast_score: int
    yolo_result: MatchResult
    frcnn_result: MatchResult


def iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    return right - left, bottom - top


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int = 4,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    for offset in range(width):
        x = x1
        while x < x2:
            draw.line([(x, y1 + offset), (min(x + dash, x2), y1 + offset)], fill=color, width=1)
            draw.line([(x, y2 - offset), (min(x + dash, x2), y2 - offset)], fill=color, width=1)
            x += dash + gap

        y = y1
        while y < y2:
            draw.line([(x1 + offset, y), (x1 + offset, min(y + dash, y2))], fill=color, width=1)
            draw.line([(x2 - offset, y), (x2 - offset, min(y + dash, y2))], fill=color, width=1)
            y += dash + gap


def match_predictions(predictions: list[dict], ground_truth: list[dict]) -> MatchResult:
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    used_gt_indices: set[int] = set()
    matched_predictions: list[dict] = []
    unmatched_predictions: list[dict] = []

    for prediction in predictions:
        best_iou = 0.0
        best_gt_index: int | None = None
        for gt_index, gt_ann in enumerate(ground_truth):
            if gt_index in used_gt_indices:
                continue
            if prediction["category_id"] != gt_ann["category_id"]:
                continue
            candidate_iou = iou_xywh(prediction["bbox"], gt_ann["bbox"])
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
            used_gt_indices.add(best_gt_index)
            matched_predictions.append(prediction)
        else:
            unmatched_predictions.append(prediction)

    missed_ground_truth = [
        gt_ann for gt_index, gt_ann in enumerate(ground_truth) if gt_index not in used_gt_indices
    ]
    return MatchResult(
        matched_predictions=matched_predictions,
        unmatched_predictions=unmatched_predictions,
        missed_ground_truth=missed_ground_truth,
    )


def prediction_maps(predictions: list[dict]) -> dict[int, list[dict]]:
    per_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= SCORE_THRESHOLD:
            per_image[int(pred["image_id"])].append(pred)
    return per_image


def select_representative_frame(gt: dict, frcnn_predictions: list[dict], yolo_predictions: list[dict]) -> ImageComparison:
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)

    frcnn_by_image = prediction_maps(frcnn_predictions)
    yolo_by_image = prediction_maps(yolo_predictions)

    best: ImageComparison | None = None
    best_key: tuple[int, int, int, int] | None = None

    for image_info in gt["images"]:
        image_id = int(image_info["id"])
        ground_truth = gt_by_image[image_id]
        if len(ground_truth) < MIN_GROUND_TRUTH_OBJECTS:
            continue

        frcnn_result = match_predictions(frcnn_by_image[image_id], ground_truth)
        yolo_result = match_predictions(yolo_by_image[image_id], ground_truth)

        contrast_score = (
            (yolo_result.tp - frcnn_result.tp)
            + (frcnn_result.fp - yolo_result.fp)
            + (frcnn_result.fn - yolo_result.fn)
        )

        tie_break = (
            contrast_score,
            len(ground_truth),
            yolo_result.tp - frcnn_result.tp,
            frcnn_result.fp - yolo_result.fp,
        )

        candidate = ImageComparison(
            image_id=image_id,
            file_name=str(image_info["file_name"]),
            ground_truth_count=len(ground_truth),
            contrast_score=contrast_score,
            yolo_result=yolo_result,
            frcnn_result=frcnn_result,
        )

        if best is None or tie_break > best_key:
            best = candidate
            best_key = tie_break

    if best is None:
        raise RuntimeError("No representative frame could be selected from the M4 test split.")
    return best


def tint(color: tuple[int, int, int], amount: float = 0.18) -> tuple[int, int, int]:
    return tuple(round(channel + (255 - channel) * amount) for channel in color)


def draw_summary_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: str,
    accent: tuple[int, int, int],
    label_font,
    value_font,
) -> int:
    label_w, _ = text_size(draw, label, label_font)
    value_w, _ = text_size(draw, value, value_font)
    chip_h = 58
    chip_w = 70 + label_w + value_w
    draw.rounded_rectangle(
        (x, y, x + chip_w, y + chip_h),
        radius=16,
        fill=tint(accent),
        outline=accent,
        width=2,
    )
    draw.rounded_rectangle((x + 16, y + 14, x + 40, y + 38), radius=8, fill=accent)
    draw.text((x + 52, y + 12), label, font=label_font, fill=TEXT_COLOR)
    draw.text((x + chip_w - value_w - 18, y + 11), value, font=value_font, fill=TEXT_COLOR)
    return chip_w


def draw_model_legend(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    model_label: str,
    result: MatchResult,
    gt_count: int,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=22, fill=WHITE, outline=BOX_EDGE, width=2)

    title_font = load_font(34, bold=True)
    chip_label_font = load_font(24)
    chip_value_font = load_font(28, bold=True)
    meta_font = load_font(24)
    meta_value_font = load_font(26, bold=True)

    draw.text((x1 + 28, y1 + 22), model_label, font=title_font, fill=TEXT_COLOR)

    chip_y = y1 + 78
    chip_x = x1 + 28
    chip_gap = 18
    chip_x += draw_summary_chip(draw, chip_x, chip_y, "TP", str(result.tp), TP_COLOR, chip_label_font, chip_value_font) + chip_gap
    chip_x += draw_summary_chip(draw, chip_x, chip_y, "FP", str(result.fp), FP_COLOR, chip_label_font, chip_value_font) + chip_gap
    chip_x += draw_summary_chip(draw, chip_x, chip_y, "FN", str(result.fn), FN_COLOR, chip_label_font, chip_value_font) + chip_gap
    draw_summary_chip(draw, chip_x, chip_y, "GT", str(gt_count), (120, 120, 120), chip_label_font, chip_value_font)

    metric_y = y1 + 154
    precision_label = "Precision"
    recall_label = "Recall"
    precision_value = f"{result.precision:.2f}"
    recall_value = f"{result.recall:.2f}"

    draw.text((x1 + 28, metric_y), precision_label, font=meta_font, fill=MUTED_TEXT)
    draw.text((x1 + 150, metric_y - 2), precision_value, font=meta_value_font, fill=TEXT_COLOR)
    draw.text((x1 + 300, metric_y), recall_label, font=meta_font, fill=MUTED_TEXT)
    draw.text((x1 + 392, metric_y - 2), recall_value, font=meta_value_font, fill=TEXT_COLOR)


def overlay_predictions(base_image: Image.Image, result: MatchResult) -> Image.Image:
    canvas = base_image.copy()
    draw = ImageDraw.Draw(canvas)

    for pred in result.matched_predictions:
        x, y, w, h = pred["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=TP_COLOR, width=5)

    for pred in result.unmatched_predictions:
        x, y, w, h = pred["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=FP_COLOR, width=5)

    for gt_ann in result.missed_ground_truth:
        x, y, w, h = gt_ann["bbox"]
        draw_dashed_rectangle(draw, (x, y, x + w, y + h), FN_COLOR, width=4)

    return canvas


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * (width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_panel(canvas: Image.Image, image: Image.Image, x: int, y: int, panel_width: int) -> int:
    draw = ImageDraw.Draw(canvas)
    panel_image = resize_to_width(image, panel_width)
    panel_x2 = x + panel_image.width
    panel_y2 = y + panel_image.height

    draw.rounded_rectangle((x - 2, y - 2, panel_x2 + 2, panel_y2 + 2), radius=20, fill=WHITE, outline=BOX_EDGE, width=2)
    canvas.paste(panel_image, (x, y))
    return panel_y2


def compose_figure(comparison: ImageComparison, yolo_overlay: Image.Image, frcnn_overlay: Image.Image) -> Image.Image:
    canvas_w = 2380
    margin = 56
    gutter = 44
    panel_w = (canvas_w - margin * 2 - gutter) // 2
    panel_gap = 34
    card_h = 224
    footer_gap = 28
    footer_h = 96

    yolo_panel = resize_to_width(yolo_overlay, panel_w)
    panel_h = yolo_panel.height
    canvas_h = margin + panel_h + panel_gap + card_h + footer_gap + footer_h + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), PANEL_BG)
    draw = ImageDraw.Draw(canvas)

    footer_font = load_font(24)
    panel_y = margin
    yolo_x = margin
    frcnn_x = margin + panel_w + gutter

    add_panel(canvas, yolo_overlay, yolo_x, panel_y, panel_w)
    add_panel(canvas, frcnn_overlay, frcnn_x, panel_y, panel_w)

    card_y = panel_y + panel_h + panel_gap
    draw_model_legend(
        draw,
        (yolo_x, card_y, yolo_x + panel_w, card_y + card_h),
        "YOLOv8l",
        comparison.yolo_result,
        comparison.ground_truth_count,
    )
    draw_model_legend(
        draw,
        (frcnn_x, card_y, frcnn_x + panel_w, card_y + card_h),
        "Faster R-CNN",
        comparison.frcnn_result,
        comparison.ground_truth_count,
    )

    footer_y = card_y + card_h + footer_gap
    footer_cell_w = (canvas_w - margin * 2) // 3
    footer_labels = [
        ("Matched prediction (TP)", TP_COLOR, "line"),
        ("Unmatched prediction (FP)", FP_COLOR, "line"),
        ("Missed ground-truth object (FN)", FN_COLOR, "dash"),
    ]

    for index, (label, color, kind) in enumerate(footer_labels):
        cell_x = margin + index * footer_cell_w
        icon_x = cell_x + 18
        icon_y = footer_y + 30
        if kind == "line":
            draw.line([(icon_x, icon_y), (icon_x + 92, icon_y)], fill=color, width=8)
        else:
            draw_dashed_rectangle(draw, (icon_x, icon_y - 18, icon_x + 92, icon_y + 18), color, width=4)
        draw.text((icon_x + 118, footer_y + 10), label, font=footer_font, fill=TEXT_COLOR)

    return canvas


def main() -> None:
    gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
    frcnn_predictions = json.loads(FRCNN_JSON.read_text(encoding="utf-8"))
    yolo_predictions = json.loads(YOLO_JSON.read_text(encoding="utf-8"))

    comparison = select_representative_frame(gt, frcnn_predictions, yolo_predictions)
    image_path = IMAGE_DIR / comparison.file_name
    base_image = Image.open(image_path).convert("RGB")

    yolo_overlay = overlay_predictions(base_image, comparison.yolo_result)
    frcnn_overlay = overlay_predictions(base_image, comparison.frcnn_result)
    figure = compose_figure(comparison, yolo_overlay, frcnn_overlay)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.save(OUT_PATH)
    print(
        f"Saved {OUT_PATH} using image {comparison.image_id} "
        f"({comparison.file_name}) | "
        f"YOLO TP/FP/FN={comparison.yolo_result.tp}/{comparison.yolo_result.fp}/{comparison.yolo_result.fn} | "
        f"FRCNN TP/FP/FN={comparison.frcnn_result.tp}/{comparison.frcnn_result.fp}/{comparison.frcnn_result.fn}"
    )


if __name__ == "__main__":
    main()
