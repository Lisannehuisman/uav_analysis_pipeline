from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENE_VIEW_CSV = ROOT / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
GEOMETRY_PRIOR_CSV = ROOT / "geometry_aware_fusion_analysis" / "outputs" / "geometry_priors.csv"
GT_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json"
PRED_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "predictions" / "YOLOv8l_M4_test_predictions.json"
RAW_IMAGE_DIR = Path(r"C:\DATA\airsim\thesis\captures\COPY OF ACTUAL DATA!\images")
OUTPUT_DIR = ROOT / "geometry_aware_fusion_analysis" / "outputs"

DEFAULT_FILE_1 = "S0-SM_tank2-elmid-radmid-az000.png"
DEFAULT_FILE_2 = "S0-SM_tank2-elmid-radnear-az315.png"
CARD_METHODS = [
    "single_view_1",
    "single_view_2",
    "mean_quality",
    "best_box",
    "geometry_prior_selector",
    "geometry_calibrated_selector",
    "viewpoint_cell_prior_selector",
    "noisy_or_best_iou",
    "support_weighted_or",
    "geometry_weighted_or_best_iou",
    "viewpoint_cell_weighted_or_best_iou",
    "hybrid_geometry_cell_weighted_or_best_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real-data multiview illustration with actual AirSim images, GT/pred boxes, and method cards."
    )
    parser.add_argument("--file-1", default=DEFAULT_FILE_1, help="First image file name.")
    parser.add_argument("--file-2", default=DEFAULT_FILE_2, help="Second image file name.")
    parser.add_argument(
        "--output-prefix",
        default="real_box_method_illustration",
        help="Prefix for output files inside geometry_aware_fusion_analysis/outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def box_center_distance(box: list[float], width: float, height: float) -> float:
    x, y, w, h = box
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    return ((cx - (width / 2.0)) ** 2 + (cy - (height / 2.0)) ** 2) ** 0.5


def method_label(method_id: str) -> str:
    return {
        "single_view_1": "Single View 1",
        "single_view_2": "Single View 2",
        "mean_quality": "Mean quality",
        "best_box": "Best box (max)",
        "geometry_prior_selector": "Geometry prior selector",
        "geometry_calibrated_selector": "Geometry-calibrated selector",
        "viewpoint_cell_prior_selector": "Viewpoint-cell selector",
        "noisy_or_best_iou": "Noisy-OR + best IoU",
        "support_weighted_or": "Support-weighted OR",
        "geometry_weighted_or_best_iou": "Geometry-weighted OR + best IoU",
        "viewpoint_cell_weighted_or_best_iou": "Viewpoint-cell OR + best IoU",
        "hybrid_geometry_cell_weighted_or_best_iou": "Hybrid geometry+cell OR",
    }[method_id]


def method_family(method_id: str) -> str:
    return {
        "single_view_1": "Reference",
        "single_view_2": "Reference",
        "mean_quality": "Naive averaging",
        "best_box": "Selection",
        "geometry_prior_selector": "Geometry-aware selection",
        "geometry_calibrated_selector": "Geometry-aware selection",
        "viewpoint_cell_prior_selector": "Geometry-aware selection",
        "noisy_or_best_iou": "Evidence accumulation",
        "support_weighted_or": "Evidence accumulation",
        "geometry_weighted_or_best_iou": "Geometry-aware accumulation",
        "viewpoint_cell_weighted_or_best_iou": "Geometry-aware accumulation",
        "hybrid_geometry_cell_weighted_or_best_iou": "Geometry-aware accumulation",
    }[method_id]


def family_colors(family: str) -> tuple[str, str]:
    return {
        "Reference": ("#f7f7f7", "#808080"),
        "Naive averaging": ("#f6f1ff", "#8467c7"),
        "Selection": ("#fff1f1", "#d26565"),
        "Geometry-aware selection": ("#fff7e8", "#d39b2d"),
        "Evidence accumulation": ("#eef8ee", "#5d9f68"),
        "Geometry-aware accumulation": ("#eef4f7", "#5d8899"),
    }[family]


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    r, g, b = hex_rgb(color)
    return (r, g, b, alpha)


def font_candidates(bold: bool = False, mono: bool = False) -> list[str]:
    if mono:
        return [
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\cour.ttf",
            r"C:\Windows\Fonts\lucon.ttf",
        ]
    if bold:
        return [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
        ]
    return [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in font_candidates(bold=bold, mono=mono):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


TITLE_FONT = load_font(44, bold=True)
SUBTITLE_FONT = load_font(20)
PANEL_TITLE_FONT = load_font(24, bold=True)
PANEL_TEXT_FONT = load_font(18, mono=True)
CARD_TITLE_FONT = load_font(22, bold=True)
CARD_FAMILY_FONT = load_font(17)
CARD_TEXT_FONT = load_font(17, mono=True)
NOTE_TITLE_FONT = load_font(24, bold=True)
NOTE_TEXT_FONT = load_font(18)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def draw_box(draw: ImageDraw.ImageDraw, xywh: tuple[int, int, int, int], fill: str, outline: str, radius: int = 18, width: int = 3) -> None:
    x, y, w, h = xywh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#666666", width: int = 6) -> None:
    draw.line([start, end], fill=hex_rgb(color), width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    arrow_size = 18
    px = -uy
    py = ux
    tip = end
    left = (
        int(end[0] - (ux * arrow_size) + (px * arrow_size * 0.6)),
        int(end[1] - (uy * arrow_size) + (py * arrow_size * 0.6)),
    )
    right = (
        int(end[0] - (ux * arrow_size) - (px * arrow_size * 0.6)),
        int(end[1] - (uy * arrow_size) - (py * arrow_size * 0.6)),
    )
    draw.polygon([tip, left, right], fill=hex_rgb(color))


def load_scene_records() -> pd.DataFrame:
    scene_df = pd.read_csv(SCENE_VIEW_CSV)
    prior_df = pd.read_csv(GEOMETRY_PRIOR_CSV)
    joined = scene_df.merge(
        prior_df[
            [
                "file_name",
                "predicted_geometry_prior",
                "geometry_weight",
                "predicted_viewpoint_cell_prior",
                "viewpoint_cell_weight",
                "predicted_hybrid_prior",
                "hybrid_weight",
            ]
        ],
        on="file_name",
        how="left",
    )
    numeric_columns = [
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
        "predicted_geometry_prior",
        "geometry_weight",
        "predicted_viewpoint_cell_prior",
        "viewpoint_cell_weight",
        "predicted_hybrid_prior",
        "hybrid_weight",
    ]
    for column in numeric_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    return joined


def load_detection_data() -> tuple[dict[str, dict[str, object]], dict[int, list[dict[str, object]]], dict[int, list[dict[str, object]]], dict[str, int]]:
    gt_data = json.loads(GT_JSON.read_text(encoding="utf-8"))
    pred_data = json.loads(PRED_JSON.read_text(encoding="utf-8"))

    image_map = {image["file_name"]: image for image in gt_data["images"]}
    gt_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in gt_data["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    pred_by_image: dict[int, list[dict[str, object]]] = {}
    for prediction in pred_data:
        pred_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    category_name_to_id = {str(category["name"]): int(category["id"]) for category in gt_data["categories"]}
    return image_map, gt_by_image, pred_by_image, category_name_to_id


def choose_focus_gt_box(
    image_record: dict[str, object],
    annotations: list[dict[str, object]],
    target_category_id: int,
) -> dict[str, object]:
    width = float(image_record["width"])
    height = float(image_record["height"])
    candidates = [annotation for annotation in annotations if int(annotation["category_id"]) == target_category_id]
    if not candidates:
        raise ValueError(f"No GT box for target category_id={target_category_id} in {image_record['file_name']}.")
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(candidates, key=lambda annotation: box_center_distance(annotation["bbox"], width, height))
    return ranked[0]


def choose_matched_target_prediction(
    predictions: list[dict[str, object]],
    target_category_id: int,
    gt_box: list[float],
) -> dict[str, object] | None:
    candidates = [prediction for prediction in predictions if int(prediction["category_id"]) == target_category_id]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda prediction: (bbox_iou(prediction["bbox"], gt_box), float(prediction["score"])),
        reverse=True,
    )
    return ranked[0]


def choose_top_predictions(predictions: list[dict[str, object]], limit: int = 10) -> list[dict[str, object]]:
    ranked = sorted(predictions, key=lambda prediction: float(prediction["score"]), reverse=True)
    return ranked[:limit]


def pair_method_scores(record_1: dict[str, object], record_2: dict[str, object]) -> pd.DataFrame:
    q1 = float(record_1["target_strict_quality_iou50"])
    q2 = float(record_2["target_strict_quality_iou50"])
    p1 = float(record_1["target_match_confidence_iou50"])
    p2 = float(record_2["target_match_confidence_iou50"])
    i1 = float(record_1["target_match_iou_at_confidence_iou50"])
    i2 = float(record_2["target_match_iou_at_confidence_iou50"])

    wg1 = float(record_1["geometry_weight"])
    wg2 = float(record_2["geometry_weight"])
    wc1 = float(record_1["viewpoint_cell_weight"])
    wc2 = float(record_2["viewpoint_cell_weight"])
    wh1 = float(record_1["hybrid_weight"])
    wh2 = float(record_2["hybrid_weight"])

    g1 = float(record_1["predicted_geometry_prior"])
    g2 = float(record_2["predicted_geometry_prior"])
    c1 = float(record_1["predicted_viewpoint_cell_prior"])
    c2 = float(record_2["predicted_viewpoint_cell_prior"])

    max_i = max(i1, i2)
    mean_i = (i1 + i2) / 2.0
    support_ratio = 1.0

    gwp1 = min(1.0, p1 * wg1)
    gwp2 = min(1.0, p2 * wg2)
    cwp1 = min(1.0, p1 * wc1)
    cwp2 = min(1.0, p2 * wc2)
    hwp1 = min(1.0, p1 * wh1)
    hwp2 = min(1.0, p2 * wh2)

    scores = [
        {
            "method_id": "single_view_1",
            "score": q1,
            "formula": "q1",
            "interpretation": f"Use only View 1 and keep its matched target box. Score = {q1:.4f}.",
        },
        {
            "method_id": "single_view_2",
            "score": q2,
            "formula": "q2",
            "interpretation": f"Use only View 2 and keep its matched target box. Score = {q2:.4f}.",
        },
        {
            "method_id": "mean_quality",
            "score": (q1 + q2) / 2.0,
            "formula": f"(q1 + q2) / 2 = ({q1:.4f} + {q2:.4f}) / 2",
            "interpretation": "Average the two view qualities directly. Both views count equally, even if one is weaker.",
        },
        {
            "method_id": "best_box",
            "score": max(q1, q2),
            "formula": f"max(q1, q2) = max({q1:.4f}, {q2:.4f})",
            "interpretation": f"Keep only the better single-view target box. Winner: {'View 1' if q1 >= q2 else 'View 2'}.",
        },
        {
            "method_id": "geometry_prior_selector",
            "score": q1 if g1 >= g2 else q2,
            "formula": f"argmax(g1, g2) with g = ({g1:.4f}, {g2:.4f})",
            "interpretation": f"Choose the view with the stronger smooth geometry prior. Winner: {'View 1' if g1 >= g2 else 'View 2'}.",
        },
        {
            "method_id": "geometry_calibrated_selector",
            "score": q1 if (p1 * wg1) >= (p2 * wg2) else q2,
            "formula": f"argmax(p*w_g) with p*w_g = ({p1 * wg1:.4f}, {p2 * wg2:.4f})",
            "interpretation": f"Choose one view after geometry-adjusting confidence. Winner: {'View 1' if (p1 * wg1) >= (p2 * wg2) else 'View 2'}.",
        },
        {
            "method_id": "viewpoint_cell_prior_selector",
            "score": q1 if c1 >= c2 else q2,
            "formula": f"argmax(c1, c2) with c = ({c1:.4f}, {c2:.4f})",
            "interpretation": f"Choose the view from the stronger discrete viewpoint cell prior. Winner: {'View 1' if c1 >= c2 else 'View 2'}.",
        },
        {
            "method_id": "noisy_or_best_iou",
            "score": noisy_or([p1, p2]) * max_i,
            "formula": f"[1-(1-p1)(1-p2)]*max(i) = {noisy_or([p1, p2]):.4f}*{max_i:.4f}",
            "interpretation": "Combine detection evidence from both views, then retain the best localization term.",
        },
        {
            "method_id": "support_weighted_or",
            "score": noisy_or([p1, p2]) * mean_i * support_ratio,
            "formula": f"[1-(1-p1)(1-p2)]*mean(i)*support = {noisy_or([p1, p2]):.4f}*{mean_i:.4f}*{support_ratio:.1f}",
            "interpretation": "Combine both views, but use mean IoU to be more conservative about localization quality.",
        },
        {
            "method_id": "geometry_weighted_or_best_iou",
            "score": noisy_or([gwp1, gwp2]) * max_i,
            "formula": f"[1-(1-p1*w_g1)(1-p2*w_g2)]*max(i) = {noisy_or([gwp1, gwp2]):.4f}*{max_i:.4f}",
            "interpretation": "Accumulate evidence, but first reweight confidence with the smooth geometry prior.",
        },
        {
            "method_id": "viewpoint_cell_weighted_or_best_iou",
            "score": noisy_or([cwp1, cwp2]) * max_i,
            "formula": f"[1-(1-p1*w_c1)(1-p2*w_c2)]*max(i) = {noisy_or([cwp1, cwp2]):.4f}*{max_i:.4f}",
            "interpretation": "Accumulate evidence, but use the discrete viewpoint-cell reliability weights before OR fusion.",
        },
        {
            "method_id": "hybrid_geometry_cell_weighted_or_best_iou",
            "score": noisy_or([hwp1, hwp2]) * max_i,
            "formula": f"[1-(1-p1*w_h1)(1-p2*w_h2)]*max(i) = {noisy_or([hwp1, hwp2]):.4f}*{max_i:.4f}",
            "interpretation": "Accumulate evidence with the hybrid of smooth geometry and discrete cell priors.",
        },
    ]
    result = pd.DataFrame(scores)
    result["method_label"] = result["method_id"].map(method_label)
    result["method_family"] = result["method_id"].map(method_family)
    result["rank_desc"] = result["score"].rank(method="min", ascending=False).astype(int)
    return result


def resize_and_draw_image(
    file_name: str,
    gt_box: list[float],
    matched_pred: dict[str, object] | None,
    predictions: list[dict[str, object]],
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    image = Image.open(RAW_IMAGE_DIR / file_name).convert("RGBA")
    image = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    sx = panel_width / 1920.0
    sy = panel_height / 1080.0

    for prediction in predictions:
        x, y, w, h = [float(value) for value in prediction["bbox"]]
        draw.rectangle(
            (x * sx, y * sy, (x + w) * sx, (y + h) * sy),
            outline=rgba("#f4d03f", 180),
            width=3,
        )

    gx, gy, gw, gh = [float(value) for value in gt_box]
    gt_pad = 6.0
    draw.rectangle(
        ((gx - gt_pad) * sx, (gy - gt_pad) * sy, (gx + gw + gt_pad) * sx, (gy + gh + gt_pad) * sy),
        outline=hex_rgb("#43a047"),
        width=9,
    )

    if matched_pred is not None:
        px, py, pw, ph = [float(value) for value in matched_pred["bbox"]]
        draw.rectangle(
            (px * sx, py * sy, (px + pw) * sx, (py + ph) * sy),
            outline=hex_rgb("#ff8c42"),
            width=7,
        )

    return Image.alpha_composite(image, overlay)


def add_label_block(
    canvas: Image.Image,
    xy: tuple[int, int],
    title: str,
    body_lines: list[str],
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    width: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    body = "\n".join(body_lines)
    title_w, title_h = text_size(draw, title, title_font)
    body_w, body_h = text_size(draw, body, body_font)
    h = 20 + title_h + 10 + body_h + 20
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + h), radius=18, fill=(255, 255, 255, 228), outline=(204, 204, 204), width=2)
    draw.multiline_text((x + 16, y + 14), title, font=title_font, fill=(17, 17, 17), spacing=4)
    draw.multiline_text((x + 16, y + 14 + title_h + 10), body, font=body_font, fill=(17, 17, 17), spacing=4)


def build_png(
    output_path: Path,
    file_1: str,
    file_2: str,
    record_1: dict[str, object],
    record_2: dict[str, object],
    gt_box_1: list[float],
    gt_box_2: list[float],
    matched_pred_1: dict[str, object] | None,
    matched_pred_2: dict[str, object] | None,
    top_predictions_1: list[dict[str, object]],
    top_predictions_2: list[dict[str, object]],
    pair_scores_df: pd.DataFrame,
) -> None:
    canvas = Image.new("RGBA", (3600, 3200), hex_rgb("#f7f7f4") + (255,))
    draw = ImageDraw.Draw(canvas)

    draw.text((80, 60), "Real-Data Example: How The Multiview Methods Process Two AirSim Views", font=TITLE_FONT, fill=(17, 17, 17))
    subtitle = (
        "Green = target GT box, orange = matched target prediction, yellow = other final detector boxes from YOLOv8l. "
        "Important: the current multiview methods do not fuse box coordinates across views; they combine per-view target evidence "
        "and usually keep a score plus the best localization term."
    )
    draw.multiline_text((80, 132), wrap(subtitle, 120), font=SUBTITLE_FONT, fill=(51, 51, 51), spacing=6)

    panel_w = 1480
    panel_h = 832
    left_x = 90
    right_x = 2030
    top_y = 300

    panel_1 = resize_and_draw_image(file_1, gt_box_1, matched_pred_1, top_predictions_1, panel_w, panel_h)
    panel_2 = resize_and_draw_image(file_2, gt_box_2, matched_pred_2, top_predictions_2, panel_w, panel_h)
    canvas.alpha_composite(panel_1, (left_x, top_y))
    canvas.alpha_composite(panel_2, (right_x, top_y))

    q1 = float(record_1["target_strict_quality_iou50"])
    q2 = float(record_2["target_strict_quality_iou50"])
    p1 = float(record_1["target_match_confidence_iou50"])
    p2 = float(record_2["target_match_confidence_iou50"])
    i1 = float(record_1["target_match_iou_at_confidence_iou50"])
    i2 = float(record_2["target_match_iou_at_confidence_iou50"])
    g1 = float(record_1["predicted_geometry_prior"])
    g2 = float(record_2["predicted_geometry_prior"])
    c1 = float(record_1["predicted_viewpoint_cell_prior"])
    c2 = float(record_2["predicted_viewpoint_cell_prior"])
    wg1 = float(record_1["geometry_weight"])
    wg2 = float(record_2["geometry_weight"])
    wc1 = float(record_1["viewpoint_cell_weight"])
    wc2 = float(record_2["viewpoint_cell_weight"])

    add_label_block(
        canvas,
        (left_x + 18, top_y + 18),
        "View 1",
        [
            f"file = {file_1}",
            f"viewpoint = {record_1['viewpoint']}",
            f"q1 = {q1:.4f}   p1 = {p1:.4f}   i1 = {i1:.4f}",
            f"g1 = {g1:.4f}   w_g1 = {wg1:.4f}",
            f"c1 = {c1:.4f}   w_c1 = {wc1:.4f}",
        ],
        PANEL_TITLE_FONT,
        PANEL_TEXT_FONT,
        650,
    )
    add_label_block(
        canvas,
        (right_x + 18, top_y + 18),
        "View 2",
        [
            f"file = {file_2}",
            f"viewpoint = {record_2['viewpoint']}",
            f"q2 = {q2:.4f}   p2 = {p2:.4f}   i2 = {i2:.4f}",
            f"g2 = {g2:.4f}   w_g2 = {wg2:.4f}",
            f"c2 = {c2:.4f}   w_c2 = {wc2:.4f}",
        ],
        PANEL_TITLE_FONT,
        PANEL_TEXT_FONT,
        650,
    )

    conn_y1 = 1190
    conn_y2 = 1325
    draw_arrow(draw, (left_x + (panel_w // 2), top_y + panel_h + 8), (left_x + (panel_w // 2), conn_y1))
    draw_arrow(draw, (right_x + (panel_w // 2), top_y + panel_h + 8), (right_x + (panel_w // 2), conn_y1))
    draw.text((980, conn_y2 - 12), "Per-view matched target signals flow downward into the combination rules", font=load_font(24), fill=(51, 51, 51))

    draw.text((80, 1390), "Same two views, processed by different methods", font=load_font(34, bold=True), fill=(17, 17, 17))
    draw.multiline_text(
        (80, 1442),
        wrap(
            "Cards below show the exact rule used on this pair. Selection methods end with one chosen box. "
            "OR-style methods use both views and output a pair score; they do not average box coordinates across image planes.",
            130,
        ),
        font=SUBTITLE_FONT,
        fill=(51, 51, 51),
        spacing=6,
    )

    cols = 4
    card_w = 825
    card_h = 330
    x_gap = 40
    y_gap = 35
    start_x = 80
    start_y = 1545

    score_lookup = pair_scores_df.set_index("method_id")
    for index, method_id in enumerate(CARD_METHODS):
        row_idx = index // cols
        col_idx = index % cols
        x = start_x + col_idx * (card_w + x_gap)
        y = start_y + row_idx * (card_h + y_gap)
        method_row = score_lookup.loc[method_id]
        family = str(method_row["method_family"])
        fill, outline = family_colors(family)
        draw_box(draw, (x, y, card_w, card_h), fill, outline, radius=18, width=3)
        draw.text((x + 18, y + 16), str(method_row["method_label"]), font=CARD_TITLE_FONT, fill=(17, 17, 17))
        draw.text((x + 18, y + 50), family, font=CARD_FAMILY_FONT, fill=hex_rgb(outline))
        body = "\n".join(
            [
                wrap(str(method_row["formula"]), 40),
                f"score = {float(method_row['score']):.4f}",
                wrap(str(method_row["interpretation"]), 42),
            ]
        )
        draw.multiline_text((x + 18, y + 84), body, font=CARD_TEXT_FONT, fill=(34, 34, 34), spacing=6)

    note_x = 80
    note_y = 2635
    note_w = 3440
    note_h = 430
    draw_box(draw, (note_x, note_y, note_w, note_h), "#edf6ed", "#6ea36e", radius=20, width=3)
    draw.text((note_x + 18, note_y + 16), "Interpretation key", font=NOTE_TITLE_FONT, fill=(16, 61, 16))
    note_text = (
        "1. `Mean quality` is usually weak because one weaker view drags the stronger one down.\n"
        "2. `Best box` and the selector methods still throw one view away.\n"
        "3. The OR-style methods are strong because both views stay alive as evidence.\n"
        "4. The geometry-aware OR variants do not invent a new 2D fused box; they mainly reweight how much each view counts before OR fusion.\n"
        "5. This is why the main conceptual jump in your results is selection versus evidence accumulation, and only after that geometry-aware refinement."
    )
    draw.multiline_text((note_x + 18, note_y + 62), note_text, font=NOTE_TEXT_FONT, fill=(31, 56, 31), spacing=8)

    canvas.convert("RGB").save(output_path)


def build_markdown(
    output_path: Path,
    png_path: Path,
    file_1: str,
    file_2: str,
    pair_scores_df: pd.DataFrame,
) -> None:
    sorted_df = pair_scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    lines = [
        "# Real Box Method Illustration",
        "",
        f"![Real-data method illustration]({png_path.as_posix()})",
        "",
        "## What This Figure Is Showing",
        "",
        f"- `View 1`: `{file_1}`",
        f"- `View 2`: `{file_2}`",
        "- Green boxes are the target ground truth.",
        "- Orange boxes are the matched target predictions that define the target metrics used in the analysis.",
        "- Yellow boxes are other final detector outputs from the prediction JSON.",
        "- The important methodological point is that the current multiview rules operate on per-view target evidence, not on a cross-view fused 2D box.",
        "",
        "## Method Scores For This Exact Pair",
        "",
        "| Rank | Method | Family | Score |",
        "| --- | --- | --- | ---: |",
    ]
    for idx, row in sorted_df.iterrows():
        lines.append(f"| {idx + 1} | `{row['method_label']}` | {row['method_family']} | {float(row['score']):.4f} |")

    lines.extend(
        [
            "",
            "## How To Read It",
            "",
            "- If a method is a `selector`, it ends by keeping one view and throwing the other away.",
            "- If a method is `OR-style accumulation`, it lets both views contribute evidence to the final pair score.",
            "- If a method is `geometry-aware`, the geometry part mostly changes how much we trust each view before combining them.",
            "- That is why the key step in your results is usually not `geometry` versus `no geometry`, but `selection` versus `evidence accumulation`.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)

    scene_df = load_scene_records()
    image_map, gt_by_image, pred_by_image, category_name_to_id = load_detection_data()

    pair_df = scene_df.loc[scene_df["file_name"].isin([args.file_1, args.file_2])].copy()
    if len(pair_df) != 2:
        raise ValueError(f"Expected exactly 2 rows for the requested files, found {len(pair_df)}.")

    record_1 = pair_df.loc[pair_df["file_name"] == args.file_1].iloc[0].to_dict()
    record_2 = pair_df.loc[pair_df["file_name"] == args.file_2].iloc[0].to_dict()

    if str(record_1["scene_key"]) != str(record_2["scene_key"]):
        raise ValueError("The two selected files must come from the same scene_key.")

    target_class_name = str(record_1["target_class"])
    if target_class_name != str(record_2["target_class"]):
        raise ValueError("The two selected files must use the same target class.")
    if target_class_name not in category_name_to_id:
        raise KeyError(f"Target class {target_class_name} not found in COCO categories.")
    target_category_id = category_name_to_id[target_class_name]

    image_record_1 = image_map[args.file_1]
    image_record_2 = image_map[args.file_2]
    image_id_1 = int(image_record_1["id"])
    image_id_2 = int(image_record_2["id"])

    gt_box_1 = choose_focus_gt_box(image_record_1, gt_by_image[image_id_1], target_category_id)["bbox"]
    gt_box_2 = choose_focus_gt_box(image_record_2, gt_by_image[image_id_2], target_category_id)["bbox"]
    matched_pred_1 = choose_matched_target_prediction(pred_by_image.get(image_id_1, []), target_category_id, gt_box_1)
    matched_pred_2 = choose_matched_target_prediction(pred_by_image.get(image_id_2, []), target_category_id, gt_box_2)
    top_predictions_1 = choose_top_predictions(pred_by_image.get(image_id_1, []), limit=10)
    top_predictions_2 = choose_top_predictions(pred_by_image.get(image_id_2, []), limit=10)

    pair_scores_df = pair_method_scores(record_1, record_2)

    prefix = args.output_prefix
    png_path = OUTPUT_DIR / f"{prefix}.png"
    csv_path = OUTPUT_DIR / f"{prefix}_pair_scores.csv"
    md_path = OUTPUT_DIR / f"{prefix}.md"

    build_png(
        png_path,
        args.file_1,
        args.file_2,
        record_1,
        record_2,
        gt_box_1,
        gt_box_2,
        matched_pred_1,
        matched_pred_2,
        top_predictions_1,
        top_predictions_2,
        pair_scores_df,
    )
    pair_scores_df.sort_values("score", ascending=False).to_csv(csv_path, index=False)
    build_markdown(md_path, png_path, args.file_1, args.file_2, pair_scores_df)


if __name__ == "__main__":
    main()
