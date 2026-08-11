from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
VIEWPOINT_RE = re.compile(r"^(?P<scene>.+)-(?P<viewpoint>el[a-z]+-rad[a-z]+-az(?P<azimuth>\d+))$", re.IGNORECASE)
TARGET_RE = re.compile(r"^S0-SM_([^-]+)-", re.IGNORECASE)
ELEVATION_SORT = {"low": 0, "mid": 1, "high": 2}
RADIUS_SORT = {"near": 0, "mid": 1, "far": 2}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = PROJECT_ROOT / "results" / "recomputed" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json"
DEFAULT_PRED = PROJECT_ROOT / "results" / "recomputed" / "detector_family_comparison" / "standardized_test_eval" / "predictions" / "YOLOv8l_M4_test_predictions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "recomputed" / "two_drone_operational"


@dataclass(frozen=True)
class ViewRecord:
    image_id: int
    file_name: str
    scene_key: str
    viewpoint: str
    elevation: str
    radius: str
    azimuth: int
    target_class: str
    target_class_id: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float
    tp: int
    fp: int
    fn: int
    num_gt: int
    num_pred: int
    target_visible: bool
    target_detected: bool
    target_precision: float
    target_recall: float
    target_f1: float
    target_ap50: float
    target_ap50_95: float
    target_best_iou: float
    target_match_confidence_iou50: float
    target_match_iou_at_confidence_iou50: float
    target_strict_quality_iou50: float
    target_tp: int
    target_fp: int
    target_fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operational method-2 analysis: compare one-view versus two-view drone detection using existing M4 GT and YOLOv8l predictions."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT), help="COCO ground-truth JSON for the M4 test split.")
    parser.add_argument("--pred-json", default=str(DEFAULT_PRED), help="COCO prediction JSON for the full-model detector on the M4 test split.")
    parser.add_argument("--score-threshold", type=float, default=0.001, help="Prediction score threshold used before matching.")
    parser.add_argument("--min-single-support", type=int, default=10, help="Minimum sample count for headline single-viewpoint rankings.")
    parser.add_argument("--min-pair-support", type=int, default=8, help="Minimum sample count for headline pair rankings.")
    parser.add_argument("--min-rescue-support", type=int, default=10, help="Minimum primary-miss count for second-drone rescue rankings.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSVs, plots, and the report.")
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    intersection = inter_w * inter_h

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0, a_max=None) * np.clip(boxes1[:, 3] - boxes1[:, 1], a_min=0, a_max=None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0, a_max=None) * np.clip(boxes2[:, 3] - boxes2[:, 1], a_min=0, a_max=None)
    union = area1[:, None] + area2[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def compute_ap(tp_flags: np.ndarray, confidences: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if tp_flags.size == 0:
        return 0.0

    order = np.argsort(-confidences)
    tp = tp_flags[order].astype(np.float32)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapezoid(mpre, mrec))


def match_detections(pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_classes: np.ndarray, gt_boxes: np.ndarray, gt_classes: np.ndarray, iou_threshold: float) -> tuple[np.ndarray, int, int, int]:
    if len(pred_boxes) == 0:
        return np.zeros((0,), dtype=np.float32), 0, 0, len(gt_boxes)

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_classes = pred_classes[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    tp_flags = np.zeros((len(pred_boxes),), dtype=np.float32)

    for pred_idx, pred_cls in enumerate(pred_classes):
        best_iou = -1.0
        best_gt = -1
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx in matched_gt or gt_cls != pred_cls:
                continue
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = float(iou)
                best_gt = gt_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp_flags[pred_idx] = 1.0

    tp = int(tp_flags.sum())
    fp = int(len(pred_boxes) - tp)
    fn = int(len(gt_boxes) - tp)
    return tp_flags, tp, fp, fn


def evaluate_single_image(pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_classes: np.ndarray, gt_boxes: np.ndarray, gt_classes: np.ndarray) -> dict[str, float]:
    tp_flags_50, tp_50, fp_50, fn_50 = match_detections(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, 0.5)
    precision = safe_divide(tp_50, tp_50 + fp_50)
    recall = safe_divide(tp_50, tp_50 + fn_50)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    ap_values: list[float] = []
    for threshold in IOU_THRESHOLDS:
        tp_flags, _, _, _ = match_detections(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, float(threshold))
        ap_values.append(compute_ap(tp_flags, pred_scores, len(gt_boxes)))

    valid_ap_values = [value for value in ap_values if not math.isnan(value)]
    ap50 = ap_values[0] if not math.isnan(ap_values[0]) else float("nan")
    ap50_95 = float(np.mean(valid_ap_values)) if valid_ap_values else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50,
        "ap50_95": ap50_95,
        "tp": float(tp_50),
        "fp": float(fp_50),
        "fn": float(fn_50),
        "num_gt": float(len(gt_boxes)),
        "num_pred": float(len(pred_boxes)),
    }


def compute_target_match_quality(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5) -> dict[str, float]:
    if len(gt_boxes) == 0:
        return {
            "target_best_iou": 0.0,
            "target_match_confidence_iou50": 0.0,
            "target_match_iou_at_confidence_iou50": 0.0,
            "target_strict_quality_iou50": 0.0,
        }

    if len(pred_boxes) == 0:
        return {
            "target_best_iou": 0.0,
            "target_match_confidence_iou50": 0.0,
            "target_match_iou_at_confidence_iou50": 0.0,
            "target_strict_quality_iou50": 0.0,
        }

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    best_iou = float(np.max(iou_matrix)) if iou_matrix.size else 0.0

    candidate_rows, candidate_cols = np.where(iou_matrix >= iou_threshold)
    if len(candidate_rows) == 0:
        return {
            "target_best_iou": best_iou,
            "target_match_confidence_iou50": 0.0,
            "target_match_iou_at_confidence_iou50": 0.0,
            "target_strict_quality_iou50": 0.0,
        }

    best_index = max(
        range(len(candidate_rows)),
        key=lambda idx: (
            float(pred_scores[candidate_rows[idx]]),
            float(iou_matrix[candidate_rows[idx], candidate_cols[idx]]),
        ),
    )
    match_confidence = float(pred_scores[candidate_rows[best_index]])
    match_iou = float(iou_matrix[candidate_rows[best_index], candidate_cols[best_index]])

    return {
        "target_best_iou": best_iou,
        "target_match_confidence_iou50": match_confidence,
        "target_match_iou_at_confidence_iou50": match_iou,
        "target_strict_quality_iou50": match_confidence * match_iou,
    }


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def strip_prefix(token: str, prefix: str) -> str:
    return token[len(prefix) :] if token.startswith(prefix) else token


def parse_viewpoint_metadata(file_name: str, known_class_names: list[str]) -> tuple[str, str, str, str, int, str]:
    stem = Path(file_name).stem
    match = VIEWPOINT_RE.match(stem)
    if not match:
        raise ValueError(f"Could not parse scene/viewpoint from image name: {file_name}")

    scene_key = match.group("scene")
    viewpoint = match.group("viewpoint").lower()
    parts = viewpoint.split("-")
    elevation = strip_prefix(parts[0], "el")
    radius = strip_prefix(parts[1], "rad")
    azimuth = int(match.group("azimuth"))

    target_match = TARGET_RE.search(stem)
    if not target_match:
        raise ValueError(f"Could not infer target class from image name: {file_name}")

    object_token = target_match.group(1).lower()
    target_class = ""
    for candidate in sorted(known_class_names, key=len, reverse=True):
        if object_token.startswith(candidate.lower()):
            target_class = candidate
            break
    if not target_class:
        fallback = re.match(r"([a-zA-Z]+)", object_token)
        if fallback:
            target_class = fallback.group(1).lower()
        else:
            raise ValueError(f"Could not map filename token '{object_token}' to a class name.")

    return scene_key, viewpoint, elevation, radius, azimuth, target_class


def viewpoint_sort_key(viewpoint: str) -> tuple[int, int, int]:
    parts = viewpoint.split("-")
    elevation = strip_prefix(parts[0], "el")
    radius = strip_prefix(parts[1], "rad")
    azimuth = int(parts[2].replace("az", ""))
    return (ELEVATION_SORT.get(elevation, 99), RADIUS_SORT.get(radius, 99), azimuth)


def azimuth_gap(a: int, b: int) -> int:
    delta = abs(a - b) % 360
    return min(delta, 360 - delta)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def build_view_records(gt_json: Path, pred_json: Path, score_threshold: float) -> list[ViewRecord]:
    gt = load_json(gt_json)
    preds = load_json(pred_json)

    class_id_to_name = {int(row["id"]): str(row["name"]) for row in gt["categories"]}
    class_name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}
    known_class_names = list(class_name_to_id.keys())

    image_meta = {int(row["id"]): row for row in gt["images"]}
    gt_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in gt["annotations"]:
        gt_by_image[int(row["image_id"])].append(row)

    pred_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in preds:
        if float(row["score"]) >= score_threshold:
            pred_by_image[int(row["image_id"])].append(row)

    records: list[ViewRecord] = []
    for image_id in sorted(image_meta):
        meta = image_meta[image_id]
        scene_key, viewpoint, elevation, radius, azimuth, target_class = parse_viewpoint_metadata(str(meta["file_name"]), known_class_names)
        target_class_id = class_name_to_id[target_class]

        gt_annotations = gt_by_image.get(image_id, [])
        pred_annotations = pred_by_image.get(image_id, [])

        gt_boxes = np.array([xywh_to_xyxy(row["bbox"]) for row in gt_annotations], dtype=np.float32)
        gt_classes = np.array([int(row["category_id"]) for row in gt_annotations], dtype=np.int32)
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int32)

        pred_boxes = np.array([xywh_to_xyxy(row["bbox"]) for row in pred_annotations], dtype=np.float32)
        pred_scores = np.array([float(row["score"]) for row in pred_annotations], dtype=np.float32)
        pred_classes = np.array([int(row["category_id"]) for row in pred_annotations], dtype=np.int32)
        if len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=np.int32)

        overall_metrics = evaluate_single_image(pred_boxes=pred_boxes, pred_scores=pred_scores, pred_classes=pred_classes, gt_boxes=gt_boxes, gt_classes=gt_classes)

        gt_target_mask = gt_classes == target_class_id
        pred_target_mask = pred_classes == target_class_id
        target_metrics = evaluate_single_image(
            pred_boxes=pred_boxes[pred_target_mask],
            pred_scores=pred_scores[pred_target_mask],
            pred_classes=pred_classes[pred_target_mask],
            gt_boxes=gt_boxes[gt_target_mask],
            gt_classes=gt_classes[gt_target_mask],
        )
        target_quality = compute_target_match_quality(
            pred_boxes=pred_boxes[pred_target_mask],
            pred_scores=pred_scores[pred_target_mask],
            gt_boxes=gt_boxes[gt_target_mask],
            iou_threshold=0.5,
        )

        records.append(
            ViewRecord(
                image_id=image_id,
                file_name=str(meta["file_name"]),
                scene_key=scene_key,
                viewpoint=viewpoint,
                elevation=elevation,
                radius=radius,
                azimuth=azimuth,
                target_class=target_class,
                target_class_id=target_class_id,
                precision=float(overall_metrics["precision"]),
                recall=float(overall_metrics["recall"]),
                f1=float(overall_metrics["f1"]),
                ap50=float(overall_metrics["ap50"]),
                ap50_95=float(overall_metrics["ap50_95"]),
                tp=int(overall_metrics["tp"]),
                fp=int(overall_metrics["fp"]),
                fn=int(overall_metrics["fn"]),
                num_gt=int(overall_metrics["num_gt"]),
                num_pred=int(overall_metrics["num_pred"]),
                target_visible=bool(int(target_metrics["num_gt"]) > 0),
                target_detected=bool(int(target_metrics["tp"]) > 0),
                target_precision=float(target_metrics["precision"]),
                target_recall=float(target_metrics["recall"]),
                target_f1=float(target_metrics["f1"]),
                target_ap50=0.0 if math.isnan(float(target_metrics["ap50"])) else float(target_metrics["ap50"]),
                target_ap50_95=0.0 if math.isnan(float(target_metrics["ap50_95"])) else float(target_metrics["ap50_95"]),
                target_best_iou=float(target_quality["target_best_iou"]),
                target_match_confidence_iou50=float(target_quality["target_match_confidence_iou50"]),
                target_match_iou_at_confidence_iou50=float(target_quality["target_match_iou_at_confidence_iou50"]),
                target_strict_quality_iou50=float(target_quality["target_strict_quality_iou50"]),
                target_tp=int(target_metrics["tp"]),
                target_fp=int(target_metrics["fp"]),
                target_fn=int(target_metrics["fn"]),
            )
        )

    return records


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scene_groups(records: list[ViewRecord]) -> dict[str, list[ViewRecord]]:
    grouped: dict[str, list[ViewRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_key].append(record)
    return {key: sorted(values, key=lambda row: viewpoint_sort_key(row.viewpoint)) for key, values in sorted(grouped.items())}


def scene_view_rows(records: list[ViewRecord]) -> list[dict[str, object]]:
    return [
        {
            "scene_key": row.scene_key,
            "file_name": row.file_name,
            "image_id": row.image_id,
            "target_class": row.target_class,
            "viewpoint": row.viewpoint,
            "elevation": row.elevation,
            "radius": row.radius,
            "azimuth": row.azimuth,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "ap50": row.ap50,
            "ap50_95": row.ap50_95,
            "tp": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "num_gt": row.num_gt,
            "num_pred": row.num_pred,
            "target_visible": int(row.target_visible),
            "target_detected": int(row.target_detected),
            "target_precision": row.target_precision,
            "target_recall": row.target_recall,
            "target_f1": row.target_f1,
            "target_ap50": row.target_ap50,
            "target_ap50_95": row.target_ap50_95,
            "target_best_iou": row.target_best_iou,
            "target_match_confidence_iou50": row.target_match_confidence_iou50,
            "target_match_iou_at_confidence_iou50": row.target_match_iou_at_confidence_iou50,
            "target_strict_quality_iou50": row.target_strict_quality_iou50,
            "target_tp": row.target_tp,
            "target_fp": row.target_fp,
            "target_fn": row.target_fn,
        }
        for row in records
    ]


def summarize_single_viewpoints(records: list[ViewRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[ViewRecord]] = defaultdict(list)
    for row in records:
        grouped[row.viewpoint].append(row)

    rows: list[dict[str, object]] = []
    for viewpoint, members in sorted(grouped.items(), key=lambda item: viewpoint_sort_key(item[0])):
        visible_count = sum(1 for row in members if row.target_visible)
        detected_count = sum(1 for row in members if row.target_detected)
        tp = sum(row.tp for row in members)
        fp = sum(row.fp for row in members)
        fn = sum(row.fn for row in members)
        rows.append(
            {
                "viewpoint": viewpoint,
                "elevation": members[0].elevation,
                "radius": members[0].radius,
                "azimuth": members[0].azimuth,
                "sample_count": len(members),
                "scene_count": len({row.scene_key for row in members}),
                "micro_precision": safe_divide(tp, tp + fp),
                "micro_recall": safe_divide(tp, tp + fn),
                "micro_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "mean_precision": mean(row.precision for row in members),
                "mean_recall": mean(row.recall for row in members),
                "mean_f1": mean(row.f1 for row in members),
                "mean_ap50": mean(row.ap50 for row in members),
                "mean_ap50_95": mean(row.ap50_95 for row in members),
                "mean_target_precision": mean(row.target_precision for row in members),
                "mean_target_recall": mean(row.target_recall for row in members),
                "mean_target_f1": mean(row.target_f1 for row in members),
                "mean_target_ap50": mean(row.target_ap50 for row in members),
                "mean_target_ap50_95": mean(row.target_ap50_95 for row in members),
                "mean_target_best_iou": mean(row.target_best_iou for row in members),
                "mean_target_match_confidence_iou50": mean(row.target_match_confidence_iou50 for row in members),
                "mean_target_strict_quality_iou50": mean(row.target_strict_quality_iou50 for row in members),
                "target_visible_rate": safe_divide(visible_count, len(members)),
                "target_detected_rate": safe_divide(detected_count, len(members)),
                "target_detected_given_visible": safe_divide(detected_count, visible_count),
            }
        )
    return rows


def summarize_single_factors(records: list[ViewRecord]) -> list[dict[str, object]]:
    factor_groups: dict[tuple[str, str], list[ViewRecord]] = defaultdict(list)
    for row in records:
        factor_groups[("elevation", row.elevation)].append(row)
        factor_groups[("radius", row.radius)].append(row)
        factor_groups[("azimuth", f"{row.azimuth:03d}")].append(row)

    rows: list[dict[str, object]] = []
    for (factor, level), members in sorted(factor_groups.items()):
        visible_count = sum(1 for row in members if row.target_visible)
        detected_count = sum(1 for row in members if row.target_detected)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "sample_count": len(members),
                "scene_count": len({row.scene_key for row in members}),
                "mean_f1": mean(row.f1 for row in members),
                "mean_ap50_95": mean(row.ap50_95 for row in members),
                "mean_target_ap50_95": mean(row.target_ap50_95 for row in members),
                "mean_target_match_confidence_iou50": mean(row.target_match_confidence_iou50 for row in members),
                "mean_target_strict_quality_iou50": mean(row.target_strict_quality_iou50 for row in members),
                "target_visible_rate": safe_divide(visible_count, len(members)),
                "target_detected_rate": safe_divide(detected_count, len(members)),
                "target_detected_given_visible": safe_divide(detected_count, visible_count),
            }
        )
    return rows


def pair_metric_row(first: ViewRecord, second: ViewRecord) -> dict[str, object]:
    pair_tp = first.tp + second.tp
    pair_fp = first.fp + second.fp
    pair_fn = first.fn + second.fn
    pair_visible = first.target_visible or second.target_visible
    pair_detected = first.target_detected or second.target_detected
    return {
        "scene_key": first.scene_key,
        "target_class": first.target_class,
        "viewpoint_1": first.viewpoint,
        "viewpoint_2": second.viewpoint,
        "viewpoint_pair": f"{first.viewpoint} + {second.viewpoint}",
        "elevation_1": first.elevation,
        "radius_1": first.radius,
        "azimuth_1": first.azimuth,
        "elevation_2": second.elevation,
        "radius_2": second.radius,
        "azimuth_2": second.azimuth,
        "same_elevation": int(first.elevation == second.elevation),
        "same_radius": int(first.radius == second.radius),
        "azimuth_gap": azimuth_gap(first.azimuth, second.azimuth),
        "precision": safe_divide(pair_tp, pair_tp + pair_fp),
        "recall": safe_divide(pair_tp, pair_tp + pair_fn),
        "f1": safe_divide(2 * pair_tp, 2 * pair_tp + pair_fp + pair_fn),
        "mean_ap50_95": mean([first.ap50_95, second.ap50_95]),
        "best_ap50_95": max(first.ap50_95, second.ap50_95),
        "mean_target_ap50_95": mean([first.target_ap50_95, second.target_ap50_95]),
        "best_target_ap50_95": max(first.target_ap50_95, second.target_ap50_95),
        "mean_target_match_confidence_iou50": mean([first.target_match_confidence_iou50, second.target_match_confidence_iou50]),
        "best_target_match_confidence_iou50": max(first.target_match_confidence_iou50, second.target_match_confidence_iou50),
        "mean_target_strict_quality_iou50": mean([first.target_strict_quality_iou50, second.target_strict_quality_iou50]),
        "best_target_strict_quality_iou50": max(first.target_strict_quality_iou50, second.target_strict_quality_iou50),
        "target_visible": int(pair_visible),
        "target_detected": int(pair_detected),
        "target_detected_given_visible": 1.0 if pair_visible and pair_detected else 0.0,
        "tp_sum": pair_tp,
        "fp_sum": pair_fp,
        "fn_sum": pair_fn,
    }


def ordered_rescue_row(primary: ViewRecord, secondary: ViewRecord) -> dict[str, object]:
    primary_miss = int(not primary.target_detected)
    rescue = int((not primary.target_detected) and secondary.target_detected)
    primary_not_visible = int(not primary.target_visible)
    visibility_rescue = int((not primary.target_visible) and secondary.target_visible)
    return {
        "scene_key": primary.scene_key,
        "target_class": primary.target_class,
        "primary_viewpoint": primary.viewpoint,
        "secondary_viewpoint": secondary.viewpoint,
        "primary_elevation": primary.elevation,
        "primary_radius": primary.radius,
        "primary_azimuth": primary.azimuth,
        "secondary_elevation": secondary.elevation,
        "secondary_radius": secondary.radius,
        "secondary_azimuth": secondary.azimuth,
        "same_elevation": int(primary.elevation == secondary.elevation),
        "same_radius": int(primary.radius == secondary.radius),
        "azimuth_gap": azimuth_gap(primary.azimuth, secondary.azimuth),
        "primary_target_visible": int(primary.target_visible),
        "primary_target_detected": int(primary.target_detected),
        "secondary_target_visible": int(secondary.target_visible),
        "secondary_target_detected": int(secondary.target_detected),
        "primary_miss": primary_miss,
        "rescue": rescue,
        "primary_not_visible": primary_not_visible,
        "visibility_rescue": visibility_rescue,
    }


def build_pair_rows(scene_groups: dict[str, list[ViewRecord]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair_rows: list[dict[str, object]] = []
    rescue_rows: list[dict[str, object]] = []

    for members in scene_groups.values():
        for first, second in combinations(members, 2):
            pair_rows.append(pair_metric_row(first, second))
        for primary, secondary in permutations(members, 2):
            rescue_rows.append(ordered_rescue_row(primary, secondary))

    return pair_rows, rescue_rows


def summarize_pair_viewpoints(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["viewpoint_pair"])].append(row)

    rows: list[dict[str, object]] = []
    for pair_key, members in sorted(grouped.items()):
        visible_count = sum(int(row["target_visible"]) for row in members)
        detected_count = sum(int(row["target_detected"]) for row in members)
        rows.append(
            {
                "viewpoint_pair": pair_key,
                "viewpoint_1": members[0]["viewpoint_1"],
                "viewpoint_2": members[0]["viewpoint_2"],
                "sample_count": len(members),
                "scene_count": len({str(row["scene_key"]) for row in members}),
                "mean_precision": mean(float(row["precision"]) for row in members),
                "mean_recall": mean(float(row["recall"]) for row in members),
                "mean_f1": mean(float(row["f1"]) for row in members),
                "mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
                "mean_best_ap50_95": mean(float(row["best_ap50_95"]) for row in members),
                "mean_target_ap50_95": mean(float(row["mean_target_ap50_95"]) for row in members),
                "mean_best_target_ap50_95": mean(float(row["best_target_ap50_95"]) for row in members),
                "mean_target_match_confidence_iou50": mean(float(row["mean_target_match_confidence_iou50"]) for row in members),
                "mean_best_target_match_confidence_iou50": mean(float(row["best_target_match_confidence_iou50"]) for row in members),
                "mean_target_strict_quality_iou50": mean(float(row["mean_target_strict_quality_iou50"]) for row in members),
                "mean_best_target_strict_quality_iou50": mean(float(row["best_target_strict_quality_iou50"]) for row in members),
                "mean_azimuth_gap": mean(float(row["azimuth_gap"]) for row in members),
                "target_visible_rate": safe_divide(visible_count, len(members)),
                "target_detected_rate": safe_divide(detected_count, len(members)),
                "target_detected_given_visible": safe_divide(detected_count, visible_count),
            }
        )
    return rows


def summarize_pair_relations(pair_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    relation_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    azimuth_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        relation_groups[f"same_elevation_{int(row['same_elevation'])}__same_radius_{int(row['same_radius'])}"].append(row)
        azimuth_groups[int(row["azimuth_gap"])].append(row)

    relation_rows: list[dict[str, object]] = []
    for label, members in sorted(relation_groups.items()):
        visible_count = sum(int(row["target_visible"]) for row in members)
        detected_count = sum(int(row["target_detected"]) for row in members)
        relation_rows.append(
            {
                "relation_label": label,
                "sample_count": len(members),
                "mean_f1": mean(float(row["f1"]) for row in members),
                "mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
                "mean_target_ap50_95": mean(float(row["mean_target_ap50_95"]) for row in members),
                "mean_target_match_confidence_iou50": mean(float(row["mean_target_match_confidence_iou50"]) for row in members),
                "mean_target_strict_quality_iou50": mean(float(row["mean_target_strict_quality_iou50"]) for row in members),
                "target_visible_rate": safe_divide(visible_count, len(members)),
                "target_detected_rate": safe_divide(detected_count, len(members)),
                "target_detected_given_visible": safe_divide(detected_count, visible_count),
            }
        )

    azimuth_rows: list[dict[str, object]] = []
    for gap, members in sorted(azimuth_groups.items()):
        visible_count = sum(int(row["target_visible"]) for row in members)
        detected_count = sum(int(row["target_detected"]) for row in members)
        azimuth_rows.append(
            {
                "azimuth_gap": gap,
                "sample_count": len(members),
                "mean_f1": mean(float(row["f1"]) for row in members),
                "mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
                "mean_target_ap50_95": mean(float(row["mean_target_ap50_95"]) for row in members),
                "mean_target_match_confidence_iou50": mean(float(row["mean_target_match_confidence_iou50"]) for row in members),
                "mean_target_strict_quality_iou50": mean(float(row["mean_target_strict_quality_iou50"]) for row in members),
                "target_visible_rate": safe_divide(visible_count, len(members)),
                "target_detected_rate": safe_divide(detected_count, len(members)),
                "target_detected_given_visible": safe_divide(detected_count, visible_count),
            }
        )

    return relation_rows, azimuth_rows


def summarize_second_drone_rescue(rescue_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    viewpoint_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    factor_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    azimuth_groups: dict[int, list[dict[str, object]]] = defaultdict(list)

    for row in rescue_rows:
        viewpoint_groups[str(row["secondary_viewpoint"])].append(row)
        factor_groups[("elevation", str(row["secondary_elevation"]))].append(row)
        factor_groups[("radius", str(row["secondary_radius"]))].append(row)
        factor_groups[("azimuth", f"{int(row['secondary_azimuth']):03d}")].append(row)
        azimuth_groups[int(row["azimuth_gap"])].append(row)

    viewpoint_summary: list[dict[str, object]] = []
    for viewpoint, members in sorted(viewpoint_groups.items(), key=lambda item: viewpoint_sort_key(item[0])):
        miss_count = sum(int(row["primary_miss"]) for row in members)
        rescue_count = sum(int(row["rescue"]) for row in members)
        not_visible_count = sum(int(row["primary_not_visible"]) for row in members)
        visibility_rescue_count = sum(int(row["visibility_rescue"]) for row in members)
        viewpoint_summary.append(
            {
                "secondary_viewpoint": viewpoint,
                "sample_count": len(members),
                "scene_count": len({str(row["scene_key"]) for row in members}),
                "primary_miss_count": miss_count,
                "rescue_count": rescue_count,
                "rescue_rate_over_all_pairs": safe_divide(rescue_count, len(members)),
                "rescue_rate_given_primary_miss": safe_divide(rescue_count, miss_count),
                "primary_not_visible_count": not_visible_count,
                "visibility_rescue_count": visibility_rescue_count,
                "visibility_rescue_given_primary_not_visible": safe_divide(visibility_rescue_count, not_visible_count),
                "mean_azimuth_gap": mean(float(row["azimuth_gap"]) for row in members),
            }
        )

    factor_summary: list[dict[str, object]] = []
    for (factor, level), members in sorted(factor_groups.items()):
        miss_count = sum(int(row["primary_miss"]) for row in members)
        rescue_count = sum(int(row["rescue"]) for row in members)
        not_visible_count = sum(int(row["primary_not_visible"]) for row in members)
        visibility_rescue_count = sum(int(row["visibility_rescue"]) for row in members)
        factor_summary.append(
            {
                "factor": factor,
                "level": level,
                "sample_count": len(members),
                "primary_miss_count": miss_count,
                "rescue_count": rescue_count,
                "rescue_rate_over_all_pairs": safe_divide(rescue_count, len(members)),
                "rescue_rate_given_primary_miss": safe_divide(rescue_count, miss_count),
                "primary_not_visible_count": not_visible_count,
                "visibility_rescue_count": visibility_rescue_count,
                "visibility_rescue_given_primary_not_visible": safe_divide(visibility_rescue_count, not_visible_count),
            }
        )

    azimuth_summary: list[dict[str, object]] = []
    for gap, members in sorted(azimuth_groups.items()):
        miss_count = sum(int(row["primary_miss"]) for row in members)
        rescue_count = sum(int(row["rescue"]) for row in members)
        not_visible_count = sum(int(row["primary_not_visible"]) for row in members)
        visibility_rescue_count = sum(int(row["visibility_rescue"]) for row in members)
        azimuth_summary.append(
            {
                "azimuth_gap": gap,
                "sample_count": len(members),
                "primary_miss_count": miss_count,
                "rescue_count": rescue_count,
                "rescue_rate_over_all_pairs": safe_divide(rescue_count, len(members)),
                "rescue_rate_given_primary_miss": safe_divide(rescue_count, miss_count),
                "primary_not_visible_count": not_visible_count,
                "visibility_rescue_count": visibility_rescue_count,
                "visibility_rescue_given_primary_not_visible": safe_divide(visibility_rescue_count, not_visible_count),
            }
        )

    return viewpoint_summary, factor_summary, azimuth_summary


def build_scene_expectations(scene_groups: dict[str, list[ViewRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene_key, members in sorted(scene_groups.items()):
        single_visible_rate = mean(float(row.target_visible) for row in members)
        single_detected_rate = mean(float(row.target_detected) for row in members)
        single_target_ap = mean(float(row.target_ap50_95) for row in members)
        single_target_confidence = mean(float(row.target_match_confidence_iou50) for row in members)
        single_target_quality = mean(float(row.target_strict_quality_iou50) for row in members)
        pair_members = [pair_metric_row(first, second) for first, second in combinations(members, 2)]
        pair_visible_rate = mean(float(row["target_visible"]) for row in pair_members) if pair_members else float("nan")
        pair_detected_rate = mean(float(row["target_detected"]) for row in pair_members) if pair_members else float("nan")
        pair_f1 = mean(float(row["f1"]) for row in pair_members) if pair_members else float("nan")
        pair_ap = mean(float(row["mean_ap50_95"]) for row in pair_members) if pair_members else float("nan")
        pair_best_ap = mean(float(row["best_ap50_95"]) for row in pair_members) if pair_members else float("nan")
        pair_target_mean_ap = mean(float(row["mean_target_ap50_95"]) for row in pair_members) if pair_members else float("nan")
        pair_target_best_ap = mean(float(row["best_target_ap50_95"]) for row in pair_members) if pair_members else float("nan")
        pair_target_confidence = mean(float(row["best_target_match_confidence_iou50"]) for row in pair_members) if pair_members else float("nan")
        pair_target_quality = mean(float(row["best_target_strict_quality_iou50"]) for row in pair_members) if pair_members else float("nan")
        single_ap = mean(row.ap50_95 for row in members)
        rows.append(
            {
                "scene_key": scene_key,
                "target_class": members[0].target_class,
                "available_view_count": len(members),
                "pair_count": len(pair_members),
                "single_expected_precision": mean(row.precision for row in members),
                "single_expected_recall": mean(row.recall for row in members),
                "single_expected_f1": mean(row.f1 for row in members),
                "single_expected_ap50_95": single_ap,
                "single_expected_target_visible_rate": single_visible_rate,
                "single_expected_target_detected_rate": single_detected_rate,
                "single_expected_target_detected_given_visible": safe_divide(single_detected_rate, single_visible_rate),
                "single_expected_target_ap50_95": single_target_ap,
                "single_expected_target_match_confidence_iou50": single_target_confidence,
                "single_expected_target_strict_quality_iou50": single_target_quality,
                "pair_expected_precision": mean(float(row["precision"]) for row in pair_members) if pair_members else float("nan"),
                "pair_expected_recall": mean(float(row["recall"]) for row in pair_members) if pair_members else float("nan"),
                "pair_expected_f1": pair_f1,
                "pair_expected_ap50_95": pair_ap,
                "pair_expected_best_available_ap50_95": pair_best_ap,
                "pair_expected_target_visible_rate": pair_visible_rate,
                "pair_expected_target_detected_rate": pair_detected_rate,
                "pair_expected_target_detected_given_visible": safe_divide(pair_detected_rate, pair_visible_rate),
                "pair_expected_mean_target_ap50_95": pair_target_mean_ap,
                "pair_expected_target_ap50_95": pair_target_best_ap,
                "pair_expected_best_target_ap50_95": pair_target_best_ap,
                "pair_expected_target_match_confidence_iou50": pair_target_confidence,
                "pair_expected_target_strict_quality_iou50": pair_target_quality,
                "delta_target_detected_rate": pair_detected_rate - single_detected_rate if pair_members else float("nan"),
                "delta_target_ap50_95": pair_target_best_ap - single_target_ap if pair_members else float("nan"),
                "delta_target_match_confidence_iou50": pair_target_confidence - single_target_confidence if pair_members else float("nan"),
                "delta_target_strict_quality_iou50": pair_target_quality - single_target_quality if pair_members else float("nan"),
                "delta_ap50_95": pair_ap - single_ap if pair_members else float("nan"),
            }
        )
    return rows


def overall_comparison(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    single_visible = mean(float(row["single_expected_target_visible_rate"]) for row in scene_rows)
    single_detected = mean(float(row["single_expected_target_detected_rate"]) for row in scene_rows)
    pair_visible = mean(float(row["pair_expected_target_visible_rate"]) for row in scene_rows)
    pair_detected = mean(float(row["pair_expected_target_detected_rate"]) for row in scene_rows)

    return [
        {
            "drone_count": 1,
            "scene_count": len(scene_rows),
            "expected_precision": mean(float(row["single_expected_precision"]) for row in scene_rows),
            "expected_recall": mean(float(row["single_expected_recall"]) for row in scene_rows),
            "expected_f1": mean(float(row["single_expected_f1"]) for row in scene_rows),
            "expected_ap50_95": mean(float(row["single_expected_ap50_95"]) for row in scene_rows),
            "expected_best_available_ap50_95": mean(float(row["single_expected_ap50_95"]) for row in scene_rows),
            "expected_target_visible_rate": single_visible,
            "expected_target_detected_rate": single_detected,
            "expected_target_detected_given_visible": safe_divide(single_detected, single_visible),
            "expected_target_ap50_95": mean(float(row["single_expected_target_ap50_95"]) for row in scene_rows),
            "expected_best_target_ap50_95": mean(float(row["single_expected_target_ap50_95"]) for row in scene_rows),
            "expected_target_match_confidence_iou50": mean(float(row["single_expected_target_match_confidence_iou50"]) for row in scene_rows),
            "expected_target_strict_quality_iou50": mean(float(row["single_expected_target_strict_quality_iou50"]) for row in scene_rows),
        },
        {
            "drone_count": 2,
            "scene_count": len(scene_rows),
            "expected_precision": mean(float(row["pair_expected_precision"]) for row in scene_rows),
            "expected_recall": mean(float(row["pair_expected_recall"]) for row in scene_rows),
            "expected_f1": mean(float(row["pair_expected_f1"]) for row in scene_rows),
            "expected_ap50_95": mean(float(row["pair_expected_ap50_95"]) for row in scene_rows),
            "expected_best_available_ap50_95": mean(float(row["pair_expected_best_available_ap50_95"]) for row in scene_rows),
            "expected_target_visible_rate": pair_visible,
            "expected_target_detected_rate": pair_detected,
            "expected_target_detected_given_visible": safe_divide(pair_detected, pair_visible),
            "expected_target_ap50_95": mean(float(row["pair_expected_target_ap50_95"]) for row in scene_rows),
            "expected_best_target_ap50_95": mean(float(row["pair_expected_best_target_ap50_95"]) for row in scene_rows),
            "expected_target_match_confidence_iou50": mean(float(row["pair_expected_target_match_confidence_iou50"]) for row in scene_rows),
            "expected_target_strict_quality_iou50": mean(float(row["pair_expected_target_strict_quality_iou50"]) for row in scene_rows),
        },
    ]


def class_comparison(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scene_rows:
        grouped[str(row["target_class"])].append(row)

    rows: list[dict[str, object]] = []
    for class_name, members in sorted(grouped.items()):
        single_detected = mean(float(row["single_expected_target_detected_rate"]) for row in members)
        pair_detected = mean(float(row["pair_expected_target_detected_rate"]) for row in members)
        single_ap = mean(float(row["single_expected_ap50_95"]) for row in members)
        pair_ap = mean(float(row["pair_expected_ap50_95"]) for row in members)
        pair_best_ap = mean(float(row["pair_expected_best_available_ap50_95"]) for row in members)
        rows.append(
            {
                "target_class": class_name,
                "scene_count": len(members),
                "single_expected_target_detected_rate": single_detected,
                "pair_expected_target_detected_rate": pair_detected,
                "delta_target_detected_rate": pair_detected - single_detected,
                "single_expected_target_ap50_95": mean(float(row["single_expected_target_ap50_95"]) for row in members),
                "pair_expected_target_ap50_95": mean(float(row["pair_expected_target_ap50_95"]) for row in members),
                "delta_target_ap50_95": mean(float(row["pair_expected_target_ap50_95"]) for row in members) - mean(float(row["single_expected_target_ap50_95"]) for row in members),
                "single_expected_target_strict_quality_iou50": mean(float(row["single_expected_target_strict_quality_iou50"]) for row in members),
                "pair_expected_target_strict_quality_iou50": mean(float(row["pair_expected_target_strict_quality_iou50"]) for row in members),
                "delta_target_strict_quality_iou50": mean(float(row["pair_expected_target_strict_quality_iou50"]) for row in members) - mean(float(row["single_expected_target_strict_quality_iou50"]) for row in members),
                "single_expected_ap50_95": single_ap,
                "pair_expected_ap50_95": pair_ap,
                "delta_ap50_95": pair_ap - single_ap,
                "pair_expected_best_available_ap50_95": pair_best_ap,
                "delta_best_available_ap50_95": pair_best_ap - single_ap,
            }
        )
    return rows


def plot_overall_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = [
        ("expected_target_match_confidence_iou50", "Target confidence"),
        ("expected_target_strict_quality_iou50", "Target strict quality"),
        ("expected_target_ap50_95", "Target AP50-95"),
        ("expected_ap50_95", "Mean AP50-95"),
        ("expected_best_available_ap50_95", "Best available AP50-95"),
        ("expected_f1", "Mean F1"),
    ]
    x = np.arange(len(metrics))
    values = [[float(row[key]) for key, _ in metrics] for row in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.36
    ax.bar(x - width / 2, values[0], width, label="1 drone", color="#1f77b4")
    ax.bar(x + width / 2, values[1], width, label="2 drones", color="#ff7f0e")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=15, ha="right")
    ax.set_title("Expected performance under stricter target-quality scoring")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_barh(rows: list[dict[str, object]], label_key: str, value_key: str, output_path: Path, title: str, xlabel: str, top_k: int = 20) -> None:
    top_rows = sorted(rows, key=lambda row: float(row[value_key]), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_rows) * 0.38)))
    labels = [str(row[label_key]) for row in top_rows][::-1]
    values = [float(row[value_key]) for row in top_rows][::-1]
    ax.barh(labels, values, color="#2ca02c")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_secondary_factor_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    elevation_rows = [row for row in rows if row["factor"] == "elevation"]
    radius_rows = [row for row in rows if row["factor"] == "radius"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, members, title in (
        (axes[0], elevation_rows, "Rescue rate by second-drone elevation"),
        (axes[1], radius_rows, "Rescue rate by second-drone radius"),
    ):
        labels = [str(row["level"]) for row in members]
        values = [float(row["rescue_rate_given_primary_miss"]) for row in members]
        ax.bar(labels, values, color="#9467bd")
        ax.set_ylim(0, max(values + [0.05]) * 1.2 if values else 0.1)
        ax.set_ylabel("Rescue rate | primary miss")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_azimuth_gap_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [str(int(row["azimuth_gap"])) for row in rows]
    values = [float(row["rescue_rate_given_primary_miss"]) for row in rows]
    ax.bar(labels, values, color="#8c564b")
    ax.set_ylabel("Rescue rate | primary miss")
    ax.set_xlabel("Azimuth gap (degrees)")
    ax.set_title("Second-drone rescue by azimuth gap")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_delta(rows: list[dict[str, object]], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["delta_target_strict_quality_iou50"]), reverse=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [str(row["target_class"]) for row in sorted_rows]
    values = [float(row["delta_target_strict_quality_iou50"]) for row in sorted_rows]
    ax.bar(labels, values, color="#d62728")
    ax.set_ylabel("2-drone minus 1-drone target strict quality")
    ax.set_title("Per-class strict target-quality gain from a second viewpoint")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(output_path: Path, overall_rows: list[dict[str, object]], class_rows: list[dict[str, object]], single_rows: list[dict[str, object]], pair_rows: list[dict[str, object]], rescue_rows: list[dict[str, object]], relation_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    overall_by_drone = {int(row["drone_count"]): row for row in overall_rows}
    best_single = sorted(single_rows, key=lambda row: (float(row["mean_target_strict_quality_iou50"]), float(row["mean_target_ap50_95"])), reverse=True)[:5]
    best_pairs = sorted(pair_rows, key=lambda row: (float(row["mean_best_target_strict_quality_iou50"]), float(row["mean_best_target_ap50_95"])), reverse=True)[:5]
    best_rescues = sorted(
        [row for row in rescue_rows if int(row["primary_miss_count"]) >= args.min_rescue_support],
        key=lambda row: (float(row["rescue_rate_given_primary_miss"]), int(row["primary_miss_count"])),
        reverse=True,
    )[:5]
    best_classes = sorted(class_rows, key=lambda row: float(row["delta_target_strict_quality_iou50"]), reverse=True)

    lines = [
        "# M4 Two-Drone Operational Analysis",
        "",
        "## What This Method Measures",
        "",
        "- The detector is held fixed: full M4 `YOLOv8l` predictions on the fixed M4 test split are reused.",
        "- One-drone mode selects one available viewpoint per scene.",
        "- Two-drone mode selects two available viewpoints from the same scene.",
        "- Binary target-found rate is still reported as a reference.",
        "- The stricter headline metrics score the target with matched confidence, target-only AP50-95, and a strict quality score equal to confidence multiplied by matched IoU at IoU>=0.50.",
        "- Standard detector image scores are also summarized over the selected image sets.",
        "",
        "## Important Interpretation Boundary",
        "",
        "- This is an operational viewpoint-availability simulation, not a retraining experiment.",
        "- The target-centric mission metric is the cleanest answer to the question 'does a second view help find the intended object?'",
        "- The accompanying precision/recall/F1/AP values describe the selected image sets, but they do not deduplicate identical real-world objects across views.",
        "",
        "## Overall Expected Comparison",
        "",
        f"- Expected target confidence, 1 drone: {float(overall_by_drone[1]['expected_target_match_confidence_iou50']):.4f}",
        f"- Expected target confidence, 2 drones: {float(overall_by_drone[2]['expected_target_match_confidence_iou50']):.4f}",
        f"- Expected target strict quality, 1 drone: {float(overall_by_drone[1]['expected_target_strict_quality_iou50']):.4f}",
        f"- Expected target strict quality, 2 drones: {float(overall_by_drone[2]['expected_target_strict_quality_iou50']):.4f}",
        f"- Gain in target strict quality: {float(overall_by_drone[2]['expected_target_strict_quality_iou50']) - float(overall_by_drone[1]['expected_target_strict_quality_iou50']):.4f}",
        f"- Expected target AP50-95, 1 drone: {float(overall_by_drone[1]['expected_target_ap50_95']):.4f}",
        f"- Expected best target AP50-95, 2 drones: {float(overall_by_drone[2]['expected_target_ap50_95']):.4f}",
        f"- Binary target found rate, 1 drone: {float(overall_by_drone[1]['expected_target_detected_rate']):.4f}",
        f"- Binary target found rate, 2 drones: {float(overall_by_drone[2]['expected_target_detected_rate']):.4f}",
        f"- Expected mean AP50-95, 1 drone: {float(overall_by_drone[1]['expected_ap50_95']):.4f}",
        f"- Expected mean AP50-95, 2 drones: {float(overall_by_drone[2]['expected_ap50_95']):.4f}",
        f"- Expected best available AP50-95, 1 drone: {float(overall_by_drone[1]['expected_best_available_ap50_95']):.4f}",
        f"- Expected best available AP50-95, 2 drones: {float(overall_by_drone[2]['expected_best_available_ap50_95']):.4f}",
        f"- Expected mean F1, 1 drone: {float(overall_by_drone[1]['expected_f1']):.4f}",
        f"- Expected mean F1, 2 drones: {float(overall_by_drone[2]['expected_f1']):.4f}",
        "",
        "## Strongest Single Viewpoints",
        "",
    ]

    for row in best_single:
        lines.append(f"- `{row['viewpoint']}`: target strict quality {float(row['mean_target_strict_quality_iou50']):.4f}, target AP50-95 {float(row['mean_target_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Strongest Two-View Combinations", ""])
    for row in best_pairs:
        lines.append(f"- `{row['viewpoint_pair']}`: best target strict quality {float(row['mean_best_target_strict_quality_iou50']):.4f}, best target AP50-95 {float(row['mean_best_target_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Best Second-Drone Rescue Viewpoints", ""])
    for row in best_rescues:
        lines.append(f"- `{row['secondary_viewpoint']}`: rescue rate given primary miss {float(row['rescue_rate_given_primary_miss']):.4f}, primary-miss support {int(row['primary_miss_count'])}")

    lines.extend(["", "## Object Classes With The Largest 2-Drone Gains", ""])
    for row in best_classes[:5]:
        lines.append(f"- `{row['target_class']}`: delta target strict quality {float(row['delta_target_strict_quality_iou50']):.4f}, delta target AP50-95 {float(row['delta_target_ap50_95']):.4f}")

    lines.extend(["", "## Pair-Relation Snapshot", ""])
    for row in relation_rows:
        lines.append(f"- `{row['relation_label']}`: target strict quality {float(row['mean_target_strict_quality_iou50']):.4f}, target AP50-95 {float(row['mean_target_ap50_95']):.4f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_json = Path(args.gt_json).resolve()
    pred_json = Path(args.pred_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_view_records(gt_json, pred_json, score_threshold=args.score_threshold)
    scene_groups = build_scene_groups(records)
    single_rows = summarize_single_viewpoints(records)
    single_factor_rows = summarize_single_factors(records)
    pair_rows, rescue_rows = build_pair_rows(scene_groups)
    pair_summary_rows = summarize_pair_viewpoints(pair_rows)
    relation_rows, pair_gap_rows = summarize_pair_relations(pair_rows)
    rescue_view_rows, rescue_factor_rows, rescue_gap_rows = summarize_second_drone_rescue(rescue_rows)
    scene_expectation_rows = build_scene_expectations(scene_groups)
    overall_rows = overall_comparison(scene_expectation_rows)
    class_rows = class_comparison(scene_expectation_rows)

    write_csv_rows(output_dir / "scene_view_records.csv", scene_view_rows(records))
    write_csv_rows(output_dir / "single_viewpoint_summary.csv", single_rows)
    write_csv_rows(output_dir / "single_factor_summary.csv", single_factor_rows)
    write_csv_rows(output_dir / "pair_records.csv", pair_rows)
    write_csv_rows(output_dir / "pair_viewpoint_summary.csv", pair_summary_rows)
    write_csv_rows(output_dir / "pair_relation_summary.csv", relation_rows)
    write_csv_rows(output_dir / "pair_azimuth_gap_summary.csv", pair_gap_rows)
    write_csv_rows(output_dir / "ordered_second_drone_rescue_records.csv", rescue_rows)
    write_csv_rows(output_dir / "second_drone_viewpoint_rescue_summary.csv", rescue_view_rows)
    write_csv_rows(output_dir / "second_drone_factor_rescue_summary.csv", rescue_factor_rows)
    write_csv_rows(output_dir / "second_drone_azimuth_gap_rescue_summary.csv", rescue_gap_rows)
    write_csv_rows(output_dir / "scene_expectation_summary.csv", scene_expectation_rows)
    write_csv_rows(output_dir / "overall_one_vs_two_summary.csv", overall_rows)
    write_csv_rows(output_dir / "class_level_one_vs_two_summary.csv", class_rows)

    plot_overall_comparison(overall_rows, output_dir / "overall_one_vs_two_comparison.png")
    filtered_single_rows = [row for row in single_rows if int(row["sample_count"]) >= args.min_single_support]
    plot_top_barh(filtered_single_rows, "viewpoint", "mean_target_strict_quality_iou50", output_dir / "top_single_viewpoints_target_found.png", "Top single viewpoints by target strict quality", "Target strict quality")
    filtered_pair_rows = [row for row in pair_summary_rows if int(row["sample_count"]) >= args.min_pair_support]
    plot_top_barh(filtered_pair_rows, "viewpoint_pair", "mean_best_target_strict_quality_iou50", output_dir / "top_pair_viewpoints_target_found.png", "Top two-view combinations by best target strict quality", "Best target strict quality")
    filtered_rescue_rows = [row for row in rescue_view_rows if int(row["primary_miss_count"]) >= args.min_rescue_support]
    plot_top_barh(filtered_rescue_rows, "secondary_viewpoint", "rescue_rate_given_primary_miss", output_dir / "top_second_drone_rescue_viewpoints.png", "Best second-drone rescue viewpoints", "Rescue rate | primary miss")
    plot_secondary_factor_summary(rescue_factor_rows, output_dir / "second_drone_factor_rescue.png")
    plot_azimuth_gap_summary(rescue_gap_rows, output_dir / "second_drone_azimuth_gap_rescue.png")
    plot_class_delta(class_rows, output_dir / "class_level_two_minus_one_target_found.png")
    write_report(output_dir / "two_drone_operational_report.md", overall_rows, class_rows, filtered_single_rows, filtered_pair_rows, filtered_rescue_rows, relation_rows, args)

    print(f"Saved two-drone operational analysis to: {output_dir}")


if __name__ == "__main__":
    main()
