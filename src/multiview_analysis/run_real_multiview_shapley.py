from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
REAL_MULTIVIEW_DIR = WORKSPACE / "data_collection" / "raw_data" / "real_same_object_multiview"
UAV_DIR = REAL_MULTIVIEW_DIR / "images"
GT_LABEL_DIR = REAL_MULTIVIEW_DIR / "labels"
DEFAULT_OUTPUT_DIR = WORKSPACE / "results" / "recomputed" / "real_multiview_shapley"

DEFAULT_RUNS = {
    "synthetic_m4": WORKSPACE / "models" / "yolov8l" / "S0_M4_yolov8l" / "weights" / "best.pt",
    "real_uav_finetuned": WORKSPACE / "models" / "real_uav_finetuned" / "weights" / "last.pt",
}

# Timestamped same-object UAV labels use this 5-class order.
# It differs from the older 10-class synthetic/review-set order.
GT_CLASS_NAMES = [
    "container",
    "male",
    "suv",
    "tower",
    "whitevan",
]

# Both YOLO checkpoints used here expose this class order.
MODEL_CLASS_NAMES = [
    "tent",
    "tank",
    "tower",
    "container",
    "whitevan",
    "suv",
    "male",
    "rock",
    "barrel",
    "tree",
]

MAIN_METRICS = [
    "detection_at_conf25_iou50",
    "best_matched_confidence_iou50",
    "best_strict_quality_iou50",
    "noisy_or_best_iou_iou50",
]

SHAPLEY_VALUE_FUNCTIONS = {
    "detection": "detection_at_conf25_iou50",
    "strict_quality": "best_strict_quality_iou50",
    "noisy_or_best_iou": "noisy_or_best_iou_iou50",
}

SHAPLEY_EFFICIENCY_TOL = 1e-8


@dataclass(frozen=True)
class ObjectGroup:
    object_id: str
    description: str
    target_class: str
    file_names: tuple[str, ...]


@dataclass(frozen=True)
class GroundTruthTarget:
    class_name: str
    xyxy: tuple[float, float, float, float]
    area: float
    source_label_count: int
    selected_by: str


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewScore:
    run_name: str
    object_id: str
    description: str
    target_class: str
    view_index: int
    file_name: str
    image_path: Path
    label_path: Path | None
    target_gt_present: bool
    target_gt_count: int
    target_gt_area: float
    target_match_confidence_iou50: float
    target_match_iou_at_confidence_iou50: float
    target_strict_quality_iou50: float
    detected_at_conf25_iou50: int
    target_prediction_count: int
    best_target_prediction_confidence: float
    best_target_prediction_iou: float


OBJECT_GROUPS = [
    ObjectGroup(
        object_id="parkinglot_car",
        description="car on parking lot",
        target_class="suv",
        file_names=(
            "2026-06-28 20.56.45.jpg",
            "2026-06-28 20.56.46.jpg",
            "2026-06-28 20.56.47.jpg",
            "2026-06-28 20.56.54.jpg",
            "2026-06-28 20.58.04.jpg",
            "2026-06-28 20.58.01.jpg",
            "2026-06-28 20.58.04.jpg",
            "2026-06-28 20.58.01.jpg",
        ),
    ),
    ObjectGroup(
        object_id="ooij_tower",
        description="tower in the Ooij",
        target_class="tower",
        file_names=(
            "2026-06-29 17.37.15.jpg",
            "2026-06-29 17.35.32.jpg",
            "2026-06-29 17.36.06.jpg",
            "2026-06-29 17.36.07.jpg",
            "2026-06-29 17.36.40.jpg",
            "2026-06-29 17.36.50.jpg",
            "2026-06-29 17.36.51.jpg",
            "2026-06-29 17.36.53.jpg",
        ),
    ),
    ObjectGroup(
        object_id="la_souris_truck",
        description="La Souris truck",
        target_class="whitevan",
        file_names=(
            "2026-07-12 21.05.57.jpg",
            "2026-07-12 21.06.04.jpg",
            "2026-07-12 21.01.06.jpg",
            "2026-07-12 21.01.23.jpg",
            "2026-07-12 21.01.30.jpg",
            "2026-07-12 21.01.33.jpg",
            "2026-07-12 21.01.49.jpg",
            "2026-07-12 21.01.50.jpg",
            "2026-07-12 21.01.51.jpg",
            "2026-07-12 21.05.27.jpg",
        ),
    ),
    ObjectGroup(
        object_id="m_truck",
        description="M truck",
        target_class="whitevan",
        file_names=(
            "2026-07-12 21.22.14.jpg",
            "2026-07-12 21.21.43.jpg",
            "2026-07-12 21.21.46.jpg",
            "2026-07-12 21.21.47.jpg",
            "2026-07-12 21.22.11.jpg",
        ),
    ),
    ObjectGroup(
        object_id="white_truck_bottendaal",
        description="white truck Bottendaal",
        target_class="whitevan",
        file_names=(
            "2026-06-30 15.33.46.jpg",
            "2026-06-30 15.33.52.jpg",
            "2026-06-30 15.34.24.jpg",
            "2026-06-30 15.34.33.jpg",
            "2026-06-30 15.35.01.jpg",
            "2026-06-30 15.33.29.jpg",
            "2026-06-30 15.33.34.jpg",
            "2026-06-30 15.33.46.jpg",
            "2026-06-30 15.33.52.jpg",
            "2026-06-30 15.34.24.jpg",
            "2026-06-30 15.34.33.jpg",
            "2026-06-30 15.35.01.jpg",
            "2026-06-30 15.33.29.jpg",
            "2026-06-30 15.33.34.jpg",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-world same-object multi-view coalition and Shapley-style analysis "
            "on the five manually identified UAV object groups."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--uav-dir", type=Path, default=UAV_DIR)
    parser.add_argument("--gt-label-dir", type=Path, default=GT_LABEL_DIR)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=sorted(DEFAULT_RUNS),
        default=["synthetic_m4", "real_uav_finetuned"],
        help="Model runs to evaluate.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for target detection matches.",
    )
    parser.add_argument(
        "--detection-conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for the binary detected/not-detected metric.",
    )
    parser.add_argument(
        "--predict-conf-threshold",
        type=float,
        default=0.001,
        help="Low model prediction threshold used before target matching.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only rebuild analysis from existing prediction CSVs in the output directory.",
    )
    return parser.parse_args()


def dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def normalize_stem(stem: str) -> str:
    base = stem.split("_jpg.rf.", 1)[0]
    base = base.split("_jpeg.rf.", 1)[0]
    return base.replace(" ", "-").replace(".", "-").replace("_", "-")


def image_key(path_or_name: Path | str) -> str:
    return normalize_stem(Path(path_or_name).stem)


def label_key(path: Path) -> str:
    return normalize_stem(path.stem)


def yolo_or_polygon_to_xyxy(coords: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(coords) == 4:
        cx, cy, box_w, box_h = coords
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        return x1, y1, x2, y2

    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height


def box_area(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0.0 else inter / union


def safe_float(value: object) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def fmt(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_manifest(uav_dir: Path, gt_label_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Path]]:
    label_by_key = {label_key(path): path for path in gt_label_dir.glob("*.txt")}
    manifest_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    image_paths_by_key: dict[str, Path] = {}

    for group in OBJECT_GROUPS:
        deduped = dedupe_keep_order(group.file_names)
        duplicate_count = len(group.file_names) - len(deduped)
        for view_index, file_name in enumerate(deduped, start=1):
            image_path = uav_dir / file_name
            key = image_key(file_name)
            label_path = label_by_key.get(key)
            image_paths_by_key[key] = image_path
            exists = image_path.exists()
            label_exists = label_path is not None and label_path.exists()
            manifest_rows.append(
                {
                    "object_id": group.object_id,
                    "description": group.description,
                    "target_class": group.target_class,
                    "view_index": view_index,
                    "file_name": file_name,
                    "image_key": key,
                    "image_path": str(image_path),
                    "label_path": "" if label_path is None else str(label_path),
                    "image_exists": int(exists),
                    "label_exists": int(label_exists),
                    "duplicates_removed_from_group": duplicate_count,
                }
            )
            if not exists or not label_exists:
                issue_rows.append(
                    {
                        "object_id": group.object_id,
                        "file_name": file_name,
                        "image_exists": int(exists),
                        "label_exists": int(label_exists),
                        "message": "Missing raw UAV image or matching YOLO ground-truth label.",
                    }
                )
    return manifest_rows, issue_rows, image_paths_by_key


def read_target_gt(
    label_path: Path | None,
    image_path: Path,
    target_class: str,
) -> GroundTruthTarget | None:
    if label_path is None or not label_path.exists() or not image_path.exists():
        return None

    with Image.open(image_path) as image:
        width, height = image.size

    candidates: list[GroundTruthTarget] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(GT_CLASS_NAMES):
            continue
        class_name = GT_CLASS_NAMES[class_id]
        if class_name != target_class:
            continue
        coords = [float(value) for value in parts[1:]]
        xyxy = yolo_or_polygon_to_xyxy(coords, width, height)
        candidates.append(
            GroundTruthTarget(
                class_name=class_name,
                xyxy=xyxy,
                area=box_area(xyxy),
                source_label_count=0,
                selected_by="largest_target_class_box",
            )
        )

    if not candidates:
        return None

    selected = max(candidates, key=lambda item: item.area)
    return GroundTruthTarget(
        class_name=selected.class_name,
        xyxy=selected.xyxy,
        area=selected.area,
        source_label_count=len(candidates),
        selected_by=selected.selected_by,
    )


def model_name_lookup(model_names: object) -> dict[int, str]:
    if isinstance(model_names, dict):
        return {int(key): str(value) for key, value in model_names.items()}
    if isinstance(model_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(model_names)}
    return {index: name for index, name in enumerate(MODEL_CLASS_NAMES)}


def run_predictions(
    run_name: str,
    weights: Path,
    image_paths: Sequence[Path],
    predict_conf_threshold: float,
    imgsz: int,
) -> dict[str, list[Prediction]]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE / "UltralyticsConfig"))
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model_name_lookup(model.names)
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=predict_conf_threshold,
        imgsz=imgsz,
        verbose=False,
        save=False,
    )

    predictions_by_key: dict[str, list[Prediction]] = {}
    for image_path, result in zip(image_paths, results):
        key = image_key(image_path)
        rows: list[Prediction] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxys = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            for xyxy_raw, confidence, class_id_raw in zip(xyxys, confidences, class_ids):
                class_id = int(class_id_raw)
                rows.append(
                    Prediction(
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                        confidence=float(confidence),
                        xyxy=tuple(float(v) for v in xyxy_raw),
                    )
                )
        predictions_by_key[key] = rows

    print(f"{run_name}: predicted {sum(len(v) for v in predictions_by_key.values())} boxes on {len(image_paths)} images.")
    return predictions_by_key


def score_view(
    run_name: str,
    manifest_row: dict[str, object],
    gt_target: GroundTruthTarget | None,
    predictions: Sequence[Prediction],
    iou_threshold: float,
    detection_conf_threshold: float,
) -> ViewScore:
    target_class = str(manifest_row["target_class"])
    image_path = Path(str(manifest_row["image_path"]))
    label_path_raw = str(manifest_row.get("label_path", ""))
    label_path = Path(label_path_raw) if label_path_raw else None

    target_predictions = [pred for pred in predictions if pred.class_name == target_class]
    best_prediction_conf = max((pred.confidence for pred in target_predictions), default=0.0)
    if gt_target is None:
        return ViewScore(
            run_name=run_name,
            object_id=str(manifest_row["object_id"]),
            description=str(manifest_row["description"]),
            target_class=target_class,
            view_index=int(manifest_row["view_index"]),
            file_name=str(manifest_row["file_name"]),
            image_path=image_path,
            label_path=label_path,
            target_gt_present=False,
            target_gt_count=0,
            target_gt_area=0.0,
            target_match_confidence_iou50=0.0,
            target_match_iou_at_confidence_iou50=0.0,
            target_strict_quality_iou50=0.0,
            detected_at_conf25_iou50=0,
            target_prediction_count=len(target_predictions),
            best_target_prediction_confidence=best_prediction_conf,
            best_target_prediction_iou=0.0,
        )

    iou_rows = [(iou_xyxy(pred.xyxy, gt_target.xyxy), pred) for pred in target_predictions]
    best_iou_any = max((overlap for overlap, _ in iou_rows), default=0.0)
    matched = [(overlap, pred) for overlap, pred in iou_rows if overlap >= iou_threshold]
    if matched:
        best_overlap, best_pred = max(matched, key=lambda item: item[0] * item[1].confidence)
        match_conf = best_pred.confidence
        match_iou = best_overlap
        strict_quality = match_conf * match_iou
        detected = int(match_conf >= detection_conf_threshold)
    else:
        match_conf = 0.0
        match_iou = 0.0
        strict_quality = 0.0
        detected = 0

    return ViewScore(
        run_name=run_name,
        object_id=str(manifest_row["object_id"]),
        description=str(manifest_row["description"]),
        target_class=target_class,
        view_index=int(manifest_row["view_index"]),
        file_name=str(manifest_row["file_name"]),
        image_path=image_path,
        label_path=label_path,
        target_gt_present=True,
        target_gt_count=gt_target.source_label_count,
        target_gt_area=gt_target.area,
        target_match_confidence_iou50=match_conf,
        target_match_iou_at_confidence_iou50=match_iou,
        target_strict_quality_iou50=strict_quality,
        detected_at_conf25_iou50=detected,
        target_prediction_count=len(target_predictions),
        best_target_prediction_confidence=best_prediction_conf,
        best_target_prediction_iou=best_iou_any,
    )


def view_score_to_row(score: ViewScore) -> dict[str, object]:
    return {
        "run_name": score.run_name,
        "object_id": score.object_id,
        "description": score.description,
        "target_class": score.target_class,
        "view_index": score.view_index,
        "file_name": score.file_name,
        "image_path": str(score.image_path),
        "label_path": "" if score.label_path is None else str(score.label_path),
        "target_gt_present": int(score.target_gt_present),
        "target_gt_count": score.target_gt_count,
        "target_gt_area": score.target_gt_area,
        "target_prediction_count": score.target_prediction_count,
        "best_target_prediction_confidence": score.best_target_prediction_confidence,
        "best_target_prediction_iou": score.best_target_prediction_iou,
        "target_match_confidence_iou50": score.target_match_confidence_iou50,
        "target_match_iou_at_confidence_iou50": score.target_match_iou_at_confidence_iou50,
        "target_strict_quality_iou50": score.target_strict_quality_iou50,
        "detected_at_conf25_iou50": score.detected_at_conf25_iou50,
    }


def score_from_row(row: dict[str, str]) -> ViewScore:
    label_path = Path(row["label_path"]) if row.get("label_path") else None
    return ViewScore(
        run_name=row["run_name"],
        object_id=row["object_id"],
        description=row["description"],
        target_class=row["target_class"],
        view_index=int(row["view_index"]),
        file_name=row["file_name"],
        image_path=Path(row["image_path"]),
        label_path=label_path,
        target_gt_present=bool(int(float(row["target_gt_present"]))),
        target_gt_count=int(float(row["target_gt_count"])),
        target_gt_area=safe_float(row["target_gt_area"]),
        target_match_confidence_iou50=safe_float(row["target_match_confidence_iou50"]),
        target_match_iou_at_confidence_iou50=safe_float(row["target_match_iou_at_confidence_iou50"]),
        target_strict_quality_iou50=safe_float(row["target_strict_quality_iou50"]),
        detected_at_conf25_iou50=int(float(row["detected_at_conf25_iou50"])),
        target_prediction_count=int(float(row["target_prediction_count"])),
        best_target_prediction_confidence=safe_float(row["best_target_prediction_confidence"]),
        best_target_prediction_iou=safe_float(row["best_target_prediction_iou"]),
    )


def noisy_or(values: Sequence[float]) -> float:
    complement = 1.0
    for value in values:
        complement *= max(0.0, 1.0 - float(value))
    return 1.0 - complement


def coalition_values(scores: Sequence[ViewScore]) -> dict[str, float]:
    matched_confidences = [score.target_match_confidence_iou50 for score in scores]
    matched_ious = [score.target_match_iou_at_confidence_iou50 for score in scores]
    best_iou = max(matched_ious, default=0.0)
    noisy_conf = noisy_or(matched_confidences)
    return {
        "detection_at_conf25_iou50": float(max((score.detected_at_conf25_iou50 for score in scores), default=0)),
        "best_matched_confidence_iou50": max(matched_confidences, default=0.0),
        "best_matched_iou_iou50": best_iou,
        "best_strict_quality_iou50": max((score.target_strict_quality_iou50 for score in scores), default=0.0),
        "noisy_or_confidence_iou50": noisy_conf,
        "noisy_or_best_iou_iou50": noisy_conf * best_iou if best_iou > 0.0 else 0.0,
    }


def group_scores(scores: Sequence[ViewScore]) -> dict[tuple[str, str], list[ViewScore]]:
    grouped: dict[tuple[str, str], list[ViewScore]] = defaultdict(list)
    for score in scores:
        if score.target_gt_present:
            grouped[(score.run_name, score.object_id)].append(score)
    return dict(grouped)


def build_coalition_rows(scores: Sequence[ViewScore], max_k: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (run_name, object_id), object_scores in group_scores(scores).items():
        object_scores = sorted(object_scores, key=lambda item: item.view_index)
        for k in range(1, min(max_k, len(object_scores)) + 1):
            for combo in itertools.combinations(object_scores, k):
                values = coalition_values(combo)
                rows.append(
                    {
                        "run_name": run_name,
                        "object_id": object_id,
                        "description": combo[0].description,
                        "target_class": combo[0].target_class,
                        "available_view_count": len(object_scores),
                        "coalition_size": k,
                        "view_indices": "|".join(str(score.view_index) for score in combo),
                        "file_names": "|".join(score.file_name for score in combo),
                        **values,
                    }
                )
    return rows


def summarize_object_coalitions(coalition_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in coalition_rows:
        grouped[(str(row["run_name"]), str(row["object_id"]), int(row["coalition_size"]))].append(row)

    rows: list[dict[str, object]] = []
    for (run_name, object_id, k), members in sorted(grouped.items()):
        first = members[0]
        summary = {
            "run_name": run_name,
            "object_id": object_id,
            "description": first["description"],
            "target_class": first["target_class"],
            "available_view_count": first["available_view_count"],
            "coalition_size": k,
            "combination_count": len(members),
        }
        for metric in coalition_values([]):
            summary[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in members]))
            summary[f"max_{metric}"] = float(np.max([float(row[metric]) for row in members]))
        rows.append(summary)
    return rows


def summarize_overall(object_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[(str(row["run_name"]), int(row["coalition_size"]))].append(row)

    rows: list[dict[str, object]] = []
    for (run_name, k), members in sorted(grouped.items()):
        summary = {
            "run_name": run_name,
            "coalition_size": k,
            "object_support": len(members),
            "combination_count": int(sum(int(row["combination_count"]) for row in members)),
        }
        for metric in coalition_values([]):
            summary[f"macro_mean_{metric}"] = float(np.mean([float(row[f"mean_{metric}"]) for row in members]))
        rows.append(summary)

    by_run = defaultdict(dict)
    for row in rows:
        by_run[str(row["run_name"])][int(row["coalition_size"])] = row

    for run_name, by_k in by_run.items():
        for k, row in by_k.items():
            for metric in coalition_values([]):
                current = float(row[f"macro_mean_{metric}"])
                previous = 0.0 if k == 1 else float(by_k.get(k - 1, {}).get(f"macro_mean_{metric}", math.nan))
                row[f"marginal_gain_vs_k_minus_1_{metric}"] = current - previous if not math.isnan(previous) else math.nan
    return sorted(rows, key=lambda row: (str(row["run_name"]), int(row["coalition_size"])))


def summarize_by_class(object_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[(str(row["run_name"]), str(row["target_class"]), int(row["coalition_size"]))].append(row)

    rows: list[dict[str, object]] = []
    for (run_name, target_class, k), members in sorted(grouped.items()):
        summary = {
            "run_name": run_name,
            "target_class": target_class,
            "coalition_size": k,
            "object_support": len(members),
            "combination_count": int(sum(int(row["combination_count"]) for row in members)),
        }
        for metric in coalition_values([]):
            summary[f"macro_mean_{metric}"] = float(np.mean([float(row[f"mean_{metric}"]) for row in members]))
        rows.append(summary)
    return rows


def dense_rank_desc(rows: Sequence[dict[str, object]], value_key: str, rank_key: str) -> None:
    ranked = sorted(rows, key=lambda row: (-float(row[value_key]), int(row["view_id"])))
    rank = 0
    previous: float | None = None
    for row in ranked:
        value = float(row[value_key])
        if previous is None or not math.isclose(value, previous, abs_tol=1e-12):
            rank += 1
            previous = value
        row[rank_key] = rank


def exact_shapley_analysis(
    scores: Sequence[ViewScore],
    tolerance: float = SHAPLEY_EFFICIENCY_TOL,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Enumerate every same-object coalition and calculate exact standard Shapley values."""
    contribution_rows: list[dict[str, object]] = []
    marginal_size_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    exact_coalition_rows: list[dict[str, object]] = []

    for (run_name, object_id), object_scores in group_scores(scores).items():
        object_scores = sorted(object_scores, key=lambda item: item.view_index)
        n = len(object_scores)
        if n == 0:
            continue

        file_names = [score.file_name for score in object_scores]
        if len(file_names) != len(set(file_names)):
            raise AssertionError(f"Duplicate filenames remain in {run_name}/{object_id}; duplicates cannot be Shapley players.")

        all_indices = tuple(range(n))
        subset_cache: dict[frozenset[int], dict[str, float]] = {}
        for subset_size in range(n + 1):
            for subset in itertools.combinations(all_indices, subset_size):
                subset_key = frozenset(subset)
                subset_scores = [object_scores[index] for index in subset]
                subset_cache[subset_key] = coalition_values(subset_scores)
                view_ids = [object_scores[index].view_index for index in subset]
                subset_files = [object_scores[index].file_name for index in subset]
                values = subset_cache[subset_key]
                exact_coalition_rows.append(
                    {
                        "model": run_name,
                        "object": object_id,
                        "target_class": object_scores[0].target_class,
                        "n_views_in_game": n,
                        "coalition_size": subset_size,
                        "coalition_view_ids": "empty" if not view_ids else "|".join(str(view_id) for view_id in view_ids),
                        "coalition_filenames": "empty" if not subset_files else "|".join(subset_files),
                        "detection": values["detection_at_conf25_iou50"],
                        "best_strict_quality": values["best_strict_quality_iou50"],
                        "noisy_or_best_iou": values["noisy_or_best_iou_iou50"],
                        "best_matched_confidence": values["best_matched_confidence_iou50"],
                        "best_matched_iou": values["best_matched_iou_iou50"],
                        "noisy_or_confidence": values["noisy_or_confidence_iou50"],
                    }
                )

        expected_coalitions = 2**n
        enumerated_coalitions = len(subset_cache)
        if enumerated_coalitions != expected_coalitions:
            raise AssertionError(
                f"{run_name}/{object_id} enumerated {enumerated_coalitions} coalitions, expected {expected_coalitions}."
            )

        phi_by_view_metric = {
            index: {value_name: 0.0 for value_name in SHAPLEY_VALUE_FUNCTIONS}
            for index in all_indices
        }
        group_rows: list[dict[str, object]] = []

        for index, score in enumerate(object_scores):
            others = [other for other in all_indices if other != index]
            size_totals: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: {value_name: [] for value_name in SHAPLEY_VALUE_FUNCTIONS}
            )

            for predecessor_size in range(n):
                subset_weight = math.factorial(predecessor_size) * math.factorial(n - predecessor_size - 1) / math.factorial(n)
                for subset in itertools.combinations(others, predecessor_size):
                    without_key = frozenset(subset)
                    with_key = frozenset((*subset, index))
                    for value_name, metric in SHAPLEY_VALUE_FUNCTIONS.items():
                        delta = subset_cache[with_key][metric] - subset_cache[without_key][metric]
                        phi_by_view_metric[index][value_name] += subset_weight * delta
                        size_totals[predecessor_size][value_name].append(delta)

            single_values = coalition_values([score])
            group_rows.append(
                {
                    "model": run_name,
                    "object": object_id,
                    "target_class": score.target_class,
                    "view_id": score.view_index,
                    "filename": score.file_name,
                    "n_views_in_game": n,
                    "single_view_detection": score.detected_at_conf25_iou50,
                    "single_view_strict_quality": score.target_strict_quality_iou50,
                    "single_view_noisy_or_quality": single_values["noisy_or_best_iou_iou50"],
                    "shapley_detection": phi_by_view_metric[index]["detection"],
                    "shapley_strict_quality": phi_by_view_metric[index]["strict_quality"],
                    "shapley_noisy_or_best_iou": phi_by_view_metric[index]["noisy_or_best_iou"],
                }
            )

            for predecessor_size, metric_values in sorted(size_totals.items()):
                predecessor_count = len(next(iter(metric_values.values()))) if metric_values else 0
                row = {
                    "model": run_name,
                    "object": object_id,
                    "target_class": score.target_class,
                    "view_id": score.view_index,
                    "filename": score.file_name,
                    "n_views_in_game": n,
                    "predecessor_coalition_size": predecessor_size,
                    "predecessor_coalition_count": predecessor_count,
                    "shapley_weight_per_coalition": (
                        math.factorial(predecessor_size) * math.factorial(n - predecessor_size - 1) / math.factorial(n)
                    ),
                    "shapley_total_weight_for_size": predecessor_count
                    * math.factorial(predecessor_size)
                    * math.factorial(n - predecessor_size - 1)
                    / math.factorial(n),
                }
                for value_name in SHAPLEY_VALUE_FUNCTIONS:
                    values = metric_values[value_name]
                    row[f"mean_marginal_{value_name}"] = float(np.mean(values)) if values else math.nan
                marginal_size_rows.append(row)

        dense_rank_desc(group_rows, "shapley_strict_quality", "rank_shapley_strict")
        dense_rank_desc(group_rows, "shapley_detection", "rank_shapley_detection")

        grand_key = frozenset(all_indices)
        for value_name, metric in SHAPLEY_VALUE_FUNCTIONS.items():
            grand_value = subset_cache[grand_key][metric]
            sum_shapley = sum(phi_by_view_metric[index][value_name] for index in all_indices)
            efficiency_error = abs(sum_shapley - grand_value)
            if efficiency_error > tolerance:
                raise AssertionError(
                    f"Shapley efficiency failed for {run_name}/{object_id}/{value_name}: {efficiency_error}."
                )
            validation_rows.append(
                {
                    "model": run_name,
                    "object": object_id,
                    "value_function": value_name,
                    "n_views": n,
                    "expected_coalitions": expected_coalitions,
                    "enumerated_coalitions": enumerated_coalitions,
                    "grand_coalition_value": grand_value,
                    "sum_shapley": sum_shapley,
                    "efficiency_error": efficiency_error,
                }
            )

        for row in group_rows:
            if (
                abs(float(row["single_view_detection"])) <= tolerance
                and abs(float(row["single_view_strict_quality"])) <= tolerance
                and abs(float(row["single_view_noisy_or_quality"])) <= tolerance
            ):
                max_abs_phi = max(
                    abs(float(row["shapley_detection"])),
                    abs(float(row["shapley_strict_quality"])),
                    abs(float(row["shapley_noisy_or_best_iou"])),
                )
                if max_abs_phi > tolerance:
                    raise AssertionError(
                        f"Null-view check failed for {run_name}/{object_id}/view {row['view_id']}: {max_abs_phi}."
                    )

        score_by_view_id = {score.view_index: score for score in object_scores}
        for left, right in itertools.combinations(group_rows, 2):
            left_score = score_by_view_id[int(left["view_id"])]
            right_score = score_by_view_id[int(right["view_id"])]
            signature_left = (
                left_score.detected_at_conf25_iou50,
                left_score.target_match_confidence_iou50,
                left_score.target_match_iou_at_confidence_iou50,
                left_score.target_strict_quality_iou50,
            )
            signature_right = (
                right_score.detected_at_conf25_iou50,
                right_score.target_match_confidence_iou50,
                right_score.target_match_iou_at_confidence_iou50,
                right_score.target_strict_quality_iou50,
            )
            if all(math.isclose(a, b, abs_tol=tolerance) for a, b in zip(signature_left, signature_right)):
                for value_name in SHAPLEY_VALUE_FUNCTIONS:
                    key = f"shapley_{value_name}"
                    if not math.isclose(float(left[key]), float(right[key]), abs_tol=tolerance):
                        raise AssertionError(
                            f"Symmetry check failed for {run_name}/{object_id}/views {left['view_id']} and {right['view_id']}."
                        )

        contribution_rows.extend(sorted(group_rows, key=lambda row: int(row["view_id"])))

    return contribution_rows, marginal_size_rows, validation_rows, exact_coalition_rows


def restricted_k_view_contribution_rows(scores: Sequence[ViewScore], max_k: int = 3) -> list[dict[str, object]]:
    """Average each view's marginal contribution over same-object coalitions up to size max_k."""
    rows: list[dict[str, object]] = []
    for (run_name, object_id), object_scores in group_scores(scores).items():
        object_scores = sorted(object_scores, key=lambda item: item.view_index)
        n = len(object_scores)
        if n == 0:
            continue

        for index, score in enumerate(object_scores):
            others = [other for other in range(n) if other != index]
            marginal_values_by_metric: dict[str, list[float]] = {metric: [] for metric in MAIN_METRICS}
            marginal_values_by_k: dict[int, dict[str, list[float]]] = {
                k: {metric: [] for metric in MAIN_METRICS}
                for k in range(1, min(max_k, n) + 1)
            }
            coalition_count_by_k: dict[int, int] = {}

            for coalition_size in range(1, min(max_k, n) + 1):
                count_for_k = 0
                for subset in itertools.combinations(others, coalition_size - 1):
                    without_scores = [object_scores[other] for other in subset]
                    with_scores = without_scores + [score]
                    without_values = coalition_values(without_scores)
                    with_values = coalition_values(with_scores)
                    count_for_k += 1
                    for metric in MAIN_METRICS:
                        delta = with_values[metric] - without_values[metric]
                        marginal_values_by_metric[metric].append(delta)
                        marginal_values_by_k[coalition_size][metric].append(delta)
                coalition_count_by_k[coalition_size] = count_for_k

            row = {
                "run_name": run_name,
                "object_id": object_id,
                "description": score.description,
                "target_class": score.target_class,
                "available_view_count": n,
                "view_index": score.view_index,
                "file_name": score.file_name,
                "coalition_count_k1_to_k3": sum(coalition_count_by_k.values()),
                "single_view_detection_at_conf25_iou50": score.detected_at_conf25_iou50,
                "single_view_matched_confidence_iou50": score.target_match_confidence_iou50,
                "single_view_matched_iou_iou50": score.target_match_iou_at_confidence_iou50,
                "single_view_strict_quality_iou50": score.target_strict_quality_iou50,
            }
            for k in range(1, max_k + 1):
                row[f"coalition_count_k{k}"] = coalition_count_by_k.get(k, 0)
            for metric in MAIN_METRICS:
                values = marginal_values_by_metric[metric]
                row[f"mean_marginal_k1_to_k3_{metric}"] = float(np.mean(values)) if values else math.nan
                row[f"positive_marginal_rate_k1_to_k3_{metric}"] = (
                    float(np.mean([1.0 if value > 1e-12 else 0.0 for value in values])) if values else math.nan
                )
                for k in range(1, max_k + 1):
                    k_values = marginal_values_by_k.get(k, {}).get(metric, [])
                    row[f"mean_marginal_k{k}_{metric}"] = float(np.mean(k_values)) if k_values else math.nan
            rows.append(row)

    return sorted(rows, key=lambda row: (str(row["run_name"]), str(row["object_id"]), int(row["view_index"])))


def plot_progression(overall_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_names = sorted({str(row["run_name"]) for row in overall_rows})
    colors = {
        "real_uav_finetuned": "#1f6f8b",
        "synthetic_m4": "#b45309",
    }
    labels = {
        "real_uav_finetuned": "Real UAV fine-tuned",
        "synthetic_m4": "Synthetic M4 only",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for run_name in run_names:
        rows = sorted([row for row in overall_rows if row["run_name"] == run_name], key=lambda row: int(row["coalition_size"]))
        x = [int(row["coalition_size"]) for row in rows]
        detection = [float(row["macro_mean_detection_at_conf25_iou50"]) for row in rows]
        strict = [float(row["macro_mean_best_strict_quality_iou50"]) for row in rows]
        fusion = [float(row["macro_mean_noisy_or_best_iou_iou50"]) for row in rows]
        color = colors.get(run_name, None)
        axes[0].plot(x, detection, marker="o", linewidth=2, label=labels.get(run_name, run_name), color=color)
        axes[1].plot(x, strict, marker="o", linewidth=2, label=labels.get(run_name, run_name), color=color)
        axes[2].plot(x, fusion, marker="o", linewidth=2, label=labels.get(run_name, run_name), color=color)

    axes[0].set_title("Detection probability by number of views")
    axes[0].set_xlabel("Views in coalition")
    axes[0].set_ylabel("Macro mean any-detected rate")
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_title("Best strict quality by number of views")
    axes[1].set_xlabel("Views in coalition")
    axes[1].set_ylabel("Macro mean best strict quality")
    axes[1].set_xticks([1, 2, 3])
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].set_title("Heuristic fusion quality by number of views")
    axes[2].set_xlabel("Views in coalition")
    axes[2].set_ylabel("Macro mean Noisy-OR x best IoU")
    axes[2].set_xticks([1, 2, 3])
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def object_short_label(object_id: str) -> str:
    return {
        "la_souris_truck": "LS",
        "m_truck": "M",
        "ooij_tower": "OT",
        "parkinglot_car": "PC",
        "white_truck_bottendaal": "WB",
    }.get(object_id, object_id)


def object_display_label(object_id: str) -> str:
    return {
        "la_souris_truck": "La Souris truck",
        "m_truck": "M truck",
        "ooij_tower": "Ooij tower",
        "parkinglot_car": "parking-lot car",
        "white_truck_bottendaal": "white truck Bottendaal",
    }.get(object_id, object_id)


def plot_exact_shapley_figures(rows: Sequence[dict[str, object]], output_dir: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    real_rows = [row for row in rows if row["model"] == "real_uav_finetuned"]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in real_rows:
        grouped[str(row["object"])].append(row)

    object_order = ["parkinglot_car", "ooij_tower", "la_souris_truck", "m_truck", "white_truck_bottendaal"]
    object_order = [object_id for object_id in object_order if object_id in grouped]
    colors = {
        "parkinglot_car": "#2563eb",
        "ooij_tower": "#7c3aed",
        "la_souris_truck": "#059669",
        "m_truck": "#dc2626",
        "white_truck_bottendaal": "#ca8a04",
    }

    fig, axes = plt.subplots(len(object_order), 1, figsize=(11, 12), sharex=True)
    if len(object_order) == 1:
        axes = [axes]
    for ax, object_id in zip(axes, object_order):
        members = sorted(grouped[object_id], key=lambda row: int(row["view_id"]))
        y = np.arange(len(members))
        values = [float(row["shapley_strict_quality"]) for row in members]
        labels = [f"v{int(row['view_id'])}" for row in members]
        ax.barh(y, values, color=colors.get(object_id, "#4b5563"), alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(object_display_label(object_id))
        ax.grid(axis="x", alpha=0.25)
        for y_pos, value in zip(y, values):
            ax.text(value + 0.003, y_pos, f"{value:.3f}", va="center", fontsize=8)
    axes[-1].set_xlabel("Exact Shapley value: strict target quality")
    fig.suptitle("Exact per-view Shapley contributions, real UAV fine-tuned model", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "exact_shapley_strict_by_view_real_finetuned.png", dpi=220)
    plt.close(fig)

    x = np.array([float(row["single_view_strict_quality"]) for row in real_rows], dtype=float)
    y = np.array([float(row["shapley_strict_quality"]) for row in real_rows], dtype=float)
    if len(real_rows) > 1 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        correlation = math.nan

    fig, ax = plt.subplots(figsize=(11, 7.5))
    for object_id in object_order:
        members = sorted(grouped[object_id], key=lambda row: int(row["view_id"]))
        xs = [float(row["single_view_strict_quality"]) for row in members]
        ys = [float(row["shapley_strict_quality"]) for row in members]
        ax.scatter(xs, ys, s=42, color=colors.get(object_id, "#4b5563"), label=object_display_label(object_id), alpha=0.9)
        for local_index, (row, x_val, y_val) in enumerate(zip(members, xs, ys)):
            x_offset = 5 if local_index % 2 == 0 else -28
            y_offset = [-14, -6, 2, 10, 18][local_index % 5]
            ax.annotate(
                f"{object_short_label(object_id)}-v{int(row['view_id'])}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                fontsize=6,
            )
    ax.set_xlabel("Single-view strict target quality")
    ax.set_ylabel("Exact Shapley value: strict target quality")
    ax.set_title("Single-view quality versus exact Shapley contribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "single_view_vs_exact_shapley_strict_real_finetuned.png", dpi=220)
    plt.close(fig)

    return {
        "real_finetuned_single_vs_shapley_strict_pearson_r": correlation,
        "real_finetuned_scatter_n_views": len(real_rows),
    }


def build_report(
    output_path: Path,
    manifest_rows: Sequence[dict[str, object]],
    issue_rows: Sequence[dict[str, object]],
    view_scores: Sequence[ViewScore],
    object_rows: Sequence[dict[str, object]],
    overall_rows: Sequence[dict[str, object]],
    class_rows: Sequence[dict[str, object]],
    shapley_rows: Sequence[dict[str, object]],
    restricted_shapley_rows: Sequence[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    run_label = {
        "real_uav_finetuned": "real UAV fine-tuned",
        "synthetic_m4": "synthetic M4 only",
    }
    usable_target_by_view: dict[tuple[str, int, str], bool] = {}
    for score in view_scores:
        usable_target_by_view[(score.object_id, score.view_index, score.file_name)] = score.target_gt_present

    usable_target_count_by_object: dict[str, int] = defaultdict(int)
    missing_target_rows: list[tuple[str, int, str, str]] = []
    for (object_id, view_index, file_name), target_present in sorted(usable_target_by_view.items()):
        if target_present:
            usable_target_count_by_object[object_id] += 1
        else:
            target_class = next(str(row["target_class"]) for row in manifest_rows if row["object_id"] == object_id)
            missing_target_rows.append((object_id, view_index, file_name, target_class))

    lines = [
        "# Real-World Same-Object Multi-View Shapley Analysis",
        "",
        "## Scope",
        "",
        "- Uses only the five user-specified real UAV object instances with multiple viewpoints.",
        "- Duplicate filenames inside a supplied object list are removed before analysis.",
        "- Coalitions are formed only within the same object instance, never across objects.",
        "- Target-class mapping: parking-lot car -> `suv`, Ooij tower -> `tower`, trucks -> `whitevan`.",
        f"- Target match rule: same-class detection with IoU >= {args.iou_threshold:.2f}; binary detection also requires confidence >= {args.detection_conf_threshold:.2f}.",
        "- Ground-truth labels for the timestamped UAV images use the 5-class order documented in the copied real multiview dataset.",
        "- If multiple target-class ground-truth boxes exist in a view, the largest target-class box is used as the named target instance proxy.",
        "",
        "## Object Groups",
        "",
        "| Object | Target class | Unique views | Labels present | Usable target views |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    grouped_manifest: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        grouped_manifest[str(row["object_id"])].append(row)
    for object_id, rows in grouped_manifest.items():
        target_class = rows[0]["target_class"]
        labels_present = sum(int(row["label_exists"]) for row in rows)
        usable_targets = usable_target_count_by_object.get(object_id, 0)
        lines.append(f"| `{object_id}` | `{target_class}` | {len(rows)} | {labels_present} | {usable_targets} |")

    if issue_rows:
        lines.extend(["", "## Manifest Issues", ""])
        for issue in issue_rows:
            lines.append(f"- `{issue['object_id']}` / `{issue['file_name']}`: {issue['message']}")

    if missing_target_rows:
        lines.extend(["", "## Target-GT Exclusions", ""])
        for object_id, view_index, file_name, target_class in missing_target_rows:
            lines.append(
                f"- `{object_id}` view {view_index} (`{file_name}`) has a label file but no `{target_class}` ground-truth box, so it is excluded from coalitions."
            )

    lines.extend(
        [
            "",
            "## Overall 1-2-3 View Progression",
            "",
            "Values are macro-averaged over objects, so a 10-view object does not dominate a 5-view object.",
            "",
            "| Run | Views | Objects | Detection rate | Marginal detection gain | Best matched confidence | Best strict quality | Noisy-OR x best IoU | Marginal Noisy-OR gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in sorted(overall_rows, key=lambda item: (str(item["run_name"]), int(item["coalition_size"]))):
        run_name = str(row["run_name"])
        lines.append(
            "| {run} | {k} | {objects} | {det} | {det_gain} | {conf} | {strict} | {fusion} | {fusion_gain} |".format(
                run=run_label.get(run_name, run_name),
                k=int(row["coalition_size"]),
                objects=int(row["object_support"]),
                det=fmt(row["macro_mean_detection_at_conf25_iou50"]),
                det_gain=fmt(row["marginal_gain_vs_k_minus_1_detection_at_conf25_iou50"]),
                conf=fmt(row["macro_mean_best_matched_confidence_iou50"]),
                strict=fmt(row["macro_mean_best_strict_quality_iou50"]),
                fusion=fmt(row["macro_mean_noisy_or_best_iou_iou50"]),
                fusion_gain=fmt(row["marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Object Progression",
            "",
            "| Run | Object | Views | Detection rate | Best strict quality | Noisy-OR x best IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(object_rows, key=lambda item: (str(item["run_name"]), str(item["object_id"]), int(item["coalition_size"]))):
        lines.append(
            "| {run} | `{obj}` | {k} | {det} | {strict} | {fusion} |".format(
                run=run_label.get(str(row["run_name"]), str(row["run_name"])),
                obj=row["object_id"],
                k=int(row["coalition_size"]),
                det=fmt(row["mean_detection_at_conf25_iou50"]),
                strict=fmt(row["mean_best_strict_quality_iou50"]),
                fusion=fmt(row["mean_noisy_or_best_iou_iou50"]),
            )
        )

    lines.extend(
        [
            "",
            "## Class-Level Progression",
            "",
            "| Run | Class | Views | Object support | Detection rate | Noisy-OR x best IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(class_rows, key=lambda item: (str(item["run_name"]), str(item["target_class"]), int(item["coalition_size"]))):
        lines.append(
            "| {run} | `{cls}` | {k} | {support} | {det} | {fusion} |".format(
                run=run_label.get(str(row["run_name"]), str(row["run_name"])),
                cls=row["target_class"],
                k=int(row["coalition_size"]),
                support=int(row["object_support"]),
                det=fmt(row["macro_mean_detection_at_conf25_iou50"]),
                fusion=fmt(row["macro_mean_noisy_or_best_iou_iou50"]),
            )
        )

    if shapley_rows:
        top_rows = sorted(
            shapley_rows,
            key=lambda row: float(row["shapley_noisy_or_best_iou_iou50"]),
            reverse=True,
        )[:12]
        lines.extend(
            [
                "",
                "## Top Shapley-Style View Contributions",
                "",
                "These rows rank named views by exact Shapley value within their same-object game, using `Noisy-OR x best IoU`.",
                "",
                "| Run | Object | View | File | Shapley Noisy-OR x best IoU | Shapley detection |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in top_rows:
            lines.append(
                "| {run} | `{obj}` | {view} | `{file}` | {fusion} | {det} |".format(
                    run=run_label.get(str(row["run_name"]), str(row["run_name"])),
                    obj=row["object_id"],
                    view=int(row["view_index"]),
                    file=row["file_name"],
                    fusion=fmt(row["shapley_noisy_or_best_iou_iou50"]),
                    det=fmt(row["shapley_detection_at_conf25_iou50"]),
                )
            )

    if restricted_shapley_rows:
        best_restricted = sorted(
            restricted_shapley_rows,
            key=lambda row: (
                str(row["run_name"]) != "real_uav_finetuned",
                str(row["object_id"]),
                -float(row["mean_marginal_k1_to_k3_noisy_or_best_iou_iou50"]),
            ),
        )[:12]
        lines.extend(
            [
                "",
                "## 1/2/3-View Per-View Contributions",
                "",
                "The full table is in `shapley_k1_to_k3_view_contributions.csv`. For each view, the reported value averages `value(coalition) - value(coalition without this view)` over all same-object coalitions of size 1, 2, or 3 that contain the view.",
                "",
                "| Run | Object | View | File | Mean marginal Noisy-OR x best IoU | Mean marginal detection |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in best_restricted:
            lines.append(
                "| {run} | `{obj}` | {view} | `{file}` | {fusion} | {det} |".format(
                    run=run_label.get(str(row["run_name"]), str(row["run_name"])),
                    obj=row["object_id"],
                    view=int(row["view_index"]),
                    file=row["file_name"],
                    fusion=fmt(row["mean_marginal_k1_to_k3_noisy_or_best_iou_iou50"]),
                    det=fmt(row["mean_marginal_k1_to_k3_detection_at_conf25_iou50"]),
                )
            )

    finetuned_by_k = {
        int(row["coalition_size"]): row
        for row in overall_rows
        if row["run_name"] == "real_uav_finetuned"
    }
    if 1 in finetuned_by_k and 2 in finetuned_by_k and 3 in finetuned_by_k:
        gain_2 = float(finetuned_by_k[2]["marginal_gain_vs_k_minus_1_detection_at_conf25_iou50"])
        gain_3 = float(finetuned_by_k[3]["marginal_gain_vs_k_minus_1_detection_at_conf25_iou50"])
        fusion_gain_2 = float(finetuned_by_k[2]["marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50"])
        fusion_gain_3 = float(finetuned_by_k[3]["marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50"])
        lines.extend(
            [
                "",
                "## Headline Interpretation",
                "",
                f"- For the real UAV fine-tuned model, adding a second view changes macro detection rate by `{fmt(gain_2)}` and `Noisy-OR x best IoU` by `{fmt(fusion_gain_2)}`.",
                f"- Adding a third view changes macro detection rate by `{fmt(gain_3)}` and `Noisy-OR x best IoU` by `{fmt(fusion_gain_3)}`.",
                "- This directly answers the count-first question: the experiment estimates the marginal value of another independent real UAV view, rather than asking which exact angle is intrinsically best.",
            ]
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `real_multiview_manifest.csv`: object/view manifest.",
            "- `per_view_target_scores.csv`: target-centric single-view match scores.",
            "- `coalition_scores_k1_to_k3.csv`: all 1-, 2-, and 3-view combinations.",
            "- `object_coalition_size_summary.csv`: per-object average coalition values.",
            "- `overall_coalition_size_summary.csv`: thesis headline 1 -> 2 -> 3 view table.",
            "- `class_coalition_size_summary.csv`: class-level aggregation.",
            "- `shapley_view_contributions.csv`: exact Shapley-style named-view contributions.",
            "- `shapley_size_conditioned_marginals.csv`: marginal contribution by added-view position.",
            "- `shapley_k1_to_k3_view_contributions.csv`: per-view contribution averaged only over 1-, 2-, and 3-view coalitions.",
            "- `shapley_k1_to_k3_view_contributions.md`: readable per-object version of the 1/2/3-view contribution table.",
            "- `real_multiview_progression.png`: progression plot.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_revised_report(
    output_path: Path,
    manifest_rows: Sequence[dict[str, object]],
    issue_rows: Sequence[dict[str, object]],
    view_scores: Sequence[ViewScore],
    object_rows: Sequence[dict[str, object]],
    overall_rows: Sequence[dict[str, object]],
    class_rows: Sequence[dict[str, object]],
    exact_shapley_rows: Sequence[dict[str, object]],
    validation_rows: Sequence[dict[str, object]],
    figure_stats: dict[str, object],
    args: argparse.Namespace,
) -> None:
    run_label = {
        "real_uav_finetuned": "Real UAV fine-tuned",
        "synthetic_m4": "Synthetic M4 only",
    }
    usable_target_by_view: dict[tuple[str, int, str], bool] = {}
    for score in view_scores:
        usable_target_by_view[(score.object_id, score.view_index, score.file_name)] = score.target_gt_present

    usable_target_count_by_object: dict[str, int] = defaultdict(int)
    missing_target_rows: list[tuple[str, int, str, str]] = []
    for (object_id, view_index, file_name), target_present in sorted(usable_target_by_view.items()):
        if target_present:
            usable_target_count_by_object[object_id] += 1
        else:
            target_class = next(str(row["target_class"]) for row in manifest_rows if row["object_id"] == object_id)
            missing_target_rows.append((object_id, view_index, file_name, target_class))

    by_run_k = {
        (str(row["run_name"]), int(row["coalition_size"])): row
        for row in overall_rows
    }
    real_k1 = by_run_k.get(("real_uav_finetuned", 1), {})
    real_k2 = by_run_k.get(("real_uav_finetuned", 2), {})
    real_k3 = by_run_k.get(("real_uav_finetuned", 3), {})

    real_rows = [row for row in exact_shapley_rows if row["model"] == "real_uav_finetuned"]
    synthetic_rows = [row for row in exact_shapley_rows if row["model"] == "synthetic_m4"]
    shapley_by_model_object: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in exact_shapley_rows:
        shapley_by_model_object[(str(row["model"]), str(row["object"]))].append(row)

    validation_by_model_value: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in validation_rows:
        validation_by_model_value[(str(row["model"]), str(row["value_function"]))].append(row)

    def macro_grand(model: str, value_function: str) -> float:
        rows = validation_by_model_value.get((model, value_function), [])
        return float(np.mean([float(row["grand_coalition_value"]) for row in rows])) if rows else math.nan

    def near_zero_count(rows: Sequence[dict[str, object]], key: str, threshold: float = 1e-6) -> int:
        return sum(1 for row in rows if abs(float(row[key])) <= threshold)

    corr = figure_stats.get("real_finetuned_single_vs_shapley_strict_pearson_r", math.nan)
    corr_text = fmt(corr) if not math.isnan(float(corr)) else "n/a"

    lines = [
        "# Revised Real-World Multi-View Shapley Analysis",
        "",
        "## Scope",
        "",
        "- Uses exactly the five same-object real UAV sequences specified by the user.",
        "- Duplicate filenames inside the supplied object lists are removed before analysis and are not counted as separate players.",
        "- Coalitions are formed only within the same physical object instance.",
        "- Target-class mapping: `parkinglot_car` -> `suv`, `ooij_tower` -> `tower`, and truck objects -> `whitevan`.",
        f"- Target detection rule: correct target class, IoU >= {args.iou_threshold:.2f}, and confidence >= {args.detection_conf_threshold:.2f}.",
        "- The Ooij tower image `2026-06-29 17.35.32.jpg` remains excluded because its label file has no tower ground-truth target.",
        "- No model retraining is performed; the analysis reuses the existing per-view prediction table when `--skip-predict` is used.",
        "",
        "## Object Groups",
        "",
        "| Object | Target class | Unique listed views | Usable target views |",
        "| --- | --- | ---: | ---: |",
    ]

    grouped_manifest: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        grouped_manifest[str(row["object_id"])].append(row)
    for object_id, rows in grouped_manifest.items():
        lines.append(
            f"| `{object_id}` | `{rows[0]['target_class']}` | {len(rows)} | {usable_target_count_by_object.get(object_id, 0)} |"
        )

    if issue_rows:
        lines.extend(["", "## Manifest Issues", ""])
        for issue in issue_rows:
            lines.append(f"- `{issue['object_id']}` / `{issue['file_name']}`: {issue['message']}")

    if missing_target_rows:
        lines.extend(["", "## Target-GT Exclusions", ""])
        for object_id, view_index, file_name, target_class in missing_target_rows:
            lines.append(
                f"- `{object_id}` view {view_index} (`{file_name}`) has no `{target_class}` target box and is excluded from Shapley games."
            )

    lines.extend(
        [
            "",
            "## Analysis A: Coalition-Size Progression",
            "",
            "Question: how much does performance improve, on average, when moving from one to two to three UAV views?",
            "",
            "Values are macro-averaged over physical object instances, so the 10-view object does not dominate the 5-view object.",
            "",
            "| Model | Views | Object support | Detection | Gain detection | Best strict quality | Gain strict quality | Noisy-OR x best IoU | Gain Noisy-OR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(overall_rows, key=lambda item: (str(item["run_name"]), int(item["coalition_size"]))):
        lines.append(
            "| {run} | {k} | {objects} | {det} | {det_gain} | {strict} | {strict_gain} | {fusion} | {fusion_gain} |".format(
                run=run_label.get(str(row["run_name"]), str(row["run_name"])),
                k=int(row["coalition_size"]),
                objects=int(row["object_support"]),
                det=fmt(row["macro_mean_detection_at_conf25_iou50"]),
                det_gain=fmt(row["marginal_gain_vs_k_minus_1_detection_at_conf25_iou50"]),
                strict=fmt(row["macro_mean_best_strict_quality_iou50"]),
                strict_gain=fmt(row["marginal_gain_vs_k_minus_1_best_strict_quality_iou50"]),
                fusion=fmt(row["macro_mean_noisy_or_best_iou_iou50"]),
                fusion_gain=fmt(row["marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50"]),
            )
        )

    lines.extend(
        [
            "",
            "## Analysis B: Exact Shapley Viewpoint Attribution",
            "",
            "Question: across all possible same-object view combinations, which individual viewpoints account for coalition value?",
            "",
            "For an object with `n` usable views, every subset of those views is enumerated exactly. The empty coalition has value 0, and each view receives the standard factorial-weighted Shapley value over predecessor coalitions of size 0 through `n-1`.",
            "",
            "Value functions:",
            "",
            "- Primary: `best strict target quality`, defined as the maximum strict target-quality score in the coalition.",
            "- Detection/rescue: 1 if any view correctly detects the target, otherwise 0.",
            "- Secondary sensitivity metric: `Noisy-OR x best IoU`. This is a heuristic fusion score. YOLO confidence is not treated as a calibrated probability, and the Noisy-OR component naturally tends to increase when additional positive-confidence observations are added.",
            "",
            "## Shapley Efficiency Validation",
            "",
            "Each row checks that the sum of view Shapley values equals the grand-coalition value. The script asserts an absolute error below `1e-8`; it also asserts no duplicate filenames remain, null views receive zero contribution, symmetric views receive equal values, and the coalition count equals `2^n`.",
            "",
            "| Model | Object | Value function | Views | Expected coalitions | Enumerated | Grand value | Sum Shapley | Error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(validation_rows, key=lambda item: (str(item["model"]), str(item["object"]), str(item["value_function"]))):
        lines.append(
            "| {model} | `{obj}` | `{value}` | {n} | {expected} | {enum} | {grand} | {sum_phi} | {error} |".format(
                model=run_label.get(str(row["model"]), str(row["model"])),
                obj=row["object"],
                value=row["value_function"],
                n=int(row["n_views"]),
                expected=int(row["expected_coalitions"]),
                enum=int(row["enumerated_coalitions"]),
                grand=fmt(row["grand_coalition_value"], 6),
                sum_phi=fmt(row["sum_shapley"], 6),
                error=fmt(row["efficiency_error"], 10),
            )
        )

    lines.extend(
        [
            "",
            "## Thesis-Level Shapley Summary",
            "",
            "Main metric: exact Shapley value for strict target quality.",
            "",
            "| Model | Object | View | Single-view quality | Exact Shapley value | Rank |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(exact_shapley_rows, key=lambda item: (str(item["model"]), str(item["object"]), int(item["view_id"]))):
        lines.append(
            "| {model} | `{obj}` | {view} | {single} | {shapley} | {rank} |".format(
                model=run_label.get(str(row["model"]), str(row["model"])),
                obj=row["object"],
                view=int(row["view_id"]),
                single=fmt(row["single_view_strict_quality"]),
                shapley=fmt(row["shapley_strict_quality"]),
                rank=int(row["rank_shapley_strict"]),
            )
        )

    lines.extend(
        [
            "",
            "## Highest Strict-Quality Shapley Views",
            "",
            "| Model | Object | Highest-ranked view(s) | Exact Shapley value |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for (model, object_id), rows in sorted(shapley_by_model_object.items()):
        best_rank = min(int(row["rank_shapley_strict"]) for row in rows)
        best_rows = [row for row in rows if int(row["rank_shapley_strict"]) == best_rank]
        view_text = ", ".join(f"v{int(row['view_id'])} `{row['filename']}`" for row in best_rows)
        best_value = max(float(row["shapley_strict_quality"]) for row in best_rows)
        lines.append(
            f"| {run_label.get(model, model)} | `{object_id}` | {view_text} | {fmt(best_value)} |"
        )

    lines.extend(
        [
            "",
            "## Fine-Tuned Versus Synthetic-Only Comparison",
            "",
            "| Model | Macro grand detection | Macro grand strict quality | Macro grand Noisy-OR x best IoU | Near-zero strict-Shapley views |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, rows in (("real_uav_finetuned", real_rows), ("synthetic_m4", synthetic_rows)):
        lines.append(
            "| {model} | {det} | {strict} | {fusion} | {zero} / {total} |".format(
                model=run_label.get(model, model),
                det=fmt(macro_grand(model, "detection")),
                strict=fmt(macro_grand(model, "strict_quality")),
                fusion=fmt(macro_grand(model, "noisy_or_best_iou")),
                zero=near_zero_count(rows, "shapley_strict_quality"),
                total=len(rows),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if real_k1 and real_k2 and real_k3:
        lines.extend(
            [
                f"- Adding a second real UAV view improves macro detection by `{fmt(real_k2['marginal_gain_vs_k_minus_1_detection_at_conf25_iou50'])}`, strict quality by `{fmt(real_k2['marginal_gain_vs_k_minus_1_best_strict_quality_iou50'])}`, and Noisy-OR x best IoU by `{fmt(real_k2['marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50'])}`.",
                f"- Adding a third real UAV view gives smaller gains: detection `{fmt(real_k3['marginal_gain_vs_k_minus_1_detection_at_conf25_iou50'])}`, strict quality `{fmt(real_k3['marginal_gain_vs_k_minus_1_best_strict_quality_iou50'])}`, and Noisy-OR x best IoU `{fmt(real_k3['marginal_gain_vs_k_minus_1_noisy_or_best_iou_iou50'])}`.",
            ]
        )
    lines.extend(
        [
            "- Viewpoints do not contribute equally: exact strict-quality Shapley values vary by object, with some views carrying most of the grand-coalition quality and null/weak views contributing approximately zero.",
            f"- The descriptive Pearson correlation between single-view strict quality and exact strict-quality Shapley contribution for the real fine-tuned model is `{corr_text}` across {figure_stats.get('real_finetuned_scatter_n_views', 0)} views. This is descriptive only; the independent experimental unit is the physical object instance, not each view or coalition.",
            "- Fine-tuning changes both the absolute detection quality and the usefulness of multi-view attribution. Synthetic-only Shapley attribution is limited when the detector fails under the synthetic-to-real domain gap.",
            "- These results should not be framed as statistically significant population estimates, because there are only five independent physical object instances.",
            "",
            "## Suggested Results Paragraph",
            "",
            "Across five real UAV object instances with multiple same-object viewpoints, the fine-tuned detector showed a clear multi-view benefit in the coalition-size analysis. Macro-averaged detection was already high with one view, but adding a second view increased both detection and target quality, while the third view produced smaller additional gains, indicating diminishing returns. Exact all-view Shapley attribution showed that the contribution of individual viewpoints was uneven: for several objects, a small number of views accounted for most of the strict target-quality value, while weak or failed views contributed little or nothing. The synthetic-only model produced much lower grand-coalition quality and many near-zero viewpoint contributions, so its Shapley values are mainly evidence of limited synthetic-to-real transfer rather than reliable viewpoint usefulness. Overall, the experiment supports the thesis distinction between synthetic controlled viewpoint analysis, real-world fine-tuning for transfer, and a limited but measurable additional benefit from combining real UAV viewpoints. Because the independent sample consists of only five physical objects, these findings should be interpreted as focused case-study evidence rather than broad statistical proof.",
            "",
            "## Output Files",
            "",
            "- `exact_shapley_all_views.csv`: exact standard Shapley values for every usable view of every object and model.",
            "- `exact_shapley_validation.csv`: Shapley efficiency and coalition-count validation table.",
            "- `exact_coalition_values_all_sizes.csv`: raw value of every coalition, including the empty and grand coalitions.",
            "- `shapley_marginal_by_predecessor_size.csv`: mean marginal contributions by predecessor coalition size; this explains Shapley values but is not a substitute for them.",
            "- `coalition_scores_k1_to_k3.csv`: retained 1-, 2-, and 3-view coalitions for Analysis A.",
            "- `overall_coalition_size_summary.csv`: macro-averaged 1 -> 2 -> 3 progression.",
            "- `object_coalition_size_summary.csv`: per-object 1 -> 2 -> 3 progression.",
            "- `class_coalition_size_summary.csv`: class-level 1 -> 2 -> 3 progression.",
            "- `exact_shapley_strict_by_view_real_finetuned.png`: requested Figure 1.",
            "- `single_view_vs_exact_shapley_strict_real_finetuned.png`: requested Figure 2.",
            "- `real_multiview_progression.png`: retained progression figure.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_restricted_shapley_markdown(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    run_label = {
        "real_uav_finetuned": "real UAV fine-tuned",
        "synthetic_m4": "synthetic M4 only",
    }
    lines = [
        "# 1/2/3-View Per-View Mean Marginal Contributions",
        "",
        "For each view, this table averages its marginal contribution over every same-object coalition of size 1, 2, or 3 that contains that view:",
        "",
        "`value(coalition) - value(coalition without this view)`",
        "",
        "This is a diagnostic mean marginal table for small coalitions only. It is not the exact Shapley result; use `exact_shapley_all_views.csv` for standard Shapley attribution.",
        "",
    ]

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_name"]), str(row["object_id"]))].append(row)

    for (run_name, object_id), members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: int(row["view_index"]))
        first = members[0]
        lines.extend(
            [
                f"## {run_label.get(run_name, run_name)} / `{object_id}`",
                "",
                f"Target class: `{first['target_class']}`. Usable views: {len(members)}.",
                "",
                "| View | File | Coalitions | Mean marginal detection | Mean marginal strict quality | Mean marginal Noisy-OR x best IoU | Single-view strict quality |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in members:
            lines.append(
                "| {view} | `{file}` | {coalitions} | {det} | {strict} | {fusion} | {single_strict} |".format(
                    view=int(row["view_index"]),
                    file=row["file_name"],
                    coalitions=int(row["coalition_count_k1_to_k3"]),
                    det=fmt(row["mean_marginal_k1_to_k3_detection_at_conf25_iou50"]),
                    strict=fmt(row["mean_marginal_k1_to_k3_best_strict_quality_iou50"]),
                    fusion=fmt(row["mean_marginal_k1_to_k3_noisy_or_best_iou_iou50"]),
                    single_strict=fmt(row["single_view_strict_quality_iou50"]),
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    manifest_rows, issue_rows, image_paths_by_key = build_manifest(args.uav_dir, args.gt_label_dir)
    write_csv(args.output_dir / "real_multiview_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "manifest_issues.csv", issue_rows)

    if issue_rows:
        print(f"Warning: manifest has {len(issue_rows)} missing image/label issues. See manifest_issues.csv.")

    usable_image_paths = sorted(
        {Path(str(row["image_path"])) for row in manifest_rows if int(row["image_exists"]) and int(row["label_exists"])},
        key=lambda path: image_key(path),
    )

    all_view_scores: list[ViewScore] = []
    per_view_csv = args.output_dir / "per_view_target_scores.csv"

    if args.skip_predict:
        all_view_scores = [score_from_row(row) for row in read_csv(per_view_csv)]
    else:
        label_by_key = {
            str(row["image_key"]): Path(str(row["label_path"]))
            for row in manifest_rows
            if str(row.get("label_path", ""))
        }

        gt_by_key_and_class: dict[tuple[str, str], GroundTruthTarget | None] = {}
        for row in manifest_rows:
            key = str(row["image_key"])
            target_class = str(row["target_class"])
            if (key, target_class) not in gt_by_key_and_class:
                gt_by_key_and_class[(key, target_class)] = read_target_gt(
                    label_by_key.get(key),
                    Path(str(row["image_path"])),
                    target_class,
                )

        for run_name in args.runs:
            predictions_by_key = run_predictions(
                run_name=run_name,
                weights=DEFAULT_RUNS[run_name],
                image_paths=usable_image_paths,
                predict_conf_threshold=args.predict_conf_threshold,
                imgsz=args.imgsz,
            )
            for row in manifest_rows:
                if not int(row["image_exists"]) or not int(row["label_exists"]):
                    continue
                key = str(row["image_key"])
                target_class = str(row["target_class"])
                all_view_scores.append(
                    score_view(
                        run_name=run_name,
                        manifest_row=row,
                        gt_target=gt_by_key_and_class[(key, target_class)],
                        predictions=predictions_by_key.get(key, []),
                        iou_threshold=args.iou_threshold,
                        detection_conf_threshold=args.detection_conf_threshold,
                    )
                )

        write_csv(per_view_csv, [view_score_to_row(score) for score in all_view_scores])

    coalition_rows = build_coalition_rows(all_view_scores, max_k=3)
    object_rows = summarize_object_coalitions(coalition_rows)
    overall_rows = summarize_overall(object_rows)
    class_rows = summarize_by_class(object_rows)
    shapley_rows, shapley_size_rows, shapley_validation_rows, exact_coalition_rows = exact_shapley_analysis(all_view_scores)
    restricted_shapley_rows = restricted_k_view_contribution_rows(all_view_scores, max_k=3)
    figure_stats = plot_exact_shapley_figures(shapley_rows, args.output_dir)

    write_csv(args.output_dir / "coalition_scores_k1_to_k3.csv", coalition_rows)
    write_csv(args.output_dir / "object_coalition_size_summary.csv", object_rows)
    write_csv(args.output_dir / "overall_coalition_size_summary.csv", overall_rows)
    write_csv(args.output_dir / "class_coalition_size_summary.csv", class_rows)
    write_csv(args.output_dir / "exact_shapley_all_views.csv", shapley_rows)
    write_csv(args.output_dir / "exact_shapley_validation.csv", shapley_validation_rows)
    write_csv(args.output_dir / "exact_coalition_values_all_sizes.csv", exact_coalition_rows)
    write_csv(args.output_dir / "shapley_marginal_by_predecessor_size.csv", shapley_size_rows)
    write_csv(args.output_dir / "shapley_view_contributions.csv", shapley_rows)
    write_csv(args.output_dir / "shapley_size_conditioned_marginals.csv", shapley_size_rows)
    write_csv(args.output_dir / "shapley_k1_to_k3_view_contributions.csv", restricted_shapley_rows)
    build_restricted_shapley_markdown(
        restricted_shapley_rows,
        args.output_dir / "shapley_k1_to_k3_view_contributions.md",
    )
    plot_progression(overall_rows, args.output_dir / "real_multiview_progression.png")
    build_revised_report(
        output_path=args.output_dir / "real_multiview_shapley_report.md",
        manifest_rows=manifest_rows,
        issue_rows=issue_rows,
        view_scores=all_view_scores,
        object_rows=object_rows,
        overall_rows=overall_rows,
        class_rows=class_rows,
        exact_shapley_rows=shapley_rows,
        validation_rows=shapley_validation_rows,
        figure_stats=figure_stats,
        args=args,
    )

    print(json.dumps(overall_rows, indent=2))
    print(f"Wrote real multiview Shapley outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
