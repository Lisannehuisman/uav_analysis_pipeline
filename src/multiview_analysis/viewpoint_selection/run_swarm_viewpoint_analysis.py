from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_RECORDS = (
    WORKSPACE / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
)
DEFAULT_VIEWPOINT_MANIFEST = (
    WORKSPACE / "viewpoint_data_separated" / "72_trained_models" / "manifests" / "viewpoint_inventory.csv"
)
DEFAULT_DATASET_AUDIT_DIR = WORKSPACE / "outputs" / "thesis_tools" / "dataset_structure_audit"
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_viewpoint_selection_analysis" / "outputs" / "integrated"

ELEVATION_SORT = {"low": 0, "mid": 1, "high": 2}
RADIUS_SORT = {"near": 0, "mid": 1, "far": 2}
AZIMUTH_VALUES = [0, 45, 90, 135, 180, 225, 270, 315]
OBJECT_CLASS_COUNT = 10


@dataclass(frozen=True)
class SceneRecord:
    scene_key: str
    file_name: str
    image_id: int
    target_class: str
    viewpoint: str
    elevation: str
    radius: str
    azimuth: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float
    tp: int
    fp: int
    fn: int
    target_visible: int
    target_detected: int
    target_precision: float
    target_recall: float
    target_f1: float
    target_ap50: float
    target_ap50_95: float
    target_match_confidence_iou50: float
    target_strict_quality_iou50: float
    target_tp: int
    target_fp: int
    target_fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated fixed-detector M4 viewpoint analysis for single-view, "
            "multi-view relationships, synergy matrices, practical subset selection, "
            "and 1/2/3-drone budget comparison."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--viewpoint-manifest", default=str(DEFAULT_VIEWPOINT_MANIFEST))
    parser.add_argument("--dataset-audit-dir", default=str(DEFAULT_DATASET_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-k", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--primary-metric", default="mean_ap50_95", choices=["mean_ap50_95", "mean_strict_quality"])
    return parser.parse_args()


def parse_float(raw: object) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return 0.0
    return float(text)


def parse_int(raw: object) -> int:
    text = str(raw).strip()
    if not text:
        return 0
    return int(float(text))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=0))


def minmax(value: float, low: float, high: float, neutral: float = 0.5) -> float:
    if high <= low:
        return neutral
    return max(0.0, min(1.0, (value - low) / (high - low)))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strip_prefix(token: str, prefix: str) -> str:
    return token[len(prefix) :] if token.startswith(prefix) else token


def viewpoint_parts(viewpoint: str) -> tuple[str, str, int]:
    parts = viewpoint.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unexpected viewpoint token: {viewpoint}")
    elevation = strip_prefix(parts[0], "el")
    radius = strip_prefix(parts[1], "rad")
    azimuth = int(parts[2].replace("az", ""))
    return elevation, radius, azimuth


def viewpoint_sort_key(viewpoint: str) -> tuple[int, int, int]:
    elevation, radius, azimuth = viewpoint_parts(viewpoint)
    return (ELEVATION_SORT.get(elevation, 99), RADIUS_SORT.get(radius, 99), azimuth)


def combo_label(viewpoints: Iterable[str]) -> str:
    return " + ".join(sorted(viewpoints, key=viewpoint_sort_key))


def azimuth_gap(first: int, second: int) -> int:
    delta = abs(first - second) % 360
    return min(delta, 360 - delta)


def max_pairwise_azimuth_gap(azimuths: list[int]) -> int:
    if len(azimuths) < 2:
        return 0
    return max(azimuth_gap(a, b) for a, b in combinations(azimuths, 2))


def mean_pairwise_azimuth_gap(azimuths: list[int]) -> float:
    if len(azimuths) < 2:
        return 0.0
    return mean(azimuth_gap(a, b) for a, b in combinations(azimuths, 2))


def ensure_scene_records(path: Path) -> None:
    if not path.is_file():
        gt_path = WORKSPACE / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json"
        pred_path = (
            WORKSPACE
            / "outputs"
            / "detector_family_comparison"
            / "standardized_test_eval"
            / "predictions"
            / "YOLOv8l_M4_test_predictions.json"
        )
        missing = [str(item) for item in [path, gt_path, pred_path] if not item.is_file()]
        raise FileNotFoundError(
            "Required cached scene records are missing. "
            "This integrated script expects the existing fixed-detector cache. Missing: "
            + "; ".join(missing)
        )


def read_scene_records(path: Path) -> list[SceneRecord]:
    ensure_scene_records(path)
    rows: list[SceneRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scene_key",
            "file_name",
            "image_id",
            "target_class",
            "viewpoint",
            "elevation",
            "radius",
            "azimuth",
            "precision",
            "recall",
            "f1",
            "ap50",
            "ap50_95",
            "tp",
            "fp",
            "fn",
            "target_visible",
            "target_detected",
            "target_precision",
            "target_recall",
            "target_f1",
            "target_ap50",
            "target_ap50_95",
            "target_match_confidence_iou50",
            "target_strict_quality_iou50",
            "target_tp",
            "target_fp",
            "target_fn",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Scene records CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                SceneRecord(
                    scene_key=row["scene_key"],
                    file_name=row["file_name"],
                    image_id=parse_int(row["image_id"]),
                    target_class=row["target_class"],
                    viewpoint=row["viewpoint"],
                    elevation=row["elevation"],
                    radius=row["radius"],
                    azimuth=parse_int(row["azimuth"]),
                    precision=parse_float(row["precision"]),
                    recall=parse_float(row["recall"]),
                    f1=parse_float(row["f1"]),
                    ap50=parse_float(row["ap50"]),
                    ap50_95=parse_float(row["ap50_95"]),
                    tp=parse_int(row["tp"]),
                    fp=parse_int(row["fp"]),
                    fn=parse_int(row["fn"]),
                    target_visible=parse_int(row["target_visible"]),
                    target_detected=parse_int(row["target_detected"]),
                    target_precision=parse_float(row["target_precision"]),
                    target_recall=parse_float(row["target_recall"]),
                    target_f1=parse_float(row["target_f1"]),
                    target_ap50=parse_float(row["target_ap50"]),
                    target_ap50_95=parse_float(row["target_ap50_95"]),
                    target_match_confidence_iou50=parse_float(row["target_match_confidence_iou50"]),
                    target_strict_quality_iou50=parse_float(row["target_strict_quality_iou50"]),
                    target_tp=parse_int(row["target_tp"]),
                    target_fp=parse_int(row["target_fp"]),
                    target_fn=parse_int(row["target_fn"]),
                )
            )
    return rows


def read_viewpoint_manifest(path: Path) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return manifest
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            viewpoint = str(row.get("viewpoint", "")).strip()
            if not viewpoint:
                continue
            manifest[viewpoint] = {
                "manifest_train_images": parse_int(row.get("train_images", 0)),
                "manifest_val_images": parse_int(row.get("val_images", 0)),
                "manifest_test_images": parse_int(row.get("test_images", 0)),
            }
    return manifest


def build_input_inventory(scene_records_path: Path, manifest_path: Path, dataset_audit_dir: Path) -> list[dict[str, object]]:
    paths = [
        ("scene_view_records", scene_records_path),
        ("viewpoint_manifest", manifest_path),
        (
            "m4_test_ground_truth",
            WORKSPACE / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json",
        ),
        (
            "m4_test_predictions",
            WORKSPACE
            / "outputs"
            / "detector_family_comparison"
            / "standardized_test_eval"
            / "predictions"
            / "YOLOv8l_M4_test_predictions.json",
        ),
        ("dataset_split_summary", dataset_audit_dir / "split_summary.csv"),
        ("dataset_instance_overlap", dataset_audit_dir / "instance_split_overlap.csv"),
        (
            "existing_swarm_protocol_summary",
            WORKSPACE / "m4_two_drone_operational_analysis" / "thesis_swarm_outputs" / "protocol_overall_summary.csv",
        ),
        (
            "existing_box_fusion_pair_rows",
            WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "pair_combo_rows.csv",
        ),
    ]
    return [
        {
            "input_name": name,
            "path": str(path),
            "exists": int(path.is_file()),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        for name, path in paths
    ]


def build_scene_groups(records: list[SceneRecord]) -> dict[str, list[SceneRecord]]:
    groups: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in records:
        groups[record.scene_key].append(record)
    for key in groups:
        groups[key].sort(key=lambda record: viewpoint_sort_key(record.viewpoint))
    return groups


def summarize_records(records: list[SceneRecord]) -> dict[str, object]:
    tp = sum(record.tp for record in records)
    fp = sum(record.fp for record in records)
    fn = sum(record.fn for record in records)
    target_tp = sum(record.target_tp for record in records)
    target_fp = sum(record.target_fp for record in records)
    target_fn = sum(record.target_fn for record in records)
    absent = [record for record in records if record.target_visible == 0]
    return {
        "sample_count": len(records),
        "scene_count": len({record.scene_key for record in records}),
        "class_count": len({record.target_class for record in records}),
        "micro_precision": safe_divide(tp, tp + fp),
        "micro_recall": safe_divide(tp, tp + fn),
        "micro_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "target_micro_precision": safe_divide(target_tp, target_tp + target_fp),
        "target_micro_recall": safe_divide(target_tp, target_tp + target_fn),
        "target_micro_f1": safe_divide(2 * target_tp, 2 * target_tp + target_fp + target_fn),
        "mean_precision": mean(record.precision for record in records),
        "mean_recall": mean(record.recall for record in records),
        "mean_f1": mean(record.f1 for record in records),
        "mean_ap50": mean(record.ap50 for record in records),
        "mean_ap50_95": mean(record.ap50_95 for record in records),
        "mean_target_precision": mean(record.target_precision for record in records),
        "mean_target_recall": mean(record.target_recall for record in records),
        "mean_target_f1": mean(record.target_f1 for record in records),
        "mean_target_ap50": mean(record.target_ap50 for record in records),
        "mean_target_ap50_95": mean(record.target_ap50_95 for record in records),
        "mean_strict_quality": mean(record.target_strict_quality_iou50 for record in records),
        "mean_match_confidence_iou50": mean(record.target_match_confidence_iou50 for record in records),
        "target_visible_rate": mean(record.target_visible for record in records),
        "target_found_rate": mean(record.target_detected for record in records),
        "target_absent_view_count": len(absent),
        "false_alarm_count_when_target_absent": sum(1 for record in absent if record.target_fp > 0),
        "false_alarm_rate_when_target_absent": safe_divide(sum(1 for record in absent if record.target_fp > 0), len(absent)),
    }


def build_viewpoint_table(records: list[SceneRecord], manifest: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    viewpoints = sorted({record.viewpoint for record in records}.union(manifest), key=viewpoint_sort_key)
    record_groups: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in records:
        record_groups[record.viewpoint].append(record)

    rows: list[dict[str, object]] = []
    for index, viewpoint in enumerate(viewpoints, start=1):
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        row: dict[str, object] = {
            "viewpoint_index": index,
            "viewpoint": viewpoint,
            "elevation": elevation,
            "radius": radius,
            "azimuth": azimuth,
            "angle_degrees": azimuth,
            "height_level": elevation,
            "distance_level": radius,
            "observed_in_scene_records": int(viewpoint in record_groups),
            "observed_test_records": len(record_groups.get(viewpoint, [])),
            "observed_test_scenes": len({record.scene_key for record in record_groups.get(viewpoint, [])}),
        }
        row.update(manifest.get(viewpoint, {}))
        rows.append(row)
    return rows


def build_per_viewpoint_metrics(records: list[SceneRecord], manifest: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    groups: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in records:
        groups[record.viewpoint].append(record)
    rows: list[dict[str, object]] = []
    for viewpoint in sorted(groups, key=viewpoint_sort_key):
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        row: dict[str, object] = {
            "viewpoint": viewpoint,
            "elevation": elevation,
            "radius": radius,
            "azimuth": azimuth,
            **summarize_records(groups[viewpoint]),
        }
        row.update(manifest.get(viewpoint, {}))
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["mean_target_ap50_95"]), -float(row["mean_strict_quality"]), str(row["viewpoint"])))
    for rank, row in enumerate(rows, start=1):
        row["global_rank_by_target_ap50_95"] = rank
    rows.sort(key=lambda row: (-float(row["mean_strict_quality"]), -float(row["mean_target_ap50_95"]), str(row["viewpoint"])))
    for rank, row in enumerate(rows, start=1):
        row["global_rank_by_strict_quality"] = rank
    rows.sort(key=lambda row: viewpoint_sort_key(str(row["viewpoint"])))
    return rows


def build_per_class_viewpoint_metrics(records: list[SceneRecord]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[SceneRecord]] = defaultdict(list)
    for record in records:
        groups[(record.target_class, record.viewpoint)].append(record)

    rows: list[dict[str, object]] = []
    for (target_class, viewpoint), members in groups.items():
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        rows.append(
            {
                "target_class": target_class,
                "viewpoint": viewpoint,
                "elevation": elevation,
                "radius": radius,
                "azimuth": azimuth,
                **summarize_records(members),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["target_class"]),
            -float(row["mean_target_ap50_95"]),
            -float(row["mean_strict_quality"]),
            str(row["viewpoint"]),
        )
    )
    current_class = None
    rank = 0
    for row in rows:
        if row["target_class"] != current_class:
            current_class = row["target_class"]
            rank = 1
        else:
            rank += 1
        row["class_rank_by_target_ap50_95"] = rank
    rows.sort(key=lambda row: (str(row["target_class"]), viewpoint_sort_key(str(row["viewpoint"]))))
    return rows


def group_label(record: SceneRecord, factor: str) -> str:
    if factor == "elevation":
        return record.elevation
    if factor == "radius":
        return record.radius
    if factor == "azimuth":
        return f"{record.azimuth:03d}"
    if factor == "elevation_radius":
        return f"{record.elevation}|{record.radius}"
    if factor == "elevation_azimuth":
        return f"{record.elevation}|az{record.azimuth:03d}"
    if factor == "radius_azimuth":
        return f"{record.radius}|az{record.azimuth:03d}"
    if factor == "full_viewpoint":
        return record.viewpoint
    raise ValueError(f"Unknown factor: {factor}")


def build_grouped_analysis(records: list[SceneRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    factors = [
        "elevation",
        "radius",
        "azimuth",
        "elevation_radius",
        "elevation_azimuth",
        "radius_azimuth",
        "full_viewpoint",
    ]
    for factor in factors:
        groups: dict[str, list[SceneRecord]] = defaultdict(list)
        for record in records:
            groups[group_label(record, factor)].append(record)
        for value, members in groups.items():
            rows.append({"factor": factor, "factor_value": value, **summarize_records(members)})
    rows.sort(key=lambda row: (str(row["factor"]), -float(row["mean_target_ap50_95"]), str(row["factor_value"])))
    return rows


def eta_squared(records: list[SceneRecord], factor: str, metric_name: str) -> dict[str, object]:
    values = np.array([getattr(record, metric_name) for record in records], dtype=float)
    total_mean = float(values.mean()) if len(values) else 0.0
    total_ss = float(((values - total_mean) ** 2).sum())
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[group_label(record, factor)].append(float(getattr(record, metric_name)))
    between_ss = 0.0
    for group_values in groups.values():
        group_array = np.array(group_values, dtype=float)
        between_ss += len(group_array) * float((group_array.mean() - total_mean) ** 2)
    return {
        "factor": factor,
        "metric": metric_name,
        "level_count": len(groups),
        "sample_count": len(records),
        "eta_squared": safe_divide(between_ss, total_ss),
        "between_group_ss": between_ss,
        "total_ss": total_ss,
    }


def build_factor_explanation(records: list[SceneRecord]) -> list[dict[str, object]]:
    factors = [
        "elevation",
        "radius",
        "azimuth",
        "elevation_radius",
        "elevation_azimuth",
        "radius_azimuth",
        "full_viewpoint",
    ]
    metrics = ["target_ap50_95", "target_strict_quality_iou50", "target_ap50", "target_f1"]
    rows = [eta_squared(records, factor, metric) for metric in metrics for factor in factors]
    metric_order = {metric: index for index, metric in enumerate(metrics)}
    rows.sort(key=lambda row: (metric_order.get(str(row["metric"]), 99), -float(row["eta_squared"])))
    return rows


def relationship_taxonomy(viewpoints: tuple[str, ...]) -> dict[str, object]:
    elevations: list[str] = []
    radii: list[str] = []
    azimuths: list[int] = []
    for viewpoint in viewpoints:
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        elevations.append(elevation)
        radii.append(radius)
        azimuths.append(azimuth)

    unique_elevations = sorted(set(elevations), key=lambda value: ELEVATION_SORT[value])
    unique_radii = sorted(set(radii), key=lambda value: RADIUS_SORT[value])
    unique_azimuths = sorted(set(azimuths))
    max_gap = max_pairwise_azimuth_gap(azimuths)
    mean_gap = mean_pairwise_azimuth_gap(azimuths)

    if len(unique_radii) == 1:
        radius_relationship = "same_radius"
    elif {"near", "far"}.issubset(unique_radii):
        radius_relationship = "near_far"
    else:
        radius_relationship = "adjacent_radius"

    if len(unique_elevations) == 1:
        elevation_relationship = "same_elevation"
    elif {"low", "high"}.issubset(unique_elevations):
        elevation_relationship = "low_high"
    else:
        elevation_relationship = "adjacent_elevation"

    if len(viewpoints) == 2:
        if max_gap == 0:
            azimuth_relationship = "same_azimuth"
        elif max_gap == 45:
            azimuth_relationship = "adjacent_45"
        elif max_gap == 90:
            azimuth_relationship = "quarter_turn_90"
        elif max_gap == 135:
            azimuth_relationship = "diagonal_135"
        elif max_gap == 180:
            azimuth_relationship = "opposite_180"
        else:
            azimuth_relationship = f"gap_{max_gap}"
    else:
        if max_gap <= 90:
            azimuth_relationship = "compact_azimuths"
        elif max_gap <= 135:
            azimuth_relationship = "medium_spread_azimuths"
        else:
            azimuth_relationship = "broad_or_opposite_azimuths"

    diversity_factors = []
    if len(unique_radii) > 1:
        diversity_factors.append("distance")
    if len(unique_elevations) > 1:
        diversity_factors.append("elevation")
    if len(unique_azimuths) > 1:
        diversity_factors.append("azimuth")

    if not diversity_factors:
        diversity_type = "no_viewpoint_diversity"
    elif len(diversity_factors) == 1:
        diversity_type = f"{diversity_factors[0]}_only"
    elif len(diversity_factors) == 2:
        diversity_type = "+".join(diversity_factors)
    else:
        diversity_type = "distance+elevation+azimuth"

    return {
        "elevation_pattern": " + ".join(unique_elevations),
        "radius_pattern": " + ".join(unique_radii),
        "azimuth_pattern": " + ".join(f"{value:03d}" for value in unique_azimuths),
        "unique_elevation_count": len(unique_elevations),
        "unique_radius_count": len(unique_radii),
        "unique_azimuth_count": len(unique_azimuths),
        "max_pairwise_azimuth_gap": max_gap,
        "mean_pairwise_azimuth_gap": mean_gap,
        "radius_relationship": radius_relationship,
        "elevation_relationship": elevation_relationship,
        "azimuth_relationship": azimuth_relationship,
        "diversity_type": diversity_type,
        "diversity_factor_count": len(diversity_factors),
    }


def combo_observation(combo: tuple[SceneRecord, ...]) -> dict[str, object]:
    viewpoints = tuple(record.viewpoint for record in combo)
    target_tp = sum(record.target_tp for record in combo)
    target_fp = sum(record.target_fp for record in combo)
    target_fn = sum(record.target_fn for record in combo)
    overall_tp = sum(record.tp for record in combo)
    overall_fp = sum(record.fp for record in combo)
    overall_fn = sum(record.fn for record in combo)
    visible_count = sum(record.target_visible for record in combo)
    detected_count = sum(record.target_detected for record in combo)
    absent_views = [record for record in combo if record.target_visible == 0]

    return {
        "scene_key": combo[0].scene_key,
        "target_class": combo[0].target_class,
        "drone_count": len(combo),
        "combination_label": combo_label(viewpoints),
        "viewpoint_1": viewpoints[0],
        "viewpoint_2": viewpoints[1] if len(viewpoints) >= 2 else "",
        "viewpoint_3": viewpoints[2] if len(viewpoints) >= 3 else "",
        **relationship_taxonomy(tuple(sorted(viewpoints, key=viewpoint_sort_key))),
        "precision": safe_divide(target_tp, target_tp + target_fp),
        "recall": safe_divide(target_tp, target_tp + target_fn),
        "f1": safe_divide(2 * target_tp, 2 * target_tp + target_fp + target_fn),
        "overall_precision": safe_divide(overall_tp, overall_tp + overall_fp),
        "overall_recall": safe_divide(overall_tp, overall_tp + overall_fn),
        "overall_f1": safe_divide(2 * overall_tp, 2 * overall_tp + overall_fp + overall_fn),
        "ap50": max(record.target_ap50 for record in combo),
        "ap50_95": max(record.target_ap50_95 for record in combo),
        "mean_ap50": mean(record.target_ap50 for record in combo),
        "mean_ap50_95": mean(record.target_ap50_95 for record in combo),
        "best_strict_quality": max(record.target_strict_quality_iou50 for record in combo),
        "mean_strict_quality": mean(record.target_strict_quality_iou50 for record in combo),
        "best_match_confidence_iou50": max(record.target_match_confidence_iou50 for record in combo),
        "target_visible_any": int(visible_count > 0),
        "target_visible_all": int(visible_count == len(combo)),
        "target_found_or": int(detected_count > 0),
        "target_found_all": int(detected_count == len(combo)),
        "target_found_majority": int(detected_count >= math.ceil(len(combo) / 2)),
        "absent_view_count": len(absent_views),
        "selected_has_absent_view": int(bool(absent_views)),
        "all_selected_views_absent": int(visible_count == 0),
        "false_alarm_any_absent_view": int(any(record.target_fp > 0 for record in absent_views)),
        "target_tp_sum": target_tp,
        "target_fp_sum": target_fp,
        "target_fn_sum": target_fn,
    }


def aggregate_combo_observations(observations: list[dict[str, object]]) -> dict[str, object]:
    first = observations[0]
    class_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        class_groups[str(row["target_class"])].append(row)

    class_means = [mean(float(row["best_strict_quality"]) for row in rows) for rows in class_groups.values()]
    class_cv = safe_divide(std(class_means), mean(class_means)) if class_means else 0.0
    class_robustness = min(1.0, len(class_means) / OBJECT_CLASS_COUNT) * max(0.0, 1.0 - min(class_cv, 1.0))

    return {
        "drone_count": first["drone_count"],
        "combination_label": first["combination_label"],
        "viewpoint_1": first["viewpoint_1"],
        "viewpoint_2": first["viewpoint_2"],
        "viewpoint_3": first["viewpoint_3"],
        "scene_count": len({row["scene_key"] for row in observations}),
        "sample_count": len(observations),
        "class_count": len(class_groups),
        "dominant_target_class": Counter(str(row["target_class"]) for row in observations).most_common(1)[0][0],
        "dominant_target_class_count": Counter(str(row["target_class"]) for row in observations).most_common(1)[0][1],
        "precision": mean(float(row["precision"]) for row in observations),
        "recall": mean(float(row["recall"]) for row in observations),
        "F1": mean(float(row["f1"]) for row in observations),
        "AP50": mean(float(row["ap50"]) for row in observations),
        "AP50-95": mean(float(row["ap50_95"]) for row in observations),
        "mean_ap50": mean(float(row["mean_ap50"]) for row in observations),
        "mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in observations),
        "mean_best_strict_quality": mean(float(row["best_strict_quality"]) for row in observations),
        "mean_mean_strict_quality": mean(float(row["mean_strict_quality"]) for row in observations),
        "mean_best_match_confidence_iou50": mean(float(row["best_match_confidence_iou50"]) for row in observations),
        "target_visible_any_rate": mean(int(row["target_visible_any"]) for row in observations),
        "target_visible_all_rate": mean(int(row["target_visible_all"]) for row in observations),
        "target_found_or_rate": mean(int(row["target_found_or"]) for row in observations),
        "target_found_majority_rate": mean(int(row["target_found_majority"]) for row in observations),
        "target_found_all_rate": mean(int(row["target_found_all"]) for row in observations),
        "mean_absent_view_count": mean(int(row["absent_view_count"]) for row in observations),
        "selected_has_absent_view_rate": mean(int(row["selected_has_absent_view"]) for row in observations),
        "all_selected_views_absent_rate": mean(int(row["all_selected_views_absent"]) for row in observations),
        "false_alarm_rate_when_any_selected_view_absent": safe_divide(
            sum(int(row["false_alarm_any_absent_view"]) for row in observations),
            sum(int(row["selected_has_absent_view"]) for row in observations),
        ),
        "class_mean_strict_quality_min": min(class_means) if class_means else 0.0,
        "class_mean_strict_quality_std": std(class_means),
        "class_mean_strict_quality_cv": class_cv,
        "class_robustness_score": class_robustness,
        **{key: first[key] for key in relationship_taxonomy(tuple(str(first[f"viewpoint_{i}"]) for i in range(1, int(first["drone_count"]) + 1))).keys()},
    }


def build_combo_summaries(records: list[SceneRecord], max_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scene_groups = build_scene_groups(records)
    by_combo: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)

    for scene_records in scene_groups.values():
        for drone_count in range(1, max_k + 1):
            if len(scene_records) < drone_count:
                continue
            for combo in combinations(scene_records, drone_count):
                key = tuple(sorted((record.viewpoint for record in combo), key=viewpoint_sort_key))
                by_combo[key].append(combo_observation(combo))

    summary_rows = [aggregate_combo_observations(rows) for rows in by_combo.values()]
    relationship_rows = [row for row in summary_rows if int(row["drone_count"]) >= 2]
    summary_rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["AP50-95"]),
            -float(row["mean_best_strict_quality"]),
            str(row["combination_label"]),
        )
    )
    relationship_rows.sort(key=lambda row: (int(row["drone_count"]), str(row["combination_label"])))
    return summary_rows, relationship_rows


def add_synergy(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in summary_rows:
        drone_count = int(row["drone_count"])
        if drone_count == 1:
            row["synergy_strict_vs_best_individual"] = 0.0
            row["synergy_ap50_95_vs_best_individual"] = 0.0
            continue

        # Recompute matched-scene constituent means from the existing observation-level
        # definition would require preserving every observation. For exact combinations,
        # the conservative approximation below compares best-available combo quality with
        # the mean constituent quality stored in the combo row. It is used for weighted
        # recommendations only; the pairwise matrix below computes exact matched-scene
        # synergy from scene records.
        row["synergy_strict_vs_best_individual"] = float(row["mean_best_strict_quality"]) - float(row["mean_mean_strict_quality"])
        row["synergy_ap50_95_vs_best_individual"] = float(row["AP50-95"]) - float(row["mean_ap50_95"])
    return summary_rows


def build_relationship_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    axes = [
        ("radius_relationship", "distance"),
        ("elevation_relationship", "elevation"),
        ("azimuth_relationship", "azimuth"),
        ("diversity_type", "mixed_diversity"),
    ]
    for drone_count in [2, 3]:
        members = [row for row in rows if int(row["drone_count"]) == drone_count]
        for field, axis in axes:
            groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in members:
                groups[str(row[field])].append(row)
            for value, group_rows in groups.items():
                ap_values = [float(row["AP50-95"]) for row in group_rows]
                strict_values = [float(row["mean_best_strict_quality"]) for row in group_rows]
                out.append(
                    {
                        "drone_count": drone_count,
                        "relationship_axis": axis,
                        "relationship_type": value,
                        "configuration_count": len(group_rows),
                        "mean_scene_count": mean(int(row["scene_count"]) for row in group_rows),
                        "median_scene_count": median(int(row["scene_count"]) for row in group_rows),
                        "mean_AP50": mean(float(row["AP50"]) for row in group_rows),
                        "mean_AP50_95": mean(ap_values),
                        "median_AP50_95": float(median(ap_values)),
                        "mean_precision": mean(float(row["precision"]) for row in group_rows),
                        "mean_recall": mean(float(row["recall"]) for row in group_rows),
                        "mean_F1": mean(float(row["F1"]) for row in group_rows),
                        "mean_strict_quality": mean(strict_values),
                        "median_strict_quality": float(median(strict_values)),
                        "mean_synergy_strict_vs_best_individual": mean(
                            float(row["synergy_strict_vs_best_individual"]) for row in group_rows
                        ),
                        "mean_class_robustness_score": mean(float(row["class_robustness_score"]) for row in group_rows),
                    }
                )
    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["relationship_axis"]),
            -float(row["mean_AP50_95"]),
        )
    )
    return out


def build_pairwise_synergy(
    records: list[SceneRecord],
    viewpoints: list[str],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    scene_groups = build_scene_groups(records)
    single_metric_by_viewpoint = {
        viewpoint: mean(record.target_strict_quality_iou50 for record in records if record.viewpoint == viewpoint)
        for viewpoint in viewpoints
    }
    matrix = np.full((len(viewpoints), len(viewpoints)), np.nan, dtype=float)
    performance_matrix = np.full((len(viewpoints), len(viewpoints)), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    index_by_viewpoint = {viewpoint: index for index, viewpoint in enumerate(viewpoints)}

    for viewpoint in viewpoints:
        index = index_by_viewpoint[viewpoint]
        matrix[index, index] = 0.0
        performance_matrix[index, index] = single_metric_by_viewpoint.get(viewpoint, 0.0)

    for first, second in combinations(viewpoints, 2):
        strict_first: list[float] = []
        strict_second: list[float] = []
        best_values: list[float] = []
        ap_best_values: list[float] = []
        support_scenes = 0
        for scene_records in scene_groups.values():
            by_viewpoint = {record.viewpoint: record for record in scene_records}
            if first not in by_viewpoint or second not in by_viewpoint:
                continue
            support_scenes += 1
            first_record = by_viewpoint[first]
            second_record = by_viewpoint[second]
            strict_first.append(first_record.target_strict_quality_iou50)
            strict_second.append(second_record.target_strict_quality_iou50)
            best_values.append(max(first_record.target_strict_quality_iou50, second_record.target_strict_quality_iou50))
            ap_best_values.append(max(first_record.target_ap50_95, second_record.target_ap50_95))

        if not best_values:
            continue
        pair_best = mean(best_values)
        best_individual_on_matched = max(mean(strict_first), mean(strict_second))
        synergy = pair_best - best_individual_on_matched
        ap_pair_best = mean(ap_best_values)
        i = index_by_viewpoint[first]
        j = index_by_viewpoint[second]
        matrix[i, j] = synergy
        matrix[j, i] = synergy
        performance_matrix[i, j] = pair_best
        performance_matrix[j, i] = pair_best
        rows.append(
            {
                "viewpoint_1": first,
                "viewpoint_2": second,
                "combination_label": combo_label((first, second)),
                "scene_count": support_scenes,
                "mean_best_pair_strict_quality": pair_best,
                "best_individual_mean_strict_quality_on_matched_scenes": best_individual_on_matched,
                "pair_synergy_strict": synergy,
                "mean_best_pair_AP50_95": ap_pair_best,
                **relationship_taxonomy(tuple(sorted((first, second), key=viewpoint_sort_key))),
            }
        )

    rows.sort(key=lambda row: (-float(row["pair_synergy_strict"]), -float(row["mean_best_pair_strict_quality"])))
    return rows, matrix, performance_matrix


def build_complementarity_groups(
    viewpoints: list[str],
    per_viewpoint_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    single_by_viewpoint = {str(row["viewpoint"]): float(row["mean_strict_quality"]) for row in per_viewpoint_rows}
    pair_members: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        pair_members[str(row["viewpoint_1"])].append(row)
        pair_members[str(row["viewpoint_2"])].append(row)

    single_values = [single_by_viewpoint.get(viewpoint, 0.0) for viewpoint in viewpoints]
    avg_synergy_values = [
        mean(float(row["pair_synergy_strict"]) for row in pair_members.get(viewpoint, [])) for viewpoint in viewpoints
    ]
    single_high = float(np.percentile(single_values, 67)) if single_values else 0.0
    synergy_high = float(np.percentile(avg_synergy_values, 67)) if avg_synergy_values else 0.0
    synergy_low = float(np.percentile(avg_synergy_values, 33)) if avg_synergy_values else 0.0

    rows: list[dict[str, object]] = []
    for viewpoint in viewpoints:
        pairs = pair_members.get(viewpoint, [])
        avg_synergy = mean(float(row["pair_synergy_strict"]) for row in pairs)
        positive_fraction = safe_divide(sum(1 for row in pairs if float(row["pair_synergy_strict"]) > 0), len(pairs))
        single_score = single_by_viewpoint.get(viewpoint, 0.0)
        if single_score >= single_high and avg_synergy >= synergy_high:
            group = "anchor_and_complementary"
        elif single_score >= single_high:
            group = "strong_standalone_anchor"
        elif avg_synergy >= synergy_high:
            group = "complementary_support_view"
        elif avg_synergy <= synergy_low or positive_fraction < 0.25:
            group = "redundant_or_low_gain"
        else:
            group = "coverage_specific"

        rows.append(
            {
                "viewpoint": viewpoint,
                "single_mean_strict_quality": single_score,
                "average_pair_synergy_strict": avg_synergy,
                "median_pair_synergy_strict": float(median([float(row["pair_synergy_strict"]) for row in pairs])) if pairs else 0.0,
                "positive_synergy_pair_fraction": positive_fraction,
                "best_partner": pairs[0]["viewpoint_2"] if pairs and pairs[0]["viewpoint_1"] == viewpoint else (pairs[0]["viewpoint_1"] if pairs else ""),
                "best_pair_synergy_strict": max((float(row["pair_synergy_strict"]) for row in pairs), default=0.0),
                "complementarity_group": group,
            }
        )
    rows.sort(key=lambda row: (-float(row["single_mean_strict_quality"]), -float(row["average_pair_synergy_strict"])))
    return rows


def deployability_score(row: dict[str, object]) -> float:
    drone_count = int(row["drone_count"])
    unique_elev = int(row.get("unique_elevation_count", 1))
    unique_radius = int(row.get("unique_radius_count", 1))
    max_gap = float(row.get("max_pairwise_azimuth_gap", 0.0))
    score = 1.0
    score -= 0.08 * max(0, unique_elev - 1)
    score -= 0.08 * max(0, unique_radius - 1)
    if drone_count >= 2 and max_gap > 135:
        score -= 0.06
    if drone_count == 3:
        score -= 0.04
    return max(0.0, min(1.0, score))


def add_weighted_scores(summary_rows: list[dict[str, object]], min_support: int) -> list[dict[str, object]]:
    scored_rows = [dict(row) for row in summary_rows if int(row["scene_count"]) >= min_support]
    by_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        by_k[int(row["drone_count"])].append(row)

    for drone_count, rows in by_k.items():
        raw_values = [float(row["AP50-95"]) for row in rows]
        comp_values = [max(0.0, float(row["synergy_strict_vs_best_individual"])) for row in rows]
        support_values = [float(row["scene_count"]) for row in rows]
        class_values = [float(row["class_robustness_score"]) for row in rows]
        for row in rows:
            raw_score = minmax(float(row["AP50-95"]), min(raw_values), max(raw_values))
            complementarity_score = 0.5 if drone_count == 1 else minmax(
                max(0.0, float(row["synergy_strict_vs_best_individual"])),
                min(comp_values),
                max(comp_values),
            )
            class_score = minmax(float(row["class_robustness_score"]), min(class_values), max(class_values))
            scene_score = minmax(float(row["scene_count"]), min(support_values), max(support_values))
            if float(row["selected_has_absent_view_rate"]) > 0:
                absence_score = 1.0 - float(row["false_alarm_rate_when_any_selected_view_absent"])
            else:
                absence_score = 0.75
            simple_score = deployability_score(row)
            weighted = (
                0.35 * raw_score
                + 0.20 * complementarity_score
                + 0.15 * class_score
                + 0.15 * scene_score
                + 0.10 * absence_score
                + 0.05 * simple_score
            )
            row.update(
                {
                    "raw_detection_score_norm": raw_score,
                    "complementarity_score_norm": complementarity_score,
                    "class_robustness_score_norm": class_score,
                    "scene_support_score_norm": scene_score,
                    "absence_safety_score": absence_score,
                    "deployability_score": simple_score,
                    "weighted_selection_score": weighted,
                    "weight_definition": "0.35 raw + 0.20 complementarity + 0.15 class robustness + 0.15 scene support + 0.10 absence safety + 0.05 deployability",
                }
            )
    scored_rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["weighted_selection_score"]),
            -float(row["AP50-95"]),
            str(row["combination_label"]),
        )
    )
    rank_by_k: Counter[int] = Counter()
    for row in scored_rows:
        drone_count = int(row["drone_count"])
        rank_by_k[drone_count] += 1
        row["rank_within_drone_budget"] = rank_by_k[drone_count]
    return scored_rows


def build_budget_comparison(scored_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    best_single_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "F1": 0.0,
        "AP50": 0.0,
        "AP50-95": 0.0,
    }
    previous_best_metrics = dict(best_single_metrics)
    for drone_count in [1, 2, 3]:
        candidates = [row for row in scored_rows if int(row["drone_count"]) == drone_count]
        all_configs = [row for row in summary_rows if int(row["drone_count"]) == drone_count]
        if not candidates or not all_configs:
            continue
        best = candidates[0]
        current_metrics = {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "F1": float(best["F1"]),
            "AP50": float(best["AP50"]),
            "AP50-95": float(best["AP50-95"]),
        }
        if drone_count == 1:
            best_single_metrics = dict(current_metrics)
            previous_best_metrics = dict(current_metrics)
        row = {
            "drone_count": drone_count,
            "best_subset_label": best["combination_label"],
            "best_subset_scene_count": best["scene_count"],
            "best_subset_weighted_score": best["weighted_selection_score"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_F1": best["F1"],
            "best_AP50": best["AP50"],
            "best_AP50_95": best["AP50-95"],
            "best_gain_precision_vs_best_single": current_metrics["precision"] - best_single_metrics["precision"],
            "best_gain_recall_vs_best_single": current_metrics["recall"] - best_single_metrics["recall"],
            "best_gain_F1_vs_best_single": current_metrics["F1"] - best_single_metrics["F1"],
            "best_gain_AP50_vs_best_single": current_metrics["AP50"] - best_single_metrics["AP50"],
            "best_gain_AP50_95_vs_best_single": current_metrics["AP50-95"] - best_single_metrics["AP50-95"],
            "best_marginal_gain_precision_vs_previous_budget": 0.0
            if drone_count == 1
            else current_metrics["precision"] - previous_best_metrics["precision"],
            "best_marginal_gain_recall_vs_previous_budget": 0.0
            if drone_count == 1
            else current_metrics["recall"] - previous_best_metrics["recall"],
            "best_marginal_gain_F1_vs_previous_budget": 0.0
            if drone_count == 1
            else current_metrics["F1"] - previous_best_metrics["F1"],
            "best_marginal_gain_AP50_vs_previous_budget": 0.0
            if drone_count == 1
            else current_metrics["AP50"] - previous_best_metrics["AP50"],
            "best_marginal_gain_AP50_95_vs_previous_budget": 0.0
            if drone_count == 1
            else current_metrics["AP50-95"] - previous_best_metrics["AP50-95"],
            "average_precision_over_all_configurations": mean(float(item["precision"]) for item in all_configs),
            "average_recall_over_all_configurations": mean(float(item["recall"]) for item in all_configs),
            "average_F1_over_all_configurations": mean(float(item["F1"]) for item in all_configs),
            "average_AP50_over_all_configurations": mean(float(item["AP50"]) for item in all_configs),
            "average_AP50_95_over_all_configurations": mean(float(item["AP50-95"]) for item in all_configs),
            "configuration_count": len(all_configs),
        }
        previous_best_metrics = dict(current_metrics)
        rows.append(row)
    return rows


def build_relationship_metadata_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metadata_fields = [
        "drone_count",
        "combination_label",
        "viewpoint_1",
        "viewpoint_2",
        "viewpoint_3",
        "scene_count",
        "sample_count",
        "class_count",
        "elevation_pattern",
        "radius_pattern",
        "azimuth_pattern",
        "unique_elevation_count",
        "unique_radius_count",
        "unique_azimuth_count",
        "max_pairwise_azimuth_gap",
        "mean_pairwise_azimuth_gap",
        "radius_relationship",
        "elevation_relationship",
        "azimuth_relationship",
        "diversity_type",
        "diversity_factor_count",
    ]
    return [{field: row.get(field, "") for field in metadata_fields} for row in rows]


def write_matrix_csv(path: Path, viewpoints: list[str], matrix: np.ndarray) -> None:
    rows: list[dict[str, object]] = []
    for i, viewpoint in enumerate(viewpoints):
        row: dict[str, object] = {"viewpoint": viewpoint}
        for j, other in enumerate(viewpoints):
            value = matrix[i, j]
            row[other] = "" if np.isnan(value) else f"{value:.6f}"
        rows.append(row)
    write_csv(path, rows, fieldnames=["viewpoint", *viewpoints])


def plot_viewpoint_heatmap(rows: list[dict[str, object]], metric: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_labels = [f"{elevation}-{radius}" for elevation in ["low", "mid", "high"] for radius in ["near", "mid", "far"]]
    values_by_key = {
        (str(row["elevation"]), str(row["radius"]), int(row["azimuth"])): float(row[metric])
        for row in rows
    }
    matrix = np.full((len(row_labels), len(AZIMUTH_VALUES)), np.nan, dtype=float)
    for i, label in enumerate(row_labels):
        elevation, radius = label.split("-")
        for j, azimuth in enumerate(AZIMUTH_VALUES):
            matrix[i, j] = values_by_key.get((elevation, radius, azimuth), np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(AZIMUTH_VALUES)))
    ax.set_xticklabels([f"{az:03d}" for az in AZIMUTH_VALUES])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation-radius")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=metric)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_grouped_bars(grouped_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors = ["elevation", "radius", "azimuth"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, factor in zip(axes, factors):
        rows = [row for row in grouped_rows if row["factor"] == factor]
        rows.sort(key=lambda row: str(row["factor_value"]))
        labels = [str(row["factor_value"]) for row in rows]
        values = [float(row["mean_target_ap50_95"]) for row in rows]
        ax.bar(labels, values, color="#3b6ea8")
        ax.set_title(factor)
        ax.set_ylabel("Mean target AP50-95")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_relationship_boxplots(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axes_info = [
        ("radius_relationship", "Distance"),
        ("elevation_relationship", "Elevation"),
        ("azimuth_relationship", "Azimuth"),
        ("diversity_type", "Mixed diversity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (field, title) in zip(axes.flatten(), axes_info):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(float(row["AP50-95"]))
        labels = sorted(groups, key=lambda key: -mean(groups[key]))
        data = [groups[label] for label in labels]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("Configuration mean AP50-95")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_synergy_heatmap(matrix: np.ndarray, viewpoints: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", aspect="auto")
    tick_step = 8
    ticks = list(range(0, len(viewpoints), tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([viewpoints[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([viewpoints[i] for i in ticks], fontsize=6)
    ax.set_title("Pairwise synergy matrix: pair best strict quality minus best individual on matched scenes")
    ax.set_xlabel("Second viewpoint")
    ax.set_ylabel("First viewpoint")
    fig.colorbar(im, ax=ax, label="Strict-quality synergy")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_budget_curves(rows: list[dict[str, object]], gain_path: Path, budget_path: Path) -> None:
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    x = [int(row["drone_count"]) for row in rows]
    best = [float(row["best_AP50_95"]) for row in rows]
    avg = [float(row["average_AP50_95_over_all_configurations"]) for row in rows]
    gains = [float(row["best_marginal_gain_AP50_95_vs_previous_budget"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar([str(item) for item in x], gains, color="#d08c3f")
    ax.set_xlabel("Drone/viewpoint budget")
    ax.set_ylabel("Marginal AP50-95 gain")
    ax.set_title("Marginal gain from adding viewpoints")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(gain_path, dpi=300, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(x, best, marker="o", linewidth=2.2, label="Best supported subset")
    ax.plot(x, avg, marker="o", linewidth=2.2, label="Average configuration")
    ax.set_xticks(x)
    ax.set_xlabel("Drone/viewpoint budget")
    ax.set_ylabel("AP50-95")
    ax.set_title("Budget vs performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(budget_path, dpi=300, facecolor="white")
    plt.close(fig)


def write_interpretation(
    path: Path,
    records: list[SceneRecord],
    viewpoint_rows: list[dict[str, object]],
    factor_rows: list[dict[str, object]],
    relationship_summary: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    scored_rows: list[dict[str, object]],
    budget_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_viewpoints = sorted(viewpoint_rows, key=lambda row: float(row["mean_target_ap50_95"]), reverse=True)[:5]
    top_pair = sorted(pair_rows, key=lambda row: float(row["pair_synergy_strict"]), reverse=True)[0] if pair_rows else None
    best_relationships = relationship_summary[:8]
    best_by_budget = {int(row["drone_count"]): row for row in budget_rows}
    best_scored = {
        k: [row for row in scored_rows if int(row["drone_count"]) == k][:5]
        for k in [1, 2, 3]
    }

    lines = [
        "# Integrated Viewpoint Swarm Analysis",
        "",
        "## Viewpoint Representation",
        "",
        "Viewpoints are represented as filename/cache tokens of the form `el{elevation}-rad{radius}-az{azimuth}`.",
        "",
        "- elevation: `low`, `mid`, `high`",
        "- radius/distance: `near`, `mid`, `far`",
        "- azimuth: `000`, `045`, `090`, `135`, `180`, `225`, `270`, `315` degrees",
        "- full space: `3 x 3 x 8 = 72` possible viewpoints",
        "",
        "## Cached Evidence",
        "",
        f"- scene/view rows: `{len(records)}`",
        f"- scenes: `{len({record.scene_key for record in records})}`",
        f"- observed viewpoints: `{len({record.viewpoint for record in records})}`",
        f"- target-absent rows: `{sum(1 for record in records if record.target_visible == 0)}`",
        "",
        "## Strongest Single Viewpoints",
        "",
    ]
    for row in top_viewpoints:
        lines.append(
            "- "
            f"`{row['viewpoint']}`: target AP50-95 `{float(row['mean_target_ap50_95']):.4f}`, "
            f"strict quality `{float(row['mean_strict_quality']):.4f}`, support `{row['scene_count']}` scenes"
        )

    lines.extend(["", "## Factor Explanation", ""])
    factor_focus = [
        row
        for row in factor_rows
        if row["metric"] in {"target_ap50_95", "target_strict_quality_iou50"}
    ][:8]
    for row in factor_focus:
        lines.append(
            "- "
            f"`{row['metric']}` explained by `{row['factor']}`: eta-squared `{float(row['eta_squared']):.4f}`"
        )

    lines.extend(["", "## Relationship Types", ""])
    for row in best_relationships:
        lines.append(
            "- "
            f"k=`{row['drone_count']}`, `{row['relationship_axis']}={row['relationship_type']}`: "
            f"mean AP50-95 `{float(row['mean_AP50_95']):.4f}`, "
            f"mean strict quality `{float(row['mean_strict_quality']):.4f}`, "
            f"configs `{row['configuration_count']}`"
        )

    lines.extend(["", "## Explicit Synergy Definition", ""])
    lines.append(
        "For a pair `(i, j)`, synergy is computed on matched scenes where both viewpoints exist: "
        "`mean(max(strict_i, strict_j)) - max(mean(strict_i), mean(strict_j))`. "
        "Positive values mean the pair adds complementary evidence beyond its better single viewpoint."
    )
    if top_pair:
        lines.append(
            f"Most complementary observed pair: `{top_pair['combination_label']}` with synergy "
            f"`{float(top_pair['pair_synergy_strict']):.4f}` over `{top_pair['scene_count']}` matched scenes."
        )

    lines.extend(["", "## Practical Recommendations", ""])
    lines.append(
        "The scripted weighted score uses: 0.35 raw detection, 0.20 complementarity, "
        "0.15 class robustness, 0.15 scene support, 0.10 absence safety, and 0.05 deployability."
    )
    for k in [1, 2, 3]:
        lines.append(f"### Budget k={k}")
        for row in best_scored[k]:
            lines.append(
                "- "
                f"`{row['combination_label']}`: weighted score `{float(row['weighted_selection_score']):.4f}`, "
                f"AP50-95 `{float(row['AP50-95']):.4f}`, support `{row['scene_count']}` scenes"
            )
        lines.append("")

    lines.extend(["## Budget Comparison", ""])
    for k in [1, 2, 3]:
        row = best_by_budget.get(k)
        if not row:
            continue
        lines.append(
            "- "
            f"k=`{k}` best supported subset `{row['best_subset_label']}`: "
            f"AP50-95 `{float(row['best_AP50_95']):.4f}`, "
            f"gain vs best single `{float(row['best_gain_AP50_95_vs_best_single']):+.4f}`, "
            f"marginal gain `{float(row['best_marginal_gain_AP50_95_vs_previous_budget']):+.4f}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is a fixed-detector, cached-prediction analysis. Pair/triple performance uses best-available target evidence across selected views, not calibrated 3D fusion or cross-view object identity tracking. Exact pair/triple subsets can be sparse because not every scene contains every viewpoint.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checklist(path: Path, scored_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Practical Viewpoint Selection Checklist",
        "",
        "Use this checklist when choosing a drone-swarm viewpoint set from the cached M4 evidence.",
        "",
        "1. Start with a high raw detection viewpoint: high `AP50-95` and high strict target quality.",
        "2. Require enough scene support; exact pairs/triples with only a few matched scenes are exploratory.",
        "3. Add viewpoints only when pair synergy is positive under the explicit matched-scene definition.",
        "4. Prefer subsets that improve class robustness rather than only improving one dominant class.",
        "5. Check target-absent false alarms; sparse absence rows mean this is a safety audit, not a full negative benchmark.",
        "6. Keep deployment simple when scores are close: fewer changes in elevation/radius are easier to fly.",
        "",
        "## Scripted Ranking Rule",
        "",
        "`weighted_selection_score = 0.35 raw_detection + 0.20 complementarity + 0.15 class_robustness + 0.15 scene_support + 0.10 absence_safety + 0.05 deployability`",
        "",
        "Top recommendations by budget:",
        "",
    ]
    for k in [1, 2, 3]:
        lines.append(f"## {k} Viewpoint(s)")
        for row in [row for row in scored_rows if int(row["drone_count"]) == k][:10]:
            lines.append(
                "- "
                f"`{row['combination_label']}`: score `{float(row['weighted_selection_score']):.4f}`, "
                f"AP50-95 `{float(row['AP50-95']):.4f}`, "
                f"synergy `{float(row['synergy_strict_vs_best_individual']):.4f}`, "
                f"class robustness `{float(row['class_robustness_score']):.4f}`, "
                f"support `{row['scene_count']}` scenes"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_records_path = Path(args.scene_records)
    manifest_path = Path(args.viewpoint_manifest)
    dataset_audit_dir = Path(args.dataset_audit_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    records = read_scene_records(scene_records_path)
    manifest = read_viewpoint_manifest(manifest_path)
    inventory_rows = build_input_inventory(scene_records_path, manifest_path, dataset_audit_dir)

    viewpoint_table = build_viewpoint_table(records, manifest)
    per_viewpoint = build_per_viewpoint_metrics(records, manifest)
    per_class_viewpoint = build_per_class_viewpoint_metrics(records)
    grouped_rows = build_grouped_analysis(records)
    factor_rows = build_factor_explanation(records)

    combo_summary, pair_triple_rows = build_combo_summaries(records, max_k=args.max_k)
    combo_summary = add_synergy(combo_summary)
    pair_triple_rows = add_synergy(pair_triple_rows)
    relationship_summary = build_relationship_summary(pair_triple_rows)

    viewpoints = [str(row["viewpoint"]) for row in sorted(viewpoint_table, key=lambda row: viewpoint_sort_key(str(row["viewpoint"])))]
    pair_synergy_rows, synergy_matrix, pair_performance_matrix = build_pairwise_synergy(records, viewpoints)
    complementarity_groups = build_complementarity_groups(viewpoints, per_viewpoint, pair_synergy_rows)

    scored_rows = add_weighted_scores(combo_summary, min_support=args.min_support)
    budget_rows = build_budget_comparison(scored_rows, combo_summary)

    top_pairs = sorted(
        [row for row in pair_triple_rows if int(row["drone_count"]) == 2 and int(row["scene_count"]) >= args.min_support],
        key=lambda row: (float(row["AP50-95"]), float(row["mean_best_strict_quality"])),
        reverse=True,
    )[: args.top_n]
    top_complementary = [
        row for row in pair_synergy_rows if int(row["scene_count"]) >= args.min_support
    ][: args.top_n]
    top_redundant = sorted(
        [row for row in pair_synergy_rows if int(row["scene_count"]) >= args.min_support],
        key=lambda row: (float(row["pair_synergy_strict"]), -float(row["mean_best_pair_strict_quality"])),
    )[: args.top_n]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "analysis_input_inventory.csv", inventory_rows)
    write_csv(output_dir / "viewpoint_table.csv", viewpoint_table)
    write_csv(output_dir / "per_viewpoint_metrics.csv", per_viewpoint)
    write_csv(output_dir / "per_class_viewpoint_metrics.csv", per_class_viewpoint)
    write_csv(output_dir / "viewpoint_grouped_analysis.csv", grouped_rows)
    write_csv(output_dir / "viewpoint_factor_explanation.csv", factor_rows)
    write_csv(output_dir / "pair_triple_relationship_metadata.csv", build_relationship_metadata_rows(pair_triple_rows))
    write_csv(output_dir / "pair_triple_performance.csv", pair_triple_rows)
    write_csv(output_dir / "relationship_type_performance_summary.csv", relationship_summary)
    write_csv(output_dir / "single_view_performance_vector.csv", per_viewpoint)
    write_matrix_csv(output_dir / "pairwise_synergy_matrix.csv", viewpoints, synergy_matrix)
    write_matrix_csv(output_dir / "pairwise_performance_matrix.csv", viewpoints, pair_performance_matrix)
    write_csv(output_dir / "pairwise_synergy_long.csv", pair_synergy_rows)
    write_csv(output_dir / "top_k_best_pairs.csv", top_pairs)
    write_csv(output_dir / "top_k_most_complementary_pairs.csv", top_complementary)
    write_csv(output_dir / "top_k_redundant_pairs.csv", top_redundant)
    write_csv(output_dir / "viewpoint_complementarity_groups.csv", complementarity_groups)
    write_csv(output_dir / "recommended_viewpoint_subsets.csv", scored_rows)
    write_csv(output_dir / "swarm_budget_comparison.csv", budget_rows)

    plot_viewpoint_heatmap(
        per_viewpoint,
        "mean_target_ap50_95",
        plots_dir / "viewpoint_heatmap_target_ap50_95.png",
        "Single-view target AP50-95 by viewpoint",
    )
    plot_viewpoint_heatmap(
        per_viewpoint,
        "mean_strict_quality",
        plots_dir / "viewpoint_heatmap_strict_quality.png",
        "Single-view strict target quality by viewpoint",
    )
    plot_grouped_bars(grouped_rows, plots_dir / "grouped_elevation_radius_azimuth.png")
    plot_relationship_boxplots(pair_triple_rows, plots_dir / "relationship_type_boxplots.png")
    plot_synergy_heatmap(synergy_matrix, viewpoints, plots_dir / "pairwise_synergy_heatmap.png")
    plot_budget_curves(
        budget_rows,
        plots_dir / "marginal_gain_by_budget.png",
        plots_dir / "budget_vs_performance.png",
    )

    write_interpretation(
        output_dir / "VIEWPOINT_SWARM_ANALYSIS_SUMMARY.md",
        records=records,
        viewpoint_rows=per_viewpoint,
        factor_rows=factor_rows,
        relationship_summary=relationship_summary,
        pair_rows=pair_synergy_rows,
        scored_rows=scored_rows,
        budget_rows=budget_rows,
    )
    write_checklist(output_dir / "PRACTICAL_VIEWPOINT_SELECTION_CHECKLIST.md", scored_rows)

    print(f"Wrote integrated swarm viewpoint analysis to {output_dir}")
    print(f"Viewpoints: {len(viewpoint_table)}")
    print(f"Pair/triple configurations: {len(pair_triple_rows)}")
    print(f"Recommended subset rows: {len(scored_rows)}")


if __name__ == "__main__":
    main()
