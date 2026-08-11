from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_two_drone_operational import (
    DEFAULT_GT,
    DEFAULT_PRED,
    ELEVATION_SORT,
    RADIUS_SORT,
    ViewRecord,
    azimuth_gap,
    build_scene_groups,
    build_view_records,
    mean,
    plot_top_barh,
    safe_divide,
    viewpoint_sort_key,
    write_csv_rows,
)


DEFAULT_OUTPUT_DIR = Path("m4_two_drone_operational_analysis") / "thesis_swarm_outputs"


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    drone_count: int
    threshold: int
    label: str
    short_label: str


PROTOCOLS = [
    Protocol("n1_any1", 1, 1, "1 drone, 1-of-1 success", "1-of-1"),
    Protocol("n2_any1", 2, 1, "2 drones, 1-of-2 OR", "1-of-2"),
    Protocol("n2_all2", 2, 2, "2 drones, 2-of-2 confirmation", "2-of-2"),
    Protocol("n3_any1", 3, 1, "3 drones, 1-of-3 OR", "1-of-3"),
    Protocol("n3_majority2", 3, 2, "3 drones, 2-of-3 confirmation", "2-of-3"),
    Protocol("n3_all3", 3, 3, "3 drones, 3-of-3 unanimous", "3-of-3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thesis-style operational swarm analysis for 1, 2, and 3 drones using the fixed M4 YOLOv8l detector outputs."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT), help="COCO ground-truth JSON for the M4 test split.")
    parser.add_argument("--pred-json", default=str(DEFAULT_PRED), help="COCO prediction JSON for the fixed full-M4 detector.")
    parser.add_argument("--score-threshold", type=float, default=0.001, help="Prediction score threshold applied before per-image matching.")
    parser.add_argument("--min-pair-support", type=int, default=8, help="Minimum sample count for headline pair rankings.")
    parser.add_argument("--min-triple-support", type=int, default=6, help="Minimum sample count for headline triple rankings.")
    parser.add_argument("--min-rescue-support", type=int, default=10, help="Minimum failure-support count for rescue summaries.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for the swarm analysis.")
    return parser.parse_args()


def protocol_map() -> dict[str, Protocol]:
    return {row.protocol_id: row for row in PROTOCOLS}


def combo_label(combo: tuple[ViewRecord, ...]) -> str:
    return " + ".join(record.viewpoint for record in combo)


def max_pairwise_azimuth_gap(values: list[int]) -> int:
    if len(values) <= 1:
        return 0
    best = 0
    for first, second in combinations(values, 2):
        best = max(best, azimuth_gap(first, second))
    return best


def metric_or_zero(value: float) -> float:
    return 0.0 if math.isnan(float(value)) else float(value)


def kth_largest(values: list[float], rank: int) -> float:
    ordered = sorted((metric_or_zero(value) for value in values), reverse=True)
    if rank <= 0 or rank > len(ordered):
        return 0.0
    return float(ordered[rank - 1])


def normalize_pattern_label(raw_pattern: str, order_map: dict[str, int]) -> str:
    tokens = [token for token in str(raw_pattern).split("|") if token]
    tokens = sorted(tokens, key=lambda token: (order_map.get(token, 99), token))
    return " + ".join(tokens)


def combo_metrics(combo: tuple[ViewRecord, ...]) -> dict[str, object]:
    viewpoints = [record.viewpoint for record in combo]
    elevations = [record.elevation for record in combo]
    radii = [record.radius for record in combo]
    azimuths = [record.azimuth for record in combo]
    target_detected_count = sum(int(record.target_detected) for record in combo)
    target_visible_count = sum(int(record.target_visible) for record in combo)
    tp_sum = sum(record.tp for record in combo)
    fp_sum = sum(record.fp for record in combo)
    fn_sum = sum(record.fn for record in combo)
    target_ap_values = [record.target_ap50_95 for record in combo]
    target_quality_values = [record.target_strict_quality_iou50 for record in combo]
    target_confidence_values = [record.target_match_confidence_iou50 for record in combo]
    target_match_iou_values = [record.target_match_iou_at_confidence_iou50 for record in combo]
    target_f1_values = [record.target_f1 for record in combo]
    target_precision_values = [record.target_precision for record in combo]
    target_recall_values = [record.target_recall for record in combo]

    return {
        "scene_key": combo[0].scene_key,
        "target_class": combo[0].target_class,
        "drone_count": len(combo),
        "combination_label": combo_label(combo),
        "viewpoint_1": viewpoints[0],
        "viewpoint_2": viewpoints[1] if len(viewpoints) >= 2 else "",
        "viewpoint_3": viewpoints[2] if len(viewpoints) >= 3 else "",
        "elevation_pattern": "|".join(elevations),
        "radius_pattern": "|".join(radii),
        "azimuth_pattern": "|".join(f"{value:03d}" for value in azimuths),
        "unique_elevation_count": len(set(elevations)),
        "unique_radius_count": len(set(radii)),
        "unique_azimuth_count": len(set(azimuths)),
        "max_pairwise_azimuth_gap": max_pairwise_azimuth_gap(azimuths),
        "mean_ap50_95": mean(record.ap50_95 for record in combo),
        "best_available_ap50_95": max(record.ap50_95 for record in combo),
        "mean_f1": mean(record.f1 for record in combo),
        "target_mean_precision": mean(target_precision_values),
        "target_mean_recall": mean(target_recall_values),
        "target_mean_f1": mean(target_f1_values),
        "target_mean_ap50_95": mean(target_ap_values),
        "target_best_ap50_95": kth_largest(target_ap_values, 1),
        "target_mean_match_confidence_iou50": mean(target_confidence_values),
        "target_best_match_confidence_iou50": kth_largest(target_confidence_values, 1),
        "target_mean_match_iou_at_confidence_iou50": mean(target_match_iou_values),
        "target_best_match_iou_at_confidence_iou50": kth_largest(target_match_iou_values, 1),
        "target_mean_strict_quality_iou50": mean(target_quality_values),
        "target_best_strict_quality_iou50": kth_largest(target_quality_values, 1),
        "target_ap50_95_rank_1": kth_largest(target_ap_values, 1),
        "target_ap50_95_rank_2": kth_largest(target_ap_values, 2),
        "target_ap50_95_rank_3": kth_largest(target_ap_values, 3),
        "target_match_confidence_iou50_rank_1": kth_largest(target_confidence_values, 1),
        "target_match_confidence_iou50_rank_2": kth_largest(target_confidence_values, 2),
        "target_match_confidence_iou50_rank_3": kth_largest(target_confidence_values, 3),
        "target_match_iou_at_confidence_iou50_rank_1": kth_largest(target_match_iou_values, 1),
        "target_match_iou_at_confidence_iou50_rank_2": kth_largest(target_match_iou_values, 2),
        "target_match_iou_at_confidence_iou50_rank_3": kth_largest(target_match_iou_values, 3),
        "target_strict_quality_iou50_rank_1": kth_largest(target_quality_values, 1),
        "target_strict_quality_iou50_rank_2": kth_largest(target_quality_values, 2),
        "target_strict_quality_iou50_rank_3": kth_largest(target_quality_values, 3),
        "precision": safe_divide(tp_sum, tp_sum + fp_sum),
        "recall": safe_divide(tp_sum, tp_sum + fn_sum),
        "f1": safe_divide(2 * tp_sum, 2 * tp_sum + fp_sum + fn_sum),
        "target_visible_any": int(target_visible_count >= 1),
        "target_detected_count": target_detected_count,
        "target_visible_count": target_visible_count,
    }


def build_combo_rows(scene_groups: dict[str, list[ViewRecord]], drone_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for members in scene_groups.values():
        if len(members) < drone_count:
            continue
        for combo in combinations(members, drone_count):
            rows.append(combo_metrics(combo))
    return rows


def add_protocol_flags(combo_rows: list[dict[str, object]], protocol: Protocol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in combo_rows:
        detected_count = int(row["target_detected_count"])
        visible_count = int(row["target_visible_count"])
        success = int(detected_count >= protocol.threshold)
        possible = int(visible_count >= protocol.threshold)
        enriched = dict(row)
        enriched.update(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_label": protocol.label,
                "threshold": protocol.threshold,
                "success": success,
                "possible": possible,
                "threshold_target_ap50_95": float(row.get(f"target_ap50_95_rank_{protocol.threshold}", 0.0)),
                "threshold_target_match_confidence_iou50": float(row.get(f"target_match_confidence_iou50_rank_{protocol.threshold}", 0.0)),
                "threshold_target_match_iou_at_confidence_iou50": float(row.get(f"target_match_iou_at_confidence_iou50_rank_{protocol.threshold}", 0.0)),
                "threshold_target_strict_quality_iou50": float(row.get(f"target_strict_quality_iou50_rank_{protocol.threshold}", 0.0)),
            }
        )
        rows.append(enriched)
    return rows


def summarize_protocol_scene_expectations(protocol_rows: list[dict[str, object]], protocol: Protocol) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in protocol_rows:
        grouped[str(row["scene_key"])].append(row)

    rows: list[dict[str, object]] = []
    for scene_key, members in sorted(grouped.items()):
        possible_count = sum(int(row["possible"]) for row in members)
        success_count = sum(int(row["success"]) for row in members)
        any_visible = sum(int(row["target_visible_any"]) for row in members)
        rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_label": protocol.label,
                "drone_count": protocol.drone_count,
                "threshold": protocol.threshold,
                "scene_key": scene_key,
                "target_class": members[0]["target_class"],
                "combination_count": len(members),
                "expected_target_visible_any_rate": safe_divide(any_visible, len(members)),
                "expected_target_visible_threshold_rate": safe_divide(possible_count, len(members)),
                "expected_target_found_rate": safe_divide(success_count, len(members)),
                "expected_target_found_given_threshold_visibility": safe_divide(success_count, possible_count),
                "expected_target_threshold_ap50_95": mean(float(row["threshold_target_ap50_95"]) for row in members),
                "expected_target_threshold_match_confidence_iou50": mean(float(row["threshold_target_match_confidence_iou50"]) for row in members),
                "expected_target_threshold_match_iou_at_confidence_iou50": mean(float(row["threshold_target_match_iou_at_confidence_iou50"]) for row in members),
                "expected_target_threshold_strict_quality_iou50": mean(float(row["threshold_target_strict_quality_iou50"]) for row in members),
                "expected_target_mean_ap50_95": mean(float(row["target_mean_ap50_95"]) for row in members),
                "expected_target_best_ap50_95": mean(float(row["target_best_ap50_95"]) for row in members),
                "expected_target_mean_strict_quality_iou50": mean(float(row["target_mean_strict_quality_iou50"]) for row in members),
                "expected_target_best_strict_quality_iou50": mean(float(row["target_best_strict_quality_iou50"]) for row in members),
                "expected_precision": mean(float(row["precision"]) for row in members),
                "expected_recall": mean(float(row["recall"]) for row in members),
                "expected_f1": mean(float(row["f1"]) for row in members),
                "expected_mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
                "expected_best_available_ap50_95": mean(float(row["best_available_ap50_95"]) for row in members),
            }
        )
    return rows


def summarize_protocol_overall(scene_rows: list[dict[str, object]]) -> dict[str, object]:
    first = scene_rows[0]
    return {
        "protocol_id": first["protocol_id"],
        "protocol_label": first["protocol_label"],
        "drone_count": first["drone_count"],
        "threshold": first["threshold"],
        "scene_count": len(scene_rows),
        "expected_target_visible_any_rate": mean(float(row["expected_target_visible_any_rate"]) for row in scene_rows),
        "expected_target_visible_threshold_rate": mean(float(row["expected_target_visible_threshold_rate"]) for row in scene_rows),
        "expected_target_found_rate": mean(float(row["expected_target_found_rate"]) for row in scene_rows),
        "expected_target_found_given_threshold_visibility": mean(float(row["expected_target_found_given_threshold_visibility"]) for row in scene_rows),
        "expected_target_threshold_ap50_95": mean(float(row["expected_target_threshold_ap50_95"]) for row in scene_rows),
        "expected_target_threshold_match_confidence_iou50": mean(float(row["expected_target_threshold_match_confidence_iou50"]) for row in scene_rows),
        "expected_target_threshold_match_iou_at_confidence_iou50": mean(float(row["expected_target_threshold_match_iou_at_confidence_iou50"]) for row in scene_rows),
        "expected_target_threshold_strict_quality_iou50": mean(float(row["expected_target_threshold_strict_quality_iou50"]) for row in scene_rows),
        "expected_target_mean_ap50_95": mean(float(row["expected_target_mean_ap50_95"]) for row in scene_rows),
        "expected_target_best_ap50_95": mean(float(row["expected_target_best_ap50_95"]) for row in scene_rows),
        "expected_target_mean_strict_quality_iou50": mean(float(row["expected_target_mean_strict_quality_iou50"]) for row in scene_rows),
        "expected_target_best_strict_quality_iou50": mean(float(row["expected_target_best_strict_quality_iou50"]) for row in scene_rows),
        "expected_precision": mean(float(row["expected_precision"]) for row in scene_rows),
        "expected_recall": mean(float(row["expected_recall"]) for row in scene_rows),
        "expected_f1": mean(float(row["expected_f1"]) for row in scene_rows),
        "expected_mean_ap50_95": mean(float(row["expected_mean_ap50_95"]) for row in scene_rows),
        "expected_best_available_ap50_95": mean(float(row["expected_best_available_ap50_95"]) for row in scene_rows),
    }


def summarize_protocol_by_class(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(str(row["protocol_id"]), str(row["target_class"]))].append(row)

    rows: list[dict[str, object]] = []
    for (protocol_id, class_name), members in sorted(grouped.items()):
        rows.append(
            {
                "protocol_id": protocol_id,
                "protocol_label": members[0]["protocol_label"],
                "drone_count": members[0]["drone_count"],
                "threshold": members[0]["threshold"],
                "target_class": class_name,
                "scene_count": len(members),
                "expected_target_found_rate": mean(float(row["expected_target_found_rate"]) for row in members),
                "expected_target_found_given_threshold_visibility": mean(float(row["expected_target_found_given_threshold_visibility"]) for row in members),
                "expected_target_threshold_ap50_95": mean(float(row["expected_target_threshold_ap50_95"]) for row in members),
                "expected_target_threshold_match_confidence_iou50": mean(float(row["expected_target_threshold_match_confidence_iou50"]) for row in members),
                "expected_target_threshold_strict_quality_iou50": mean(float(row["expected_target_threshold_strict_quality_iou50"]) for row in members),
                "expected_target_best_ap50_95": mean(float(row["expected_target_best_ap50_95"]) for row in members),
                "expected_target_best_strict_quality_iou50": mean(float(row["expected_target_best_strict_quality_iou50"]) for row in members),
                "expected_best_available_ap50_95": mean(float(row["expected_best_available_ap50_95"]) for row in members),
                "expected_mean_ap50_95": mean(float(row["expected_mean_ap50_95"]) for row in members),
            }
        )
    return rows


def summarize_exact_combinations(protocol_rows: list[dict[str, object]], protocol: Protocol) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in protocol_rows:
        grouped[str(row["combination_label"])].append(row)

    rows: list[dict[str, object]] = []
    for combination_key, members in sorted(grouped.items()):
        possible_count = sum(int(row["possible"]) for row in members)
        success_count = sum(int(row["success"]) for row in members)
        rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_label": protocol.label,
                "drone_count": protocol.drone_count,
                "threshold": protocol.threshold,
                "combination_label": combination_key,
                "viewpoint_1": members[0]["viewpoint_1"],
                "viewpoint_2": members[0]["viewpoint_2"],
                "viewpoint_3": members[0]["viewpoint_3"],
                "sample_count": len(members),
                "scene_count": len({str(row["scene_key"]) for row in members}),
                "mean_max_pairwise_azimuth_gap": mean(float(row["max_pairwise_azimuth_gap"]) for row in members),
                "mean_unique_elevation_count": mean(float(row["unique_elevation_count"]) for row in members),
                "mean_unique_radius_count": mean(float(row["unique_radius_count"]) for row in members),
                "expected_target_found_rate": safe_divide(success_count, len(members)),
                "expected_target_found_given_threshold_visibility": safe_divide(success_count, possible_count),
                "expected_threshold_visibility_rate": safe_divide(possible_count, len(members)),
                "expected_target_threshold_ap50_95": mean(float(row["threshold_target_ap50_95"]) for row in members),
                "expected_target_threshold_match_confidence_iou50": mean(float(row["threshold_target_match_confidence_iou50"]) for row in members),
                "expected_target_threshold_strict_quality_iou50": mean(float(row["threshold_target_strict_quality_iou50"]) for row in members),
                "expected_target_best_ap50_95": mean(float(row["target_best_ap50_95"]) for row in members),
                "expected_target_best_strict_quality_iou50": mean(float(row["target_best_strict_quality_iou50"]) for row in members),
                "expected_best_available_ap50_95": mean(float(row["best_available_ap50_95"]) for row in members),
                "expected_mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
                "expected_f1": mean(float(row["f1"]) for row in members),
            }
        )
    return rows


def summarize_triple_diversity(protocol_rows: list[dict[str, object]], protocol: Protocol) -> list[dict[str, object]]:
    if protocol.drone_count != 3:
        return []

    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in protocol_rows:
        key = (
            int(row["unique_elevation_count"]),
            int(row["unique_radius_count"]),
            int(row["max_pairwise_azimuth_gap"]),
        )
        grouped[key].append(row)

    rows: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        possible_count = sum(int(row["possible"]) for row in members)
        success_count = sum(int(row["success"]) for row in members)
        rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_label": protocol.label,
                "unique_elevation_count": key[0],
                "unique_radius_count": key[1],
                "max_pairwise_azimuth_gap": key[2],
                "sample_count": len(members),
                "expected_target_found_rate": safe_divide(success_count, len(members)),
                "expected_target_found_given_threshold_visibility": safe_divide(success_count, possible_count),
                "expected_threshold_visibility_rate": safe_divide(possible_count, len(members)),
                "expected_target_threshold_ap50_95": mean(float(row["threshold_target_ap50_95"]) for row in members),
                "expected_target_threshold_match_confidence_iou50": mean(float(row["threshold_target_match_confidence_iou50"]) for row in members),
                "expected_target_threshold_strict_quality_iou50": mean(float(row["threshold_target_strict_quality_iou50"]) for row in members),
                "expected_target_best_ap50_95": mean(float(row["target_best_ap50_95"]) for row in members),
                "expected_target_best_strict_quality_iou50": mean(float(row["target_best_strict_quality_iou50"]) for row in members),
                "expected_best_available_ap50_95": mean(float(row["best_available_ap50_95"]) for row in members),
                "expected_mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
            }
        )
    return rows


def summarize_pattern_members(protocol: Protocol, members: list[dict[str, object]]) -> dict[str, object]:
    possible_count = sum(int(row["possible"]) for row in members)
    success_count = sum(int(row["success"]) for row in members)
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_label": protocol.label,
        "drone_count": protocol.drone_count,
        "threshold": protocol.threshold,
        "sample_count": len(members),
        "scene_count": len({str(row["scene_key"]) for row in members}),
        "expected_target_found_rate": safe_divide(success_count, len(members)),
        "expected_target_found_given_threshold_visibility": safe_divide(success_count, possible_count),
        "expected_threshold_visibility_rate": safe_divide(possible_count, len(members)),
        "expected_target_threshold_ap50_95": mean(float(row["threshold_target_ap50_95"]) for row in members),
        "expected_target_threshold_match_confidence_iou50": mean(float(row["threshold_target_match_confidence_iou50"]) for row in members),
        "expected_target_threshold_strict_quality_iou50": mean(float(row["threshold_target_strict_quality_iou50"]) for row in members),
        "expected_target_best_ap50_95": mean(float(row["target_best_ap50_95"]) for row in members),
        "expected_target_best_strict_quality_iou50": mean(float(row["target_best_strict_quality_iou50"]) for row in members),
        "expected_best_available_ap50_95": mean(float(row["best_available_ap50_95"]) for row in members),
        "expected_mean_ap50_95": mean(float(row["mean_ap50_95"]) for row in members),
    }


def summarize_pair_angle_patterns(protocol_rows: list[dict[str, object]], protocol: Protocol) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if protocol.drone_count != 2:
        return [], [], [], []

    elevation_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    radius_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    gap_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    structure_groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)

    for row in protocol_rows:
        elevation_groups[normalize_pattern_label(str(row["elevation_pattern"]), ELEVATION_SORT)].append(row)
        radius_groups[normalize_pattern_label(str(row["radius_pattern"]), RADIUS_SORT)].append(row)
        gap_groups[int(row["max_pairwise_azimuth_gap"])].append(row)
        structure_groups[
            (
                int(row["unique_elevation_count"]),
                int(row["unique_radius_count"]),
                int(row["max_pairwise_azimuth_gap"]),
            )
        ].append(row)

    elevation_rows: list[dict[str, object]] = []
    for label, members in sorted(elevation_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["elevation_pattern"] = label
        elevation_rows.append(summary)

    radius_rows: list[dict[str, object]] = []
    for label, members in sorted(radius_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["radius_pattern"] = label
        radius_rows.append(summary)

    gap_rows: list[dict[str, object]] = []
    for gap, members in sorted(gap_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["azimuth_gap"] = gap
        gap_rows.append(summary)

    structure_rows: list[dict[str, object]] = []
    for key, members in sorted(structure_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["unique_elevation_count"] = key[0]
        summary["unique_radius_count"] = key[1]
        summary["max_pairwise_azimuth_gap"] = key[2]
        summary["structure_label"] = f"ue={key[0]}, ur={key[1]}, gap={key[2]}"
        structure_rows.append(summary)

    return elevation_rows, radius_rows, gap_rows, structure_rows


def summarize_triple_angle_patterns(protocol_rows: list[dict[str, object]], protocol: Protocol) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if protocol.drone_count != 3:
        return [], [], [], []

    elevation_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    radius_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    gap_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    structure_groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)

    for row in protocol_rows:
        elevation_groups[normalize_pattern_label(str(row["elevation_pattern"]), ELEVATION_SORT)].append(row)
        radius_groups[normalize_pattern_label(str(row["radius_pattern"]), RADIUS_SORT)].append(row)
        gap_groups[int(row["max_pairwise_azimuth_gap"])].append(row)
        structure_groups[
            (
                int(row["unique_elevation_count"]),
                int(row["unique_radius_count"]),
                int(row["max_pairwise_azimuth_gap"]),
            )
        ].append(row)

    elevation_rows: list[dict[str, object]] = []
    for label, members in sorted(elevation_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["elevation_pattern"] = label
        elevation_rows.append(summary)

    radius_rows: list[dict[str, object]] = []
    for label, members in sorted(radius_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["radius_pattern"] = label
        radius_rows.append(summary)

    gap_rows: list[dict[str, object]] = []
    for gap, members in sorted(gap_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["max_pairwise_azimuth_gap"] = gap
        gap_rows.append(summary)

    structure_rows: list[dict[str, object]] = []
    for key, members in sorted(structure_groups.items()):
        summary = summarize_pattern_members(protocol, members)
        summary["unique_elevation_count"] = key[0]
        summary["unique_radius_count"] = key[1]
        summary["max_pairwise_azimuth_gap"] = key[2]
        summary["structure_label"] = f"ue={key[0]}, ur={key[1]}, gap={key[2]}"
        structure_rows.append(summary)

    return elevation_rows, radius_rows, gap_rows, structure_rows


def build_data_readiness(records: list[ViewRecord], scene_groups: dict[str, list[ViewRecord]], gt_json: Path, pred_json: Path, score_threshold: float) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    scene_rows: list[dict[str, object]] = []
    for scene_key, members in sorted(scene_groups.items()):
        scene_rows.append(
            {
                "scene_key": scene_key,
                "target_class": members[0].target_class,
                "available_view_count": len(members),
                "supports_1_drone": int(len(members) >= 1),
                "supports_2_drones": int(len(members) >= 2),
                "supports_3_drones": int(len(members) >= 3),
                "target_visible_rate_across_views": mean(float(row.target_visible) for row in members),
                "target_detected_rate_across_views": mean(float(row.target_detected) for row in members),
                "target_mean_ap50_95_across_views": mean(float(row.target_ap50_95) for row in members),
                "target_mean_match_confidence_iou50_across_views": mean(float(row.target_match_confidence_iou50) for row in members),
                "target_mean_strict_quality_iou50_across_views": mean(float(row.target_strict_quality_iou50) for row in members),
            }
        )

    readiness_row = {
        "gt_json": str(gt_json),
        "pred_json": str(pred_json),
        "score_threshold": score_threshold,
        "image_count": len(records),
        "scene_count": len(scene_groups),
        "absolute_viewpoint_count": len({row.viewpoint for row in records}),
        "min_views_per_scene": min(len(members) for members in scene_groups.values()),
        "mean_views_per_scene": mean(len(members) for members in scene_groups.values()),
        "max_views_per_scene": max(len(members) for members in scene_groups.values()),
        "scenes_supporting_2_drones": sum(int(row["supports_2_drones"]) for row in scene_rows),
        "scenes_supporting_3_drones": sum(int(row["supports_3_drones"]) for row in scene_rows),
        "target_visible_rate_across_all_views": mean(float(row.target_visible) for row in records),
        "target_detected_rate_across_all_views": mean(float(row.target_detected) for row in records),
        "target_mean_ap50_95_across_all_views": mean(float(row.target_ap50_95) for row in records),
        "target_mean_match_confidence_iou50_across_all_views": mean(float(row.target_match_confidence_iou50) for row in records),
        "target_mean_strict_quality_iou50_across_all_views": mean(float(row.target_strict_quality_iou50) for row in records),
        "filename_target_absent_fraction": 1.0 - mean(float(row.target_visible) for row in records),
    }

    inventory_rows = [
        {"name": "gt_json", "path": str(gt_json), "exists": int(gt_json.exists()), "size_bytes": gt_json.stat().st_size if gt_json.exists() else 0},
        {"name": "pred_json", "path": str(pred_json), "exists": int(pred_json.exists()), "size_bytes": pred_json.stat().st_size if pred_json.exists() else 0},
    ]

    return [readiness_row], scene_rows, inventory_rows


def build_incremental_rescue_rows(scene_groups: dict[str, list[ViewRecord]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    second_rows: list[dict[str, object]] = []
    third_rows: list[dict[str, object]] = []

    for members in scene_groups.values():
        for primary, secondary in permutations(members, 2):
            primary_miss = int(not primary.target_detected)
            rescue = int((not primary.target_detected) and secondary.target_detected)
            second_rows.append(
                {
                    "scene_key": primary.scene_key,
                    "target_class": primary.target_class,
                    "primary_viewpoint": primary.viewpoint,
                    "secondary_viewpoint": secondary.viewpoint,
                    "primary_miss": primary_miss,
                    "rescue": rescue,
                    "azimuth_gap": azimuth_gap(primary.azimuth, secondary.azimuth),
                    "secondary_elevation": secondary.elevation,
                    "secondary_radius": secondary.radius,
                    "delta_best_target_ap50_95": max(primary.target_ap50_95, secondary.target_ap50_95) - primary.target_ap50_95,
                    "delta_best_target_strict_quality_iou50": max(primary.target_strict_quality_iou50, secondary.target_strict_quality_iou50) - primary.target_strict_quality_iou50,
                }
            )

        for first, second, third in permutations(members, 3):
            first_two_miss = int((not first.target_detected) and (not second.target_detected))
            third_or_rescue = int(first_two_miss and third.target_detected)
            first_two_support = int(first.target_detected) + int(second.target_detected)
            upgrade_possible = int(first_two_support == 1)
            majority_upgrade = int(upgrade_possible and third.target_detected)
            third_rows.append(
                {
                    "scene_key": first.scene_key,
                    "target_class": first.target_class,
                    "first_viewpoint": first.viewpoint,
                    "second_viewpoint": second.viewpoint,
                    "third_viewpoint": third.viewpoint,
                    "first_two_miss": first_two_miss,
                    "third_or_rescue": third_or_rescue,
                    "first_two_support_equals_one": upgrade_possible,
                    "third_majority_upgrade": majority_upgrade,
                    "mean_gap_to_first_two": mean([azimuth_gap(third.azimuth, first.azimuth), azimuth_gap(third.azimuth, second.azimuth)]),
                    "third_elevation": third.elevation,
                    "third_radius": third.radius,
                    "delta_best_target_ap50_95": max(first.target_ap50_95, second.target_ap50_95, third.target_ap50_95) - max(first.target_ap50_95, second.target_ap50_95),
                    "delta_best_target_strict_quality_iou50": max(first.target_strict_quality_iou50, second.target_strict_quality_iou50, third.target_strict_quality_iou50) - max(first.target_strict_quality_iou50, second.target_strict_quality_iou50),
                }
            )

    return second_rows, third_rows


def summarize_rescue(rows: list[dict[str, object]], key_name: str, failure_key: str, rescue_key: str, extra_factor_name: str, extra_radius_name: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)

    summary: list[dict[str, object]] = []
    for label, members in sorted(grouped.items(), key=lambda item: viewpoint_sort_key(item[0])):
        failure_count = sum(int(row[failure_key]) for row in members)
        rescue_count = sum(int(row[rescue_key]) for row in members)
        summary.append(
            {
                key_name: label,
                "sample_count": len(members),
                "failure_count": failure_count,
                "rescue_count": rescue_count,
                "rescue_rate_over_all": safe_divide(rescue_count, len(members)),
                "rescue_rate_given_failure": safe_divide(rescue_count, failure_count),
                "mean_azimuth_gap": mean(float(row.get("azimuth_gap", row.get("mean_gap_to_first_two", 0.0))) for row in members),
                "mode_elevation": members[0][extra_factor_name],
                "mode_radius": members[0][extra_radius_name],
                "mean_delta_best_target_ap50_95": mean(float(row.get("delta_best_target_ap50_95", 0.0)) for row in members),
                "mean_delta_best_target_strict_quality_iou50": mean(float(row.get("delta_best_target_strict_quality_iou50", 0.0)) for row in members),
            }
        )
    return summary


def summarize_protocol_deltas(overall_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_protocol = {str(row["protocol_id"]): row for row in overall_rows}
    baseline = by_protocol["n1_any1"]
    rows: list[dict[str, object]] = []
    for row in overall_rows:
        rows.append(
            {
                "protocol_id": row["protocol_id"],
                "protocol_label": row["protocol_label"],
                "drone_count": row["drone_count"],
                "delta_target_found_rate_vs_n1_any1": float(row["expected_target_found_rate"]) - float(baseline["expected_target_found_rate"]),
                "delta_target_threshold_ap50_95_vs_n1_any1": float(row["expected_target_threshold_ap50_95"]) - float(baseline["expected_target_threshold_ap50_95"]),
                "delta_target_threshold_match_confidence_iou50_vs_n1_any1": float(row["expected_target_threshold_match_confidence_iou50"]) - float(baseline["expected_target_threshold_match_confidence_iou50"]),
                "delta_target_threshold_strict_quality_iou50_vs_n1_any1": float(row["expected_target_threshold_strict_quality_iou50"]) - float(baseline["expected_target_threshold_strict_quality_iou50"]),
                "delta_best_available_ap50_95_vs_n1_any1": float(row["expected_best_available_ap50_95"]) - float(baseline["expected_best_available_ap50_95"]),
            }
        )
    return rows


def summarize_class_deltas(class_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = {
        str(row["target_class"]): row
        for row in class_rows
        if str(row["protocol_id"]) == "n1_any1"
    }
    rows: list[dict[str, object]] = []
    for row in class_rows:
        base = baseline[str(row["target_class"])]
        rows.append(
            {
                "protocol_id": row["protocol_id"],
                "protocol_label": row["protocol_label"],
                "drone_count": row["drone_count"],
                "target_class": row["target_class"],
                "delta_target_found_rate_vs_n1_any1": float(row["expected_target_found_rate"]) - float(base["expected_target_found_rate"]),
                "delta_target_threshold_ap50_95_vs_n1_any1": float(row["expected_target_threshold_ap50_95"]) - float(base["expected_target_threshold_ap50_95"]),
                "delta_target_threshold_strict_quality_iou50_vs_n1_any1": float(row["expected_target_threshold_strict_quality_iou50"]) - float(base["expected_target_threshold_strict_quality_iou50"]),
                "delta_best_available_ap50_95_vs_n1_any1": float(row["expected_best_available_ap50_95"]) - float(base["expected_best_available_ap50_95"]),
            }
        )
    return rows


def plot_protocol_comparison(overall_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["short_label"]) if "short_label" in row else str(row["protocol_id"]) for row in overall_rows]
    x = np.arange(len(overall_rows))
    threshold_confidence = [float(row["expected_target_threshold_match_confidence_iou50"]) for row in overall_rows]
    threshold_quality = [float(row["expected_target_threshold_strict_quality_iou50"]) for row in overall_rows]
    threshold_ap = [float(row["expected_target_threshold_ap50_95"]) for row in overall_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    ax.bar(x - width, threshold_confidence, width, label="Threshold target confidence", color="#1f77b4")
    ax.bar(x, threshold_quality, width, label="Threshold target strict quality", color="#ff7f0e")
    ax.bar(x + width, threshold_ap, width, label="Threshold target AP50-95", color="#2ca02c")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Swarm protocol comparison with stricter target-quality metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_direct_drone_count_strict_quality(overall_rows: list[dict[str, object]], output_path: Path) -> None:
    row_map = {str(row["protocol_id"]): row for row in overall_rows}
    direct_rows = [
        ("1 drone", row_map["n1_any1"]),
        ("2 drones", row_map["n2_any1"]),
        ("3 drones", row_map["n3_any1"]),
    ]

    labels = [label for label, _ in direct_rows]
    strict_quality = [float(row["expected_target_threshold_strict_quality_iou50"]) for _, row in direct_rows]
    target_ap = [float(row["expected_target_threshold_ap50_95"]) for _, row in direct_rows]
    x = np.arange(len(direct_rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x,
        strict_quality,
        width=0.58,
        color=["#5B8FF9", "#2CA58D", "#F28E2B"],
        edgecolor="#1f1f1f",
        linewidth=0.8,
        zorder=3,
    )
    ax.plot(x, target_ap, color="#C73E1D", marker="o", linewidth=2.2, markersize=7, label="Threshold target AP50-95", zorder=4)

    baseline = strict_quality[0]
    for idx, (bar, value) in enumerate(zip(bars, strict_quality)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if idx > 0:
            delta = value - baseline
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.045,
                f"+{delta:.3f} vs 1",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1f1f1f",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Direct Swarm Comparison: Strict Target Quality by Drone Count")
    ax.grid(axis="y", linestyle="--", alpha=0.28, zorder=0)
    ax.legend(frameon=False, loc="lower right")

    caption = (
        "Bars: threshold target strict quality (confidence x matched IoU).\n"
        "Line: threshold target AP50-95. Protocols: 1-of-1, 1-of-2, 1-of-3."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_readiness(scene_rows: list[dict[str, object]], output_path: Path) -> None:
    counts = [int(row["available_view_count"]) for row in scene_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts, bins=range(min(counts), max(counts) + 2), color="#17becf", edgecolor="white")
    ax.set_xlabel("Available viewpoints per scene")
    ax.set_ylabel("Number of scenes")
    ax.set_title("Scene support for 1/2/3-drone simulation")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pattern_summary(rows: list[dict[str, object]], label_key: str, output_path: Path, title: str, xlabel: str, sort_mode: str = "score") -> None:
    if not rows:
        return

    if sort_mode == "numeric_label":
        ordered_rows = sorted(rows, key=lambda row: float(row[label_key]))
    else:
        ordered_rows = sorted(rows, key=lambda row: float(row["expected_target_threshold_strict_quality_iou50"]), reverse=True)

    labels = [str(row[label_key]) for row in ordered_rows]
    values = [float(row["expected_target_threshold_strict_quality_iou50"]) for row in ordered_rows]
    support = [int(row["sample_count"]) for row in ordered_rows]

    fig_height = max(4.8, len(rows) * 0.55)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(labels, values, color="#2CA58D")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, score, count in zip(bars, values, support):
        ax.text(
            min(score + 0.004, 0.995),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}  (n={count})",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_protocol_matrix(class_rows: list[dict[str, object]], output_path: Path) -> None:
    protocols = [row.protocol_id for row in PROTOCOLS]
    class_names = sorted({str(row["target_class"]) for row in class_rows})
    value_map = {(str(row["target_class"]), str(row["protocol_id"])): float(row["expected_target_threshold_strict_quality_iou50"]) for row in class_rows}
    grid = np.array([[value_map.get((class_name, protocol_id), np.nan) for protocol_id in protocols] for class_name in class_names], dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.45)))
    image = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(np.arange(len(protocols)))
    ax.set_xticklabels(protocols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title("Per-class strict target quality by swarm protocol")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Threshold target strict quality")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(output_path: Path, readiness_rows: list[dict[str, object]], overall_rows: list[dict[str, object]], class_delta_rows: list[dict[str, object]], pair_rows: list[dict[str, object]], triple_rows: list[dict[str, object]], second_rescue_rows: list[dict[str, object]], third_rescue_rows: list[dict[str, object]]) -> None:
    readiness = readiness_rows[0]
    overall_by_id = {str(row["protocol_id"]): row for row in overall_rows}
    best_pair_or = sorted([row for row in pair_rows if row["protocol_id"] == "n2_any1"], key=lambda row: (float(row["expected_target_threshold_strict_quality_iou50"]), float(row["expected_target_threshold_ap50_95"])), reverse=True)[:5]
    best_pair_confirm = sorted([row for row in pair_rows if row["protocol_id"] == "n2_all2"], key=lambda row: (float(row["expected_target_threshold_strict_quality_iou50"]), float(row["expected_target_threshold_ap50_95"])), reverse=True)[:5]
    best_triple_or = sorted([row for row in triple_rows if row["protocol_id"] == "n3_any1"], key=lambda row: (float(row["expected_target_threshold_strict_quality_iou50"]), float(row["expected_target_threshold_ap50_95"])), reverse=True)[:5]
    best_triple_majority = sorted([row for row in triple_rows if row["protocol_id"] == "n3_majority2"], key=lambda row: (float(row["expected_target_threshold_strict_quality_iou50"]), float(row["expected_target_threshold_ap50_95"])), reverse=True)[:5]
    biggest_two_drone_gain = sorted([row for row in class_delta_rows if row["protocol_id"] == "n2_any1"], key=lambda row: float(row["delta_target_threshold_strict_quality_iou50_vs_n1_any1"]), reverse=True)[:5]
    biggest_three_drone_gain = sorted([row for row in class_delta_rows if row["protocol_id"] == "n3_any1"], key=lambda row: float(row["delta_target_threshold_strict_quality_iou50_vs_n1_any1"]), reverse=True)[:5]

    lines = [
        "# Thesis-Style Swarm Detection Analysis",
        "",
        "## Data Readiness",
        "",
        f"- Images available: {int(readiness['image_count'])}",
        f"- Scenes available: {int(readiness['scene_count'])}",
        f"- Absolute viewpoints available: {int(readiness['absolute_viewpoint_count'])}",
        f"- Views per scene: min {int(readiness['min_views_per_scene'])}, mean {float(readiness['mean_views_per_scene']):.2f}, max {int(readiness['max_views_per_scene'])}",
        f"- Scenes supporting 3-drone simulation: {int(readiness['scenes_supporting_3_drones'])}",
        f"- Filename target visible in GT across all views: {float(readiness['target_visible_rate_across_all_views']):.4f}",
        f"- Filename target detected across all views: {float(readiness['target_detected_rate_across_all_views']):.4f}",
        f"- Target mean AP50-95 across all views: {float(readiness['target_mean_ap50_95_across_all_views']):.4f}",
        f"- Target mean matched confidence at IoU>=0.50: {float(readiness['target_mean_match_confidence_iou50_across_all_views']):.4f}",
        f"- Target mean strict quality at IoU>=0.50: {float(readiness['target_mean_strict_quality_iou50_across_all_views']):.4f}",
        "",
        "## Protocol Overview",
        "",
    ]

    for protocol in PROTOCOLS:
        row = overall_by_id[protocol.protocol_id]
        lines.append(
            f"- `{protocol.short_label}`: threshold target confidence {float(row['expected_target_threshold_match_confidence_iou50']):.4f}, threshold strict quality {float(row['expected_target_threshold_strict_quality_iou50']):.4f}, threshold target AP50-95 {float(row['expected_target_threshold_ap50_95']):.4f}, binary found reference {float(row['expected_target_found_rate']):.4f}"
        )

    lines.extend(["", "## Best 2-Drone OR Pairs", ""])
    for row in best_pair_or:
        lines.append(f"- `{row['combination_label']}`: threshold strict quality {float(row['expected_target_threshold_strict_quality_iou50']):.4f}, threshold target AP50-95 {float(row['expected_target_threshold_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Best 2-Drone Confirmation Pairs", ""])
    for row in best_pair_confirm:
        lines.append(f"- `{row['combination_label']}`: threshold strict quality {float(row['expected_target_threshold_strict_quality_iou50']):.4f}, threshold target AP50-95 {float(row['expected_target_threshold_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Best 3-Drone OR Triples", ""])
    for row in best_triple_or:
        lines.append(f"- `{row['combination_label']}`: threshold strict quality {float(row['expected_target_threshold_strict_quality_iou50']):.4f}, threshold target AP50-95 {float(row['expected_target_threshold_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Best 3-Drone 2-of-3 Confirmation Triples", ""])
    for row in best_triple_majority:
        lines.append(f"- `{row['combination_label']}`: threshold strict quality {float(row['expected_target_threshold_strict_quality_iou50']):.4f}, threshold target AP50-95 {float(row['expected_target_threshold_ap50_95']):.4f}, support {int(row['sample_count'])}")

    lines.extend(["", "## Strongest Incremental Rescue Views", ""])
    for row in sorted(second_rescue_rows, key=lambda row: float(row["rescue_rate_given_failure"]), reverse=True)[:5]:
        lines.append(f"- Second drone `{row['secondary_viewpoint']}`: rescue | primary miss {float(row['rescue_rate_given_failure']):.4f}, mean strict-quality lift {float(row['mean_delta_best_target_strict_quality_iou50']):.4f}")
    for row in sorted(third_rescue_rows, key=lambda row: float(row["rescue_rate_given_failure"]), reverse=True)[:5]:
        lines.append(f"- Third drone `{row['third_viewpoint']}`: rescue | first two miss {float(row['rescue_rate_given_failure']):.4f}, mean strict-quality lift {float(row['mean_delta_best_target_strict_quality_iou50']):.4f}")

    lines.extend(["", "## Largest Class Gains For 2-Drone OR", ""])
    for row in biggest_two_drone_gain:
        lines.append(f"- `{row['target_class']}`: delta threshold strict quality {float(row['delta_target_threshold_strict_quality_iou50_vs_n1_any1']):.4f}, delta threshold AP50-95 {float(row['delta_target_threshold_ap50_95_vs_n1_any1']):.4f}")

    lines.extend(["", "## Largest Class Gains For 3-Drone OR", ""])
    for row in biggest_three_drone_gain:
        lines.append(f"- `{row['target_class']}`: delta threshold strict quality {float(row['delta_target_threshold_strict_quality_iou50_vs_n1_any1']):.4f}, delta threshold AP50-95 {float(row['delta_target_threshold_ap50_95_vs_n1_any1']):.4f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_json = Path(args.gt_json).resolve()
    pred_json = Path(args.pred_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_view_records(gt_json, pred_json, args.score_threshold)
    scene_groups = build_scene_groups(records)

    readiness_rows, scene_availability_rows, inventory_rows = build_data_readiness(records, scene_groups, gt_json, pred_json, args.score_threshold)
    combo_rows_by_n = {1: build_combo_rows(scene_groups, 1), 2: build_combo_rows(scene_groups, 2), 3: build_combo_rows(scene_groups, 3)}

    protocol_scene_rows: list[dict[str, object]] = []
    protocol_overall_rows: list[dict[str, object]] = []
    protocol_class_rows: list[dict[str, object]] = []
    pair_protocol_rows: list[dict[str, object]] = []
    triple_protocol_rows: list[dict[str, object]] = []
    triple_diversity_rows: list[dict[str, object]] = []
    pair_elevation_pattern_rows: list[dict[str, object]] = []
    pair_radius_pattern_rows: list[dict[str, object]] = []
    pair_gap_pattern_rows: list[dict[str, object]] = []
    pair_structure_pattern_rows: list[dict[str, object]] = []
    triple_elevation_pattern_rows: list[dict[str, object]] = []
    triple_radius_pattern_rows: list[dict[str, object]] = []
    triple_gap_pattern_rows: list[dict[str, object]] = []
    triple_structure_pattern_rows: list[dict[str, object]] = []

    for protocol in PROTOCOLS:
        protocol_combo_rows = add_protocol_flags(combo_rows_by_n[protocol.drone_count], protocol)
        scene_rows = summarize_protocol_scene_expectations(protocol_combo_rows, protocol)
        overall_row = summarize_protocol_overall(scene_rows)
        overall_row["short_label"] = protocol.short_label
        protocol_scene_rows.extend(scene_rows)
        protocol_overall_rows.append(overall_row)
        protocol_class_rows.extend(summarize_protocol_by_class(scene_rows))

        exact_rows = summarize_exact_combinations(protocol_combo_rows, protocol)
        if protocol.drone_count == 2:
            pair_protocol_rows.extend(exact_rows)
            elevation_rows, radius_rows, gap_rows, structure_rows = summarize_pair_angle_patterns(protocol_combo_rows, protocol)
            pair_elevation_pattern_rows.extend(elevation_rows)
            pair_radius_pattern_rows.extend(radius_rows)
            pair_gap_pattern_rows.extend(gap_rows)
            pair_structure_pattern_rows.extend(structure_rows)
        elif protocol.drone_count == 3:
            triple_protocol_rows.extend(exact_rows)
            triple_diversity_rows.extend(summarize_triple_diversity(protocol_combo_rows, protocol))
            elevation_rows, radius_rows, gap_rows, structure_rows = summarize_triple_angle_patterns(protocol_combo_rows, protocol)
            triple_elevation_pattern_rows.extend(elevation_rows)
            triple_radius_pattern_rows.extend(radius_rows)
            triple_gap_pattern_rows.extend(gap_rows)
            triple_structure_pattern_rows.extend(structure_rows)

    second_rescue_records, third_rescue_records = build_incremental_rescue_rows(scene_groups)
    second_rescue_summary = summarize_rescue(second_rescue_records, "secondary_viewpoint", "primary_miss", "rescue", "secondary_elevation", "secondary_radius")
    third_or_rescue_summary = summarize_rescue(third_rescue_records, "third_viewpoint", "first_two_miss", "third_or_rescue", "third_elevation", "third_radius")
    third_upgrade_summary = summarize_rescue(third_rescue_records, "third_viewpoint", "first_two_support_equals_one", "third_majority_upgrade", "third_elevation", "third_radius")
    protocol_delta_rows = summarize_protocol_deltas(protocol_overall_rows)
    class_delta_rows = summarize_class_deltas(protocol_class_rows)

    write_csv_rows(output_dir / "data_readiness_summary.csv", readiness_rows)
    write_csv_rows(output_dir / "input_file_inventory.csv", inventory_rows)
    write_csv_rows(output_dir / "scene_availability_summary.csv", scene_availability_rows)
    write_csv_rows(output_dir / "protocol_scene_expectation_summary.csv", protocol_scene_rows)
    write_csv_rows(output_dir / "protocol_overall_summary.csv", protocol_overall_rows)
    write_csv_rows(output_dir / "protocol_delta_vs_single_summary.csv", protocol_delta_rows)
    write_csv_rows(output_dir / "protocol_class_summary.csv", protocol_class_rows)
    write_csv_rows(output_dir / "protocol_class_delta_vs_single_summary.csv", class_delta_rows)
    write_csv_rows(output_dir / "pair_protocol_summary.csv", pair_protocol_rows)
    write_csv_rows(output_dir / "triple_protocol_summary.csv", triple_protocol_rows)
    write_csv_rows(output_dir / "triple_diversity_summary.csv", triple_diversity_rows)
    write_csv_rows(output_dir / "pair_elevation_pattern_summary.csv", pair_elevation_pattern_rows)
    write_csv_rows(output_dir / "pair_radius_pattern_summary.csv", pair_radius_pattern_rows)
    write_csv_rows(output_dir / "pair_azimuth_gap_pattern_summary.csv", pair_gap_pattern_rows)
    write_csv_rows(output_dir / "pair_structure_pattern_summary.csv", pair_structure_pattern_rows)
    write_csv_rows(output_dir / "triple_elevation_pattern_summary.csv", triple_elevation_pattern_rows)
    write_csv_rows(output_dir / "triple_radius_pattern_summary.csv", triple_radius_pattern_rows)
    write_csv_rows(output_dir / "triple_azimuth_gap_pattern_summary.csv", triple_gap_pattern_rows)
    write_csv_rows(output_dir / "triple_structure_pattern_summary.csv", triple_structure_pattern_rows)
    write_csv_rows(output_dir / "second_drone_rescue_summary.csv", second_rescue_summary)
    write_csv_rows(output_dir / "third_drone_or_rescue_summary.csv", third_or_rescue_summary)
    write_csv_rows(output_dir / "third_drone_confirmation_upgrade_summary.csv", third_upgrade_summary)

    plot_readiness(scene_availability_rows, output_dir / "scene_view_count_distribution.png")
    plot_protocol_comparison(protocol_overall_rows, output_dir / "protocol_comparison.png")
    plot_direct_drone_count_strict_quality(protocol_overall_rows, output_dir / "direct_one_two_three_strict_quality.png")
    plot_class_protocol_matrix(protocol_class_rows, output_dir / "protocol_class_matrix.png")

    filtered_pair_or = [row for row in pair_protocol_rows if row["protocol_id"] == "n2_any1" and int(row["sample_count"]) >= args.min_pair_support]
    filtered_pair_confirm = [row for row in pair_protocol_rows if row["protocol_id"] == "n2_all2" and int(row["sample_count"]) >= args.min_pair_support]
    filtered_triple_or = [row for row in triple_protocol_rows if row["protocol_id"] == "n3_any1" and int(row["sample_count"]) >= args.min_triple_support]
    filtered_triple_majority = [row for row in triple_protocol_rows if row["protocol_id"] == "n3_majority2" and int(row["sample_count"]) >= args.min_triple_support]
    filtered_second_rescue = [row for row in second_rescue_summary if int(row["failure_count"]) >= args.min_rescue_support]
    filtered_third_rescue = [row for row in third_or_rescue_summary if int(row["failure_count"]) >= args.min_rescue_support]
    filtered_third_upgrade = [row for row in third_upgrade_summary if int(row["failure_count"]) >= args.min_rescue_support]
    filtered_pair_elevation_or = [row for row in pair_elevation_pattern_rows if row["protocol_id"] == "n2_any1"]
    filtered_pair_radius_or = [row for row in pair_radius_pattern_rows if row["protocol_id"] == "n2_any1"]
    filtered_pair_gap_or = [row for row in pair_gap_pattern_rows if row["protocol_id"] == "n2_any1"]
    filtered_triple_elevation_or = [row for row in triple_elevation_pattern_rows if row["protocol_id"] == "n3_any1" and int(row["sample_count"]) >= 5]
    filtered_triple_radius_or = [row for row in triple_radius_pattern_rows if row["protocol_id"] == "n3_any1" and int(row["sample_count"]) >= 5]
    filtered_triple_gap_or = [row for row in triple_gap_pattern_rows if row["protocol_id"] == "n3_any1" and int(row["sample_count"]) >= 5]
    filtered_pair_structure_or = [row for row in pair_structure_pattern_rows if row["protocol_id"] == "n2_any1" and int(row["sample_count"]) >= 8]
    filtered_triple_structure_or = [row for row in triple_structure_pattern_rows if row["protocol_id"] == "n3_any1" and int(row["sample_count"]) >= 5]

    plot_top_barh(filtered_pair_or, "combination_label", "expected_target_threshold_strict_quality_iou50", output_dir / "top_pairs_or.png", "Top 2-drone OR combinations", "Threshold target strict quality")
    plot_top_barh(filtered_pair_confirm, "combination_label", "expected_target_threshold_strict_quality_iou50", output_dir / "top_pairs_confirmation.png", "Top 2-drone confirmation combinations", "Threshold target strict quality")
    plot_top_barh(filtered_triple_or, "combination_label", "expected_target_threshold_strict_quality_iou50", output_dir / "top_triples_or.png", "Top 3-drone OR combinations", "Threshold target strict quality")
    plot_top_barh(filtered_triple_majority, "combination_label", "expected_target_threshold_strict_quality_iou50", output_dir / "top_triples_majority.png", "Top 3-drone 2-of-3 combinations", "Threshold target strict quality")
    plot_top_barh(filtered_second_rescue, "secondary_viewpoint", "rescue_rate_given_failure", output_dir / "top_second_drone_rescue.png", "Best second-drone rescue viewpoints", "Rescue rate | primary miss")
    plot_top_barh(filtered_third_rescue, "third_viewpoint", "rescue_rate_given_failure", output_dir / "top_third_drone_or_rescue.png", "Best third-drone OR rescue viewpoints", "Rescue rate | first two miss")
    plot_top_barh(filtered_third_upgrade, "third_viewpoint", "rescue_rate_given_failure", output_dir / "top_third_drone_confirmation_upgrade.png", "Best third-drone majority-upgrade viewpoints", "Upgrade rate | first two have one support")
    plot_top_barh(filtered_second_rescue, "secondary_viewpoint", "mean_delta_best_target_strict_quality_iou50", output_dir / "top_second_drone_quality_lift.png", "Best second-drone strict-quality lift", "Mean strict-quality lift")
    plot_top_barh(filtered_third_rescue, "third_viewpoint", "mean_delta_best_target_strict_quality_iou50", output_dir / "top_third_drone_quality_lift.png", "Best third-drone strict-quality lift", "Mean strict-quality lift")
    plot_pattern_summary(filtered_pair_elevation_or, "elevation_pattern", output_dir / "pair_elevation_patterns_or.png", "2-Drone OR: Elevation patterns by strict target quality", "Expected threshold target strict quality")
    plot_pattern_summary(filtered_pair_radius_or, "radius_pattern", output_dir / "pair_radius_patterns_or.png", "2-Drone OR: Radius patterns by strict target quality", "Expected threshold target strict quality")
    plot_pattern_summary(filtered_pair_gap_or, "azimuth_gap", output_dir / "pair_azimuth_gap_patterns_or.png", "2-Drone OR: Azimuth gap patterns by strict target quality", "Expected threshold target strict quality", sort_mode="numeric_label")
    plot_pattern_summary(filtered_triple_elevation_or, "elevation_pattern", output_dir / "triple_elevation_patterns_or.png", "3-Drone OR: Elevation patterns by strict target quality", "Expected threshold target strict quality")
    plot_pattern_summary(filtered_triple_radius_or, "radius_pattern", output_dir / "triple_radius_patterns_or.png", "3-Drone OR: Radius patterns by strict target quality", "Expected threshold target strict quality")
    plot_pattern_summary(filtered_triple_gap_or, "max_pairwise_azimuth_gap", output_dir / "triple_azimuth_gap_patterns_or.png", "3-Drone OR: Max azimuth-gap patterns by strict target quality", "Expected threshold target strict quality", sort_mode="numeric_label")
    plot_top_barh(filtered_pair_structure_or, "structure_label", "expected_target_threshold_strict_quality_iou50", output_dir / "pair_structure_patterns_or.png", "2-Drone OR: Top structure buckets", "Threshold target strict quality", top_k=20)
    plot_top_barh(filtered_triple_structure_or, "structure_label", "expected_target_threshold_strict_quality_iou50", output_dir / "triple_structure_patterns_or.png", "3-Drone OR: Top structure buckets", "Threshold target strict quality", top_k=20)

    write_report(output_dir / "swarm_thesis_report.md", readiness_rows, protocol_overall_rows, class_delta_rows, filtered_pair_or + filtered_pair_confirm, filtered_triple_or + filtered_triple_majority, filtered_second_rescue, filtered_third_rescue)
    print(f"Saved thesis-style swarm analysis to: {output_dir}")


if __name__ == "__main__":
    main()

