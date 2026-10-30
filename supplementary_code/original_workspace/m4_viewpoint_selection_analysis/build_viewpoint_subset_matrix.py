from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
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
DEFAULT_PAIR_FUSION_ROWS = (
    WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "pair_combo_rows.csv"
)
DEFAULT_TRIPLE_FUSION_ROWS = (
    WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "triple_combo_rows.csv"
)
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_viewpoint_selection_analysis" / "outputs"

ELEVATION_SORT = {"low": 0, "mid": 1, "high": 2}
RADIUS_SORT = {"near": 0, "mid": 1, "far": 2}


@dataclass(frozen=True)
class ViewRecord:
    scene_key: str
    file_name: str
    image_id: int
    target_class: str
    viewpoint: str
    elevation: str
    radius: str
    azimuth: int
    target_visible: int
    target_detected: int
    target_ap50_95: float
    target_match_confidence_iou50: float
    target_strict_quality_iou50: float
    target_fp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build all observed 1/2/3-viewpoint subset rankings and 72x72 pair "
            "matrices from the cached M4 fixed-detector scene records."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--viewpoint-manifest", default=str(DEFAULT_VIEWPOINT_MANIFEST))
    parser.add_argument("--pair-fusion-rows", default=str(DEFAULT_PAIR_FUSION_ROWS))
    parser.add_argument("--triple-fusion-rows", default=str(DEFAULT_TRIPLE_FUSION_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-k", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=25)
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


def combo_key_from_label(label: str) -> tuple[str, ...]:
    viewpoints = [part.strip() for part in str(label).split("+") if part.strip()]
    return tuple(sorted(viewpoints, key=viewpoint_sort_key))


def combo_label(viewpoints: Iterable[str]) -> str:
    return " + ".join(sorted(viewpoints, key=viewpoint_sort_key))


def azimuth_gap(first: int, second: int) -> int:
    delta = abs(first - second) % 360
    return min(delta, 360 - delta)


def max_pairwise_azimuth_gap(azimuths: list[int]) -> int:
    if len(azimuths) < 2:
        return 0
    return max(azimuth_gap(a, b) for a, b in combinations(azimuths, 2))


def ordered_pattern(values: Iterable[str], order: dict[str, int]) -> str:
    return " + ".join(sorted(values, key=lambda value: (order.get(value, 99), value)))


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_scene_records(path: Path) -> list[ViewRecord]:
    rows: list[ViewRecord] = []
    with ensure_file(path, "Scene records CSV").open("r", newline="", encoding="utf-8") as handle:
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
            "target_visible",
            "target_detected",
            "target_ap50_95",
            "target_match_confidence_iou50",
            "target_strict_quality_iou50",
            "target_fp",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Scene records CSV is missing columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                ViewRecord(
                    scene_key=row["scene_key"],
                    file_name=row["file_name"],
                    image_id=parse_int(row["image_id"]),
                    target_class=row["target_class"],
                    viewpoint=row["viewpoint"],
                    elevation=row["elevation"],
                    radius=row["radius"],
                    azimuth=parse_int(row["azimuth"]),
                    target_visible=parse_int(row["target_visible"]),
                    target_detected=parse_int(row["target_detected"]),
                    target_ap50_95=parse_float(row["target_ap50_95"]),
                    target_match_confidence_iou50=parse_float(row["target_match_confidence_iou50"]),
                    target_strict_quality_iou50=parse_float(row["target_strict_quality_iou50"]),
                    target_fp=parse_int(row["target_fp"]),
                )
            )
    return rows


def read_viewpoint_manifest(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest: dict[str, dict[str, int]] = {}
        for row in reader:
            viewpoint = row.get("viewpoint", "").strip()
            if not viewpoint:
                continue
            manifest[viewpoint] = {
                "manifest_train_images": parse_int(row.get("train_images", 0)),
                "manifest_val_images": parse_int(row.get("val_images", 0)),
                "manifest_test_images": parse_int(row.get("test_images", 0)),
            }
    return manifest


def read_fusion_rows(paths: Iterable[Path]) -> dict[tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Counter[tuple[str, ...]] = Counter()
    numeric_fields = [
        "best_box_quality",
        "fused_quality_noisy_or_max_iou",
        "fused_quality_support_weighted_or",
        "gain_noisy_vs_best",
        "gain_support_vs_best",
        "support_ratio",
    ]

    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = combo_key_from_label(row["combination_label"])
                counts[key] += 1
                for field in numeric_fields:
                    grouped[key][field] += parse_float(row.get(field, 0.0))

    out: dict[tuple[str, ...], dict[str, float]] = {}
    for key, total in grouped.items():
        count = counts[key]
        out[key] = {f"fusion_mean_{field}": value / count for field, value in total.items()}
        out[key]["fusion_sample_count"] = float(count)
    return out


def build_scene_groups(records: list[ViewRecord]) -> dict[str, list[ViewRecord]]:
    groups: dict[str, list[ViewRecord]] = defaultdict(list)
    for record in records:
        groups[record.scene_key].append(record)
    for scene_key in groups:
        groups[scene_key].sort(key=lambda row: viewpoint_sort_key(row.viewpoint))
    return groups


def new_accumulator(key: tuple[str, ...]) -> dict[str, object]:
    return {
        "key": key,
        "sample_count": 0,
        "scene_keys": set(),
        "target_classes": Counter(),
        "target_visible_any_sum": 0,
        "target_visible_all_sum": 0,
        "target_found_or_sum": 0,
        "target_found_all_sum": 0,
        "target_found_majority_sum": 0,
        "target_absent_view_sum": 0,
        "selected_has_absent_view_sum": 0,
        "all_selected_absent_sum": 0,
        "false_alarm_any_absent_view_sum": 0,
        "false_alarm_all_absent_sum": 0,
        "best_ap_sum": 0.0,
        "mean_ap_sum": 0.0,
        "best_strict_sum": 0.0,
        "mean_strict_sum": 0.0,
        "best_confidence_sum": 0.0,
        "mean_confidence_sum": 0.0,
        "quality_by_viewpoint_sum": defaultdict(float),
        "ap_by_viewpoint_sum": defaultdict(float),
    }


def accumulate_combo(acc: dict[str, object], combo: tuple[ViewRecord, ...]) -> None:
    sample_count = int(acc["sample_count"]) + 1
    acc["sample_count"] = sample_count
    acc["scene_keys"].add(combo[0].scene_key)
    acc["target_classes"][combo[0].target_class] += 1

    visible = [record.target_visible for record in combo]
    detected = [record.target_detected for record in combo]
    ap_values = [record.target_ap50_95 for record in combo]
    strict_values = [record.target_strict_quality_iou50 for record in combo]
    confidence_values = [record.target_match_confidence_iou50 for record in combo]

    visible_count = sum(visible)
    detected_count = sum(detected)
    absent_view_count = len(combo) - visible_count
    false_alarm_on_absent = any((record.target_visible == 0 and record.target_fp > 0) for record in combo)

    acc["target_visible_any_sum"] += int(visible_count > 0)
    acc["target_visible_all_sum"] += int(visible_count == len(combo))
    acc["target_found_or_sum"] += int(detected_count > 0)
    acc["target_found_all_sum"] += int(detected_count == len(combo))
    acc["target_found_majority_sum"] += int(detected_count >= math.ceil(len(combo) / 2))
    acc["target_absent_view_sum"] += absent_view_count
    acc["selected_has_absent_view_sum"] += int(absent_view_count > 0)
    acc["all_selected_absent_sum"] += int(visible_count == 0)
    acc["false_alarm_any_absent_view_sum"] += int(false_alarm_on_absent)
    acc["false_alarm_all_absent_sum"] += int(visible_count == 0 and false_alarm_on_absent)
    acc["best_ap_sum"] += max(ap_values)
    acc["mean_ap_sum"] += mean(ap_values)
    acc["best_strict_sum"] += max(strict_values)
    acc["mean_strict_sum"] += mean(strict_values)
    acc["best_confidence_sum"] += max(confidence_values)
    acc["mean_confidence_sum"] += mean(confidence_values)

    quality_sums: defaultdict[str, float] = acc["quality_by_viewpoint_sum"]
    ap_sums: defaultdict[str, float] = acc["ap_by_viewpoint_sum"]
    for record in combo:
        quality_sums[record.viewpoint] += record.target_strict_quality_iou50
        ap_sums[record.viewpoint] += record.target_ap50_95


def subset_geometry(key: tuple[str, ...]) -> dict[str, object]:
    elevations: list[str] = []
    radii: list[str] = []
    azimuths: list[int] = []
    for viewpoint in key:
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        elevations.append(elevation)
        radii.append(radius)
        azimuths.append(azimuth)
    return {
        "elevation_pattern": ordered_pattern(elevations, ELEVATION_SORT),
        "radius_pattern": ordered_pattern(radii, RADIUS_SORT),
        "unique_elevation_count": len(set(elevations)),
        "unique_radius_count": len(set(radii)),
        "max_pairwise_azimuth_gap": max_pairwise_azimuth_gap(azimuths),
        "mean_pairwise_azimuth_gap": mean(
            azimuth_gap(a, b) for a, b in combinations(azimuths, 2)
        )
        if len(azimuths) >= 2
        else 0.0,
    }


def finalize_accumulator(
    acc: dict[str, object],
    fusion_stats: dict[tuple[str, ...], dict[str, float]] | None = None,
) -> dict[str, object]:
    key: tuple[str, ...] = acc["key"]
    sample_count = int(acc["sample_count"])
    drone_count = len(key)
    class_counter: Counter[str] = acc["target_classes"]
    quality_sums: defaultdict[str, float] = acc["quality_by_viewpoint_sum"]
    ap_sums: defaultdict[str, float] = acc["ap_by_viewpoint_sum"]

    mean_best_strict = safe_divide(float(acc["best_strict_sum"]), sample_count)
    constituent_strict_means = [safe_divide(quality_sums[viewpoint], sample_count) for viewpoint in key]
    constituent_ap_means = [safe_divide(ap_sums[viewpoint], sample_count) for viewpoint in key]
    best_constituent_strict = max(constituent_strict_means) if constituent_strict_means else 0.0
    best_constituent_ap = max(constituent_ap_means) if constituent_ap_means else 0.0

    row: dict[str, object] = {
        "drone_count": drone_count,
        "combination_label": combo_label(key),
        "viewpoint_1": key[0] if len(key) > 0 else "",
        "viewpoint_2": key[1] if len(key) > 1 else "",
        "viewpoint_3": key[2] if len(key) > 2 else "",
        "sample_count": sample_count,
        "scene_count": len(acc["scene_keys"]),
        "dominant_target_class": class_counter.most_common(1)[0][0] if class_counter else "",
        "dominant_target_class_count": class_counter.most_common(1)[0][1] if class_counter else 0,
        **subset_geometry(key),
        "target_visible_any_rate": safe_divide(acc["target_visible_any_sum"], sample_count),
        "target_visible_all_rate": safe_divide(acc["target_visible_all_sum"], sample_count),
        "target_found_or_rate": safe_divide(acc["target_found_or_sum"], sample_count),
        "target_found_majority_rate": safe_divide(acc["target_found_majority_sum"], sample_count),
        "target_found_all_rate": safe_divide(acc["target_found_all_sum"], sample_count),
        "mean_best_target_ap50_95": safe_divide(float(acc["best_ap_sum"]), sample_count),
        "mean_mean_target_ap50_95": safe_divide(float(acc["mean_ap_sum"]), sample_count),
        "mean_best_strict_quality": mean_best_strict,
        "mean_mean_strict_quality": safe_divide(float(acc["mean_strict_sum"]), sample_count),
        "mean_best_match_confidence_iou50": safe_divide(float(acc["best_confidence_sum"]), sample_count),
        "mean_mean_match_confidence_iou50": safe_divide(float(acc["mean_confidence_sum"]), sample_count),
        "best_constituent_mean_strict_quality_on_matched_scenes": best_constituent_strict,
        "complementarity_vs_best_single_strict_quality": mean_best_strict - best_constituent_strict,
        "best_constituent_mean_ap50_95_on_matched_scenes": best_constituent_ap,
        "complementarity_vs_best_single_ap50_95": safe_divide(float(acc["best_ap_sum"]), sample_count) - best_constituent_ap,
        "mean_absent_view_count": safe_divide(acc["target_absent_view_sum"], sample_count),
        "selected_has_absent_view_rate": safe_divide(acc["selected_has_absent_view_sum"], sample_count),
        "all_selected_views_absent_rate": safe_divide(acc["all_selected_absent_sum"], sample_count),
        "false_alarm_rate_when_any_selected_view_absent": safe_divide(
            acc["false_alarm_any_absent_view_sum"], acc["selected_has_absent_view_sum"]
        ),
        "false_alarm_rate_when_all_selected_views_absent": safe_divide(
            acc["false_alarm_all_absent_sum"], acc["all_selected_absent_sum"]
        ),
    }

    if fusion_stats and key in fusion_stats:
        row.update(fusion_stats[key])
    else:
        row.update(
            {
                "fusion_sample_count": 0.0,
                "fusion_mean_best_box_quality": "",
                "fusion_mean_fused_quality_noisy_or_max_iou": "",
                "fusion_mean_fused_quality_support_weighted_or": "",
                "fusion_mean_gain_noisy_vs_best": "",
                "fusion_mean_gain_support_vs_best": "",
                "fusion_mean_support_ratio": "",
            }
        )
    return row


def aggregate_subset_scores(
    records: list[ViewRecord],
    max_k: int = 3,
    fusion_stats: dict[tuple[str, ...], dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    scene_groups = build_scene_groups(records)
    accumulators: dict[tuple[str, ...], dict[str, object]] = {}

    for scene_records in scene_groups.values():
        by_viewpoint = {record.viewpoint: record for record in scene_records}
        ordered_records = [by_viewpoint[key] for key in sorted(by_viewpoint, key=viewpoint_sort_key)]
        for drone_count in range(1, max_k + 1):
            if len(ordered_records) < drone_count:
                continue
            for combo in combinations(ordered_records, drone_count):
                key = tuple(record.viewpoint for record in combo)
                if key not in accumulators:
                    accumulators[key] = new_accumulator(key)
                accumulate_combo(accumulators[key], combo)

    rows = [finalize_accumulator(acc, fusion_stats=fusion_stats) for acc in accumulators.values()]
    rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["mean_best_strict_quality"]),
            -int(row["scene_count"]),
            str(row["combination_label"]),
        )
    )
    return rows


def build_viewpoint_inventory(
    records: list[ViewRecord],
    manifest: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    by_viewpoint: dict[str, list[ViewRecord]] = defaultdict(list)
    for record in records:
        by_viewpoint[record.viewpoint].append(record)

    rows: list[dict[str, object]] = []
    for viewpoint in sorted(by_viewpoint, key=viewpoint_sort_key):
        members = by_viewpoint[viewpoint]
        absent_members = [record for record in members if record.target_visible == 0]
        row = {
            "viewpoint": viewpoint,
            "scene_count": len({record.scene_key for record in members}),
            "sample_count": len(members),
            "target_visible_rate": mean(record.target_visible for record in members),
            "target_found_rate": mean(record.target_detected for record in members),
            "mean_target_ap50_95": mean(record.target_ap50_95 for record in members),
            "mean_strict_quality": mean(record.target_strict_quality_iou50 for record in members),
            "mean_match_confidence_iou50": mean(record.target_match_confidence_iou50 for record in members),
            "absent_view_count": len(absent_members),
            "false_alarm_rate_when_target_absent": safe_divide(
                sum(1 for record in absent_members if record.target_fp > 0), len(absent_members)
            ),
        }
        row.update(manifest.get(viewpoint, {}))
        rows.append(row)
    return rows


def float_or_nan(value: object) -> float:
    if value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_pair_matrix(
    path: Path,
    viewpoints: list[str],
    pair_rows: list[dict[str, object]],
    single_rows: list[dict[str, object]],
    metric: str,
    diagonal_metric: str | None = None,
) -> np.ndarray:
    diagonal_metric = diagonal_metric or metric
    values_by_pair = {
        combo_key_from_label(str(row["combination_label"])): float_or_nan(row.get(metric, ""))
        for row in pair_rows
    }
    diagonal_by_viewpoint = {
        str(row["combination_label"]): float_or_nan(row.get(diagonal_metric, ""))
        for row in single_rows
    }

    matrix = np.full((len(viewpoints), len(viewpoints)), np.nan, dtype=float)
    for i, first in enumerate(viewpoints):
        for j, second in enumerate(viewpoints):
            if i == j:
                matrix[i, j] = diagonal_by_viewpoint.get(first, np.nan)
            else:
                key = tuple(sorted((first, second), key=viewpoint_sort_key))
                matrix[i, j] = values_by_pair.get(key, np.nan)

    csv_rows: list[dict[str, object]] = []
    for i, viewpoint in enumerate(viewpoints):
        row: dict[str, object] = {"viewpoint": viewpoint}
        for j, column_viewpoint in enumerate(viewpoints):
            value = matrix[i, j]
            row[column_viewpoint] = "" if np.isnan(value) else f"{value:.6f}"
        csv_rows.append(row)
    write_csv(path, csv_rows, fieldnames=["viewpoint", *viewpoints])
    return matrix


def plot_matrix(
    matrix: np.ndarray,
    viewpoints: list[str],
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    tick_step = 8
    ticks = list(range(0, len(viewpoints), tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([viewpoints[index] for index in ticks], rotation=90, fontsize=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([viewpoints[index] for index in ticks], fontsize=6)
    for boundary in range(8, len(viewpoints), 8):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Second viewpoint, sorted by elevation/radius/azimuth")
    ax.set_ylabel("First viewpoint, sorted by elevation/radius/azimuth")
    fig.colorbar(im, ax=ax, shrink=0.82, label=colorbar_label)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_gain_curve(rows: list[dict[str, object]], output_path: Path, min_support: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for drone_count in [1, 2, 3]:
        members = [
            row
            for row in rows
            if int(row["drone_count"]) == drone_count and int(row["scene_count"]) >= min_support
        ]
        if not members:
            continue
        values = [float(row["mean_best_strict_quality"]) for row in members]
        summary_rows.append(
            {
                "drone_count": drone_count,
                "best": max(values),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }
        )

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = [row["drone_count"] for row in summary_rows]
    ax.plot(x, [row["best"] for row in summary_rows], marker="o", linewidth=2.2, label="Best subset")
    ax.plot(x, [row["p90"] for row in summary_rows], marker="o", linewidth=1.8, label="90th percentile")
    ax.plot(x, [row["median"] for row in summary_rows], marker="o", linewidth=1.8, label="Median subset")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Number of selected viewpoints")
    ax.set_ylabel("Mean best strict target quality")
    ax.set_title("Subset quality by swarm size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_top_subsets(rows: list[dict[str, object]], output_path: Path, min_support: int, top_n: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=True)
    for axis, drone_count in zip(axes, [1, 2, 3]):
        members = [
            row
            for row in rows
            if int(row["drone_count"]) == drone_count and int(row["scene_count"]) >= min_support
        ]
        members = sorted(
            members,
            key=lambda row: (float(row["mean_best_strict_quality"]), int(row["scene_count"])),
            reverse=True,
        )[:top_n]
        labels = [str(row["combination_label"]) for row in members][::-1]
        values = [float(row["mean_best_strict_quality"]) for row in members][::-1]
        y = np.arange(len(labels))
        axis.barh(y, values, color="#3b6ea8")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=6)
        axis.set_xlabel("Mean best strict target quality")
        axis.set_title(f"Top {drone_count}-view subsets")
        axis.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def build_factor_pattern_summary(rows: list[dict[str, object]], min_support: int) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["scene_count"]) < min_support:
            continue
        if int(row["drone_count"]) == 1:
            continue
        key = (int(row["drone_count"]), str(row["elevation_pattern"]), str(row["radius_pattern"]))
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for (drone_count, elevation_pattern, radius_pattern), members in grouped.items():
        out.append(
            {
                "drone_count": drone_count,
                "elevation_pattern": elevation_pattern,
                "radius_pattern": radius_pattern,
                "combination_count": len(members),
                "mean_scene_count": mean(int(row["scene_count"]) for row in members),
                "mean_best_strict_quality": mean(float(row["mean_best_strict_quality"]) for row in members),
                "mean_complementarity_strict_quality": mean(
                    float(row["complementarity_vs_best_single_strict_quality"]) for row in members
                ),
                "mean_target_found_or_rate": mean(float(row["target_found_or_rate"]) for row in members),
                "mean_false_alarm_rate_when_any_selected_view_absent": mean(
                    float(row["false_alarm_rate_when_any_selected_view_absent"]) for row in members
                ),
            }
        )
    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["mean_best_strict_quality"]),
            str(row["elevation_pattern"]),
            str(row["radius_pattern"]),
        )
    )
    return out


def plot_factor_patterns(rows: list[dict[str, object]], output_path: Path, top_n: int = 18) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(rows, key=lambda row: float(row["mean_best_strict_quality"]), reverse=True)[:top_n]
    labels = [
        f"k={row['drone_count']} | {row['elevation_pattern']} | {row['radius_pattern']}"
        for row in members
    ][::-1]
    values = [float(row["mean_best_strict_quality"]) for row in members][::-1]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4f8f6f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean best strict target quality")
    ax.set_title("Best geometry patterns among supported subsets")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def build_best_subsets(rows: list[dict[str, object]], min_support: int, top_n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for drone_count in [1, 2, 3]:
        members = [
            row
            for row in rows
            if int(row["drone_count"]) == drone_count and int(row["scene_count"]) >= min_support
        ]
        members.sort(
            key=lambda row: (float(row["mean_best_strict_quality"]), int(row["scene_count"])),
            reverse=True,
        )
        for rank, row in enumerate(members[:top_n], start=1):
            out.append({"rank_within_drone_count": rank, **row})
    return out


def write_report(
    output_path: Path,
    records: list[ViewRecord],
    subset_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    min_support: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_count = len({record.scene_key for record in records})
    viewpoint_count = len({record.viewpoint for record in records})
    absent_count = sum(1 for record in records if record.target_visible == 0)

    lines = [
        "# Viewpoint Subset Matrix Report",
        "",
        "## Inputs",
        f"- Scene/view records: `{DEFAULT_SCENE_RECORDS}`",
        f"- Records: `{len(records)}`",
        f"- Scene keys: `{scene_count}`",
        f"- Absolute viewpoints: `{viewpoint_count}`",
        f"- Target-absent views: `{absent_count}` (`{safe_divide(absent_count, len(records)):.4f}`)",
        "",
        "## Method Boundary",
        "",
        "This analysis reuses cached fixed-detector M4 records. It ranks viewpoint subsets by target-centric evidence available in the selected views. It does not retrain detectors and does not assume calibrated 3D geometry.",
        "",
        "## Support Distribution",
        "",
        "Exact fixed subsets are only evaluated on scenes where all selected viewpoints are present. This matters most for triples.",
        "",
    ]
    for drone_count in [1, 2, 3]:
        members = [row for row in subset_rows if int(row["drone_count"]) == drone_count]
        supports = sorted(int(row["scene_count"]) for row in members)
        if not supports:
            continue
        median_support = supports[len(supports) // 2]
        lines.append(
            "- "
            f"k=`{drone_count}`: `{len(members)}` observed subsets, "
            f"support min/median/max = `{supports[0]}`/`{median_support}`/`{supports[-1]}` scenes"
        )
    lines.extend(
        [
            "",
            "Use the scene-split validation script before treating high-scoring sparse 2-view or 3-view exact subsets as robust recommendations.",
            "",
        "## Best Supported Subsets",
        "",
        f"Minimum support for the headline list: `{min_support}` scenes.",
        "",
        ]
    )
    for drone_count in [1, 2, 3]:
        members = [row for row in best_rows if int(row["drone_count"]) == drone_count][:5]
        lines.append(f"### {drone_count} Viewpoint(s)")
        if not members:
            lines.append("- No supported subset found.")
            lines.append("")
            continue
        for row in members:
            lines.append(
                "- "
                f"`{row['combination_label']}`: strict quality `{float(row['mean_best_strict_quality']):.4f}`, "
                f"target found OR `{float(row['target_found_or_rate']):.4f}`, "
                f"support `{int(row['scene_count'])}` scenes"
            )
        lines.append("")

    lines.extend(
        [
            "## Generated Files",
            "",
            "- `viewpoint_inventory.csv`",
            "- `subset_scores.csv`",
            "- `best_subsets_by_k.csv`",
            "- `pair_matrix_strict_quality.csv`",
            "- `pair_matrix_complementarity.csv`",
            "- `pair_matrix_support.csv`",
            "- `pair_matrix_support_weighted_fusion.csv`",
            "- `factor_pattern_summary.csv`",
            "- `plots/pair_matrix_strict_quality.png`",
            "- `plots/pair_matrix_complementarity.png`",
            "- `plots/subset_size_gain_curve.png`",
            "- `plots/top_subsets_by_k.png`",
            "- `plots/factor_pattern_summary.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_records_path = Path(args.scene_records)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    records = read_scene_records(scene_records_path)
    manifest = read_viewpoint_manifest(Path(args.viewpoint_manifest))
    fusion_stats = read_fusion_rows([Path(args.pair_fusion_rows), Path(args.triple_fusion_rows)])

    inventory_rows = build_viewpoint_inventory(records, manifest)
    subset_rows = aggregate_subset_scores(records, max_k=args.max_k, fusion_stats=fusion_stats)
    best_rows = build_best_subsets(subset_rows, min_support=args.min_support, top_n=args.top_n)
    pattern_rows = build_factor_pattern_summary(subset_rows, min_support=args.min_support)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "viewpoint_inventory.csv", inventory_rows)
    write_csv(output_dir / "subset_scores.csv", subset_rows)
    write_csv(output_dir / "best_subsets_by_k.csv", best_rows)
    write_csv(output_dir / "factor_pattern_summary.csv", pattern_rows)

    viewpoints = [row["viewpoint"] for row in inventory_rows]
    single_rows = [row for row in subset_rows if int(row["drone_count"]) == 1]
    pair_rows = [row for row in subset_rows if int(row["drone_count"]) == 2]

    strict_matrix = write_pair_matrix(
        output_dir / "pair_matrix_strict_quality.csv",
        viewpoints,
        pair_rows,
        single_rows,
        metric="mean_best_strict_quality",
    )
    complementarity_matrix = write_pair_matrix(
        output_dir / "pair_matrix_complementarity.csv",
        viewpoints,
        pair_rows,
        single_rows,
        metric="complementarity_vs_best_single_strict_quality",
    )
    support_matrix = write_pair_matrix(
        output_dir / "pair_matrix_support.csv",
        viewpoints,
        pair_rows,
        single_rows,
        metric="scene_count",
        diagonal_metric="scene_count",
    )
    fusion_matrix = write_pair_matrix(
        output_dir / "pair_matrix_support_weighted_fusion.csv",
        viewpoints,
        pair_rows,
        single_rows,
        metric="fusion_mean_fused_quality_support_weighted_or",
        diagonal_metric="mean_best_strict_quality",
    )

    plot_matrix(
        strict_matrix,
        viewpoints,
        plots_dir / "pair_matrix_strict_quality.png",
        "Pair matrix: mean best strict target quality",
        "Strict quality",
        cmap="viridis",
    )
    plot_matrix(
        complementarity_matrix,
        viewpoints,
        plots_dir / "pair_matrix_complementarity.png",
        "Pair matrix: complementarity over best single",
        "Strict-quality gain",
        cmap="coolwarm",
    )
    plot_matrix(
        support_matrix,
        viewpoints,
        plots_dir / "pair_matrix_support.png",
        "Pair matrix: matched scene support",
        "Scene count",
        cmap="magma",
    )
    if not np.isnan(fusion_matrix).all():
        plot_matrix(
            fusion_matrix,
            viewpoints,
            plots_dir / "pair_matrix_support_weighted_fusion.png",
            "Pair matrix: support-weighted late-fusion quality",
            "Fusion quality",
            cmap="viridis",
        )
    plot_gain_curve(subset_rows, plots_dir / "subset_size_gain_curve.png", min_support=args.min_support)
    plot_top_subsets(subset_rows, plots_dir / "top_subsets_by_k.png", min_support=args.min_support, top_n=min(args.top_n, 15))
    plot_factor_patterns(pattern_rows, plots_dir / "factor_pattern_summary.png")

    write_report(
        output_dir / "viewpoint_subset_matrix_report.md",
        records=records,
        subset_rows=subset_rows,
        best_rows=best_rows,
        min_support=args.min_support,
    )

    print(f"Wrote viewpoint subset matrix analysis to {output_dir}")
    print(f"Subset rows: {len(subset_rows)}")
    print(f"Best-subset rows: {len(best_rows)}")


if __name__ == "__main__":
    main()
