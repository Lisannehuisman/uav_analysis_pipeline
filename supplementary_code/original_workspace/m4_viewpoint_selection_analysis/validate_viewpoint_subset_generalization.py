from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_viewpoint_subset_matrix import (
    DEFAULT_SCENE_RECORDS,
    WORKSPACE,
    ViewRecord,
    accumulate_combo,
    aggregate_subset_scores,
    build_scene_groups,
    combo_label,
    finalize_accumulator,
    mean,
    new_accumulator,
    parse_float,
    parse_int,
    read_scene_records,
    safe_divide,
    subset_geometry,
    viewpoint_sort_key,
    write_csv,
)


DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_viewpoint_selection_analysis" / "outputs" / "generalization"
DEFAULT_DATASET_AUDIT_DIR = WORKSPACE / "outputs" / "thesis_tools" / "dataset_structure_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate viewpoint subset selection with repeated scene-split "
            "held-out evaluation and audit target-absent false alarms."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--dataset-audit-dir", default=str(DEFAULT_DATASET_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--max-k", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument(
        "--selection-metric",
        default="mean_best_strict_quality",
        choices=[
            "mean_best_strict_quality",
            "mean_best_target_ap50_95",
            "target_found_or_rate",
            "fusion_mean_fused_quality_support_weighted_or",
        ],
    )
    return parser.parse_args()


def scene_target_class(records: list[ViewRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in records:
        out.setdefault(record.scene_key, record.target_class)
    return out


def stratified_scene_folds(
    records: list[ViewRecord],
    folds: int,
    seed: int,
) -> list[set[str]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for scene_key, target_class in scene_target_class(records).items():
        by_class[target_class].append(scene_key)

    fold_sets = [set() for _ in range(folds)]
    for target_class in sorted(by_class):
        scenes = sorted(by_class[target_class])
        rng.shuffle(scenes)
        for index, scene_key in enumerate(scenes):
            fold_sets[index % folds].add(scene_key)
    return fold_sets


def filter_records_by_scenes(records: list[ViewRecord], scenes: set[str]) -> list[ViewRecord]:
    return [record for record in records if record.scene_key in scenes]


def empty_combo_eval(key: tuple[str, ...]) -> dict[str, object]:
    return {
        "drone_count": len(key),
        "combination_label": combo_label(key),
        "viewpoint_1": key[0] if len(key) > 0 else "",
        "viewpoint_2": key[1] if len(key) > 1 else "",
        "viewpoint_3": key[2] if len(key) > 2 else "",
        "sample_count": 0,
        "scene_count": 0,
        "dominant_target_class": "",
        "dominant_target_class_count": 0,
        **subset_geometry(key),
        "target_visible_any_rate": 0.0,
        "target_visible_all_rate": 0.0,
        "target_found_or_rate": 0.0,
        "target_found_majority_rate": 0.0,
        "target_found_all_rate": 0.0,
        "mean_best_target_ap50_95": 0.0,
        "mean_mean_target_ap50_95": 0.0,
        "mean_best_strict_quality": 0.0,
        "mean_mean_strict_quality": 0.0,
        "mean_best_match_confidence_iou50": 0.0,
        "mean_mean_match_confidence_iou50": 0.0,
        "best_constituent_mean_strict_quality_on_matched_scenes": 0.0,
        "complementarity_vs_best_single_strict_quality": 0.0,
        "best_constituent_mean_ap50_95_on_matched_scenes": 0.0,
        "complementarity_vs_best_single_ap50_95": 0.0,
        "mean_absent_view_count": 0.0,
        "selected_has_absent_view_rate": 0.0,
        "all_selected_views_absent_rate": 0.0,
        "false_alarm_rate_when_any_selected_view_absent": 0.0,
        "false_alarm_rate_when_all_selected_views_absent": 0.0,
    }


def evaluate_combo(records: list[ViewRecord], key: tuple[str, ...]) -> dict[str, object]:
    scene_groups = build_scene_groups(records)
    acc = new_accumulator(key)
    for scene_records in scene_groups.values():
        by_viewpoint = {record.viewpoint: record for record in scene_records}
        if not all(viewpoint in by_viewpoint for viewpoint in key):
            continue
        combo = tuple(by_viewpoint[viewpoint] for viewpoint in key)
        accumulate_combo(acc, combo)
    if int(acc["sample_count"]) == 0:
        return empty_combo_eval(key)
    return finalize_accumulator(acc, fusion_stats=None)


def combo_key_from_row(row: dict[str, object]) -> tuple[str, ...]:
    viewpoints = [str(row[f"viewpoint_{index}"]) for index in [1, 2, 3] if row.get(f"viewpoint_{index}")]
    return tuple(sorted(viewpoints, key=viewpoint_sort_key))


def metric_value(row: dict[str, object], metric: str) -> float:
    value = row.get(metric, 0.0)
    if value == "":
        return 0.0
    return float(value)


def select_top_subsets(
    train_records: list[ViewRecord],
    max_k: int,
    top_n: int,
    min_support: int,
    metric: str,
) -> dict[int, list[dict[str, object]]]:
    rows = aggregate_subset_scores(train_records, max_k=max_k, fusion_stats=None)
    selected: dict[int, list[dict[str, object]]] = {}
    for drone_count in range(1, max_k + 1):
        members = [
            row
            for row in rows
            if int(row["drone_count"]) == drone_count and int(row["scene_count"]) >= min_support
        ]
        members.sort(
            key=lambda row: (metric_value(row, metric), int(row["scene_count"])),
            reverse=True,
        )
        selected[drone_count] = members[:top_n]
    return selected


def run_scene_split_validation(
    records: list[ViewRecord],
    folds: int,
    repeats: int,
    seed: int,
    max_k: int,
    top_n: int,
    min_support: int,
    metric: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_scenes = set(scene_target_class(records))
    for repeat_index in range(repeats):
        fold_sets = stratified_scene_folds(records, folds=folds, seed=seed + repeat_index)
        for fold_index, heldout_scenes in enumerate(fold_sets):
            train_scenes = all_scenes.difference(heldout_scenes)
            train_records = filter_records_by_scenes(records, train_scenes)
            heldout_records = filter_records_by_scenes(records, heldout_scenes)
            selected = select_top_subsets(
                train_records=train_records,
                max_k=max_k,
                top_n=top_n,
                min_support=min_support,
                metric=metric,
            )

            for drone_count in range(1, max_k + 1):
                for rank, train_row in enumerate(selected.get(drone_count, []), start=1):
                    key = combo_key_from_row(train_row)
                    heldout_row = evaluate_combo(heldout_records, key)
                    rows.append(
                        {
                            "repeat": repeat_index + 1,
                            "fold": fold_index + 1,
                            "drone_count": drone_count,
                            "selected_rank": rank,
                            "combination_label": combo_label(key),
                            "viewpoint_1": key[0] if len(key) > 0 else "",
                            "viewpoint_2": key[1] if len(key) > 1 else "",
                            "viewpoint_3": key[2] if len(key) > 2 else "",
                            "selection_scene_count": int(train_row["scene_count"]),
                            "heldout_scene_count": int(heldout_row["scene_count"]),
                            "selection_metric_name": metric,
                            "selection_metric_value": metric_value(train_row, metric),
                            "selection_mean_best_strict_quality": float(train_row["mean_best_strict_quality"]),
                            "heldout_mean_best_strict_quality": float(heldout_row["mean_best_strict_quality"]),
                            "generalization_gap_strict_quality": float(train_row["mean_best_strict_quality"])
                            - float(heldout_row["mean_best_strict_quality"]),
                            "selection_mean_best_target_ap50_95": float(train_row["mean_best_target_ap50_95"]),
                            "heldout_mean_best_target_ap50_95": float(heldout_row["mean_best_target_ap50_95"]),
                            "selection_target_found_or_rate": float(train_row["target_found_or_rate"]),
                            "heldout_target_found_or_rate": float(heldout_row["target_found_or_rate"]),
                            "heldout_selected_has_absent_view_rate": float(
                                heldout_row["selected_has_absent_view_rate"]
                            ),
                            "heldout_false_alarm_rate_when_any_selected_view_absent": float(
                                heldout_row["false_alarm_rate_when_any_selected_view_absent"]
                            ),
                            "heldout_all_selected_views_absent_rate": float(
                                heldout_row["all_selected_views_absent_rate"]
                            ),
                        }
                    )
    return rows


def summarize_generalization_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for drone_count in sorted({int(row["drone_count"]) for row in rows}):
        members = [row for row in rows if int(row["drone_count"]) == drone_count]
        out.append(
            {
                "drone_count": drone_count,
                "trial_count": len(members),
                "mean_selection_scene_count": mean(parse_int(row["selection_scene_count"]) for row in members),
                "mean_heldout_scene_count": mean(parse_int(row["heldout_scene_count"]) for row in members),
                "mean_selection_strict_quality": mean(
                    parse_float(row["selection_mean_best_strict_quality"]) for row in members
                ),
                "mean_heldout_strict_quality": mean(
                    parse_float(row["heldout_mean_best_strict_quality"]) for row in members
                ),
                "mean_generalization_gap_strict_quality": mean(
                    parse_float(row["generalization_gap_strict_quality"]) for row in members
                ),
                "median_generalization_gap_strict_quality": float(
                    np.median([parse_float(row["generalization_gap_strict_quality"]) for row in members])
                ),
                "mean_selection_target_found_or_rate": mean(
                    parse_float(row["selection_target_found_or_rate"]) for row in members
                ),
                "mean_heldout_target_found_or_rate": mean(
                    parse_float(row["heldout_target_found_or_rate"]) for row in members
                ),
                "mean_heldout_false_alarm_rate_when_any_selected_view_absent": mean(
                    parse_float(row["heldout_false_alarm_rate_when_any_selected_view_absent"])
                    for row in members
                ),
            }
        )
    return out


def summarize_selection_frequency(
    rows: list[dict[str, object]],
    repeats: int,
    folds: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (int(row["drone_count"]), str(row["combination_label"]))
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    denominator = repeats * folds
    for (drone_count, label), members in grouped.items():
        first = members[0]
        out.append(
            {
                "drone_count": drone_count,
                "combination_label": label,
                "viewpoint_1": first.get("viewpoint_1", ""),
                "viewpoint_2": first.get("viewpoint_2", ""),
                "viewpoint_3": first.get("viewpoint_3", ""),
                "selection_count": len(members),
                "selection_fraction_of_scene_splits": safe_divide(len(members), denominator),
                "mean_selected_rank": mean(parse_int(row["selected_rank"]) for row in members),
                "mean_selection_metric_value": mean(parse_float(row["selection_metric_value"]) for row in members),
                "mean_selection_strict_quality": mean(
                    parse_float(row["selection_mean_best_strict_quality"]) for row in members
                ),
                "mean_heldout_strict_quality": mean(
                    parse_float(row["heldout_mean_best_strict_quality"]) for row in members
                ),
                "mean_generalization_gap_strict_quality": mean(
                    parse_float(row["generalization_gap_strict_quality"]) for row in members
                ),
                "mean_heldout_target_found_or_rate": mean(
                    parse_float(row["heldout_target_found_or_rate"]) for row in members
                ),
                "mean_heldout_scene_count": mean(parse_int(row["heldout_scene_count"]) for row in members),
            }
        )

    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -int(row["selection_count"]),
            -float(row["mean_heldout_strict_quality"]),
            str(row["combination_label"]),
        )
    )
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_generalization_readiness(records: list[ViewRecord], audit_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scene_count = len({record.scene_key for record in records})
    viewpoint_count = len({record.viewpoint for record in records})
    absent_count = sum(1 for record in records if record.target_visible == 0)
    rows.extend(
        [
            {
                "scope": "cached_m4_test_records",
                "item": "record_count",
                "value": len(records),
                "interpretation": "Cached fixed-detector M4 test image/view rows available.",
            },
            {
                "scope": "cached_m4_test_records",
                "item": "scene_count",
                "value": scene_count,
                "interpretation": "Scene keys available for scene-split held-out validation.",
            },
            {
                "scope": "cached_m4_test_records",
                "item": "viewpoint_count",
                "value": viewpoint_count,
                "interpretation": "Absolute viewpoint tokens available in the M4 test cache.",
            },
            {
                "scope": "cached_m4_test_records",
                "item": "target_absent_view_count",
                "value": absent_count,
                "interpretation": "Views where the filename target has no ground-truth box.",
            },
            {
                "scope": "cached_m4_test_records",
                "item": "target_absent_view_fraction",
                "value": f"{safe_divide(absent_count, len(records)):.6f}",
                "interpretation": "Sparse absence signal; useful for an audit, not a large negative benchmark.",
            },
        ]
    )

    split_rows = read_csv_rows(audit_dir / "split_summary.csv")
    for row in split_rows:
        rows.append(
            {
                "scope": "dataset_structure_audit",
                "item": f"{row.get('split', '')}_image_count",
                "value": row.get("image_count", ""),
                "interpretation": "Read from outputs/thesis_tools/dataset_structure_audit/split_summary.csv.",
            }
        )

    overlap_rows = read_csv_rows(audit_dir / "instance_split_overlap.csv")
    if overlap_rows:
        split_count_counter = Counter(parse_int(row.get("split_count", 0)) for row in overlap_rows)
        total_instances = len(overlap_rows)
        all_three = split_count_counter.get(3, 0)
        rows.append(
            {
                "scope": "dataset_structure_audit",
                "item": "object_instances_in_overlap_table",
                "value": total_instances,
                "interpretation": "Unique filename-derived object instances in the audit table.",
            }
        )
        rows.append(
            {
                "scope": "dataset_structure_audit",
                "item": "object_instances_present_in_train_val_test",
                "value": all_three,
                "interpretation": (
                    "If high, the test split should be described as held-out images/views, "
                    "not true novel-object-instance generalization."
                ),
            }
        )
        rows.append(
            {
                "scope": "dataset_structure_audit",
                "item": "fraction_instances_present_in_train_val_test",
                "value": f"{safe_divide(all_three, total_instances):.6f}",
                "interpretation": "Use this to qualify seen/unseen claims.",
            }
        )

    for split_name, gt_path, pred_path in [
        (
            "test",
            WORKSPACE / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json",
            WORKSPACE
            / "outputs"
            / "detector_family_comparison"
            / "standardized_test_eval"
            / "predictions"
            / "YOLOv8l_M4_test_predictions.json",
        ),
        (
            "val",
            WORKSPACE / "outputs" / "detector_family_comparison" / "standardized_val_eval" / "ground_truth" / "M4_val_gt.json",
            WORKSPACE
            / "outputs"
            / "detector_family_comparison"
            / "standardized_val_eval"
            / "predictions"
            / "YOLOv8l_M4_val_predictions.json",
        ),
        (
            "train",
            WORKSPACE / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "ground_truth" / "M4_train_gt.json",
            WORKSPACE
            / "outputs"
            / "detector_family_comparison"
            / "standardized_train_eval"
            / "predictions"
            / "YOLOv8l_M4_train_predictions.json",
        ),
    ]:
        rows.append(
            {
                "scope": "cache_inventory",
                "item": f"{split_name}_gt_json_exists",
                "value": int(gt_path.is_file()),
                "interpretation": str(gt_path),
            }
        )
        rows.append(
            {
                "scope": "cache_inventory",
                "item": f"{split_name}_prediction_json_exists",
                "value": int(pred_path.is_file()),
                "interpretation": str(pred_path),
            }
        )

    return rows


def build_absence_audit(records: list[ViewRecord]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    absent_records = [record for record in records if record.target_visible == 0]
    groups: dict[tuple[str, str], list[ViewRecord]] = defaultdict(list)
    for record in absent_records:
        groups[("overall", "all")].append(record)
        groups[("target_class", record.target_class)].append(record)
        groups[("viewpoint", record.viewpoint)].append(record)
        groups[("elevation", record.elevation)].append(record)
        groups[("radius", record.radius)].append(record)
        groups[("azimuth", f"{record.azimuth:03d}")].append(record)

    audit_rows: list[dict[str, object]] = []
    for (group_type, group_value), members in groups.items():
        false_alarm_count = sum(1 for record in members if record.target_fp > 0)
        audit_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "target_absent_view_count": len(members),
                "false_alarm_view_count": false_alarm_count,
                "false_alarm_rate": safe_divide(false_alarm_count, len(members)),
                "mean_target_fp_when_absent": mean(record.target_fp for record in members),
            }
        )
    audit_rows.sort(
        key=lambda row: (
            str(row["group_type"]),
            -int(row["target_absent_view_count"]),
            -float(row["false_alarm_rate"]),
            str(row["group_value"]),
        )
    )

    example_rows = [
        {
            "scene_key": record.scene_key,
            "file_name": record.file_name,
            "image_id": record.image_id,
            "target_class": record.target_class,
            "viewpoint": record.viewpoint,
            "elevation": record.elevation,
            "radius": record.radius,
            "azimuth": record.azimuth,
            "target_fp": record.target_fp,
            "target_match_confidence_iou50": record.target_match_confidence_iou50,
        }
        for record in absent_records
        if record.target_fp > 0
    ]
    example_rows.sort(key=lambda row: (-int(row["target_fp"]), str(row["file_name"])))
    return audit_rows, example_rows


def plot_generalization_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(summary_rows))
    labels = [str(row["drone_count"]) for row in summary_rows]
    width = 0.36
    selection = [float(row["mean_selection_strict_quality"]) for row in summary_rows]
    heldout = [float(row["mean_heldout_strict_quality"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, selection, width=width, label="Selected scenes", color="#3b6ea8")
    ax.bar(x + width / 2, heldout, width=width, label="Held-out scenes", color="#d08c3f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of selected viewpoints")
    ax.set_ylabel("Mean best strict target quality")
    ax.set_title("Scene-split subset validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_selection_stability(frequency_rows: list[dict[str, object]], output_path: Path, top_n: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(
        frequency_rows,
        key=lambda row: (int(row["selection_count"]), float(row["mean_heldout_strict_quality"])),
        reverse=True,
    )[:top_n]
    labels = [f"k={row['drone_count']} | {row['combination_label']}" for row in members][::-1]
    values = [float(row["selection_fraction_of_scene_splits"]) for row in members][::-1]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4f8f6f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Selection frequency across scene splits")
    ax.set_title("Most stable recommended viewpoint subsets")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_absence_audit(absence_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_rows = [
        row
        for row in absence_rows
        if row["group_type"] == "target_class" and int(row["target_absent_view_count"]) > 0
    ]
    class_rows.sort(key=lambda row: float(row["false_alarm_rate"]), reverse=True)
    labels = [str(row["group_value"]) for row in class_rows][::-1]
    rates = [float(row["false_alarm_rate"]) for row in class_rows][::-1]
    counts = [int(row["target_absent_view_count"]) for row in class_rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, rates, color="#9b4d4d")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label} (n={count})" for label, count in zip(labels, counts)])
    ax.set_xlabel("False-alarm rate when filename target is absent")
    ax.set_title("Target-absent view audit by class")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def write_report(
    output_path: Path,
    readiness_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    frequency_rows: list[dict[str, object]],
    absence_rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_fraction = next(
        (
            row["value"]
            for row in readiness_rows
            if row["item"] == "fraction_instances_present_in_train_val_test"
        ),
        "unknown",
    )
    absent_overall = next(
        (row for row in absence_rows if row["group_type"] == "overall" and row["group_value"] == "all"),
        None,
    )

    lines = [
        "# Viewpoint Subset Generalization Report",
        "",
        "## Interpretation Boundary",
        "",
        "This script performs repeated scene-split validation inside the cached M4 test records. The held-out fold is unseen by the subset-selection step, but it should not be described as a guaranteed novel-object-instance split.",
        "",
        f"Dataset audit fraction of instances present in train, val, and test: `{overlap_fraction}`.",
        "",
        "## Scene-Split Result",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- "
            f"k=`{row['drone_count']}`: selected strict quality `{float(row['mean_selection_strict_quality']):.4f}`, "
            f"held-out strict quality `{float(row['mean_heldout_strict_quality']):.4f}`, "
            f"mean gap `{float(row['mean_generalization_gap_strict_quality']):.4f}`"
        )

    lines.extend(
        [
            "",
            "## Selection Frequency And Held-Out Support",
            "",
            "Rows with near-zero held-out support are evidence that exact fixed 2-view or 3-view combinations are sparse in the current cache. Treat those as a data-coverage limitation, not as deployable recommendations.",
            "",
        ]
    )
    for drone_count in sorted({int(row["drone_count"]) for row in frequency_rows}):
        members = [row for row in frequency_rows if int(row["drone_count"]) == drone_count][:5]
        lines.append(f"### k={drone_count}")
        if not members:
            lines.append("- No selected subsets.")
            lines.append("")
            continue
        for row in members:
            lines.append(
                "- "
                f"`{row['combination_label']}`: "
                f"selected in `{float(row['selection_fraction_of_scene_splits']):.3f}` of scene splits, "
                f"held-out strict quality `{float(row['mean_heldout_strict_quality']):.4f}`, "
                f"held-out support `{float(row['mean_heldout_scene_count']):.2f}` scenes"
            )
        lines.append("")

    lines.extend(["", "## Target-Absent Audit", ""])
    if absent_overall:
        lines.append(
            "- "
            f"Target-absent views: `{absent_overall['target_absent_view_count']}`; "
            f"false-alarm views: `{absent_overall['false_alarm_view_count']}`; "
            f"false-alarm rate `{float(absent_overall['false_alarm_rate']):.4f}`."
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `scene_split_generalization_trials.csv`",
            "- `scene_split_generalization_summary.csv`",
            "- `top_subset_selection_frequency.csv`",
            "- `recommended_subsets_scene_split.csv`",
            "- `generalization_readiness.csv`",
            "- `target_absent_viewpoint_audit.csv`",
            "- `target_absent_false_alarm_examples.csv`",
            "- `plots/scene_split_generalization_gap.png`",
            "- `plots/selection_stability_top_subsets.png`",
            "- `plots/target_absent_false_alarm_rates.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    records = read_scene_records(Path(args.scene_records))

    trial_rows = run_scene_split_validation(
        records=records,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.random_seed,
        max_k=args.max_k,
        top_n=args.top_n,
        min_support=args.min_support,
        metric=args.selection_metric,
    )
    summary_rows = summarize_generalization_trials(trial_rows)
    frequency_rows = summarize_selection_frequency(trial_rows, repeats=args.repeats, folds=args.folds)
    recommended_rows = sorted(
        frequency_rows,
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["selection_fraction_of_scene_splits"]),
            -float(row["mean_heldout_strict_quality"]),
            str(row["combination_label"]),
        ),
    )
    readiness_rows = build_generalization_readiness(records, Path(args.dataset_audit_dir))
    absence_rows, absence_examples = build_absence_audit(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scene_split_generalization_trials.csv", trial_rows)
    write_csv(output_dir / "scene_split_generalization_summary.csv", summary_rows)
    write_csv(output_dir / "top_subset_selection_frequency.csv", frequency_rows)
    write_csv(output_dir / "recommended_subsets_scene_split.csv", recommended_rows)
    write_csv(output_dir / "generalization_readiness.csv", readiness_rows)
    write_csv(output_dir / "target_absent_viewpoint_audit.csv", absence_rows)
    write_csv(output_dir / "target_absent_false_alarm_examples.csv", absence_examples)

    plot_generalization_summary(summary_rows, plots_dir / "scene_split_generalization_gap.png")
    plot_selection_stability(frequency_rows, plots_dir / "selection_stability_top_subsets.png")
    plot_absence_audit(absence_rows, plots_dir / "target_absent_false_alarm_rates.png")
    write_report(
        output_dir / "generalization_validation_report.md",
        readiness_rows=readiness_rows,
        summary_rows=summary_rows,
        frequency_rows=recommended_rows,
        absence_rows=absence_rows,
    )

    print(f"Wrote viewpoint subset generalization validation to {output_dir}")
    print(f"Scene-split trial rows: {len(trial_rows)}")
    print(f"Stable subset rows: {len(frequency_rows)}")


if __name__ == "__main__":
    main()
