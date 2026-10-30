from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY_CSV = ROOT / "geometry_ground_truth_analysis" / "outputs" / "view_geometry_table.csv"
DEFAULT_SCENE_RECORDS_CSV = ROOT / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = ROOT / "clutter_grouping_analysis" / "outputs"

SCENE_GROUP_ORDER = ["Q1 low clutter", "Q2 mid-low clutter", "Q3 mid-high clutter", "Q4 high clutter"]
VIEW_GROUP_ORDER = ["0", "1-4", "5-9", "10+"]
SCENE_GROUP_COLORS = {
    "Q1 low clutter": "#4c78a8",
    "Q2 mid-low clutter": "#f58518",
    "Q3 mid-high clutter": "#54a24b",
    "Q4 high clutter": "#e45756",
}

METRICS = [
    ("scene_mean_target_ap50_95", "Target AP50-95"),
    ("scene_mean_target_strict_quality_iou50", "Target Strict Quality"),
    ("scene_mean_target_match_confidence_iou50", "Target Match Confidence"),
    ("scene_mean_target_detected_rate", "Target Detected Rate"),
]


@dataclass(frozen=True)
class JoinedRecord:
    scene_key: str
    file_name: str
    target_class: str
    distractor_count: int
    num_label_boxes: int
    num_target_class_boxes: int
    target_visible: int
    target_detected: int
    target_ap50_95: float
    target_strict_quality_iou50: float
    target_match_confidence_iou50: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether clutter groupings correlate with and significantly affect "
            "target detection quality."
        )
    )
    parser.add_argument("--geometry-csv", default=str(DEFAULT_GEOMETRY_CSV))
    parser.add_argument("--scene-records-csv", default=str(DEFAULT_SCENE_RECORDS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int(raw: object) -> int:
    text = str(raw).strip()
    if not text:
        return 0
    return int(float(text))


def parse_float(raw: object) -> float:
    text = str(raw).strip()
    if not text:
        return 0.0
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return 0.0
    return float(text)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1))


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def p_to_text(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    if value < 1e-4:
        return "<0.0001"
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_geometry_index(path: Path) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    with ensure_file(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row["file_name"]).strip()
            num_label_boxes = parse_int(row["num_label_boxes"])
            num_target_class_boxes = parse_int(row["num_target_class_boxes"])
            distractor_count = max(0, num_label_boxes - num_target_class_boxes)
            index[file_name] = {
                "scene_key": row["scene_key"],
                "num_label_boxes": num_label_boxes,
                "num_target_class_boxes": num_target_class_boxes,
                "distractor_count": distractor_count,
            }
    return index


def join_records(geometry_csv: Path, scene_records_csv: Path) -> list[JoinedRecord]:
    geometry_index = read_geometry_index(geometry_csv)
    records: list[JoinedRecord] = []
    with ensure_file(scene_records_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row["file_name"]).strip()
            if file_name not in geometry_index:
                continue
            geometry = geometry_index[file_name]
            records.append(
                JoinedRecord(
                    scene_key=str(row["scene_key"]).strip(),
                    file_name=file_name,
                    target_class=str(row["target_class"]).strip(),
                    distractor_count=int(geometry["distractor_count"]),
                    num_label_boxes=int(geometry["num_label_boxes"]),
                    num_target_class_boxes=int(geometry["num_target_class_boxes"]),
                    target_visible=parse_int(row["target_visible"]),
                    target_detected=parse_int(row["target_detected"]),
                    target_ap50_95=parse_float(row["target_ap50_95"]),
                    target_strict_quality_iou50=parse_float(row["target_strict_quality_iou50"]),
                    target_match_confidence_iou50=parse_float(row["target_match_confidence_iou50"]),
                )
            )
    return records


def view_clutter_group(distractor_count: int) -> str:
    if distractor_count == 0:
        return "0"
    if 1 <= distractor_count <= 4:
        return "1-4"
    if 5 <= distractor_count <= 9:
        return "5-9"
    return "10+"


def build_joined_rows(records: list[JoinedRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "scene_key": record.scene_key,
                "file_name": record.file_name,
                "target_class": record.target_class,
                "distractor_count": record.distractor_count,
                "view_clutter_group": view_clutter_group(record.distractor_count),
                "num_label_boxes": record.num_label_boxes,
                "num_target_class_boxes": record.num_target_class_boxes,
                "target_visible": record.target_visible,
                "target_detected": record.target_detected,
                "target_ap50_95": record.target_ap50_95,
                "target_strict_quality_iou50": record.target_strict_quality_iou50,
                "target_match_confidence_iou50": record.target_match_confidence_iou50,
            }
        )
    return rows


def build_view_group_summary(records: list[JoinedRecord]) -> list[dict[str, object]]:
    groups: dict[str, list[JoinedRecord]] = defaultdict(list)
    for record in records:
        groups[view_clutter_group(record.distractor_count)].append(record)

    rows: list[dict[str, object]] = []
    for group in VIEW_GROUP_ORDER:
        members = groups.get(group, [])
        if not members:
            continue
        rows.append(
            {
                "view_clutter_group": group,
                "sample_count": len(members),
                "scene_count": len({member.scene_key for member in members}),
                "mean_distractor_count": mean([float(member.distractor_count) for member in members]),
                "mean_target_visible_rate": mean([float(member.target_visible) for member in members]),
                "mean_target_detected_rate": mean([float(member.target_detected) for member in members]),
                "mean_target_ap50_95": mean([member.target_ap50_95 for member in members]),
                "mean_target_strict_quality_iou50": mean(
                    [member.target_strict_quality_iou50 for member in members]
                ),
                "mean_target_match_confidence_iou50": mean(
                    [member.target_match_confidence_iou50 for member in members]
                ),
            }
        )
    return rows


def build_scene_rows(records: list[JoinedRecord]) -> list[dict[str, object]]:
    groups: dict[str, list[JoinedRecord]] = defaultdict(list)
    for record in records:
        groups[record.scene_key].append(record)

    rows: list[dict[str, object]] = []
    for scene_key, members in sorted(groups.items()):
        distractors = [float(member.distractor_count) for member in members]
        visible_members = [member for member in members if member.target_visible > 0]
        rows.append(
            {
                "scene_key": scene_key,
                "target_class": members[0].target_class,
                "view_count": len(members),
                "scene_mean_distractor_count": mean(distractors),
                "scene_median_distractor_count": median(distractors),
                "scene_max_distractor_count": max(int(value) for value in distractors),
                "scene_mean_target_visible_rate": mean([float(member.target_visible) for member in members]),
                "scene_mean_target_detected_rate": mean([float(member.target_detected) for member in members]),
                "scene_detected_given_visible": (
                    mean([float(member.target_detected) for member in visible_members]) if visible_members else 0.0
                ),
                "scene_mean_target_ap50_95": mean([member.target_ap50_95 for member in members]),
                "scene_mean_target_strict_quality_iou50": mean(
                    [member.target_strict_quality_iou50 for member in members]
                ),
                "scene_mean_target_match_confidence_iou50": mean(
                    [member.target_match_confidence_iou50 for member in members]
                ),
            }
        )
    return rows


def compute_scene_quantiles(scene_rows: list[dict[str, object]]) -> tuple[float, float, float]:
    values = [float(row["scene_mean_distractor_count"]) for row in scene_rows]
    q1, q2, q3 = np.quantile(np.array(values, dtype=float), [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def assign_scene_group(value: float, q1: float, q2: float, q3: float) -> str:
    if value <= q1:
        return "Q1 low clutter"
    if value <= q2:
        return "Q2 mid-low clutter"
    if value <= q3:
        return "Q3 mid-high clutter"
    return "Q4 high clutter"


def annotate_scene_groups(scene_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    q1, q2, q3 = compute_scene_quantiles(scene_rows)
    quantiles = {"q1": q1, "median": q2, "q3": q3}

    annotated: list[dict[str, object]] = []
    for row in scene_rows:
        copied = dict(row)
        copied["scene_clutter_group"] = assign_scene_group(
            float(copied["scene_mean_distractor_count"]),
            q1,
            q2,
            q3,
        )
        annotated.append(copied)
    return annotated, quantiles


def build_scene_group_summary(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scene_rows:
        groups[str(row["scene_clutter_group"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for group in SCENE_GROUP_ORDER:
        members = groups.get(group, [])
        if not members:
            continue
        summary_rows.append(
            {
                "scene_clutter_group": group,
                "scene_count": len(members),
                "view_count_total": sum(int(member["view_count"]) for member in members),
                "mean_scene_distractor_count": mean(
                    [float(member["scene_mean_distractor_count"]) for member in members]
                ),
                "min_scene_distractor_count": min(
                    float(member["scene_mean_distractor_count"]) for member in members
                ),
                "max_scene_distractor_count": max(
                    float(member["scene_mean_distractor_count"]) for member in members
                ),
                "mean_target_visible_rate": mean(
                    [float(member["scene_mean_target_visible_rate"]) for member in members]
                ),
                "mean_target_detected_rate": mean(
                    [float(member["scene_mean_target_detected_rate"]) for member in members]
                ),
                "mean_detected_given_visible": mean(
                    [float(member["scene_detected_given_visible"]) for member in members]
                ),
                "mean_target_ap50_95": mean(
                    [float(member["scene_mean_target_ap50_95"]) for member in members]
                ),
                "std_target_ap50_95": safe_std(
                    [float(member["scene_mean_target_ap50_95"]) for member in members]
                ),
                "mean_target_strict_quality_iou50": mean(
                    [float(member["scene_mean_target_strict_quality_iou50"]) for member in members]
                ),
                "std_target_strict_quality_iou50": safe_std(
                    [float(member["scene_mean_target_strict_quality_iou50"]) for member in members]
                ),
                "mean_target_match_confidence_iou50": mean(
                    [float(member["scene_mean_target_match_confidence_iou50"]) for member in members]
                ),
                "std_target_match_confidence_iou50": safe_std(
                    [float(member["scene_mean_target_match_confidence_iou50"]) for member in members]
                ),
            }
        )
    return summary_rows


def cliffs_delta(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return 0.0
    greater = 0
    lower = 0
    for left in first:
        for right in second:
            if left > right:
                greater += 1
            elif left < right:
                lower += 1
    return float((greater - lower) / (len(first) * len(second)))


def holm_adjust(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: float(item[1]["p_value"]))

    adjusted: dict[int, float] = {}
    running_max = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(indexed, start=1):
        raw = float(row["p_value"])
        candidate = (total - rank + 1) * raw
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["p_value_holm"] = adjusted[index]
        copied["significant_holm_0_05"] = int(adjusted[index] < 0.05)
        output.append(copied)
    return output


def build_correlation_rows(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    x_values = [float(row["scene_mean_distractor_count"]) for row in scene_rows]
    rows: list[dict[str, object]] = []
    for metric_key, metric_label in METRICS:
        y_values = [float(row[metric_key]) for row in scene_rows]
        rho, p_value = spearmanr(x_values, y_values)
        rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "scene_count": len(scene_rows),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "significant_0_05": int(float(p_value) < 0.05),
            }
        )
    return rows


def build_omnibus_rows(scene_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scene_rows:
        by_group[str(row["scene_clutter_group"])].append(row)

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for metric_key, metric_label in METRICS:
        ordered_groups: list[tuple[str, list[float]]] = []
        for group in SCENE_GROUP_ORDER:
            values = [float(row[metric_key]) for row in by_group.get(group, [])]
            if values:
                ordered_groups.append((group, values))
        statistic, p_value = kruskal(*[values for _, values in ordered_groups])
        omnibus_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "group_count": len(ordered_groups),
                "test_name": "Kruskal-Wallis",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant_0_05": int(float(p_value) < 0.05),
            }
        )

        raw_pairwise_rows: list[dict[str, object]] = []
        for (group_a, values_a), (group_b, values_b) in combinations(ordered_groups, 2):
            statistic_u, p_pair = mannwhitneyu(values_a, values_b, alternative="two-sided")
            raw_pairwise_rows.append(
                {
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(values_a),
                    "n_b": len(values_b),
                    "mean_a": mean(values_a),
                    "mean_b": mean(values_b),
                    "median_a": median(values_a),
                    "median_b": median(values_b),
                    "delta_mean_a_minus_b": mean(values_a) - mean(values_b),
                    "u_statistic": float(statistic_u),
                    "p_value": float(p_pair),
                    "cliffs_delta": cliffs_delta(values_a, values_b),
                }
            )
        pairwise_rows.extend(holm_adjust(raw_pairwise_rows))

    return omnibus_rows, pairwise_rows


def plot_clutter_distribution(records: list[JoinedRecord], scene_rows: list[dict[str, object]], output_path: Path) -> None:
    view_counts = [record.distractor_count for record in records]
    scene_counts = [float(row["scene_mean_distractor_count"]) for row in scene_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(view_counts, bins=range(0, max(view_counts) + 2), color="#4c78a8", alpha=0.85)
    axes[0].set_title("View-level distractor count distribution")
    axes[0].set_xlabel("Distractor count")
    axes[0].set_ylabel("View count")

    axes[1].hist(scene_counts, bins=16, color="#f58518", alpha=0.85)
    axes[1].set_title("Scene mean distractor distribution")
    axes[1].set_xlabel("Mean distractor count per scene")
    axes[1].set_ylabel("Scene count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scene_metric_scatter(
    scene_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x_values = np.array([float(row["scene_mean_distractor_count"]) for row in scene_rows], dtype=float)
    correlations = {str(row["metric_key"]): row for row in correlation_rows}

    for axis, (metric_key, metric_label) in zip(axes.flatten(), METRICS):
        y_values = np.array([float(row[metric_key]) for row in scene_rows], dtype=float)
        for group in SCENE_GROUP_ORDER:
            group_rows = [row for row in scene_rows if row["scene_clutter_group"] == group]
            if not group_rows:
                continue
            axis.scatter(
                [float(row["scene_mean_distractor_count"]) for row in group_rows],
                [float(row[metric_key]) for row in group_rows],
                s=34,
                alpha=0.82,
                label=group,
                color=SCENE_GROUP_COLORS[group],
                edgecolors="white",
                linewidths=0.45,
            )
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        axis.plot(x_line, slope * x_line + intercept, color="#222222", linestyle="--", linewidth=1.2)

        rho = float(correlations[metric_key]["spearman_rho"])
        p_value = float(correlations[metric_key]["p_value"])
        axis.set_title(metric_label)
        axis.set_xlabel("Scene mean distractor count")
        axis.set_ylabel(metric_label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.15)
        axis.text(
            0.03,
            0.06,
            f"Spearman rho = {rho:.3f}\np = {p_to_text(p_value)}",
            transform=axis.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scene_group_boxplots(scene_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    label_map = {
        "Q1 low clutter": "Q1",
        "Q2 mid-low clutter": "Q2",
        "Q3 mid-high clutter": "Q3",
        "Q4 high clutter": "Q4",
    }

    for axis, (metric_key, metric_label) in zip(axes.flatten(), METRICS):
        data = []
        tick_labels = []
        colors = []
        for group in SCENE_GROUP_ORDER:
            group_values = [
                float(row[metric_key]) for row in scene_rows if str(row["scene_clutter_group"]) == group
            ]
            if not group_values:
                continue
            data.append(group_values)
            tick_labels.append(label_map[group])
            colors.append(SCENE_GROUP_COLORS[group])

        boxplot = axis.boxplot(data, patch_artist=True, tick_labels=tick_labels, showfliers=True)
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        for median_line in boxplot["medians"]:
            median_line.set_color("#111111")
            median_line.set_linewidth(1.5)

        axis.set_title(metric_label)
        axis.set_xlabel("Scene clutter quartile")
        axis.set_ylabel(metric_label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_view_group_lines(view_summary_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    x_positions = np.arange(len(view_summary_rows))
    labels = [str(row["view_clutter_group"]) for row in view_summary_rows]

    plot_specs = [
        ("mean_target_ap50_95", "Mean target AP50-95", "#4c78a8"),
        ("mean_target_strict_quality_iou50", "Mean target strict quality", "#f58518"),
        ("mean_target_detected_rate", "Mean target detected rate", "#54a24b"),
    ]

    for axis, (metric_key, title, color) in zip(axes, plot_specs):
        values = [float(row[metric_key]) for row in view_summary_rows]
        axis.plot(x_positions, values, marker="o", linewidth=2.0, color=color)
        axis.set_title(title)
        axis.set_xlabel("View clutter group")
        axis.set_ylabel("Mean value")
        axis.set_xticks(x_positions, labels)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output_path: Path,
    records: list[JoinedRecord],
    scene_rows: list[dict[str, object]],
    quantiles: dict[str, float],
    view_summary_rows: list[dict[str, object]],
    scene_group_summary_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
    omnibus_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
) -> None:
    strongest_correlation = max(correlation_rows, key=lambda row: abs(float(row["spearman_rho"])))
    significant_omnibus = [row for row in omnibus_rows if int(row["significant_0_05"]) == 1]
    significant_pairwise = [row for row in pairwise_rows if int(row["significant_holm_0_05"]) == 1]

    lines = [
        "# Clutter Grouping Analysis",
        "",
        "## Purpose",
        "",
        "This analysis tests whether clutter groupings, operationalized as non-target objects in view, are associated with target detection quality and whether the clutter groups differ significantly.",
        "",
        "## Inputs",
        "",
        f"- Geometry table: `{DEFAULT_GEOMETRY_CSV}`",
        f"- Scene detection table: `{DEFAULT_SCENE_RECORDS_CSV}`",
        "",
        "## Operationalization",
        "",
        "- View-level clutter is defined as `num_label_boxes - num_target_class_boxes`.",
        "- View-level bins: `0`, `1-4`, `5-9`, `10+` distractors.",
        "- Scene-level clutter uses the mean distractor count across evaluated views of a scene.",
        "- Significance testing is done at the scene level to avoid treating multiple correlated views from the same scene as fully independent.",
        "",
        "## Scene clutter quartile cutoffs",
        "",
        f"- Q1 cutoff: `{fmt(quantiles['q1'])}` mean distractors",
        f"- Median cutoff: `{fmt(quantiles['median'])}` mean distractors",
        f"- Q3 cutoff: `{fmt(quantiles['q3'])}` mean distractors",
        "",
        "## Dataset coverage",
        "",
        f"- Joined evaluation views: `{len(records)}`",
        f"- Unique evaluated scenes: `{len(scene_rows)}`",
        "",
        "## Headline findings",
        "",
        (
            f"- Strongest scene-level monotonic association: `{strongest_correlation['metric_label']}` "
            f"with Spearman rho `{fmt(float(strongest_correlation['spearman_rho']), 3)}` and "
            f"p `{p_to_text(float(strongest_correlation['p_value']))}`."
        ),
    ]

    best_group = max(
        scene_group_summary_rows,
        key=lambda row: float(row["mean_target_ap50_95"]),
    )
    worst_group = min(
        scene_group_summary_rows,
        key=lambda row: float(row["mean_target_ap50_95"]),
    )
    lines.append(
        f"- Mean target AP50-95 falls from `{fmt(float(best_group['mean_target_ap50_95']))}` in `{best_group['scene_clutter_group']}` "
        f"to `{fmt(float(worst_group['mean_target_ap50_95']))}` in `{worst_group['scene_clutter_group']}`."
    )
    if significant_omnibus:
        lines.append(
            f"- `{len(significant_omnibus)}/{len(omnibus_rows)}` scene-level omnibus tests are significant at `p < 0.05`."
        )
    else:
        lines.append("- No scene-level omnibus test reached `p < 0.05`.")

    lines.extend(
        [
            "",
            "## View-level clutter summary",
            "",
        ]
    )
    for row in view_summary_rows:
        lines.append(
            f"- Group `{row['view_clutter_group']}`: `n={row['sample_count']}`, mean AP50-95 `{fmt(float(row['mean_target_ap50_95']))}`, "
            f"mean strict quality `{fmt(float(row['mean_target_strict_quality_iou50']))}`, "
            f"mean detected rate `{fmt(float(row['mean_target_detected_rate']))}`."
        )

    lines.extend(
        [
            "",
            "## Scene-level group summary",
            "",
        ]
    )
    for row in scene_group_summary_rows:
        lines.append(
            f"- `{row['scene_clutter_group']}`: `n={row['scene_count']}` scenes, mean clutter `{fmt(float(row['mean_scene_distractor_count']))}`, "
            f"mean AP50-95 `{fmt(float(row['mean_target_ap50_95']))}`, "
            f"mean strict quality `{fmt(float(row['mean_target_strict_quality_iou50']))}`, "
            f"mean detected rate `{fmt(float(row['mean_target_detected_rate']))}`."
        )

    lines.extend(
        [
            "",
            "## Correlations",
            "",
        ]
    )
    for row in correlation_rows:
        lines.append(
            f"- `{row['metric_label']}`: Spearman rho `{fmt(float(row['spearman_rho']), 3)}`, p `{p_to_text(float(row['p_value']))}`."
        )

    lines.extend(
        [
            "",
            "## Omnibus tests across scene clutter quartiles",
            "",
        ]
    )
    for row in omnibus_rows:
        lines.append(
            f"- `{row['metric_label']}`: Kruskal-Wallis H `{fmt(float(row['statistic']), 3)}`, p `{p_to_text(float(row['p_value']))}`."
        )

    lines.extend(
        [
            "",
            "## Pairwise scene-group differences after Holm correction",
            "",
        ]
    )
    if significant_pairwise:
        for row in sorted(
            significant_pairwise,
            key=lambda item: (str(item["metric_label"]), float(item["p_value_holm"])),
        ):
            lines.append(
                f"- `{row['metric_label']}`: `{row['group_a']}` vs `{row['group_b']}`, Holm-adjusted p `{p_to_text(float(row['p_value_holm']))}`, "
                f"mean delta `{fmt(float(row['delta_mean_a_minus_b']))}`, Cliff's delta `{fmt(float(row['cliffs_delta']), 3)}`."
            )
    else:
        lines.append("- No pairwise difference remained significant after Holm correction.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    geometry_csv = Path(args.geometry_csv).resolve()
    scene_records_csv = Path(args.scene_records_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    records = join_records(geometry_csv, scene_records_csv)
    joined_rows = build_joined_rows(records)
    view_summary_rows = build_view_group_summary(records)
    scene_rows_raw = build_scene_rows(records)
    scene_rows, quantiles = annotate_scene_groups(scene_rows_raw)
    scene_group_summary_rows = build_scene_group_summary(scene_rows)
    correlation_rows = build_correlation_rows(scene_rows)
    omnibus_rows, pairwise_rows = build_omnibus_rows(scene_rows)

    write_csv(output_dir / "joined_view_records.csv", joined_rows)
    write_csv(output_dir / "view_clutter_group_summary.csv", view_summary_rows)
    write_csv(output_dir / "scene_clutter_summary.csv", scene_rows)
    write_csv(output_dir / "scene_clutter_group_summary.csv", scene_group_summary_rows)
    write_csv(
        output_dir / "scene_clutter_quantiles.csv",
        [
            {
                "q1_scene_mean_distractor_count": quantiles["q1"],
                "median_scene_mean_distractor_count": quantiles["median"],
                "q3_scene_mean_distractor_count": quantiles["q3"],
            }
        ],
    )
    write_csv(output_dir / "scene_clutter_correlations.csv", correlation_rows)
    write_csv(output_dir / "scene_clutter_omnibus_tests.csv", omnibus_rows)
    write_csv(output_dir / "scene_clutter_pairwise_tests.csv", pairwise_rows)

    plot_clutter_distribution(records, scene_rows, output_dir / "clutter_distribution.png")
    plot_scene_metric_scatter(scene_rows, correlation_rows, output_dir / "scene_metric_scatter.png")
    plot_scene_group_boxplots(scene_rows, output_dir / "scene_group_boxplots.png")
    plot_view_group_lines(view_summary_rows, output_dir / "view_group_metric_lines.png")

    build_report(
        output_dir / "analysis_report.md",
        records,
        scene_rows,
        quantiles,
        view_summary_rows,
        scene_group_summary_rows,
        correlation_rows,
        omnibus_rows,
        pairwise_rows,
    )

    print(f"Saved clutter grouping analysis to: {output_dir}")


if __name__ == "__main__":
    main()
