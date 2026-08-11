from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = ROOT / "m4_two_drone_operational_analysis"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from analyze_two_drone_operational import (  # noqa: E402
    DEFAULT_GT,
    DEFAULT_PRED,
    ViewRecord,
    azimuth_gap,
    build_scene_groups,
    build_view_records,
    mean,
    safe_divide,
)


DEFAULT_OUTPUT_DIR = Path("m4_cross_view_box_fusion_analysis") / "outputs"
ELEVATION_ORDER = {"low": 0, "mid": 1, "high": 2}
RADIUS_ORDER = {"near": 0, "mid": 1, "far": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative cross-view matched-box fusion analysis for the fixed YOLOv8l M4 detector."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT), help="COCO ground-truth JSON for the M4 test split.")
    parser.add_argument("--pred-json", default=str(DEFAULT_PRED), help="COCO prediction JSON for the fixed full-M4 detector.")
    parser.add_argument("--score-threshold", type=float, default=0.001, help="Prediction score threshold applied before matching.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSVs, plots, and the report.")
    parser.add_argument("--min-pair-support", type=int, default=8, help="Minimum support for headline pair summaries.")
    parser.add_argument("--min-triple-support", type=int, default=6, help="Minimum support for headline triple summaries.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path)


def require_existing_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def normalize_pattern(values: list[str], order_map: dict[str, int]) -> str:
    return " + ".join(sorted(values, key=lambda value: (order_map.get(value, 99), value)))


def combo_features(combo: tuple[ViewRecord, ...]) -> dict[str, object]:
    matched = [row for row in combo if row.target_detected]
    matched_conf = [float(row.target_match_confidence_iou50) for row in matched]
    matched_iou = [float(row.target_match_iou_at_confidence_iou50) for row in matched]
    matched_quality = [float(row.target_strict_quality_iou50) for row in matched]

    drone_count = len(combo)
    elevations = [str(row.elevation) for row in combo]
    radii = [str(row.radius) for row in combo]
    azimuths = [int(row.azimuth) for row in combo]

    support_count = len(matched)
    support_ratio = safe_divide(support_count, drone_count)
    noisy_or_conf = noisy_or(matched_conf)
    best_iou = max(matched_iou) if matched_iou else 0.0
    mean_iou = mean(matched_iou) if matched_iou else 0.0

    return {
        "scene_key": combo[0].scene_key,
        "target_class": combo[0].target_class,
        "drone_count": drone_count,
        "combination_label": " + ".join(row.viewpoint for row in combo),
        "viewpoint_1": combo[0].viewpoint,
        "viewpoint_2": combo[1].viewpoint if drone_count >= 2 else "",
        "viewpoint_3": combo[2].viewpoint if drone_count >= 3 else "",
        "elevation_pattern": normalize_pattern(elevations, ELEVATION_ORDER),
        "radius_pattern": normalize_pattern(radii, RADIUS_ORDER),
        "azimuth_gap_max": max((azimuth_gap(a, b) for a, b in combinations(azimuths, 2)), default=0),
        "support_count": support_count,
        "support_ratio": support_ratio,
        "mean_target_ap50_95": mean(float(row.target_ap50_95) for row in combo),
        "best_target_ap50_95": max(float(row.target_ap50_95) for row in combo),
        "best_box_quality": max(matched_quality) if matched_quality else 0.0,
        "best_box_confidence": max(matched_conf) if matched_conf else 0.0,
        "best_box_iou": best_iou,
        "noisy_or_confidence": noisy_or_conf,
        "fused_quality_noisy_or_max_iou": noisy_or_conf * best_iou,
        "fused_quality_support_weighted_or": noisy_or_conf * mean_iou * support_ratio,
        "mean_matched_iou": mean_iou,
        "max_matched_iou": best_iou,
    }


def build_combo_rows(scene_groups: dict[str, list[ViewRecord]], drone_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for members in scene_groups.values():
        if len(members) < drone_count:
            continue
        for combo in combinations(members, drone_count):
            rows.append(combo_features(combo))
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_overall(rows: list[dict[str, object]], drone_count: int) -> list[dict[str, object]]:
    return [
        {
            "drone_count": drone_count,
            "policy_id": "best_box",
            "policy_label": "Best matched box",
            "sample_count": len(rows),
            "mean_quality": mean(float(row["best_box_quality"]) for row in rows),
            "median_quality": float(np.median([float(row["best_box_quality"]) for row in rows])),
        },
        {
            "drone_count": drone_count,
            "policy_id": "noisy_or_max_iou",
            "policy_label": "Noisy-OR confidence + best IoU",
            "sample_count": len(rows),
            "mean_quality": mean(float(row["fused_quality_noisy_or_max_iou"]) for row in rows),
            "median_quality": float(np.median([float(row["fused_quality_noisy_or_max_iou"]) for row in rows])),
        },
        {
            "drone_count": drone_count,
            "policy_id": "support_weighted_or",
            "policy_label": "Support-weighted noisy-OR fusion",
            "sample_count": len(rows),
            "mean_quality": mean(float(row["fused_quality_support_weighted_or"]) for row in rows),
            "median_quality": float(np.median([float(row["fused_quality_support_weighted_or"]) for row in rows])),
        },
    ]


def summarize_by_combination(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["combination_label"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for combination_label, members in sorted(grouped.items()):
        summary_rows.append(
            {
                "combination_label": combination_label,
                "viewpoint_1": members[0]["viewpoint_1"],
                "viewpoint_2": members[0]["viewpoint_2"],
                "viewpoint_3": members[0]["viewpoint_3"],
                "drone_count": members[0]["drone_count"],
                "sample_count": len(members),
                "mean_best_box_quality": mean(float(row["best_box_quality"]) for row in members),
                "mean_noisy_or_max_iou": mean(float(row["fused_quality_noisy_or_max_iou"]) for row in members),
                "mean_support_weighted_or": mean(float(row["fused_quality_support_weighted_or"]) for row in members),
                "mean_support_ratio": mean(float(row["support_ratio"]) for row in members),
                "mean_target_ap50_95": mean(float(row["best_target_ap50_95"]) for row in members),
                "mean_azimuth_gap_max": mean(float(row["azimuth_gap_max"]) for row in members),
                "elevation_pattern": members[0]["elevation_pattern"],
                "radius_pattern": members[0]["radius_pattern"],
            }
        )
    return summary_rows


def summarize_pair_patterns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["elevation_pattern"]), int(row["azimuth_gap_max"]))].append(row)

    summary: list[dict[str, object]] = []
    for (elevation_pattern, azimuth_gap_max), members in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary.append(
            {
                "elevation_pattern": elevation_pattern,
                "azimuth_gap_max": azimuth_gap_max,
                "sample_count": len(members),
                "mean_best_box_quality": mean(float(row["best_box_quality"]) for row in members),
                "mean_support_weighted_or": mean(float(row["fused_quality_support_weighted_or"]) for row in members),
                "mean_gain_vs_best_box": mean(
                    float(row["fused_quality_support_weighted_or"]) - float(row["best_box_quality"]) for row in members
                ),
            }
        )
    return summary


def plot_overall_quality(overall_rows: list[dict[str, object]], output_path: Path) -> None:
    policy_order = ["best_box", "noisy_or_max_iou", "support_weighted_or"]
    policy_labels = {
        "best_box": "Best box",
        "noisy_or_max_iou": "Noisy-OR + best IoU",
        "support_weighted_or": "Support-weighted",
    }
    color_map = {
        2: "#1f77b4",
        3: "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = np.arange(len(policy_order))
    for drone_count in sorted({int(row["drone_count"]) for row in overall_rows}):
        values = []
        for policy_id in policy_order:
            row = next(item for item in overall_rows if int(item["drone_count"]) == drone_count and item["policy_id"] == policy_id)
            values.append(float(row["mean_quality"]))
        ax.plot(x, values, marker="o", linewidth=2.4, markersize=8, color=color_map[drone_count], label=f"{drone_count} views")

    ax.set_xticks(x, [policy_labels[key] for key in policy_order])
    ax.set_ylabel("Mean fused target strict quality")
    ax.set_title("Cross-view matched-box fusion quality by policy")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    gains_noisy = [float(row["fused_quality_noisy_or_max_iou"]) - float(row["best_box_quality"]) for row in rows]
    gains_support = [float(row["fused_quality_support_weighted_or"]) - float(row["best_box_quality"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    bins = np.linspace(min(gains_noisy + gains_support), max(gains_noisy + gains_support), 30)

    axes[0].hist(gains_noisy, bins=bins, color="#4c78a8", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("2-view gain: noisy-OR + best IoU vs best box")
    axes[0].set_xlabel("Fusion gain in strict quality")
    axes[0].set_ylabel("Number of scene-level pairs")

    axes[1].hist(gains_support, bins=bins, color="#f58518", alpha=0.85)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("2-view gain: support-weighted fusion vs best box")
    axes[1].set_xlabel("Fusion gain in strict quality")
    axes[1].set_ylabel("Number of scene-level pairs")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_combinations(rows: list[dict[str, object]], score_key: str, output_path: Path, title: str, top_n: int) -> None:
    top_rows = sorted(rows, key=lambda row: (float(row[score_key]), float(row["mean_support_ratio"])), reverse=True)[:top_n]
    labels = [str(row["combination_label"]) for row in top_rows][::-1]
    values = [float(row[score_key]) for row in top_rows][::-1]

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.42)), constrained_layout=True)
    ax.barh(labels, values, color="#2ca02c")
    ax.set_xlabel("Mean fused target strict quality")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_pattern_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    elevation_labels = sorted({str(row["elevation_pattern"]) for row in rows})
    gap_labels = [0, 45, 90, 135, 180]
    matrix = np.full((len(elevation_labels), len(gap_labels)), np.nan)

    elevation_lookup = {label: idx for idx, label in enumerate(elevation_labels)}
    gap_lookup = {gap: idx for idx, gap in enumerate(gap_labels)}
    for row in rows:
        gap = int(row["azimuth_gap_max"])
        if gap not in gap_lookup:
            continue
        matrix[elevation_lookup[str(row["elevation_pattern"])], gap_lookup[gap]] = float(row["mean_gain_vs_best_box"])

    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.05
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(gap_labels)), [str(g) for g in gap_labels])
    ax.set_yticks(range(len(elevation_labels)), elevation_labels)
    ax.set_xlabel("Maximum azimuth gap")
    ax.set_ylabel("Elevation pattern")
    ax.set_title("Where support-weighted fusion gains over simple best-box selection")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean gain vs best-box baseline")

    for r_idx in range(matrix.shape[0]):
        for c_idx in range(matrix.shape[1]):
            value = matrix[r_idx, c_idx]
            if math.isnan(value):
                continue
            text_color = "white" if abs(value) > 0.03 else "black"
            ax.text(c_idx, r_idx, f"{value:+.03f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    pair_overall: list[dict[str, object]],
    triple_overall: list[dict[str, object]],
    pair_summary: list[dict[str, object]],
    triple_summary: list[dict[str, object]],
) -> None:
    def get_row(rows: list[dict[str, object]], policy_id: str) -> dict[str, object]:
        return next(row for row in rows if row["policy_id"] == policy_id)

    pair_best = get_row(pair_overall, "best_box")
    pair_noisy = get_row(pair_overall, "noisy_or_max_iou")
    pair_support = get_row(pair_overall, "support_weighted_or")
    triple_best = get_row(triple_overall, "best_box")
    triple_noisy = get_row(triple_overall, "noisy_or_max_iou")
    triple_support = get_row(triple_overall, "support_weighted_or")

    best_pair_combo = max(pair_summary, key=lambda row: float(row["mean_support_weighted_or"]))
    best_triple_combo = max(triple_summary, key=lambda row: float(row["mean_support_weighted_or"]))

    lines = [
        "# Cross-View Matched-Box Fusion Report",
        "",
        "## What This Adds",
        "",
        "- This experiment keeps the full-viewpoint detector fixed and tests conservative late fusion at matched-target-box level.",
        "- It is more box-aware than the earlier `1-of-2` / `1-of-3` operational success rules, but it is still not geometric reprojection fusion.",
        "- Because the dataset lacks camera calibration and cross-view object IDs, the fusion is restricted to the intended target object and is ground-truth anchored.",
        "",
        "## 2-View Overall Quality",
        "",
        f"- Best-box baseline: `{float(pair_best['mean_quality']):.4f}`",
        f"- Noisy-OR + best IoU: `{float(pair_noisy['mean_quality']):.4f}`",
        f"- Support-weighted noisy-OR: `{float(pair_support['mean_quality']):.4f}`",
        f"- Noisy-OR gain vs best-box: `{float(pair_noisy['mean_quality']) - float(pair_best['mean_quality']):+.4f}`",
        f"- Support-weighted gain vs best-box: `{float(pair_support['mean_quality']) - float(pair_best['mean_quality']):+.4f}`",
        "",
        "## 3-View Overall Quality",
        "",
        f"- Best-box baseline: `{float(triple_best['mean_quality']):.4f}`",
        f"- Noisy-OR + best IoU: `{float(triple_noisy['mean_quality']):.4f}`",
        f"- Support-weighted noisy-OR: `{float(triple_support['mean_quality']):.4f}`",
        f"- Noisy-OR gain vs best-box: `{float(triple_noisy['mean_quality']) - float(triple_best['mean_quality']):+.4f}`",
        f"- Support-weighted gain vs best-box: `{float(triple_support['mean_quality']) - float(triple_best['mean_quality']):+.4f}`",
        "",
        "## Best Combinations Under Support-Weighted Fusion",
        "",
        f"- Best pair: `{best_pair_combo['combination_label']}` with mean fused quality `{float(best_pair_combo['mean_support_weighted_or']):.4f}`",
        f"- Best triple: `{best_triple_combo['combination_label']}` with mean fused quality `{float(best_triple_combo['mean_support_weighted_or']):.4f}`",
        "",
        "## Interpretation",
        "",
        "- If noisy-OR beats best-box, then there is usable cross-view confidence accumulation even without geometry.",
        "- If support-weighted fusion beats best-box, then some viewpoint combinations provide genuinely corroborating target evidence rather than just a single rescue view.",
        "- If support-weighted fusion loses to best-box, then the multiview gain is mostly from having a rescue view available, not from true agreement between matched boxes.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_run_summary(output_dir: Path, pair_overall: list[dict[str, object]], triple_overall: list[dict[str, object]]) -> None:
    def get_quality(rows: list[dict[str, object]], policy_id: str) -> float:
        return float(next(row["mean_quality"] for row in rows if row["policy_id"] == policy_id))

    pair_best = get_quality(pair_overall, "best_box")
    pair_support = get_quality(pair_overall, "support_weighted_or")
    triple_best = get_quality(triple_overall, "best_box")
    triple_support = get_quality(triple_overall, "support_weighted_or")

    print(f"Saved cross-view box fusion analysis to: {output_dir}")
    print(
        "2-view mean quality | "
        f"best-box={pair_best:.4f}, "
        f"support-weighted={pair_support:.4f}, "
        f"delta={pair_support - pair_best:+.4f}"
    )
    print(
        "3-view mean quality | "
        f"best-box={triple_best:.4f}, "
        f"support-weighted={triple_support:.4f}, "
        f"delta={triple_support - triple_best:+.4f}"
    )


def main() -> None:
    args = parse_args()
    gt_json = require_existing_file(resolve_project_path(args.gt_json), "Ground-truth JSON")
    pred_json = require_existing_file(resolve_project_path(args.pred_json), "Prediction JSON")
    output_dir = resolve_project_path(args.output_dir)
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    records = build_view_records(gt_json, pred_json, args.score_threshold)
    scene_groups = build_scene_groups(records)

    pair_combo_rows = build_combo_rows(scene_groups, drone_count=2)
    triple_combo_rows = build_combo_rows(scene_groups, drone_count=3)

    for row in pair_combo_rows:
        row["gain_noisy_vs_best"] = float(row["fused_quality_noisy_or_max_iou"]) - float(row["best_box_quality"])
        row["gain_support_vs_best"] = float(row["fused_quality_support_weighted_or"]) - float(row["best_box_quality"])
    for row in triple_combo_rows:
        row["gain_noisy_vs_best"] = float(row["fused_quality_noisy_or_max_iou"]) - float(row["best_box_quality"])
        row["gain_support_vs_best"] = float(row["fused_quality_support_weighted_or"]) - float(row["best_box_quality"])

    pair_overall = summarize_overall(pair_combo_rows, drone_count=2)
    triple_overall = summarize_overall(triple_combo_rows, drone_count=3)
    overall_rows = pair_overall + triple_overall

    pair_summary = summarize_by_combination(pair_combo_rows)
    triple_summary = summarize_by_combination(triple_combo_rows)
    pair_pattern_rows = summarize_pair_patterns(pair_combo_rows)

    filtered_pair_summary = [row for row in pair_summary if int(row["sample_count"]) >= args.min_pair_support]
    filtered_triple_summary = [row for row in triple_summary if int(row["sample_count"]) >= args.min_triple_support]

    write_csv_rows(output_dir / "pair_combo_rows.csv", pair_combo_rows)
    write_csv_rows(output_dir / "triple_combo_rows.csv", triple_combo_rows)
    write_csv_rows(output_dir / "policy_overall_summary.csv", overall_rows)
    write_csv_rows(output_dir / "pair_policy_summary.csv", pair_summary)
    write_csv_rows(output_dir / "triple_policy_summary.csv", triple_summary)
    write_csv_rows(output_dir / "pair_support_weighted_pattern_summary.csv", pair_pattern_rows)

    plot_overall_quality(overall_rows, plots_dir / "policy_overall_quality.png")
    plot_gain_distribution(pair_combo_rows, plots_dir / "pair_fusion_gain_distribution.png")
    plot_top_combinations(filtered_pair_summary, "mean_support_weighted_or", plots_dir / "top_pairs_support_weighted.png", "Top 2-view combinations by support-weighted fusion", 15)
    plot_top_combinations(filtered_triple_summary, "mean_support_weighted_or", plots_dir / "top_triples_support_weighted.png", "Top 3-view combinations by support-weighted fusion", 10)
    plot_pair_pattern_heatmap(pair_pattern_rows, plots_dir / "pair_support_weighted_pattern_heatmap.png")
    write_report(output_dir / "box_fusion_report.md", pair_overall, triple_overall, filtered_pair_summary or pair_summary, filtered_triple_summary or triple_summary)
    print_run_summary(output_dir, pair_overall, triple_overall)


if __name__ == "__main__":
    main()
