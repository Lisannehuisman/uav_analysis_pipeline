from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_swarm_viewpoint_analysis import (
    DEFAULT_SCENE_RECORDS,
    WORKSPACE,
    build_scene_groups,
    combo_observation,
    mean,
    read_scene_records,
    safe_divide,
    viewpoint_sort_key,
    write_csv,
)


DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_viewpoint_selection_analysis" / "outputs" / "robustness"

RELATIONSHIP_AXES = {
    "distance": "radius_relationship",
    "elevation": "elevation_relationship",
    "azimuth": "azimuth_relationship",
    "mixed_diversity": "diversity_type",
}

METRICS = {
    "AP50-95": "ap50_95",
    "AP50": "ap50",
    "strict_quality": "best_strict_quality",
    "precision": "precision",
    "recall": "recall",
    "F1": "f1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Robustness analysis for viewpoint-diversity conclusions: factor-level "
            "bootstrap confidence intervals plus matched-scene relationship comparisons."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-k", type=int, default=3, choices=[2, 3])
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-scenes", type=int, default=5)
    return parser.parse_args()


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_mean(values: list[float], iterations: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.array(values, dtype=float)
    if len(array) == 1:
        value = float(array[0])
        return value, value, value, 0.0
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    boot = array[indices].mean(axis=1)
    lower, upper = percentile_ci(boot)
    return float(array.mean()), lower, upper, float(array.std(ddof=0))


def bootstrap_paired_diff(
    left_by_scene: dict[str, float],
    right_by_scene: dict[str, float],
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    common_scenes = sorted(set(left_by_scene).intersection(right_by_scene))
    if not common_scenes:
        return {
            "common_scene_count": 0,
            "mean_difference": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "approx_two_sided_p": 1.0,
        }

    diffs = np.array([left_by_scene[scene] - right_by_scene[scene] for scene in common_scenes], dtype=float)
    if len(diffs) == 1:
        diff = float(diffs[0])
        p_value = 0.0 if diff != 0 else 1.0
        return {
            "common_scene_count": 1,
            "mean_difference": diff,
            "ci95_low": diff,
            "ci95_high": diff,
            "approx_two_sided_p": p_value,
        }

    indices = rng.integers(0, len(diffs), size=(iterations, len(diffs)))
    boot = diffs[indices].mean(axis=1)
    lower, upper = percentile_ci(boot)
    frac_le_zero = float(np.mean(boot <= 0))
    frac_ge_zero = float(np.mean(boot >= 0))
    p_value = min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero))
    return {
        "common_scene_count": len(common_scenes),
        "mean_difference": float(diffs.mean()),
        "ci95_low": lower,
        "ci95_high": upper,
        "approx_two_sided_p": p_value,
    }


def build_combo_observations(records, max_k: int) -> list[dict[str, object]]:
    scene_groups = build_scene_groups(records)
    rows: list[dict[str, object]] = []
    for scene_records in scene_groups.values():
        for drone_count in range(2, max_k + 1):
            if len(scene_records) < drone_count:
                continue
            for combo in combinations(scene_records, drone_count):
                row = combo_observation(combo)
                row["ap50_95"] = row["ap50_95"]
                row["ap50"] = row["ap50"]
                rows.append(row)
    return rows


def build_scene_relationship_metric_table(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[int, str, str, str], int] = defaultdict(int)

    for row in observations:
        drone_count = int(row["drone_count"])
        scene_key = str(row["scene_key"])
        for axis_name, field_name in RELATIONSHIP_AXES.items():
            relationship_type = str(row[field_name])
            counts[(drone_count, axis_name, relationship_type, scene_key)] += 1
            for metric_label, metric_field in METRICS.items():
                grouped[(drone_count, axis_name, relationship_type, scene_key, metric_label)].append(
                    float(row[metric_field])
                )

    rows: list[dict[str, object]] = []
    for (drone_count, axis_name, relationship_type, scene_key, metric_label), values in grouped.items():
        rows.append(
            {
                "drone_count": drone_count,
                "relationship_axis": axis_name,
                "relationship_type": relationship_type,
                "scene_key": scene_key,
                "metric": metric_label,
                "scene_mean_value": mean(values),
                "combo_count_in_scene": counts[(drone_count, axis_name, relationship_type, scene_key)],
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["relationship_axis"]),
            str(row["relationship_type"]),
            str(row["metric"]),
            str(row["scene_key"]),
        )
    )
    return rows


def summarize_bootstrap_ci(
    scene_metric_rows: list[dict[str, object]],
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in scene_metric_rows:
        grouped[
            (
                int(row["drone_count"]),
                str(row["relationship_axis"]),
                str(row["relationship_type"]),
                str(row["metric"]),
            )
        ].append(row)

    out: list[dict[str, object]] = []
    for (drone_count, axis_name, relationship_type, metric_label), rows in grouped.items():
        values = [float(row["scene_mean_value"]) for row in rows]
        mean_value, low, high, sd = bootstrap_mean(values, iterations=iterations, rng=rng)
        out.append(
            {
                "drone_count": drone_count,
                "relationship_axis": axis_name,
                "relationship_type": relationship_type,
                "metric": metric_label,
                "scene_count": len(rows),
                "total_combo_count": sum(int(row["combo_count_in_scene"]) for row in rows),
                "mean_scene_value": mean_value,
                "ci95_low": low,
                "ci95_high": high,
                "scene_sd": sd,
            }
        )
    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["metric"]),
            str(row["relationship_axis"]),
            -float(row["mean_scene_value"]),
        )
    )
    return out


def build_matched_scene_pairwise_comparisons(
    scene_metric_rows: list[dict[str, object]],
    iterations: int,
    rng: np.random.Generator,
    min_scenes: int,
) -> list[dict[str, object]]:
    value_maps: dict[tuple[int, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in scene_metric_rows:
        key = (
            int(row["drone_count"]),
            str(row["relationship_axis"]),
            str(row["relationship_type"]),
            str(row["metric"]),
        )
        value_maps[key][str(row["scene_key"])] = float(row["scene_mean_value"])

    available: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    for drone_count, axis_name, relationship_type, metric_label in value_maps:
        available[(drone_count, axis_name, metric_label)].append(relationship_type)

    out: list[dict[str, object]] = []
    for (drone_count, axis_name, metric_label), relationship_types in available.items():
        relationship_types = sorted(set(relationship_types))
        for left, right in combinations(relationship_types, 2):
            left_map = value_maps[(drone_count, axis_name, left, metric_label)]
            right_map = value_maps[(drone_count, axis_name, right, metric_label)]
            diff = bootstrap_paired_diff(left_map, right_map, iterations=iterations, rng=rng)
            if int(diff["common_scene_count"]) < min_scenes:
                continue
            out.append(
                {
                    "drone_count": drone_count,
                    "relationship_axis": axis_name,
                    "metric": metric_label,
                    "left_relationship_type": left,
                    "right_relationship_type": right,
                    "comparison": f"{left} minus {right}",
                    **diff,
                    "left_mean_on_common_scenes": mean(left_map[scene] for scene in set(left_map).intersection(right_map)),
                    "right_mean_on_common_scenes": mean(right_map[scene] for scene in set(left_map).intersection(right_map)),
                }
            )
    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["metric"]),
            str(row["relationship_axis"]),
            -abs(float(row["mean_difference"])),
        )
    )
    return out


def build_common_scene_axis_summary(scene_metric_rows: list[dict[str, object]], min_scenes: int) -> list[dict[str, object]]:
    by_axis_metric: dict[tuple[int, str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in scene_metric_rows:
        if row["metric"] not in {"AP50-95", "strict_quality"}:
            continue
        key = (int(row["drone_count"]), str(row["relationship_axis"]), str(row["metric"]))
        by_axis_metric[key][str(row["relationship_type"])][str(row["scene_key"])] = float(row["scene_mean_value"])

    out: list[dict[str, object]] = []
    for (drone_count, axis_name, metric_label), relation_maps in by_axis_metric.items():
        relationship_types = sorted(relation_maps)
        if len(relationship_types) < 2:
            continue
        common_scenes = set.intersection(*(set(relation_maps[item]) for item in relationship_types))
        if len(common_scenes) < min_scenes:
            continue
        for relationship_type in relationship_types:
            values = [relation_maps[relationship_type][scene] for scene in common_scenes]
            out.append(
                {
                    "drone_count": drone_count,
                    "relationship_axis": axis_name,
                    "metric": metric_label,
                    "relationship_type": relationship_type,
                    "common_scene_count": len(common_scenes),
                    "mean_value_on_common_scenes": mean(values),
                    "median_value_on_common_scenes": float(np.median(np.array(values, dtype=float))),
                }
            )
    out.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["metric"]),
            str(row["relationship_axis"]),
            -float(row["mean_value_on_common_scenes"]),
        )
    )
    return out


def build_robust_recommendations(
    bootstrap_rows: list[dict[str, object]],
    matched_rows: list[dict[str, object]],
    min_scenes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for boot_row in bootstrap_rows:
        if boot_row["metric"] != "AP50-95":
            continue
        if int(boot_row["scene_count"]) < min_scenes:
            continue
        comparisons = [
            row
            for row in matched_rows
            if int(row["drone_count"]) == int(boot_row["drone_count"])
            and row["relationship_axis"] == boot_row["relationship_axis"]
            and row["metric"] == "AP50-95"
            and (
                row["left_relationship_type"] == boot_row["relationship_type"]
                or row["right_relationship_type"] == boot_row["relationship_type"]
            )
        ]
        wins = 0
        losses = 0
        for row in comparisons:
            diff = float(row["mean_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            if row["left_relationship_type"] == boot_row["relationship_type"]:
                if low > 0:
                    wins += 1
                elif high < 0:
                    losses += 1
            else:
                if high < 0:
                    wins += 1
                elif low > 0:
                    losses += 1
        rows.append(
            {
                "drone_count": boot_row["drone_count"],
                "relationship_axis": boot_row["relationship_axis"],
                "relationship_type": boot_row["relationship_type"],
                "scene_count": boot_row["scene_count"],
                "mean_AP50_95": boot_row["mean_scene_value"],
                "ci95_low": boot_row["ci95_low"],
                "ci95_high": boot_row["ci95_high"],
                "significant_matched_wins": wins,
                "significant_matched_losses": losses,
                "robustness_score": float(boot_row["mean_scene_value"]) + 0.01 * wins - 0.02 * losses,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            str(row["relationship_axis"]),
            -float(row["robustness_score"]),
        )
    )
    return rows


def plot_bootstrap_intervals(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in rows
        if row["metric"] == "AP50-95" and int(row["drone_count"]) in {2, 3}
    ]
    axes = sorted({(int(row["drone_count"]), str(row["relationship_axis"])) for row in selected})
    fig, subplots = plt.subplots(len(axes), 1, figsize=(10, max(4, 2.4 * len(axes))), constrained_layout=True)
    if len(axes) == 1:
        subplots = [subplots]
    for ax, (drone_count, axis_name) in zip(subplots, axes):
        members = [
            row
            for row in selected
            if int(row["drone_count"]) == drone_count and row["relationship_axis"] == axis_name
        ]
        members.sort(key=lambda row: float(row["mean_scene_value"]))
        labels = [str(row["relationship_type"]) for row in members]
        means = np.array([float(row["mean_scene_value"]) for row in members], dtype=float)
        lows = np.array([float(row["ci95_low"]) for row in members], dtype=float)
        highs = np.array([float(row["ci95_high"]) for row in members], dtype=float)
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#3b6ea8")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Scene-normalized AP50-95 with bootstrap 95% CI")
        ax.set_title(f"k={drone_count}, {axis_name}")
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_matched_differences(rows: list[dict[str, object]], output_path: Path, top_n: int = 24) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in rows
        if row["metric"] == "AP50-95" and int(row["common_scene_count"]) >= 5
    ]
    selected.sort(key=lambda row: abs(float(row["mean_difference"])), reverse=True)
    selected = selected[:top_n]
    selected.sort(key=lambda row: float(row["mean_difference"]))
    labels = [
        f"k={row['drone_count']} {row['relationship_axis']}: {row['comparison']}"
        for row in selected
    ]
    means = np.array([float(row["mean_difference"]) for row in selected], dtype=float)
    lows = np.array([float(row["ci95_low"]) for row in selected], dtype=float)
    highs = np.array([float(row["ci95_high"]) for row in selected], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#8f4f6f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Paired scene AP50-95 difference with bootstrap 95% CI")
    ax.set_title("Largest matched-scene relationship differences")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def write_report(
    output_path: Path,
    bootstrap_rows: list[dict[str, object]],
    matched_rows: list[dict[str, object]],
    common_rows: list[dict[str, object]],
    recommendation_rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_boot = [
        row
        for row in bootstrap_rows
        if row["metric"] == "AP50-95" and int(row["drone_count"]) in {2, 3}
    ]
    top_boot.sort(key=lambda row: (int(row["drone_count"]), str(row["relationship_axis"]), -float(row["mean_scene_value"])))
    top_recs = recommendation_rows[:18]
    significant = [
        row
        for row in matched_rows
        if row["metric"] == "AP50-95"
        and (float(row["ci95_low"]) > 0 or float(row["ci95_high"]) < 0)
    ]
    significant.sort(key=lambda row: abs(float(row["mean_difference"])), reverse=True)

    lines = [
        "# Robust Viewpoint Diversity Analysis",
        "",
        "## What This Adds",
        "",
        "This analysis strengthens the factor-level pair/triple conclusions using two checks:",
        "",
        "1. Bootstrap confidence intervals over scenes, not over raw combinations.",
        "2. Matched-scene pairwise comparisons between relationship types, so each difference is computed only on scenes where both relationship types are available.",
        "",
        "This directly addresses the sparse exact-pair/exact-triple support problem.",
        "",
        "## Strongest Scene-Normalized Relationship Types",
        "",
    ]

    seen_axis: set[tuple[int, str]] = set()
    for row in top_boot:
        key = (int(row["drone_count"]), str(row["relationship_axis"]))
        if key in seen_axis:
            continue
        seen_axis.add(key)
        lines.append(
            "- "
            f"k=`{row['drone_count']}`, `{row['relationship_axis']}` best type: "
            f"`{row['relationship_type']}` with AP50-95 `{float(row['mean_scene_value']):.4f}` "
            f"(95% CI `{float(row['ci95_low']):.4f}` to `{float(row['ci95_high']):.4f}`, "
            f"`{row['scene_count']}` scenes)"
        )

    lines.extend(["", "## Matched-Scene Differences", ""])
    if significant:
        for row in significant[:10]:
            lines.append(
                "- "
                f"k=`{row['drone_count']}`, `{row['relationship_axis']}`, `{row['comparison']}`: "
                f"mean difference `{float(row['mean_difference']):+.4f}`, "
                f"95% CI `{float(row['ci95_low']):+.4f}` to `{float(row['ci95_high']):+.4f}`, "
                f"common scenes `{row['common_scene_count']}`"
            )
    else:
        lines.append("- No AP50-95 relationship differences had a bootstrap CI excluding zero at the current support threshold.")

    lines.extend(["", "## Robust Recommendations", ""])
    for row in top_recs:
        lines.append(
            "- "
            f"k=`{row['drone_count']}`, `{row['relationship_axis']}={row['relationship_type']}`: "
            f"mean AP50-95 `{float(row['mean_AP50_95']):.4f}`, "
            f"CI `{float(row['ci95_low']):.4f}` to `{float(row['ci95_high']):.4f}`, "
            f"wins `{row['significant_matched_wins']}`, losses `{row['significant_matched_losses']}`"
        )

    lines.extend(
        [
            "",
            "## Thesis Interpretation",
            "",
            "Use this robustness layer to support claims about relationship patterns, not exact viewpoint identities. If a relationship type has a high scene-normalized mean but its matched-scene CI overlaps competing types, phrase the conclusion as a tendency rather than a statistically decisive result.",
            "",
            "A strong thesis-safe claim is: evaluate exact top pairs/triples as candidates, but base general swarm-design guidance on relationship types that remain strong under scene-normalized bootstrap and matched-scene checks.",
            "",
            "## Generated Files",
            "",
            "- `scene_relationship_metric_table.csv`",
            "- `relationship_bootstrap_ci.csv`",
            "- `matched_scene_relationship_differences.csv`",
            "- `common_scene_axis_summary.csv`",
            "- `robust_relationship_recommendations.csv`",
            "- `plots/bootstrap_relationship_ci.png`",
            "- `plots/matched_scene_differences.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    rng = np.random.default_rng(args.seed)

    records = read_scene_records(Path(args.scene_records))
    observations = build_combo_observations(records, max_k=args.max_k)
    scene_metric_rows = build_scene_relationship_metric_table(observations)
    bootstrap_rows = summarize_bootstrap_ci(scene_metric_rows, iterations=args.bootstrap_iters, rng=rng)
    matched_rows = build_matched_scene_pairwise_comparisons(
        scene_metric_rows,
        iterations=args.bootstrap_iters,
        rng=rng,
        min_scenes=args.min_scenes,
    )
    common_rows = build_common_scene_axis_summary(scene_metric_rows, min_scenes=args.min_scenes)
    recommendation_rows = build_robust_recommendations(
        bootstrap_rows,
        matched_rows,
        min_scenes=args.min_scenes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scene_relationship_metric_table.csv", scene_metric_rows)
    write_csv(output_dir / "relationship_bootstrap_ci.csv", bootstrap_rows)
    write_csv(output_dir / "matched_scene_relationship_differences.csv", matched_rows)
    write_csv(output_dir / "common_scene_axis_summary.csv", common_rows)
    write_csv(output_dir / "robust_relationship_recommendations.csv", recommendation_rows)
    plot_bootstrap_intervals(bootstrap_rows, plots_dir / "bootstrap_relationship_ci.png")
    plot_matched_differences(matched_rows, plots_dir / "matched_scene_differences.png")
    write_report(
        output_dir / "ROBUST_VIEWPOINT_DIVERSITY_SUMMARY.md",
        bootstrap_rows=bootstrap_rows,
        matched_rows=matched_rows,
        common_rows=common_rows,
        recommendation_rows=recommendation_rows,
    )

    print(f"Wrote robust viewpoint diversity analysis to {output_dir}")
    print(f"Scene relationship metric rows: {len(scene_metric_rows)}")
    print(f"Bootstrap summary rows: {len(bootstrap_rows)}")
    print(f"Matched-scene comparison rows: {len(matched_rows)}")


if __name__ == "__main__":
    main()
