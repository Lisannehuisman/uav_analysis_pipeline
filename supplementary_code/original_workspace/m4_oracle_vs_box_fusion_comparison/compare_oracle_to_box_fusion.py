from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE_SCENE = WORKSPACE / "m4_two_drone_operational_analysis" / "thesis_swarm_outputs" / "protocol_scene_expectation_summary.csv"
DEFAULT_PAIR_ROWS = WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "pair_combo_rows.csv"
DEFAULT_TRIPLE_ROWS = WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "triple_combo_rows.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_oracle_vs_box_fusion_comparison" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the current oracle-best-available multiview evaluation to the existing box-fusion policies."
    )
    parser.add_argument("--oracle-scene-csv", default=str(DEFAULT_ORACLE_SCENE))
    parser.add_argument("--pair-csv", default=str(DEFAULT_PAIR_ROWS))
    parser.add_argument("--triple-csv", default=str(DEFAULT_TRIPLE_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def parse_float(raw: str) -> float:
    text = str(raw).strip()
    if not text:
        return 0.0
    return float(text)


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def aggregate_fusion_scene_rows(combo_rows: list[dict[str, str]], drone_count: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in combo_rows:
        grouped[(row["scene_key"], row["target_class"])].append(row)

    rows: list[dict[str, object]] = []
    for (scene_key, target_class), members in sorted(grouped.items()):
        rows.append(
            {
                "drone_count": drone_count,
                "scene_key": scene_key,
                "target_class": target_class,
                "combination_count": len(members),
                "fusion_noisy_or_max_iou_quality": mean(
                    [parse_float(row["fused_quality_noisy_or_max_iou"]) for row in members]
                ),
                "fusion_support_weighted_or_quality": mean(
                    [parse_float(row["fused_quality_support_weighted_or"]) for row in members]
                ),
                "fusion_mean_target_ap50_95_reference": mean(
                    [parse_float(row["mean_target_ap50_95"]) for row in members]
                ),
            }
        )
    return rows


def build_matched_scene_rows(
    oracle_scene_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    triple_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    oracle_rows = [
        row
        for row in oracle_scene_rows
        if row["protocol_id"] in {"n2_any1", "n3_any1"}
    ]
    oracle_lookup = {
        (int(row["drone_count"]), row["scene_key"], row["target_class"]): row
        for row in oracle_rows
    }

    fusion_scene_rows = (
        aggregate_fusion_scene_rows(pair_rows, 2)
        + aggregate_fusion_scene_rows(triple_rows, 3)
    )

    matched: list[dict[str, object]] = []
    for row in fusion_scene_rows:
        key = (int(row["drone_count"]), str(row["scene_key"]), str(row["target_class"]))
        oracle = oracle_lookup.get(key)
        if oracle is None:
            continue
        matched.append(
            {
                "drone_count": int(row["drone_count"]),
                "scene_key": str(row["scene_key"]),
                "target_class": str(row["target_class"]),
                "combination_count": int(row["combination_count"]),
                "oracle_expected_target_threshold_strict_quality_iou50": parse_float(
                    oracle["expected_target_threshold_strict_quality_iou50"]
                ),
                "oracle_expected_target_threshold_ap50_95": parse_float(
                    oracle["expected_target_threshold_ap50_95"]
                ),
                "fusion_noisy_or_max_iou_quality": float(row["fusion_noisy_or_max_iou_quality"]),
                "fusion_support_weighted_or_quality": float(row["fusion_support_weighted_or_quality"]),
                "gap_noisy_or_vs_oracle": float(row["fusion_noisy_or_max_iou_quality"])
                - parse_float(oracle["expected_target_threshold_strict_quality_iou50"]),
                "gap_support_weighted_vs_oracle": float(row["fusion_support_weighted_or_quality"])
                - parse_float(oracle["expected_target_threshold_strict_quality_iou50"]),
            }
        )
    return matched


def summarize_overall(matched_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in matched_rows:
        grouped[int(row["drone_count"])].append(row)

    rows: list[dict[str, object]] = []
    for drone_count, members in sorted(grouped.items()):
        oracle = mean([float(row["oracle_expected_target_threshold_strict_quality_iou50"]) for row in members])
        noisy = mean([float(row["fusion_noisy_or_max_iou_quality"]) for row in members])
        support = mean([float(row["fusion_support_weighted_or_quality"]) for row in members])
        rows.append(
            {
                "drone_count": drone_count,
                "scene_count": len(members),
                "oracle_best_available_quality": oracle,
                "fusion_noisy_or_max_iou_quality": noisy,
                "fusion_support_weighted_or_quality": support,
                "gap_noisy_or_vs_oracle": noisy - oracle,
                "gap_support_weighted_vs_oracle": support - oracle,
            }
        )
    return rows


def summarize_by_class(matched_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in matched_rows:
        grouped[(int(row["drone_count"]), str(row["target_class"]))].append(row)

    rows: list[dict[str, object]] = []
    for (drone_count, target_class), members in sorted(grouped.items()):
        oracle = mean([float(row["oracle_expected_target_threshold_strict_quality_iou50"]) for row in members])
        noisy = mean([float(row["fusion_noisy_or_max_iou_quality"]) for row in members])
        support = mean([float(row["fusion_support_weighted_or_quality"]) for row in members])
        rows.append(
            {
                "drone_count": drone_count,
                "target_class": target_class,
                "scene_count": len(members),
                "oracle_best_available_quality": oracle,
                "fusion_noisy_or_max_iou_quality": noisy,
                "fusion_support_weighted_or_quality": support,
                "gap_noisy_or_vs_oracle": noisy - oracle,
                "gap_support_weighted_vs_oracle": support - oracle,
            }
        )
    return rows


def plot_overall(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    x = [float(row["drone_count"]) for row in summary_rows]
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([value - width for value in x], [float(row["oracle_best_available_quality"]) for row in summary_rows], width, label="Oracle / current method", color="#355C7D")
    ax.bar([value for value in x], [float(row["fusion_noisy_or_max_iou_quality"]) for row in summary_rows], width, label="Noisy-OR + best IoU", color="#C06C84")
    ax.bar([value + width for value in x], [float(row["fusion_support_weighted_or_quality"]) for row in summary_rows], width, label="Support-weighted OR", color="#F0A202")
    ax.set_xticks(x)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Scene-averaged strict quality")
    ax.set_title("Current Method vs Late Fusion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class(
    class_rows: list[dict[str, object]],
    drone_count: int,
    output_path: Path,
    x_min: float = 0.6,
) -> None:
    rows = [row for row in class_rows if int(row["drone_count"]) == drone_count]
    rows.sort(key=lambda row: float(row["oracle_best_available_quality"]), reverse=True)
    labels = [str(row["target_class"]) for row in rows]
    y = list(range(len(rows)))
    h = 0.22

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(rows) + 1.5)))
    ax.barh([value - h for value in y], [float(row["oracle_best_available_quality"]) for row in rows], h, label="Oracle / current method", color="#355C7D")
    ax.barh([value for value in y], [float(row["fusion_noisy_or_max_iou_quality"]) for row in rows], h, label="Noisy-OR + best IoU", color="#C06C84")
    ax.barh([value + h for value in y], [float(row["fusion_support_weighted_or_quality"]) for row in rows], h, label="Support-weighted OR", color="#F0A202")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Scene-averaged strict quality")
    ax.set_title(f"Per-Class Comparison for {drone_count} Views")
    values = [
        float(row["oracle_best_available_quality"]) for row in rows
    ] + [
        float(row["fusion_noisy_or_max_iou_quality"]) for row in rows
    ] + [
        float(row["fusion_support_weighted_or_quality"]) for row in rows
    ]
    current_max = max(values) if values else x_min + 0.01
    right_limit = max(current_max * 1.01, x_min + 0.01)
    ax.set_xlim(left=x_min, right=right_limit)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(overall_rows: list[dict[str, object]], class_rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Current Method vs Late Fusion Comparison",
        "",
        "This report compares the current method to the late-fusion policies on the same scene-balanced evaluation base:",
        "",
        "- `oracle / current method`: the current evaluation, which takes the best target strict-quality available inside the selected view set",
        "- `late fusion`: the existing deployable policies from `m4_cross_view_box_fusion_analysis`",
        "",
        "The `best_box` policy is intentionally omitted from the outputs here because it is exactly identical to the oracle/current-method score for every overall and per-class row in this dataset.",
        "",
        "## Overall",
        "",
    ]
    for row in overall_rows:
        lines.extend(
            [
                f"### {int(row['drone_count'])} views",
                f"- Oracle / current method: `{fmt(float(row['oracle_best_available_quality']))}`",
                f"- Noisy-OR + best IoU: `{fmt(float(row['fusion_noisy_or_max_iou_quality']))}` (gap vs oracle `{fmt(float(row['gap_noisy_or_vs_oracle']))}`)",
                f"- Support-weighted OR: `{fmt(float(row['fusion_support_weighted_or_quality']))}` (gap vs oracle `{fmt(float(row['gap_support_weighted_vs_oracle']))}`)",
                "",
            ]
        )

    lines.extend(["## Per Class Highlights", ""])
    for drone_count in [2, 3]:
        rows = [row for row in class_rows if int(row["drone_count"]) == drone_count]
        if not rows:
            continue
        best_support = max(rows, key=lambda row: float(row["fusion_support_weighted_or_quality"]))
        closest_gap = min(rows, key=lambda row: abs(float(row["gap_support_weighted_vs_oracle"])))
        highest_above = max(rows, key=lambda row: float(row["gap_support_weighted_vs_oracle"]))
        lowest_below = min(rows, key=lambda row: float(row["gap_support_weighted_vs_oracle"]))
        lines.extend(
            [
                f"### {drone_count} views",
                f"- Highest support-weighted fused quality: `{best_support['target_class']}` at `{fmt(float(best_support['fusion_support_weighted_or_quality']))}`",
                f"- Closest support-weighted result to oracle: `{closest_gap['target_class']}` at gap `{fmt(float(closest_gap['gap_support_weighted_vs_oracle']))}`",
                f"- Largest positive support-weighted gap vs oracle: `{highest_above['target_class']}` at `{fmt(float(highest_above['gap_support_weighted_vs_oracle']))}`",
                f"- Largest negative support-weighted gap vs oracle: `{lowest_below['target_class']}` at `{fmt(float(lowest_below['gap_support_weighted_vs_oracle']))}`",
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    oracle_scene_rows = read_csv_rows(Path(args.oracle_scene_csv))
    pair_rows = read_csv_rows(Path(args.pair_csv))
    triple_rows = read_csv_rows(Path(args.triple_csv))

    matched_scene_rows = build_matched_scene_rows(oracle_scene_rows, pair_rows, triple_rows)
    overall_rows = summarize_overall(matched_scene_rows)
    class_rows = summarize_by_class(matched_scene_rows)

    write_csv(output_dir / "matched_scene_policy_comparison.csv", matched_scene_rows)
    write_csv(output_dir / "overall_policy_comparison.csv", overall_rows)
    write_csv(output_dir / "class_policy_comparison.csv", class_rows)

    plot_overall(overall_rows, plots_dir / "overall_policy_comparison.png")
    plot_per_class(class_rows, 2, plots_dir / "per_class_policy_comparison_k2.png")
    plot_per_class(class_rows, 3, plots_dir / "per_class_policy_comparison_k3.png")
    write_report(overall_rows, class_rows, output_dir / "oracle_vs_box_fusion_report.md")


if __name__ == "__main__":
    main()
