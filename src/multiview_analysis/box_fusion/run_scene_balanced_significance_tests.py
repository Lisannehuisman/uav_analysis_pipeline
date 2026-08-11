from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_ROWS = WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "pair_combo_rows.csv"
DEFAULT_TRIPLE_ROWS = WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "triple_combo_rows.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_cross_view_box_fusion_analysis" / "outputs" / "significance_tests"


@dataclass(frozen=True)
class ComparisonSpec:
    left_key: str
    left_label: str
    right_key: str
    right_label: str


COMPARISONS = [
    ComparisonSpec("noisy_or", "Noisy-OR + best IoU", "best_box", "Best box"),
    ComparisonSpec("support", "Support-weighted noisy-OR", "best_box", "Best box"),
    ComparisonSpec("noisy_or", "Noisy-OR + best IoU", "support", "Support-weighted noisy-OR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scene-balanced paired significance tests for the late-fusion box-combination results."
    )
    parser.add_argument("--pair-rows", default=str(DEFAULT_PAIR_ROWS))
    parser.add_argument("--triple-rows", default=str(DEFAULT_TRIPLE_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def load_scene_balanced_rows(path: Path) -> list[dict[str, object]]:
    by_scene: dict[str, dict[str, list[float] | str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scene_key = row["scene_key"]
            current = by_scene.setdefault(
                scene_key,
                {
                    "scene_key": scene_key,
                    "target_class": row["target_class"],
                    "best_box": [],
                    "noisy_or": [],
                    "support": [],
                },
            )
            current["best_box"].append(float(row["best_box_quality"]))
            current["noisy_or"].append(float(row["fused_quality_noisy_or_max_iou"]))
            current["support"].append(float(row["fused_quality_support_weighted_or"]))

    scene_rows: list[dict[str, object]] = []
    for scene_key, values in sorted(by_scene.items()):
        scene_rows.append(
            {
                "scene_key": scene_key,
                "target_class": values["target_class"],
                "best_box": float(np.mean(values["best_box"])),
                "noisy_or": float(np.mean(values["noisy_or"])),
                "support": float(np.mean(values["support"])),
            }
        )
    return scene_rows


def bootstrap_mean_ci(deltas: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(deltas), size=(n_bootstrap, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def permutation_p_value(deltas: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(deltas)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(deltas)))
    null_means = np.abs((signs * deltas).mean(axis=1))
    exceed = int(np.sum(null_means >= observed))
    return float((exceed + 1) / (n_permutations + 1))


def holm_adjust_in_place(rows: list[dict[str, object]], p_key: str, adjusted_key: str) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    total = len(indexed)
    adjusted_values = [0.0] * total
    running_max = 0.0
    for rank, (original_index, row) in enumerate(indexed, start=1):
        candidate = min(1.0, float(row[p_key]) * (total - rank + 1))
        running_max = max(running_max, candidate)
        adjusted_values[original_index] = running_max
    for row, adjusted_value in zip(rows, adjusted_values, strict=False):
        row[adjusted_key] = float(adjusted_value)


def summarize_drone_count(
    scene_rows: list[dict[str, object]],
    drone_count: int,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    scene_export_rows: list[dict[str, object]] = []

    scene_rows_sorted = sorted(scene_rows, key=lambda row: str(row["scene_key"]))
    for scene_row in scene_rows_sorted:
        scene_export_rows.append(
            {
                "drone_count": drone_count,
                "scene_key": scene_row["scene_key"],
                "target_class": scene_row["target_class"],
                "best_box_scene_mean": scene_row["best_box"],
                "noisy_or_scene_mean": scene_row["noisy_or"],
                "support_scene_mean": scene_row["support"],
                "delta_noisy_vs_best": float(scene_row["noisy_or"]) - float(scene_row["best_box"]),
                "delta_support_vs_best": float(scene_row["support"]) - float(scene_row["best_box"]),
                "delta_noisy_vs_support": float(scene_row["noisy_or"]) - float(scene_row["support"]),
            }
        )

    for comparison in COMPARISONS:
        left = np.array([float(row[comparison.left_key]) for row in scene_rows_sorted], dtype=float)
        right = np.array([float(row[comparison.right_key]) for row in scene_rows_sorted], dtype=float)
        deltas = left - right
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap, rng)
        rows.append(
            {
                "drone_count": drone_count,
                "comparison": f"{comparison.left_label} minus {comparison.right_label}",
                "n_scenes": len(scene_rows_sorted),
                "left_mean_scene_balanced": float(np.mean(left)),
                "right_mean_scene_balanced": float(np.mean(right)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)),
                "cohen_dz": float(np.mean(deltas) / np.std(deltas, ddof=1)),
                "positive_scene_count": int(np.sum(deltas > 0.0)),
                "negative_scene_count": int(np.sum(deltas < 0.0)),
                "zero_scene_count": int(np.sum(np.isclose(deltas, 0.0))),
                "positive_scene_rate": float(np.mean(deltas > 0.0)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p_two_sided": permutation_p_value(deltas, n_permutations, rng),
            }
        )

    return rows, scene_export_rows


def write_markdown_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(int(row["drone_count"]), []).append(row)

    lines = [
        "# Scene-Balanced Late-Fusion Significance Tests",
        "",
        "- The statistical unit is the scene, not the raw coalition row.",
        "- For each scene, coalition scores were averaged across all available combinations for that scene.",
        "- Uncertainty is reported with paired scene bootstrap confidence intervals.",
        "- Significance is tested with a two-sided paired sign-flip permutation test on the per-scene deltas.",
        "",
    ]

    for drone_count in sorted(grouped):
        lines.extend(
            [
                f"## {drone_count}-view scene-balanced comparisons",
                "",
                "| Comparison | Left mean | Right mean | Mean delta | 95% CI | Positive scenes | Permutation p | Holm p |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in grouped[drone_count]:
            lines.append(
                f"| {row['comparison']} | "
                f"{format_float(float(row['left_mean_scene_balanced']))} | "
                f"{format_float(float(row['right_mean_scene_balanced']))} | "
                f"{format_float(float(row['mean_delta']))} | "
                f"[{format_float(float(row['bootstrap_ci_low']))}, {format_float(float(row['bootstrap_ci_high']))}] | "
                f"{row['positive_scene_count']}/{row['n_scenes']} | "
                f"{format_float(float(row['permutation_p_two_sided']))} | "
                f"{format_float(float(row['holm_adjusted_p']))} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pair_scene_rows = load_scene_balanced_rows(Path(args.pair_rows))
    triple_scene_rows = load_scene_balanced_rows(Path(args.triple_rows))

    pair_summary_rows, pair_scene_export_rows = summarize_drone_count(
        pair_scene_rows,
        drone_count=2,
        n_bootstrap=args.bootstrap_samples,
        n_permutations=args.permutation_samples,
        rng=rng,
    )
    triple_summary_rows, triple_scene_export_rows = summarize_drone_count(
        triple_scene_rows,
        drone_count=3,
        n_bootstrap=args.bootstrap_samples,
        n_permutations=args.permutation_samples,
        rng=rng,
    )

    summary_rows = pair_summary_rows + triple_summary_rows
    holm_adjust_in_place(summary_rows, "permutation_p_two_sided", "holm_adjusted_p")
    scene_rows = pair_scene_export_rows + triple_scene_export_rows

    summary_path = output_dir / "scene_balanced_fusion_significance_summary.csv"
    scene_path = output_dir / "scene_balanced_fusion_scene_means.csv"
    report_path = output_dir / "scene_balanced_fusion_significance_report.md"

    write_csv(summary_path, summary_rows)
    write_csv(scene_path, scene_rows)
    write_markdown_report(report_path, summary_rows)

    print(summary_path)
    print(scene_path)
    print(report_path)


if __name__ == "__main__":
    main()
