from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from m4_marginal_viewpoint_value_analysis.compute_ring_shapley_noisy_or_best_iou import (
    AZIMUTHS,
    DEFAULT_SCENE_RECORDS,
    build_scene_lookup,
    coalition_value_for_scene,
    exact_shapley_from_values,
    load_records,
    ring_scene_keys,
    ring_sort_key,
)

DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_marginal_viewpoint_value_analysis" / "outputs" / "ring_shapley_significance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap scene-level uncertainty for exact 8-player ring Shapley values."
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
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


def build_ring_scene_value_matrix(
    ring_id: str,
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    ring_scenes: list[str],
) -> np.ndarray:
    full_mask = (1 << len(AZIMUTHS)) - 1
    matrix = np.zeros((len(ring_scenes), full_mask + 1), dtype=float)
    for scene_index, scene_key in enumerate(ring_scenes):
        scene_records = scene_lookup[scene_key]
        for coalition_mask in range(full_mask + 1):
            matrix[scene_index, coalition_mask] = coalition_value_for_scene(scene_records, ring_id, coalition_mask)
    return matrix


def bootstrap_ring(
    ring_id: str,
    coalition_scene_matrix: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    observed_coalition_values = coalition_scene_matrix.mean(axis=0)
    observed_shapley = exact_shapley_from_values(observed_coalition_values.tolist(), n_players=len(AZIMUTHS))
    order = np.argsort(observed_shapley)[::-1]
    top_index = int(order[0])
    second_index = int(order[1])
    top_azimuth = AZIMUTHS[top_index]
    second_azimuth = AZIMUTHS[second_index]
    observed_gap = float(observed_shapley[top_index] - observed_shapley[second_index])

    sample_indices = rng.integers(0, coalition_scene_matrix.shape[0], size=(bootstrap_samples, coalition_scene_matrix.shape[0]))
    bootstrap_shapley = np.zeros((bootstrap_samples, len(AZIMUTHS)), dtype=float)
    top_counts = np.zeros(len(AZIMUTHS), dtype=int)

    for sample_idx in range(bootstrap_samples):
        boot_values = coalition_scene_matrix[sample_indices[sample_idx]].mean(axis=0)
        boot_shapley = exact_shapley_from_values(boot_values.tolist(), n_players=len(AZIMUTHS))
        bootstrap_shapley[sample_idx] = boot_shapley
        top_counts[int(np.argmax(boot_shapley))] += 1

    detail_rows: list[dict[str, object]] = []
    for azimuth_index, azimuth in enumerate(AZIMUTHS):
        boot_values = bootstrap_shapley[:, azimuth_index]
        detail_rows.append(
            {
                "ring_id": ring_id,
                "azimuth": azimuth,
                "observed_shapley": float(observed_shapley[azimuth_index]),
                "bootstrap_mean": float(np.mean(boot_values)),
                "bootstrap_ci_low": float(np.quantile(boot_values, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot_values, 0.975)),
                "selected_as_top_rate": float(top_counts[azimuth_index] / bootstrap_samples),
            }
        )

    gap_samples = bootstrap_shapley[:, top_index] - bootstrap_shapley[:, second_index]
    summary_row = {
        "ring_id": ring_id,
        "n_scenes": coalition_scene_matrix.shape[0],
        "observed_top_azimuth": top_azimuth,
        "observed_top_shapley": float(observed_shapley[top_index]),
        "observed_second_azimuth": second_azimuth,
        "observed_second_shapley": float(observed_shapley[second_index]),
        "observed_gap_top_minus_second": observed_gap,
        "gap_bootstrap_ci_low": float(np.quantile(gap_samples, 0.025)),
        "gap_bootstrap_ci_high": float(np.quantile(gap_samples, 0.975)),
        "gap_positive_rate": float(np.mean(gap_samples > 0.0)),
        "top_selected_rate": float(top_counts[top_index] / bootstrap_samples),
    }
    return detail_rows, summary_row


def write_markdown_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Bootstrap Uncertainty For Exact Ring Shapley",
        "",
        "- Scenes are resampled with replacement within each ring.",
        "- For each bootstrap sample, the exact 8-player Shapley values are recomputed from the resampled scene coalition values.",
        "- The key inferential quantity is the bootstrap interval for the gap between the observed top azimuth and the observed runner-up in each ring.",
        "",
        "| Ring | Scenes | Top azimuth | Runner-up | Observed gap | 95% bootstrap CI for gap | P(gap > 0) | Top stays top |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['ring_id']}` | {row['n_scenes']} | "
            f"`az{int(row['observed_top_azimuth']):03d}` | "
            f"`az{int(row['observed_second_azimuth']):03d}` | "
            f"{format_float(float(row['observed_gap_top_minus_second']))} | "
            f"[{format_float(float(row['gap_bootstrap_ci_low']))}, {format_float(float(row['gap_bootstrap_ci_high']))}] | "
            f"{format_float(float(row['gap_positive_rate']))} | "
            f"{format_float(float(row['top_selected_rate']))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    records = load_records(Path(args.scene_records))
    scene_lookup = build_scene_lookup(records)
    ring_ids = sorted({str(record["ring_id"]) for record in records}, key=ring_sort_key)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for ring_id in ring_ids:
        ring_scenes = ring_scene_keys(ring_id, scene_lookup)
        coalition_scene_matrix = build_ring_scene_value_matrix(ring_id, scene_lookup, ring_scenes)
        ring_detail_rows, ring_summary_row = bootstrap_ring(
            ring_id,
            coalition_scene_matrix,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
        )
        detail_rows.extend(ring_detail_rows)
        summary_rows.append(ring_summary_row)

    summary_rows.sort(key=lambda row: ring_sort_key(str(row["ring_id"])))

    detail_path = output_dir / "ring_shapley_bootstrap_detail.csv"
    summary_path = output_dir / "ring_shapley_bootstrap_summary.csv"
    report_path = output_dir / "ring_shapley_bootstrap_report.md"

    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    write_markdown_report(report_path, summary_rows)

    print(detail_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
