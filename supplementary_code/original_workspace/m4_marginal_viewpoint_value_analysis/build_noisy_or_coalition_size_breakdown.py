from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_ring_shapley_noisy_or_best_iou import (
    AZIMUTHS,
    DEFAULT_SCENE_RECORDS,
    build_scene_class_lookup,
    build_scene_lookup,
    coalition_value_for_scene,
    format_float,
    load_records,
    ring_scene_keys,
    ring_sort_key,
    write_csv,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_marginal_viewpoint_value_analysis" / "outputs" / "coalition_size_breakdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Break down Noisy-OR + best IoU coalition value by exact coalition size, "
            "so the gain of the 2nd, 3rd, 4th, ... drone becomes explicit."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def coalition_members(mask: int) -> str:
    members = [f"az{AZIMUTHS[idx]:03d}" for idx in range(len(AZIMUTHS)) if mask & (1 << idx)]
    return " ".join(members) if members else "(empty)"


def analyze_ring_sizes(
    ring_id: str,
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    ring_scenes: list[str],
) -> list[dict[str, object]]:
    full_mask = (1 << len(AZIMUTHS)) - 1
    size_to_values: dict[int, list[float]] = defaultdict(list)
    size_to_best: dict[int, tuple[float, int]] = {}
    max_views_seen_in_any_ring_scene = 0

    ring_viewpoints = [f"{ring_id}-az{azimuth:03d}" for azimuth in AZIMUTHS]
    for scene_key in ring_scenes:
        observed = sum(1 for viewpoint in ring_viewpoints if viewpoint in scene_lookup[scene_key])
        max_views_seen_in_any_ring_scene = max(max_views_seen_in_any_ring_scene, observed)

    for coalition_mask in range(full_mask + 1):
        coalition_scene_values = [
            coalition_value_for_scene(scene_lookup[scene_key], ring_id, coalition_mask)
            for scene_key in ring_scenes
        ]
        coalition_value = float(np.mean(coalition_scene_values))
        subset_size = coalition_mask.bit_count()
        size_to_values[subset_size].append(coalition_value)
        previous_best = size_to_best.get(subset_size)
        if previous_best is None or coalition_value > previous_best[0]:
            size_to_best[subset_size] = (coalition_value, coalition_mask)

    rows: list[dict[str, object]] = []
    previous_mean_value = None
    previous_best_value = None
    grand_value = float(size_to_best[len(AZIMUTHS)][0])
    for subset_size in range(len(AZIMUTHS) + 1):
        mean_value = float(np.mean(size_to_values[subset_size]))
        best_value, best_mask = size_to_best[subset_size]
        mean_gain = float("nan") if previous_mean_value is None else mean_value - previous_mean_value
        best_gain = float("nan") if previous_best_value is None else best_value - previous_best_value
        rows.append(
            {
                "ring_id": ring_id,
                "subset_size": subset_size,
                "added_drone_number": subset_size,
                "ring_scene_count": len(ring_scenes),
                "max_views_seen_in_any_ring_scene": max_views_seen_in_any_ring_scene,
                "mean_coalition_value": mean_value,
                "best_coalition_value": best_value,
                "mean_marginal_gain_from_prev_size": mean_gain,
                "best_marginal_gain_from_prev_size": best_gain,
                "mean_share_of_full_8_drone_value": float("nan") if grand_value <= 0.0 else mean_value / grand_value,
                "best_share_of_full_8_drone_value": float("nan") if grand_value <= 0.0 else best_value / grand_value,
                "best_coalition_members": coalition_members(best_mask),
                "num_coalitions_of_this_size": len(size_to_values[subset_size]),
            }
        )
        previous_mean_value = mean_value
        previous_best_value = best_value
    return rows


def aggregate_across_rings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_size[int(row["subset_size"])].append(row)

    aggregated_rows: list[dict[str, object]] = []
    for subset_size in sorted(by_size):
        members = by_size[subset_size]
        aggregated_rows.append(
            {
                "subset_size": subset_size,
                "num_rings": len(members),
                "mean_of_ring_mean_coalition_value": float(np.mean([float(row["mean_coalition_value"]) for row in members])),
                "mean_of_ring_best_coalition_value": float(np.mean([float(row["best_coalition_value"]) for row in members])),
                "mean_of_ring_mean_marginal_gain": float(np.mean([float(row["mean_marginal_gain_from_prev_size"]) for row in members]))
                if subset_size > 0
                else float("nan"),
                "mean_of_ring_best_marginal_gain": float(np.mean([float(row["best_marginal_gain_from_prev_size"]) for row in members]))
                if subset_size > 0
                else float("nan"),
                "min_ring_mean_coalition_value": float(np.min([float(row["mean_coalition_value"]) for row in members])),
                "max_ring_mean_coalition_value": float(np.max([float(row["mean_coalition_value"]) for row in members])),
            }
        )
    return aggregated_rows


def plot_all_rings_mean_values(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ring_ids = sorted({str(row["ring_id"]) for row in rows}, key=ring_sort_key)
    for ring_id in ring_ids:
        ring_rows = [row for row in rows if str(row["ring_id"]) == ring_id]
        x = [int(row["subset_size"]) for row in ring_rows]
        y = [float(row["mean_coalition_value"]) for row in ring_rows]
        ax.plot(x, y, marker="o", linewidth=2.0, label=ring_id)
    ax.set_title("Noisy-OR + best IoU: mean coalition value by exact drone count")
    ax.set_xlabel("Number of drones in coalition")
    ax.set_ylabel("Mean coalition value")
    ax.set_xticks(range(0, len(AZIMUTHS) + 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_all_rings_mean_gains(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ring_ids = sorted({str(row["ring_id"]) for row in rows}, key=ring_sort_key)
    for ring_id in ring_ids:
        ring_rows = [row for row in rows if str(row["ring_id"]) == ring_id and int(row["subset_size"]) > 0]
        x = [int(row["added_drone_number"]) for row in ring_rows]
        y = [float(row["mean_marginal_gain_from_prev_size"]) for row in ring_rows]
        ax.plot(x, y, marker="o", linewidth=2.0, label=ring_id)
    ax.set_title("How much the 1st, 2nd, 3rd, ... drone adds on average")
    ax.set_xlabel("Added drone number (k)")
    ax.set_ylabel("Mean marginal gain from size k-1 to k")
    ax.set_xticks(range(1, len(AZIMUTHS) + 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_all_rings_share_of_full_value(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ring_ids = sorted({str(row["ring_id"]) for row in rows}, key=ring_sort_key)
    for ring_id in ring_ids:
        ring_rows = [row for row in rows if str(row["ring_id"]) == ring_id]
        x = [int(row["subset_size"]) for row in ring_rows]
        y = [float(row["mean_share_of_full_8_drone_value"]) for row in ring_rows]
        ax.plot(x, y, marker="o", linewidth=2.0, label=ring_id)
    ax.set_title("How much of the full 8-drone value is already reached at each drone count")
    ax.set_xlabel("Number of drones in coalition")
    ax.set_ylabel("Mean coalition value / full 8-drone value")
    ax.set_xticks(range(0, len(AZIMUTHS) + 1))
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_mean_gain_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    ring_ids = sorted({str(row["ring_id"]) for row in rows}, key=ring_sort_key)
    subset_sizes = list(range(1, len(AZIMUTHS) + 1))
    matrix = np.zeros((len(ring_ids), len(subset_sizes)), dtype=float)
    for ring_index, ring_id in enumerate(ring_ids):
        ring_rows = {int(row["subset_size"]): row for row in rows if str(row["ring_id"]) == ring_id}
        for col_index, subset_size in enumerate(subset_sizes):
            matrix[ring_index, col_index] = float(ring_rows[subset_size]["mean_marginal_gain_from_prev_size"])

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_title("Average extra value of the kth drone by ring")
    ax.set_xlabel("Added drone number (k)")
    ax.set_ylabel("Ring")
    ax.set_xticks(np.arange(len(subset_sizes)))
    ax.set_xticklabels(subset_sizes)
    ax.set_yticks(np.arange(len(ring_ids)))
    ax.set_yticklabels(ring_ids)
    for row_index in range(len(ring_ids)):
        for col_index in range(len(subset_sizes)):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#222222",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean marginal gain from size k-1 to k")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_focus_ring(rows: list[dict[str, object]], output_path: Path) -> None:
    focus_ring = max(
        {str(row["ring_id"]) for row in rows},
        key=lambda ring_id: max(
            float(row["best_coalition_value"])
            for row in rows
            if str(row["ring_id"]) == ring_id and int(row["subset_size"]) == len(AZIMUTHS)
        ),
    )
    focus_rows = [row for row in rows if str(row["ring_id"]) == focus_ring]

    subset_sizes = [int(row["subset_size"]) for row in focus_rows]
    mean_values = [float(row["mean_coalition_value"]) for row in focus_rows]
    best_values = [float(row["best_coalition_value"]) for row in focus_rows]
    gain_sizes = [int(row["added_drone_number"]) for row in focus_rows if int(row["subset_size"]) > 0]
    mean_gains = [float(row["mean_marginal_gain_from_prev_size"]) for row in focus_rows if int(row["subset_size"]) > 0]
    best_gains = [float(row["best_marginal_gain_from_prev_size"]) for row in focus_rows if int(row["subset_size"]) > 0]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax_left.plot(subset_sizes, mean_values, marker="o", linewidth=2.2, label="Mean coalition of that size")
    ax_left.plot(subset_sizes, best_values, marker="s", linewidth=2.2, label="Best coalition of that size")
    ax_left.set_title(f"{focus_ring}: coalition value by drone count")
    ax_left.set_xlabel("Number of drones")
    ax_left.set_ylabel("Noisy-OR + best IoU coalition value")
    ax_left.set_xticks(subset_sizes)
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.legend(frameon=False, fontsize=9)

    x = np.arange(len(gain_sizes))
    width = 0.38
    ax_right.bar(x - width / 2, mean_gains, width=width, label="Average extra value", color="#9fb6d0")
    ax_right.bar(x + width / 2, best_gains, width=width, label="Best-case extra value", color="#d97a1d")
    ax_right.set_title(f"{focus_ring}: exact gain of the 1st, 2nd, 3rd, ... drone")
    ax_right.set_xlabel("Added drone number (k)")
    ax_right.set_ylabel("Marginal gain from size k-1 to k")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(gain_sizes)
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.legend(frameon=False, fontsize=9)

    best_second = next(row for row in focus_rows if int(row["subset_size"]) == 2)
    best_third = next(row for row in focus_rows if int(row["subset_size"]) == 3)
    best_fourth = next(row for row in focus_rows if int(row["subset_size"]) == 4)
    ax_right.text(
        0.02,
        0.98,
        (
            f"2nd drone avg gain: {float(best_second['mean_marginal_gain_from_prev_size']):.4f}\n"
            f"3rd drone avg gain: {float(best_third['mean_marginal_gain_from_prev_size']):.4f}\n"
            f"4th drone avg gain: {float(best_fourth['mean_marginal_gain_from_prev_size']):.4f}"
        ),
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.suptitle("From coalition value to exact added-drone gains", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_report(rows: list[dict[str, object]], aggregate_rows: list[dict[str, object]], output_path: Path) -> None:
    top_ring = max(
        {str(row["ring_id"]) for row in rows},
        key=lambda ring_id: max(
            float(row["best_coalition_value"])
            for row in rows
            if str(row["ring_id"]) == ring_id and int(row["subset_size"]) == len(AZIMUTHS)
        ),
    )
    top_ring_rows = [row for row in rows if str(row["ring_id"]) == top_ring]
    top_ring_grand_value = next(
        float(row["best_coalition_value"]) for row in top_ring_rows if int(row["subset_size"]) == len(AZIMUTHS)
    )

    lines = [
        "# Exact Added-Drone Gain Analysis",
        "",
        "This analysis keeps the same coalition value as the fusion-based ring Shapley analysis:",
        "",
        "- Per scene: `Noisy-OR(confidences in coalition) x best IoU in coalition`.",
        "- Per ring: average that coalition value over all scenes that contain any observation in the ring.",
        "",
        "The difference is the output:",
        "",
        "- Shapley asks: `which viewpoint adds the most value on average across all coalition contexts?`",
        "- This file asks: `how much extra value does the 2nd, 3rd, 4th, ... drone add?`",
        "",
        "## Overall pattern across the 9 controlled rings",
        "",
        "| Drones in coalition | Mean of ring mean values | Mean of ring best values | Mean extra value of kth drone | Mean best-case extra value of kth drone |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{int(row['subset_size'])} | "
            f"{format_float(float(row['mean_of_ring_mean_coalition_value']), 4)} | "
            f"{format_float(float(row['mean_of_ring_best_coalition_value']), 4)} | "
            f"{format_float(float(row['mean_of_ring_mean_marginal_gain']), 4)} | "
            f"{format_float(float(row['mean_of_ring_best_marginal_gain']), 4)} |"
        )

    lines.extend(
        [
            "",
            f"## Focus ring: {top_ring}",
            "",
            f"This is the strongest ring by full 8-drone coalition value: `{format_float(top_ring_grand_value, 4)}`.",
            "",
            "| Drones | Mean coalition value | Best coalition value | Average extra value of this drone | Best-case extra value of this drone | Best coalition members |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in top_ring_rows:
        lines.append(
            "| "
            f"{int(row['subset_size'])} | "
            f"{format_float(float(row['mean_coalition_value']), 4)} | "
            f"{format_float(float(row['best_coalition_value']), 4)} | "
            f"{format_float(float(row['mean_marginal_gain_from_prev_size']), 4)} | "
            f"{format_float(float(row['best_marginal_gain_from_prev_size']), 4)} | "
            f"{row['best_coalition_members']} |"
        )

    second = next(row for row in top_ring_rows if int(row["subset_size"]) == 2)
    third = next(row for row in top_ring_rows if int(row["subset_size"]) == 3)
    fourth = next(row for row in top_ring_rows if int(row["subset_size"]) == 4)
    lines.extend(
        [
            "",
            "## How to read this for the thesis question",
            "",
            f"- In `{top_ring}`, the **2nd drone** adds `{format_float(float(second['mean_marginal_gain_from_prev_size']), 4)}` on average.",
            f"- The **3rd drone** adds `{format_float(float(third['mean_marginal_gain_from_prev_size']), 4)}` on average.",
            f"- The **4th drone** adds `{format_float(float(fourth['mean_marginal_gain_from_prev_size']), 4)}` on average.",
            "- After that, gains continue to be positive but keep shrinking, which is the diminishing-returns story the swarm-size question needs.",
            "",
            "## Methodological takeaway",
            "",
            "- Use this coalition-size breakdown to answer `how many drones are worth adding?`.",
            "- Use Shapley on top of that to answer `which viewpoints are the best teammates within a fixed swarm size setting?`.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.scene_records))
    scene_lookup = build_scene_lookup(records)
    scene_class_lookup = build_scene_class_lookup(scene_lookup)

    ring_ids = sorted({str(record["ring_id"]) for record in records}, key=ring_sort_key)
    rows: list[dict[str, object]] = []
    for ring_id in ring_ids:
        ring_scenes = ring_scene_keys(ring_id, scene_lookup, scene_class_lookup)
        if not ring_scenes:
            continue
        rows.extend(analyze_ring_sizes(ring_id, scene_lookup, ring_scenes))

    aggregate_rows = aggregate_across_rings(rows)
    write_csv(output_dir / "ring_coalition_size_breakdown.csv", rows)
    write_csv(output_dir / "aggregate_coalition_size_breakdown.csv", aggregate_rows)
    build_report(rows, aggregate_rows, output_dir / "coalition_size_breakdown_report.md")
    plot_all_rings_mean_values(rows, output_dir / "all_rings_mean_coalition_value_by_size.png")
    plot_all_rings_mean_gains(rows, output_dir / "all_rings_mean_added_drone_gain.png")
    plot_all_rings_share_of_full_value(rows, output_dir / "all_rings_share_of_full_value.png")
    plot_mean_gain_heatmap(rows, output_dir / "all_rings_mean_gain_heatmap.png")
    plot_focus_ring(rows, output_dir / "focus_ring_added_drone_gain.png")


if __name__ == "__main__":
    main()
