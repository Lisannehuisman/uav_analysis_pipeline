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
    analyze_ring_for_scene_keys,
    build_scene_class_lookup,
    build_scene_lookup,
    exact_shapley_from_values,
    format_float,
    load_records,
    ring_scene_keys,
    ring_sort_key,
    viewpoint_for_ring,
    write_csv,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = WORKSPACE / "results" / "recomputed" / "shapley_swarm_size_profiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose exact ring Shapley values by coalition size so the marginal value "
            "of the 1st, 2nd, 3rd, ... added view is explicit."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def build_size_conditioned_rows(
    ring_id: str,
    coalition_values: list[float],
    summary_row: dict[str, object],
) -> list[dict[str, object]]:
    n_players = len(AZIMUTHS)
    full_mask = (1 << n_players) - 1
    shapley_values = exact_shapley_from_values(coalition_values, n_players=n_players)
    rows: list[dict[str, object]] = []

    for player_index, azimuth in enumerate(AZIMUTHS):
        bit = 1 << player_index
        total_shapley = float(shapley_values[player_index])
        cumulative_component = 0.0
        for base_size in range(n_players):
            marginals = [
                float(coalition_values[base_mask | bit]) - float(coalition_values[base_mask])
                for base_mask in range(full_mask + 1)
                if not (base_mask & bit) and base_mask.bit_count() == base_size
            ]
            mean_marginal = nanmean(marginals)
            component = mean_marginal / n_players
            cumulative_component += component
            share = float("nan") if math.isclose(total_shapley, 0.0, abs_tol=1e-12) else component / total_shapley
            cumulative_share = (
                float("nan")
                if math.isclose(total_shapley, 0.0, abs_tol=1e-12)
                else cumulative_component / total_shapley
            )
            rows.append(
                {
                    "ring_id": ring_id,
                    "viewpoint": viewpoint_for_ring(ring_id, azimuth),
                    "azimuth": azimuth,
                    "base_coalition_size": base_size,
                    "added_drone_number": base_size + 1,
                    "num_contexts_of_this_size": len(marginals),
                    "mean_marginal_gain": mean_marginal,
                    "min_marginal_gain": float(np.min(marginals)),
                    "max_marginal_gain": float(np.max(marginals)),
                    "shapley_component_from_this_size": component,
                    "cumulative_shapley_through_this_size": cumulative_component,
                    "total_shapley_value": total_shapley,
                    "share_of_total_shapley_from_this_size": share,
                    "cumulative_share_of_total_shapley": cumulative_share,
                    "grand_value_all_8_players": float(summary_row["grand_value_all_8_players"]),
                    "ring_scene_count": int(summary_row["ring_scene_count"]),
                }
            )

    rows.sort(
        key=lambda row: (
            ring_sort_key(str(row["ring_id"])),
            int(row["added_drone_number"]),
            float(row["mean_marginal_gain"]),
            int(row["azimuth"]),
        ),
        reverse=False,
    )
    return rows


def build_ring_size_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ring_id"]), int(row["added_drone_number"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (ring_id, added_drone_number), members in sorted(
        grouped.items(),
        key=lambda item: (ring_sort_key(item[0][0]), item[0][1]),
    ):
        ordered = sorted(
            members,
            key=lambda row: (
                float(row["mean_marginal_gain"]),
                float(row["shapley_component_from_this_size"]),
                -int(row["azimuth"]),
            ),
            reverse=True,
        )
        top_row = ordered[0]
        second_row = ordered[1] if len(ordered) > 1 else ordered[0]
        summary_rows.append(
            {
                "ring_id": ring_id,
                "base_coalition_size": int(top_row["base_coalition_size"]),
                "added_drone_number": added_drone_number,
                "mean_marginal_gain_across_all_azimuths": nanmean(
                    [float(row["mean_marginal_gain"]) for row in ordered]
                ),
                "mean_shapley_component_across_all_azimuths": nanmean(
                    [float(row["shapley_component_from_this_size"]) for row in ordered]
                ),
                "mean_cumulative_share_of_total_shapley": nanmean(
                    [float(row["cumulative_share_of_total_shapley"]) for row in ordered]
                ),
                "top_viewpoint": str(top_row["viewpoint"]),
                "top_azimuth": int(top_row["azimuth"]),
                "top_mean_marginal_gain": float(top_row["mean_marginal_gain"]),
                "top_shapley_component_from_this_size": float(top_row["shapley_component_from_this_size"]),
                "second_viewpoint": str(second_row["viewpoint"]),
                "second_azimuth": int(second_row["azimuth"]),
                "second_mean_marginal_gain": float(second_row["mean_marginal_gain"]),
                "top_minus_second_marginal_gap": float(top_row["mean_marginal_gain"]) - float(second_row["mean_marginal_gain"]),
            }
        )
    return summary_rows


def build_aggregate_summary(
    detail_rows: list[dict[str, object]],
    ring_size_summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    detail_by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    top_by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        detail_by_size[int(row["added_drone_number"])].append(row)
    for row in ring_size_summary_rows:
        top_by_size[int(row["added_drone_number"])].append(row)

    aggregate_rows: list[dict[str, object]] = []
    for added_drone_number in sorted(detail_by_size):
        detail_members = detail_by_size[added_drone_number]
        top_members = top_by_size[added_drone_number]
        aggregate_rows.append(
            {
                "added_drone_number": added_drone_number,
                "base_coalition_size": added_drone_number - 1,
                "num_ring_player_profiles": len(detail_members),
                "num_rings": len(top_members),
                "mean_marginal_gain_across_all_ring_players": nanmean(
                    [float(row["mean_marginal_gain"]) for row in detail_members]
                ),
                "mean_shapley_component_across_all_ring_players": nanmean(
                    [float(row["shapley_component_from_this_size"]) for row in detail_members]
                ),
                "mean_share_of_total_shapley_from_this_size": nanmean(
                    [float(row["share_of_total_shapley_from_this_size"]) for row in detail_members]
                ),
                "mean_cumulative_share_of_total_shapley": nanmean(
                    [float(row["cumulative_share_of_total_shapley"]) for row in detail_members]
                ),
                "mean_top_marginal_gain_across_rings": nanmean(
                    [float(row["top_mean_marginal_gain"]) for row in top_members]
                ),
                "mean_top_minus_second_gap_across_rings": nanmean(
                    [float(row["top_minus_second_marginal_gap"]) for row in top_members]
                ),
            }
        )
    return aggregate_rows


def plot_aggregate_progression(aggregate_rows: list[dict[str, object]], output_path: Path) -> None:
    x = [int(row["added_drone_number"]) for row in aggregate_rows]
    mean_gain = [float(row["mean_marginal_gain_across_all_ring_players"]) for row in aggregate_rows]
    mean_top_gain = [float(row["mean_top_marginal_gain_across_rings"]) for row in aggregate_rows]
    cumulative_share = [float(row["mean_cumulative_share_of_total_shapley"]) for row in aggregate_rows]

    fig, ax_left = plt.subplots(figsize=(9.5, 5.5))
    ax_left.plot(x, mean_gain, marker="o", linewidth=2.2, color="#1f6f8b", label="Mean marginal gain")
    ax_left.plot(x, mean_top_gain, marker="s", linewidth=2.2, color="#d97706", label="Mean top azimuth gain")
    ax_left.set_xlabel("Added drone number (k)")
    ax_left.set_ylabel("Exact mean marginal gain")
    ax_left.set_xticks(x)
    ax_left.grid(axis="y", alpha=0.25)

    ax_right = ax_left.twinx()
    ax_right.plot(
        x,
        cumulative_share,
        marker="^",
        linewidth=2.0,
        color="#4d7c0f",
        label="Cumulative share of total Shapley",
    )
    ax_right.set_ylabel("Cumulative share of final Shapley")
    ax_right.set_ylim(0.0, 1.02)

    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, frameon=False, loc="center right")
    ax_left.set_title("Size-conditioned Shapley progression across controlled rings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_focus_ring_heatmap(
    detail_rows: list[dict[str, object]],
    focus_ring: str,
    output_path: Path,
) -> None:
    ring_rows = [row for row in detail_rows if str(row["ring_id"]) == focus_ring]
    matrix = np.full((len(AZIMUTHS), len(AZIMUTHS)), np.nan, dtype=float)
    for row in ring_rows:
        azimuth_index = AZIMUTHS.index(int(row["azimuth"]))
        size_index = int(row["added_drone_number"]) - 1
        matrix[azimuth_index, size_index] = float(row["mean_marginal_gain"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title(f"{focus_ring}: marginal value of each azimuth by swarm size")
    ax.set_xlabel("Added drone number (k)")
    ax.set_ylabel("Azimuth")
    ax.set_xticks(np.arange(len(AZIMUTHS)))
    ax.set_xticklabels(range(1, len(AZIMUTHS) + 1))
    ax.set_yticks(np.arange(len(AZIMUTHS)))
    ax.set_yticklabels([f"az{azimuth:03d}" for azimuth in AZIMUTHS])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#102a43",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean marginal gain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_focus_ring_top_progression(
    ring_size_summary_rows: list[dict[str, object]],
    focus_ring: str,
    output_path: Path,
) -> None:
    ring_rows = [row for row in ring_size_summary_rows if str(row["ring_id"]) == focus_ring]
    x = [int(row["added_drone_number"]) for row in ring_rows]
    y = [float(row["top_mean_marginal_gain"]) for row in ring_rows]
    labels = [f"az{int(row['top_azimuth']):03d}" for row in ring_rows]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(x, y, color="#2b6cb0")
    ax.set_title(f"{focus_ring}: best azimuth at each swarm size")
    ax.set_xlabel("Added drone number (k)")
    ax.set_ylabel("Top mean marginal gain")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    for bar, label in zip(bars, labels, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_report(
    aggregate_rows: list[dict[str, object]],
    ring_size_summary_rows: list[dict[str, object]],
    focus_ring: str,
    focus_ring_grand_value: float,
    output_path: Path,
) -> None:
    focus_rows = [row for row in ring_size_summary_rows if str(row["ring_id"]) == focus_ring]
    lines = [
        "# Size-Conditioned Shapley Progression",
        "",
        "This analysis keeps the exact same fusion game as the controlled ring Shapley analysis.",
        "",
        "- Coalition value: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.",
        "- Players: the 8 azimuths inside one fixed M4 ring.",
        "- Difference from the standard report: Shapley is decomposed by coalition size.",
        "",
        "For one viewpoint `u`, the exact Shapley value can be rewritten as:",
        "",
        "`phi(u) = (1 / n) * sum_k average_{|S| = k}[v(S union {u}) - v(S)]`",
        "",
        "So each row below answers a swarm-size question directly:",
        "",
        "- `k = 1`: how much does a view add as the first drone?",
        "- `k = 2`: how much does it add when one drone is already present?",
        "- `k = 3`: how much does it add when a pair is already present?",
        "- and so on.",
        "",
        "## Overall progression across the 9 controlled rings",
        "",
        "| Added drone number | Mean marginal gain | Mean exact Shapley component from this size | Mean cumulative share of final Shapley | Mean best azimuth gain | Mean top-vs-runner-up gap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{int(row['added_drone_number'])} | "
            f"{format_float(float(row['mean_marginal_gain_across_all_ring_players']), 4)} | "
            f"{format_float(float(row['mean_shapley_component_across_all_ring_players']), 4)} | "
            f"{format_float(float(row['mean_cumulative_share_of_total_shapley']), 4)} | "
            f"{format_float(float(row['mean_top_marginal_gain_across_rings']), 4)} | "
            f"{format_float(float(row['mean_top_minus_second_gap_across_rings']), 4)} |"
        )

    lines.extend(
        [
            "",
            f"## Focus ring: {focus_ring}",
            "",
            f"This is the strongest ring by full 8-drone coalition value: `{format_float(focus_ring_grand_value, 4)}`.",
            "",
            "| Added drone number | Mean marginal gain across azimuths | Best azimuth at this size | Top marginal gain | Runner-up | Gap |",
            "| ---: | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for row in focus_rows:
        lines.append(
            "| "
            f"{int(row['added_drone_number'])} | "
            f"{format_float(float(row['mean_marginal_gain_across_all_azimuths']), 4)} | "
            f"`az{int(row['top_azimuth']):03d}` | "
            f"{format_float(float(row['top_mean_marginal_gain']), 4)} | "
            f"`az{int(row['second_azimuth']):03d}` | "
            f"{format_float(float(row['top_minus_second_marginal_gap']), 4)} |"
        )

    second_row = next(row for row in aggregate_rows if int(row["added_drone_number"]) == 2)
    third_row = next(row for row in aggregate_rows if int(row["added_drone_number"]) == 3)
    fourth_row = next(row for row in aggregate_rows if int(row["added_drone_number"]) == 4)
    lines.extend(
        [
            "",
            "## Thesis reading",
            "",
            f"- The average gain of the **2nd drone** is `{format_float(float(second_row['mean_marginal_gain_across_all_ring_players']), 4)}`.",
            f"- The average gain of the **3rd drone** is `{format_float(float(third_row['mean_marginal_gain_across_all_ring_players']), 4)}`.",
            f"- The average gain of the **4th drone** is `{format_float(float(fourth_row['mean_marginal_gain_across_all_ring_players']), 4)}`.",
            "- Gains remain positive but shrink with swarm size, so the main question becomes `how many angles are worth adding?` before `which exact angle is best?`.",
            "- Within any fixed swarm size, the per-ring top azimuth still tells you which angle is the best teammate at that stage.",
            "",
            "## Practical takeaway",
            "",
            "- Use `aggregate_shapley_swarm_size_summary.csv` when the headline question is the number of views.",
            "- Use `ring_shapley_swarm_size_summary.csv` when the question is which angle to add at swarm size `k`.",
            "- Use the standard ring Shapley report when you want one overall teammate ranking aggregated across all coalition sizes.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.scene_records))
    scene_lookup = build_scene_lookup(records)
    scene_class_lookup = build_scene_class_lookup(scene_lookup)
    ring_ids = sorted({str(record["ring_id"]) for record in records}, key=ring_sort_key)

    detail_rows: list[dict[str, object]] = []
    ring_full_values: dict[str, float] = {}
    for ring_id in ring_ids:
        scene_keys = ring_scene_keys(ring_id, scene_lookup, scene_class_lookup=scene_class_lookup, target_class=None)
        if not scene_keys:
            continue
        _detail, summary_row, coalition_values, _conditional_rows, _conditional_summary_rows = analyze_ring_for_scene_keys(
            ring_id,
            scene_lookup,
            scene_keys,
            group="overall",
            target_class="all",
        )
        ring_full_values[ring_id] = float(summary_row["grand_value_all_8_players"])
        detail_rows.extend(build_size_conditioned_rows(ring_id, coalition_values, summary_row))

    ring_size_summary_rows = build_ring_size_summary(detail_rows)
    aggregate_rows = build_aggregate_summary(detail_rows, ring_size_summary_rows)

    focus_ring = max(ring_full_values, key=ring_full_values.get)
    focus_ring_grand_value = ring_full_values[focus_ring]

    write_csv(output_dir / "ring_shapley_swarm_size_detail.csv", detail_rows)
    write_csv(output_dir / "ring_shapley_swarm_size_summary.csv", ring_size_summary_rows)
    write_csv(output_dir / "aggregate_shapley_swarm_size_summary.csv", aggregate_rows)
    build_report(
        aggregate_rows,
        ring_size_summary_rows,
        focus_ring,
        focus_ring_grand_value,
        output_dir / "shapley_swarm_size_progression_report.md",
    )
    plot_aggregate_progression(aggregate_rows, output_dir / "aggregate_shapley_swarm_size_progression.png")
    plot_focus_ring_heatmap(detail_rows, focus_ring, output_dir / "focus_ring_shapley_swarm_size_heatmap.png")
    plot_focus_ring_top_progression(
        ring_size_summary_rows,
        focus_ring,
        output_dir / "focus_ring_top_azimuth_by_swarm_size.png",
    )


if __name__ == "__main__":
    main()
