from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_RECORDS = WORKSPACE / "results" / "intermediate" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "results" / "recomputed" / "shapley_explanation"
AZIMUTHS = list(range(0, 360, 45))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple, actual-data explanation of Noisy-OR + best IoU Shapley for one M4 ring."
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--ring-id", default="elmid-radnear")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_float(raw: object) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    return float(text)


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def viewpoint_for_ring(ring_id: str, azimuth: int) -> str:
    return f"{ring_id}-az{azimuth:03d}"


def azimuths_for_mask(coalition_mask: int) -> list[int]:
    return [AZIMUTHS[idx] for idx in range(len(AZIMUTHS)) if coalition_mask & (1 << idx)]


def coalition_members_text(coalition_mask: int) -> str:
    members = [f"az{azimuth:03d}" for azimuth in azimuths_for_mask(coalition_mask)]
    return " + ".join(members) if members else "(empty)"


def best_mask_for_size(coalition_values: list[float], subset_size: int) -> int:
    candidate_masks = [mask for mask in range(len(coalition_values)) if mask.bit_count() == subset_size]
    if not candidate_masks:
        raise ValueError(f"No coalition found for subset size {subset_size}")
    return max(candidate_masks, key=lambda mask: (float(coalition_values[mask]), -mask))


def conditional_gain_rows_for_base_mask(
    ring_id: str,
    coalition_values: list[float],
    base_mask: int,
) -> list[dict[str, object]]:
    base_value = float(coalition_values[base_mask])
    rows: list[dict[str, object]] = []
    for player_index, azimuth in enumerate(AZIMUTHS):
        bit = 1 << player_index
        if base_mask & bit:
            continue
        expanded_mask = base_mask | bit
        expanded_value = float(coalition_values[expanded_mask])
        rows.append(
            {
                "ring_id": ring_id,
                "base_coalition_mask": base_mask,
                "base_coalition_size": base_mask.bit_count(),
                "base_coalition_members": coalition_members_text(base_mask),
                "candidate_azimuth": azimuth,
                "candidate_viewpoint": viewpoint_for_ring(ring_id, azimuth),
                "expanded_coalition_mask": expanded_mask,
                "expanded_coalition_size": expanded_mask.bit_count(),
                "expanded_coalition_members": coalition_members_text(expanded_mask),
                "base_value": base_value,
                "expanded_value": expanded_value,
                "delta_value": expanded_value - base_value,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["delta_value"]),
            float(row["expanded_value"]),
            -int(row["candidate_azimuth"]),
        ),
        reverse=True,
    )
    return rows


def load_records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "scene_key": row["scene_key"],
                    "viewpoint": row["viewpoint"],
                    "ring_id": f"el{row['elevation']}-rad{row['radius']}",
                    "azimuth": int(float(row["azimuth"])),
                    "confidence": 0.0
                    if math.isnan(parse_float(row["target_match_confidence_iou50"]))
                    else parse_float(row["target_match_confidence_iou50"]),
                    "iou": 0.0
                    if math.isnan(parse_float(row["target_match_iou_at_confidence_iou50"]))
                    else parse_float(row["target_match_iou_at_confidence_iou50"]),
                    "strict_quality": 0.0
                    if math.isnan(parse_float(row["target_strict_quality_iou50"]))
                    else parse_float(row["target_strict_quality_iou50"]),
                }
            )
    return rows


def build_scene_lookup(records: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    by_scene: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in records:
        by_scene[str(record["scene_key"])][str(record["viewpoint"])] = record
    return by_scene


@lru_cache(maxsize=None)
def factorial(value: int) -> int:
    return math.factorial(value)


def exact_shapley_from_values(coalition_values: list[float], n_players: int) -> np.ndarray:
    shapley = np.zeros(n_players, dtype=float)
    n_factorial = factorial(n_players)
    full_mask = (1 << n_players) - 1
    for player in range(n_players):
        bit = 1 << player
        for mask in range(full_mask + 1):
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            weight = factorial(subset_size) * factorial(n_players - subset_size - 1) / n_factorial
            shapley[player] += weight * (coalition_values[mask | bit] - coalition_values[mask])
    return shapley


def coalition_value_for_scene(
    scene_records: dict[str, dict[str, object]],
    ring_id: str,
    coalition_mask: int,
) -> float:
    confidences: list[float] = []
    best_iou = 0.0
    for player_index, azimuth in enumerate(AZIMUTHS):
        if not (coalition_mask & (1 << player_index)):
            continue
        record = scene_records.get(viewpoint_for_ring(ring_id, azimuth))
        if record is None:
            continue
        confidence = float(record["confidence"])
        iou = float(record["iou"])
        if confidence > 0.0:
            confidences.append(confidence)
        if iou > best_iou:
            best_iou = iou
    return noisy_or(confidences) * best_iou if best_iou > 0.0 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.scene_records))
    scene_lookup = build_scene_lookup(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ring_id = str(args.ring_id)
    ring_viewpoints = [viewpoint_for_ring(ring_id, azimuth) for azimuth in AZIMUTHS]
    ring_scenes = sorted(
        scene_key
        for scene_key, scene_records in scene_lookup.items()
        if any(viewpoint in scene_records for viewpoint in ring_viewpoints)
    )
    if not ring_scenes:
        raise ValueError(f"No scenes found for ring {ring_id}")

    full_mask = (1 << len(AZIMUTHS)) - 1
    coalition_values: list[float] = []
    coalition_rows: list[dict[str, object]] = []
    size_to_values: dict[int, list[float]] = defaultdict(list)
    size_to_best: dict[int, tuple[float, int]] = {}
    for coalition_mask in range(full_mask + 1):
        coalition_scene_values = [
            coalition_value_for_scene(scene_lookup[scene_key], ring_id, coalition_mask)
            for scene_key in ring_scenes
        ]
        coalition_value = float(np.mean(coalition_scene_values))
        coalition_values.append(coalition_value)
        subset_size = coalition_mask.bit_count()
        size_to_values[subset_size].append(coalition_value)
        previous = size_to_best.get(subset_size)
        if previous is None or coalition_value > previous[0]:
            size_to_best[subset_size] = (coalition_value, coalition_mask)

        members = [f"az{AZIMUTHS[idx]:03d}" for idx in range(len(AZIMUTHS)) if coalition_mask & (1 << idx)]
        coalition_rows.append(
            {
                "ring_id": ring_id,
                "coalition_mask": coalition_mask,
                "subset_size": subset_size,
                "coalition_members": " ".join(members) if members else "(empty)",
                "coalition_value": coalition_value,
            }
        )

    shapley = exact_shapley_from_values(coalition_values, n_players=len(AZIMUTHS))

    viewpoint_rows: list[dict[str, object]] = []
    for player_index, azimuth in enumerate(AZIMUTHS):
        singleton_mask = 1 << player_index
        viewpoint_rows.append(
            {
                "ring_id": ring_id,
                "azimuth": azimuth,
                "viewpoint": viewpoint_for_ring(ring_id, azimuth),
                "singleton_value": float(coalition_values[singleton_mask]),
                "shapley_value": float(shapley[player_index]),
            }
        )
    viewpoint_rows.sort(key=lambda row: int(row["azimuth"]))

    size_rows: list[dict[str, object]] = []
    for subset_size in sorted(size_to_values):
        best_value, best_mask = size_to_best[subset_size]
        size_rows.append(
            {
                "ring_id": ring_id,
                "subset_size": subset_size,
                "mean_coalition_value": float(np.mean(size_to_values[subset_size])),
                "best_coalition_value": best_value,
                "best_coalition_members": coalition_members_text(best_mask),
                "num_coalitions": len(size_to_values[subset_size]),
            }
        )

    top_singleton_mask = best_mask_for_size(coalition_values, 1)
    best_pair_mask = best_mask_for_size(coalition_values, 2)
    top_singleton_extension = conditional_gain_rows_for_base_mask(ring_id, coalition_values, top_singleton_mask)[0]
    best_pair_extension = conditional_gain_rows_for_base_mask(ring_id, coalition_values, best_pair_mask)[0]
    conditional_rows = []
    for subset_size in range(1, len(AZIMUTHS)):
        base_mask = best_mask_for_size(coalition_values, subset_size)
        best_extension = conditional_gain_rows_for_base_mask(ring_id, coalition_values, base_mask)[0]
        conditional_rows.append(
            {
                "ring_id": ring_id,
                "base_coalition_size": subset_size,
                "base_coalition_members": coalition_members_text(base_mask),
                "base_value": float(coalition_values[base_mask]),
                "best_addition_azimuth": int(best_extension["candidate_azimuth"]),
                "best_addition_viewpoint": str(best_extension["candidate_viewpoint"]),
                "delta_value": float(best_extension["delta_value"]),
                "expanded_coalition_members": str(best_extension["expanded_coalition_members"]),
                "expanded_value": float(best_extension["expanded_value"]),
            }
        )

    base_name = ring_id.replace("-", "_")
    write_csv(output_dir / f"{base_name}_coalition_values.csv", coalition_rows)
    write_csv(output_dir / f"{base_name}_coalition_size_profile.csv", size_rows)
    write_csv(output_dir / f"{base_name}_viewpoint_shapley.csv", viewpoint_rows)
    write_csv(output_dir / f"{base_name}_conditional_extensions.csv", conditional_rows)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

    subset_sizes = [int(row["subset_size"]) for row in size_rows]
    mean_values = [float(row["mean_coalition_value"]) for row in size_rows]
    best_values = [float(row["best_coalition_value"]) for row in size_rows]
    ax_left.plot(subset_sizes, mean_values, marker="o", linewidth=2.2, label="Mean over all coalitions")
    ax_left.plot(subset_sizes, best_values, marker="s", linewidth=2.2, label="Best coalition of that size")
    ax_left.set_title(f"{ring_id}: coalition value by team size")
    ax_left.set_xlabel("Coalition size")
    ax_left.set_ylabel("Scene-balanced Noisy-OR + best IoU value")
    ax_left.set_xticks(subset_sizes)
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.legend(frameon=False, fontsize=9)

    az_labels = [f"{int(row['azimuth']):03d}" for row in viewpoint_rows]
    x = np.arange(len(az_labels))
    width = 0.38
    singleton_values = [float(row["singleton_value"]) for row in viewpoint_rows]
    shapley_values = [float(row["shapley_value"]) for row in viewpoint_rows]
    ax_right.bar(x - width / 2, singleton_values, width=width, label="Singleton coalition value v({u})", color="#9fb6d0")
    ax_right.bar(x + width / 2, shapley_values, width=width, label="Exact Shapley value", color="#d97a1d")
    ax_right.set_title(f"{ring_id}: singleton coalition value versus Shapley")
    ax_right.set_xlabel("Azimuth")
    ax_right.set_ylabel("Value")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(az_labels)
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.legend(frameon=False, fontsize=9)

    top_singleton = max(viewpoint_rows, key=lambda row: float(row["singleton_value"]))
    top_shapley = max(viewpoint_rows, key=lambda row: float(row["shapley_value"]))
    ax_right.text(
        0.02,
        0.98,
        f"Top singleton in this ring: az{int(top_singleton['azimuth']):03d}\nTop Shapley in this ring: az{int(top_shapley['azimuth']):03d}",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.suptitle("How Noisy-OR + best IoU becomes a Shapley game", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(output_dir / f"{base_name}_shapley_explanation.png", dpi=220)
    plt.close(fig)

    top_singleton_az = int(top_singleton["azimuth"])
    top_shapley_az = int(top_shapley["azimuth"])
    explanation_lines = [
        f"# Noisy-OR Shapley Explanation for {ring_id}",
        "",
        f"- Input scenes: `{len(ring_scenes)}`",
        "- Players: the 8 azimuth viewpoints in this fixed elevation-radius ring.",
        "- Coalition value: per scene, fuse the selected viewpoints with `Noisy-OR + best IoU`, then average over scenes.",
        "- Singleton coalition value means `v({u})`: the score of a one-view coalition containing exactly one azimuth in this fixed ring.",
        "- Shapley value: for each azimuth, average how much it increases coalition value when added to all possible subsets of the other 7 azimuths.",
        "- `Highest Shapley azimuth` therefore means: the azimuth with the largest exact average marginal contribution **within this ring only**, not one universal best angle across all rings.",
        "- For one concrete current coalition `C`, the next-view rule is `Delta(u | C) = v(C union {u}) - v(C)` under the same Noisy-OR + best IoU coalition value.",
        "",
        "## Reading the plot",
        "",
        "- Left panel: the coalition value grows as more viewpoints are available, but with diminishing returns.",
        "- Right panel: singleton value and Shapley value are not identical.",
        f"- In this ring, the best singleton azimuth is `az{top_singleton_az:03d}` with `v({{az{top_singleton_az:03d}}}) = {float(top_singleton['singleton_value']):.4f}`, but the highest Shapley azimuth is `az{top_shapley_az:03d}` with `phi = {float(top_shapley['shapley_value']):.4f}`.",
        "- That means the best individual viewpoint is not automatically the best collaborative viewpoint.",
        "",
        "## Concrete next-view example in the same ring",
        "",
        f"- Starting from the best singleton coalition `C = {{{coalition_members_text(top_singleton_mask)}}}`, the best second azimuth is `az{int(top_singleton_extension['candidate_azimuth']):03d}`.",
        f"- That raises coalition value from `{float(top_singleton_extension['base_value']):.4f}` to `{float(top_singleton_extension['expanded_value']):.4f}`, so `Delta(u | C) = {float(top_singleton_extension['delta_value']):.4f}`.",
        f"- Starting from the best pair `C = {{{coalition_members_text(best_pair_mask)}}}`, the best third azimuth is `az{int(best_pair_extension['candidate_azimuth']):03d}`.",
        f"- That raises coalition value from `{float(best_pair_extension['base_value']):.4f}` to `{float(best_pair_extension['expanded_value']):.4f}`, so `Delta(u | C) = {float(best_pair_extension['delta_value']):.4f}`.",
        "- This is the right operational quantity if the question is not `who is best on average?` but `which extra drone should I add to the team I already have?`.",
        "",
        "## Why this is still the best fusion method",
        "",
        "- The coalition game uses the thesis-best coalition rule directly: `Noisy-OR + best IoU`.",
        "- So Shapley is no longer answering `which viewpoint has the best max score?`.",
        "- Instead, Shapley answers `which viewpoint adds the most value on average when coalition value is defined by the best fusion rule?`.",
        "- The conditional gain table then complements that with the concrete decision rule for extending one specific current coalition.",
    ]
    (output_dir / f"{base_name}_shapley_explanation.md").write_text("\n".join(explanation_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
