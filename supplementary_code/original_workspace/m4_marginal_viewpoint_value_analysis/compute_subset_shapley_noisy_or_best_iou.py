from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compute_ring_shapley_noisy_or_best_iou import (
    AZIMUTHS,
    DEFAULT_SCENE_RECORDS,
    ELEVATION_ORDER,
    RADIUS_ORDER,
    WORKSPACE,
    build_scene_class_lookup,
    build_scene_lookup,
    exact_shapley_from_values,
    format_float,
    load_records,
    noisy_or,
    write_csv,
)


DEFAULT_OUTPUT_DIR = (
    WORKSPACE / "m4_marginal_viewpoint_value_analysis" / "outputs" / "subset_shapley_noisy_or_best_iou"
)
DEFAULT_SUBSET_NAME = "full_grid_perimeter_8"

PERIMETER_RING_IDS = [
    "ellow-radnear",
    "ellow-radmid",
    "ellow-radfar",
    "elmid-radfar",
    "elhigh-radfar",
    "elhigh-radmid",
    "elhigh-radnear",
    "elmid-radnear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact Shapley values for an arbitrary viewpoint subset using "
            "Noisy-OR + best IoU as the coalition value."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset-name", default=DEFAULT_SUBSET_NAME)
    parser.add_argument(
        "--selection-mode",
        default="full_grid_perimeter",
        choices=["full_grid_perimeter", "manual"],
        help=(
            "How to choose the players. "
            "`full_grid_perimeter` spreads 8 viewpoints over the outer 3x3 ring grid "
            "and uses all 8 azimuths once. `manual` uses --viewpoints as-is."
        ),
    )
    parser.add_argument(
        "--viewpoints",
        nargs="*",
        default=[],
        help="Explicit viewpoints to analyze when --selection-mode manual is used.",
    )
    parser.add_argument(
        "--min-scenes",
        type=int,
        default=25,
        help="Warn when the selected subset has fewer than this many scenes with any selected viewpoint.",
    )
    return parser.parse_args()


def strip_prefix(token: str, prefix: str) -> str:
    return token[len(prefix) :] if token.startswith(prefix) else token


def viewpoint_parts(viewpoint: str) -> tuple[str, str, int]:
    parts = viewpoint.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unexpected viewpoint token: {viewpoint}")
    elevation = strip_prefix(parts[0], "el")
    radius = strip_prefix(parts[1], "rad")
    azimuth = int(parts[2].replace("az", ""))
    return elevation, radius, azimuth


def viewpoint_sort_key(viewpoint: str) -> tuple[int, int, int]:
    elevation, radius, azimuth = viewpoint_parts(viewpoint)
    return (ELEVATION_ORDER.get(elevation, 99), RADIUS_ORDER.get(radius, 99), azimuth)


def ring_id_for_viewpoint(viewpoint: str) -> str:
    elevation, radius, _azimuth = viewpoint_parts(viewpoint)
    return f"el{elevation}-rad{radius}"


def coalition_members_text(coalition_mask: int, players: list[str]) -> str:
    members = [players[idx] for idx in range(len(players)) if coalition_mask & (1 << idx)]
    return " + ".join(members) if members else "(empty)"


def best_mask_for_size(coalition_values: list[float], subset_size: int) -> int:
    candidate_masks = [mask for mask in range(len(coalition_values)) if mask.bit_count() == subset_size]
    if not candidate_masks:
        raise ValueError(f"No coalition found for subset size {subset_size}")
    return max(candidate_masks, key=lambda mask: (float(coalition_values[mask]), -mask))


def select_full_grid_perimeter(available_viewpoints: list[str]) -> list[str]:
    available = set(available_viewpoints)
    selected = [f"{ring_id}-az{azimuth:03d}" for ring_id, azimuth in zip(PERIMETER_RING_IDS, AZIMUTHS, strict=True)]
    missing = sorted(viewpoint for viewpoint in selected if viewpoint not in available)
    if missing:
        raise ValueError(f"Full-grid perimeter selection could not find viewpoints: {missing}")
    return selected


def resolve_players(args: argparse.Namespace, available_viewpoints: list[str]) -> tuple[list[str], str]:
    if args.selection_mode == "manual":
        if not args.viewpoints:
            raise ValueError("--selection-mode manual requires at least one viewpoint via --viewpoints")
        players = sorted(dict.fromkeys(args.viewpoints), key=viewpoint_sort_key)
    else:
        players = select_full_grid_perimeter(available_viewpoints)

    missing = sorted(viewpoint for viewpoint in players if viewpoint not in set(available_viewpoints))
    if missing:
        raise ValueError(f"Selected viewpoints are not present in the scene records: {missing}")
    if len(players) < 2:
        raise ValueError("At least two viewpoints are required for a multi-player Shapley analysis")

    selection_label = (
        "8 viewpoints on the perimeter of the 3x3 elevation/radius grid, one distinct azimuth each"
        if args.selection_mode == "full_grid_perimeter"
        else "manual viewpoint subset"
    )
    return players, selection_label


def coalition_value_for_scene(
    scene_records: dict[str, dict[str, object]],
    players: list[str],
    coalition_mask: int,
) -> float:
    confidences: list[float] = []
    best_iou = 0.0
    for player_index, viewpoint in enumerate(players):
        if not (coalition_mask & (1 << player_index)):
            continue
        record = scene_records.get(viewpoint)
        if record is None:
            continue
        confidence = float(record["confidence"])
        iou = float(record["iou"])
        if confidence > 0.0:
            confidences.append(confidence)
        if iou > best_iou:
            best_iou = iou
    return noisy_or(confidences) * best_iou if best_iou > 0.0 else 0.0


def subset_scene_keys(
    players: list[str],
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    scene_class_lookup: dict[str, str] | None = None,
    target_class: str | None = None,
) -> list[str]:
    player_set = set(players)
    selected: list[str] = []
    for scene_key, scene_records in scene_lookup.items():
        if target_class is not None and scene_class_lookup is not None and scene_class_lookup[scene_key] != target_class:
            continue
        if any(viewpoint in player_set for viewpoint in scene_records):
            selected.append(scene_key)
    selected.sort()
    return selected


def conditional_gain_rows_for_base_mask(
    subset_name: str,
    players: list[str],
    coalition_values: list[float],
    base_mask: int,
    *,
    group: str,
    target_class: str,
) -> list[dict[str, object]]:
    base_value = float(coalition_values[base_mask])
    rows: list[dict[str, object]] = []
    for player_index, viewpoint in enumerate(players):
        bit = 1 << player_index
        if base_mask & bit:
            continue
        expanded_mask = base_mask | bit
        expanded_value = float(coalition_values[expanded_mask])
        rows.append(
            {
                "group": group,
                "target_class": target_class,
                "subset_name": subset_name,
                "player_count": len(players),
                "base_coalition_mask": base_mask,
                "base_coalition_size": base_mask.bit_count(),
                "base_coalition_members": coalition_members_text(base_mask, players),
                "candidate_player_index": player_index + 1,
                "candidate_viewpoint": viewpoint,
                "expanded_coalition_mask": expanded_mask,
                "expanded_coalition_size": expanded_mask.bit_count(),
                "expanded_coalition_members": coalition_members_text(expanded_mask, players),
                "base_value": base_value,
                "expanded_value": expanded_value,
                "delta_value": expanded_value - base_value,
                "rank_within_base_coalition": 0,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["delta_value"]),
            float(row["expanded_value"]),
            -int(row["candidate_player_index"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank_within_base_coalition"] = rank
    return rows


def build_conditional_gain_rows(
    subset_name: str,
    players: list[str],
    coalition_values: list[float],
    *,
    group: str,
    target_class: str,
) -> list[dict[str, object]]:
    full_mask = (1 << len(players)) - 1
    rows: list[dict[str, object]] = []
    for base_mask in range(full_mask + 1):
        if base_mask == full_mask:
            continue
        rows.extend(
            conditional_gain_rows_for_base_mask(
                subset_name,
                players,
                coalition_values,
                base_mask,
                group=group,
                target_class=target_class,
            )
        )

    rows.sort(
        key=lambda row: (
            str(row["subset_name"]),
            int(row["base_coalition_size"]),
            int(row["base_coalition_mask"]),
            int(row["rank_within_base_coalition"]),
        )
    )
    return rows


def summarize_best_extensions_by_size(
    subset_name: str,
    players: list[str],
    coalition_values: list[float],
    *,
    group: str,
    target_class: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base_size in range(1, len(players)):
        base_mask = best_mask_for_size(coalition_values, base_size)
        best_extension = conditional_gain_rows_for_base_mask(
            subset_name,
            players,
            coalition_values,
            base_mask,
            group=group,
            target_class=target_class,
        )[0]
        rows.append(
            {
                "group": group,
                "target_class": target_class,
                "subset_name": subset_name,
                "player_count": len(players),
                "base_coalition_mask": base_mask,
                "base_coalition_size": base_size,
                "base_coalition_members": coalition_members_text(base_mask, players),
                "base_value": float(coalition_values[base_mask]),
                "best_addition_player_index": best_extension["candidate_player_index"],
                "best_addition_viewpoint": best_extension["candidate_viewpoint"],
                "expanded_coalition_mask": best_extension["expanded_coalition_mask"],
                "expanded_coalition_size": best_extension["expanded_coalition_size"],
                "expanded_coalition_members": best_extension["expanded_coalition_members"],
                "expanded_value": best_extension["expanded_value"],
                "delta_value": best_extension["delta_value"],
            }
        )
    return rows


def analyze_subset_for_scene_keys(
    subset_name: str,
    players: list[str],
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    scene_keys: list[str],
    *,
    group: str,
    target_class: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[float],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not scene_keys:
        raise ValueError(f"No scenes contain any observations for subset {subset_name} group {group}")

    coalition_values: list[float] = []
    max_views_per_scene = 0
    complete_subset_scene_count = 0
    for scene_key in scene_keys:
        observed = sum(1 for viewpoint in players if viewpoint in scene_lookup[scene_key])
        max_views_per_scene = max(max_views_per_scene, observed)
        if observed == len(players):
            complete_subset_scene_count += 1

    full_mask = (1 << len(players)) - 1
    for coalition_mask in range(full_mask + 1):
        coalition_scene_values = [
            coalition_value_for_scene(scene_lookup[scene_key], players, coalition_mask) for scene_key in scene_keys
        ]
        coalition_values.append(float(np.mean(coalition_scene_values)))

    shapley = exact_shapley_from_values(coalition_values, n_players=len(players))
    conditional_rows = build_conditional_gain_rows(
        subset_name,
        players,
        coalition_values,
        group=group,
        target_class=target_class,
    )
    conditional_summary_rows = summarize_best_extensions_by_size(
        subset_name,
        players,
        coalition_values,
        group=group,
        target_class=target_class,
    )

    grand_value = float(coalition_values[full_mask])
    empty_value = float(coalition_values[0])
    top_singleton_mask = best_mask_for_size(coalition_values, 1)
    top_singleton_index = next(index for index in range(len(players)) if top_singleton_mask & (1 << index))
    top_singleton_value = float(coalition_values[top_singleton_mask])
    top_singleton_extension = next(row for row in conditional_summary_rows if int(row["base_coalition_size"]) == 1)
    best_pair_extension = next(row for row in conditional_summary_rows if int(row["base_coalition_size"]) == 2)

    detail_rows: list[dict[str, object]] = []
    for player_index, viewpoint in enumerate(players):
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        observed_scene_count = 0
        matched_scene_count = 0
        observed_strict_values: list[float] = []
        coalition_mask = 1 << player_index
        singleton_value = float(coalition_values[coalition_mask])
        for scene_key in scene_keys:
            record = scene_lookup[scene_key].get(viewpoint)
            if record is None:
                continue
            observed_scene_count += 1
            observed_strict_values.append(float(record["strict_quality"]))
            if int(record["target_detected"]) > 0 and float(record["iou"]) > 0.0:
                matched_scene_count += 1

        detail_rows.append(
            {
                "group": group,
                "target_class": target_class,
                "subset_name": subset_name,
                "player_count": len(players),
                "player_index": player_index + 1,
                "viewpoint": viewpoint,
                "ring_id": ring_id_for_viewpoint(viewpoint),
                "elevation": elevation,
                "radius": radius,
                "azimuth": azimuth,
                "rank": 0,
                "shapley_value": float(shapley[player_index]),
                "share_of_subset_value": float(shapley[player_index] / grand_value) if grand_value else 0.0,
                "singleton_subset_scene_value": singleton_value,
                "mean_strict_quality_on_observed_rows": float(np.mean(observed_strict_values))
                if observed_strict_values
                else float("nan"),
                "observed_scene_count": observed_scene_count,
                "observed_scene_rate_within_subset": observed_scene_count / len(scene_keys),
                "matched_scene_count": matched_scene_count,
                "matched_scene_rate_within_subset": matched_scene_count / len(scene_keys),
                "subset_scene_count": len(scene_keys),
                "complete_subset_scene_count": complete_subset_scene_count,
                "max_views_seen_in_any_subset_scene": max_views_per_scene,
                "grand_value_all_players": grand_value,
                "empty_coalition_value": empty_value,
            }
        )

    detail_rows.sort(
        key=lambda row: (
            float(row["shapley_value"]),
            float(row["singleton_subset_scene_value"]),
            -int(row["player_index"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(detail_rows, start=1):
        row["rank"] = rank

    summary_row = {
        "group": group,
        "target_class": target_class,
        "subset_name": subset_name,
        "player_count": len(players),
        "subset_scene_count": len(scene_keys),
        "complete_subset_scene_count": complete_subset_scene_count,
        "max_views_seen_in_any_subset_scene": max_views_per_scene,
        "grand_value_all_players": grand_value,
        "empty_coalition_value": empty_value,
        "selected_viewpoints": " | ".join(players),
        "selected_ring_ids": " | ".join(ring_id_for_viewpoint(viewpoint) for viewpoint in players),
        "selected_azimuths": " | ".join(f"az{viewpoint_parts(viewpoint)[2]:03d}" for viewpoint in players),
        "top_viewpoint": detail_rows[0]["viewpoint"],
        "top_player_index": detail_rows[0]["player_index"],
        "top_shapley_value": detail_rows[0]["shapley_value"],
        "second_viewpoint": detail_rows[1]["viewpoint"],
        "second_player_index": detail_rows[1]["player_index"],
        "second_shapley_value": detail_rows[1]["shapley_value"],
        "top_minus_second_shapley_gap": float(detail_rows[0]["shapley_value"]) - float(detail_rows[1]["shapley_value"]),
        "top_singleton_viewpoint": players[top_singleton_index],
        "top_singleton_player_index": top_singleton_index + 1,
        "top_singleton_value": top_singleton_value,
        "top_singleton_best_addition_player_index": top_singleton_extension["best_addition_player_index"],
        "top_singleton_best_addition_viewpoint": top_singleton_extension["best_addition_viewpoint"],
        "top_singleton_best_addition_delta": top_singleton_extension["delta_value"],
        "top_singleton_expanded_members": top_singleton_extension["expanded_coalition_members"],
        "top_singleton_expanded_value": top_singleton_extension["expanded_value"],
        "best_pair_members": best_pair_extension["base_coalition_members"],
        "best_pair_value": best_pair_extension["base_value"],
        "best_pair_best_addition_player_index": best_pair_extension["best_addition_player_index"],
        "best_pair_best_addition_viewpoint": best_pair_extension["best_addition_viewpoint"],
        "best_pair_best_addition_delta": best_pair_extension["delta_value"],
        "best_pair_expanded_members": best_pair_extension["expanded_coalition_members"],
        "best_pair_expanded_value": best_pair_extension["expanded_value"],
    }
    return detail_rows, summary_row, coalition_values, conditional_rows, conditional_summary_rows


def write_selected_viewpoints(
    output_path: Path,
    subset_name: str,
    selection_mode: str,
    selection_label: str,
    players: list[str],
) -> None:
    rows: list[dict[str, object]] = []
    for player_index, viewpoint in enumerate(players, start=1):
        elevation, radius, azimuth = viewpoint_parts(viewpoint)
        rows.append(
            {
                "subset_name": subset_name,
                "selection_mode": selection_mode,
                "selection_label": selection_label,
                "player_index": player_index,
                "viewpoint": viewpoint,
                "ring_id": ring_id_for_viewpoint(viewpoint),
                "elevation": elevation,
                "radius": radius,
                "azimuth": azimuth,
            }
        )
    write_csv(output_path, rows)


def write_markdown_report(
    output_path: Path,
    scene_records_path: Path,
    selection_mode: str,
    selection_label: str,
    players: list[str],
    detail_rows: list[dict[str, object]],
    summary_row: dict[str, object],
    min_scenes: int,
) -> None:
    scene_count = int(summary_row["subset_scene_count"])
    warning = " low support" if scene_count < min_scenes else ""
    lines: list[str] = [
        "# Exact Viewpoint-Subset Shapley Analysis",
        "",
        "## Setup",
        "",
        f"- Input scene records: `{scene_records_path}`",
        f"- Subset name: `{summary_row['subset_name']}`",
        f"- Selection mode: `{selection_mode}`",
        f"- Selection rule: {selection_label}",
        f"- Player count: `{summary_row['player_count']}`",
        "- Coalition value uses `Noisy-OR + best IoU` exactly, matching the ring Shapley analysis.",
        f"- Shapley is computed exactly over all `2^{len(players)} = {1 << len(players)}` coalitions.",
        "- Missing selected viewpoints inside a scene are treated as unavailable rather than imputed.",
        "- The subset is evaluated on scenes that contain at least one of the selected viewpoints.",
        "",
        "## Selected Players",
        "",
        "| Player | Viewpoint | Ring | Azimuth |",
        "| ---: | --- | --- | ---: |",
    ]
    for player_index, viewpoint in enumerate(players, start=1):
        lines.append(
            f"| {player_index} | `{viewpoint}` | `{ring_id_for_viewpoint(viewpoint)}` | `az{viewpoint_parts(viewpoint)[2]:03d}` |"
        )

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Scenes with any selected viewpoint: `{scene_count}`{warning}",
            f"- Complete scenes with all selected viewpoints observed: `{summary_row['complete_subset_scene_count']}`",
            f"- Maximum selected views observed in one scene: `{summary_row['max_views_seen_in_any_subset_scene']}`",
            f"- Grand coalition value: `{format_float(float(summary_row['grand_value_all_players']))}`",
            "",
            "## Top Players",
            "",
            f"- Best singleton: `{summary_row['top_singleton_viewpoint']}` with `v({{u}}) = {format_float(float(summary_row['top_singleton_value']))}`.",
            f"- Top Shapley player: `{summary_row['top_viewpoint']}` with `phi = {format_float(float(summary_row['top_shapley_value']))}`.",
            f"- Runner-up: `{summary_row['second_viewpoint']}` with gap `{format_float(float(summary_row['top_minus_second_shapley_gap']))}`.",
            "",
            "## Conditional Next-View Example Inside The Same Subset Game",
            "",
            "| Current coalition C | Best addition u | Delta(u | C) | New value |",
            "| --- | --- | ---: | ---: |",
            f"| `{summary_row['top_singleton_viewpoint']}` | `{summary_row['top_singleton_best_addition_viewpoint']}` | "
            f"{format_float(float(summary_row['top_singleton_best_addition_delta']))} | "
            f"{format_float(float(summary_row['top_singleton_expanded_value']))} |",
            f"| `{summary_row['best_pair_members']}` | `{summary_row['best_pair_best_addition_viewpoint']}` | "
            f"{format_float(float(summary_row['best_pair_best_addition_delta']))} | "
            f"{format_float(float(summary_row['best_pair_expanded_value']))} |",
            "",
            "## Player Ranking",
            "",
            "| Rank | Player | Viewpoint | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in detail_rows:
        lines.append(
            f"| {row['rank']} | {row['player_index']} | `{row['viewpoint']}` | "
            f"{format_float(float(row['shapley_value']))} | "
            f"{format_float(float(row['singleton_subset_scene_value']))} | "
            f"{row['observed_scene_count']} | {row['matched_scene_count']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_class_report(
    output_path: Path,
    per_class_summary_rows: list[dict[str, object]],
    min_scenes: int,
) -> None:
    lines: list[str] = [
        "# Per-Class Exact Viewpoint-Subset Shapley Analysis",
        "",
        "This report repeats the exact subset Shapley analysis per target class.",
        "",
        "| Class | Scenes | Top viewpoint | Top Shapley | Runner-up | Gap |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for row in sorted(per_class_summary_rows, key=lambda item: str(item["target_class"])):
        scene_label = f"{row['subset_scene_count']}"
        if int(row["subset_scene_count"]) < min_scenes:
            scene_label += " low"
        lines.append(
            f"| `{row['target_class']}` | {scene_label} | `{row['top_viewpoint']}` | "
            f"{format_float(float(row['top_shapley_value']))} | `{row['second_viewpoint']}` | "
            f"{format_float(float(row['top_minus_second_shapley_gap']))} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_conditional_gain_report(
    output_path: Path,
    conditional_summary_rows: list[dict[str, object]],
) -> None:
    grouped: dict[int, dict[str, object]] = {
        int(row["base_coalition_size"]): row for row in conditional_summary_rows
    }
    lines: list[str] = [
        "# Exact Conditional Next-View Gains On The Subset Fusion Game",
        "",
        "- This report uses the same coalition value as subset Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.",
        "- Shapley answers the average teammate question.",
        "- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.",
        "",
        "| Coalition size | Current coalition C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |",
        "| ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for coalition_size in sorted(grouped):
        row = grouped[coalition_size]
        lines.append(
            f"| {coalition_size} | `{row['base_coalition_members']}` | "
            f"{format_float(float(row['base_value']))} | "
            f"`{row['best_addition_viewpoint']}` | "
            f"{format_float(float(row['delta_value']))} | "
            f"{format_float(float(row['expanded_value']))} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_shapley_bars(
    detail_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    sorted_rows = sorted(detail_rows, key=lambda row: int(row["rank"]))
    labels = [f"P{int(row['player_index'])}\n{row['viewpoint']}" for row in sorted_rows]
    values = [float(row["shapley_value"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(13.5, 6.6), constrained_layout=True)
    bars = ax.bar(range(len(sorted_rows)), values, color="#2b6cb0")
    ax.set_xticks(range(len(sorted_rows)), labels, rotation=25, ha="right")
    ax.set_ylabel("Shapley value")
    ax.set_title("Exact Shapley values for the selected viewpoint subset")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{value:.3f}", ha="center", va="bottom")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scene_records_path = Path(args.scene_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(scene_records_path)
    available_viewpoints = sorted({str(record["viewpoint"]) for record in records}, key=viewpoint_sort_key)
    players, selection_label = resolve_players(args, available_viewpoints)
    scene_lookup = build_scene_lookup(records)
    scene_class_lookup = build_scene_class_lookup(scene_lookup)
    target_classes = sorted({str(record["target_class"]) for record in records})

    detail_rows, summary_row, _coalition_values, conditional_detail_rows, conditional_summary_rows = (
        analyze_subset_for_scene_keys(
            args.subset_name,
            players,
            scene_lookup,
            subset_scene_keys(players, scene_lookup),
            group="overall",
            target_class="all",
        )
    )

    per_class_detail_rows: list[dict[str, object]] = []
    per_class_summary_rows: list[dict[str, object]] = []
    for target_class in target_classes:
        scene_keys = subset_scene_keys(
            players,
            scene_lookup,
            scene_class_lookup=scene_class_lookup,
            target_class=target_class,
        )
        if not scene_keys:
            continue
        class_detail_rows, class_summary_row, _class_values, _class_conditional_rows, _class_conditional_summary = (
            analyze_subset_for_scene_keys(
                args.subset_name,
                players,
                scene_lookup,
                scene_keys,
                group="per_class",
                target_class=target_class,
            )
        )
        per_class_detail_rows.extend(class_detail_rows)
        per_class_summary_rows.append(class_summary_row)

    per_class_summary_rows.sort(key=lambda row: str(row["target_class"]))
    conditional_summary_rows.sort(key=lambda row: int(row["base_coalition_size"]))

    selected_viewpoints_path = output_dir / "selected_viewpoints.csv"
    detail_path = output_dir / "subset_shapley_noisy_or_best_iou.csv"
    summary_path = output_dir / "subset_shapley_noisy_or_best_iou_summary.csv"
    report_path = output_dir / "subset_shapley_noisy_or_best_iou_report.md"
    bar_path = output_dir / "subset_shapley_noisy_or_best_iou_bars.png"
    conditional_detail_path = output_dir / "subset_shapley_noisy_or_best_iou_conditional_gain.csv"
    conditional_summary_path = output_dir / "subset_shapley_noisy_or_best_iou_conditional_gain_summary.csv"
    conditional_report_path = output_dir / "subset_shapley_noisy_or_best_iou_conditional_gain_report.md"
    per_class_detail_path = output_dir / "subset_shapley_noisy_or_best_iou_by_class.csv"
    per_class_summary_path = output_dir / "subset_shapley_noisy_or_best_iou_by_class_summary.csv"
    per_class_report_path = output_dir / "subset_shapley_noisy_or_best_iou_by_class_report.md"

    write_selected_viewpoints(selected_viewpoints_path, args.subset_name, args.selection_mode, selection_label, players)
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, [summary_row])
    write_markdown_report(
        report_path,
        scene_records_path,
        args.selection_mode,
        selection_label,
        players,
        detail_rows,
        summary_row,
        args.min_scenes,
    )
    plot_shapley_bars(detail_rows, bar_path)
    write_csv(conditional_detail_path, conditional_detail_rows)
    write_csv(conditional_summary_path, conditional_summary_rows)
    write_conditional_gain_report(conditional_report_path, conditional_summary_rows)
    write_csv(per_class_detail_path, per_class_detail_rows)
    write_csv(per_class_summary_path, per_class_summary_rows)
    write_per_class_report(per_class_report_path, per_class_summary_rows, args.min_scenes)

    print(selected_viewpoints_path)
    print(detail_path)
    print(summary_path)
    print(report_path)
    print(bar_path)
    print(conditional_detail_path)
    print(conditional_summary_path)
    print(conditional_report_path)
    print(per_class_detail_path)
    print(per_class_summary_path)
    print(per_class_report_path)


if __name__ == "__main__":
    main()
