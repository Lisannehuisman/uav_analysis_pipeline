from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_RECORDS = WORKSPACE / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "m4_marginal_viewpoint_value_analysis" / "outputs" / "ring_shapley_noisy_or_best_iou"

ELEVATION_ORDER = {"low": 0, "mid": 1, "high": 2}
RADIUS_ORDER = {"near": 0, "mid": 1, "far": 2}
AZIMUTHS = list(range(0, 360, 45))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact 8-player Shapley values for each controlled M4 viewpoint ring "
            "using Noisy-OR + best IoU as the coalition value."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--min-scenes-per-ring",
        type=int,
        default=25,
        help="Warn when a ring has fewer than this many scenes with any observation in the ring.",
    )
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


def ring_sort_key(ring_id: str) -> tuple[int, int]:
    elevation_token, radius_token = ring_id.split("-")
    elevation = elevation_token.replace("el", "")
    radius = radius_token.replace("rad", "")
    return (ELEVATION_ORDER.get(elevation, 99), RADIUS_ORDER.get(radius, 99))


def viewpoint_for_ring(ring_id: str, azimuth: int) -> str:
    return f"{ring_id}-az{azimuth:03d}"


def azimuths_for_mask(coalition_mask: int) -> list[int]:
    return [AZIMUTHS[idx] for idx in range(len(AZIMUTHS)) if coalition_mask & (1 << idx)]


def coalition_members_text(coalition_mask: int) -> str:
    members = [f"az{azimuth:03d}" for azimuth in azimuths_for_mask(coalition_mask)]
    return " + ".join(members) if members else "(empty)"


def format_float(value: float, digits: int = 6) -> str:
    if math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scene_key",
            "target_class",
            "viewpoint",
            "elevation",
            "radius",
            "azimuth",
            "target_match_confidence_iou50",
            "target_match_iou_at_confidence_iou50",
            "target_strict_quality_iou50",
            "target_detected",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Scene records CSV is missing columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "scene_key": row["scene_key"],
                    "target_class": row["target_class"],
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
                    "target_detected": int(parse_float(row["target_detected"])),
                }
            )
    return rows


def build_scene_lookup(records: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    by_scene: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in records:
        by_scene[str(record["scene_key"])][str(record["viewpoint"])] = record
    return by_scene


def build_scene_class_lookup(scene_lookup: dict[str, dict[str, dict[str, object]]]) -> dict[str, str]:
    scene_class: dict[str, str] = {}
    for scene_key, scene_records in scene_lookup.items():
        first_record = next(iter(scene_records.values()))
        scene_class[scene_key] = str(first_record["target_class"])
    return scene_class


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
            marginal = coalition_values[mask | bit] - coalition_values[mask]
            shapley[player] += weight * marginal
    return shapley


def best_mask_for_size(coalition_values: list[float], subset_size: int) -> int:
    candidate_masks = [mask for mask in range(len(coalition_values)) if mask.bit_count() == subset_size]
    if not candidate_masks:
        raise ValueError(f"No coalition found for subset size {subset_size}")
    return max(candidate_masks, key=lambda mask: (float(coalition_values[mask]), -mask))


def conditional_gain_rows_for_base_mask(
    ring_id: str,
    coalition_values: list[float],
    base_mask: int,
    *,
    group: str,
    target_class: str,
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
                "group": group,
                "target_class": target_class,
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
                "rank_within_base_coalition": 0,
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
    for rank, row in enumerate(rows, start=1):
        row["rank_within_base_coalition"] = rank
    return rows


def build_conditional_gain_rows(
    ring_id: str,
    coalition_values: list[float],
    *,
    group: str,
    target_class: str,
) -> list[dict[str, object]]:
    full_mask = (1 << len(AZIMUTHS)) - 1
    rows: list[dict[str, object]] = []
    for base_mask in range(full_mask + 1):
        if base_mask == full_mask:
            continue
        rows.extend(
            conditional_gain_rows_for_base_mask(
                ring_id,
                coalition_values,
                base_mask,
                group=group,
                target_class=target_class,
            )
        )

    rows.sort(
        key=lambda row: (
            str(row["ring_id"]),
            int(row["base_coalition_size"]),
            int(row["base_coalition_mask"]),
            int(row["rank_within_base_coalition"]),
        )
    )
    return rows


def summarize_best_extensions_by_size(
    ring_id: str,
    coalition_values: list[float],
    *,
    group: str,
    target_class: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base_size in range(1, len(AZIMUTHS)):
        base_mask = best_mask_for_size(coalition_values, base_size)
        best_extension = conditional_gain_rows_for_base_mask(
            ring_id,
            coalition_values,
            base_mask,
            group=group,
            target_class=target_class,
        )[0]
        rows.append(
            {
                "group": group,
                "target_class": target_class,
                "ring_id": ring_id,
                "base_coalition_mask": base_mask,
                "base_coalition_size": base_size,
                "base_coalition_members": coalition_members_text(base_mask),
                "base_value": float(coalition_values[base_mask]),
                "best_addition_azimuth": best_extension["candidate_azimuth"],
                "best_addition_viewpoint": best_extension["candidate_viewpoint"],
                "expanded_coalition_mask": best_extension["expanded_coalition_mask"],
                "expanded_coalition_size": best_extension["expanded_coalition_size"],
                "expanded_coalition_members": best_extension["expanded_coalition_members"],
                "expanded_value": best_extension["expanded_value"],
                "delta_value": best_extension["delta_value"],
            }
        )
    return rows


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


def ring_scene_keys(
    ring_id: str,
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    scene_class_lookup: dict[str, str] | None = None,
    target_class: str | None = None,
) -> list[str]:
    ring_viewpoints = [viewpoint_for_ring(ring_id, azimuth) for azimuth in AZIMUTHS]
    selected: list[str] = []
    for scene_key, scene_records in scene_lookup.items():
        if target_class is not None and scene_class_lookup is not None and scene_class_lookup[scene_key] != target_class:
            continue
        if any(viewpoint in scene_records for viewpoint in ring_viewpoints):
            selected.append(scene_key)
    selected.sort()
    return selected


def analyze_ring_for_scene_keys(
    ring_id: str,
    scene_lookup: dict[str, dict[str, dict[str, object]]],
    ring_scenes: list[str],
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
    ring_viewpoints = [viewpoint_for_ring(ring_id, azimuth) for azimuth in AZIMUTHS]

    if not ring_scenes:
        raise ValueError(f"No scenes contain any observations for ring {ring_id} group {group}")

    coalition_values: list[float] = []
    max_views_per_scene = 0
    complete_ring_scene_count = 0
    for scene_key in ring_scenes:
        observed = sum(1 for viewpoint in ring_viewpoints if viewpoint in scene_lookup[scene_key])
        max_views_per_scene = max(max_views_per_scene, observed)
        if observed == len(ring_viewpoints):
            complete_ring_scene_count += 1

    full_mask = (1 << len(AZIMUTHS)) - 1
    for coalition_mask in range(full_mask + 1):
        coalition_scene_values = [
            coalition_value_for_scene(scene_lookup[scene_key], ring_id, coalition_mask)
            for scene_key in ring_scenes
        ]
        coalition_values.append(float(np.mean(coalition_scene_values)))

    shapley = exact_shapley_from_values(coalition_values, n_players=len(AZIMUTHS))
    conditional_rows = build_conditional_gain_rows(
        ring_id,
        coalition_values,
        group=group,
        target_class=target_class,
    )
    conditional_summary_rows = summarize_best_extensions_by_size(
        ring_id,
        coalition_values,
        group=group,
        target_class=target_class,
    )
    grand_value = float(coalition_values[full_mask])
    empty_value = float(coalition_values[0])
    top_singleton_mask = best_mask_for_size(coalition_values, 1)
    top_singleton_azimuth = azimuths_for_mask(top_singleton_mask)[0]
    top_singleton_value = float(coalition_values[top_singleton_mask])
    top_singleton_extension = next(row for row in conditional_summary_rows if int(row["base_coalition_size"]) == 1)
    best_pair_extension = next(row for row in conditional_summary_rows if int(row["base_coalition_size"]) == 2)

    detail_rows: list[dict[str, object]] = []
    for player_index, azimuth in enumerate(AZIMUTHS):
        viewpoint = viewpoint_for_ring(ring_id, azimuth)
        observed_scene_count = 0
        matched_scene_count = 0
        observed_strict_values: list[float] = []
        coalition_mask = 1 << player_index
        singleton_value = float(coalition_values[coalition_mask])
        for scene_key in ring_scenes:
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
                "ring_id": ring_id,
                "viewpoint": viewpoint,
                "azimuth": azimuth,
                "rank": 0,
                "shapley_value": float(shapley[player_index]),
                "share_of_ring_value": float(shapley[player_index] / grand_value) if grand_value else 0.0,
                "singleton_ring_scene_value": singleton_value,
                "mean_strict_quality_on_observed_rows": float(np.mean(observed_strict_values))
                if observed_strict_values
                else float("nan"),
                "observed_scene_count": observed_scene_count,
                "observed_scene_rate_within_ring": observed_scene_count / len(ring_scenes),
                "matched_scene_count": matched_scene_count,
                "matched_scene_rate_within_ring": matched_scene_count / len(ring_scenes),
                "ring_scene_count": len(ring_scenes),
                "complete_ring_scene_count": complete_ring_scene_count,
                "max_views_seen_in_any_ring_scene": max_views_per_scene,
                "grand_value_all_8_players": grand_value,
                "empty_coalition_value": empty_value,
            }
        )

    detail_rows.sort(
        key=lambda row: (float(row["shapley_value"]), float(row["singleton_ring_scene_value"])),
        reverse=True,
    )
    for rank, row in enumerate(detail_rows, start=1):
        row["rank"] = rank

    summary_row = {
        "group": group,
        "target_class": target_class,
        "ring_id": ring_id,
        "ring_scene_count": len(ring_scenes),
        "complete_ring_scene_count": complete_ring_scene_count,
        "max_views_seen_in_any_ring_scene": max_views_per_scene,
        "grand_value_all_8_players": grand_value,
        "empty_coalition_value": empty_value,
        "top_viewpoint": detail_rows[0]["viewpoint"],
        "top_azimuth": detail_rows[0]["azimuth"],
        "top_shapley_value": detail_rows[0]["shapley_value"],
        "second_viewpoint": detail_rows[1]["viewpoint"],
        "second_azimuth": detail_rows[1]["azimuth"],
        "second_shapley_value": detail_rows[1]["shapley_value"],
        "top_minus_second_shapley_gap": float(detail_rows[0]["shapley_value"]) - float(detail_rows[1]["shapley_value"]),
        "top_singleton_viewpoint": viewpoint_for_ring(ring_id, top_singleton_azimuth),
        "top_singleton_azimuth": top_singleton_azimuth,
        "top_singleton_value": top_singleton_value,
        "top_singleton_best_addition_azimuth": top_singleton_extension["best_addition_azimuth"],
        "top_singleton_best_addition_viewpoint": top_singleton_extension["best_addition_viewpoint"],
        "top_singleton_best_addition_delta": top_singleton_extension["delta_value"],
        "top_singleton_expanded_members": top_singleton_extension["expanded_coalition_members"],
        "top_singleton_expanded_value": top_singleton_extension["expanded_value"],
        "best_pair_members": best_pair_extension["base_coalition_members"],
        "best_pair_value": best_pair_extension["base_value"],
        "best_pair_best_addition_azimuth": best_pair_extension["best_addition_azimuth"],
        "best_pair_best_addition_viewpoint": best_pair_extension["best_addition_viewpoint"],
        "best_pair_best_addition_delta": best_pair_extension["delta_value"],
        "best_pair_expanded_members": best_pair_extension["expanded_coalition_members"],
        "best_pair_expanded_value": best_pair_extension["expanded_value"],
    }
    return detail_rows, summary_row, coalition_values, conditional_rows, conditional_summary_rows


def analyze_ring(
    ring_id: str,
    scene_lookup: dict[str, dict[str, dict[str, object]]],
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[float],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    ring_scenes = ring_scene_keys(ring_id, scene_lookup)
    return analyze_ring_for_scene_keys(ring_id, scene_lookup, ring_scenes, group="overall", target_class="all")


def write_markdown_report(
    output_path: Path,
    scene_records_path: Path,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    min_scenes_per_ring: int,
) -> None:
    lines: list[str] = [
        "# Controlled 8-Azimuth Ring Shapley Analysis",
        "",
        "## Setup",
        "",
        f"- Input scene records: `{scene_records_path}`",
        "- Players are the 8 azimuth viewpoints within each fixed M4 ring (`elevation x radius`).",
        "- Coalition value uses `Noisy-OR + best IoU` exactly, not the older independent `max` utility.",
        "- Singleton coalition value means `v({u})`: the exact one-view coalition value of one azimuth inside one fixed ring.",
        "- Shapley is computed exactly over all `2^8 = 256` coalitions.",
        "- The `top Shapley azimuth` is therefore only the winner within that one fixed ring, not one global best angle across all rings.",
        "- For a concrete existing coalition `C`, the next-view question uses the same game directly: `Delta(u | C) = v(C union {u}) - v(C)`.",
        "- Because the current cache contains no scene with a full observed 8-view ring, missing ring viewpoints are treated as unavailable inside the coalition rather than pretending the ring is complete.",
        "- Each ring is evaluated on the scenes that contain at least one observation from that ring.",
        "",
        "## Ring Coverage",
        "",
        "| Ring | Ring scenes | Complete 8-view scenes | Max views seen in one scene | Grand value | Best singleton | Top Shapley |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        warning = " (low support)" if int(row["ring_scene_count"]) < min_scenes_per_ring else ""
        lines.append(
            f"| `{row['ring_id']}`{warning} | {row['ring_scene_count']} | {row['complete_ring_scene_count']} | "
            f"{row['max_views_seen_in_any_ring_scene']} | {format_float(float(row['grand_value_all_8_players']))} | "
            f"`az{int(row['top_singleton_azimuth']):03d}` ({format_float(float(row['top_singleton_value']))}) | "
            f"`az{int(row['top_azimuth']):03d}` ({format_float(float(row['top_shapley_value']))}) |"
        )

    lines.extend(["", "## Top Azimuth Per Ring", ""])
    for row in summary_rows:
        lines.append(
            f"- `{row['ring_id']}`: best singleton coalition is `az{int(row['top_singleton_azimuth']):03d}` "
            f"with `v({{u}}) = {format_float(float(row['top_singleton_value']))}`, while the top Shapley azimuth is "
            f"`az{int(row['top_azimuth']):03d}` (`{row['top_viewpoint']}`) with `phi = {format_float(float(row['top_shapley_value']))}`; "
            f"runner-up `az{int(row['second_azimuth']):03d}` with gap `{format_float(float(row['top_minus_second_shapley_gap']))}`."
        )

    lines.extend(["", "## Conditional Next-View Example Within The Same Ring Game", ""])
    lines.append("| Ring | Best singleton C | Best 2nd azimuth | Delta(u | C) | New value | Best pair C | Best 3rd azimuth | Delta(u | C) | New value |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| `{row['ring_id']}` | `{row['top_singleton_viewpoint']}` | "
            f"`az{int(row['top_singleton_best_addition_azimuth']):03d}` | "
            f"{format_float(float(row['top_singleton_best_addition_delta']))} | "
            f"{format_float(float(row['top_singleton_expanded_value']))} | "
            f"`{row['best_pair_members']}` | "
            f"`az{int(row['best_pair_best_addition_azimuth']):03d}` | "
            f"{format_float(float(row['best_pair_best_addition_delta']))} | "
            f"{format_float(float(row['best_pair_expanded_value']))} |"
        )

    lines.extend(["", "## Per-Ring Rankings", ""])
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["ring_id"])].append(row)

    for ring_id in sorted(grouped, key=ring_sort_key):
        lines.extend([f"### {ring_id}", ""])
        lines.append("| Rank | Azimuth | Shapley phi_u | Singleton coalition value v({u}) | Observed scenes | Matched scenes |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in grouped[ring_id]:
            lines.append(
                f"| {row['rank']} | `{int(row['azimuth']):03d}` | {format_float(float(row['shapley_value']))} | "
                f"{format_float(float(row['singleton_ring_scene_value']))} | {row['observed_scene_count']} | {row['matched_scene_count']} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_class_report(
    output_path: Path,
    per_class_summary_rows: list[dict[str, object]],
    min_scenes_per_ring: int,
) -> None:
    lines: list[str] = [
        "# Per-Class Controlled 8-Azimuth Ring Shapley Analysis",
        "",
        "This report repeats the exact 8-player ring Shapley analysis per target class.",
        "",
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_class_summary_rows:
        grouped[str(row["target_class"])].append(row)

    for target_class in sorted(grouped):
        lines.extend([f"## {target_class}", ""])
        lines.append("| Ring | Scenes | Top azimuth | Top Shapley | Runner-up | Gap |")
        lines.append("| --- | ---: | --- | ---: | --- | ---: |")
        for row in sorted(grouped[target_class], key=lambda item: ring_sort_key(str(item["ring_id"]))):
            scene_label = f"{row['ring_scene_count']}"
            if int(row["ring_scene_count"]) < min_scenes_per_ring:
                scene_label += " low"
            lines.append(
                f"| `{row['ring_id']}` | {scene_label} | `az{int(row['top_azimuth']):03d}` | "
                f"{format_float(float(row['top_shapley_value']))} | `az{int(row['second_azimuth']):03d}` | "
                f"{format_float(float(row['top_minus_second_shapley_gap']))} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_conditional_gain_report(
    output_path: Path,
    conditional_summary_rows: list[dict[str, object]],
) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in conditional_summary_rows:
        grouped[str(row["ring_id"])].append(row)

    lines: list[str] = [
        "# Exact Conditional Next-View Gains On The Ring Fusion Game",
        "",
        "- This report uses the same coalition value as ring Shapley: `v(S) = mean_scene[Noisy-OR(S) x best IoU(S)]`.",
        "- Shapley answers the average teammate question.",
        "- `Delta(u | C) = v(C union {u}) - v(C)` answers the concrete next-view question for one current coalition `C`.",
        "",
        "## Best next addition for the best singleton in each ring",
        "",
        "| Ring | Current singleton C | v(C) | Best addition u | Delta(u | C) | v(C union {u}) |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for ring_id in sorted(grouped, key=ring_sort_key):
        singleton_row = next(row for row in grouped[ring_id] if int(row["base_coalition_size"]) == 1)
        lines.append(
            f"| `{ring_id}` | `{singleton_row['base_coalition_members']}` | "
            f"{format_float(float(singleton_row['base_value']))} | "
            f"`az{int(singleton_row['best_addition_azimuth']):03d}` | "
            f"{format_float(float(singleton_row['delta_value']))} | "
            f"{format_float(float(singleton_row['expanded_value']))} |"
        )

    lines.extend(
        [
            "",
            "## Best next addition for the best pair in each ring",
            "",
            "| Ring | Current pair C | v(C) | Best third azimuth u | Delta(u | C) | v(C union {u}) |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for ring_id in sorted(grouped, key=ring_sort_key):
        pair_row = next(row for row in grouped[ring_id] if int(row["base_coalition_size"]) == 2)
        lines.append(
            f"| `{ring_id}` | `{pair_row['base_coalition_members']}` | "
            f"{format_float(float(pair_row['base_value']))} | "
            f"`az{int(pair_row['best_addition_azimuth']):03d}` | "
            f"{format_float(float(pair_row['delta_value']))} | "
            f"{format_float(float(pair_row['expanded_value']))} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_shapley_heatmap(detail_rows: list[dict[str, object]], output_path: Path) -> None:
    ring_ids = sorted({str(row["ring_id"]) for row in detail_rows}, key=ring_sort_key)
    matrix = np.full((len(ring_ids), len(AZIMUTHS)), np.nan, dtype=float)
    ring_lookup = {ring_id: index for index, ring_id in enumerate(ring_ids)}
    az_lookup = {azimuth: index for index, azimuth in enumerate(AZIMUTHS)}
    for row in detail_rows:
        matrix[ring_lookup[str(row["ring_id"])], az_lookup[int(row["azimuth"])]] = float(row["shapley_value"])

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(AZIMUTHS)), [f"az{az:03d}" for az in AZIMUTHS], rotation=0)
    ax.set_yticks(range(len(ring_ids)), ring_ids)
    ax.set_xlabel("Azimuth player")
    ax.set_ylabel("Controlled viewpoint ring")
    ax.set_title("Exact 8-player Shapley by M4 ring using Noisy-OR + best IoU")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Shapley value")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            label = "n/a" if math.isnan(value) else f"{value:.3f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color="#102a43")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_azimuths(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["ring_id"]) for row in summary_rows]
    values = [float(row["top_shapley_value"]) for row in summary_rows]
    azimuth_labels = [f"az{int(row['top_azimuth']):03d}" for row in summary_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    bars = ax.barh(labels, values, color="#2b6cb0")
    ax.invert_yaxis()
    ax.set_xlabel("Top azimuth Shapley value")
    ax.set_title("Best azimuth within each controlled M4 ring")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for bar, azimuth_label in zip(bars, azimuth_labels, strict=False):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, azimuth_label, va="center", ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scene_records_path = Path(args.scene_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(scene_records_path)
    scene_lookup = build_scene_lookup(records)
    scene_class_lookup = build_scene_class_lookup(scene_lookup)
    ring_ids = sorted({str(record["ring_id"]) for record in records}, key=ring_sort_key)
    target_classes = sorted({str(record["target_class"]) for record in records})

    all_detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    conditional_detail_rows: list[dict[str, object]] = []
    conditional_summary_rows: list[dict[str, object]] = []
    for ring_id in ring_ids:
        detail_rows, summary_row, _coalition_values, ring_conditional_rows, ring_conditional_summary_rows = analyze_ring(
            ring_id,
            scene_lookup,
        )
        all_detail_rows.extend(detail_rows)
        summary_rows.append(summary_row)
        conditional_detail_rows.extend(ring_conditional_rows)
        conditional_summary_rows.extend(ring_conditional_summary_rows)

    summary_rows.sort(key=lambda row: ring_sort_key(str(row["ring_id"])))
    conditional_summary_rows.sort(
        key=lambda row: (ring_sort_key(str(row["ring_id"])), int(row["base_coalition_size"]))
    )
    per_class_detail_rows: list[dict[str, object]] = []
    per_class_summary_rows: list[dict[str, object]] = []
    for target_class in target_classes:
        for ring_id in ring_ids:
            scene_keys = ring_scene_keys(
                ring_id,
                scene_lookup,
                scene_class_lookup=scene_class_lookup,
                target_class=target_class,
            )
            if not scene_keys:
                continue
            detail_rows, summary_row, _coalition_values, _conditional_rows, _conditional_summary_rows = analyze_ring_for_scene_keys(
                ring_id,
                scene_lookup,
                scene_keys,
                group="per_class",
                target_class=target_class,
            )
            per_class_detail_rows.extend(detail_rows)
            per_class_summary_rows.append(summary_row)

    per_class_summary_rows.sort(key=lambda row: (str(row["target_class"]), *ring_sort_key(str(row["ring_id"])) ))
    detail_path = output_dir / "ring_shapley_noisy_or_best_iou.csv"
    summary_path = output_dir / "ring_shapley_noisy_or_best_iou_summary.csv"
    report_path = output_dir / "ring_shapley_noisy_or_best_iou_report.md"
    heatmap_path = output_dir / "ring_shapley_noisy_or_best_iou_heatmap.png"
    top_path = output_dir / "ring_shapley_noisy_or_best_iou_top_azimuths.png"
    per_class_detail_path = output_dir / "ring_shapley_noisy_or_best_iou_by_class.csv"
    per_class_summary_path = output_dir / "ring_shapley_noisy_or_best_iou_by_class_summary.csv"
    per_class_report_path = output_dir / "ring_shapley_noisy_or_best_iou_by_class_report.md"
    conditional_detail_path = output_dir / "ring_shapley_noisy_or_best_iou_conditional_gain.csv"
    conditional_summary_path = output_dir / "ring_shapley_noisy_or_best_iou_conditional_gain_summary.csv"
    conditional_report_path = output_dir / "ring_shapley_noisy_or_best_iou_conditional_gain_report.md"

    write_csv(detail_path, all_detail_rows)
    write_csv(summary_path, summary_rows)
    write_markdown_report(report_path, scene_records_path, all_detail_rows, summary_rows, args.min_scenes_per_ring)
    plot_shapley_heatmap(all_detail_rows, heatmap_path)
    plot_top_azimuths(summary_rows, top_path)
    write_csv(conditional_detail_path, conditional_detail_rows)
    write_csv(conditional_summary_path, conditional_summary_rows)
    write_conditional_gain_report(conditional_report_path, conditional_summary_rows)
    write_csv(per_class_detail_path, per_class_detail_rows)
    write_csv(per_class_summary_path, per_class_summary_rows)
    write_per_class_report(per_class_report_path, per_class_summary_rows, args.min_scenes_per_ring)

    print(detail_path)
    print(summary_path)
    print(report_path)
    print(heatmap_path)
    print(top_path)
    print(conditional_detail_path)
    print(conditional_summary_path)
    print(conditional_report_path)
    print(per_class_detail_path)
    print(per_class_summary_path)
    print(per_class_report_path)


if __name__ == "__main__":
    main()
