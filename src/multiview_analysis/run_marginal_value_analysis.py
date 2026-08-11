from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_RECORDS = WORKSPACE / "results" / "intermediate" / "scene_view_records.csv"
DEFAULT_PROTOCOL_OVERALL = WORKSPACE / "results" / "tables" / "one_vs_two_summary.csv"
DEFAULT_PROTOCOL_CLASS = WORKSPACE / "results" / "tables" / "one_vs_two_class_summary.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "results" / "recomputed" / "marginal_viewpoint_value"

PRIMARY_LABEL = "Target Strict Quality"


@dataclass(frozen=True)
class SceneViewRecord:
    scene_key: str
    target_class: str
    viewpoint: str
    elevation: str
    radius: str
    azimuth: int
    target_visible: int
    target_detected: int
    target_ap50_95: float
    target_strict_quality_iou50: float
    target_match_confidence_iou50: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operationalize marginal viewpoint value, diminishing returns, and "
            "viewpoint complementarity using the current M4 fixed-detector outputs."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--protocol-overall", default=str(DEFAULT_PROTOCOL_OVERALL))
    parser.add_argument("--protocol-class", default=str(DEFAULT_PROTOCOL_CLASS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--min-pair-support", type=int, default=8)
    parser.add_argument("--min-triple-support", type=int, default=3)
    return parser.parse_args()


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def parse_float(raw: str) -> float:
    text = str(raw).strip()
    if text == "":
        return 0.0
    lowered = text.lower()
    if lowered in {"nan", "none"}:
        return 0.0
    return float(text)


def parse_int(raw: str) -> int:
    text = str(raw).strip()
    if text == "":
        return 0
    return int(float(text))


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_scene_records(path: Path) -> list[SceneViewRecord]:
    rows: list[SceneViewRecord] = []
    with ensure_file(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                SceneViewRecord(
                    scene_key=row["scene_key"],
                    target_class=row["target_class"],
                    viewpoint=row["viewpoint"],
                    elevation=row["elevation"],
                    radius=row["radius"],
                    azimuth=parse_int(row["azimuth"]),
                    target_visible=parse_int(row["target_visible"]),
                    target_detected=parse_int(row["target_detected"]),
                    target_ap50_95=parse_float(row["target_ap50_95"]),
                    target_strict_quality_iou50=parse_float(row["target_strict_quality_iou50"]),
                    target_match_confidence_iou50=parse_float(row["target_match_confidence_iou50"]),
                )
            )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with ensure_file(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_scene_groups(records: list[SceneViewRecord]) -> dict[str, list[SceneViewRecord]]:
    grouped: dict[str, list[SceneViewRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_key].append(record)
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda item: item.viewpoint)
    return dict(grouped)


def summarize_single_view_landscape(records: list[SceneViewRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[SceneViewRecord]] = defaultdict(list)
    for record in records:
        grouped[record.viewpoint].append(record)

    rows: list[dict[str, object]] = []
    for viewpoint, members in grouped.items():
        exemplar = members[0]
        rows.append(
            {
                "viewpoint": viewpoint,
                "elevation": exemplar.elevation,
                "radius": exemplar.radius,
                "azimuth": exemplar.azimuth,
                "scene_count": len({member.scene_key for member in members}),
                "sample_count": len(members),
                "mean_target_ap50_95": mean([member.target_ap50_95 for member in members]),
                "mean_target_strict_quality_iou50": mean(
                    [member.target_strict_quality_iou50 for member in members]
                ),
                "mean_target_match_confidence_iou50": mean(
                    [member.target_match_confidence_iou50 for member in members]
                ),
                "target_visible_rate": mean([float(member.target_visible) for member in members]),
                "target_detected_rate": mean([float(member.target_detected) for member in members]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["mean_target_strict_quality_iou50"]),
            float(row["mean_target_ap50_95"]),
            str(row["viewpoint"]),
        ),
        reverse=True,
    )


def summarize_pair_complementarity(scene_groups: dict[str, list[SceneViewRecord]]) -> list[dict[str, object]]:
    observations: dict[tuple[str, str], list[tuple[SceneViewRecord, SceneViewRecord]]] = defaultdict(list)
    for members in scene_groups.values():
        for first, second in combinations(members, 2):
            observations[(first.viewpoint, second.viewpoint)].append((first, second))

    rows: list[dict[str, object]] = []
    for (viewpoint_1, viewpoint_2), members in observations.items():
        first_ap = [pair[0].target_ap50_95 for pair in members]
        second_ap = [pair[1].target_ap50_95 for pair in members]
        first_strict = [pair[0].target_strict_quality_iou50 for pair in members]
        second_strict = [pair[1].target_strict_quality_iou50 for pair in members]
        first_conf = [pair[0].target_match_confidence_iou50 for pair in members]
        second_conf = [pair[1].target_match_confidence_iou50 for pair in members]

        pair_or_ap = mean([max(a, b) for a, b in zip(first_ap, second_ap)])
        pair_or_strict = mean([max(a, b) for a, b in zip(first_strict, second_strict)])
        pair_or_conf = mean([max(a, b) for a, b in zip(first_conf, second_conf)])
        first_mean_ap = mean(first_ap)
        second_mean_ap = mean(second_ap)
        first_mean_strict = mean(first_strict)
        second_mean_strict = mean(second_strict)
        first_mean_conf = mean(first_conf)
        second_mean_conf = mean(second_conf)
        best_single_ap = max(first_mean_ap, second_mean_ap)
        best_single_strict = max(first_mean_strict, second_mean_strict)
        best_single_conf = max(first_mean_conf, second_mean_conf)

        rows.append(
            {
                "target_class": members[0][0].target_class,
                "viewpoint_pair": f"{viewpoint_1} + {viewpoint_2}",
                "viewpoint_1": viewpoint_1,
                "viewpoint_2": viewpoint_2,
                "scene_count": len(members),
                "mean_viewpoint_1_target_ap50_95": first_mean_ap,
                "mean_viewpoint_2_target_ap50_95": second_mean_ap,
                "expected_pair_or_target_ap50_95": pair_or_ap,
                "best_single_target_ap50_95_on_matched_scenes": best_single_ap,
                "pair_complementarity_gain_target_ap50_95": pair_or_ap - best_single_ap,
                "mean_viewpoint_1_target_strict_quality_iou50": first_mean_strict,
                "mean_viewpoint_2_target_strict_quality_iou50": second_mean_strict,
                "expected_pair_or_target_strict_quality_iou50": pair_or_strict,
                "best_single_target_strict_quality_iou50_on_matched_scenes": best_single_strict,
                "pair_complementarity_gain_target_strict_quality_iou50": pair_or_strict - best_single_strict,
                "mean_viewpoint_1_target_match_confidence_iou50": first_mean_conf,
                "mean_viewpoint_2_target_match_confidence_iou50": second_mean_conf,
                "expected_pair_or_target_match_confidence_iou50": pair_or_conf,
                "best_single_target_match_confidence_iou50_on_matched_scenes": best_single_conf,
                "pair_complementarity_gain_target_match_confidence_iou50": pair_or_conf - best_single_conf,
                "viewpoint_1_scene_win_rate": mean(
                    [1.0 if a > b else 0.0 for a, b in zip(first_strict, second_strict)]
                ),
                "viewpoint_2_scene_win_rate": mean(
                    [1.0 if b > a else 0.0 for a, b in zip(first_strict, second_strict)]
                ),
                "tie_rate": mean(
                    [
                        1.0 if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) else 0.0
                        for a, b in zip(first_strict, second_strict)
                    ]
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["pair_complementarity_gain_target_strict_quality_iou50"]),
            float(row["expected_pair_or_target_strict_quality_iou50"]),
            int(row["scene_count"]),
        ),
        reverse=True,
    )


def summarize_triple_gain(scene_groups: dict[str, list[SceneViewRecord]]) -> list[dict[str, object]]:
    observations: dict[tuple[str, str, str], list[tuple[SceneViewRecord, SceneViewRecord, SceneViewRecord]]] = defaultdict(list)
    for members in scene_groups.values():
        for first, second, third in combinations(members, 3):
            observations[(first.viewpoint, second.viewpoint, third.viewpoint)].append((first, second, third))

    rows: list[dict[str, object]] = []
    for key, members in observations.items():
        values_ap = [(row[0].target_ap50_95, row[1].target_ap50_95, row[2].target_ap50_95) for row in members]
        values_strict = [
            (
                row[0].target_strict_quality_iou50,
                row[1].target_strict_quality_iou50,
                row[2].target_strict_quality_iou50,
            )
            for row in members
        ]

        mean12_ap = mean([max(a, b) for a, b, _ in values_ap])
        mean13_ap = mean([max(a, c) for a, _, c in values_ap])
        mean23_ap = mean([max(b, c) for _, b, c in values_ap])
        triple_ap = mean([max(a, b, c) for a, b, c in values_ap])

        mean12_strict = mean([max(a, b) for a, b, _ in values_strict])
        mean13_strict = mean([max(a, c) for a, _, c in values_strict])
        mean23_strict = mean([max(b, c) for _, b, c in values_strict])
        triple_strict = mean([max(a, b, c) for a, b, c in values_strict])

        ap_pair_rows = [
            (f"{key[0]} + {key[1]}", mean12_ap),
            (f"{key[0]} + {key[2]}", mean13_ap),
            (f"{key[1]} + {key[2]}", mean23_ap),
        ]
        strict_pair_rows = [
            (f"{key[0]} + {key[1]}", mean12_strict),
            (f"{key[0]} + {key[2]}", mean13_strict),
            (f"{key[1]} + {key[2]}", mean23_strict),
        ]
        best_pair_for_ap = max(ap_pair_rows, key=lambda row: row[1])
        best_pair_for_strict = max(strict_pair_rows, key=lambda row: row[1])

        rows.append(
            {
                "target_class": members[0][0].target_class,
                "viewpoint_triple": " + ".join(key),
                "viewpoint_1": key[0],
                "viewpoint_2": key[1],
                "viewpoint_3": key[2],
                "scene_count": len(members),
                "expected_pair_12_target_ap50_95": mean12_ap,
                "expected_pair_13_target_ap50_95": mean13_ap,
                "expected_pair_23_target_ap50_95": mean23_ap,
                "expected_best_pair_target_ap50_95_on_matched_scenes": best_pair_for_ap[1],
                "best_pair_for_target_ap50_95": best_pair_for_ap[0],
                "expected_triple_or_target_ap50_95": triple_ap,
                "third_view_gain_target_ap50_95": triple_ap - best_pair_for_ap[1],
                "expected_pair_12_target_strict_quality_iou50": mean12_strict,
                "expected_pair_13_target_strict_quality_iou50": mean13_strict,
                "expected_pair_23_target_strict_quality_iou50": mean23_strict,
                "expected_best_pair_target_strict_quality_iou50_on_matched_scenes": best_pair_for_strict[1],
                "best_pair_for_target_strict_quality_iou50": best_pair_for_strict[0],
                "expected_triple_or_target_strict_quality_iou50": triple_strict,
                "third_view_gain_target_strict_quality_iou50": triple_strict - best_pair_for_strict[1],
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["third_view_gain_target_strict_quality_iou50"]),
            float(row["expected_triple_or_target_strict_quality_iou50"]),
            int(row["scene_count"]),
        ),
        reverse=True,
    )


def summarize_viewpoint_marginal_contributions(
    pair_rows: list[dict[str, object]],
    triple_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pair_stage_ap: dict[str, list[float]] = defaultdict(list)
    pair_stage_strict: dict[str, list[float]] = defaultdict(list)
    third_stage_ap: dict[str, list[float]] = defaultdict(list)
    third_stage_strict: dict[str, list[float]] = defaultdict(list)
    pair_support: dict[str, int] = defaultdict(int)
    triple_support: dict[str, int] = defaultdict(int)

    for row in pair_rows:
        scene_count = int(row["scene_count"])
        v1 = str(row["viewpoint_1"])
        v2 = str(row["viewpoint_2"])
        pair_or_ap = float(row["expected_pair_or_target_ap50_95"])
        pair_or_strict = float(row["expected_pair_or_target_strict_quality_iou50"])
        v1_ap = float(row["mean_viewpoint_1_target_ap50_95"])
        v2_ap = float(row["mean_viewpoint_2_target_ap50_95"])
        v1_strict = float(row["mean_viewpoint_1_target_strict_quality_iou50"])
        v2_strict = float(row["mean_viewpoint_2_target_strict_quality_iou50"])

        pair_stage_ap[v1].append(pair_or_ap - v2_ap)
        pair_stage_ap[v2].append(pair_or_ap - v1_ap)
        pair_stage_strict[v1].append(pair_or_strict - v2_strict)
        pair_stage_strict[v2].append(pair_or_strict - v1_strict)
        pair_support[v1] += scene_count
        pair_support[v2] += scene_count

    for row in triple_rows:
        scene_count = int(row["scene_count"])
        triple_ap = float(row["expected_triple_or_target_ap50_95"])
        triple_strict = float(row["expected_triple_or_target_strict_quality_iou50"])
        v1 = str(row["viewpoint_1"])
        v2 = str(row["viewpoint_2"])
        v3 = str(row["viewpoint_3"])

        third_stage_ap[v1].append(triple_ap - float(row["expected_pair_23_target_ap50_95"]))
        third_stage_ap[v2].append(triple_ap - float(row["expected_pair_13_target_ap50_95"]))
        third_stage_ap[v3].append(triple_ap - float(row["expected_pair_12_target_ap50_95"]))
        third_stage_strict[v1].append(triple_strict - float(row["expected_pair_23_target_strict_quality_iou50"]))
        third_stage_strict[v2].append(triple_strict - float(row["expected_pair_13_target_strict_quality_iou50"]))
        third_stage_strict[v3].append(triple_strict - float(row["expected_pair_12_target_strict_quality_iou50"]))
        triple_support[v1] += scene_count
        triple_support[v2] += scene_count
        triple_support[v3] += scene_count

    viewpoints = sorted(set(pair_stage_strict) | set(third_stage_strict))
    rows: list[dict[str, object]] = []
    for viewpoint in viewpoints:
        pair_ap_mean = mean(pair_stage_ap.get(viewpoint, []))
        pair_strict_mean = mean(pair_stage_strict.get(viewpoint, []))
        third_ap_mean = mean(third_stage_ap.get(viewpoint, []))
        third_strict_mean = mean(third_stage_strict.get(viewpoint, []))
        combined_marginal_ap = mean([value for value in [pair_ap_mean, third_ap_mean] if not math.isnan(value)])
        combined_marginal_strict = mean([value for value in [pair_strict_mean, third_strict_mean] if not math.isnan(value)])
        rows.append(
            {
                "viewpoint": viewpoint,
                "pair_stage_observation_count": len(pair_stage_strict.get(viewpoint, [])),
                "third_stage_observation_count": len(third_stage_strict.get(viewpoint, [])),
                "pair_stage_scene_support": pair_support.get(viewpoint, 0),
                "third_stage_scene_support": triple_support.get(viewpoint, 0),
                "pair_stage_mean_marginal_target_ap50_95": pair_ap_mean,
                "third_stage_mean_marginal_target_ap50_95": third_ap_mean,
                "combined_marginal_target_ap50_95": combined_marginal_ap,
                "pair_stage_mean_marginal_target_strict_quality_iou50": pair_strict_mean,
                "third_stage_mean_marginal_target_strict_quality_iou50": third_strict_mean,
                "combined_marginal_target_strict_quality_iou50": combined_marginal_strict,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["combined_marginal_target_strict_quality_iou50"]),
            float(row["combined_marginal_target_ap50_95"]),
        ),
        reverse=True,
    )


def summarize_multi_view_gain(protocol_overall_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wanted = {"n1_any1": 1, "n2_any1": 2, "n3_any1": 3}
    filtered = [row for row in protocol_overall_rows if row["protocol_id"] in wanted]
    filtered.sort(key=lambda row: wanted[row["protocol_id"]])

    summary_rows: list[dict[str, object]] = []
    previous_ap = float("nan")
    previous_strict = float("nan")
    first_ap = float("nan")
    first_strict = float("nan")
    last_ap = float("nan")
    last_strict = float("nan")

    for index, row in enumerate(filtered):
        drone_count = int(row["drone_count"])
        ap = parse_float(row["expected_target_threshold_ap50_95"])
        strict = parse_float(row["expected_target_threshold_strict_quality_iou50"])
        if index == 0:
            first_ap = ap
            first_strict = strict
        last_ap = ap
        last_strict = strict
        summary_rows.append(
            {
                "drone_count": drone_count,
                "protocol_id": row["protocol_id"],
                "protocol_label": row["protocol_label"],
                "expected_target_threshold_ap50_95": ap,
                "expected_target_threshold_strict_quality_iou50": strict,
                "delta_target_threshold_ap50_95_vs_previous_k": 0.0 if math.isnan(previous_ap) else ap - previous_ap,
                "delta_target_threshold_strict_quality_iou50_vs_previous_k": 0.0
                if math.isnan(previous_strict)
                else strict - previous_strict,
                "delta_target_threshold_ap50_95_vs_k1": 0.0 if math.isnan(first_ap) else ap - first_ap,
                "delta_target_threshold_strict_quality_iou50_vs_k1": 0.0
                if math.isnan(first_strict)
                else strict - first_strict,
            }
        )
        previous_ap = ap
        previous_strict = strict

    total_ap_gain = last_ap - first_ap if len(summary_rows) >= 2 else 0.0
    total_strict_gain = last_strict - first_strict if len(summary_rows) >= 2 else 0.0

    for row in summary_rows:
        ap_gain = float(row["delta_target_threshold_ap50_95_vs_k1"])
        strict_gain = float(row["delta_target_threshold_strict_quality_iou50_vs_k1"])
        row["fraction_of_total_ap50_95_gain_captured"] = 0.0 if math.isclose(total_ap_gain, 0.0) else ap_gain / total_ap_gain
        row["fraction_of_total_strict_quality_gain_captured"] = 0.0 if math.isclose(total_strict_gain, 0.0) else strict_gain / total_strict_gain

    targets: list[dict[str, object]] = []
    for threshold in [0.90, 0.95]:
        for metric_key, label in [
            ("fraction_of_total_ap50_95_gain_captured", "AP50-95"),
            ("fraction_of_total_strict_quality_gain_captured", PRIMARY_LABEL),
        ]:
            chosen = next((row for row in summary_rows if float(row[metric_key]) >= threshold), summary_rows[-1])
            targets.append(
                {
                    "gain_fraction_target": threshold,
                    "metric": label,
                    "minimum_drone_count": int(chosen["drone_count"]),
                    "protocol_label": chosen["protocol_label"],
                    "achieved_fraction": float(chosen[metric_key]),
                }
            )

    return summary_rows, targets


def summarize_class_gains(protocol_class_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_class: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in protocol_class_rows:
        if row["protocol_id"] not in {"n1_any1", "n2_any1", "n3_any1"}:
            continue
        drone_count = parse_int(row["drone_count"])
        by_class[row["target_class"]][drone_count] = {
            "ap": parse_float(row["expected_target_threshold_ap50_95"]),
            "strict": parse_float(row["expected_target_threshold_strict_quality_iou50"]),
            "found_rate": parse_float(row["expected_target_found_rate"]),
            "scene_count": parse_float(row["scene_count"]),
        }

    rows: list[dict[str, object]] = []
    for target_class, per_k in by_class.items():
        if not all(k in per_k for k in [1, 2, 3]):
            continue
        rows.append(
            {
                "target_class": target_class,
                "scene_count": int(per_k[1]["scene_count"]),
                "k1_target_ap50_95": per_k[1]["ap"],
                "k2_target_ap50_95": per_k[2]["ap"],
                "k3_target_ap50_95": per_k[3]["ap"],
                "delta_target_ap50_95_1_to_2": per_k[2]["ap"] - per_k[1]["ap"],
                "delta_target_ap50_95_2_to_3": per_k[3]["ap"] - per_k[2]["ap"],
                "delta_target_ap50_95_1_to_3": per_k[3]["ap"] - per_k[1]["ap"],
                "k1_target_strict_quality_iou50": per_k[1]["strict"],
                "k2_target_strict_quality_iou50": per_k[2]["strict"],
                "k3_target_strict_quality_iou50": per_k[3]["strict"],
                "delta_target_strict_quality_iou50_1_to_2": per_k[2]["strict"] - per_k[1]["strict"],
                "delta_target_strict_quality_iou50_2_to_3": per_k[3]["strict"] - per_k[2]["strict"],
                "delta_target_strict_quality_iou50_1_to_3": per_k[3]["strict"] - per_k[1]["strict"],
                "k1_target_found_rate": per_k[1]["found_rate"],
                "k2_target_found_rate": per_k[2]["found_rate"],
                "k3_target_found_rate": per_k[3]["found_rate"],
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["delta_target_strict_quality_iou50_1_to_2"]),
            float(row["delta_target_strict_quality_iou50_1_to_3"]),
        ),
        reverse=True,
    )


def summarize_best_combinations(
    single_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    triple_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    best_single = single_rows[0]
    best_pair = max(pair_rows, key=lambda row: float(row["expected_pair_or_target_strict_quality_iou50"]))
    best_triple = max(triple_rows, key=lambda row: float(row["expected_triple_or_target_strict_quality_iou50"]))

    return [
        {
            "drone_count": 1,
            "combination_label": best_single["viewpoint"],
            "scene_count": int(best_single["scene_count"]),
            "expected_target_ap50_95": float(best_single["mean_target_ap50_95"]),
            "expected_target_strict_quality_iou50": float(best_single["mean_target_strict_quality_iou50"]),
            "complementarity_or_gain_target_strict_quality_iou50": 0.0,
        },
        {
            "drone_count": 2,
            "combination_label": best_pair["viewpoint_pair"],
            "scene_count": int(best_pair["scene_count"]),
            "expected_target_ap50_95": float(best_pair["expected_pair_or_target_ap50_95"]),
            "expected_target_strict_quality_iou50": float(best_pair["expected_pair_or_target_strict_quality_iou50"]),
            "complementarity_or_gain_target_strict_quality_iou50": float(
                best_pair["pair_complementarity_gain_target_strict_quality_iou50"]
            ),
        },
        {
            "drone_count": 3,
            "combination_label": best_triple["viewpoint_triple"],
            "scene_count": int(best_triple["scene_count"]),
            "expected_target_ap50_95": float(best_triple["expected_triple_or_target_ap50_95"]),
            "expected_target_strict_quality_iou50": float(best_triple["expected_triple_or_target_strict_quality_iou50"]),
            "complementarity_or_gain_target_strict_quality_iou50": float(
                best_triple["third_view_gain_target_strict_quality_iou50"]
            ),
        },
    ]


def summarize_single_view_landscape_by_class(records: list[SceneViewRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[SceneViewRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.target_class, record.viewpoint)].append(record)

    rows: list[dict[str, object]] = []
    for (target_class, viewpoint), members in grouped.items():
        exemplar = members[0]
        rows.append(
            {
                "target_class": target_class,
                "viewpoint": viewpoint,
                "elevation": exemplar.elevation,
                "radius": exemplar.radius,
                "azimuth": exemplar.azimuth,
                "scene_count": len({member.scene_key for member in members}),
                "sample_count": len(members),
                "mean_target_ap50_95": mean([member.target_ap50_95 for member in members]),
                "mean_target_strict_quality_iou50": mean(
                    [member.target_strict_quality_iou50 for member in members]
                ),
                "mean_target_match_confidence_iou50": mean(
                    [member.target_match_confidence_iou50 for member in members]
                ),
                "target_visible_rate": mean([float(member.target_visible) for member in members]),
                "target_detected_rate": mean([float(member.target_detected) for member in members]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            str(row["target_class"]),
            -float(row["mean_target_strict_quality_iou50"]),
            -float(row["mean_target_ap50_95"]),
            str(row["viewpoint"]),
        ),
    )


def slugify(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in str(text)]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "class"


def plot_barh(
    rows: list[dict[str, object]],
    label_key: str,
    value_key: str,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    selected = list(reversed(rows[:top_n]))
    labels = [str(row[label_key]) for row in selected]
    values = [float(row[value_key]) for row in selected]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(selected) + 1.5)))
    ax.barh(range(len(selected)), values, color="#2E6F95")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(PRIMARY_LABEL)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_multi_view_gain(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    x = [int(row["drone_count"]) for row in summary_rows]
    ap = [float(row["expected_target_threshold_ap50_95"]) for row in summary_rows]
    strict = [float(row["expected_target_threshold_strict_quality_iou50"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, ap, marker="o", linewidth=2, color="#8C5A2B", label="Target AP50-95")
    ax.plot(x, strict, marker="o", linewidth=2, color="#1B7F5A", label=PRIMARY_LABEL)
    ax.set_xticks(x)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Expected performance")
    ax.set_title("Diminishing Returns From Additional Views")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_class_gain_comparison(class_rows: list[dict[str, object]], output_path: Path) -> None:
    selected = sorted(
        class_rows,
        key=lambda row: float(row["delta_target_strict_quality_iou50_1_to_2"]),
        reverse=True,
    )
    labels = [str(row["target_class"]) for row in selected]
    delta12 = [float(row["delta_target_strict_quality_iou50_1_to_2"]) for row in selected]
    delta23 = [float(row["delta_target_strict_quality_iou50_2_to_3"]) for row in selected]
    y = list(range(len(selected)))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.45 * len(selected) + 1.5)), sharey=True)
    axes[0].barh(y, delta12, color="#2E6F95")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Strict Quality Gain: 1 -> 2 Views")
    axes[0].set_xlabel("Delta strict quality")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, delta23, color="#8C5A2B")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Strict Quality Gain: 2 -> 3 Views")
    axes[1].set_xlabel("Delta strict quality")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Per-Class Diminishing Returns")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_class_gain_curve(class_row: dict[str, object], output_path: Path) -> None:
    x = [1, 2, 3]
    ap = [
        float(class_row["k1_target_ap50_95"]),
        float(class_row["k2_target_ap50_95"]),
        float(class_row["k3_target_ap50_95"]),
    ]
    strict = [
        float(class_row["k1_target_strict_quality_iou50"]),
        float(class_row["k2_target_strict_quality_iou50"]),
        float(class_row["k3_target_strict_quality_iou50"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, ap, marker="o", linewidth=2, color="#8C5A2B", label="Target AP50-95")
    ax.plot(x, strict, marker="o", linewidth=2, color="#1B7F5A", label=PRIMARY_LABEL)
    ax.set_xticks(x)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Expected performance")
    ax.set_title(f"{class_row['target_class']}: 1 vs 2 vs 3 Views")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_per_class_outputs(
    records: list[SceneViewRecord],
    pair_rows: list[dict[str, object]],
    triple_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    plots_dir: Path,
    output_dir: Path,
    top_n: int,
    min_pair_support: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_class_plot_dir = plots_dir / "per_class"
    per_class_plot_dir.mkdir(parents=True, exist_ok=True)

    records_by_class: dict[str, list[SceneViewRecord]] = defaultdict(list)
    for record in records:
        records_by_class[record.target_class].append(record)

    pair_by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        pair_by_class[str(row["target_class"])].append(row)

    triple_by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in triple_rows:
        triple_by_class[str(row["target_class"])].append(row)

    class_lookup = {str(row["target_class"]): row for row in class_rows}
    single_by_class_rows: list[dict[str, object]] = []
    marginal_by_class_rows: list[dict[str, object]] = []
    plot_index_rows: list[dict[str, object]] = []

    for target_class in sorted(records_by_class.keys()):
        class_slug = slugify(target_class)
        class_single_rows = summarize_single_view_landscape(records_by_class[target_class])
        class_pair_rows = pair_by_class.get(target_class, [])
        class_supported_pair_rows = [
            row for row in class_pair_rows if int(row["scene_count"]) >= min_pair_support
        ]
        if not class_supported_pair_rows:
            class_supported_pair_rows = class_pair_rows
        class_triple_rows = triple_by_class.get(target_class, [])
        class_marginal_rows = summarize_viewpoint_marginal_contributions(
            class_pair_rows,
            class_triple_rows,
        )
        class_summary_row = class_lookup.get(target_class)

        for row in class_single_rows:
            single_by_class_rows.append({"target_class": target_class, **row})
        for row in class_marginal_rows:
            marginal_by_class_rows.append({"target_class": target_class, **row})

        if class_single_rows:
            single_plot = per_class_plot_dir / f"{class_slug}_single_view_landscape_top.png"
            plot_barh(
                class_single_rows,
                label_key="viewpoint",
                value_key="mean_target_strict_quality_iou50",
                title=f"{target_class}: Strongest Single Views",
                output_path=single_plot,
                top_n=top_n,
            )
            plot_index_rows.append(
                {
                    "target_class": target_class,
                    "plot_type": "single_view_landscape",
                    "path": str(single_plot),
                }
            )

        if class_supported_pair_rows:
            pair_plot = per_class_plot_dir / f"{class_slug}_pair_complementarity_top.png"
            plot_barh(
                class_supported_pair_rows,
                label_key="viewpoint_pair",
                value_key="pair_complementarity_gain_target_strict_quality_iou50",
                title=f"{target_class}: Most Complementary Pairs",
                output_path=pair_plot,
                top_n=top_n,
            )
            plot_index_rows.append(
                {
                    "target_class": target_class,
                    "plot_type": "pair_complementarity",
                    "path": str(pair_plot),
                }
            )

        if class_marginal_rows:
            marginal_plot = per_class_plot_dir / f"{class_slug}_viewpoint_marginal_scores_top.png"
            plot_barh(
                class_marginal_rows,
                label_key="viewpoint",
                value_key="combined_marginal_target_strict_quality_iou50",
                title=f"{target_class}: Highest Marginal View Scores",
                output_path=marginal_plot,
                top_n=top_n,
            )
            plot_index_rows.append(
                {
                    "target_class": target_class,
                    "plot_type": "viewpoint_marginal_scores",
                    "path": str(marginal_plot),
                }
            )

        if class_summary_row is not None:
            gain_plot = per_class_plot_dir / f"{class_slug}_multi_view_gain.png"
            plot_single_class_gain_curve(class_summary_row, gain_plot)
            plot_index_rows.append(
                {
                    "target_class": target_class,
                    "plot_type": "multi_view_gain",
                    "path": str(gain_plot),
                }
            )

    write_csv(output_dir / "single_view_landscape_by_class.csv", single_by_class_rows)
    write_csv(output_dir / "viewpoint_marginal_contribution_by_class.csv", marginal_by_class_rows)
    write_csv(output_dir / "per_class_plot_index.csv", plot_index_rows)
    return single_by_class_rows, marginal_by_class_rows, plot_index_rows


def build_report(
    single_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    triple_rows: list[dict[str, object]],
    marginal_rows: list[dict[str, object]],
    gain_rows: list[dict[str, object]],
    gain_targets: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    output_path: Path,
    top_n: int,
    min_pair_support: int,
    min_triple_support: int,
    max_triple_support: int,
) -> None:
    gain_by_k = {int(row["drone_count"]): row for row in gain_rows}
    k2 = gain_by_k[2]
    k3 = gain_by_k[3]
    lines = [
        "# Marginal Viewpoint Value Report",
        "",
        "This report reframes the current M4 swarm outputs around a thesis question:",
        "",
        "> how much new information do extra UAV views still add, and when do they mostly become redundant?",
        "",
        "## 1. Diminishing Returns",
        "",
        f"- `1 -> 2` views: target AP50-95 changes by `{fmt(float(k2['delta_target_threshold_ap50_95_vs_previous_k']))}` and {PRIMARY_LABEL.lower()} changes by `{fmt(float(k2['delta_target_threshold_strict_quality_iou50_vs_previous_k']))}`.",
        f"- `2 -> 3` views: target AP50-95 changes by `{fmt(float(k3['delta_target_threshold_ap50_95_vs_previous_k']))}` and {PRIMARY_LABEL.lower()} changes by `{fmt(float(k3['delta_target_threshold_strict_quality_iou50_vs_previous_k']))}`.",
        f"- `2` views already capture `{fmt(float(k2['fraction_of_total_ap50_95_gain_captured']) * 100.0, 1)}%` of the total `1 -> 3` AP50-95 gain.",
        f"- `2` views already capture `{fmt(float(k2['fraction_of_total_strict_quality_gain_captured']) * 100.0, 1)}%` of the total `1 -> 3` {PRIMARY_LABEL.lower()} gain.",
        "",
        "Gain targets:",
    ]
    for row in gain_targets:
        lines.append(
            f"- `{int(float(row['minimum_drone_count']))}` view(s) are enough to capture `{fmt(float(row['gain_fraction_target']) * 100.0, 0)}%` of the available `{row['metric']}` gain (achieved `{fmt(float(row['achieved_fraction']) * 100.0, 1)}%`)."
        )

    lines.extend(["", "## 2. Strongest Single Views", ""])
    for row in single_rows[:top_n]:
        lines.append(
            f"- `{row['viewpoint']}`: strict quality `{fmt(float(row['mean_target_strict_quality_iou50']))}`, AP50-95 `{fmt(float(row['mean_target_ap50_95']))}`, scenes `{int(row['scene_count'])}`."
        )

    lines.extend(
        [
            "",
            "## 3. Most Complementary Pairs",
            "",
            f"Only pairs with at least `{min_pair_support}` matched scenes are used in this headline section.",
            "",
            "Pair complementarity is defined here as:",
            "",
            "`E[max(view_i, view_j)] - max(E[view_i], E[view_j])`",
            "",
            "evaluated on the matched scene subset where both viewpoints are available.",
            "",
        ]
    )
    for row in pair_rows[:top_n]:
        lines.append(
            f"- `{row['viewpoint_pair']}`: complementarity gain `{fmt(float(row['pair_complementarity_gain_target_strict_quality_iou50']))}`, pair strict quality `{fmt(float(row['expected_pair_or_target_strict_quality_iou50']))}`, matched scenes `{int(row['scene_count'])}`."
        )

    lines.extend(
        [
            "",
            "## 4. Triples With Useful Third Views",
            "",
            f"Only triples with at least `{min_triple_support}` matched scenes are used in this headline section.",
            f"The maximum exact-triple overlap in the current dataset is `{max_triple_support}` scenes.",
            "",
            "Third-view gain is defined here as:",
            "",
            "`E[max(view_i, view_j, view_k)] - max(E[max(view_i, view_j)], E[max(view_i, view_k)], E[max(view_j, view_k)])`",
            "",
            "again on the matched scene subset where all three viewpoints are available.",
            "",
        ]
    )
    for row in triple_rows[:top_n]:
        lines.append(
            f"- `{row['viewpoint_triple']}`: third-view strict-quality gain `{fmt(float(row['third_view_gain_target_strict_quality_iou50']))}`, best pair `{row['best_pair_for_target_strict_quality_iou50']}`, matched scenes `{int(row['scene_count'])}`."
        )

    lines.extend(
        [
            "",
            "## 5. Observed Marginal View Scores",
            "",
            "The score below averages a viewpoint's marginal contribution when added to:",
            "",
            "- one existing view",
            "- an existing pair",
            "",
            "This is an observed-coalition marginal summary, not a Shapley analysis.",
            "",
        ]
    )
    for row in marginal_rows[:top_n]:
        lines.append(
            f"- `{row['viewpoint']}`: combined marginal strict quality `{fmt(float(row['combined_marginal_target_strict_quality_iou50']))}`, pair-stage marginal `{fmt(float(row['pair_stage_mean_marginal_target_strict_quality_iou50']))}`, third-stage marginal `{fmt(float(row['third_stage_mean_marginal_target_strict_quality_iou50']))}`."
        )

    lines.extend(["", "## 6. Object Classes With The Largest Multi-View Gain", ""])
    for row in class_rows[: min(top_n, len(class_rows))]:
        lines.append(
            f"- `{row['target_class']}`: strict-quality gain `1 -> 2` = `{fmt(float(row['delta_target_strict_quality_iou50_1_to_2']))}`, `1 -> 3` = `{fmt(float(row['delta_target_strict_quality_iou50_1_to_3']))}`."
        )

    lines.extend(
        [
            "",
            "## 7. Thesis Interpretation",
            "",
            "These outputs support a thesis narrative in which:",
            "",
            "- single-view performance maps the baseline information landscape;",
            "- `k-view` curves quantify diminishing returns instead of only asking whether multiview beats single-view;",
            "- complementarity is about marginal added coverage, not just individually strong viewpoints;",
            "- the best swarm size is therefore the smallest `k` that captures most of the available gain.",
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

    scene_records = read_scene_records(Path(args.scene_records))
    scene_groups = build_scene_groups(scene_records)
    protocol_overall_rows = read_csv_rows(Path(args.protocol_overall))
    protocol_class_rows = read_csv_rows(Path(args.protocol_class))

    single_rows = summarize_single_view_landscape(scene_records)
    pair_rows = summarize_pair_complementarity(scene_groups)
    triple_rows = summarize_triple_gain(scene_groups)
    marginal_rows = summarize_viewpoint_marginal_contributions(pair_rows, triple_rows)
    gain_rows, gain_targets = summarize_multi_view_gain(protocol_overall_rows)
    class_rows = summarize_class_gains(protocol_class_rows)
    supported_pair_rows = [row for row in pair_rows if int(row["scene_count"]) >= args.min_pair_support]
    supported_triple_rows = [row for row in triple_rows if int(row["scene_count"]) >= args.min_triple_support]
    max_triple_support = max(int(row["scene_count"]) for row in triple_rows) if triple_rows else 0

    if not supported_pair_rows:
        supported_pair_rows = pair_rows
    if not supported_triple_rows:
        supported_triple_rows = triple_rows
    best_combo_rows = summarize_best_combinations(single_rows, supported_pair_rows, supported_triple_rows)

    write_csv(output_dir / "single_view_landscape.csv", single_rows)
    write_csv(output_dir / "pair_complementarity_summary.csv", pair_rows)
    write_csv(output_dir / "triple_third_view_gain_summary.csv", triple_rows)
    write_csv(output_dir / "pair_complementarity_supported_summary.csv", supported_pair_rows)
    write_csv(output_dir / "triple_third_view_gain_supported_summary.csv", supported_triple_rows)
    write_csv(output_dir / "viewpoint_marginal_contribution_summary.csv", marginal_rows)
    write_csv(output_dir / "multi_view_gain_summary.csv", gain_rows)
    write_csv(output_dir / "gain_capture_targets.csv", gain_targets)
    write_csv(output_dir / "class_multi_view_gain_summary.csv", class_rows)
    write_csv(output_dir / "best_combinations_by_drone_count.csv", best_combo_rows)

    build_per_class_outputs(
        records=scene_records,
        pair_rows=pair_rows,
        triple_rows=triple_rows,
        class_rows=class_rows,
        plots_dir=plots_dir,
        output_dir=output_dir,
        top_n=args.top_n,
        min_pair_support=args.min_pair_support,
    )

    plot_barh(
        single_rows,
        label_key="viewpoint",
        value_key="mean_target_strict_quality_iou50",
        title="Strongest Single Views",
        output_path=plots_dir / "single_view_landscape_top.png",
        top_n=args.top_n,
    )
    plot_barh(
        supported_pair_rows,
        label_key="viewpoint_pair",
        value_key="pair_complementarity_gain_target_strict_quality_iou50",
        title="Most Complementary Pairs",
        output_path=plots_dir / "pair_complementarity_top.png",
        top_n=args.top_n,
    )
    plot_barh(
        supported_triple_rows,
        label_key="viewpoint_triple",
        value_key="third_view_gain_target_strict_quality_iou50",
        title="Triples With The Largest Third-View Gain",
        output_path=plots_dir / "third_view_gain_top.png",
        top_n=args.top_n,
    )
    plot_barh(
        marginal_rows,
        label_key="viewpoint",
        value_key="combined_marginal_target_strict_quality_iou50",
        title="Highest Observed Marginal View Scores",
        output_path=plots_dir / "viewpoint_marginal_scores_top.png",
        top_n=args.top_n,
    )
    plot_multi_view_gain(gain_rows, plots_dir / "multi_view_diminishing_returns.png")
    plot_class_gain_comparison(class_rows, plots_dir / "class_gain_comparison.png")

    build_report(
        single_rows=single_rows,
        pair_rows=supported_pair_rows,
        triple_rows=supported_triple_rows,
        marginal_rows=marginal_rows,
        gain_rows=gain_rows,
        gain_targets=gain_targets,
        class_rows=class_rows,
        output_path=output_dir / "marginal_value_report.md",
        top_n=args.top_n,
        min_pair_support=args.min_pair_support,
        min_triple_support=args.min_triple_support,
        max_triple_support=max_triple_support,
    )


if __name__ == "__main__":
    main()
