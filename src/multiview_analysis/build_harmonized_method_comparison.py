from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_VIEW_CSV = ROOT / "results" / "intermediate" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "recomputed" / "multiview_method_comparison"


@dataclass(frozen=True)
class MethodMeta:
    method_id: str
    method_label: str
    method_family: str
    question_type: str
    strict_quality_alias: str
    supports_ap50_95: bool
    supports_found_rate: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a harmonized multiview method comparison so all currently used "
            "coalition methods can be compared on the same strict-quality metric."
        )
    )
    parser.add_argument(
        "--scene-view-csv",
        default=str(DEFAULT_SCENE_VIEW_CSV),
        help="CSV with per-scene, per-view target metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV summaries and the markdown report.",
    )
    return parser.parse_args()


def parse_float(raw: object) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    return float(text)


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if not math.isnan(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def odds_fusion(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    odds_product = 1.0
    for probability in probabilities:
        p = clamp_probability(float(probability))
        odds_product *= p / (1.0 - p)
    return odds_product / (1.0 + odds_product)


def kth_largest(values: list[float], rank: int) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    if rank <= 0 or rank > len(ordered):
        return 0.0
    return float(ordered[rank - 1])


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


def load_scene_records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "scene_key": row["scene_key"],
                    "target_class": row["target_class"],
                    "viewpoint": row["viewpoint"],
                    "target_ap50_95": 0.0 if math.isnan(parse_float(row["target_ap50_95"])) else parse_float(row["target_ap50_95"]),
                    "target_detected": int(parse_float(row["target_detected"])),
                    "target_match_confidence_iou50": 0.0
                    if math.isnan(parse_float(row["target_match_confidence_iou50"]))
                    else parse_float(row["target_match_confidence_iou50"]),
                    "target_match_iou_at_confidence_iou50": 0.0
                    if math.isnan(parse_float(row["target_match_iou_at_confidence_iou50"]))
                    else parse_float(row["target_match_iou_at_confidence_iou50"]),
                    "target_strict_quality_iou50": 0.0
                    if math.isnan(parse_float(row["target_strict_quality_iou50"]))
                    else parse_float(row["target_strict_quality_iou50"]),
                }
            )
    return rows


def build_method_catalog() -> dict[str, MethodMeta]:
    rows = [
        MethodMeta(
            "single_view_reference",
            "Single-view reference",
            "Reference",
            "Reference",
            "",
            True,
            True,
            "Scene-balanced baseline over available single viewpoints.",
        ),
        MethodMeta(
            "threshold_1_of_2",
            "1-of-2 OR",
            "Operational threshold",
            "Availability / rescue",
            "Same strict-quality score as best_box for 2 views.",
            True,
            True,
            "Uses the best strict-quality observation and best target AP50-95 among two views.",
        ),
        MethodMeta(
            "threshold_2_of_2",
            "2-of-2 confirmation",
            "Operational threshold",
            "Confirmation / consensus",
            "Not the same as unanimous_best_box.",
            True,
            True,
            "Uses the second-best strict-quality observation, so both views must support the target.",
        ),
        MethodMeta(
            "threshold_1_of_3",
            "1-of-3 OR",
            "Operational threshold",
            "Availability / rescue",
            "Same strict-quality score as best_box for 3 views.",
            True,
            True,
            "Uses the best strict-quality observation among three views.",
        ),
        MethodMeta(
            "threshold_2_of_3",
            "2-of-3 confirmation",
            "Operational threshold",
            "Confirmation / consensus",
            "",
            True,
            True,
            "Uses the second-best strict-quality observation among three views.",
        ),
        MethodMeta(
            "threshold_3_of_3",
            "3-of-3 unanimous",
            "Operational threshold",
            "Confirmation / consensus",
            "Not the same as unanimous_best_box for 3 views.",
            True,
            True,
            "Uses the third-best strict-quality observation, so all three views must support the target.",
        ),
        MethodMeta(
            "best_box",
            "Best box (max)",
            "Selection / rescue",
            "Availability / rescue",
            "Equals threshold_1_of_n on strict quality.",
            False,
            False,
            "Keeps only the strongest strict-quality target observation in the coalition.",
        ),
        MethodMeta(
            "mean_quality",
            "Mean quality",
            "Naive aggregation",
            "Naive pooling",
            "",
            False,
            False,
            "Averages strict-quality values across selected views.",
        ),
        MethodMeta(
            "unanimous_best_box",
            "Unanimous best box",
            "Strict confirmation",
            "Confirmation / consensus",
            "Requires full support, but keeps the best strict-quality observation once that support exists.",
            False,
            False,
            "Returns the best strict-quality observation only if every selected view supports the target.",
        ),
        MethodMeta(
            "noisy_or_best_iou",
            "Noisy-OR + best IoU",
            "Probabilistic late fusion",
            "Probabilistic accumulation",
            "",
            False,
            False,
            "Combines confidence with noisy-OR and couples it to the best matched IoU.",
        ),
        MethodMeta(
            "noisy_or_mean_iou",
            "Noisy-OR + mean IoU",
            "Probabilistic late fusion",
            "Probabilistic accumulation",
            "",
            False,
            False,
            "Combines confidence with noisy-OR and couples it to the mean matched IoU.",
        ),
        MethodMeta(
            "support_weighted_or",
            "Support-weighted OR",
            "Probabilistic late fusion",
            "Conservative corroboration",
            "",
            False,
            False,
            "Adds a support-ratio penalty so corroboration matters, not only confidence accumulation.",
        ),
        MethodMeta(
            "roy_odds_best_iou",
            "Odds-product + best IoU",
            "Odds-product fusion",
            "Probabilistic corroboration",
            "",
            False,
            False,
            "Applies odds-product fusion to confidence proxies, then couples the fused probability to the best matched IoU.",
        ),
        MethodMeta(
            "roy_odds_mean_iou",
            "Odds-product + mean IoU",
            "Odds-product fusion",
            "Probabilistic corroboration",
            "",
            False,
            False,
            "Applies odds-product fusion to confidence proxies, then couples the fused probability to the mean matched IoU.",
        ),
    ]
    return {row.method_id: row for row in rows}


def combo_metrics(records: list[dict[str, object]]) -> dict[str, float]:
    drone_count = len(records)
    qualities = [float(record["target_strict_quality_iou50"]) for record in records]
    aps = [float(record["target_ap50_95"]) for record in records]
    detected_count = sum(int(record["target_detected"]) for record in records)
    confidences = [
        float(record["target_match_confidence_iou50"])
        for record in records
        if float(record["target_match_confidence_iou50"]) > 0.0
    ]
    ious = [
        float(record["target_match_iou_at_confidence_iou50"])
        for record in records
        if float(record["target_match_iou_at_confidence_iou50"]) > 0.0
    ]
    support_ratio = len(ious) / float(drone_count) if drone_count else 0.0
    noisy = noisy_or(confidences)
    roy_odds = odds_fusion(confidences)
    best_quality = max(qualities) if qualities else 0.0
    average_quality = mean(qualities) if qualities else 0.0
    unanimous_best = best_quality if math.isclose(support_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12) else 0.0
    best_iou = max(ious) if ious else 0.0
    average_iou = mean(ious) if ious else 0.0

    row: dict[str, float] = {
        "best_box": best_quality,
        "mean_quality": average_quality,
        "unanimous_best_box": unanimous_best,
        "noisy_or_best_iou": noisy * best_iou,
        "noisy_or_mean_iou": noisy * average_iou,
        "support_weighted_or": noisy * average_iou * support_ratio,
        "roy_odds_best_iou": roy_odds * best_iou,
        "roy_odds_mean_iou": roy_odds * average_iou,
    }
    for rank in range(1, drone_count + 1):
        row[f"threshold_{rank}_of_{drone_count}_strict_quality"] = kth_largest(qualities, rank)
        row[f"threshold_{rank}_of_{drone_count}_ap50_95"] = kth_largest(aps, rank)
        row[f"threshold_{rank}_of_{drone_count}_found_rate"] = float(detected_count >= rank)
    return row


def build_combo_metric_rows(scene_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in scene_records:
        grouped[str(record["scene_key"])].append(record)

    rows: list[dict[str, object]] = []
    for scene_key, records in sorted(grouped.items()):
        target_class = str(records[0]["target_class"])

        single_values = [float(record["target_strict_quality_iou50"]) for record in records]
        single_aps = [float(record["target_ap50_95"]) for record in records]
        single_found = [1.0 if int(record["target_detected"]) > 0 else 0.0 for record in records]
        rows.append(
            {
                "scene_key": scene_key,
                "target_class": target_class,
                "drone_count": 1,
                "method_id": "single_view_reference",
                "strict_quality": mean(single_values),
                "target_ap50_95": mean(single_aps),
                "target_found_rate": mean(single_found),
            }
        )

        for drone_count in (2, 3):
            if len(records) < drone_count:
                continue
            for combo in combinations(records, drone_count):
                metrics = combo_metrics(list(combo))
                for rank in range(1, drone_count + 1):
                    method_id = f"threshold_{rank}_of_{drone_count}"
                    rows.append(
                        {
                            "scene_key": scene_key,
                            "target_class": target_class,
                            "drone_count": drone_count,
                            "method_id": method_id,
                            "strict_quality": float(metrics[f"{method_id}_strict_quality"]),
                            "target_ap50_95": float(metrics[f"{method_id}_ap50_95"]),
                            "target_found_rate": float(metrics[f"{method_id}_found_rate"]),
                        }
                    )
                for method_id in (
                    "best_box",
                    "mean_quality",
                    "unanimous_best_box",
                    "noisy_or_best_iou",
                    "noisy_or_mean_iou",
                    "support_weighted_or",
                    "roy_odds_best_iou",
                    "roy_odds_mean_iou",
                ):
                    rows.append(
                        {
                            "scene_key": scene_key,
                            "target_class": target_class,
                            "drone_count": drone_count,
                            "method_id": method_id,
                            "strict_quality": float(metrics[method_id]),
                            "target_ap50_95": float("nan"),
                            "target_found_rate": float("nan"),
                        }
                    )
    return rows


def summarize_scene_balanced(
    combo_metric_rows: list[dict[str, object]], catalog: dict[str, MethodMeta]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in combo_metric_rows:
        grouped[(int(row["drone_count"]), str(row["method_id"]), str(row["scene_key"]))].append(row)

    per_scene_rows: list[dict[str, object]] = []
    for (drone_count, method_id, scene_key), members in sorted(grouped.items()):
        per_scene_rows.append(
            {
                "drone_count": drone_count,
                "method_id": method_id,
                "scene_key": scene_key,
                "target_class": str(members[0]["target_class"]),
                "scene_expected_strict_quality": mean([float(row["strict_quality"]) for row in members]),
                "scene_expected_target_ap50_95": mean([float(row["target_ap50_95"]) for row in members]),
                "scene_expected_target_found_rate": mean([float(row["target_found_rate"]) for row in members]),
            }
        )

    overall_grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_scene_rows:
        overall_grouped[(int(row["drone_count"]), str(row["method_id"]))].append(row)

    baseline_members = overall_grouped[(1, "single_view_reference")]
    baseline_strict = mean([float(row["scene_expected_strict_quality"]) for row in baseline_members])
    baseline_ap = mean([float(row["scene_expected_target_ap50_95"]) for row in baseline_members])

    rows: list[dict[str, object]] = []
    for (drone_count, method_id), members in sorted(overall_grouped.items()):
        meta = catalog[method_id]
        strict_quality = mean([float(row["scene_expected_strict_quality"]) for row in members])
        ap50_95 = mean([float(row["scene_expected_target_ap50_95"]) for row in members])
        found_rate = mean([float(row["scene_expected_target_found_rate"]) for row in members])
        rows.append(
            {
                "drone_count": drone_count,
                "method_id": method_id,
                "method_label": meta.method_label,
                "method_family": meta.method_family,
                "question_type": meta.question_type,
                "scene_count": len(members),
                "mean_scene_expected_strict_quality": strict_quality,
                "delta_vs_single_reference_strict_quality": strict_quality - baseline_strict,
                "mean_scene_expected_target_ap50_95": ap50_95 if meta.supports_ap50_95 else float("nan"),
                "delta_vs_single_reference_target_ap50_95": (
                    ap50_95 - baseline_ap if meta.supports_ap50_95 else float("nan")
                ),
                "mean_scene_expected_target_found_rate": found_rate if meta.supports_found_rate else float("nan"),
                "strict_quality_alias": meta.strict_quality_alias,
                "supports_ap50_95": int(meta.supports_ap50_95),
                "supports_found_rate": int(meta.supports_found_rate),
                "notes": meta.notes,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["drone_count"]),
            -float(row["mean_scene_expected_strict_quality"]),
            str(row["method_label"]),
        )
    )

    rows_by_count: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_count[int(row["drone_count"])].append(row)

    for drone_count, count_rows in rows_by_count.items():
        best_strict = max(float(row["mean_scene_expected_strict_quality"]) for row in count_rows)
        current_rank = 0
        previous_strict: float | None = None
        for row in count_rows:
            strict = float(row["mean_scene_expected_strict_quality"])
            if previous_strict is None or not math.isclose(strict, previous_strict, rel_tol=0.0, abs_tol=1e-12):
                current_rank += 1
                previous_strict = strict
            row["rank_within_drone_count"] = current_rank
            row["relative_gain_vs_single_reference_percent"] = (
                ((strict - baseline_strict) / baseline_strict) * 100.0 if baseline_strict else float("nan")
            )
            row["regret_vs_best_in_same_drone_count"] = best_strict - strict

    return rows


def fmt(value: float) -> str:
    if math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.4f}"


def write_markdown_report(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    lines: list[str] = [
        "# Harmonized Multiview Method Comparison",
        "",
        "This report compares the current multiview coalition methods on one shared headline metric:",
        "",
        "- `mean_scene_expected_strict_quality`",
        "",
        "This is the only metric that is principled across all currently used coalition methods.",
        "",
        "Secondary metrics are only shown where they remain methodologically meaningful:",
        "",
        "- `mean_scene_expected_target_ap50_95` is shown for threshold / protocol methods;",
        "- `mean_scene_expected_target_found_rate` is shown for threshold / protocol methods;",
        "- late-fusion methods are therefore compared directly on strict quality, not on AP50-95.",
        "",
        "Odds-product note:",
        "",
        "- The independent-detection idea `1 - product(1 - p_i)` is mathematically the same aggregation used by the existing `noisy-OR` fusion methods when detector confidence is treated as a probability proxy.",
        "- The additional corroboration variant added here is the odds-product fusion rule.",
        "",
        "## Key Equivalences",
        "",
        "- `1-of-2 OR` and `best_box` are identical on strict quality for 2-view coalitions.",
        "- `1-of-3 OR` and `best_box` are identical on strict quality for 3-view coalitions.",
        "- `2-of-2` is not the same as `unanimous_best_box`: `2-of-2` uses the weaker of the two matched qualities, while `unanimous_best_box` keeps the strongest quality once both views support the target.",
        "- `3-of-3` is not the same as a hypothetical 3-view unanimous best-box rule for the same reason.",
        "- `noisy-OR` methods operationalize independent-detection accumulation; `odds-product` methods provide an alternative corroboration rule.",
        "",
    ]

    available_counts = sorted({int(row["drone_count"]) for row in summary_rows})
    for drone_count in available_counts:
        lines.extend([f"## {drone_count}-View Methods", ""])
        lines.append("| Rank | Method | Family | Strict quality | Delta vs single | Relative gain % | Regret vs best | AP50-95 | Found rate |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_rows:
            if int(row["drone_count"]) != drone_count:
                continue
            lines.append(
                "| "
                + f"{int(row['rank_within_drone_count'])} | "
                + f"`{row['method_label']}` | {row['method_family']} | {fmt(float(row['mean_scene_expected_strict_quality']))} | "
                + f"{fmt(float(row['delta_vs_single_reference_strict_quality']))} | "
                + f"{fmt(float(row['relative_gain_vs_single_reference_percent']))} | "
                + f"{fmt(float(row['regret_vs_best_in_same_drone_count']))} | "
                + f"{fmt(float(row['mean_scene_expected_target_ap50_95']))} | "
                + f"{fmt(float(row['mean_scene_expected_target_found_rate']))} |"
            )
        lines.append("")

    best_by_count: dict[int, dict[str, object]] = {}
    for row in summary_rows:
        drone_count = int(row["drone_count"])
        current = best_by_count.get(drone_count)
        if current is None or float(row["mean_scene_expected_strict_quality"]) > float(
            current["mean_scene_expected_strict_quality"]
        ):
            best_by_count[drone_count] = row

    lines.extend(["## Best Method Per View Count", ""])
    for drone_count in sorted(best_by_count):
        row = best_by_count[drone_count]
        lines.append(
            f"- `{drone_count}` view(s): `{row['method_label']}` with strict quality "
            f"`{fmt(float(row['mean_scene_expected_strict_quality']))}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use this harmonized table when you want to compare methods directly on one shared coalition-quality metric.",
            "- Use protocol-specific AP50-95 and found-rate columns only as secondary information inside the threshold family.",
            "- Keep marginal analyses and ring Shapley separate from coalition-method ranking: they measure value attribution, not coalition quality.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_view_csv = Path(args.scene_view_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_records = load_scene_records(scene_view_csv)
    catalog = build_method_catalog()
    combo_metric_rows = build_combo_metric_rows(scene_records)
    summary_rows = summarize_scene_balanced(combo_metric_rows, catalog)

    combo_path = output_dir / "harmonized_combo_metric_rows.csv"
    summary_path = output_dir / "harmonized_method_summary.csv"
    report_path = output_dir / "harmonized_method_comparison_report.md"

    write_csv(combo_path, combo_metric_rows)
    write_csv(summary_path, summary_rows)
    write_markdown_report(summary_rows, report_path)


if __name__ == "__main__":
    main()
