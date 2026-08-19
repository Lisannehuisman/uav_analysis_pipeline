from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank best and worst viewpoint combinations per object from cached per-image metrics."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Cached per-image CSV for a single model.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to infer class names.",
    )
    parser.add_argument(
        "--metric",
        default="ap50_95",
        choices=sorted(VALID_METRICS),
        help="Metric used to rank viewpoint quality.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/best_viewpoint_analysis",
        help="Folder for analysis CSV outputs.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    seen: list[str] = []
    for row in rows:
        class_name = row.get("class_name", "").strip()
        if class_name and class_name not in seen:
            seen.append(class_name)
    return seen


def parse_record(image_path: str, class_names: list[str]) -> dict[str, str] | None:
    stem = Path(image_path).stem
    object_token_match = re.search(r"^S0-SM_([^-]+)-", stem, re.IGNORECASE)
    if not object_token_match:
        return None
    object_token = object_token_match.group(1).lower()

    class_name = None
    for candidate in sorted(class_names, key=len, reverse=True):
        if object_token.startswith(candidate.lower()):
            class_name = candidate
            break
    if class_name is None:
        fallback = re.match(r"([a-zA-Z]+)", object_token)
        if not fallback:
            return None
        class_name = fallback.group(1).lower()

    azimuth_match = re.search(r"-az(\d+)", stem, re.IGNORECASE)
    elevation_match = re.search(r"-el([a-z]+)", stem, re.IGNORECASE)
    radius_match = re.search(r"-rad([a-z]+)", stem, re.IGNORECASE)
    if not azimuth_match or not elevation_match or not radius_match:
        return None

    elevation = elevation_match.group(1).lower()
    if elevation == "ellow":
        elevation = "low"
    radius = radius_match.group(1).lower()

    return {
        "class_name": class_name,
        "azimuth": azimuth_match.group(1),
        "elevation": elevation,
        "radius": radius,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarize_group(values_by_key: dict[str, list[float]]) -> tuple[dict[str, float | str], dict[str, float | str]]:
    ranked = sorted(
        (
            {
                "group": key,
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
            }
            for key, values in values_by_key.items()
            if values
        ),
        key=lambda row: row["median"],
    )
    return ranked[-1], ranked[0]


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_csv = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(per_class_csv)
    rows = read_csv_rows(per_image_path)

    per_object_combo: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_object_azimuth: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_object_elevation: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_object_radius: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        parsed = parse_record(row["image"], class_names)
        if parsed is None:
            continue
        metric_value = row.get(args.metric)
        if metric_value in (None, ""):
            continue
        value = float(metric_value)
        class_name = parsed["class_name"]
        combo_key = f"az{parsed['azimuth']}_el{parsed['elevation']}_rad{parsed['radius']}"
        per_object_combo[class_name][combo_key].append(value)
        per_object_azimuth[class_name][parsed["azimuth"]].append(value)
        per_object_elevation[class_name][parsed["elevation"]].append(value)
        per_object_radius[class_name][parsed["radius"]].append(value)

    combo_rows: list[dict[str, float | str]] = []
    factor_rows: list[dict[str, float | str]] = []

    for class_name in sorted(per_object_combo.keys()):
        best_combo, worst_combo = summarize_group(per_object_combo[class_name])
        combo_rows.append(
            {
                "class_name": class_name,
                "best_combo": best_combo["group"],
                "best_combo_count": best_combo["count"],
                "best_combo_mean": best_combo["mean"],
                "best_combo_median": best_combo["median"],
                "worst_combo": worst_combo["group"],
                "worst_combo_count": worst_combo["count"],
                "worst_combo_mean": worst_combo["mean"],
                "worst_combo_median": worst_combo["median"],
                "combo_median_gap": float(best_combo["median"]) - float(worst_combo["median"]),
            }
        )

        for group_type, source in (
            ("azimuth", per_object_azimuth[class_name]),
            ("elevation", per_object_elevation[class_name]),
            ("radius", per_object_radius[class_name]),
        ):
            best_group, worst_group = summarize_group(source)
            factor_rows.append(
                {
                    "class_name": class_name,
                    "group_type": group_type,
                    "best_group": best_group["group"],
                    "best_count": best_group["count"],
                    "best_mean": best_group["mean"],
                    "best_median": best_group["median"],
                    "worst_group": worst_group["group"],
                    "worst_count": worst_group["count"],
                    "worst_mean": worst_group["mean"],
                    "worst_median": worst_group["median"],
                    "median_gap": float(best_group["median"]) - float(worst_group["median"]),
                }
            )

    with (output_dir / f"best_worst_combo_{args.metric}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "best_combo",
                "best_combo_count",
                "best_combo_mean",
                "best_combo_median",
                "worst_combo",
                "worst_combo_count",
                "worst_combo_mean",
                "worst_combo_median",
                "combo_median_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(combo_rows)

    with (output_dir / f"best_worst_factors_{args.metric}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "group_type",
                "best_group",
                "best_count",
                "best_mean",
                "best_median",
                "worst_group",
                "worst_count",
                "worst_mean",
                "worst_median",
                "median_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(factor_rows)

    print(f"Saved viewpoint analysis to: {output_dir}")


if __name__ == "__main__":
    main()



# python analyze_best_viewpoints.py `
#   --per-image ".\comparison_output\per_image_metrics_model_b.csv" `
#   --per-class-csv ".\comparison_output\per_class_ap50_95.csv" `
#   --metric ap50_95 `
#   --output-dir ".\comparison_output\best_viewpoint_analysis_s0_m4"
