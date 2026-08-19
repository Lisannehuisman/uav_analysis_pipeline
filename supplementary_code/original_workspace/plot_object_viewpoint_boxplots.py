from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRIC = "ap50_95"
VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot viewpoint-sensitive boxplots per object class from cached per-image metrics."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Cached per-image CSV for a single model.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to infer valid class names and stable plotting order.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label to show in figure titles.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=sorted(VALID_METRICS),
        help="Metric to visualize against viewpoint variables.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/object_viewpoint_boxplots",
        help="Folder for per-object viewpoint figures.",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay individual viewpoint samples as jittered points.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.55,
        help="Transparency for jittered viewpoint points.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    names: list[str] = []
    for row in rows:
        class_name = row.get("class_name", "").strip()
        if class_name and class_name not in names:
            names.append(class_name)
    return names


def parse_viewpoint_tokens(image_path: str, class_names: list[str]) -> dict[str, str] | None:
    stem = Path(image_path).stem.lower()
    object_token_match = re.search(r"^S0-SM_([^-]+)-", Path(image_path).stem, re.IGNORECASE)
    if not object_token_match:
        return None
    object_token = object_token_match.group(1).lower()

    class_name = None
    for candidate in sorted(class_names, key=len, reverse=True):
        candidate_lower = candidate.lower()
        if object_token.startswith(candidate_lower):
            class_name = candidate
            break

    if class_name is None:
        fallback = re.match(r"([a-zA-Z]+)", object_token)
        if fallback:
            class_name = fallback.group(1).lower()
        else:
            return None

    azimuth_match = re.search(r"-az(\d+)", stem)
    elevation_match = re.search(r"-el([a-z]+)", stem)
    radius_match = re.search(r"-rad([a-z]+)", stem)

    if not azimuth_match or not elevation_match or not radius_match:
        return None

    elevation = elevation_match.group(1)
    if elevation == "ellow":
        elevation = "low"
    elif elevation == "elmid":
        elevation = "mid"
    elif elevation == "elhigh":
        elevation = "high"

    radius = radius_match.group(1)

    return {
        "class_name": class_name,
        "azimuth": azimuth_match.group(1),
        "elevation": elevation,
        "radius": radius,
    }


def build_records(rows: list[dict[str, str]], class_names: list[str], metric: str) -> list[dict[str, str | float]]:
    records: list[dict[str, str | float]] = []
    for row in rows:
        tokens = parse_viewpoint_tokens(row["image"], class_names)
        if tokens is None:
            continue
        metric_value = row.get(metric)
        if metric_value in (None, ""):
            continue
        records.append(
            {
                "class_name": tokens["class_name"],
                "azimuth": tokens["azimuth"],
                "elevation": tokens["elevation"],
                "radius": tokens["radius"],
                "metric_value": float(metric_value),
            }
        )
    return records


def ordered_unique(values: list[str], preferred: list[str] | None = None) -> list[str]:
    unique = list(dict.fromkeys(values))
    if preferred is None:
        try:
            return sorted(unique, key=lambda value: int(value))
        except ValueError:
            return sorted(unique)
    preferred_present = [value for value in preferred if value in unique]
    extras = [value for value in unique if value not in preferred_present]
    return preferred_present + extras


def grouped_values(records: list[dict[str, str | float]], group_key: str) -> tuple[list[str], list[list[float]]]:
    values_by_group: dict[str, list[float]] = defaultdict(list)
    for record in records:
        values_by_group[str(record[group_key])].append(float(record["metric_value"]))

    if group_key == "elevation":
        groups = ordered_unique(list(values_by_group.keys()), preferred=ELEVATION_ORDER)
    elif group_key == "radius":
        groups = ordered_unique(list(values_by_group.keys()), preferred=RADIUS_ORDER)
    else:
        groups = ordered_unique(list(values_by_group.keys()))

    return groups, [values_by_group[group] for group in groups]


def draw_group_boxplot(
    ax,
    records: list[dict[str, str | float]],
    group_key: str,
    title: str,
    show_points: bool,
    point_alpha: float,
) -> None:
    groups, data = grouped_values(records, group_key)
    if not groups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    box = ax.boxplot(data, tick_labels=groups, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe2f3")
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("#d62728")
        median.set_linewidth(1.6)

    if show_points:
        rng = np.random.default_rng(42)
        for idx, values in enumerate(data, start=1):
            if not values:
                continue
            jitter_x = rng.normal(idx, 0.045, size=len(values))
            ax.scatter(
                jitter_x,
                values,
                s=16,
                alpha=point_alpha,
                color="#1f77b4",
                edgecolors="none",
                zorder=3,
            )

    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def create_class_figure(
    class_name: str,
    class_records: list[dict[str, str | float]],
    metric: str,
    label: str,
    output_path: Path,
    show_points: bool,
    point_alpha: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    draw_group_boxplot(
        axes[0],
        class_records,
        "azimuth",
        f"{class_name}: azimuth",
        show_points=show_points,
        point_alpha=point_alpha,
    )
    draw_group_boxplot(
        axes[1],
        class_records,
        "elevation",
        f"{class_name}: elevation",
        show_points=show_points,
        point_alpha=point_alpha,
    )
    draw_group_boxplot(
        axes[2],
        class_records,
        "radius",
        f"{class_name}: radius",
        show_points=show_points,
        point_alpha=point_alpha,
    )
    axes[0].set_ylabel(metric.upper())
    fig.suptitle(f"{label} viewpoint sensitivity for {class_name} ({metric.upper()})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_csv(output_path: Path, records: list[dict[str, str | float]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "group_type", "group_value", "count", "mean", "median"],
        )
        writer.writeheader()

        for class_name in sorted({str(record["class_name"]) for record in records}):
            class_records = [record for record in records if str(record["class_name"]) == class_name]
            for group_key in ("azimuth", "elevation", "radius"):
                groups, data = grouped_values(class_records, group_key)
                for group_value, values in zip(groups, data):
                    if not values:
                        continue
                    sorted_values = sorted(values)
                    count = len(sorted_values)
                    midpoint = count // 2
                    median = (
                        sorted_values[midpoint]
                        if count % 2 == 1
                        else (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
                    )
                    writer.writerow(
                        {
                            "class_name": class_name,
                            "group_type": group_key,
                            "group_value": group_value,
                            "count": count,
                            "mean": sum(sorted_values) / count,
                            "median": median,
                        }
                    )


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_csv = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(per_class_csv)
    rows = read_csv_rows(per_image_path)
    records = build_records(rows, class_names, args.metric)

    class_order = class_names or sorted({str(record["class_name"]) for record in records})
    for class_name in class_order:
        class_records = [record for record in records if str(record["class_name"]) == class_name]
        if not class_records:
            continue
        create_class_figure(
            class_name=class_name,
            class_records=class_records,
            metric=args.metric,
            label=args.label,
            output_path=output_dir / f"{class_name}_{args.metric}_viewpoint_boxplots.png",
            show_points=args.show_points,
            point_alpha=args.point_alpha,
        )

    write_summary_csv(output_dir / f"viewpoint_summary_{args.metric}.csv", records)
    print(f"Saved per-object viewpoint figures to: {output_dir}")


if __name__ == "__main__":
    main()
