from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a figure showing the best viewpoint per object using cached per-image metrics."
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
        help="Metric used to define the best viewpoint.",
    )
    parser.add_argument("--label", default="S0_M4", help="Model label shown in the report title.")
    parser.add_argument(
        "--output-dir",
        default="comparison_output/best_viewpoint_report",
        help="Folder where the summary figure and CSV will be written.",
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


def parse_image_metadata(image_path: str, class_names: list[str]) -> dict[str, str] | None:
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


def summarize_best_viewpoints(
    rows: list[dict[str, str]],
    class_names: list[str],
    metric: str,
) -> list[dict[str, str | float]]:
    grouped: dict[str, dict[str, list[tuple[float, str]]]] = defaultdict(lambda: defaultdict(list))
    azimuth_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    elevation_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    radius_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        parsed = parse_image_metadata(row["image"], class_names)
        if parsed is None:
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        metric_value = float(value)
        key = f"az{parsed['azimuth']}_el{parsed['elevation']}_rad{parsed['radius']}"
        grouped[parsed["class_name"]][key].append((metric_value, row["image"]))
        azimuth_grouped[parsed["class_name"]][parsed["azimuth"]].append(metric_value)
        elevation_grouped[parsed["class_name"]][parsed["elevation"]].append(metric_value)
        radius_grouped[parsed["class_name"]][parsed["radius"]].append(metric_value)

    best_rows: list[dict[str, str | float]] = []
    for class_name in sorted(grouped.keys()):
        best_key = None
        best_median = -1.0
        best_mean = -1.0
        best_count = 0
        best_image = None

        for viewpoint_key, values in grouped[class_name].items():
            metric_values = [item[0] for item in values]
            current_median = median(metric_values)
            current_mean = mean(metric_values)
            if current_median > best_median or (
                math.isclose(current_median, best_median) and current_mean > best_mean
            ):
                best_key = viewpoint_key
                best_median = current_median
                best_mean = current_mean
                best_count = len(metric_values)
                best_image = max(values, key=lambda item: item[0])[1]

        if best_key is None or best_image is None:
            continue

        parts = re.match(r"az(\d+)_el([a-z]+)_rad([a-z]+)", best_key)
        if parts is None:
            continue

        best_azimuth = max(
            azimuth_grouped[class_name].items(),
            key=lambda item: (median(item[1]), mean(item[1]), len(item[1])),
        )
        best_elevation = max(
            elevation_grouped[class_name].items(),
            key=lambda item: (median(item[1]), mean(item[1]), len(item[1])),
        )
        best_radius = max(
            radius_grouped[class_name].items(),
            key=lambda item: (median(item[1]), mean(item[1]), len(item[1])),
        )

        best_rows.append(
            {
                "class_name": class_name,
                "best_single_azimuth": parts.group(1),
                "best_single_elevation": parts.group(2),
                "best_single_radius": parts.group(3),
                "best_single_median": best_median,
                "best_single_mean": best_mean,
                "best_single_count": best_count,
                "best_average_azimuth": best_azimuth[0],
                "best_average_elevation": best_elevation[0],
                "best_average_radius": best_radius[0],
                "best_average_azimuth_median": median(best_azimuth[1]),
                "best_average_elevation_median": median(best_elevation[1]),
                "best_average_radius_median": median(best_radius[1]),
                "image_path": best_image,
            }
        )

    return best_rows


def write_summary_csv(output_path: Path, rows: list[dict[str, str | float]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "best_single_azimuth",
                "best_single_elevation",
                "best_single_radius",
                "best_single_median",
                "best_single_mean",
                "best_single_count",
                "best_average_azimuth",
                "best_average_elevation",
                "best_average_radius",
                "best_average_azimuth_median",
                "best_average_elevation_median",
                "best_average_radius_median",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def add_table(ax, rows: list[dict[str, str | float]], metric: str) -> None:
    ax.axis("off")
    headers = [
        "Object",
        "Best single",
        f"{metric} med",
        "Best avg viewpoint",
        "Az med",
        "El med",
        "Rad med",
    ]
    cell_text = [
        [
            str(row["class_name"]),
            f"az{row['best_single_azimuth']} / {row['best_single_elevation']} / {row['best_single_radius']}",
            f"{float(row['best_single_median']):.3f}",
            f"az{row['best_average_azimuth']} / {row['best_average_elevation']} / {row['best_average_radius']}",
            f"{float(row['best_average_azimuth_median']):.3f}",
            f"{float(row['best_average_elevation_median']):.3f}",
            f"{float(row['best_average_radius_median']):.3f}",
        ]
        for row in rows
    ]
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)


def add_image_grid(fig, rows: list[dict[str, str | float]], start_y: float = 0.02) -> None:
    cols = 5
    rows_count = math.ceil(len(rows) / cols)
    left_margin = 0.04
    right_margin = 0.04
    h_gap = 0.02
    v_gap = 0.055
    usable_width = 1 - left_margin - right_margin - h_gap * (cols - 1)
    box_width = usable_width / cols
    box_height = 0.22

    for idx, row in enumerate(rows):
        grid_row = idx // cols
        grid_col = idx % cols
        left = left_margin + grid_col * (box_width + h_gap)
        bottom = start_y + (rows_count - 1 - grid_row) * (box_height + v_gap)

        image_ax = fig.add_axes([left, bottom, box_width, box_height])
        image_ax.axis("off")
        image = mpimg.imread(str(row["image_path"]))
        image_ax.imshow(image)
        image_ax.set_title(
            (
                f"{row['class_name']}\n"
                f"best single: az{row['best_single_azimuth']} | el{row['best_single_elevation']} | rad{row['best_single_radius']}\n"
                f"best avg: az{row['best_average_azimuth']} | {row['best_average_elevation']} | {row['best_average_radius']}"
            ),
            fontsize=9,
        )


def create_report_figure(
    rows: list[dict[str, str | float]],
    metric: str,
    label: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Best single viewpoint per object for {label}", fontsize=24, y=0.98)

    table_ax = fig.add_axes([0.04, 0.68, 0.92, 0.23])
    add_table(table_ax, rows, metric)
    add_image_grid(fig, rows, start_y=0.04)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_csv = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(per_class_csv)
    rows = read_csv_rows(per_image_path)
    best_rows = summarize_best_viewpoints(rows, class_names, args.metric)

    summary_csv = output_dir / f"best_viewpoint_report_{args.metric}.csv"
    summary_png = output_dir / f"best_viewpoint_report_{args.metric}.png"
    write_summary_csv(summary_csv, best_rows)
    create_report_figure(best_rows, args.metric, args.label, summary_png)

    print(f"Saved report image to: {summary_png}")
    print(f"Saved report CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
