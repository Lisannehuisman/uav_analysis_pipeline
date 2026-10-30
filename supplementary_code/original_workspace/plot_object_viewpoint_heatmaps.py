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


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-object viewpoint heatmaps from cached per-image metrics."
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
        help="Metric to visualize in the heatmaps.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label shown in figure titles.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/object_viewpoint_heatmaps",
        help="Folder where heatmap figures and summaries will be written.",
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


def parse_viewpoint_tokens(image_path: str, class_names: list[str]) -> dict[str, str] | None:
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
    return sum(values) / len(values) if values else float("nan")


def ordered_azimuths(records: list[dict[str, str | float]]) -> list[str]:
    azimuths = sorted({str(record["azimuth"]) for record in records}, key=lambda value: int(value))
    return azimuths


def build_records(
    rows: list[dict[str, str]],
    class_names: list[str],
    metric: str,
) -> list[dict[str, str | float]]:
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


def build_heatmap_matrix(
    records: list[dict[str, str | float]],
    radius: str,
    azimuths: list[str],
) -> np.ndarray:
    matrix = np.full((len(ELEVATION_ORDER), len(azimuths)), np.nan, dtype=float)
    for row_idx, elevation in enumerate(ELEVATION_ORDER):
        for col_idx, azimuth in enumerate(azimuths):
            values = [
                float(record["metric_value"])
                for record in records
                if str(record["radius"]) == radius
                and str(record["elevation"]) == elevation
                and str(record["azimuth"]) == azimuth
            ]
            matrix[row_idx, col_idx] = mean(values)
    return matrix


def best_cell(matrix: np.ndarray) -> tuple[int, int] | None:
    if np.isnan(matrix).all():
        return None
    return tuple(np.unravel_index(np.nanargmax(matrix), matrix.shape))


def create_class_heatmap_figure(
    class_name: str,
    class_records: list[dict[str, str | float]],
    metric: str,
    label: str,
    output_path: Path,
) -> list[dict[str, str | float]]:
    azimuths = ordered_azimuths(class_records)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    summaries: list[dict[str, str | float]] = []

    vmin = 0.0
    vmax = 1.0

    for idx, radius in enumerate(RADIUS_ORDER):
        matrix = build_heatmap_matrix(class_records, radius, azimuths)
        ax = axes[idx]
        im = ax.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{class_name}: radius={radius}")
        ax.set_xticks(range(len(azimuths)))
        ax.set_xticklabels(azimuths, rotation=45, ha="right")
        ax.set_xlabel("Azimuth")
        if idx == 0:
            ax.set_yticks(range(len(ELEVATION_ORDER)))
            ax.set_yticklabels(ELEVATION_ORDER)
            ax.set_ylabel("Elevation")
        else:
            ax.set_yticks(range(len(ELEVATION_ORDER)))
            ax.set_yticklabels([])

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=7,
                )

        best = best_cell(matrix)
        if best is not None:
            best_row, best_col = best
            ax.scatter(best_col, best_row, s=160, facecolors="none", edgecolors="white", linewidths=2)
            summaries.append(
                {
                    "class_name": class_name,
                    "radius": radius,
                    "best_elevation": ELEVATION_ORDER[best_row],
                    "best_azimuth": azimuths[best_col],
                    "best_score": float(matrix[best_row, best_col]),
                }
            )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(metric.upper())
    fig.suptitle(f"{label} best viewpoint heatmaps for {class_name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return summaries


def write_summary_csv(output_path: Path, rows: list[dict[str, str | float]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "radius", "best_elevation", "best_azimuth", "best_score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_overall_best_csv(output_path: Path, rows: list[dict[str, str | float]]) -> None:
    best_rows: list[dict[str, str | float]] = []
    grouped: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_name"])].append(row)

    for class_name, class_rows in sorted(grouped.items()):
        best = max(class_rows, key=lambda row: float(row["best_score"]))
        best_rows.append(best)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "radius", "best_elevation", "best_azimuth", "best_score"],
        )
        writer.writeheader()
        writer.writerows(best_rows)


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_csv = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(per_class_csv)
    rows = read_csv_rows(per_image_path)
    records = build_records(rows, class_names, args.metric)

    summaries: list[dict[str, str | float]] = []
    class_order = class_names or sorted({str(record["class_name"]) for record in records})
    for class_name in class_order:
        class_records = [record for record in records if str(record["class_name"]) == class_name]
        if not class_records:
            continue
        summaries.extend(
            create_class_heatmap_figure(
                class_name=class_name,
                class_records=class_records,
                metric=args.metric,
                label=args.label,
                output_path=output_dir / f"{class_name}_{args.metric}_heatmaps.png",
            )
        )

    write_summary_csv(output_dir / f"best_per_radius_{args.metric}.csv", summaries)
    write_overall_best_csv(output_dir / f"best_overall_{args.metric}.csv", summaries)
    print(f"Saved per-object heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()


# python plot_object_viewpoint_heatmaps.py `
#   --per-image ".\comparison_output\per_image_metrics_model_b.csv" `
#   --per-class-csv ".\comparison_output\per_class_ap50_95.csv" `
#   --metric ap50_95 `
#   --label "S0_M4" `
#   --output-dir ".\comparison_output\object_viewpoint_heatmaps_s0_m4"
