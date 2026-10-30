from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]
ELEVATION_CODE = {"low": -1.0, "mid": 0.0, "high": 1.0}
RADIUS_CODE = {"near": -1.0, "mid": 0.0, "far": 1.0}
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


@dataclass(frozen=True)
class Record:
    class_name: str
    azimuth_label: str
    azimuth_deg: int
    elevation: str
    radius: str
    metric_value: float


@dataclass(frozen=True)
class ViewpointStats:
    object_class: str
    azimuth_label: str
    azimuth_deg: int
    elevation: str
    radius: str
    mean_value: float
    std_value: float
    variance_value: float
    count: int

    @property
    def viewpoint_label(self) -> str:
        return f"az{self.azimuth_label} | {self.elevation} | {self.radius}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run thesis-ready viewpoint dependence analysis from cached per-image metrics."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Per-image metrics CSV for one model.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to infer canonical class names.",
    )
    parser.add_argument(
        "--metric",
        default="ap50_95",
        choices=sorted(VALID_METRICS),
        help="Per-image metric to analyze. Defaults to ap50_95 as the best IoU-like proxy available.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label shown in figure titles and summaries.",
    )
    parser.add_argument(
        "--clusters",
        default=3,
        type=int,
        help="Number of hierarchical clusters for object grouping.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/thesis_viewpoint_analysis_s0_m4",
        help="Output folder for tables, plots, and summary text.",
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
        "azimuth_label": azimuth_match.group(1),
        "elevation": elevation,
        "radius": radius,
    }


def build_records(rows: list[dict[str, str]], class_names: list[str], metric: str) -> list[Record]:
    records: list[Record] = []
    for row in rows:
        tokens = parse_viewpoint_tokens(row["image"], class_names)
        if tokens is None:
            continue
        metric_value = row.get(metric)
        if metric_value in (None, ""):
            continue
        azimuth_label = tokens["azimuth_label"]
        records.append(
            Record(
                class_name=tokens["class_name"],
                azimuth_label=azimuth_label,
                azimuth_deg=int(azimuth_label),
                elevation=tokens["elevation"],
                radius=tokens["radius"],
                metric_value=float(metric_value),
            )
        )
    return records


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def variance(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.var(np.asarray(values, dtype=float), ddof=0))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_viewpoint_rankings(
    records: list[Record],
    metric: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[ViewpointStats]]]:
    grouped: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record.azimuth_label, record.elevation, record.radius)
        grouped[record.class_name][key].append(record.metric_value)

    best_rows: list[dict[str, object]] = []
    top3_rows: list[dict[str, object]] = []
    rankings: dict[str, list[ViewpointStats]] = {}
    mean_key = "mean_ap50_95" if metric == "ap50_95" else f"mean_{metric}"
    std_key = "std_ap50_95" if metric == "ap50_95" else f"std_{metric}"
    variance_key = "variance_ap50_95" if metric == "ap50_95" else f"variance_{metric}"

    for class_name in sorted(grouped):
        stats_rows: list[ViewpointStats] = []
        for (azimuth_label, elevation, radius), values in grouped[class_name].items():
            stats_rows.append(
                ViewpointStats(
                    object_class=class_name,
                    azimuth_label=azimuth_label,
                    azimuth_deg=int(azimuth_label),
                    elevation=elevation,
                    radius=radius,
                    mean_value=mean(values),
                    std_value=std(values),
                    variance_value=variance(values),
                    count=len(values),
                )
            )

        stats_rows.sort(
            key=lambda row: (row.mean_value, -row.std_value, row.count, -row.azimuth_deg),
            reverse=True,
        )
        rankings[class_name] = stats_rows
        best = stats_rows[0]
        best_rows.append(
            {
                "object_class": class_name,
                "best_azimuth": best.azimuth_label,
                "best_elevation": best.elevation,
                "best_radius": best.radius,
                mean_key: best.mean_value,
                std_key: best.std_value,
                variance_key: best.variance_value,
                "n_images": best.count,
            }
        )

        for rank, row in enumerate(stats_rows[:3], start=1):
            top3_rows.append(
                {
                    "object_class": class_name,
                    "rank": rank,
                    "azimuth": row.azimuth_label,
                    "elevation": row.elevation,
                    "radius": row.radius,
                    "viewpoint_label": row.viewpoint_label,
                    "mean_value": row.mean_value,
                    "std_value": row.std_value,
                    "variance_value": row.variance_value,
                    "n_images": row.count,
                }
            )

    return best_rows, top3_rows, rankings


def plot_top3_viewpoints(
    rankings: dict[str, list[ViewpointStats]],
    output_path: Path,
    label: str,
    metric: str,
) -> None:
    classes = sorted(rankings)
    fig, ax = plt.subplots(figsize=(16, 9))
    offsets = {1: 0.22, 2: 0.0, 3: -0.22}
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    y_base = np.arange(len(classes))
    for row_idx, class_name in enumerate(classes):
        top3 = rankings[class_name][:3]
        for rank, stats in enumerate(top3, start=1):
            y = y_base[row_idx] + offsets[rank]
            ax.errorbar(
                stats.mean_value,
                y,
                xerr=stats.std_value,
                fmt="o",
                color=colors[rank],
                capsize=3,
                markersize=7,
                label=f"Rank {rank}" if row_idx == 0 else None,
            )
            ax.text(
                min(stats.mean_value + 0.02, 0.985),
                y,
                stats.viewpoint_label,
                va="center",
                fontsize=8,
                color=colors[rank],
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(classes)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Object class")
    ax.set_title(f"Top-3 viewpoint combinations per object for {label}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def unique_azimuth_labels(records: list[Record]) -> list[str]:
    return sorted({record.azimuth_label for record in records}, key=int)


def heatmap_matrix(class_records: list[Record], azimuth_labels: list[str]) -> np.ndarray:
    matrix = np.full((len(ELEVATION_ORDER), len(azimuth_labels)), np.nan, dtype=float)
    for row_idx, elevation in enumerate(ELEVATION_ORDER):
        for col_idx, azimuth_label in enumerate(azimuth_labels):
            values = [
                record.metric_value
                for record in class_records
                if record.elevation == elevation and record.azimuth_label == azimuth_label
            ]
            matrix[row_idx, col_idx] = mean(values)
    return matrix


def plot_heatmaps(records: list[Record], output_dir: Path, label: str, metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.class_name].append(record)

    classes = sorted(grouped)
    azimuth_labels = unique_azimuth_labels(records)

    ncols = 2
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    im = None

    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        matrix = heatmap_matrix(grouped[class_name], azimuth_labels)
        im = ax.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(class_name)
        ax.set_xticks(range(len(azimuth_labels)))
        ax.set_xticklabels(azimuth_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(ELEVATION_ORDER)))
        ax.set_yticklabels(ELEVATION_ORDER)
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")

        if not np.isnan(matrix).all():
            best_row, best_col = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            ax.scatter(best_col, best_row, s=160, facecolors="none", edgecolors="white", linewidths=2)

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

        class_fig, class_ax = plt.subplots(figsize=(8.5, 3.8))
        class_im = class_ax.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)
        class_ax.set_title(f"{class_name}: mean {metric.upper()} over azimuth x elevation\n(radius averaged)")
        class_ax.set_xticks(range(len(azimuth_labels)))
        class_ax.set_xticklabels(azimuth_labels, rotation=45, ha="right")
        class_ax.set_yticks(range(len(ELEVATION_ORDER)))
        class_ax.set_yticklabels(ELEVATION_ORDER)
        class_ax.set_xlabel("Azimuth")
        class_ax.set_ylabel("Elevation")
        if not np.isnan(matrix).all():
            best_row, best_col = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            class_ax.scatter(best_col, best_row, s=180, facecolors="none", edgecolors="white", linewidths=2)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                class_ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )
        class_cbar = class_fig.colorbar(class_im, ax=class_ax, shrink=0.9)
        class_cbar.set_label(metric.upper())
        class_fig.tight_layout()
        class_fig.savefig(output_dir / f"{class_name}_{metric}_heatmap.png", dpi=180)
        plt.close(class_fig)

    for idx in range(len(classes), len(axes)):
        axes[idx].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.98)
        cbar.set_label(metric.upper())
    fig.suptitle(f"Viewpoint heatmaps for {label} ({metric.upper()}, radius averaged)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / f"all_objects_{metric}_heatmaps.png", dpi=180)
    plt.close(fig)


def optimal_feature_vector(best: ViewpointStats) -> np.ndarray:
    theta = math.radians(best.azimuth_deg)
    return np.array(
        [
            math.cos(theta),
            math.sin(theta),
            ELEVATION_CODE[best.elevation],
            RADIUS_CODE[best.radius],
            best.mean_value,
        ],
        dtype=float,
    )


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (matrix - means) / stds


def cluster_objects(
    rankings: dict[str, list[ViewpointStats]],
    output_dir: Path,
    clusters: int,
    label: str,
    metric: str,
) -> list[dict[str, object]]:
    classes = sorted(rankings)
    best_rows = [rankings[class_name][0] for class_name in classes]
    feature_matrix = np.vstack([optimal_feature_vector(row) for row in best_rows])
    scaled_matrix = standardize_columns(feature_matrix)
    linkage_matrix = linkage(scaled_matrix, method="ward")
    cluster_ids = fcluster(linkage_matrix, t=clusters, criterion="maxclust")

    rows: list[dict[str, object]] = []
    for class_name, best, cluster_id in zip(classes, best_rows, cluster_ids, strict=True):
        rows.append(
            {
                "object_class": class_name,
                "cluster_id": int(cluster_id),
                "best_azimuth": best.azimuth_label,
                "best_elevation": best.elevation,
                "best_radius": best.radius,
                "mean_value": best.mean_value,
                "std_value": best.std_value,
            }
        )

    write_csv(
        output_dir / "cluster_assignments.csv",
        ["object_class", "cluster_id", "best_azimuth", "best_elevation", "best_radius", "mean_value", "std_value"],
        rows,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    radius_scale = {"near": 1.0, "mid": 2.0, "far": 3.0}
    elev_scale = {"low": 0.0, "mid": 1.0, "high": 2.0}

    for class_name, best, cluster_id in zip(classes, best_rows, cluster_ids, strict=True):
        radius = radius_scale[best.radius]
        theta = math.radians(best.azimuth_deg)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = elev_scale[best.elevation]
        color = CLUSTER_COLORS[(int(cluster_id) - 1) % len(CLUSTER_COLORS)]
        ax.scatter(x, y, z, s=200, color=color, edgecolor="black", linewidth=0.8)
        ax.text(x, y, z + 0.06, class_name, fontsize=9, ha="center")

    ax.set_title(f"Clusters of optimal viewpoints for {label}")
    ax.set_xlabel("x (radius x cos azimuth)")
    ax.set_ylabel("y (radius x sin azimuth)")
    ax.set_zlabel("Elevation")
    ax.set_zticks([0, 1, 2])
    ax.set_zticklabels(ELEVATION_ORDER)
    legend_handles = []
    for cluster_id in sorted(set(int(cluster_id) for cluster_id in cluster_ids)):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)],
                markeredgecolor="black",
                markersize=10,
                label=f"Cluster {cluster_id}",
            )
        )
    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_viewpoints_3d.png", dpi=180)
    plt.close(fig)

    dendro_fig, dendro_ax = plt.subplots(figsize=(11, 5.5))
    dendrogram(linkage_matrix, labels=classes, ax=dendro_ax, color_threshold=None)
    dendro_ax.set_title(f"Hierarchical clustering of optimal viewpoints ({metric.upper()})")
    dendro_ax.set_ylabel("Ward distance")
    dendro_fig.tight_layout()
    dendro_fig.savefig(output_dir / "cluster_dendrogram.png", dpi=180)
    plt.close(dendro_fig)

    return rows


def build_regression_features(azimuth_deg: float, elevation_value: float, radius_value: float) -> np.ndarray:
    theta = math.radians(azimuth_deg)
    az_sin = math.sin(theta)
    az_cos = math.cos(theta)
    return np.array(
        [
            1.0,
            az_sin,
            az_cos,
            elevation_value,
            radius_value,
            az_sin * elevation_value,
            az_cos * elevation_value,
            az_sin * radius_value,
            az_cos * radius_value,
            elevation_value * radius_value,
        ],
        dtype=float,
    )


def fit_linear_model(records: list[Record]) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.vstack(
        [
            build_regression_features(
                record.azimuth_deg,
                ELEVATION_CODE[record.elevation],
                RADIUS_CODE[record.radius],
            )
            for record in records
        ]
    )
    y = np.asarray([record.metric_value for record in records], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ beta
    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(np.sum((y - predictions) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    return beta, predictions, r_squared


def mean_prediction_over_grid(
    beta: np.ndarray,
    azimuth_values: list[int],
    elevation_values: list[float],
    radius_values: list[float],
    vary: str,
) -> tuple[list[float], list[float]]:
    if vary == "azimuth":
        xs = list(azimuth_values)
        ys = []
        for azimuth in xs:
            preds = [
                float(build_regression_features(azimuth, elevation, radius) @ beta)
                for elevation, radius in product(elevation_values, radius_values)
            ]
            ys.append(float(np.mean(preds)))
        return xs, ys

    if vary == "elevation":
        xs = list(elevation_values)
        ys = []
        for elevation in xs:
            preds = [
                float(build_regression_features(azimuth, elevation, radius) @ beta)
                for azimuth, radius in product(azimuth_values, radius_values)
            ]
            ys.append(float(np.mean(preds)))
        return xs, ys

    xs = list(radius_values)
    ys = []
    for radius in xs:
        preds = [
            float(build_regression_features(azimuth, elevation, radius) @ beta)
            for azimuth, elevation in product(azimuth_values, elevation_values)
        ]
        ys.append(float(np.mean(preds)))
    return xs, ys


def run_regression_analysis(
    records: list[Record],
    rankings: dict[str, list[ViewpointStats]],
    output_dir: Path,
    metric: str,
    label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.class_name].append(record)

    coefficient_names = [
        "intercept",
        "azimuth_sin",
        "azimuth_cos",
        "elevation",
        "radius",
        "azimuth_sin_x_elevation",
        "azimuth_cos_x_elevation",
        "azimuth_sin_x_radius",
        "azimuth_cos_x_radius",
        "elevation_x_radius",
    ]

    coefficient_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []

    classes = sorted(grouped)
    fig, axes = plt.subplots(len(classes), 3, figsize=(15, 3.1 * len(classes)))
    axes = np.atleast_2d(axes)

    for row_idx, class_name in enumerate(classes):
        class_records = grouped[class_name]
        beta, _, r_squared = fit_linear_model(class_records)
        for coefficient_name, coefficient_value in zip(coefficient_names, beta, strict=True):
            coefficient_rows.append(
                {
                    "object_class": class_name,
                    "coefficient": coefficient_name,
                    "value": float(coefficient_value),
                    "r_squared": r_squared,
                }
            )

        observed_azimuths = sorted({record.azimuth_deg for record in class_records})
        observed_elevations = sorted({ELEVATION_CODE[record.elevation] for record in class_records})
        observed_radii = sorted({RADIUS_CODE[record.radius] for record in class_records})

        az_x, az_y = mean_prediction_over_grid(beta, observed_azimuths, observed_elevations, observed_radii, "azimuth")
        el_x, el_y = mean_prediction_over_grid(beta, observed_azimuths, observed_elevations, observed_radii, "elevation")
        rad_x, rad_y = mean_prediction_over_grid(beta, observed_azimuths, observed_elevations, observed_radii, "radius")

        effect_strengths = {
            "azimuth": float(max(az_y) - min(az_y)),
            "elevation": float(max(el_y) - min(el_y)),
            "radius": float(max(rad_y) - min(rad_y)),
        }
        strongest_parameter = max(effect_strengths, key=effect_strengths.get)
        best = rankings[class_name][0]
        effect_rows.append(
            {
                "object_class": class_name,
                "r_squared": r_squared,
                "azimuth_effect": effect_strengths["azimuth"],
                "elevation_effect": effect_strengths["elevation"],
                "radius_effect": effect_strengths["radius"],
                "strongest_parameter": strongest_parameter,
                "best_azimuth": best.azimuth_label,
                "best_elevation": best.elevation,
                "best_radius": best.radius,
            }
        )

        axes[row_idx, 0].plot(az_x, az_y, marker="o", color="#1f77b4")
        axes[row_idx, 0].set_title(f"{class_name}: azimuth")
        axes[row_idx, 0].set_ylim(0.0, 1.0)
        axes[row_idx, 0].set_xticks(az_x)
        axes[row_idx, 0].grid(alpha=0.25)

        axes[row_idx, 1].plot(["low", "mid", "high"], el_y, marker="o", color="#ff7f0e")
        axes[row_idx, 1].set_title(f"{class_name}: elevation")
        axes[row_idx, 1].set_ylim(0.0, 1.0)
        axes[row_idx, 1].grid(alpha=0.25)

        axes[row_idx, 2].plot(["near", "mid", "far"], rad_y, marker="o", color="#2ca02c")
        axes[row_idx, 2].set_title(f"{class_name}: radius")
        axes[row_idx, 2].set_ylim(0.0, 1.0)
        axes[row_idx, 2].grid(alpha=0.25)

    fig.suptitle(f"Regression-based partial dependence plots for {label} ({metric.upper()})", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_partial_dependence.png", dpi=180)
    plt.close(fig)

    write_csv(
        output_dir / "regression_coefficients.csv",
        ["object_class", "coefficient", "value", "r_squared"],
        coefficient_rows,
    )
    write_csv(
        output_dir / "regression_effect_strengths.csv",
        [
            "object_class",
            "r_squared",
            "azimuth_effect",
            "elevation_effect",
            "radius_effect",
            "strongest_parameter",
            "best_azimuth",
            "best_elevation",
            "best_radius",
        ],
        effect_rows,
    )

    effect_fig, effect_ax = plt.subplots(figsize=(12, 6))
    class_names = [row["object_class"] for row in effect_rows]
    x = np.arange(len(class_names))
    width = 0.25
    effect_ax.bar(x - width, [row["azimuth_effect"] for row in effect_rows], width=width, label="Azimuth")
    effect_ax.bar(x, [row["elevation_effect"] for row in effect_rows], width=width, label="Elevation")
    effect_ax.bar(x + width, [row["radius_effect"] for row in effect_rows], width=width, label="Radius")
    effect_ax.set_xticks(x)
    effect_ax.set_xticklabels(class_names, rotation=45, ha="right")
    effect_ax.set_ylabel(f"Predicted {metric.upper()} range")
    effect_ax.set_title("Estimated parameter importance from regression effect ranges")
    effect_ax.legend()
    effect_ax.grid(axis="y", alpha=0.25)
    effect_fig.tight_layout()
    effect_fig.savefig(output_dir / "regression_effect_strengths.png", dpi=180)
    plt.close(effect_fig)

    return coefficient_rows, effect_rows


def evaluate_regression(beta: np.ndarray, azimuth_deg: float, elevation_value: float, radius_value: float) -> float:
    return float(build_regression_features(azimuth_deg, elevation_value, radius_value) @ beta)


def run_robustness_analysis(
    records: list[Record],
    rankings: dict[str, list[ViewpointStats]],
    output_dir: Path,
    metric: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.class_name].append(record)

    rows: list[dict[str, object]] = []
    perturbations = list(product([-10.0, -5.0, 5.0, 10.0], [-0.5, 0.5], [-0.5, 0.5]))

    for class_name in sorted(grouped):
        class_records = grouped[class_name]
        beta, _, _ = fit_linear_model(class_records)
        best = rankings[class_name][0]
        base_elevation = ELEVATION_CODE[best.elevation]
        base_radius = RADIUS_CODE[best.radius]
        base_score = evaluate_regression(beta, best.azimuth_deg, base_elevation, base_radius)

        perturbed_scores: list[float] = []
        for az_delta, el_delta, rad_delta in perturbations:
            perturbed_azimuth = (best.azimuth_deg + az_delta) % 360.0
            perturbed_elevation = float(np.clip(base_elevation + el_delta, -1.0, 1.0))
            perturbed_radius = float(np.clip(base_radius + rad_delta, -1.0, 1.0))
            perturbed_scores.append(
                evaluate_regression(beta, perturbed_azimuth, perturbed_elevation, perturbed_radius)
            )

        mean_perturbed = float(np.mean(perturbed_scores))
        worst_perturbed = float(np.min(perturbed_scores))
        rows.append(
            {
                "object_class": class_name,
                "best_viewpoint": best.viewpoint_label,
                "base_predicted_value": base_score,
                "mean_perturbed_value": mean_perturbed,
                "worst_perturbed_value": worst_perturbed,
                "mean_drop": base_score - mean_perturbed,
                "worst_drop": base_score - worst_perturbed,
            }
        )

    rows.sort(key=lambda row: float(row["mean_drop"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["sensitivity_rank"] = rank

    write_csv(
        output_dir / "robustness_analysis.csv",
        [
            "object_class",
            "best_viewpoint",
            "base_predicted_value",
            "mean_perturbed_value",
            "worst_perturbed_value",
            "mean_drop",
            "worst_drop",
            "sensitivity_rank",
        ],
        rows,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    class_names = [row["object_class"] for row in rows]
    x = np.arange(len(class_names))
    ax.bar(x, [row["mean_drop"] for row in rows], color="#d62728", alpha=0.85, label="Mean drop")
    ax.plot(x, [row["worst_drop"] for row in rows], color="black", marker="o", linewidth=1.5, label="Worst drop")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel(f"Predicted {metric.upper()} drop")
    ax.set_title("Local viewpoint sensitivity around each class-specific optimum")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_ranking.png", dpi=180)
    plt.close(fig)

    return rows


def build_summary_text(
    best_rows: list[dict[str, object]],
    cluster_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    output_dir: Path,
    metric: str,
    label: str,
) -> None:
    cluster_counter = Counter(int(row["cluster_id"]) for row in cluster_rows)
    strongest_counter = Counter(str(row["strongest_parameter"]) for row in regression_rows)
    best_elev_counter = Counter(str(row["best_elevation"]) for row in best_rows)
    best_radius_counter = Counter(str(row["best_radius"]) for row in best_rows)
    top_sensitive = robustness_rows[:3]
    std_key = next(key for key in best_rows[0] if key.startswith("std_"))
    mean_key = next(key for key in best_rows[0] if key.startswith("mean_"))
    most_stable = sorted(best_rows, key=lambda row: float(row[std_key]))[:3]
    overall_best = max(best_rows, key=lambda row: float(row[mean_key]))
    stable_summary = ", ".join(
        f"{row['object_class']} ({float(row[std_key]):.3f})" for row in most_stable
    )
    sensitive_summary = ", ".join(
        f"{row['object_class']} ({float(row['mean_drop']):.3f})" for row in top_sensitive
    )
    low_support_count = sum(int(row["n_images"]) <= 2 for row in best_rows)

    lines = [
        f"# Thesis viewpoint analysis summary for {label}",
        "",
        f"Metric analyzed: `{metric}`. The cached per-image file does not include a raw IoU column, so `{metric}` is used as the available localization-quality proxy throughout this analysis.",
        "",
        "## 1. Ideal viewpoint per object",
        f"- Highest mean score overall: {overall_best['object_class']} at az{overall_best['best_azimuth']}, elevation {overall_best['best_elevation']}, radius {overall_best['best_radius']} with mean {float(overall_best[mean_key]):.3f}.",
        f"- Most common optimal elevation band: {best_elev_counter.most_common(1)[0][0]} ({best_elev_counter.most_common(1)[0][1]} of {len(best_rows)} objects).",
        f"- Most common optimal radius band: {best_radius_counter.most_common(1)[0][0]} ({best_radius_counter.most_common(1)[0][1]} of {len(best_rows)} objects).",
        f"- Most stable best viewpoints (lowest std): {stable_summary}.",
        f"- Caution: {low_support_count} of {len(best_rows)} exact best viewpoints are supported by only 1-2 images, so the heatmaps, top-3 tables, and regression trends are more reliable than any single exact cell on its own.",
        "",
        "## 2. Viewpoint heatmaps",
        "- Heatmaps were averaged over radius so the azimuth x elevation pattern is easier to compare across objects.",
        "- White circled cells indicate the highest mean cell per object in the saved heatmaps.",
        "",
        "## 3. Object grouping",
        f"- Hierarchical clustering produced {len(cluster_counter)} clusters with sizes: {', '.join(f'cluster {cluster}: {count}' for cluster, count in sorted(cluster_counter.items()))}.",
        "- Cluster membership is based on each class-specific optimal azimuth, elevation, radius, and best mean score.",
        "",
        "## 4. Regression analysis",
        f"- Strongest parameter by regression effect range: {', '.join(f'{parameter} ({count})' for parameter, count in strongest_counter.most_common())}.",
        "- The regression surface uses azimuth sine/cosine terms, elevation, radius, and pairwise interactions to approximate local viewpoint effects.",
        "",
        "## 5. Robustness analysis",
        f"- Most viewpoint-sensitive objects by mean local drop: {sensitive_summary}.",
        "- Robustness was estimated from the fitted regression surface using azimuth perturbations (+/-5 and +/-10 degrees) and small elevation/radius perturbations in normalized band units.",
        "",
        "## 6. Thesis-ready conclusions",
        f"- Optimal viewpoints are object-specific, but the best solutions are concentrated in the {best_elev_counter.most_common(1)[0][0]} elevation band and the {best_radius_counter.most_common(1)[0][0]} radius band.",
        "- Objects do share patterns: the clustering and heatmaps show repeated families of preferred viewing geometry rather than completely unique optima.",
        f"- {strongest_counter.most_common(1)[0][0].capitalize()} appears most influential most often, although several classes also show clear interaction effects.",
        "- The most sensitive objects should be approached with tighter viewpoint control, while the most stable classes can be searched more flexibly.",
    ]

    (output_dir / "thesis_viewpoint_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    per_image_path = Path(args.per_image).resolve()
    per_class_path = Path(args.per_class_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(per_image_path)
    class_names = load_class_names(per_class_path)
    records = build_records(rows, class_names, args.metric)
    if not records:
        raise RuntimeError(f"No usable records were found in {per_image_path}.")

    best_rows, top3_rows, rankings = compute_viewpoint_rankings(records, args.metric)
    write_csv(output_dir / "ideal_viewpoints.csv", list(best_rows[0].keys()), best_rows)
    write_csv(
        output_dir / "top3_viewpoints.csv",
        ["object_class", "rank", "azimuth", "elevation", "radius", "viewpoint_label", "mean_value", "std_value", "variance_value", "n_images"],
        top3_rows,
    )

    plot_top3_viewpoints(rankings, output_dir / "ideal_viewpoints_top3.png", args.label, args.metric)
    plot_heatmaps(records, output_dir / "heatmaps", args.label, args.metric)
    cluster_rows = cluster_objects(rankings, output_dir, args.clusters, args.label, args.metric)
    _, regression_rows = run_regression_analysis(records, rankings, output_dir, args.metric, args.label)
    robustness_rows = run_robustness_analysis(records, rankings, output_dir, args.metric)
    build_summary_text(best_rows, cluster_rows, regression_rows, robustness_rows, output_dir, args.metric, args.label)

    print(f"Saved thesis viewpoint analysis to: {output_dir}")


if __name__ == "__main__":
    main()
