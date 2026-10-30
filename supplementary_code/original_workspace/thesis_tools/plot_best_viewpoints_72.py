from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}
AZIMUTH_ORDER = ["000", "045", "090", "135", "180", "225", "270", "315"]
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]
RADIUS_SCALE = {"near": 1.0, "mid": 2.0, "far": 3.0}
ELEVATION_SCALE = {"low": 0.0, "mid": 1.0, "high": 2.0}
ELEVATION_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}

FIGURE_BACKGROUND = "#f8fafc"
PANEL_BACKGROUND = "#ffffff"
PANE_COLOR = (0.97, 0.98, 0.99, 1.0)
GRID_COLOR = (0.56, 0.60, 0.69, 0.26)
RING_COLOR = (0.35, 0.41, 0.55, 0.28)
RADIAL_COLOR = (0.35, 0.41, 0.55, 0.16)
BEST_MEAN_FACE = "#f97316"
BEST_MEAN_EDGE = "#b91c1c"
BEST_SINGLE_FACE = "#ffffff"
BEST_SINGLE_EDGE = "#0f172a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all 72 viewpoints for each object and highlight the best viewpoint."
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
        help="Per-image metric to analyze.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label shown in the plot title.",
    )
    parser.add_argument(
        "--min-samples",
        default=3,
        type=int,
        help="Minimum support required before a viewpoint can be selected as the best-by-mean cell.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/thesis_tools/best_viewpoints_72",
        help="Folder for the 72-viewpoint plots and CSV summaries.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_present(row: dict[str, str], candidates: list[str]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def resolve_metric_value(row: dict[str, str], metric: str) -> float | None:
    value = first_present(row, [metric, f"target_{metric}"])
    if value is None:
        return None
    return float(value)


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


def infer_class_names(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        class_name = first_present(row, ["class_name", "target_class"])
        if class_name is None:
            continue
        clean_name = class_name.strip()
        if clean_name and clean_name not in seen:
            seen.append(clean_name)
    return seen


def parse_viewpoint_tokens(
    image_path: str,
    class_names: list[str],
    fallback_class_name: str | None = None,
) -> dict[str, str] | None:
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
        if fallback_class_name:
            class_name = fallback_class_name.strip()
        else:
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


def collect_stats(
    rows: list[dict[str, str]],
    class_names: list[str],
    metric: str,
) -> dict[str, dict[tuple[str, str, str], dict[str, float]]]:
    grouped_values: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        image_path = first_present(row, ["image", "file_name"])
        if image_path is None:
            continue
        fallback_class_name = first_present(row, ["class_name", "target_class"])
        tokens = parse_viewpoint_tokens(image_path, class_names, fallback_class_name=fallback_class_name)
        if tokens is None:
            continue
        metric_value = resolve_metric_value(row, metric)
        if metric_value is None:
            continue
        key = (tokens["azimuth"], tokens["elevation"], tokens["radius"])
        grouped_values[tokens["class_name"]][key].append(metric_value)

    stats: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}
    for class_name, class_rows in grouped_values.items():
        stats[class_name] = {}
        for azimuth in AZIMUTH_ORDER:
            for elevation in ELEVATION_ORDER:
                for radius in RADIUS_ORDER:
                    key = (azimuth, elevation, radius)
                    values = class_rows.get(key, [])
                    if values:
                        stats[class_name][key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(np.asarray(values), ddof=0)) if len(values) > 1 else 0.0,
                            "max": float(np.max(values)),
                            "count": float(len(values)),
                        }
                    else:
                        stats[class_name][key] = {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "count": 0.0}
    return stats


def choose_best_mean(stats: dict[tuple[str, str, str], dict[str, float]], min_samples: int) -> tuple[str, str, str]:
    supported = [
        (key, values)
        for key, values in stats.items()
        if int(values["count"]) >= min_samples and not math.isnan(values["mean"])
    ]
    if not supported:
        supported = [
            (key, values)
            for key, values in stats.items()
            if int(values["count"]) > 0 and not math.isnan(values["mean"])
        ]
    supported.sort(key=lambda item: (item[1]["mean"], item[1]["count"], -item[1]["std"]), reverse=True)
    return supported[0][0]


def choose_best_single(stats: dict[tuple[str, str, str], dict[str, float]]) -> tuple[str, str, str]:
    available = [(key, values) for key, values in stats.items() if int(values["count"]) > 0 and not math.isnan(values["max"])]
    available.sort(key=lambda item: (item[1]["max"], item[1]["count"]), reverse=True)
    return available[0][0]


def draw_reference_rings(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 240)
    for elevation in ELEVATION_ORDER:
        z = ELEVATION_SCALE[elevation]
        for radius in RADIUS_ORDER:
            radius_scale = RADIUS_SCALE[radius]
            x = radius_scale * np.cos(theta)
            y = radius_scale * np.sin(theta)
            z_arr = np.full_like(theta, z, dtype=float)
            ax.plot(x, y, z_arr, color=RING_COLOR, linewidth=1.1, zorder=0)

        for azimuth in AZIMUTH_ORDER:
            angle = math.radians(int(azimuth))
            x_vals = [RADIUS_SCALE["near"] * math.cos(angle), RADIUS_SCALE["far"] * math.cos(angle)]
            y_vals = [RADIUS_SCALE["near"] * math.sin(angle), RADIUS_SCALE["far"] * math.sin(angle)]
            z_vals = [z, z]
            ax.plot(x_vals, y_vals, z_vals, color=RADIAL_COLOR, linewidth=0.7, zorder=0)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1.0, 1.0, 0.78))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 1, 2])
    ax.set_zticklabels([ELEVATION_LABELS[key] for key in ELEVATION_ORDER], fontsize=10)
    ax.set_xlim(-3.35, 3.35)
    ax.set_ylim(-3.35, 3.35)
    ax.set_zlim(-0.1, 2.15)
    ax.xaxis.pane.set_facecolor(PANE_COLOR)
    ax.yaxis.pane.set_facecolor(PANE_COLOR)
    ax.zaxis.pane.set_facecolor(PANE_COLOR)
    ax.xaxis.pane.set_edgecolor((0.85, 0.88, 0.92, 1.0))
    ax.yaxis.pane.set_edgecolor((0.85, 0.88, 0.92, 1.0))
    ax.zaxis.pane.set_edgecolor((0.85, 0.88, 0.92, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.8


def metric_display_name(metric: str) -> str:
    return "AP50-95" if metric == "ap50_95" else metric.upper()


def viewpoint_label(key: tuple[str, str, str]) -> str:
    return f"az {key[0]} | {ELEVATION_LABELS[key[1]]} | {key[2]}"


def write_summary(output_path: Path, stats_by_class: dict[str, dict[tuple[str, str, str], dict[str, float]]], min_samples: int) -> None:
    rows: list[dict[str, object]] = []
    for class_name in sorted(stats_by_class):
        class_stats = stats_by_class[class_name]
        best_mean_key = choose_best_mean(class_stats, min_samples)
        best_single_key = choose_best_single(class_stats)
        best_mean = class_stats[best_mean_key]
        best_single = class_stats[best_single_key]
        rows.append(
            {
                "object_class": class_name,
                "best_mean_azimuth": best_mean_key[0],
                "best_mean_elevation": best_mean_key[1],
                "best_mean_radius": best_mean_key[2],
                "best_mean_value": best_mean["mean"],
                "best_mean_std": best_mean["std"],
                "best_mean_n_images": int(best_mean["count"]),
                "best_single_azimuth": best_single_key[0],
                "best_single_elevation": best_single_key[1],
                "best_single_radius": best_single_key[2],
                "best_single_value": best_single["max"],
                "best_single_n_images": int(best_single["count"]),
                "selection_rule": f"Best mean among cells with n>={min_samples}, fallback to any observed cell",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    stats_by_class: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    min_samples: int,
    label: str,
    metric: str,
    output_path: Path,
) -> None:
    classes = sorted(stats_by_class)
    fig = plt.figure(figsize=(18, 23), facecolor=FIGURE_BACKGROUND)
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.cividis
    metric_label = metric_display_name(metric)

    for idx, class_name in enumerate(classes, start=1):
        ax = fig.add_subplot(5, 2, idx, projection="3d")
        class_stats = stats_by_class[class_name]
        best_mean_key = choose_best_mean(class_stats, min_samples)
        best_single_key = choose_best_single(class_stats)
        draw_reference_rings(ax)
        style_axes(ax)

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        colors: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []

        for azimuth in AZIMUTH_ORDER:
            theta = math.radians(int(azimuth))
            for elevation in ELEVATION_ORDER:
                for radius in RADIUS_ORDER:
                    key = (azimuth, elevation, radius)
                    info = class_stats[key]
                    radius_scale = RADIUS_SCALE[radius]
                    xs.append(radius_scale * math.cos(theta))
                    ys.append(radius_scale * math.sin(theta))
                    zs.append(ELEVATION_SCALE[elevation])
                    if math.isnan(info["mean"]):
                        colors.append((0.88, 0.90, 0.93, 0.9))
                    else:
                        colors.append(cmap(color_norm(info["mean"])))
                    sizes.append(34 + 18 * float(info["count"]))

        ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolor="#0f172a", linewidth=0.35, alpha=0.96, depthshade=False)

        marker_specs = [
            (best_single_key, "s", BEST_SINGLE_FACE, BEST_SINGLE_EDGE, 165, 1.8),
            (best_mean_key, "*", BEST_MEAN_FACE, BEST_MEAN_EDGE, 300, 1.6),
        ]
        if best_mean_key == best_single_key:
            marker_specs = [
                (best_single_key, "s", BEST_SINGLE_FACE, BEST_SINGLE_EDGE, 190, 1.9),
                (best_mean_key, "*", BEST_MEAN_FACE, BEST_MEAN_EDGE, 255, 1.5),
            ]

        for key, marker, face_color, edge_color, size, line_width in marker_specs:
            theta = math.radians(int(key[0]))
            x = RADIUS_SCALE[key[2]] * math.cos(theta)
            y = RADIUS_SCALE[key[2]] * math.sin(theta)
            z = ELEVATION_SCALE[key[1]]
            ax.scatter(
                [x],
                [y],
                [z],
                marker=marker,
                s=size,
                c=[face_color],
                edgecolors=edge_color,
                linewidth=line_width,
                depthshade=False,
                zorder=8,
            )

        ax.set_title(class_name, fontsize=14, fontweight="bold", pad=10)
        ax.text2D(
            0.5,
            -0.08,
            f"Mean: {viewpoint_label(best_mean_key)}  |  Single: {viewpoint_label(best_single_key)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color="#475569",
        )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=cmap),
        ax=fig.axes,
        shrink=0.76,
        pad=0.03,
        fraction=0.04,
    )
    cbar.set_label(metric_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor=BEST_MEAN_FACE,
            markeredgecolor=BEST_MEAN_EDGE,
            markeredgewidth=1.4,
            markersize=14,
            label="Best mean viewpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=BEST_SINGLE_FACE,
            markeredgecolor=BEST_SINGLE_EDGE,
            markeredgewidth=1.6,
            markersize=9,
            label="Best single-image viewpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=cmap(0.72),
            markeredgecolor="#0f172a",
            markeredgewidth=0.4,
            markersize=8,
            label="Point color = mean score, point size = support",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        f"All 72 viewpoints per object for {label}",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.953,
        "Clearer overview of the 72 sampled viewpoints for each object class",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    fig.subplots_adjust(left=0.05, right=0.86, bottom=0.04, top=0.90, hspace=0.24, wspace=0.06)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_per_object(
    stats_by_class: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    min_samples: int,
    metric: str,
    output_dir: Path,
) -> None:
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.cividis
    metric_label = metric_display_name(metric)

    for class_name in sorted(stats_by_class):
        fig = plt.figure(figsize=(9.4, 7.4), facecolor=FIGURE_BACKGROUND)
        ax = fig.add_subplot(111, projection="3d")
        class_stats = stats_by_class[class_name]
        best_mean_key = choose_best_mean(class_stats, min_samples)
        best_single_key = choose_best_single(class_stats)
        draw_reference_rings(ax)
        style_axes(ax)

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        colors: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []

        for azimuth in AZIMUTH_ORDER:
            theta = math.radians(int(azimuth))
            for elevation in ELEVATION_ORDER:
                for radius in RADIUS_ORDER:
                    key = (azimuth, elevation, radius)
                    info = class_stats[key]
                    radius_scale = RADIUS_SCALE[radius]
                    xs.append(radius_scale * math.cos(theta))
                    ys.append(radius_scale * math.sin(theta))
                    zs.append(ELEVATION_SCALE[elevation])
                    colors.append((0.88, 0.90, 0.93, 0.9) if math.isnan(info["mean"]) else cmap(color_norm(info["mean"])))
                    sizes.append(42 + 24 * float(info["count"]))

        ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolor="#0f172a", linewidth=0.4, alpha=0.97, depthshade=False)

        marker_specs = [
            (best_single_key, "s", BEST_SINGLE_FACE, BEST_SINGLE_EDGE, 200, 1.8),
            (best_mean_key, "*", BEST_MEAN_FACE, BEST_MEAN_EDGE, 340, 1.6),
        ]
        if best_mean_key == best_single_key:
            marker_specs = [
                (best_single_key, "s", BEST_SINGLE_FACE, BEST_SINGLE_EDGE, 230, 1.9),
                (best_mean_key, "*", BEST_MEAN_FACE, BEST_MEAN_EDGE, 290, 1.5),
            ]

        for key, marker, face_color, edge_color, size, line_width in marker_specs:
            theta = math.radians(int(key[0]))
            x = RADIUS_SCALE[key[2]] * math.cos(theta)
            y = RADIUS_SCALE[key[2]] * math.sin(theta)
            z = ELEVATION_SCALE[key[1]]
            ax.scatter(
                [x],
                [y],
                [z],
                marker=marker,
                s=size,
                c=[face_color],
                edgecolors=edge_color,
                linewidth=line_width,
                depthshade=False,
                zorder=8,
            )

        ax.set_title(class_name, fontsize=17, fontweight="bold", pad=12)
        ax.text2D(
            0.5,
            0.04,
            f"{metric_label} best mean: {class_stats[best_mean_key]['mean']:.3f} at {viewpoint_label(best_mean_key)}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10.5,
            color="#334155",
        )
        ax.text2D(
            0.5,
            0.00,
            f"Best single-image score: {class_stats[best_single_key]['max']:.3f} at {viewpoint_label(best_single_key)}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10.0,
            color="#475569",
        )
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.10, top=0.88)
        fig.savefig(object_dir / f"{class_name}_72_viewpoints.png", dpi=300)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(Path(args.per_image).resolve())
    class_names = load_class_names(Path(args.per_class_csv).resolve())
    if not class_names:
        class_names = infer_class_names(rows)

    stats_by_class = collect_stats(rows, class_names, args.metric)
    write_summary(output_dir / "best_viewpoints_72_summary.csv", stats_by_class, args.min_samples)
    plot_overview(
        stats_by_class,
        args.min_samples,
        args.label,
        args.metric,
        output_dir / "all_objects_best_viewpoints_72.png",
    )
    plot_per_object(stats_by_class, args.min_samples, args.metric, output_dir)
    print(f"Saved 72-viewpoint best-viewpoint plots to: {output_dir}")


if __name__ == "__main__":
    main()
