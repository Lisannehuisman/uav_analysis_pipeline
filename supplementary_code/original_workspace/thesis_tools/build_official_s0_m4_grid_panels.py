from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_RECORDS = ROOT / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
DEFAULT_ANALYSIS_DIR = ROOT / "s0_m4" / "thesis_viewpoint_analysis_s0_m4"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thesis_tools" / "official_s0_m4_grid_panels"

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
BEST_FACE = "#f97316"
BEST_EDGE = "#b91c1c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-object grid-style panels using official S0_M4 best viewpoints."
    )
    parser.add_argument(
        "--scene-records",
        default=str(DEFAULT_SCENE_RECORDS),
        help="Scene-view records CSV with per-image ap50_95 values for the fixed YOLOv8l_M4 model.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Folder containing official S0_M4 ideal_viewpoints.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where overview and per-object grid panels will be saved.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_stats(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str, str], dict[str, float]]]:
    grouped_values: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        class_name = row["target_class"].strip()
        azimuth = row["azimuth"].strip().zfill(3)
        elevation = row["elevation"].strip().lower()
        radius = row["radius"].strip().lower()
        if azimuth not in AZIMUTH_ORDER or elevation not in ELEVATION_ORDER or radius not in RADIUS_ORDER:
            continue
        metric_value = row.get("ap50_95")
        if metric_value in (None, ""):
            continue
        grouped_values[class_name][(azimuth, elevation, radius)].append(float(metric_value))

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
                            "count": float(len(values)),
                        }
                    else:
                        stats[class_name][key] = {"mean": float("nan"), "count": 0.0}
    return stats


def load_official_best(path: Path) -> dict[str, tuple[str, str, str, float, int]]:
    best_map: dict[str, tuple[str, str, str, float, int]] = {}
    for row in read_csv_rows(path):
        best_map[row["object_class"]] = (
            row["best_azimuth"].zfill(3),
            row["best_elevation"].lower(),
            row["best_radius"].lower(),
            float(row["mean_ap50_95"]),
            int(row["n_images"]),
        )
    return best_map


def draw_reference_rings(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 240)
    for elevation in ELEVATION_ORDER:
        z = ELEVATION_SCALE[elevation]
        for radius in RADIUS_ORDER:
            radius_scale = RADIUS_SCALE[radius]
            x = radius_scale * np.cos(theta)
            y = radius_scale * np.sin(theta)
            ax.plot(x, y, np.full_like(theta, z), color=RING_COLOR, linewidth=1.1, zorder=0)

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


def viewpoint_label(key: tuple[str, str, str]) -> str:
    return f"az {key[0]} | {ELEVATION_LABELS[key[1]]} | {key[2]}"


def plot_panel(
    ax: plt.Axes,
    class_name: str,
    class_stats: dict[tuple[str, str, str], dict[str, float]],
    best_key: tuple[str, str, str],
    best_mean: float,
    best_support: int,
    show_caption: bool = True,
) -> None:
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.cividis
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
                xs.append(RADIUS_SCALE[radius] * math.cos(theta))
                ys.append(RADIUS_SCALE[radius] * math.sin(theta))
                zs.append(ELEVATION_SCALE[elevation])
                if math.isnan(info["mean"]):
                    colors.append((0.88, 0.90, 0.93, 0.9))
                else:
                    colors.append(cmap(color_norm(info["mean"])))
                sizes.append(34 + 18 * float(info["count"]))

    ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolor="#0f172a", linewidth=0.35, alpha=0.96, depthshade=False)

    theta = math.radians(int(best_key[0]))
    x = RADIUS_SCALE[best_key[2]] * math.cos(theta)
    y = RADIUS_SCALE[best_key[2]] * math.sin(theta)
    z = ELEVATION_SCALE[best_key[1]]
    ax.scatter(
        [x],
        [y],
        [z],
        marker="*",
        s=300,
        c=[BEST_FACE],
        edgecolors=BEST_EDGE,
        linewidth=1.6,
        depthshade=False,
        zorder=8,
    )

    ax.set_title(class_name, fontsize=14, fontweight="bold", pad=10)
    if show_caption:
        ax.text2D(
            0.5,
            -0.08,
            f"Official best: {viewpoint_label(best_key)}  |  mean {best_mean:.3f}  |  n = {best_support}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color="#475569",
        )


def build_overview(
    stats_by_class: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    official_best: dict[str, tuple[str, str, str, float, int]],
    output_path: Path,
) -> None:
    classes = sorted(stats_by_class)
    fig = plt.figure(figsize=(18, 23), facecolor=FIGURE_BACKGROUND)

    for idx, class_name in enumerate(classes, start=1):
        ax = fig.add_subplot(5, 2, idx, projection="3d")
        best = official_best[class_name]
        plot_panel(
            ax=ax,
            class_name=class_name,
            class_stats=stats_by_class[class_name],
            best_key=(best[0], best[1], best[2]),
            best_mean=best[3],
            best_support=best[4],
        )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=plt.cm.cividis),
        ax=fig.axes,
        shrink=0.76,
        pad=0.03,
        fraction=0.04,
    )
    cbar.set_label("AP50-95", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "All 72 viewpoints per object for S0_M4",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.953,
        "Grid-style overview using fixed YOLOv8l_M4 image-level AP50-95, with official S0_M4 best viewpoints highlighted",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    fig.subplots_adjust(left=0.05, right=0.86, bottom=0.04, top=0.90, hspace=0.24, wspace=0.06)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_per_object_panels(
    stats_by_class: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    official_best: dict[str, tuple[str, str, str, float, int]],
    output_dir: Path,
) -> None:
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    for class_name in sorted(stats_by_class):
        fig = plt.figure(figsize=(9.4, 7.4), facecolor=FIGURE_BACKGROUND)
        ax = fig.add_subplot(111, projection="3d")
        best = official_best[class_name]
        plot_panel(
            ax=ax,
            class_name=class_name,
            class_stats=stats_by_class[class_name],
            best_key=(best[0], best[1], best[2]),
            best_mean=best[3],
            best_support=best[4],
        )
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.10, top=0.88)
        fig.savefig(object_dir / f"{class_name}_official_s0_m4_grid.png", dpi=300)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    scene_records = Path(args.scene_records).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_by_class = collect_stats(read_csv_rows(scene_records))
    official_best = load_official_best(analysis_dir / "ideal_viewpoints.csv")

    build_overview(stats_by_class, official_best, output_dir / "all_objects_official_s0_m4_grid.png")
    build_per_object_panels(stats_by_class, official_best, output_dir)
    print(f"Saved official S0_M4 grid panels to: {output_dir}")


if __name__ == "__main__":
    main()
