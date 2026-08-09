from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
    }
)


RESULTS_DIR = Path("factor_level_viewpoint_analysis") / "results_s0_m4"
OUTPUT_DIR = RESULTS_DIR / "main_section_visuals"
AZIMUTH_ORDER = ["000", "045", "090", "135", "180", "225", "270", "315"]
ELEVATION_ORDER = ["low", "mid", "high"]
RADIUS_ORDER = ["near", "mid", "far"]
FACTOR_COLORS = {
    "azimuth": "#2563eb",
    "elevation": "#ea580c",
    "radius": "#16a34a",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def load_factor_summary(path: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "object_class": row["object_class"],
                "best_azimuth": row["best_azimuth"],
                "best_azimuth_mean": to_float(row["best_azimuth_mean"]),
                "best_azimuth_n": int(float(row["best_azimuth_n"])),
                "best_elevation": row["best_elevation"],
                "best_elevation_mean": to_float(row["best_elevation_mean"]),
                "best_elevation_n": int(float(row["best_elevation_n"])),
                "best_radius": row["best_radius"],
                "best_radius_mean": to_float(row["best_radius_mean"]),
                "best_radius_n": int(float(row["best_radius_n"])),
                "strongest_factor": row["strongest_factor"],
                "azimuth_effect_size": to_float(row["azimuth_effect_size"]),
                "elevation_effect_size": to_float(row["elevation_effect_size"]),
                "radius_effect_size": to_float(row["radius_effect_size"]),
            }
        )
    return parsed


def load_stats(path: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "object_class": row["object_class"],
                "factor": row["factor"],
                "level": row["level"],
                "mean_value": to_float(row["mean_value"]),
                "std_value": to_float(row["std_value"]),
                "variance_value": to_float(row["variance_value"]),
                "n_images": int(float(row["n_images"])),
            }
        )
    return parsed


def plot_best_levels_table(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    rows = sorted(summary_rows, key=lambda row: str(row["object_class"]))
    headers = [
        "Object",
        "Best azimuth",
        "Best elevation",
        "Best radius",
        "Strongest factor",
    ]
    cell_text = [
        [
            str(row["object_class"]),
            f"{row['best_azimuth']} ({row['best_azimuth_mean']:.2f})",
            f"{row['best_elevation']} ({row['best_elevation_mean']:.2f})",
            f"{row['best_radius']} ({row['best_radius_mean']:.2f})",
            str(row["strongest_factor"]),
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.5)

    header_color = "#16324f"
    for col_idx in range(len(headers)):
        table[(0, col_idx)].set_facecolor(header_color)
        table[(0, col_idx)].set_text_props(color="white", weight="bold")

    strongest_colors = {
        "azimuth": "#d9edf7",
        "elevation": "#fde2b3",
        "radius": "#d9f2d9",
    }
    for row_idx, row in enumerate(rows, start=1):
        table[(row_idx, 4)].set_facecolor(strongest_colors.get(str(row["strongest_factor"]), "white"))

    ax.set_title("Best viewpoint factor levels per object", fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_effect_heatmap(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    rows = sorted(summary_rows, key=lambda row: str(row["object_class"]))
    labels = [str(row["object_class"]) for row in rows]
    matrix = np.array(
        [
            [
                float(row["azimuth_effect_size"]),
                float(row["elevation_effect_size"]),
                float(row["radius_effect_size"]),
            ]
            for row in rows
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Azimuth", "Elevation", "Radius"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("How strongly each factor changes performance")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Best-minus-worst mean AP50-95")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def prepare_factor_series(stats_rows: list[dict[str, object]], factor: str, level_order: list[str]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in stats_rows:
        if row["factor"] != factor:
            continue
        grouped[str(row["object_class"])][str(row["level"])] = float(row["mean_value"])

    classes = sorted(grouped)
    matrix = np.array(
        [[grouped[object_class].get(level, np.nan) for level in level_order] for object_class in classes],
        dtype=float,
    )
    return classes, matrix


def save_publication_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)


def sort_rows_by_elevation_effect(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        summary_rows,
        key=lambda row: (-float(row["elevation_effect_size"]), str(row["object_class"])),
    )


def plot_ranked_factor_effects(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    rows = sort_rows_by_elevation_effect(summary_rows)
    labels = [str(row["object_class"]) for row in rows]
    azimuth = np.array([float(row["azimuth_effect_size"]) for row in rows], dtype=float)
    elevation = np.array([float(row["elevation_effect_size"]) for row in rows], dtype=float)
    radius = np.array([float(row["radius_effect_size"]) for row in rows], dtype=float)

    y = np.arange(len(rows), dtype=float)
    row_min = np.minimum(np.minimum(azimuth, elevation), radius)
    row_max = np.maximum(np.maximum(azimuth, elevation), radius)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hlines(y, row_min, row_max, color="#cbd5e1", linewidth=1.0, zorder=1)
    ax.scatter(azimuth, y, s=34, color=FACTOR_COLORS["azimuth"], label="Azimuth", zorder=3)
    ax.scatter(elevation, y, s=44, color=FACTOR_COLORS["elevation"], marker="s", label="Elevation", zorder=4)
    ax.scatter(radius, y, s=34, color=FACTOR_COLORS["radius"], marker="D", label="Radius", zorder=3)

    for idx, value in enumerate(elevation):
        ax.text(value + 0.006, y[idx], f"{value:.2f}", va="center", ha="left", fontsize=8.5, color="#7c2d12", weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(float(np.nanmax(elevation)), float(np.nanmax(azimuth)), float(np.nanmax(radius))) + 0.08)
    ax.set_xlabel("Best-minus-worst mean AP50-95")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.legend(loc="lower right", frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_publication_figure(fig, output_path)
    plt.close(fig)


def plot_elevation_story(stats_rows: list[dict[str, object]], output_path: Path) -> None:
    classes, matrix = prepare_factor_series(stats_rows, "elevation", ELEVATION_ORDER)
    gains = matrix[:, -1] - matrix[:, 0]
    order = np.argsort(gains)[::-1]
    ordered_classes = [classes[idx] for idx in order]
    ordered_matrix = matrix[order]
    mean_profile = np.nanmean(ordered_matrix, axis=0)
    q25 = np.nanpercentile(ordered_matrix, 25, axis=0)
    q75 = np.nanpercentile(ordered_matrix, 75, axis=0)
    x = np.arange(len(ELEVATION_ORDER), dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    for row in ordered_matrix:
        ax.plot(x, row, color="#94a3b8", linewidth=1.1, marker="o", markersize=3.5, alpha=0.75, zorder=2)

    ax.fill_between(x, q25, q75, color="#fed7aa", alpha=0.7, zorder=1)
    ax.plot(
        x,
        mean_profile,
        color=FACTOR_COLORS["elevation"],
        linewidth=2.6,
        marker="o",
        markersize=6.5,
        zorder=4,
    )

    for xi, value in zip(x, mean_profile, strict=True):
        ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, color="#9a3412")

    end_values = ordered_matrix[:, -1].copy()
    sorted_idx = np.argsort(end_values)
    label_positions = end_values.copy()
    min_gap = 0.010
    max_label_y = 0.935
    min_label_y = 0.80
    for pos in range(1, len(sorted_idx)):
        prev_idx = sorted_idx[pos - 1]
        idx = sorted_idx[pos]
        label_positions[idx] = max(label_positions[idx], label_positions[prev_idx] + min_gap)
    overflow = label_positions[sorted_idx[-1]] - max_label_y
    if overflow > 0:
        label_positions -= overflow
    label_positions[sorted_idx[0]] = max(label_positions[sorted_idx[0]], min_label_y)
    for pos in range(1, len(sorted_idx)):
        prev_idx = sorted_idx[pos - 1]
        idx = sorted_idx[pos]
        label_positions[idx] = max(label_positions[idx], label_positions[prev_idx] + min_gap)

    for object_class, value_y, label_y in zip(ordered_classes, end_values, label_positions, strict=True):
        ax.plot([x[-1] + 0.01, x[-1] + 0.10], [value_y, label_y], color="#94a3b8", linewidth=0.75, alpha=0.9, clip_on=False)
        ax.text(
            x[-1] + 0.12,
            label_y,
            object_class,
            ha="left",
            va="center",
            fontsize=7.4,
            color="#475569",
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Low\n(+2 m)", "Mid\n(+12 m)", "High\n(+22 m)"])
    ax.set_xlabel("Elevation band")
    ax.set_ylabel("Mean AP50-95")
    ax.set_ylim(0.48, 0.96)
    ax.set_xlim(-0.10, 2.58)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], color="#94a3b8", linewidth=1.1, marker="o", markersize=3.5, label="Object-specific profiles"),
        Line2D([0], [0], color=FACTOR_COLORS["elevation"], linewidth=2.6, marker="o", markersize=5.5, label="Object-balanced mean"),
        Patch(facecolor="#fed7aa", edgecolor="none", alpha=0.7, label="Interquartile range"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    fig.tight_layout()
    save_publication_figure(fig, output_path)
    plt.close(fig)


def plot_azimuth_story(summary_rows: list[dict[str, object]], stats_rows: list[dict[str, object]], output_path: Path) -> None:
    rows = sorted(
        summary_rows,
        key=lambda row: (AZIMUTH_ORDER.index(str(row["best_azimuth"])), -float(row["azimuth_effect_size"]), str(row["object_class"])),
    )
    lookup: dict[str, dict[str, float]] = defaultdict(dict)
    for row in stats_rows:
        if row["factor"] != "azimuth":
            continue
        lookup[str(row["object_class"])][str(row["level"])] = float(row["mean_value"])

    matrix = np.array(
        [[lookup[str(row["object_class"])].get(level, np.nan) for level in AZIMUTH_ORDER] for row in rows],
        dtype=float,
    )
    labels = [str(row["object_class"]) for row in rows]
    deviations = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    vmax = float(np.nanmax(np.abs(deviations))) + 0.005

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(deviations, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(AZIMUTH_ORDER)))
    ax.set_xticklabels(AZIMUTH_ORDER)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Object class")
    ax.set_xticks(np.arange(-0.5, len(AZIMUTH_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx, row in enumerate(rows):
        best_idx = AZIMUTH_ORDER.index(str(row["best_azimuth"]))
        ax.add_patch(Rectangle((best_idx - 0.5, row_idx - 0.5), 1.0, 1.0, fill=False, edgecolor="white", linewidth=2.2))
        ax.text(
            best_idx,
            row_idx,
            f"{matrix[row_idx, best_idx]:.2f}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="white" if abs(deviations[row_idx, best_idx]) > 0.05 else "black",
            weight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Deviation from class mean AP50-95")
    fig.tight_layout()
    save_publication_figure(fig, output_path)
    plt.close(fig)


def plot_global_trends(
    azimuth_rows: list[dict[str, object]],
    elevation_rows: list[dict[str, object]],
    radius_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    configs = [
        ("Azimuth", azimuth_rows, AZIMUTH_ORDER, "#1f77b4"),
        ("Elevation", elevation_rows, ELEVATION_ORDER, "#ff7f0e"),
        ("Radius", radius_rows, RADIUS_ORDER, "#2ca02c"),
    ]

    for ax, (title, rows, order, color) in zip(axes, configs, strict=True):
        _, matrix = prepare_factor_series(rows, str(rows[0]["factor"]), order)
        means = np.nanmean(matrix, axis=0)
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)

        x = np.arange(len(order))
        ax.plot(x, means, marker="o", color=color, linewidth=2)
        ax.fill_between(x, mins, maxs, color=color, alpha=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylim(0.45, 0.98)
        ax.set_title(title)
        ax.set_ylabel("Object-balanced mean AP50-95")
        ax.grid(alpha=0.25)

    fig.suptitle("Overall viewpoint trends across objects", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overall_factor_profiles(
    azimuth_rows: list[dict[str, object]],
    elevation_rows: list[dict[str, object]],
    radius_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    configs = [
        ("azimuth", "Azimuth", azimuth_rows, AZIMUTH_ORDER, FACTOR_COLORS["azimuth"]),
        ("elevation", "Elevation", elevation_rows, ELEVATION_ORDER, FACTOR_COLORS["elevation"]),
        ("radius", "Radius", radius_rows, RADIUS_ORDER, FACTOR_COLORS["radius"]),
    ]

    matrices: list[np.ndarray] = []
    mean_series: list[np.ndarray] = []
    q25_series: list[np.ndarray] = []
    q75_series: list[np.ndarray] = []
    for factor_name, _, rows, order, _ in configs:
        _, matrix = prepare_factor_series(rows, factor_name, order)
        matrices.append(matrix)
        mean_series.append(np.nanmean(matrix, axis=0))
        q25_series.append(np.nanpercentile(matrix, 25, axis=0))
        q75_series.append(np.nanpercentile(matrix, 75, axis=0))

    y_min = min(float(np.nanmin(q25)) for q25 in q25_series) - 0.03
    y_max = max(float(np.nanmax(q75)) for q75 in q75_series) + 0.03

    for ax, (factor_name, title, _, order, color), means, q25, q75 in zip(
        axes,
        configs,
        mean_series,
        q25_series,
        q75_series,
        strict=True,
    ):
        x = np.arange(len(order), dtype=float)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, means, color=color, linewidth=2.0, marker="o", markersize=4.5)

        best_idx = int(np.nanargmax(means))
        ax.scatter([x[best_idx]], [means[best_idx]], color=color, s=42, zorder=4)
        ax.text(
            x[best_idx],
            means[best_idx] + 0.013,
            f"{means[best_idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=color,
        )

        if factor_name == "azimuth":
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=45, ha="right")
        elif factor_name == "elevation":
            ax.set_xticks(x)
            ax.set_xticklabels(["Low", "Mid", "High"])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(["Near", "Mid", "Far"])

        ax.set_title(title, pad=4)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Object-balanced mean AP50-95")
    fig.tight_layout(w_pad=1.0)
    save_publication_figure(fig, output_path)
    plt.close(fig)


def plot_dashboard(
    summary_rows: list[dict[str, object]],
    azimuth_rows: list[dict[str, object]],
    elevation_rows: list[dict[str, object]],
    radius_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    rows = sorted(summary_rows, key=lambda row: str(row["object_class"]))
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.1, 1.0])

    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    headers = ["Object", "Best azimuth", "Best elevation", "Best radius", "Strongest"]
    cell_text = [
        [
            str(row["object_class"]),
            str(row["best_azimuth"]),
            str(row["best_elevation"]),
            str(row["best_radius"]),
            str(row["strongest_factor"]),
        ]
        for row in rows
    ]
    table = ax_table.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for col_idx in range(len(headers)):
        table[(0, col_idx)].set_facecolor("#16324f")
        table[(0, col_idx)].set_text_props(color="white", weight="bold")
    ax_table.set_title("Main viewpoint story at a glance", fontsize=16, pad=10)

    ax_heatmap = fig.add_subplot(gs[1, 0])
    matrix = np.array(
        [
            [
                float(row["azimuth_effect_size"]),
                float(row["elevation_effect_size"]),
                float(row["radius_effect_size"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    im = ax_heatmap.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax_heatmap.set_xticks(range(3))
    ax_heatmap.set_xticklabels(["Azimuth", "Elevation", "Radius"])
    ax_heatmap.set_yticks(range(len(rows)))
    ax_heatmap.set_yticklabels([str(row["object_class"]) for row in rows])
    ax_heatmap.set_title("Effect size by factor")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax_heatmap.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Best-minus-worst mean AP50-95")

    ax_trends = fig.add_subplot(gs[1, 1])
    x_positions: list[float] = []
    x_labels: list[str] = []
    current_x = 0.0
    trend_specs = [
        ("azimuth", azimuth_rows, AZIMUTH_ORDER, "#1f77b4"),
        ("elevation", elevation_rows, ELEVATION_ORDER, "#ff7f0e"),
        ("radius", radius_rows, RADIUS_ORDER, "#2ca02c"),
    ]
    for factor_name, rows_for_factor, order, color in trend_specs:
        _, factor_matrix = prepare_factor_series(rows_for_factor, factor_name, order)
        means = np.nanmean(factor_matrix, axis=0)
        xs = np.arange(len(order), dtype=float) + current_x
        ax_trends.plot(xs, means, marker="o", linewidth=2.2, color=color, label=factor_name.capitalize())
        x_positions.extend(xs.tolist())
        x_labels.extend(order)
        current_x = xs[-1] + 1.6
    ax_trends.set_xticks(x_positions)
    ax_trends.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_trends.set_ylim(0.45, 0.98)
    ax_trends.set_ylabel("Object-balanced mean AP50-95")
    ax_trends.set_title("Overall trends across objects")
    ax_trends.grid(alpha=0.25)
    ax_trends.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    strongest_counts = Counter(str(row["strongest_factor"]) for row in summary_rows)
    best_elev_counts = Counter(str(row["best_elevation"]) for row in summary_rows)
    best_radius_counts = Counter(str(row["best_radius"]) for row in summary_rows)
    best_az_counts = Counter(str(row["best_azimuth"]) for row in summary_rows)

    sorted_by_elev = sorted(summary_rows, key=lambda row: float(row["elevation_effect_size"]), reverse=True)
    strongest_objects = ", ".join(str(row["object_class"]) for row in sorted_by_elev[:3])

    text = "\n".join(
        [
            "# Main-section viewpoint summary",
            "",
            "This folder is a readability layer on top of the factor-level analysis.",
            "",
            "Key patterns:",
            f"- Strongest factor counts: {dict(strongest_counts)}",
            f"- Best elevation counts: {dict(best_elev_counts)}",
            f"- Best radius counts: {dict(best_radius_counts)}",
            f"- Best azimuth counts: {dict(best_az_counts)}",
            f"- Objects with the strongest elevation dependence: {strongest_objects}",
            "",
            "Suggested thesis story:",
            "- Elevation is the dominant viewpoint factor.",
            "- High elevation is preferred for every object class in this dataset.",
            "- Radius matters less than elevation, and azimuth matters least overall.",
            "- Near and mid range are usually preferable to far range.",
            "- Use the heatmap and dashboard here for a compact main-text presentation.",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    factor_summary = load_factor_summary(RESULTS_DIR / "factor_best_summary.csv")
    azimuth_rows = load_stats(RESULTS_DIR / "azimuth_stats.csv")
    elevation_rows = load_stats(RESULTS_DIR / "elevation_stats.csv")
    radius_rows = load_stats(RESULTS_DIR / "radius_stats.csv")

    plot_best_levels_table(factor_summary, OUTPUT_DIR / "best_levels_table.png")
    plot_effect_heatmap(factor_summary, OUTPUT_DIR / "effect_size_heatmap.png")
    plot_global_trends(azimuth_rows, elevation_rows, radius_rows, OUTPUT_DIR / "global_factor_trends.png")
    plot_overall_factor_profiles(
        azimuth_rows,
        elevation_rows,
        radius_rows,
        OUTPUT_DIR / "overall_factor_profiles.png",
    )
    plot_ranked_factor_effects(factor_summary, OUTPUT_DIR / "factor_effect_ranking.png")
    plot_elevation_story(elevation_rows, OUTPUT_DIR / "elevation_story.png")
    plot_azimuth_story(factor_summary, azimuth_rows, OUTPUT_DIR / "azimuth_story.png")
    plot_dashboard(
        factor_summary,
        azimuth_rows,
        elevation_rows,
        radius_rows,
        OUTPUT_DIR / "viewpoint_main_dashboard.png",
    )
    write_summary(factor_summary, OUTPUT_DIR / "viewpoint_main_summary.md")
    print(f"Saved main-section viewpoint visuals to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
