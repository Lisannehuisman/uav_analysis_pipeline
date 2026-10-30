from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = WORKSPACE / "m4_marginal_viewpoint_value_analysis" / "outputs" / "ring_shapley_noisy_or_best_iou"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"

ELEVATION_ORDER = {"low": 0, "mid": 1, "high": 2}
RADIUS_ORDER = {"near": 0, "mid": 1, "far": 2}
CLASS_ORDER = [
    "barrel",
    "container",
    "male",
    "rock",
    "suv",
    "tank",
    "tent",
    "tower",
    "tree",
    "whitevan",
]
AZIMUTHS = list(range(0, 360, 45))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create actual-data per-class Shapley plots from the Noisy-OR + best IoU ring analysis."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--focus-ring",
        default="elmid-radnear",
        help="Ring to show in the detailed per-class azimuth plot.",
    )
    return parser.parse_args()


def ring_sort_key(ring_id: str) -> tuple[int, int]:
    elevation_token, radius_token = ring_id.split("-")
    elevation = elevation_token.replace("el", "")
    radius = radius_token.replace("rad", "")
    return (ELEVATION_ORDER.get(elevation, 99), RADIUS_ORDER.get(radius, 99))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_heatmap_top_shapley(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    rings = sorted({row["ring_id"] for row in summary_rows}, key=ring_sort_key)
    classes = [name for name in CLASS_ORDER if any(row["target_class"] == name for row in summary_rows)]
    value_grid = np.full((len(classes), len(rings)), np.nan, dtype=float)
    annotation_grid: list[list[str]] = [["" for _ in rings] for _ in classes]

    class_index = {name: idx for idx, name in enumerate(classes)}
    ring_index = {name: idx for idx, name in enumerate(rings)}

    for row in summary_rows:
        c_idx = class_index[row["target_class"]]
        r_idx = ring_index[row["ring_id"]]
        top_az = int(float(row["top_azimuth"]))
        value = float(row["top_shapley_value"])
        value_grid[c_idx, r_idx] = value
        annotation_grid[c_idx][r_idx] = f"az{top_az:03d}\n{value:.3f}"

    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = plt.cm.YlGnBu
    im = ax.imshow(value_grid, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(rings)))
    ax.set_xticklabels(rings, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_title("Top Shapley score per object class and controlled ring", fontsize=16, pad=16)

    for c_idx in range(len(classes)):
        for r_idx in range(len(rings)):
            label = annotation_grid[c_idx][r_idx]
            if not label:
                continue
            value = value_grid[c_idx, r_idx]
            text_color = "white" if value >= np.nanmean(value_grid) else "#1f2933"
            ax.text(r_idx, c_idx, label, ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Top Shapley value", rotation=90)
    fig.text(
        0.012,
        0.012,
        "Each cell shows the best azimuth within that ring for that object class, plus its exact Shapley score.",
        fontsize=10,
        color="#5b6673",
    )
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_best_ring_per_class(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    best_rows: list[dict[str, str]] = []
    for target_class in CLASS_ORDER:
        class_rows = [row for row in summary_rows if row["target_class"] == target_class]
        if not class_rows:
            continue
        best_rows.append(max(class_rows, key=lambda row: float(row["top_shapley_value"])))

    best_rows.sort(key=lambda row: float(row["top_shapley_value"]))

    labels = [row["target_class"] for row in best_rows]
    values = [float(row["top_shapley_value"]) for row in best_rows]
    notes = [f"{row['ring_id']} | az{int(float(row['top_azimuth'])):03d}" for row in best_rows]

    fig, ax = plt.subplots(figsize=(13, 8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#2f6f9f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Best exact Shapley value across all controlled rings")
    ax.set_title("Best ring and azimuth per object class", fontsize=16, pad=14)
    ax.grid(axis="x", alpha=0.25)

    for bar, value, note in zip(bars, values, notes):
        ax.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{note} | {value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )

    fig.text(
        0.012,
        0.012,
        "This plot answers: for each object class, where is the strongest collaborative viewpoint configuration overall?",
        fontsize=10,
        color="#5b6673",
    )
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_focus_ring_facets(detail_rows: list[dict[str, str]], focus_ring: str, output_path: Path) -> None:
    ring_rows = [row for row in detail_rows if row["ring_id"] == focus_ring]
    classes = [name for name in CLASS_ORDER if any(row["target_class"] == name for row in ring_rows)]
    if not classes:
        raise ValueError(f"No rows found for focus ring {focus_ring}")

    fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, target_class in zip(axes, classes):
        class_rows = [row for row in ring_rows if row["target_class"] == target_class]
        class_rows.sort(key=lambda row: int(float(row["azimuth"])))
        azimuths = [int(float(row["azimuth"])) for row in class_rows]
        shapley_values = [float(row["shapley_value"]) for row in class_rows]
        top_row = max(class_rows, key=lambda row: float(row["shapley_value"]))
        top_az = int(float(top_row["azimuth"]))

        colors = ["#d97706" if az == top_az else "#94a9c2" for az in azimuths]
        ax.bar(azimuths, shapley_values, width=30, color=colors)
        ax.plot(azimuths, shapley_values, color="#335c81", linewidth=1.5, alpha=0.85)
        ax.set_title(
            f"{target_class} | top az{top_az:03d} | {float(top_row['shapley_value']):.3f}",
            fontsize=10.5,
        )
        ax.set_xticks(AZIMUTHS)
        ax.set_xticklabels([f"{az:03d}" for az in AZIMUTHS], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(float(row["shapley_value"]) for row in ring_rows) * 1.15)

    for ax in axes[len(classes) :]:
        ax.axis("off")

    fig.suptitle(
        f"Per-object exact Shapley scores across azimuths in ring {focus_ring}",
        fontsize=17,
        y=0.995,
    )
    fig.text(0.5, 0.015, "Azimuth", ha="center", fontsize=11)
    fig.text(0.015, 0.5, "Exact Shapley value", va="center", rotation=90, fontsize=11)
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.975))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_plot_note(path: Path, focus_ring: str) -> None:
    lines = [
        "# Per-class Shapley plots",
        "",
        "This folder contains three actual-data plots derived from the exact `Noisy-OR + best IoU` ring-based Shapley analysis.",
        "",
        "## Files",
        "",
        "- `per_class_top_shapley_heatmap.png`",
        "  - Rows are object classes, columns are controlled rings.",
        "  - Each cell shows the top azimuth and top Shapley score for that class-ring combination.",
        "- `per_class_best_ring_bar.png`",
        "  - One row per object class.",
        "  - Shows the strongest class-specific Shapley score anywhere in the ring grid, with its ring and azimuth.",
        f"- `per_class_{focus_ring.replace('-', '_')}_azimuth_facets.png`",
        f"  - Small multiples for the focused ring `{focus_ring}`.",
        "  - Shows the full azimuth-wise Shapley profile for each object class, not just the winner.",
        "",
        "## Best use",
        "",
        "- Use the heatmap for a compact supervisor or thesis overview.",
        "- Use the bar chart if you want the cleanest summary answer to `what is the best collaborative angle per object?`.",
        "- Use the focused ring facets when you want to show that pooled results hide strong object-specific differences.",
        "",
        "## Input sources",
        "",
        "- `ring_shapley_noisy_or_best_iou_by_class.csv`",
        "- `ring_shapley_noisy_or_best_iou_by_class_summary.csv`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_csv_rows(input_dir / "ring_shapley_noisy_or_best_iou_by_class_summary.csv")
    detail_rows = load_csv_rows(input_dir / "ring_shapley_noisy_or_best_iou_by_class.csv")

    save_heatmap_top_shapley(summary_rows, output_dir / "per_class_top_shapley_heatmap.png")
    save_best_ring_per_class(summary_rows, output_dir / "per_class_best_ring_bar.png")
    save_focus_ring_facets(
        detail_rows,
        focus_ring=str(args.focus_ring),
        output_path=output_dir / f"per_class_{str(args.focus_ring).replace('-', '_')}_azimuth_facets.png",
    )
    write_plot_note(output_dir / "README.md", focus_ring=str(args.focus_ring))


if __name__ == "__main__":
    main()
