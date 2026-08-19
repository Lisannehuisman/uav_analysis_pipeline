from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_partial_pair_results import (
    OUTPUT_DIR,
    REPORTS_DIR,
    CompletedRow,
    completed_rows,
    parse_viewpoint,
    read_csv_rows,
    viewpoint_stats,
)


PLOTS_DIR = OUTPUT_DIR / "plots"


def short_viewpoint(viewpoint: str) -> str:
    elevation, radius, azimuth = parse_viewpoint(viewpoint)
    elevation_map = {"ellow": "L", "elmid": "M", "elhigh": "H"}
    radius_map = {"radnear": "N", "radmid": "M", "radfar": "F"}
    return f"{elevation_map.get(elevation, elevation)}-{radius_map.get(radius, radius)}-{azimuth:03d}"


def pair_label(row: CompletedRow) -> str:
    return f"{row.pair_id}: {short_viewpoint(row.viewpoint_1)} + {short_viewpoint(row.viewpoint_2)}"


def plot_top_pairs(rows: list[CompletedRow], output_path: Path, title: str, top_k: int = 20) -> None:
    top_rows = sorted(rows, key=lambda item: item.map50_95, reverse=True)[: min(top_k, len(rows))]
    fig, ax = plt.subplots(figsize=(14, max(7, len(top_rows) * 0.45)))
    labels = [pair_label(row) for row in top_rows]
    values = [row.map50_95 for row in top_rows]
    ax.barh(labels[::-1], values[::-1], color="#1f77b4")
    ax.set_xlabel("mAP50-95 on full fixed M4 test")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_viewpoint_strengths(rows: list[dict[str, object]], output_path: Path, title: str, top_k: int = 15) -> None:
    top_rows = rows[: min(top_k, len(rows))]
    fig, ax = plt.subplots(figsize=(14, max(7, len(top_rows) * 0.45)))
    labels = [short_viewpoint(str(row["viewpoint"])) for row in top_rows]
    values = [float(row["avg_mAP50-95"]) for row in top_rows]
    counts = [int(row["completed_pair_count"]) for row in top_rows]
    bars = ax.barh(labels[::-1], values[::-1], color="#ff7f0e")
    ax.set_xlabel("Average mAP50-95 across completed pairs containing the viewpoint")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts[::-1]):
        ax.text(
            bar.get_width() + 0.0008,
            bar.get_y() + bar.get_height() / 2,
            f"n={count}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_factor_patterns(rows: list[CompletedRow], output_path: Path, title_suffix: str, top_k: int = 25) -> None:
    top_rows = sorted(rows, key=lambda item: item.map50_95, reverse=True)[: min(top_k, len(rows))]

    elevation_counts = Counter()
    radius_counts = Counter()
    azimuth_counts = Counter()
    for row in top_rows:
        for viewpoint in (row.viewpoint_1, row.viewpoint_2):
            elevation, radius, azimuth = parse_viewpoint(viewpoint)
            elevation_counts[elevation] += 1
            radius_counts[radius] += 1
            azimuth_counts[f"{azimuth:03d}"] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    elev_order = ["ellow", "elmid", "elhigh"]
    elev_labels = ["low", "mid", "high"]
    elev_values = [elevation_counts.get(key, 0) for key in elev_order]
    axes[0].bar(elev_labels, elev_values, color="#2ca02c")
    axes[0].set_title(f"Elevation mix in top {len(top_rows)} pairs{title_suffix}")
    axes[0].set_ylabel("Viewpoint occurrences")

    radius_order = ["radnear", "radmid", "radfar"]
    radius_labels = ["near", "mid", "far"]
    radius_values = [radius_counts.get(key, 0) for key in radius_order]
    axes[1].bar(radius_labels, radius_values, color="#9467bd")
    axes[1].set_title(f"Radius mix in top {len(top_rows)} pairs{title_suffix}")

    azimuth_top = azimuth_counts.most_common(8)
    axes[2].bar([key for key, _ in azimuth_top], [value for _, value in azimuth_top], color="#d62728")
    axes[2].set_title(f"Most common azimuths in top {len(top_rows)} pairs{title_suffix}")
    axes[2].set_xlabel("Azimuth")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_scatter(rows: list[CompletedRow], output_path: Path, title: str, highlight_k: int = 15) -> None:
    sorted_rows = sorted(rows, key=lambda item: item.map50_95, reverse=True)
    highlights = sorted_rows[: min(highlight_k, len(sorted_rows))]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        [row.map50_95 for row in rows],
        [row.f1 for row in rows],
        s=16,
        alpha=0.35,
        color="#7f7f7f",
        label="Completed pairs",
    )
    ax.scatter(
        [row.map50_95 for row in highlights],
        [row.f1 for row in highlights],
        s=36,
        alpha=0.9,
        color="#1f77b4",
        label=f"Top {len(highlights)} pairs",
    )

    for row in highlights[:8]:
        ax.annotate(
            row.pair_id,
            (row.map50_95, row.f1),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("mAP50-95")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    master_csv = REPORTS_DIR / "master_results.csv"
    if not master_csv.exists():
        raise SystemExit(
            f"Could not find synced Ponyland report at {master_csv}. "
            "Run sync_from_ponyland.ps1 first."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = read_csv_rows(master_csv)
    completed = completed_rows(all_rows)
    if not completed:
        raise SystemExit("No completed pair results available in the current snapshot.")
    failed_count = sum(
        1
        for row in all_rows
        if "failed"
        in {
            row.get("subset_status", ""),
            row.get("training_status", ""),
            row.get("option_a_status", ""),
            row.get("option_b_status", ""),
        }
    )
    pending_count = len(all_rows) - len(completed) - failed_count
    sweep_finished = pending_count == 0

    viewpoint_summary = viewpoint_stats(completed)

    top_title = "Top duo-viewpoint pairs in full completed sweep" if sweep_finished else "Top completed duo-viewpoint pairs so far"
    viewpoint_title = "Strongest individual viewpoints in full sweep" if sweep_finished else "Current strongest individual viewpoints"
    factor_suffix = " in full sweep" if sweep_finished else " so far"
    scatter_title = "Final pair performance landscape" if sweep_finished else "Current pair performance landscape"

    plot_top_pairs(completed, PLOTS_DIR / "top_completed_pairs_current.png", title=top_title, top_k=20)
    plot_viewpoint_strengths(viewpoint_summary, PLOTS_DIR / "strongest_viewpoints_current.png", title=viewpoint_title, top_k=15)
    plot_factor_patterns(completed, PLOTS_DIR / "factor_patterns_top25_current.png", title_suffix=factor_suffix, top_k=25)
    plot_metric_scatter(completed, PLOTS_DIR / "pair_metric_scatter_current.png", title=scatter_title, highlight_k=15)

    print(f"Wrote plots under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
