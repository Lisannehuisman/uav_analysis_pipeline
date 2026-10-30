from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize best/worst viewpoint analysis CSV outputs."
    )
    parser.add_argument(
        "--combo-csv",
        default="comparison_output/best_viewpoint_analysis_s0_m4/best_worst_combo_ap50_95.csv",
        help="CSV with best and worst full viewpoint combinations per object.",
    )
    parser.add_argument(
        "--factors-csv",
        default="comparison_output/best_viewpoint_analysis_s0_m4/best_worst_factors_ap50_95.csv",
        help="CSV with best and worst azimuth/elevation/radius factors per object.",
    )
    parser.add_argument(
        "--label",
        default="S0_M4",
        help="Model label used in figure titles.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output/best_viewpoint_analysis_s0_m4/figures",
        help="Folder where visualizations will be written.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_combo_gap(rows: list[dict[str, str]], label: str, output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: to_float(row, "combo_median_gap"), reverse=True)
    classes = [row["class_name"] for row in ordered]
    gaps = [to_float(row, "combo_median_gap") for row in ordered]

    fig, ax = plt.subplots(figsize=(12, max(5, len(classes) * 0.5)))
    y = np.arange(len(classes))
    ax.barh(y, gaps, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("Best-worst median gap")
    ax.set_title(f"{label} viewpoint sensitivity by object")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_combo_best_worst(rows: list[dict[str, str]], label: str, output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: to_float(row, "combo_median_gap"), reverse=True)
    classes = [row["class_name"] for row in ordered]
    best_vals = [to_float(row, "best_combo_median") for row in ordered]
    worst_vals = [to_float(row, "worst_combo_median") for row in ordered]

    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(12, max(5, len(classes) * 0.5)))
    ax.scatter(best_vals, y, color="#2ca02c", label="Best combo median", s=40)
    ax.scatter(worst_vals, y, color="#d62728", label="Worst combo median", s=40)
    for idx, row in enumerate(ordered):
        ax.plot(
            [to_float(row, "worst_combo_median"), to_float(row, "best_combo_median")],
            [idx, idx],
            color="#7f7f7f",
            linewidth=1.2,
            alpha=0.8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Median score")
    ax.set_title(f"{label} best vs worst full viewpoint combination")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_factor_heatmap(rows: list[dict[str, str]], label: str, output_path: Path) -> None:
    class_names = sorted({row["class_name"] for row in rows})
    factor_order = ["azimuth", "elevation", "radius"]
    matrix = np.full((len(class_names), len(factor_order)), np.nan, dtype=float)

    for row in rows:
        class_idx = class_names.index(row["class_name"])
        factor_idx = factor_order.index(row["group_type"])
        matrix[class_idx, factor_idx] = to_float(row, "median_gap")

    fig, ax = plt.subplots(figsize=(8, max(5, len(class_names) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(factor_order)))
    ax.set_xticklabels(factor_order)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title(f"{label} factor sensitivity heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best-worst median gap")

    for row_idx in range(len(class_names)):
        for col_idx in range(len(factor_order)):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.12 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_factor_best_worst(rows: list[dict[str, str]], label: str, output_path: Path) -> None:
    factor_order = ["azimuth", "elevation", "radius"]
    class_names = sorted({row["class_name"] for row in rows})

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(class_names) * 0.45)), sharey=True)
    for idx, factor in enumerate(factor_order):
        factor_rows = [row for row in rows if row["group_type"] == factor]
        factor_rows = sorted(factor_rows, key=lambda row: class_names.index(row["class_name"]))
        y = np.arange(len(factor_rows))
        best_vals = [to_float(row, "best_median") for row in factor_rows]
        worst_vals = [to_float(row, "worst_median") for row in factor_rows]

        axes[idx].scatter(best_vals, y, color="#2ca02c", label="Best", s=32)
        axes[idx].scatter(worst_vals, y, color="#d62728", label="Worst", s=32)
        for row_i, row in enumerate(factor_rows):
            axes[idx].plot(
                [to_float(row, "worst_median"), to_float(row, "best_median")],
                [row_i, row_i],
                color="#7f7f7f",
                linewidth=1.0,
                alpha=0.7,
            )
        axes[idx].set_title(factor.capitalize())
        axes[idx].set_xlim(0, 1)
        axes[idx].grid(axis="x", linestyle="--", alpha=0.3)
        axes[idx].set_xlabel("Median score")
        if idx == 0:
            axes[idx].set_yticks(y)
            axes[idx].set_yticklabels([row["class_name"] for row in factor_rows])
        else:
            axes[idx].set_yticks(y)
            axes[idx].set_yticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{label} best vs worst factor medians", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_text_summary(combo_rows: list[dict[str, str]], factor_rows: list[dict[str, str]], output_path: Path) -> None:
    most_sensitive = max(combo_rows, key=lambda row: to_float(row, "combo_median_gap"))
    least_sensitive = min(combo_rows, key=lambda row: to_float(row, "combo_median_gap"))

    strongest_factors: dict[str, tuple[str, float]] = {}
    for factor in ("azimuth", "elevation", "radius"):
        subset = [row for row in factor_rows if row["group_type"] == factor]
        if subset:
            best = max(subset, key=lambda row: to_float(row, "median_gap"))
            strongest_factors[factor] = (best["class_name"], to_float(best, "median_gap"))

    lines = [
        f"Most viewpoint-sensitive object: {most_sensitive['class_name']} (gap={to_float(most_sensitive, 'combo_median_gap'):.3f})",
        f"Least viewpoint-sensitive object: {least_sensitive['class_name']} (gap={to_float(least_sensitive, 'combo_median_gap'):.3f})",
    ]
    for factor, (class_name, gap) in strongest_factors.items():
        lines.append(f"Strongest {factor} effect: {class_name} (gap={gap:.3f})")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    combo_csv = Path(args.combo_csv).resolve()
    factors_csv = Path(args.factors_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_rows = read_rows(combo_csv)
    factor_rows = read_rows(factors_csv)

    plot_combo_gap(combo_rows, args.label, output_dir / "combo_gap_ranking.png")
    plot_combo_best_worst(combo_rows, args.label, output_dir / "combo_best_vs_worst.png")
    plot_factor_heatmap(factor_rows, args.label, output_dir / "factor_gap_heatmap.png")
    plot_factor_best_worst(factor_rows, args.label, output_dir / "factor_best_vs_worst.png")
    write_text_summary(combo_rows, factor_rows, output_dir / "summary.txt")

    print(f"Saved viewpoint visualizations to: {output_dir}")


if __name__ == "__main__":
    main()



# python visualize_best_viewpoints.py `
#   --combo-csv ".\comparison_output\best_viewpoint_analysis_s0_m4\best_worst_combo_ap50_95.csv" `
#   --factors-csv ".\comparison_output\best_viewpoint_analysis_s0_m4\best_worst_factors_ap50_95.csv" `
#   --label "S0_M4" `
#   --output-dir ".\comparison_output\best_viewpoint_analysis_s0_m4\figures"
