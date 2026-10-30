from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_config import DETECTOR_ORDER, REGIME_ORDER


COLORS = {
    "YOLOv8n": "#4C78A8",
    "YOLOv8l": "#F58518",
    "Faster R-CNN": "#54A24B",
}

MARKERS = {
    "YOLOv8n": "o",
    "YOLOv8l": "s",
    "Faster R-CNN": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mAP50:95 across detector families and viewpoint-exposure regimes.",
    )
    parser.add_argument("--summary-csv", required=True, help="Path to standardized comparison summary CSV.")
    parser.add_argument("--output", required=True, help="PNG path for the simplified mAP50:95 figure.")
    parser.add_argument("--pdf-output", help="Optional PDF path for LaTeX inclusion.")
    parser.add_argument(
        "--title",
        default="Test mAP50:95 by viewpoint-exposure regime",
        help="Figure title.",
    )
    return parser.parse_args()


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_regime_map50_95(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    pdf_output_path: Path | None = None,
) -> None:
    lookup = {(row["regime"], row["detector"]): row for row in rows}
    x = np.arange(len(REGIME_ORDER))

    fig, axis = plt.subplots(figsize=(10.5, 5.4))

    for detector in DETECTOR_ORDER:
        values = [float(lookup[(regime, detector)]["map50_95"]) for regime in REGIME_ORDER]
        axis.plot(
            x,
            values,
            color=COLORS[detector],
            marker=MARKERS[detector],
            markersize=7,
            linewidth=2.8,
            label=detector,
        )

    selected_x = REGIME_ORDER.index("M4")
    selected_y = float(lookup[("M4", "YOLOv8l")]["map50_95"])
    axis.scatter(
        [selected_x],
        [selected_y],
        s=230,
        facecolors="none",
        edgecolors="#222222",
        linewidths=2.2,
        zorder=6,
    )
    axis.scatter(
        [selected_x],
        [selected_y],
        s=95,
        marker="*",
        color="#222222",
        zorder=7,
    )
    axis.annotate(
        f"Selected model\nYOLOv8l-M4\n{selected_y:.3f}",
        xy=(selected_x, selected_y),
        xytext=(-105, -18),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.2},
    )

    axis.set_title(title, fontsize=15, fontweight="bold", pad=18)
    axis.set_xlabel("Training regime", labelpad=12)
    axis.set_ylabel("mAP50:95", labelpad=10)
    axis.set_xticks(x)
    axis.set_xticklabels(REGIME_ORDER)
    axis.set_ylim(0.35, 0.68)
    axis.set_yticks([0.35, 0.45, 0.55, 0.65])
    axis.grid(axis="y", linestyle="--", alpha=0.32)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    if pdf_output_path is not None:
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_path = Path(args.output).resolve()
    pdf_output_path = Path(args.pdf_output).resolve() if args.pdf_output else None
    rows = load_summary_rows(summary_csv)
    plot_regime_map50_95(rows, output_path, args.title, pdf_output_path)
    print(f"Saved simplified mAP50:95 figure to: {output_path}")
    if pdf_output_path is not None:
        print(f"Saved simplified mAP50:95 PDF to: {pdf_output_path}")


if __name__ == "__main__":
    main()
