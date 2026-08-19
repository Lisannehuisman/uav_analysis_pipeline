from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "s0_m4" / "thesis_viewpoint_analysis_s0_m4"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thesis_tools" / "official_s0_m4_best_viewpoint_cards"
HEATMAP_DIRNAME = "heatmaps_avgoverradius"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate per-object cards from the official S0_M4 viewpoint analysis outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Folder containing ideal_viewpoints.csv, top3_viewpoints.csv, and heatmaps_avgoverradius.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where separate per-object cards will be written.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_score(value: str) -> str:
    return f"{float(value):.3f}"


def viewpoint_text(azimuth: str, elevation: str, radius: str) -> str:
    return f"az{azimuth} | {elevation} | {radius}"


def load_top3_by_object(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        grouped[row["object_class"]].append(row)
    for object_class in grouped:
        grouped[object_class].sort(key=lambda row: int(row["rank"]))
    return grouped


def build_card(
    object_row: dict[str, str],
    top3_rows: list[dict[str, str]],
    heatmap_path: Path,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(10.5, 7.6), facecolor="#f8fafc")
    fig.suptitle(object_row["object_class"], fontsize=24, fontweight="bold", y=0.965)

    heatmap_ax = fig.add_axes([0.07, 0.20, 0.64, 0.66])
    heatmap_ax.imshow(mpimg.imread(str(heatmap_path)))
    heatmap_ax.axis("off")

    info_ax = fig.add_axes([0.74, 0.20, 0.22, 0.66])
    info_ax.axis("off")

    best_text = viewpoint_text(
        object_row["best_azimuth"],
        object_row["best_elevation"],
        object_row["best_radius"],
    )
    info_lines = [
        "Official S0_M4 best viewpoint",
        best_text,
        "",
        f"Mean AP50-95: {fmt_score(object_row['mean_ap50_95'])}",
        f"Support: n = {object_row['n_images']}",
        f"Std: {fmt_score(object_row['std_ap50_95'])}",
        "",
        "Top 3 exact viewpoints",
    ]

    y = 1.0
    for line in info_lines:
        info_ax.text(
            0.0,
            y,
            line,
            ha="left",
            va="top",
            fontsize=11 if line else 8,
            fontweight="bold" if line in {"Official S0_M4 best viewpoint", "Top 3 exact viewpoints"} else "normal",
            color="#0f172a",
            transform=info_ax.transAxes,
        )
        y -= 0.075 if line else 0.04

    rank_colors = {1: "#2563eb", 2: "#f97316", 3: "#16a34a"}
    for row in top3_rows:
        rank = int(row["rank"])
        label = viewpoint_text(row["azimuth"], row["elevation"], row["radius"])
        info_ax.text(
            0.0,
            y,
            f"#{rank}  {label}",
            ha="left",
            va="top",
            fontsize=10.5,
            color=rank_colors.get(rank, "#334155"),
            fontweight="bold",
            transform=info_ax.transAxes,
        )
        y -= 0.06
        info_ax.text(
            0.0,
            y,
            f"mean {fmt_score(row['mean_value'])} | n = {row['n_images']}",
            ha="left",
            va="top",
            fontsize=9.5,
            color="#475569",
            transform=info_ax.transAxes,
        )
        y -= 0.08

    fig.text(
        0.07,
        0.08,
        "Heatmap shows azimuth x elevation averaged over radius. The exact best viewpoint above comes from the official ideal_viewpoints.csv.",
        fontsize=10,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_copy(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    objects_dir = output_dir / "objects"

    ideal_csv = analysis_dir / "ideal_viewpoints.csv"
    top3_csv = analysis_dir / "top3_viewpoints.csv"
    heatmap_dir = analysis_dir / HEATMAP_DIRNAME

    ideal_rows = read_csv_rows(ideal_csv)
    top3_by_object = load_top3_by_object(top3_csv)

    for row in ideal_rows:
        object_class = row["object_class"]
        heatmap_path = heatmap_dir / f"{object_class}_ap50_95_heatmap.png"
        output_path = objects_dir / f"{object_class}_official_s0_m4_best_viewpoint.png"
        build_card(
            object_row=row,
            top3_rows=top3_by_object.get(object_class, []),
            heatmap_path=heatmap_path,
            output_path=output_path,
        )

    write_summary_copy(ideal_rows, output_dir / "ideal_viewpoints_official_s0_m4.csv")
    print(f"Saved official S0_M4 viewpoint cards to: {objects_dir}")


if __name__ == "__main__":
    main()
