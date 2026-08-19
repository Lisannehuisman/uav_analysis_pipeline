from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


AZIMUTH_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATION_ORDER = ["ellow", "elmid", "elhigh"]
RADIUS_ORDER = ["radnear", "radmid", "radfar"]
ELEVATION_LABELS = {"ellow": "Low", "elmid": "Mid", "elhigh": "High"}
RADIUS_LABELS = {"radnear": "Near", "radmid": "Mid", "radfar": "Far"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-grid visualization for the 72 single-viewpoint training models."
    )
    parser.add_argument(
        "--master-results",
        default="viewpoint_data_separated/72_trained_models/reports/master_results.csv",
        help="CSV with completed single-viewpoint training results.",
    )
    parser.add_argument(
        "--output",
        default="viewpoint_data_separated/72_trained_models/plots/single_viewpoint_angle_grid.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many strongest cells to outline.",
    )
    return parser.parse_args()


def parse_viewpoint(viewpoint: str) -> tuple[str, str, int]:
    elevation, radius, azimuth = viewpoint.split("-")
    return elevation, radius, int(azimuth.replace("az", ""))


def read_completed_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("evaluation_status") != "completed":
                continue
            rows.append(
                {
                    "single_id": row["single_id"],
                    "viewpoint": row["viewpoint"],
                    "mAP50-95": float(row["mAP50-95"]),
                    "mAP50": float(row["mAP50"]),
                    "F1": float(row["F1"]),
                }
            )
    return rows


def row_order() -> list[tuple[str, str]]:
    return [(elevation, radius) for elevation in ELEVATION_ORDER for radius in RADIUS_ORDER]


def row_labels() -> list[str]:
    return [
        f"{ELEVATION_LABELS[elevation]} | {RADIUS_LABELS[radius]}"
        for elevation, radius in row_order()
    ]


def build_lookup(rows: list[dict[str, object]]) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, object]]]:
    row_index = {pair: idx for idx, pair in enumerate(row_order())}
    col_index = {azimuth: idx for idx, azimuth in enumerate(AZIMUTH_ORDER)}
    grid = np.full((len(row_index), len(col_index)), np.nan, dtype=float)
    cell_rows: dict[tuple[int, int], dict[str, object]] = {}

    ordered = sorted(rows, key=lambda row: float(row["mAP50-95"]), reverse=True)
    rank_by_viewpoint = {str(row["viewpoint"]): idx for idx, row in enumerate(ordered, start=1)}

    for row in rows:
        elevation, radius, azimuth = parse_viewpoint(str(row["viewpoint"]))
        r_idx = row_index[(elevation, radius)]
        c_idx = col_index[azimuth]
        grid[r_idx, c_idx] = float(row["mAP50-95"])
        cell_rows[(r_idx, c_idx)] = {
            **row,
            "rank": rank_by_viewpoint[str(row["viewpoint"])],
            "azimuth": azimuth,
            "elevation": elevation,
            "radius": radius,
        }

    return grid, cell_rows


def text_color(value: float, vmin: float, vmax: float) -> str:
    midpoint = (vmin + vmax) / 2.0
    return "white" if value < midpoint else "black"


def main() -> None:
    args = parse_args()
    master_results = Path(args.master_results).resolve()
    output = Path(args.output).resolve()

    rows = read_completed_rows(master_results)
    if not rows:
        raise SystemExit(f"No completed rows found in {master_results}.")

    grid, cell_rows = build_lookup(rows)
    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    image = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(AZIMUTH_ORDER)), labels=[f"{azimuth:03d}°" for azimuth in AZIMUTH_ORDER])
    ax.set_yticks(range(len(row_labels())), labels=row_labels())
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation | Radius")
    ax.set_title("72 single-view training models: one cell = one training viewpoint")

    ax.set_xticks(np.arange(-0.5, len(AZIMUTH_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels()), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    for separator in (2.5, 5.5):
        ax.axhline(separator, color="white", linewidth=2.4, alpha=0.95)

    ax.text(
        7.85,
        -1.0,
        "Top viewpoints outlined in white",
        ha="right",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    top_k = max(0, int(args.top_k))
    for (r_idx, c_idx), row in cell_rows.items():
        value = float(row["mAP50-95"])
        rank = int(row["rank"])
        color = text_color(value, vmin, vmax)

        ax.text(
            c_idx,
            r_idx,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold" if rank <= top_k else "normal",
        )

        if rank <= top_k:
            ax.add_patch(
                Rectangle(
                    (c_idx - 0.5, r_idx - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=2.4,
                )
            )
            ax.text(
                c_idx - 0.44,
                r_idx - 0.34,
                f"#{rank}",
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )

    best_row = max(rows, key=lambda row: float(row["mAP50-95"]))
    best_view = str(best_row["viewpoint"])
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="mAP50-95 on full fixed M4 test")
    fig.text(
        0.01,
        0.01,
        (
            f"Best single-view training viewpoint: {best_view} "
            f"(mAP50-95 = {float(best_row['mAP50-95']):.4f})."
        ),
        fontsize=10,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
