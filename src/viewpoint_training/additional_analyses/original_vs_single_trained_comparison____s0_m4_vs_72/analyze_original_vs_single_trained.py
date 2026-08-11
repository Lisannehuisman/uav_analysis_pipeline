from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AZIMUTH_ORDER = ["000", "045", "090", "135", "180", "225", "270", "315"]
ELEVATION_ORDER = ["ellow", "elmid", "elhigh"]
RADIUS_ORDER = ["radnear", "radmid", "radfar"]
ELEVATION_TO_SHORT = {"ellow": "low", "elmid": "mid", "elhigh": "high"}
ELEVATION_FROM_SHORT = {"low": "ellow", "mid": "elmid", "high": "elhigh"}
RADIUS_TO_SHORT = {"radnear": "near", "radmid": "mid", "radfar": "far"}
RADIUS_FROM_SHORT = {"near": "radnear", "mid": "radmid", "far": "radfar"}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def comparison_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_original_viewpoint(image_path: str) -> str | None:
    az_match = re.search(r"-az(\d+)", image_path, re.IGNORECASE)
    el_match = re.search(r"-el([a-z]+)", image_path, re.IGNORECASE)
    rad_match = re.search(r"-rad([a-z]+)", image_path, re.IGNORECASE)
    if not (az_match and el_match and rad_match):
        return None

    elevation_short = el_match.group(1).lower()
    if elevation_short == "ellow":
        elevation_short = "low"
    radius_short = rad_match.group(1).lower()
    azimuth = az_match.group(1)

    elevation = ELEVATION_FROM_SHORT.get(elevation_short)
    radius = RADIUS_FROM_SHORT.get(radius_short)
    if elevation is None or radius is None:
        return None
    return f"{elevation}-{radius}-az{azimuth}"


def load_original_scores(per_image_csv: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with per_image_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            viewpoint = parse_original_viewpoint(row.get("image", ""))
            value = row.get("ap50_95")
            if viewpoint is None or value in (None, ""):
                continue
            grouped[viewpoint].append(float(value))

    rows: list[dict[str, object]] = []
    for viewpoint in sorted(grouped):
        values = grouped[viewpoint]
        rows.append(
            {
                "viewpoint": viewpoint,
                "original_count": len(values),
                "original_mean_ap50_95": float(np.mean(values)),
                "original_median_ap50_95": float(median(values)),
                "original_std_ap50_95": float(np.std(np.asarray(values), ddof=0)) if len(values) > 1 else 0.0,
            }
        )
    return rows


def load_single_training_scores(master_results_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with master_results_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("evaluation_status") != "completed":
                continue
            rows.append(
                {
                    "single_id": row["single_id"],
                    "viewpoint": row["viewpoint"],
                    "number_of_train_images": int(row["number_of_train_images"]),
                    "number_of_val_images": int(row["number_of_val_images"]),
                    "number_of_test_images": int(row["number_of_test_images"]),
                    "trained_precision": float(row["precision"]),
                    "trained_recall": float(row["recall"]),
                    "trained_f1": float(row["F1"]),
                    "trained_map50": float(row["mAP50"]),
                    "trained_map50_95": float(row["mAP50-95"]),
                }
            )
    return rows


def add_factor_columns(rows: list[dict[str, object]]) -> None:
    for row in rows:
        viewpoint = str(row["viewpoint"])
        elevation, radius, azimuth = viewpoint.split("-")
        row["elevation"] = elevation
        row["radius"] = radius
        row["azimuth"] = azimuth.replace("az", "")
        row["grid_row_label"] = f"{ELEVATION_TO_SHORT[elevation]} | {RADIUS_TO_SHORT[radius]}"


def assign_descending_rank(rows: list[dict[str, object]], score_key: str, rank_key: str, pct_key: str) -> None:
    ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
    total = len(ordered)
    for idx, row in enumerate(ordered, start=1):
        row[rank_key] = idx
        if total == 1:
            row[pct_key] = 1.0
        else:
            row[pct_key] = 1.0 - ((idx - 1) / (total - 1))


def standardize(rows: list[dict[str, object]], source_key: str, target_key: str) -> None:
    values = np.asarray([float(row[source_key]) for row in rows], dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    for row in rows:
        row[target_key] = 0.0 if std == 0 else (float(row[source_key]) - mean) / std


def merge_rows(
    original_rows: list[dict[str, object]],
    trained_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_viewpoint: dict[str, dict[str, object]] = {}
    for row in original_rows:
        by_viewpoint[str(row["viewpoint"])] = dict(row)
    for row in trained_rows:
        viewpoint = str(row["viewpoint"])
        merged = by_viewpoint.setdefault(viewpoint, {"viewpoint": viewpoint})
        merged.update(row)

    merged_rows = [row for row in by_viewpoint.values() if "original_mean_ap50_95" in row and "trained_map50_95" in row]
    add_factor_columns(merged_rows)
    assign_descending_rank(merged_rows, "original_mean_ap50_95", "original_rank", "original_pct")
    assign_descending_rank(merged_rows, "trained_map50_95", "trained_rank", "trained_pct")
    standardize(merged_rows, "original_mean_ap50_95", "original_z")
    standardize(merged_rows, "trained_map50_95", "trained_z")

    for row in merged_rows:
        row["pct_delta"] = float(row["trained_pct"]) - float(row["original_pct"])
        row["z_delta"] = float(row["trained_z"]) - float(row["original_z"])
        row["rank_delta"] = int(row["original_rank"]) - int(row["trained_rank"])
        row["abs_rank_delta"] = abs(int(row["rank_delta"]))
    return merged_rows


def spearman_rank_correlation(rows: list[dict[str, object]]) -> float:
    orig = np.asarray([float(row["original_rank"]) for row in rows], dtype=float)
    train = np.asarray([float(row["trained_rank"]) for row in rows], dtype=float)
    if len(orig) < 2:
        return float("nan")
    return float(np.corrcoef(orig, train)[0, 1])


def top_n(rows: list[dict[str, object]], key: str, n: int = 10, reverse: bool = True) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:n]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def factor_summary(rows: list[dict[str, object]], factor_key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[factor_key])].append(row)

    summary: list[dict[str, object]] = []
    for factor_value in sorted(grouped):
        group_rows = grouped[factor_value]
        summary.append(
            {
                factor_key: factor_value,
                "count": len(group_rows),
                "original_mean_ap50_95": float(np.mean([float(r["original_mean_ap50_95"]) for r in group_rows])),
                "trained_mean_map50_95": float(np.mean([float(r["trained_map50_95"]) for r in group_rows])),
                "mean_pct_delta": float(np.mean([float(r["pct_delta"]) for r in group_rows])),
                "mean_rank_delta": float(np.mean([float(r["rank_delta"]) for r in group_rows])),
            }
        )
    return summary


def viewpoint_matrix(rows: list[dict[str, object]], value_key: str) -> np.ndarray:
    matrix = np.full((len(ELEVATION_ORDER) * len(RADIUS_ORDER), len(AZIMUTH_ORDER)), np.nan)
    row_lookup = {
        (elevation, radius): idx
        for idx, (elevation, radius) in enumerate(
            [(elevation, radius) for elevation in ELEVATION_ORDER for radius in RADIUS_ORDER]
        )
    }
    col_lookup = {azimuth: idx for idx, azimuth in enumerate(AZIMUTH_ORDER)}
    for row in rows:
        matrix[row_lookup[(str(row["elevation"]), str(row["radius"]))], col_lookup[str(row["azimuth"])]] = float(row[value_key])
    return matrix


def grid_labels() -> list[str]:
    return [f"{ELEVATION_TO_SHORT[e]} | {RADIUS_TO_SHORT[r]}" for e in ELEVATION_ORDER for r in RADIUS_ORDER]


def plot_dual_heatmaps(rows: list[dict[str, object]], output_path: Path) -> None:
    original_matrix = viewpoint_matrix(rows, "original_mean_ap50_95")
    trained_matrix = viewpoint_matrix(rows, "trained_map50_95")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    row_labels = grid_labels()

    im0 = axes[0].imshow(original_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("Original S4 model: viewpoint as test-time observation")
    axes[0].set_xticks(range(len(AZIMUTH_ORDER)), labels=AZIMUTH_ORDER)
    axes[0].set_yticks(range(len(row_labels)), labels=row_labels)
    axes[0].set_xlabel("Azimuth")
    axes[0].set_ylabel("Elevation | Radius")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Mean per-image AP50-95")

    im1 = axes[1].imshow(trained_matrix, cmap="magma", aspect="auto")
    axes[1].set_title("Single-view trained model: viewpoint as only training view")
    axes[1].set_xticks(range(len(AZIMUTH_ORDER)), labels=AZIMUTH_ORDER)
    axes[1].set_yticks(range(len(row_labels)), labels=row_labels)
    axes[1].set_xlabel("Azimuth")
    axes[1].set_ylabel("Elevation | Radius")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Full-test mAP50-95")

    for ax in axes:
        ax.set_xticks(np.arange(-0.5, len(AZIMUTH_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("Same 72 viewpoints, but two different roles", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_percentile_delta_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    matrix = viewpoint_matrix(rows, "pct_delta")
    row_labels = grid_labels()
    vmax = float(np.nanmax(np.abs(matrix)))
    vmax = max(vmax, 0.25)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Normalized shift: stronger as training viewpoint vs stronger as original test viewpoint")
    ax.set_xticks(range(len(AZIMUTH_ORDER)), labels=AZIMUTH_ORDER)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation | Radius")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentile shift (trained - original)")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if math.isnan(value) or abs(value) < 0.18:
                continue
            text_color = "white" if abs(value) > 0.28 else "black"
            ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(AZIMUTH_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rank_shift_dumbbell(rows: list[dict[str, object]], output_path: Path, top_k: int = 20) -> None:
    selected = sorted(rows, key=lambda row: (float(row["abs_rank_delta"]), float(row["trained_rank"])), reverse=True)[:top_k]
    selected = sorted(selected, key=lambda row: float(row["rank_delta"]), reverse=True)
    labels = [str(row["viewpoint"]) for row in selected]
    y = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    for idx, row in enumerate(selected):
        original_rank = float(row["original_rank"])
        trained_rank = float(row["trained_rank"])
        color = "#1a9850" if trained_rank < original_rank else "#d73027"
        ax.plot([original_rank, trained_rank], [idx, idx], color=color, linewidth=2.4, alpha=0.9)
        ax.scatter(original_rank, idx, color="#4c78a8", s=70, zorder=3, label="Original S4 rank" if idx == 0 else None)
        ax.scatter(trained_rank, idx, color="#f58518", s=70, zorder=3, label="Single-trained rank" if idx == 0 else None)

    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("Rank among 72 viewpoints (1 = strongest)")
    ax.set_title("Viewpoints with the largest role reversal")
    ax.invert_xaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_percentile_scatter(rows: list[dict[str, object]], output_path: Path) -> None:
    x = np.asarray([float(row["original_pct"]) for row in rows], dtype=float)
    y = np.asarray([float(row["trained_pct"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.scatter(x, y, s=55, c=[float(row["pct_delta"]) for row in rows], cmap="coolwarm", alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel("Original S4 viewpoint percentile")
    ax.set_ylabel("Single-trained viewpoint percentile")
    ax.set_title("Do good observation viewpoints also make good training viewpoints?")
    ax.grid(alpha=0.25)

    notable = sorted(rows, key=lambda row: float(row["abs_rank_delta"]), reverse=True)[:10]
    for row in notable:
        ax.text(float(row["original_pct"]) + 0.01, float(row["trained_pct"]) + 0.01, str(row["viewpoint"]), fontsize=8)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    spearman = spearman_rank_correlation(rows)
    best_original = max(rows, key=lambda row: float(row["original_mean_ap50_95"]))
    best_trained = max(rows, key=lambda row: float(row["trained_map50_95"]))
    largest_training_wins = top_n(rows, "rank_delta", 8, reverse=True)
    largest_test_wins = top_n(rows, "rank_delta", 8, reverse=False)

    top10_original = {str(row["viewpoint"]) for row in top_n(rows, "original_mean_ap50_95", 10)}
    top10_trained = {str(row["viewpoint"]) for row in top_n(rows, "trained_map50_95", 10)}
    overlap = sorted(top10_original & top10_trained)

    elevation_rows = factor_summary(rows, "elevation")
    radius_rows = factor_summary(rows, "radius")

    lines = [
        "# Original S4 vs Single-Viewpoint-Trained Comparison",
        "",
        "## What Is Being Compared",
        "",
        "- `Original S4 viewpoint strength`: how strong a viewpoint is as a test-time observation under the original full-viewpoint model.",
        "- `Single-trained viewpoint strength`: how well a model generalizes when trained on only that one viewpoint and evaluated on the full fixed M4 test set.",
        "- These are related but not identical roles, so rank shifts are more interpretable than raw score gaps.",
        "",
        "## Headline Results",
        "",
        f"- Viewpoints compared: `{len(rows)}`",
        f"- Spearman rank correlation between the two orderings: `{spearman:.3f}`",
        f"- Best original S4 observation viewpoint: `{best_original['viewpoint']}` with mean per-image AP50-95 `{float(best_original['original_mean_ap50_95']):.4f}`",
        f"- Best single-trained viewpoint: `{best_trained['viewpoint']}` with full-test mAP50-95 `{float(best_trained['trained_map50_95']):.4f}`",
        f"- Overlap between top-10 original viewpoints and top-10 trained viewpoints: `{len(overlap)} / 10`",
        "",
        "## Largest Positive Training Shifts",
        "",
    ]

    for row in largest_training_wins:
        lines.append(
            f"- `{row['viewpoint']}`: original rank `{row['original_rank']}` -> trained rank `{row['trained_rank']}`"
            f" (`{int(row['rank_delta']):+d}` places better as a training viewpoint)"
        )

    lines.extend(["", "## Largest Negative Training Shifts", ""])
    for row in largest_test_wins:
        lines.append(
            f"- `{row['viewpoint']}`: original rank `{row['original_rank']}` -> trained rank `{row['trained_rank']}`"
            f" (`{int(row['rank_delta']):+d}` places worse as a training viewpoint)"
        )

    lines.extend(["", "## Factor-Level Signals", "", "### Elevation", ""])
    for row in elevation_rows:
        lines.append(
            f"- `{row['elevation']}`: original mean `{float(row['original_mean_ap50_95']):.4f}`, "
            f"trained mean `{float(row['trained_mean_map50_95']):.4f}`, "
            f"mean percentile shift `{float(row['mean_pct_delta']):+.3f}`"
        )

    lines.extend(["", "### Radius", ""])
    for row in radius_rows:
        lines.append(
            f"- `{row['radius']}`: original mean `{float(row['original_mean_ap50_95']):.4f}`, "
            f"trained mean `{float(row['trained_mean_map50_95']):.4f}`, "
            f"mean percentile shift `{float(row['mean_pct_delta']):+.3f}`"
        )

    lines.extend(["", "## Top-10 Overlap", ""])
    if overlap:
        for viewpoint in overlap:
            lines.append(f"- `{viewpoint}`")
    else:
        lines.append("- No overlap between the two top-10 lists.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    workspace = workspace_root()
    comp_root = comparison_root()
    output_dir = comp_root / "outputs"
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    original_csv = workspace / "comparison_output" / "per_image_metrics_model_b.csv"
    single_csv = workspace / "viewpoint_data_separated" / "s4_individual_viewpoints" / "reports" / "master_results.csv"

    original_rows = load_original_scores(original_csv)
    trained_rows = load_single_training_scores(single_csv)
    merged_rows = merge_rows(original_rows, trained_rows)

    write_csv(output_dir / "original_vs_single_trained_viewpoints.csv", merged_rows)
    write_csv(output_dir / "elevation_summary.csv", factor_summary(merged_rows, "elevation"))
    write_csv(output_dir / "radius_summary.csv", factor_summary(merged_rows, "radius"))
    write_summary(output_dir / "original_vs_single_trained_summary.md", merged_rows)

    plot_dual_heatmaps(merged_rows, plots_dir / "original_vs_trained_heatmaps.png")
    plot_percentile_delta_heatmap(merged_rows, plots_dir / "normalized_delta_heatmap.png")
    plot_rank_shift_dumbbell(merged_rows, plots_dir / "rank_shift_dumbbell_top20.png")
    plot_percentile_scatter(merged_rows, plots_dir / "original_vs_trained_percentile_scatter.png")


if __name__ == "__main__":
    main()
