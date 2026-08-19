from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_JSON = Path("outputs") / "m4_pair_subset_experiment" / "presentation" / "supervisor_update_data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation assets for the supervisor update on the M4 viewpoint-pair training pilot.",
    )
    parser.add_argument(
        "--data-json",
        default=str(DEFAULT_DATA_JSON),
        help="JSON file containing the confirmed pilot status and metrics.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pilot_progress(data: dict, output_dir: Path) -> Path:
    pairs = data["pilot_pairs"]
    completed = [pair for pair in pairs if pair["status"] == "completed"]
    running = [pair for pair in pairs if pair["status"] == "running"]
    failed = [pair for pair in pairs if pair["status"] == "failed"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    counts = [len(completed), len(running), len(failed)]
    labels = ["Completed", "Running", "Failed"]
    colors = ["#2e8b57", "#f0ad4e", "#d9534f"]
    total = max(1, sum(counts))

    left = 0
    for label, count, color in zip(labels, counts, colors):
        if count == 0:
            continue
        ax.barh(["Pilot status"], [count], left=left, color=color, height=0.5)
        ax.text(left + count / 2, 0, f"{label}\n{count}", ha="center", va="center", color="white", fontsize=12, weight="bold")
        left += count

    ax.set_xlim(0, total)
    ax.set_xticks(range(total + 1))
    ax.set_xlabel("Number of pilot pairs")
    ax.set_title("Pilot execution status (interim)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    detail_lines = [
        "Completed: " + ", ".join(pair["pair_id"] for pair in completed) if completed else "Completed: none",
        "Running: " + ", ".join(pair["pair_id"] for pair in running) if running else "Running: none",
        "Failed: " + ", ".join(pair["pair_id"] for pair in failed) if failed else "Failed: none",
    ]
    fig.text(0.08, 0.02, "\n".join(detail_lines), fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path = output_dir / "pilot_progress.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_best_vs_baseline(data: dict, output_dir: Path) -> Path:
    baseline = data["baseline_metrics"]
    best = data["best_so_far"]
    metric_keys = ["f1", "map50", "map50_95"]
    display_names = ["F1", "mAP50", "mAP50-95"]
    baseline_values = [baseline[key] for key in metric_keys]
    best_values = [best["metrics"][key] for key in metric_keys]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x_positions = list(range(len(metric_keys)))
    width = 0.35

    ax.bar([x - width / 2 for x in x_positions], baseline_values, width=width, label="Full M4 baseline", color="#4c78a8")
    ax.bar([x + width / 2 for x in x_positions], best_values, width=width, label=f"Best pilot so far ({best['pair_id']})", color="#f58518")

    for x_pos, value in zip([x - width / 2 for x in x_positions], baseline_values):
        ax.text(x_pos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    for x_pos, value in zip([x + width / 2 for x in x_positions], best_values):
        ax.text(x_pos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Best interim pilot pair vs full M4 YOLOv8l baseline")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    output_path = output_dir / "best_so_far_vs_baseline.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pilot_pair_table(data: dict, output_dir: Path) -> Path:
    rows = [
        [pair["pair_id"], pair["label"], pair["type"], pair["status"].capitalize()]
        for pair in data["pilot_pairs"]
    ]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Pair", "Viewpoints", "Pilot rationale", "Status"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#dce6f2")
            cell.set_text_props(weight="bold")
            continue
        if col_idx == 3:
            value = rows[row_idx - 1][3].lower()
            if value == "completed":
                cell.set_facecolor("#d9ead3")
            elif value == "running":
                cell.set_facecolor("#fce5cd")
            elif value == "failed":
                cell.set_facecolor("#f4cccc")

    ax.set_title("Pilot pair design and current status", pad=12, fontsize=13)
    output_path = output_dir / "pilot_pair_overview.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    data_json = Path(args.data_json).resolve()
    data = read_json(data_json)
    output_dir = data_json.parent
    ensure_output_dir(output_dir)

    created = [
        plot_pilot_progress(data, output_dir),
        plot_best_vs_baseline(data, output_dir),
        plot_pilot_pair_table(data, output_dir),
    ]
    for path in created:
        print(f"Wrote asset: {path}")


if __name__ == "__main__":
    main()
