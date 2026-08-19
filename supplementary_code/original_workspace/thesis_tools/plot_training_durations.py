from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_family_comparison.comparison_config import DETECTOR_ORDER, MODEL_RUNS, REGIME_ORDER


DEFAULT_OUTPUT_DIR = Path("outputs") / "thesis_tools" / "training_duration_analysis"


def read_yolo_duration(run_dir: Path) -> dict[str, float | str]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing YOLO results.csv in {run_dir}")

    with results_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {results_path}")

    last_row = rows[-1]
    return {
        "duration_seconds": float(last_row["time"]),
        "epochs_or_iters": int(float(last_row["epoch"])),
        "method": "Exact cumulative training time from YOLO results.csv",
    }


def resolve_frcnn_artifact_dir(run_dir: Path) -> Path:
    if (run_dir / "metrics.json").exists():
        return run_dir

    nested_candidates = [path for path in run_dir.iterdir() if path.is_dir() and (path / "metrics.json").exists()]
    if nested_candidates:
        return nested_candidates[0]

    explicit = run_dir / run_dir.name
    if (explicit / "metrics.json").exists():
        return explicit

    raise FileNotFoundError(f"Could not find metrics.json under {run_dir}")


def read_frcnn_duration(run_dir: Path) -> dict[str, float | str]:
    artifact_dir = resolve_frcnn_artifact_dir(run_dir)
    metrics_path = artifact_dir / "metrics.json"

    rows: list[dict[str, float | int]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    timed_rows = [row for row in rows if "time" in row and "iteration" in row]
    if not timed_rows:
        raise ValueError(f"No iteration timing rows found in {metrics_path}")

    max_iteration = int(max(row["iteration"] for row in rows if "iteration" in row))
    per_iteration_times = np.array([float(row["time"]) for row in timed_rows], dtype=float)
    seconds = float(np.median(per_iteration_times) * max_iteration)
    return {
        "duration_seconds": seconds,
        "epochs_or_iters": max_iteration,
        "method": "Estimated as median logged iteration time x final iteration from Faster R-CNN metrics.json",
    }


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detector in DETECTOR_ORDER:
        for regime in REGIME_ORDER:
            run_dir = Path(MODEL_RUNS[detector][regime])
            if detector.startswith("YOLO"):
                info = read_yolo_duration(run_dir)
            else:
                info = read_frcnn_duration(run_dir)
            seconds = float(info["duration_seconds"])
            rows.append(
                {
                    "detector": detector,
                    "regime": regime,
                    "duration_seconds": seconds,
                    "duration_minutes": seconds / 60.0,
                    "duration_hours": seconds / 3600.0,
                    "epochs_or_iters": info["epochs_or_iters"],
                    "method": info["method"],
                    "run_dir": str(run_dir),
                }
            )
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "detector",
                "regime",
                "duration_seconds",
                "duration_minutes",
                "duration_hours",
                "epochs_or_iters",
                "method",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_regime_bars(rows: list[dict[str, object]], output_path: Path) -> None:
    rows_by_detector = {
        detector: {row["regime"]: row for row in rows if row["detector"] == detector}
        for detector in DETECTOR_ORDER
    }

    x = np.arange(len(REGIME_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"YOLOv8n": "#1f77b4", "YOLOv8l": "#ff7f0e", "Faster R-CNN": "#2ca02c"}

    for idx, detector in enumerate(DETECTOR_ORDER):
        offsets = {0: -width, 1: 0.0, 2: width}
        heights = [float(rows_by_detector[detector][regime]["duration_hours"]) for regime in REGIME_ORDER]
        bars = ax.bar(
            x + offsets[idx],
            heights,
            width=width,
            label=detector,
            color=colors.get(detector),
        )
        for bar, height in zip(bars, heights, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.03,
                f"{height:.2f}h",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_ORDER)
    ax.set_ylabel("Training duration (hours)")
    ax.set_title("Training duration per regime and detector family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_detector_averages(rows: list[dict[str, object]], output_path: Path) -> None:
    averages = []
    for detector in DETECTOR_ORDER:
        detector_rows = [row for row in rows if row["detector"] == detector]
        averages.append(np.mean([float(row["duration_hours"]) for row in detector_rows]))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(DETECTOR_ORDER, averages, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for bar, value in zip(bars, averages, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}h", ha="center", fontsize=9)
    ax.set_ylabel("Average training duration (hours)")
    ax.set_title("Average training duration per detector family")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(output_path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Training duration summary", ""]
    for detector in DETECTOR_ORDER:
        detector_rows = [row for row in rows if row["detector"] == detector]
        avg_hours = np.mean([float(row["duration_hours"]) for row in detector_rows])
        lines.append(f"- {detector}: mean duration {avg_hours:.2f}h across {len(detector_rows)} regimes.")
    lines.append("")
    lines.append("Method notes:")
    lines.append("- YOLO durations are read directly from the final cumulative `time` column in `results.csv`.")
    lines.append("- Faster R-CNN durations are estimated from logged iteration time in `metrics.json` because an explicit cumulative wall-clock total was not stored.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    write_csv(output_dir / "training_durations.csv", rows)
    plot_regime_bars(rows, output_dir / "training_durations_by_regime.png")
    plot_detector_averages(rows, output_dir / "training_durations_by_detector.png")
    write_summary(output_dir / "training_duration_summary.md", rows)
    print(f"Saved training duration analysis to: {output_dir}")


if __name__ == "__main__":
    main()
