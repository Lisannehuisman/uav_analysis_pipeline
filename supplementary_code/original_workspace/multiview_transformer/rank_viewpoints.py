from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from multiview_transformer.common import read_csv_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiview evaluation CSVs into viewpoint rankings.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing combo_*_summary.csv outputs.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top combinations to surface per table.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def contribution_rows(summary_rows: list[dict[str, str]], combo_size: int) -> list[dict[str, object]]:
    predicted_lookup: dict[str, list[float]] = defaultdict(list)
    actual_lookup: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        viewpoints = [str(row[f"viewpoint_{index}"]) for index in range(1, combo_size + 1) if row.get(f"viewpoint_{index}")]
        for viewpoint in viewpoints:
            predicted_lookup[viewpoint].append(float(row["mean_predicted_set_score"]))
            actual_lookup[viewpoint].append(float(row["mean_actual_set_score"]))

    rows: list[dict[str, object]] = []
    for viewpoint in sorted(predicted_lookup):
        rows.append(
            {
                "viewpoint": viewpoint,
                "combo_size": combo_size,
                "combo_count": len(predicted_lookup[viewpoint]),
                "mean_predicted_combo_score": mean(predicted_lookup[viewpoint]),
                "mean_actual_combo_score": mean(actual_lookup[viewpoint]),
            }
        )
    rows.sort(key=lambda row: (-float(row["mean_predicted_combo_score"]), -float(row["mean_actual_combo_score"])))
    return rows


def by_class_rows(scene_rows: list[dict[str, str]], combo_size: int, top_k: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(str(row["target_class"]), str(row["combination_label"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (target_class, combination_label), members in grouped.items():
        row: dict[str, object] = {
            "target_class": target_class,
            "combination_label": combination_label,
            "combo_size": combo_size,
            "scene_count": len(members),
            "mean_predicted_set_score": mean([float(item["predicted_set_score"]) for item in members]),
            "mean_actual_set_score": mean([float(item["actual_set_score"]) for item in members]),
        }
        for index in range(1, combo_size + 1):
            row[f"viewpoint_{index}"] = members[0].get(f"viewpoint_{index}", "")
        summary_rows.append(row)

    selected: list[dict[str, object]] = []
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_class[str(row["target_class"])].append(row)
    for target_class, members in by_class.items():
        members.sort(key=lambda row: (-float(row["mean_predicted_set_score"]), -float(row["mean_actual_set_score"])))
        selected.extend(members[:top_k])
    return selected


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    ranking_dir = eval_dir / "rankings"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    markdown_lines = ["# Multiview ranking summary", ""]
    for summary_path in sorted(eval_dir.glob("combo_*_summary.csv")):
        combo_size = int(summary_path.stem.split("_")[1])
        scene_path = eval_dir / f"combo_{combo_size}_scene_predictions.csv"
        summary_rows = read_csv_rows(summary_path)
        scene_rows = read_csv_rows(scene_path) if scene_path.exists() else []
        if not summary_rows:
            continue

        top_rows = summary_rows[: args.top_k]
        top_path = ranking_dir / f"top_combo_{combo_size}.csv"
        write_csv_rows(top_path, fieldnames=list(top_rows[0].keys()), rows=top_rows)

        contribution = contribution_rows(summary_rows=summary_rows, combo_size=combo_size)
        if contribution:
            contribution_path = ranking_dir / f"viewpoint_contributions_combo_{combo_size}.csv"
            write_csv_rows(contribution_path, fieldnames=list(contribution[0].keys()), rows=contribution)

        if scene_rows:
            per_class = by_class_rows(scene_rows=scene_rows, combo_size=combo_size, top_k=min(3, args.top_k))
            if per_class:
                per_class_path = ranking_dir / f"top_combo_{combo_size}_by_class.csv"
                write_csv_rows(per_class_path, fieldnames=list(per_class[0].keys()), rows=per_class)

        best = top_rows[0]
        markdown_lines.extend(
            [
                f"## Combo Size {combo_size}",
                "",
                f"- Best combination by predicted score: `{best['combination_label']}`",
                f"- Mean predicted set score: `{float(best['mean_predicted_set_score']):.4f}`",
                f"- Mean actual set score: `{float(best['mean_actual_set_score']):.4f}`",
                f"- Scene support: `{best['scene_count']}`",
                "",
            ]
        )

    (ranking_dir / "ranking_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Wrote rankings to {ranking_dir}")


if __name__ == "__main__":
    main()
