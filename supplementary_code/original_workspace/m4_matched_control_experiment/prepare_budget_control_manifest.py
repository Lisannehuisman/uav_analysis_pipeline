from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from matched_control_lib import (
    DEFAULT_CONTROL_EXPERIMENT_ROOT,
    control_manifest_path,
    ensure_control_experiment_root,
    write_csv_rows,
)


DEFAULT_SINGLE_RESULTS = Path("viewpoint_data_separated") / "72_trained_models" / "reports" / "master_results.csv"
DEFAULT_PAIR_RESULTS = Path("viewpoint_data_separated") / "m4_pair_results" / "snapshot" / "reports" / "master_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create equal-image-count full-M4 matched controls for the train/val budgets "
            "that appear in the single-view and pair-view result tables."
        )
    )
    parser.add_argument("--experiment-root", default=str(DEFAULT_CONTROL_EXPERIMENT_ROOT))
    parser.add_argument("--single-results-csv", default=str(DEFAULT_SINGLE_RESULTS))
    parser.add_argument("--pair-results-csv", default=str(DEFAULT_PAIR_RESULTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Sampling seeds for each budget control.")
    parser.add_argument(
        "--grouping",
        choices=["exact", "train_only"],
        default="train_only",
        help=(
            "`exact` emits one control per unique (train,val) budget. "
            "`train_only` emits one control per unique train count and uses the modal val count for that train count."
        ),
    )
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-pair", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_single_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("evaluation_status") == "completed"]


def completed_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("option_a_status") == "completed"]


def modal_value(values: list[int]) -> int:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def summarize_budgets(rows: list[dict[str, str]], grouping: str) -> list[dict[str, object]]:
    if grouping == "exact":
        grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(int(row["number_of_train_images"]), int(row["number_of_val_images"]))].append(row)
        budget_rows: list[dict[str, object]] = []
        for (train_count, val_count), group_rows in sorted(grouped.items()):
            budget_rows.append(
                {
                    "train_count": train_count,
                    "val_count": val_count,
                    "reference_mAP50-95": sum(float(row["mAP50-95"]) for row in group_rows) / len(group_rows),
                    "reference_mAP50": sum(float(row["mAP50"]) for row in group_rows) / len(group_rows),
                    "reference_F1": sum(float(row["F1"]) for row in group_rows) / len(group_rows),
                    "source_count": len(group_rows),
                }
            )
        return budget_rows

    grouped_train: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_train[int(row["number_of_train_images"])].append(row)

    budget_rows = []
    for train_count, group_rows in sorted(grouped_train.items()):
        val_count = modal_value([int(row["number_of_val_images"]) for row in group_rows])
        budget_rows.append(
            {
                "train_count": train_count,
                "val_count": val_count,
                "reference_mAP50-95": sum(float(row["mAP50-95"]) for row in group_rows) / len(group_rows),
                "reference_mAP50": sum(float(row["mAP50"]) for row in group_rows) / len(group_rows),
                "reference_F1": sum(float(row["F1"]) for row in group_rows) / len(group_rows),
                "source_count": len(group_rows),
            }
        )
    return budget_rows


def budget_control_id(prefix: str, train_count: int, val_count: int, seed: int) -> str:
    return f"mc_{prefix}_t{train_count:03d}_v{val_count:03d}_s{seed:02d}"


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    ensure_control_experiment_root(experiment_root)

    manifest_rows: list[dict[str, object]] = []
    next_index = 1

    if not args.skip_single:
        single_rows = completed_single_rows(read_rows(Path(args.single_results_csv).resolve()))
        for budget in summarize_budgets(single_rows, args.grouping):
            for seed in args.seeds:
                manifest_rows.append(
                    {
                        "control_index": next_index,
                        "control_id": budget_control_id("single_budget", int(budget["train_count"]), int(budget["val_count"]), seed),
                        "label": f"M4 matched to single-view budget {int(budget['train_count'])}/{int(budget['val_count'])} (seed {seed})",
                        "source_group": "single_budget",
                        "source_id": f"single_budget_t{int(budget['train_count'])}_v{int(budget['val_count'])}",
                        "source_label": (
                            f"Single-view budget: train={int(budget['train_count'])}, val={int(budget['val_count'])}, "
                            f"n={int(budget['source_count'])}"
                        ),
                        "train_count": int(budget["train_count"]),
                        "val_count": int(budget["val_count"]),
                        "seed": seed,
                        "sampling_strategy": "stratified_viewpoint",
                        "reference_mAP50-95": float(budget["reference_mAP50-95"]),
                        "reference_mAP50": float(budget["reference_mAP50"]),
                        "reference_F1": float(budget["reference_F1"]),
                    }
                )
                next_index += 1

    if not args.skip_pair:
        pair_rows = completed_pair_rows(read_rows(Path(args.pair_results_csv).resolve()))
        for budget in summarize_budgets(pair_rows, args.grouping):
            for seed in args.seeds:
                manifest_rows.append(
                    {
                        "control_index": next_index,
                        "control_id": budget_control_id("pair_budget", int(budget["train_count"]), int(budget["val_count"]), seed),
                        "label": f"M4 matched to pair-view budget {int(budget['train_count'])}/{int(budget['val_count'])} (seed {seed})",
                        "source_group": "pair_budget",
                        "source_id": f"pair_budget_t{int(budget['train_count'])}_v{int(budget['val_count'])}",
                        "source_label": (
                            f"Pair-view budget: train={int(budget['train_count'])}, val={int(budget['val_count'])}, "
                            f"n={int(budget['source_count'])}"
                        ),
                        "train_count": int(budget["train_count"]),
                        "val_count": int(budget["val_count"]),
                        "seed": seed,
                        "sampling_strategy": "stratified_viewpoint",
                        "reference_mAP50-95": float(budget["reference_mAP50-95"]),
                        "reference_mAP50": float(budget["reference_mAP50"]),
                        "reference_F1": float(budget["reference_F1"]),
                    }
                )
                next_index += 1

    if not manifest_rows:
        raise SystemExit("No budget controls were requested.")

    manifest_path = control_manifest_path(experiment_root)
    write_csv_rows(
        manifest_path,
        [
            "control_index",
            "control_id",
            "label",
            "source_group",
            "source_id",
            "source_label",
            "train_count",
            "val_count",
            "seed",
            "sampling_strategy",
            "reference_mAP50-95",
            "reference_mAP50",
            "reference_F1",
        ],
        manifest_rows,
    )

    print(f"Wrote budget-control manifest: {manifest_path}")
    print(f"Jobs: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
