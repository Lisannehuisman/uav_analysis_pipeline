from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "m4_pair_partial_analysis" / "data" / "current_snapshot"
REPORTS_DIR = SNAPSHOT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "m4_pair_partial_analysis" / "outputs"


@dataclass(frozen=True)
class CompletedRow:
    pair_id: str
    viewpoint_1: str
    viewpoint_2: str
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    delta_map50_95_vs_full_m4: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def completed_rows(rows: list[dict[str, str]]) -> list[CompletedRow]:
    completed: list[CompletedRow] = []
    for row in rows:
        if row.get("option_a_status") != "completed":
            continue
        map50_95 = float_or_nan(row.get("mAP50-95", ""))
        if math.isnan(map50_95):
            continue
        completed.append(
            CompletedRow(
                pair_id=row["pair_id"],
                viewpoint_1=row["viewpoint_1"],
                viewpoint_2=row["viewpoint_2"],
                precision=float_or_nan(row.get("precision", "")),
                recall=float_or_nan(row.get("recall", "")),
                f1=float_or_nan(row.get("F1", "")),
                map50=float_or_nan(row.get("mAP50", "")),
                map50_95=map50_95,
                delta_map50_95_vs_full_m4=float_or_nan(row.get("delta_mAP50-95_vs_full_M4", "")),
            )
        )
    return completed


def parse_viewpoint(viewpoint: str) -> tuple[str, str, int]:
    elevation, radius, azimuth = viewpoint.split("-")
    return elevation, radius, int(azimuth.replace("az", ""))


def viewpoint_stats(rows: list[CompletedRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[CompletedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.viewpoint_1].append(row)
        grouped[row.viewpoint_2].append(row)

    stats: list[dict[str, object]] = []
    for viewpoint, matches in grouped.items():
        best = max(matches, key=lambda item: item.map50_95)
        elevation, radius, azimuth = parse_viewpoint(viewpoint)
        stats.append(
            {
                "viewpoint": viewpoint,
                "elevation": elevation,
                "radius": radius,
                "azimuth": azimuth,
                "completed_pair_count": len(matches),
                "avg_mAP50-95": mean(item.map50_95 for item in matches),
                "avg_mAP50": mean(item.map50 for item in matches),
                "avg_F1": mean(item.f1 for item in matches),
                "best_pair_id_with_viewpoint": best.pair_id,
                "best_pair_mAP50-95_with_viewpoint": best.map50_95,
            }
        )
    stats.sort(key=lambda item: (-float(item["avg_mAP50-95"]), str(item["viewpoint"])))
    return stats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_factor_patterns(rows: list[CompletedRow], top_k: int = 25) -> dict[str, object]:
    top_rows = sorted(rows, key=lambda item: item.map50_95, reverse=True)[: min(top_k, len(rows))]

    elevation_counter = Counter()
    radius_counter = Counter()
    viewpoint_counter = Counter()
    for row in top_rows:
        for viewpoint in (row.viewpoint_1, row.viewpoint_2):
            elevation, radius, _ = parse_viewpoint(viewpoint)
            elevation_counter[elevation] += 1
            radius_counter[radius] += 1
            viewpoint_counter[viewpoint] += 1

    return {
        "top_k": len(top_rows),
        "top_elevations": elevation_counter.most_common(),
        "top_radii": radius_counter.most_common(),
        "top_viewpoints": viewpoint_counter.most_common(10),
    }


def rows_to_dicts(rows: list[CompletedRow]) -> list[dict[str, object]]:
    return [
        {
            "pair_id": row.pair_id,
            "viewpoint_1": row.viewpoint_1,
            "viewpoint_2": row.viewpoint_2,
            "precision": row.precision,
            "recall": row.recall,
            "F1": row.f1,
            "mAP50": row.map50,
            "mAP50-95": row.map50_95,
            "delta_mAP50-95_vs_full_M4": row.delta_map50_95_vs_full_m4,
        }
        for row in rows
    ]


def main() -> None:
    master_csv = REPORTS_DIR / "master_results.csv"
    if not master_csv.exists():
        raise SystemExit(
            f"Could not find synced Ponyland report at {master_csv}. "
            "Run sync_from_ponyland.ps1 first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = read_csv_rows(master_csv)
    completed = completed_rows(all_rows)

    total_pairs = len(all_rows)
    completed_count = len(completed)
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
    pending_count = total_pairs - completed_count - failed_count
    sweep_finished = pending_count == 0

    completed_sorted = sorted(completed, key=lambda item: item.map50_95, reverse=True)
    top_25 = completed_sorted[:25]
    top_10 = completed_sorted[:10]
    best_map = completed_sorted[0] if completed_sorted else None
    best_f1 = max(completed, key=lambda item: item.f1) if completed else None
    best_map50 = max(completed, key=lambda item: item.map50) if completed else None

    viewpoint_summary = viewpoint_stats(completed)
    factor_summary = summarize_factor_patterns(completed, top_k=25) if completed else {}

    if completed:
        write_csv(OUTPUT_DIR / "completed_pairs_current_snapshot.csv", rows_to_dicts(completed_sorted))
        write_csv(OUTPUT_DIR / "top_25_pairs_by_map50_95.csv", rows_to_dicts(top_25))
        write_csv(OUTPUT_DIR / "viewpoint_pair_contribution_scores.csv", viewpoint_summary)

    title = "# Final M4 Pair-Sweep Summary" if sweep_finished else "# Current Partial M4 Pair-Sweep Summary"

    lines = [
        title,
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Source snapshot: `{SNAPSHOT_ROOT}`",
        "",
        "## Sweep Status",
        "",
        f"- Total defined pairs: {total_pairs}",
        f"- Completed Option A evaluations: {completed_count}",
        f"- Failed pairs so far: {failed_count}",
        f"- Remaining pending / incomplete pairs: {pending_count}",
        f"- Completion rate: {(completed_count / total_pairs * 100):.1f}%",
    ]

    if sweep_finished and failed_count == 0:
        lines.append("- Sweep status: all defined pairs completed successfully.")
    elif sweep_finished:
        lines.append(
            f"- Sweep status: execution finished, but {failed_count} pairs failed and are excluded from metric-based rankings."
        )
    else:
        lines.append("- Sweep status: still in progress.")

    if best_map is not None:
        lines.extend(
            [
                "",
                "## Best Current Duo Viewpoints",
                "",
                f"- Best by mAP50-95: `{best_map.pair_id}` = `{best_map.viewpoint_1}` + `{best_map.viewpoint_2}`",
                f"- Scores: mAP50-95 `{best_map.map50_95:.4f}`, mAP50 `{best_map.map50:.4f}`, F1 `{best_map.f1:.4f}`, precision `{best_map.precision:.4f}`, recall `{best_map.recall:.4f}`",
                f"- Delta vs full M4 baseline (mAP50-95): `{best_map.delta_map50_95_vs_full_m4:.4f}`",
            ]
        )

        if sweep_finished and failed_count == 0:
            lines.append("- This is now the best-performing pair in the completed full sweep.")
        elif sweep_finished:
            lines.append("- This is the best-performing pair among all successfully evaluated pairs.")

    if best_map50 is not None and best_map50.pair_id != best_map.pair_id:
        lines.append(f"- Best by mAP50: `{best_map50.pair_id}` ({best_map50.map50:.4f})")
    if best_f1 is not None and best_f1.pair_id != best_map.pair_id:
        lines.append(f"- Best by F1: `{best_f1.pair_id}` ({best_f1.f1:.4f})")

    if top_10:
        lines.extend(["", "## Top 10 Completed Pairs So Far", ""])
        for idx, row in enumerate(top_10, start=1):
            lines.append(
                f"{idx}. `{row.pair_id}`: `{row.viewpoint_1}` + `{row.viewpoint_2}` | mAP50-95 `{row.map50_95:.4f}` | mAP50 `{row.map50:.4f}` | F1 `{row.f1:.4f}`"
            )

    if viewpoint_summary:
        lines.extend(["", "## Current Strongest Individual Viewpoints", ""])
        for idx, row in enumerate(viewpoint_summary[:10], start=1):
            lines.append(
                f"{idx}. `{row['viewpoint']}` | avg mAP50-95 across completed pairs `{float(row['avg_mAP50-95']):.4f}` | completed pair count `{row['completed_pair_count']}` | best pair `{row['best_pair_id_with_viewpoint']}`"
            )

    if factor_summary:
        lines.extend(
            [
                "",
                "## What The Current Duo Viewpoints Seem To Suggest",
                "",
                f"- Based on the current top {factor_summary['top_k']} completed pairs.",
                f"- Most common elevation labels in those top pairs: {factor_summary['top_elevations']}",
                f"- Most common radius labels in those top pairs: {factor_summary['top_radii']}",
                f"- Most frequently recurring viewpoints in those top pairs: {factor_summary['top_viewpoints']}",
            ]
        )
        if sweep_finished and failed_count == 0:
            lines.append("- These patterns now reflect the completed full sweep.")
        elif sweep_finished:
            lines.append("- These patterns reflect all successful pair evaluations, with failed pairs excluded.")
        else:
            lines.append("- These patterns are still provisional because the full 2556-pair sweep is not finished yet.")

    summary_path = OUTPUT_DIR / "current_partial_results_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    if completed:
        print(f"Wrote completed pairs CSV: {OUTPUT_DIR / 'completed_pairs_current_snapshot.csv'}")
        print(f"Wrote top-25 CSV: {OUTPUT_DIR / 'top_25_pairs_by_map50_95.csv'}")
        print(f"Wrote viewpoint contribution CSV: {OUTPUT_DIR / 'viewpoint_pair_contribution_scores.csv'}")


if __name__ == "__main__":
    main()
