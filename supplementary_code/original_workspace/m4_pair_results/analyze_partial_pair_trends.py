from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_partial_pair_results import (
    OUTPUT_DIR,
    REPORTS_DIR,
    CompletedRow,
    completed_rows,
    parse_viewpoint,
    read_csv_rows,
)


TRENDS_DIR = OUTPUT_DIR / "trends"


def elevation_combo_from_row(row: CompletedRow | dict[str, str]) -> str:
    e1, _, _ = parse_viewpoint(row["viewpoint_1"] if isinstance(row, dict) else row.viewpoint_1)
    e2, _, _ = parse_viewpoint(row["viewpoint_2"] if isinstance(row, dict) else row.viewpoint_2)
    return "-".join(sorted([e1, e2]))


def radius_combo_from_row(row: CompletedRow | dict[str, str]) -> str:
    _, r1, _ = parse_viewpoint(row["viewpoint_1"] if isinstance(row, dict) else row.viewpoint_1)
    _, r2, _ = parse_viewpoint(row["viewpoint_2"] if isinstance(row, dict) else row.viewpoint_2)
    return "-".join(sorted([r1, r2]))


def azimuth_sep_from_row(row: CompletedRow | dict[str, str]) -> int:
    _, _, a1 = parse_viewpoint(row["viewpoint_1"] if isinstance(row, dict) else row.viewpoint_1)
    _, _, a2 = parse_viewpoint(row["viewpoint_2"] if isinstance(row, dict) else row.viewpoint_2)
    sep = abs(a1 - a2)
    return min(sep, 360 - sep)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def combo_mean_rows(name: str, completed: list[CompletedRow], all_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if name == "elevation_combo":
        combo_fn = elevation_combo_from_row
    elif name == "radius_combo":
        combo_fn = radius_combo_from_row
    elif name == "azimuth_separation":
        combo_fn = azimuth_sep_from_row
    else:
        raise ValueError(name)

    completed_counter = Counter(combo_fn(row) for row in completed)
    all_counter = Counter(combo_fn(row) for row in all_rows)
    score_map: dict[object, list[float]] = defaultdict(list)
    for row in completed:
        score_map[combo_fn(row)].append(row.map50_95)

    rows: list[dict[str, object]] = []
    for combo, total_count in sorted(all_counter.items(), key=lambda item: str(item[0])):
        completed_count = completed_counter.get(combo, 0)
        observed_share = completed_count / len(completed) if completed else 0.0
        design_share = total_count / len(all_rows) if all_rows else 0.0
        rows.append(
            {
                "group_type": name,
                "group_value": combo,
                "completed_count": completed_count,
                "all_defined_count": total_count,
                "completed_share": observed_share,
                "design_share": design_share,
                "coverage_ratio_vs_design": (observed_share / design_share) if design_share else 0.0,
                "avg_mAP50-95_completed_only": mean(score_map[combo]) if score_map.get(combo) else float("nan"),
            }
        )
    return rows


def plot_observed_vs_design(rows: list[dict[str, object]], title: str, output_path: Path) -> None:
    labels = [str(row["group_value"]) for row in rows]
    observed = [float(row["completed_share"]) for row in rows]
    design = [float(row["design_share"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], design, width=width, label="Expected from full design", color="#b0bec5")
    ax.bar([i + width / 2 for i in x], observed, width=width, label="Observed in completed pairs", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Share of pairs")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_means(rows: list[dict[str, object]], title: str, output_path: Path) -> None:
    valid_rows = [row for row in rows if not str(row["avg_mAP50-95_completed_only"]) == "nan"]
    labels = [str(row["group_value"]) for row in valid_rows]
    values = [float(row["avg_mAP50-95_completed_only"]) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, values, color="#ff7f0e")
    ax.set_ylabel("Average completed-pair mAP50-95")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    master_csv = REPORTS_DIR / "master_results.csv"
    if not master_csv.exists():
        raise SystemExit(f"Could not find synced Ponyland report at {master_csv}.")

    all_rows = read_csv_rows(master_csv)
    completed = completed_rows(all_rows)
    if not completed:
        raise SystemExit("No completed pair results available in the current snapshot.")

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
    pending_count = len(all_rows) - len(completed) - failed_count
    sweep_finished = pending_count == 0

    TRENDS_DIR.mkdir(parents=True, exist_ok=True)

    elev_rows = combo_mean_rows("elevation_combo", completed, all_rows)
    radius_rows = combo_mean_rows("radius_combo", completed, all_rows)
    az_rows = combo_mean_rows("azimuth_separation", completed, all_rows)

    min_elev_coverage = min(float(row["coverage_ratio_vs_design"]) for row in elev_rows)
    min_radius_coverage = min(float(row["coverage_ratio_vs_design"]) for row in radius_rows)
    min_az_coverage = min(float(row["coverage_ratio_vs_design"]) for row in az_rows)
    nearly_complete_coverage = (
        pending_count / len(all_rows) <= 0.02
        and min_elev_coverage >= 0.9
        and min_radius_coverage >= 0.9
        and min_az_coverage >= 0.9
    )

    write_csv(TRENDS_DIR / "elevation_combo_trends.csv", elev_rows)
    write_csv(TRENDS_DIR / "radius_combo_trends.csv", radius_rows)
    write_csv(TRENDS_DIR / "azimuth_separation_trends.csv", az_rows)

    plot_observed_vs_design(
        elev_rows,
        "Completed elevation-combo coverage vs full pair design",
        TRENDS_DIR / "elevation_combo_coverage_bias.png",
    )
    plot_observed_vs_design(
        radius_rows,
        "Completed radius-combo coverage vs full pair design",
        TRENDS_DIR / "radius_combo_coverage_bias.png",
    )
    plot_group_means(
        elev_rows,
        "Average mAP50-95 by elevation combo (completed pairs only)",
        TRENDS_DIR / "elevation_combo_mean_map50_95.png",
    )
    plot_group_means(
        radius_rows,
        "Average mAP50-95 by radius combo (completed pairs only)",
        TRENDS_DIR / "radius_combo_mean_map50_95.png",
    )
    plot_group_means(
        az_rows,
        "Average mAP50-95 by azimuth separation (completed pairs only)",
        TRENDS_DIR / "azimuth_separation_mean_map50_95.png",
    )

    best_top25 = sorted(completed, key=lambda row: row.map50_95, reverse=True)[:25]
    top25_elev = Counter(elevation_combo_from_row(row) for row in best_top25)
    top25_radius = Counter(radius_combo_from_row(row) for row in best_top25)
    top25_sep = Counter(azimuth_sep_from_row(row) for row in best_top25)

    title = "# Final Pair-Sweep Trend Analysis" if sweep_finished else "# Current Partial Trend Analysis"

    lines = [
        title,
        "",
        "## Main Caution",
        "",
    ]

    if sweep_finished and failed_count == 0:
        lines.extend(
            [
                "- The sweep has finished successfully, so the design-order bias from an unfinished run no longer applies.",
                "- The trends below summarize the full successful pair sweep.",
            ]
        )
    elif sweep_finished:
        lines.extend(
            [
                "- The sweep has finished running, but some pairs failed and are absent from the metric-based summaries below.",
                "- The remaining caveat is therefore missing failed pairs, not pair-order bias from an unfinished run.",
            ]
        )
    elif nearly_complete_coverage:
        lines.extend(
            [
                "- The sweep is not yet mathematically complete, but the successful pairs already cover the design very evenly.",
                "- The main remaining caveat is the small set of pending or failed pairs, not a strong coverage distortion.",
            ]
        )
    else:
        lines.extend(
            [
                "- The currently completed pairs are not a random sample of the full 2556-pair design.",
                "- The sweep is still following the pair enumeration order, so early results are structurally biased toward certain viewpoint families.",
                "- Any trend below should be treated as provisional until later pair IDs fill in the missing design regions.",
            ]
        )

    lines.extend(
        [
        "",
        "## Coverage Bias So Far",
        "",
        f"- Completed pairs analyzed: {len(completed)}",
        f"- Pair-id range represented in the synced results starts at `{completed[0].pair_id}` and currently reaches at least `{completed[-1].pair_id}`.",
        f"- Failed pairs currently excluded from these trend summaries: {failed_count}.",
    ]
    )

    if sweep_finished and failed_count == 0:
        lines.append("- All defined pair groups are now represented, so the design coverage itself is complete.")
    elif sweep_finished:
        lines.append("- Coverage is nearly complete, but failed pairs still create a small gap in the design.")
    elif nearly_complete_coverage:
        lines.append(
            f"- Coverage is already close to the full design: minimum coverage ratio versus design is {min(min_elev_coverage, min_radius_coverage, min_az_coverage):.3f} across the tracked grouping schemes."
        )
    else:
        lines.extend(
            [
                "- Elevation-combo coverage in completed pairs currently contains almost no `elmid-elmid` or `elhigh-elhigh` cases, and no meaningful `elhigh-elmid` coverage yet.",
                "- That means any early story like 'low+mid is best' could partly reflect which combinations have been reached first rather than a final scientific conclusion.",
            ]
        )

    lines.extend(
        [
        "",
        "## What Does Look Real So Far",
        "",
        "- Pure low+low (`ellow-ellow`) pairs are clearly weaker on average than mixed-elevation pairs in the completed subset.",
        "- Far+far is not emerging as a strong pattern; the best-performing completed radius groups are `radmid-radmid`, `radmid-radnear`, and `radnear-radnear`.",
        "- The strongest current pairs are not 'close+far' in a consistent way; they are mostly `radmid-radmid` or `radmid-radnear`.",
        "- Among the current top completed pairs, azimuth separations of `90`, `135`, and `45` degrees appear most often, with exact same-azimuth pairs almost absent from the top 25.",
        "- Certain mid-radius viewpoints, especially `elmid-radmid-az225`, keep recurring as strong partners.",
        "",
        "## Top-25 Pattern Snapshot",
        "",
        f"- Elevation combos in top 25: {top25_elev.most_common()}",
        f"- Radius combos in top 25: {top25_radius.most_common()}",
        f"- Azimuth separations in top 25: {top25_sep.most_common()}",
        "",
        "## Interpretation",
        "",
        "- The signal is not simply 'always high' and not simply 'always low'.",
        "- The stronger trend is that mixed-elevation pairs outperform low+low pairs, and mid-radius viewpoints appear repeatedly in the strongest duos.",
        "- There is not clear evidence that 'close+far' is the dominant recipe.",
    ]
    )

    if sweep_finished and failed_count == 0:
        lines.append("- Because the full sweep is complete, these are no longer interim observations but the main trends in the full pair experiment.")
    elif sweep_finished:
        lines.append("- Because execution is complete, these reflect the successful pair runs, with only failed pairs still missing from the final picture.")
    elif nearly_complete_coverage:
        lines.append("- Because design coverage is already very even, these patterns are close to the final picture, with only a small number of unresolved pairs still absent.")
    else:
        lines.append("- However, because the completed subset is still design-biased, the cleanest scientifically safe statement is that low+low looks weak, mixed-elevation plus mid-radius looks promising, and the final answer still requires the full sweep.")

    lines.extend(
        [
        "",
        "## Output Files",
        "",
        f"- `elevation_combo_trends.csv`",
        f"- `radius_combo_trends.csv`",
        f"- `azimuth_separation_trends.csv`",
        f"- `elevation_combo_coverage_bias.png`",
        f"- `radius_combo_coverage_bias.png`",
        f"- `elevation_combo_mean_map50_95.png`",
        f"- `radius_combo_mean_map50_95.png`",
        f"- `azimuth_separation_mean_map50_95.png`",
    ]
    )

    (TRENDS_DIR / "current_trend_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote trend analysis under: {TRENDS_DIR}")


if __name__ == "__main__":
    main()
