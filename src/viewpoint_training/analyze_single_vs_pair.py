from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = PROJECT_ROOT / "results" / "tables"
SINGLE_CSV = TABLES / "single_view_sweep_master_results.csv"
PAIR_CSV = TABLES / "pair_view_sweep_master_results.csv"
MATCHED_CONTROL_CSV = TABLES / "equal_budget_control_table.csv"
OUTPUTS = PROJECT_ROOT / "results" / "recomputed" / "single_vs_pair"
PLOTS = OUTPUTS / "plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def short_label(viewpoint: str) -> str:
    parts = viewpoint.split("-")
    elevation = parts[0].replace("ellow", "L").replace("elmid", "M").replace("elhigh", "H")
    radius = parts[1].replace("radnear", "N").replace("radmid", "M").replace("radfar", "F")
    azimuth = parts[2].replace("az", "")
    return f"{elevation}-{radius}-{azimuth}"


def elevation_combo(a: str, b: str) -> str:
    first = a.split("-")[0]
    second = b.split("-")[0]
    return "-".join(sorted((first, second)))


def load_data():
    single_rows = [row for row in read_csv(SINGLE_CSV) if row.get("evaluation_status") == "completed"]
    pair_rows = [row for row in read_csv(PAIR_CSV) if row.get("option_a_status") == "completed"]

    singles_by_viewpoint = {row["viewpoint"]: row for row in single_rows}
    singles_sorted = sorted(single_rows, key=lambda row: float(row["mAP50-95"]), reverse=True)
    single_rank = {row["viewpoint"]: index for index, row in enumerate(singles_sorted, start=1)}

    return single_rows, pair_rows, singles_by_viewpoint, single_rank


def load_matched_controls() -> list[dict[str, str]]:
    if not MATCHED_CONTROL_CSV.exists():
        return []
    rows = read_csv(MATCHED_CONTROL_CSV)
    return [row for row in rows if row.get("evaluation_status") == "completed"]


def build_budget_control_lookup(control_rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in control_rows:
        grouped[(int(row["number_of_train_images"]), int(row["number_of_val_images"]))].append(row)

    summary: dict[tuple[int, int], dict[str, object]] = {}
    for key, rows in grouped.items():
        summary[key] = {
            "count": len(rows),
            "mean_map50_95": float(np.mean([float(row["mAP50-95"]) for row in rows])),
            "mean_map50": float(np.mean([float(row["mAP50"]) for row in rows])),
            "mean_f1": float(np.mean([float(row["F1"]) for row in rows])),
            "control_ids": ",".join(sorted(str(row["control_id"]) for row in rows)),
        }
    return summary


def build_enriched_pair_rows(pair_rows, singles_by_viewpoint, single_rank, budget_control_lookup):
    enriched: list[dict[str, object]] = []
    for row in pair_rows:
        first = singles_by_viewpoint[row["viewpoint_1"]]
        second = singles_by_viewpoint[row["viewpoint_2"]]
        pair_map = float(row["mAP50-95"])
        first_map = float(first["mAP50-95"])
        second_map = float(second["mAP50-95"])
        best_single = max(first_map, second_map)
        mean_single = (first_map + second_map) / 2.0
        worst_single = min(first_map, second_map)
        budget_key = (int(row["number_of_train_images"]), int(row["number_of_val_images"]))
        budget_control = budget_control_lookup.get(budget_key)

        enriched.append(
            {
                "pair_id": row["pair_id"],
                "viewpoint_1": row["viewpoint_1"],
                "viewpoint_2": row["viewpoint_2"],
                "pair_map50_95": pair_map,
                "pair_map50": float(row["mAP50"]),
                "pair_f1": float(row["F1"]),
                "single_1_map50_95": first_map,
                "single_2_map50_95": second_map,
                "single_1_rank": single_rank[row["viewpoint_1"]],
                "single_2_rank": single_rank[row["viewpoint_2"]],
                "best_constituent_single_map50_95": best_single,
                "mean_constituent_single_map50_95": mean_single,
                "worst_constituent_single_map50_95": worst_single,
                "pair_minus_best_single": pair_map - best_single,
                "pair_minus_mean_single": pair_map - mean_single,
                "pair_minus_worst_single": pair_map - worst_single,
                "number_of_train_images": budget_key[0],
                "number_of_val_images": budget_key[1],
                "budget_control_map50_95": float(budget_control["mean_map50_95"]) if budget_control else float("nan"),
                "pair_minus_budget_control": (pair_map - float(budget_control["mean_map50_95"])) if budget_control else float("nan"),
                "budget_control_run_count": int(budget_control["count"]) if budget_control else 0,
                "budget_control_ids": str(budget_control["control_ids"]) if budget_control else "",
                "beats_best_single": pair_map > best_single,
                "beats_mean_single": pair_map > mean_single,
                "elevation_combo": elevation_combo(row["viewpoint_1"], row["viewpoint_2"]),
            }
        )
    return enriched


def build_viewpoint_summary(single_rows, enriched_pairs):
    pair_values_by_viewpoint: dict[str, list[float]] = defaultdict(list)
    pair_lifts_by_viewpoint: dict[str, list[float]] = defaultdict(list)
    best_pair_by_viewpoint: dict[str, float] = defaultdict(lambda: float("-inf"))

    for row in enriched_pairs:
        pair_map = float(row["pair_map50_95"])
        for key in ("viewpoint_1", "viewpoint_2"):
            viewpoint = str(row[key])
            pair_values_by_viewpoint[viewpoint].append(pair_map)
            pair_lifts_by_viewpoint[viewpoint].append(pair_map - float(row[f"single_{1 if key == 'viewpoint_1' else 2}_map50_95"]))
            best_pair_by_viewpoint[viewpoint] = max(best_pair_by_viewpoint[viewpoint], pair_map)

    summary_rows: list[dict[str, object]] = []
    for row in single_rows:
        viewpoint = row["viewpoint"]
        single_map = float(row["mAP50-95"])
        pair_values = pair_values_by_viewpoint.get(viewpoint, [])
        pair_lifts = pair_lifts_by_viewpoint.get(viewpoint, [])
        mean_pair_map = float(sum(pair_values) / len(pair_values)) if pair_values else float("nan")
        mean_lift = float(sum(pair_lifts) / len(pair_lifts)) if pair_lifts else float("nan")
        best_lift = float(best_pair_by_viewpoint[viewpoint] - single_map) if pair_values else float("nan")
        summary_rows.append(
            {
                "viewpoint": viewpoint,
                "single_map50_95": single_map,
                "single_map50": float(row["mAP50"]),
                "single_f1": float(row["F1"]),
                "mean_pair_map50_95_when_included": mean_pair_map,
                "best_pair_map50_95_when_included": best_pair_by_viewpoint[viewpoint] if pair_values else float("nan"),
                "mean_lift_when_paired": mean_lift,
                "best_lift_when_paired": best_lift,
                "completed_pair_count": len(pair_values),
            }
        )
    return summary_rows


def plot_pair_vs_best_single(enriched_pairs: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.array([float(row["best_constituent_single_map50_95"]) for row in enriched_pairs])
    y = np.array([float(row["pair_map50_95"]) for row in enriched_pairs])
    c = np.array([float(row["pair_minus_best_single"]) for row in enriched_pairs])

    scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.65, s=22, edgecolors="none")
    low = min(float(x.min()), float(y.min()))
    high = max(float(x.max()), float(y.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#d62728", linewidth=1.5)
    ax.set_xlabel("Best constituent single-view mAP50-95")
    ax.set_ylabel("Pair-trained mAP50-95")
    ax.set_title("Pair performance vs best constituent single-view baseline")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pair minus best single")
    fig.tight_layout()
    fig.savefig(PLOTS / "pair_vs_best_single_scatter.png", dpi=180)
    plt.close(fig)


def plot_regime_distribution_comparison(single_rows: list[dict[str, str]], pair_rows: list[dict[str, str]]) -> None:
    single_values = np.array([float(row["mAP50-95"]) for row in single_rows], dtype=float)
    pair_values = np.array([float(row["mAP50-95"]) for row in pair_rows], dtype=float)

    datasets = [single_values, pair_values]
    positions = [1, 0]
    labels = [f"Single-view sweep (n={len(single_values)})", f"Pair-view sweep (n={len(pair_values)})"]
    colors = ["#4c78a8", "#f58518"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    violins = ax.violinplot(
        datasets,
        positions=positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)

    boxplot = ax.boxplot(
        datasets,
        positions=positions,
        vert=False,
        widths=0.2,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#2f2f2f", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.6)

    for index, whisker in enumerate(boxplot["whiskers"]):
        whisker.set_color(colors[index // 2])
    for index, cap in enumerate(boxplot["caps"]):
        cap.set_color(colors[index // 2])

    low = min(float(single_values.min()), float(pair_values.min())) - 0.015
    high = max(float(single_values.max()), float(pair_values.max())) + 0.08
    text_x = max(float(single_values.max()), float(pair_values.max())) + 0.01

    for ypos, values, color in zip(positions, datasets, colors):
        mean_value = float(np.mean(values))
        best_value = float(np.max(values))

        ax.scatter(mean_value, ypos, marker="o", s=78, color=color, edgecolors="white", linewidth=0.9, zorder=4)
        ax.scatter(best_value, ypos, marker="D", s=72, color=color, edgecolors="white", linewidth=0.9, zorder=4)
        ax.text(
            text_x,
            ypos,
            f"mean {mean_value:.3f} | best {best_value:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color=color,
        )

    ax.set_xlim(low, high)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mAP50-95 on the full M4 test split")
    ax.set_title("Single-view vs pair-view sweeps on the same test metric", fontsize=17)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    mean_gap = float(np.mean(pair_values) - np.mean(single_values))
    best_gap = float(np.max(pair_values) - np.max(single_values))
    ax.text(
        0.02,
        0.03,
        f"Mean pair - mean single: {mean_gap:+.3f} | best pair - best single: {best_gap:+.3f}",
        transform=ax.transAxes,
        fontsize=12,
        color="#555555",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#6b6b6b", markeredgecolor="white", markersize=8, label="Mean"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#6b6b6b", markeredgecolor="white", markersize=8, label="Best"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(PLOTS / "single_vs_pair_regime_distribution.png", dpi=180)
    plt.close(fig)


def plot_lift_histogram(enriched_pairs: list[dict[str, object]]) -> None:
    values = np.array([float(row["pair_minus_best_single"]) for row in enriched_pairs], dtype=float)
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    positive_share = 100.0 * float(np.mean(values > 0.0))

    bins = np.linspace(float(values.min()), float(values.max()), 37)
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    colors = ["#d95f02" if center <= 0.0 else "#2ca02c" for center in centers]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.bar(edges[:-1], counts, width=widths, align="edge", color=colors, edgecolor="white", alpha=0.92)
    ax.axvline(0.0, linestyle="--", color="#444444", linewidth=1.5, label="Break-even")
    ax.axvline(mean_value, color="#1f77b4", linewidth=2.0, label=f"Mean {mean_value:+.3f}")
    ax.axvline(median_value, color="#9467bd", linewidth=2.0, label=f"Median {median_value:+.3f}")
    ax.set_xlabel("Pair minus best constituent single (mAP50-95)")
    ax.set_ylabel("Number of completed pairs")
    ax.set_title("Pair lift over the stronger single-view baseline", fontsize=17)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.text(
        0.03,
        0.95,
        f"{positive_share:.1f}% of pairs beat their better single-view constituent",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9d9d9"},
    )
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "pair_lift_vs_best_single_distribution.png", dpi=180)
    plt.close(fig)


def plot_top_synergy_pairs(enriched_pairs: list[dict[str, object]]) -> None:
    top = sorted(enriched_pairs, key=lambda row: float(row["pair_minus_best_single"]), reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(14, 8))
    labels = [
        f"{row['pair_id']}: {short_label(str(row['viewpoint_1']))} + {short_label(str(row['viewpoint_2']))}"
        for row in top
    ]
    values = [float(row["pair_minus_best_single"]) for row in top]
    ax.barh(labels[::-1], values[::-1], color="#2ca02c")
    ax.set_xlabel("Gain over best constituent single (mAP50-95)")
    ax.set_title("Top synergy pairs")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "top_synergy_pairs.png", dpi=180)
    plt.close(fig)


def plot_elevation_combo_lift(enriched_pairs: list[dict[str, object]]) -> None:
    order = ["ellow-ellow", "ellow-elmid", "ellow-elhigh", "elmid-elmid", "elhigh-elmid", "elhigh-elhigh"]
    combo_values: dict[str, list[float]] = defaultdict(list)
    for row in enriched_pairs:
        combo_values[str(row["elevation_combo"])].append(float(row["pair_minus_best_single"]))

    labels = [combo for combo in order if combo in combo_values]
    means = [sum(combo_values[label]) / len(combo_values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, means, color="#9467bd")
    ax.axhline(0.0, linestyle="--", color="#d62728", linewidth=1.5)
    ax.set_ylabel("Mean gain over best single (mAP50-95)")
    ax.set_title("Average pair lift by elevation combination")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "elevation_combo_lift_vs_best_single.png", dpi=180)
    plt.close(fig)


def plot_viewpoint_lift_summary(viewpoint_rows: list[dict[str, object]]) -> None:
    top = sorted(
        [row for row in viewpoint_rows if not math.isnan(float(row["mean_lift_when_paired"]))],
        key=lambda row: float(row["mean_lift_when_paired"]),
        reverse=True,
    )[:20]
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [short_label(str(row["viewpoint"])) for row in top]
    values = [float(row["mean_lift_when_paired"]) for row in top]
    ax.barh(labels[::-1], values[::-1], color="#ff7f0e")
    ax.set_xlabel("Mean pair lift over this viewpoint's own single-view model")
    ax.set_title("Viewpoints that benefit most from being paired")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "viewpoint_mean_lift_when_paired.png", dpi=180)
    plt.close(fig)


def plot_pair_lift_heatmap(enriched_pairs: list[dict[str, object]], single_rows: list[dict[str, str]], top_n: int = 18) -> None:
    top_viewpoints = [
        row["viewpoint"]
        for row in sorted(single_rows, key=lambda row: float(row["mAP50-95"]), reverse=True)[:top_n]
    ]
    matrix = np.full((len(top_viewpoints), len(top_viewpoints)), np.nan)
    index = {viewpoint: idx for idx, viewpoint in enumerate(top_viewpoints)}

    for row in enriched_pairs:
        first = str(row["viewpoint_1"])
        second = str(row["viewpoint_2"])
        if first not in index or second not in index:
            continue
        i = index[first]
        j = index[second]
        value = float(row["pair_minus_best_single"])
        matrix[i, j] = value
        matrix[j, i] = value

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-0.05, vmax=0.16)
    ax.set_xticks(range(len(top_viewpoints)))
    ax.set_yticks(range(len(top_viewpoints)))
    labels = [short_label(viewpoint) for viewpoint in top_viewpoints]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("Pair lift over best constituent single for top single-view viewpoints")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pair minus best constituent single (mAP50-95)")
    fig.tight_layout()
    fig.savefig(PLOTS / "pair_lift_heatmap_top_singles.png", dpi=180)
    plt.close(fig)


def plot_pair_rank_lift_map(enriched_pairs: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    better_rank = np.array(
        [min(int(row["single_1_rank"]), int(row["single_2_rank"])) for row in enriched_pairs],
        dtype=float,
    )
    worse_rank = np.array(
        [max(int(row["single_1_rank"]), int(row["single_2_rank"])) for row in enriched_pairs],
        dtype=float,
    )
    lift = np.array([float(row["pair_minus_best_single"]) for row in enriched_pairs], dtype=float)
    sizes = 20 + 220 * np.clip(lift + 0.05, 0.0, None)
    scatter = ax.scatter(
        better_rank,
        worse_rank,
        c=lift,
        s=sizes,
        cmap="viridis",
        alpha=0.55,
        edgecolors="none",
    )
    ax.set_xlabel("Better constituent single rank")
    ax.set_ylabel("Worse constituent single rank")
    ax.set_title("Where pair lift appears across constituent single-view ranks")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pair minus best constituent single (mAP50-95)")
    fig.tight_layout()
    fig.savefig(PLOTS / "pair_rank_lift_map.png", dpi=180)
    plt.close(fig)


def write_summary(single_rows, pair_rows, enriched_pairs, viewpoint_rows, control_rows) -> None:
    best_single = max(single_rows, key=lambda row: float(row["mAP50-95"]))
    best_pair = max(enriched_pairs, key=lambda row: float(row["pair_map50_95"]))
    synergy_top = sorted(enriched_pairs, key=lambda row: float(row["pair_minus_best_single"]), reverse=True)[:10]
    lift_values = [float(row["pair_minus_best_single"]) for row in enriched_pairs]
    positive_lift = sum(1 for value in lift_values if value > 0)

    best_single_map = float(best_single["mAP50-95"])
    best_pair_map = float(best_pair["pair_map50_95"])
    mean_single_train = sum(float(row["number_of_train_images"]) for row in single_rows) / len(single_rows)
    mean_pair_train = sum(float(row["number_of_train_images"]) for row in pair_rows) / len(pair_rows)

    lines = [
        "# Single vs Pair Training Comparison",
        "",
        "## Coverage",
        "",
        f"- Completed single-view models: {len(single_rows)} / 72",
        f"- Completed pair models: {len(pair_rows)} / 2556",
        "",
        "## Best Models",
        "",
        f"- Best single-view training viewpoint: `{best_single['single_id']}` = `{best_single['viewpoint']}` with `mAP50-95 = {best_single_map:.4f}`",
        f"- Best pair-trained model: `{best_pair['pair_id']}` = `{best_pair['viewpoint_1']}` + `{best_pair['viewpoint_2']}` with `mAP50-95 = {best_pair_map:.4f}`",
        f"- Best pair improvement over best single overall: `{best_pair_map - best_single_map:+.4f}` mAP50-95",
        "",
        "## Pair vs Single Baselines",
        "",
        f"- Pairs beating their better constituent single: `{positive_lift} / {len(enriched_pairs)}` ({100 * positive_lift / len(enriched_pairs):.1f}%)",
        f"- Mean pair lift over best constituent single: `{sum(lift_values) / len(lift_values):+.4f}` mAP50-95",
        f"- Median pair lift over best constituent single: `{float(np.median(np.array(lift_values))):+.4f}` mAP50-95",
        "",
        "## Data Volume Caveat",
        "",
        f"- Mean single-view training images: `{mean_single_train:.1f}`",
        f"- Mean pair-view training images: `{mean_pair_train:.1f}`",
        "- Pair models therefore usually see about twice as many training images as single-view models.",
        "- This comparison is still scientifically useful, but it reflects both viewpoint complementarity and added image count.",
        "",
        "## Recommended Fair Comparison",
        "",
        "- The cleanest fairness fix is **not** to duplicate single-view images. That would only repeat the same viewpoint evidence and would not create a genuinely stronger baseline.",
        "- Instead, compare each restricted model against an **equal-image-count M4 control** trained on the full viewpoint space with the same train/val image counts.",
        "- This isolates `image count` from `viewpoint restriction`: if the matched M4 control still wins, the gap is due to viewpoint diversity rather than just having fewer images.",
        "",
        "## Top Synergy Pairs",
        "",
    ]
    if control_rows:
        pair_budget_rows = [row for row in enriched_pairs if not math.isnan(float(row["pair_minus_budget_control"]))]
        if pair_budget_rows:
            pair_budget_gaps = [float(row["pair_minus_budget_control"]) for row in pair_budget_rows]
            lines.extend(
                [
                    "### Current matched-control status",
                    "",
                    f"- Completed matched-control runs available locally: `{len(control_rows)}`",
                    f"- Pair rows with an exact completed budget-matched M4 control: `{len(pair_budget_rows)} / {len(enriched_pairs)}`",
                    f"- Mean pair gap vs matched M4 control: `{float(np.mean(np.array(pair_budget_gaps))):+.4f}` mAP50-95",
                    f"- Median pair gap vs matched M4 control: `{float(np.median(np.array(pair_budget_gaps))):+.4f}` mAP50-95",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "### Current matched-control status",
                "",
                "- No completed matched-control runs were found yet under `outputs/m4_matched_control_experiment/reports/master_results.csv`.",
                "- The repo now includes tooling to generate those controls by budget before rerunning this comparison.",
                "",
            ]
        )

    lines.extend(
        [
            "### Practical recommendation",
            "",
            "- For the thesis headline, compare `best single` against its matched M4 control and `best pair` against its matched M4 control.",
            "- For the sweep-level story, keep the existing pair-vs-single plots, but frame them explicitly as `restricted-view vs restricted-view` comparisons rather than fully fair count-controlled comparisons.",
            "",
        ]
    )

    for row in synergy_top:
        lines.append(
            f"- `{row['pair_id']}`: `{row['viewpoint_1']}` + `{row['viewpoint_2']}` -> "
            f"`{float(row['pair_minus_best_single']):+.4f}` over the better constituent single"
        )

    top_view_lift = sorted(
        [row for row in viewpoint_rows if not math.isnan(float(row["mean_lift_when_paired"]))],
        key=lambda row: float(row["mean_lift_when_paired"]),
        reverse=True,
    )[:10]
    lines.extend(["", "## Viewpoints That Gain Most From Pairing", ""])
    for row in top_view_lift:
        lines.append(
            f"- `{row['viewpoint']}`: single `mAP50-95 = {float(row['single_map50_95']):.4f}`, "
            f"mean pair lift `{float(row['mean_lift_when_paired']):+.4f}`"
        )

    (OUTPUTS / "single_vs_pair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    single_rows, pair_rows, singles_by_viewpoint, single_rank = load_data()
    control_rows = load_matched_controls()
    budget_control_lookup = build_budget_control_lookup(control_rows)
    enriched_pairs = build_enriched_pair_rows(pair_rows, singles_by_viewpoint, single_rank, budget_control_lookup)
    viewpoint_rows = build_viewpoint_summary(single_rows, enriched_pairs)

    write_csv(
        OUTPUTS / "pair_vs_single_enriched.csv",
        [
            "pair_id",
            "viewpoint_1",
            "viewpoint_2",
            "pair_map50_95",
            "pair_map50",
            "pair_f1",
            "single_1_map50_95",
            "single_2_map50_95",
            "single_1_rank",
            "single_2_rank",
            "best_constituent_single_map50_95",
            "mean_constituent_single_map50_95",
            "worst_constituent_single_map50_95",
            "pair_minus_best_single",
            "pair_minus_mean_single",
            "pair_minus_worst_single",
            "number_of_train_images",
            "number_of_val_images",
            "budget_control_map50_95",
            "pair_minus_budget_control",
            "budget_control_run_count",
            "budget_control_ids",
            "beats_best_single",
            "beats_mean_single",
            "elevation_combo",
        ],
        enriched_pairs,
    )

    write_csv(
        OUTPUTS / "viewpoint_pairing_lift_summary.csv",
        [
            "viewpoint",
            "single_map50_95",
            "single_map50",
            "single_f1",
            "mean_pair_map50_95_when_included",
            "best_pair_map50_95_when_included",
            "mean_lift_when_paired",
            "best_lift_when_paired",
            "completed_pair_count",
        ],
        viewpoint_rows,
    )

    plot_pair_vs_best_single(enriched_pairs)
    plot_regime_distribution_comparison(single_rows, pair_rows)
    plot_lift_histogram(enriched_pairs)
    plot_top_synergy_pairs(enriched_pairs)
    plot_elevation_combo_lift(enriched_pairs)
    plot_viewpoint_lift_summary(viewpoint_rows)
    plot_pair_lift_heatmap(enriched_pairs, single_rows)
    plot_pair_rank_lift_map(enriched_pairs)
    write_summary(single_rows, pair_rows, enriched_pairs, viewpoint_rows, control_rows)

    print(f"Wrote comparison outputs under: {OUTPUTS}")


if __name__ == "__main__":
    main()
