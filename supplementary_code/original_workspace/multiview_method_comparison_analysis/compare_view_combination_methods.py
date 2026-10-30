from __future__ import annotations

import argparse
from itertools import combinations, permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_VIEW_CSV = ROOT / "m4_two_drone_operational_analysis" / "outputs" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = ROOT / "multiview_method_comparison_analysis" / "outputs"


EVALUABLE_METHODS = [
    "best_box",
    "mean_quality",
    "unanimous_best_box",
    "noisy_or_best_iou",
    "noisy_or_mean_iou",
    "support_weighted_or",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiview combination rules for 1-view and 2-view target-centric "
            "object detection using cached scene-view records."
        )
    )
    parser.add_argument(
        "--scene-view-csv",
        default=str(DEFAULT_SCENE_VIEW_CSV),
        help="CSV with per-scene, per-view target metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV summaries, plots, and the markdown report.",
    )
    parser.add_argument(
        "--min-added-view-support",
        type=int,
        default=8,
        help="Minimum support for headline added-view summaries.",
    )
    return parser.parse_args()


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def coalition_score(records: list[dict[str, object]], method_id: str) -> float:
    qualities = [float(record["target_strict_quality_iou50"]) for record in records]
    confidences = [
        float(record["target_match_confidence_iou50"])
        for record in records
        if float(record["target_match_confidence_iou50"]) > 0.0
    ]
    ious = [
        float(record["target_match_iou_at_confidence_iou50"])
        for record in records
        if float(record["target_match_iou_at_confidence_iou50"]) > 0.0
    ]
    support_ratio = len(ious) / len(records) if records else 0.0

    if method_id == "best_box":
        return max(qualities) if qualities else 0.0
    if method_id == "mean_quality":
        return float(np.mean(qualities)) if qualities else 0.0
    if method_id == "unanimous_best_box":
        return max(qualities) if qualities and support_ratio == 1.0 else 0.0
    if method_id == "noisy_or_best_iou":
        return noisy_or(confidences) * (max(ious) if ious else 0.0)
    if method_id == "noisy_or_mean_iou":
        return noisy_or(confidences) * (float(np.mean(ious)) if ious else 0.0)
    if method_id == "support_weighted_or":
        return noisy_or(confidences) * (float(np.mean(ious)) if ious else 0.0) * support_ratio
    raise KeyError(f"Unknown method_id: {method_id}")


def method_label(method_id: str) -> str:
    return {
        "best_box": "Best box (max)",
        "mean_quality": "Mean quality",
        "unanimous_best_box": "2-of-2 unanimous best box",
        "noisy_or_best_iou": "Noisy-OR + best IoU",
        "noisy_or_mean_iou": "Noisy-OR + mean IoU",
        "support_weighted_or": "Support-weighted OR",
    }[method_id]


def method_family(method_id: str) -> str:
    return {
        "best_box": "Best-view rescue",
        "mean_quality": "Naive pooling",
        "unanimous_best_box": "Strict confirmation",
        "noisy_or_best_iou": "Probabilistic accumulation",
        "noisy_or_mean_iou": "Probabilistic accumulation",
        "support_weighted_or": "Conservative corroboration",
    }[method_id]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_feasibility_matrix() -> pd.DataFrame:
    rows = [
        {
            "method_family": "Best-view rescue (max)",
            "evaluable_now": True,
            "what_it_does": "Keeps the strongest single matched target observation.",
            "requirements_missing_if_not_evaluable": "",
        },
        {
            "method_family": "Mean pooling",
            "evaluable_now": True,
            "what_it_does": "Averages target quality across selected views.",
            "requirements_missing_if_not_evaluable": "",
        },
        {
            "method_family": "Probabilistic score fusion (noisy-OR)",
            "evaluable_now": True,
            "what_it_does": "Combines view confidences as accumulating evidence, then couples them to localization quality.",
            "requirements_missing_if_not_evaluable": "",
        },
        {
            "method_family": "Conservative corroborative fusion",
            "evaluable_now": True,
            "what_it_does": "Rewards both evidence strength and multi-view support ratio.",
            "requirements_missing_if_not_evaluable": "",
        },
        {
            "method_family": "Soft-NMS across views",
            "evaluable_now": False,
            "what_it_does": "Softly suppresses overlapping boxes instead of hard removal.",
            "requirements_missing_if_not_evaluable": (
                "Needs a valid way to compare or reproject boxes across views into a common frame."
            ),
        },
        {
            "method_family": "Weighted Boxes Fusion",
            "evaluable_now": False,
            "what_it_does": "Confidence-weighted box averaging across aligned detections.",
            "requirements_missing_if_not_evaluable": (
                "Needs cross-view box correspondence in a shared image/world frame or valid reprojection."
            ),
        },
        {
            "method_family": "Geometry-aware fusion",
            "evaluable_now": False,
            "what_it_does": "Uses calibration and camera geometry to fuse evidence consistently across views.",
            "requirements_missing_if_not_evaluable": (
                "Needs camera intrinsics/extrinsics and explicit cross-view geometric linkage."
            ),
        },
        {
            "method_family": "Learned query/attention multiview fusion",
            "evaluable_now": False,
            "what_it_does": "Learns cross-view feature interaction directly from multiview training data.",
            "requirements_missing_if_not_evaluable": (
                "Needs a trainable multiview model, synchronized multiview supervision, and a new training pipeline."
            ),
        },
    ]
    return pd.DataFrame(rows)


def load_scene_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "azimuth",
        "target_ap50_95",
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
        "target_detected",
        "target_visible",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def build_pair_rows(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_key, scene_group in scene_df.groupby("scene_key", sort=True):
        records = scene_group.to_dict("records")
        for left, right in combinations(records, 2):
            pair_records = [left, right]
            row = {
                "scene_key": scene_key,
                "target_class": left["target_class"],
                "viewpoint_1": left["viewpoint"],
                "viewpoint_2": right["viewpoint"],
                "ordered_label": f"{left['viewpoint']} + {right['viewpoint']}",
                "best_constituent_single": max(
                    float(left["target_strict_quality_iou50"]),
                    float(right["target_strict_quality_iou50"]),
                ),
            }
            for method_id in EVALUABLE_METHODS:
                row[f"{method_id}_score"] = coalition_score(pair_records, method_id)
            rows.append(row)
    return pd.DataFrame(rows)


def build_scene_balanced_summary(scene_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    single_scene = (
        scene_df.groupby("scene_key")["target_strict_quality_iou50"]
        .mean()
        .reset_index(name="scene_expected_quality")
    )
    overall_single = float(single_scene["scene_expected_quality"].mean())

    rows: list[dict[str, object]] = [
        {
            "drone_count": 1,
            "method_id": "single_view_reference",
            "method_label": "Single-view reference",
            "method_family": "Reference",
            "mean_scene_expected_quality": overall_single,
            "gain_vs_single_reference": 0.0,
        }
    ]

    for method_id in EVALUABLE_METHODS:
        scene_expectation = (
            pair_df.groupby("scene_key")[f"{method_id}_score"]
            .mean()
            .reset_index(name="scene_expected_quality")
        )
        mean_quality = float(scene_expectation["scene_expected_quality"].mean())
        rows.append(
            {
                "drone_count": 2,
                "method_id": method_id,
                "method_label": method_label(method_id),
                "method_family": method_family(method_id),
                "mean_scene_expected_quality": mean_quality,
                "gain_vs_single_reference": mean_quality - overall_single,
            }
        )
    return pd.DataFrame(rows)


def build_ordered_pair_gain_rows(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_key, scene_group in scene_df.groupby("scene_key", sort=True):
        records = scene_group.to_dict("records")
        for primary, secondary in permutations(records, 2):
            pair_records = [primary, secondary]
            primary_quality = float(primary["target_strict_quality_iou50"])
            best_single = max(primary_quality, float(secondary["target_strict_quality_iou50"]))
            primary_detected = int(primary["target_detected"]) > 0

            for method_id in EVALUABLE_METHODS:
                pair_score = coalition_score(pair_records, method_id)
                rows.append(
                    {
                        "scene_key": scene_key,
                        "target_class": primary["target_class"],
                        "primary_viewpoint": primary["viewpoint"],
                        "secondary_viewpoint": secondary["viewpoint"],
                        "method_id": method_id,
                        "method_label": method_label(method_id),
                        "method_family": method_family(method_id),
                        "primary_single_quality": primary_quality,
                        "best_constituent_single_quality": best_single,
                        "pair_score": pair_score,
                        "lift_vs_primary": pair_score - primary_quality,
                        "lift_vs_best_constituent": pair_score - best_single,
                        "primary_miss": int(not primary_detected),
                        "pair_success": int(pair_score > 0.0),
                        "rescue_when_primary_miss": int((not primary_detected) and (pair_score > 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_added_viewpoints(
    ordered_rows: pd.DataFrame, min_support: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_summary = (
        ordered_rows.groupby(
            ["method_id", "method_label", "method_family", "primary_viewpoint", "secondary_viewpoint"],
            as_index=False,
        )
        .agg(
            scene_count=("scene_key", "count"),
            mean_primary_single_quality=("primary_single_quality", "mean"),
            mean_pair_score=("pair_score", "mean"),
            mean_lift_vs_primary=("lift_vs_primary", "mean"),
            mean_lift_vs_best_constituent=("lift_vs_best_constituent", "mean"),
            primary_miss_count=("primary_miss", "sum"),
            rescue_count=("rescue_when_primary_miss", "sum"),
        )
    )
    pair_summary["rescue_rate_given_primary_miss"] = np.where(
        pair_summary["primary_miss_count"] > 0,
        pair_summary["rescue_count"] / pair_summary["primary_miss_count"],
        np.nan,
    )

    added_view_summary = (
        ordered_rows.groupby(
            ["method_id", "method_label", "method_family", "secondary_viewpoint"],
            as_index=False,
        )
        .agg(
            scene_count=("scene_key", "count"),
            primary_viewpoint_count=("primary_viewpoint", "nunique"),
            mean_lift_vs_primary=("lift_vs_primary", "mean"),
            mean_lift_vs_best_constituent=("lift_vs_best_constituent", "mean"),
            primary_miss_count=("primary_miss", "sum"),
            rescue_count=("rescue_when_primary_miss", "sum"),
        )
    )
    added_view_summary["rescue_rate_given_primary_miss"] = np.where(
        added_view_summary["primary_miss_count"] > 0,
        added_view_summary["rescue_count"] / added_view_summary["primary_miss_count"],
        np.nan,
    )

    headline = (
        added_view_summary.loc[added_view_summary["scene_count"] >= min_support]
        .sort_values(["method_id", "mean_lift_vs_primary"], ascending=[True, False])
        .groupby("method_id", as_index=False)
        .head(10)
        .copy()
    )

    return pair_summary, added_view_summary, headline


def plot_overall_method_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    pair_only = summary_df.loc[summary_df["drone_count"] == 2].copy()
    pair_only = pair_only.sort_values("mean_scene_expected_quality", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    bars = ax.bar(pair_only["method_label"], pair_only["mean_scene_expected_quality"], color="#4c78a8")
    for bar, gain in zip(bars, pair_only["gain_vs_single_reference"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}\n(+{gain:.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Scene-balanced expected target quality")
    ax.set_title("Two-view combination methods compared against the single-view reference")
    ax.tick_params(axis="x", rotation=22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_added_viewpoints(headline_df: pd.DataFrame, output_path: Path) -> None:
    if headline_df.empty:
        return

    methods = headline_df["method_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(methods), 1, figsize=(10.2, 3.3 * len(methods)))
    if len(methods) == 1:
        axes = [axes]

    for ax, method_id in zip(axes, methods):
        subset = headline_df.loc[headline_df["method_id"] == method_id].nlargest(8, "mean_lift_vs_primary")
        ax.barh(subset["secondary_viewpoint"], subset["mean_lift_vs_primary"], color="#72b7b2")
        ax.invert_yaxis()
        ax.set_title(method_label(method_id))
        ax.set_xlabel("Mean lift versus primary view")

    fig.suptitle("Best added viewpoints by fusion method", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    feasibility_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    headline_df: pd.DataFrame,
) -> None:
    single_reference = overall_df.loc[overall_df["method_id"] == "single_view_reference"].iloc[0]
    pair_best = overall_df.loc[overall_df["drone_count"] == 2].sort_values(
        "mean_scene_expected_quality", ascending=False
    )
    best_method = pair_best.iloc[0]
    worst_method = pair_best.iloc[-1]

    lines = [
        "# Multiview method comparison report",
        "",
        "## Purpose",
        "",
        "This report compares viewpoint-combination rules that can be evaluated honestly with the current cached project data.",
        "It is designed to answer two questions:",
        "",
        "- if one view is already available, which two-view combination rules improve target-centric detection quality most;",
        "- and which second viewpoints add the most value when they are appended to a primary view.",
        "",
        "## How this comparison is computed",
        "",
        "- Input data: per-scene, per-view target metrics from `scene_view_records.csv`.",
        "- Single-view reference: the scene-balanced mean of `target_strict_quality_iou50` over all available single viewpoints.",
        "- Two-view comparison: for every scene and every unordered pair of viewpoints, the script evaluates each fusion rule on the same underlying per-view target records.",
        "- Added-viewpoint analysis: for every ordered pair `(primary, secondary)`, the script measures how much the pair score improves over the primary-only score and over the best constituent single score.",
        "- Headline added viewpoints: the report ranks secondary viewpoints by their average lift across all primaries they were paired with, subject to the support threshold.",
        "",
        "## Important boundary",
        "",
        "The current cache supports score-level and conservative late-fusion comparisons, but it does not yet support full geometry-aware box fusion across views.",
        "That means methods such as Weighted Boxes Fusion or calibration-based reprojection fusion are discussed in the literature map, but not evaluated as if they were already valid on the present files.",
        "",
        "## Single-view reference",
        "",
        f"- Scene-balanced single-view reference quality: `{single_reference['mean_scene_expected_quality']:.4f}`",
        "",
        "## Best two-view method in the current cache",
        "",
        f"- Best method: `{best_method['method_label']}`",
        f"- Method family: `{best_method['method_family']}`",
        f"- Scene-balanced two-view quality: `{best_method['mean_scene_expected_quality']:.4f}`",
        f"- Gain versus single-view reference: `{best_method['gain_vs_single_reference']:+.4f}`",
        "",
        "## Weakest two-view method in the current cache",
        "",
        f"- Weakest method: `{worst_method['method_label']}`",
        f"- Scene-balanced two-view quality: `{worst_method['mean_scene_expected_quality']:.4f}`",
        f"- Gain versus single-view reference: `{worst_method['gain_vs_single_reference']:+.4f}`",
        "",
        "## Interpreting the method families",
        "",
        "- `Best box (max)` is the rescue-view baseline: the second view helps if either view is good.",
        "- `Mean quality` tests the naive intuition that all views should simply be averaged.",
        "- `2-of-2 unanimous best box` is a strict confirmation rule and will often trade recall for agreement.",
        "- `Noisy-OR` methods treat multiple views as accumulating evidence rather than competing single boxes.",
        "- `Support-weighted OR` is the most conservative evaluable corroboration rule because it rewards both evidence strength and multi-view agreement.",
        "",
        "## Feasibility map",
        "",
        "The CSV `method_feasibility_matrix.csv` records which families are evaluable now and which require extra calibration, correspondence, or multiview training.",
        "",
        "## Strong added viewpoints by method",
        "",
    ]

    if headline_df.empty:
        lines.append("No added-view headline rows met the current support threshold.")
    else:
        for method_id in headline_df["method_id"].drop_duplicates().tolist():
            subset = headline_df.loc[headline_df["method_id"] == method_id].nlargest(5, "mean_lift_vs_primary")
            lines.append(f"### {method_label(method_id)}")
            lines.append("")
            for _, row in subset.iterrows():
                rescue_text = (
                    "n/a"
                    if pd.isna(row["rescue_rate_given_primary_miss"])
                    else f"{float(row['rescue_rate_given_primary_miss']):.4f}"
                )
                lines.append(
                    f"- `{row['secondary_viewpoint']}`: mean lift vs primary `{float(row['mean_lift_vs_primary']):+.4f}`, "
                    f"lift vs best constituent `{float(row['mean_lift_vs_best_constituent']):+.4f}`, "
                    f"support `{int(row['scene_count'])}`, rescue|primary miss `{rescue_text}`"
                )
            lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `overall_method_summary.csv`",
            "- `pair_method_rows.csv`",
            "- `ordered_pair_method_rows.csv`",
            "- `ordered_pair_gain_summary.csv`",
            "- `added_viewpoint_summary.csv`",
            "- `added_viewpoint_headlines.csv`",
            "- `overall_method_comparison.png`",
            "- `top_added_viewpoints_by_method.png`",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_view_path = Path(args.scene_view_csv)
    if not scene_view_path.is_absolute():
        scene_view_path = ROOT / scene_view_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    ensure_dir(output_dir)

    scene_df = load_scene_records(scene_view_path)
    feasibility_df = build_feasibility_matrix()
    pair_df = build_pair_rows(scene_df)
    overall_df = build_scene_balanced_summary(scene_df, pair_df)
    ordered_rows = build_ordered_pair_gain_rows(scene_df)
    pair_gain_summary, added_view_summary, headline_df = summarize_added_viewpoints(
        ordered_rows, args.min_added_view_support
    )

    feasibility_df.to_csv(output_dir / "method_feasibility_matrix.csv", index=False)
    pair_df.to_csv(output_dir / "pair_method_rows.csv", index=False)
    overall_df.to_csv(output_dir / "overall_method_summary.csv", index=False)
    ordered_rows.to_csv(output_dir / "ordered_pair_method_rows.csv", index=False)
    pair_gain_summary.to_csv(output_dir / "ordered_pair_gain_summary.csv", index=False)
    added_view_summary.to_csv(output_dir / "added_viewpoint_summary.csv", index=False)
    headline_df.to_csv(output_dir / "added_viewpoint_headlines.csv", index=False)

    plot_overall_method_summary(overall_df, output_dir / "overall_method_comparison.png")
    plot_top_added_viewpoints(headline_df, output_dir / "top_added_viewpoints_by_method.png")
    write_report(output_dir / "method_comparison_report.md", feasibility_df, overall_df, headline_df)

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
