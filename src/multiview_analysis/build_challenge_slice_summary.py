from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_VIEW_CSV = ROOT / "results" / "intermediate" / "scene_view_records.csv"
DEFAULT_GT_JSON = (
    ROOT
    / "results"
    / "recomputed"
    / "detector_family_comparison"
    / "standardized_test_eval"
    / "ground_truth"
    / "M4_test_gt.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "recomputed" / "box_fusion"

BASELINE_METHOD = "best_box"
METHOD_META = [
    ("support_weighted_or", "Support-weighted OR", "#8c5a2b"),
    ("roy_odds_mean_iou", "Odds-product + mean IoU", "#7a8fbf"),
    ("noisy_or_mean_iou", "Noisy-OR + mean IoU", "#4c9f70"),
    ("roy_odds_best_iou", "Odds-product + best IoU", "#457b9d"),
    ("noisy_or_best_iou", "Noisy-OR + best IoU", "#d95f02"),
]
ALL_METHODS = [(BASELINE_METHOD, "Best box", "#4c4c4c"), *METHOD_META]

SLICE_ORDER = [
    "full_ordered_pair_cache",
    "weak_primary_quality",
    "small_target_box",
    "low_primary_iou",
    "far_primary_view",
    "hard_scene",
    "primary_miss_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build challenge-slice summaries for the ordered-pair matched-box fusion "
            "cache so late-fusion conclusions are backed by explicit slice artifacts."
        )
    )
    parser.add_argument(
        "--scene-view-csv",
        default=str(DEFAULT_SCENE_VIEW_CSV),
        help="CSV with per-scene, per-view target metrics.",
    )
    parser.add_argument(
        "--gt-json",
        default=str(DEFAULT_GT_JSON),
        help="COCO ground-truth JSON used to estimate the primary-view target-box scale proxy.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for challenge-slice CSV, markdown, and plot outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def odds_fusion(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    odds_product = 1.0
    for probability in probabilities:
        p = clamp_probability(probability)
        odds_product *= p / (1.0 - p)
    return odds_product / (1.0 + odds_product)


def load_scene_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "image_id",
        "target_detected",
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def load_target_box_ratio_proxy(gt_json: Path, scene_df: pd.DataFrame) -> dict[int, float]:
    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    category_by_id = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    image_meta = {int(row["id"]): row for row in payload["images"]}

    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for image_id in image_meta:
        annotations_by_image[image_id] = []
    for row in payload["annotations"]:
        annotations_by_image[int(row["image_id"])].append(row)

    ratios: dict[int, float] = {}
    unique_views = scene_df[["image_id", "target_class"]].drop_duplicates()
    for image_id, target_class in unique_views.itertuples(index=False):
        image_id = int(image_id)
        meta = image_meta[image_id]
        image_area = float(meta["width"]) * float(meta["height"])
        target_areas = [
            float(annotation["area"])
            for annotation in annotations_by_image.get(image_id, [])
            if category_by_id[int(annotation["category_id"])] == str(target_class)
        ]
        ratios[image_id] = (max(target_areas) / image_area) if target_areas and image_area > 0.0 else 0.0
    return ratios


def attach_slice_features(scene_df: pd.DataFrame, gt_json: Path) -> pd.DataFrame:
    df = scene_df.copy()
    df["image_id"] = df["image_id"].astype(int)
    df["target_box_ratio_proxy"] = df["image_id"].map(load_target_box_ratio_proxy(gt_json, df)).fillna(0.0)
    scene_mean_quality = (
        df.groupby("scene_key", as_index=False)["target_strict_quality_iou50"]
        .mean()
        .rename(columns={"target_strict_quality_iou50": "scene_mean_single_quality"})
    )
    return df.merge(scene_mean_quality, on="scene_key", how="left")


def combo_scores(records: list[dict[str, object]]) -> dict[str, float]:
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
    support_ratio = len(ious) / float(len(records)) if records else 0.0
    noisy = noisy_or(confidences)
    roy_odds = odds_fusion(confidences)
    best_iou = max(ious) if ious else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "best_box": max(qualities) if qualities else 0.0,
        "support_weighted_or": noisy * mean_iou * support_ratio,
        "roy_odds_mean_iou": roy_odds * mean_iou,
        "noisy_or_mean_iou": noisy * mean_iou,
        "roy_odds_best_iou": roy_odds * best_iou,
        "noisy_or_best_iou": noisy * best_iou,
    }


def build_ordered_pair_rows(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_key, scene_group in scene_df.groupby("scene_key", sort=True):
        records = scene_group.to_dict("records")
        for primary, secondary in permutations(records, 2):
            scores = combo_scores([primary, secondary])
            row = {
                "scene_key": scene_key,
                "target_class": str(primary["target_class"]),
                "primary_viewpoint": str(primary["viewpoint"]),
                "secondary_viewpoint": str(secondary["viewpoint"]),
                "primary_single_quality": float(primary["target_strict_quality_iou50"]),
                "primary_iou": float(primary["target_match_iou_at_confidence_iou50"]),
                "primary_detected": int(primary["target_detected"]),
                "primary_radius": str(primary["radius"]),
                "primary_target_box_ratio_proxy": float(primary["target_box_ratio_proxy"]),
                "scene_mean_single_quality": float(primary["scene_mean_single_quality"]),
            }
            row.update(scores)
            rows.append(row)
    return pd.DataFrame(rows)


def slice_metadata(scene_df: pd.DataFrame, pair_df: pd.DataFrame) -> list[dict[str, object]]:
    weak_primary_threshold = float(scene_df["target_strict_quality_iou50"].quantile(0.25))
    small_box_threshold = float(scene_df["target_box_ratio_proxy"].quantile(0.25))
    low_iou_threshold = float(scene_df["target_match_iou_at_confidence_iou50"].quantile(0.25))
    hard_scene_threshold = float(scene_df.groupby("scene_key")["target_strict_quality_iou50"].mean().quantile(0.25))

    return [
        {
            "slice_id": "full_ordered_pair_cache",
            "slice_label": "Full ordered-pair cache",
            "definition": "All ordered primary-secondary view pairs.",
            "mask": pair_df.index == pair_df.index,
            "threshold_value": float("nan"),
        },
        {
            "slice_id": "weak_primary_quality",
            "slice_label": "Weak primary quality",
            "definition": f"Primary strict quality in the bottom quartile (<= {weak_primary_threshold:.4f}).",
            "mask": pair_df["primary_single_quality"] <= weak_primary_threshold,
            "threshold_value": weak_primary_threshold,
        },
        {
            "slice_id": "small_target_box",
            "slice_label": "Small target box",
            "definition": (
                "Primary-view target-box scale proxy in the bottom quartile based on the "
                f"largest target-class ground-truth box ratio (<= {small_box_threshold:.4f})."
            ),
            "mask": pair_df["primary_target_box_ratio_proxy"] <= small_box_threshold,
            "threshold_value": small_box_threshold,
        },
        {
            "slice_id": "low_primary_iou",
            "slice_label": "Low primary IoU",
            "definition": f"Primary matched IoU in the bottom quartile (<= {low_iou_threshold:.4f}).",
            "mask": pair_df["primary_iou"] <= low_iou_threshold,
            "threshold_value": low_iou_threshold,
        },
        {
            "slice_id": "far_primary_view",
            "slice_label": "Far primary view",
            "definition": "Primary radius is `far`.",
            "mask": pair_df["primary_radius"] == "far",
            "threshold_value": float("nan"),
        },
        {
            "slice_id": "hard_scene",
            "slice_label": "Hard scene",
            "definition": (
                "Scene-level mean single-view strict quality in the bottom quartile "
                f"(<= {hard_scene_threshold:.4f})."
            ),
            "mask": pair_df["scene_mean_single_quality"] <= hard_scene_threshold,
            "threshold_value": hard_scene_threshold,
        },
        {
            "slice_id": "primary_miss_only",
            "slice_label": "Primary miss only",
            "definition": "Primary view does not detect the target.",
            "mask": pair_df["primary_detected"] == 0,
            "threshold_value": float("nan"),
        },
    ]


def summarize_slices(pair_df: pd.DataFrame, slice_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_row in slice_rows:
        subset = pair_df.loc[slice_row["mask"]].copy()
        if subset.empty:
            continue

        best_box_mean = float(subset["best_box"].mean())
        for method_id, method_label, _ in ALL_METHODS:
            mean_score = float(subset[method_id].mean())
            positive_gain_rate = 0.0
            if method_id != BASELINE_METHOD:
                positive_gain_rate = float((subset[method_id] > subset[BASELINE_METHOD]).mean())
            rows.append(
                {
                    "slice_id": slice_row["slice_id"],
                    "slice_label": slice_row["slice_label"],
                    "definition": slice_row["definition"],
                    "threshold_value": slice_row["threshold_value"],
                    "row_count": len(subset),
                    "scene_count": int(subset["scene_key"].nunique()),
                    "method_id": method_id,
                    "method_label": method_label,
                    "mean_score": mean_score,
                    "delta_vs_best_box": mean_score - best_box_mean,
                    "positive_gain_rate_vs_best_box": positive_gain_rate,
                }
            )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    order_lookup = {slice_id: idx for idx, slice_id in enumerate(SLICE_ORDER)}
    summary_df["slice_order"] = summary_df["slice_id"].map(order_lookup).fillna(999).astype(int)
    summary_df = summary_df.sort_values(
        ["slice_order", "delta_vs_best_box", "method_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    summary_df["rank_within_slice"] = (
        summary_df.groupby("slice_id")["mean_score"].rank(method="dense", ascending=False).astype(int)
    )
    return summary_df


def build_headline_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    slice_meta = (
        summary_df[["slice_id", "slice_label", "definition", "row_count", "scene_count", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")
        .copy()
    )

    wide = (
        summary_df.pivot(index="slice_id", columns="method_id", values="mean_score")
        .reset_index()
        .merge(slice_meta, on="slice_id", how="left")
    )

    for method_id, _, _ in METHOD_META:
        wide[f"{method_id}_delta_vs_best_box"] = wide[method_id] - wide[BASELINE_METHOD]

    column_order = [
        "slice_id",
        "slice_label",
        "row_count",
        "scene_count",
        "definition",
        "best_box",
        "support_weighted_or",
        "support_weighted_or_delta_vs_best_box",
        "roy_odds_mean_iou",
        "roy_odds_mean_iou_delta_vs_best_box",
        "noisy_or_mean_iou",
        "noisy_or_mean_iou_delta_vs_best_box",
        "roy_odds_best_iou",
        "roy_odds_best_iou_delta_vs_best_box",
        "noisy_or_best_iou",
        "noisy_or_best_iou_delta_vs_best_box",
    ]
    return wide.sort_values("slice_order")[column_order]


def plot_slice_deltas(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.loc[summary_df["method_id"] != BASELINE_METHOD].copy()
    plot_df = plot_df.loc[plot_df["slice_id"] != "primary_miss_only"].copy()
    plot_df = plot_df.sort_values(["slice_order", "rank_within_slice", "method_label"])

    slice_labels = [
        plot_df.loc[plot_df["slice_id"] == slice_id, "slice_label"].iloc[0]
        for slice_id in SLICE_ORDER
        if slice_id in set(plot_df["slice_id"])
    ]
    y_base = np.arange(len(slice_labels))
    offsets = np.linspace(-0.24, 0.24, len(METHOD_META))

    fig, ax = plt.subplots(figsize=(11.2, 6.3), constrained_layout=True)
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.9)

    for offset, (method_id, method_label, color) in zip(offsets, METHOD_META):
        method_rows = (
            plot_df.loc[plot_df["method_id"] == method_id]
            .set_index("slice_id")
            .reindex([slice_id for slice_id in SLICE_ORDER if slice_id in set(plot_df["slice_id"])])
            .reset_index()
        )
        ax.plot(
            method_rows["delta_vs_best_box"],
            y_base + offset,
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=6.5,
            label=method_label,
        )

    ax.set_yticks(y_base, slice_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean strict-quality gain versus best-box selection")
    ax.set_title("Top fusion methods across the full cache and harder corroboration slices")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.legend(loc="lower right", frameon=True)

    x_max = float(plot_df["delta_vs_best_box"].max()) if not plot_df.empty else 0.1
    ax.set_xlim(-0.005, max(0.09, x_max + 0.01))

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Challenge-Slice Fusion Summary",
        "",
        "This report re-evaluates the ordered-pair matched-box fusion cache on harder slices of the fixed-detector benchmark.",
        "",
        "## Why this exists",
        "",
        "- The main late-fusion figure shows overall means on the full ordered-pair cache.",
        "- This companion artifact checks whether accumulation-based fusion pulls further ahead when the primary observation is harder.",
        "- The analysis stays on the ordered-pair cache so the results remain directly comparable to the existing matched-box fusion section.",
        "",
        "## Slice definitions",
        "",
    ]

    slice_meta = (
        summary_df[["slice_id", "slice_label", "definition", "row_count", "scene_count", "slice_order"]]
        .drop_duplicates()
        .sort_values("slice_order")
    )
    for _, row in slice_meta.iterrows():
        lines.append(
            f"- `{row['slice_label']}`: {row['definition']} "
            f"(rows `{int(row['row_count'])}`, scenes `{int(row['scene_count'])}`)."
        )

    lines.extend(
        [
            "",
            "## Headline deltas versus best-box selection",
            "",
            "| Slice | Best box | Support-weighted OR | Odds-product + mean IoU | Noisy-OR + mean IoU | Odds-product + best IoU | Noisy-OR + best IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    headline = build_headline_table(summary_df).sort_values(
        "slice_id",
        key=lambda column: column.map({slice_id: idx for idx, slice_id in enumerate(SLICE_ORDER)}),
    )
    for _, row in headline.iterrows():
        lines.append(
            "| "
            + f"{row['slice_label']} | "
            + f"{row['best_box']:.4f} | "
            + f"{row['support_weighted_or_delta_vs_best_box']:+.4f} | "
            + f"{row['roy_odds_mean_iou_delta_vs_best_box']:+.4f} | "
            + f"{row['noisy_or_mean_iou_delta_vs_best_box']:+.4f} | "
            + f"{row['roy_odds_best_iou_delta_vs_best_box']:+.4f} | "
            + f"{row['noisy_or_best_iou_delta_vs_best_box']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Reading note",
            "",
            "- `Primary miss only` is mostly a rescue slice, so accumulation methods collapse toward the same score as best-box selection once only one supporting view remains.",
            "- The strongest corroboration pattern is therefore expected on weak-primary, small-target, low-IoU, and hard-scene slices rather than on pure primary-miss rescue rows.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    scene_view_csv = Path(args.scene_view_csv)
    if not scene_view_csv.is_absolute():
        scene_view_csv = ROOT / scene_view_csv
    gt_json = Path(args.gt_json)
    if not gt_json.is_absolute():
        gt_json = ROOT / gt_json
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    ensure_dir(output_dir)
    ensure_dir(output_dir / "plots")

    scene_df = attach_slice_features(load_scene_records(scene_view_csv), gt_json)
    pair_df = build_ordered_pair_rows(scene_df)
    slice_rows = slice_metadata(scene_df, pair_df)
    summary_df = summarize_slices(pair_df, slice_rows)
    headline_df = build_headline_table(summary_df)

    summary_df.to_csv(output_dir / "challenge_slice_method_summary.csv", index=False)
    headline_df.to_csv(output_dir / "challenge_slice_headline_summary.csv", index=False)
    write_markdown_report(summary_df, output_dir / "challenge_slice_report.md")
    plot_slice_deltas(summary_df, output_dir / "plots" / "challenge_slice_top5_deltas.png")


if __name__ == "__main__":
    main()
