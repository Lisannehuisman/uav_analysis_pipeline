from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs"


LITERATURE_ROWS = [
    {
        "paper_key": "Du2018_UAVBenchmark",
        "year": 2018,
        "domain": "UAV detection benchmark",
        "main_takeaway": (
            "UAV detection is harder than ground-view detection because altitude, "
            "camera view, occlusion, density, and camera motion vary strongly."
        ),
        "relation_to_this_project": (
            "Motivates why a fixed single viewpoint is fragile and why viewpoint-aware "
            "analysis is necessary."
        ),
        "url": "https://openaccess.thecvf.com/content_ECCV_2018/html/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.html",
    },
    {
        "paper_key": "Wu2019_NDFT",
        "year": 2019,
        "domain": "UAV robustness",
        "main_takeaway": (
            "Altitude and view-angle changes act as UAV-specific nuisances, and models "
            "benefit from explicitly learning robustness to those factors."
        ),
        "relation_to_this_project": (
            "Supports our factor-level breakdown by elevation, radius, and azimuth."
        ),
        "url": "https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Delving_Into_Robust_Object_Detection_From_Unmanned_Aerial_Vehicles_A_ICCV_2019_paper.pdf",
    },
    {
        "paper_key": "Nassar2019_MultiViewInstanceDetection",
        "year": 2019,
        "domain": "Cross-view detection",
        "main_takeaway": (
            "Jointly modeling appearance and geometric soft constraints across views "
            "improves multi-view instance detection."
        ),
        "relation_to_this_project": (
            "Supports the idea that multi-view benefit is not only rescue-view gain but "
            "also cross-view corroboration."
        ),
        "url": "https://openaccess.thecvf.com/content_ICCV_2019/html/Nassar_Simultaneous_Multi-View_Instance_Detection_With_Learned_Geometric_Soft-Constraints_ICCV_2019_paper.html",
    },
    {
        "paper_key": "Vora2023_GeneralizedMVD",
        "year": 2023,
        "domain": "Generalization in multi-view detection",
        "main_takeaway": (
            "Generalization must be tested across camera count, camera position, and new "
            "scenes; many multi-view systems overfit a single camera configuration."
        ),
        "relation_to_this_project": (
            "Directly matches our concern that exact top viewpoint pairs may not "
            "generalize without scene-normalized validation."
        ),
        "url": "https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Vora_Bringing_Generalization_to_Deep_Multi-View_Pedestrian_Detection_WACVW_2023_paper.html",
    },
    {
        "paper_key": "Chen2023_VEDet",
        "year": 2023,
        "domain": "Multi-view 3D detection",
        "main_takeaway": (
            "Viewpoint consistency and viewpoint-equivariant learning improve multi-view "
            "object detection."
        ),
        "relation_to_this_project": (
            "Supports our interpretation that extra views help because they add structured, "
            "viewpoint-consistent evidence."
        ),
        "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Viewpoint_Equivariance_for_Multi-View_3D_Object_Detection_CVPR_2023_paper.html",
    },
    {
        "paper_key": "Hou2024_ViewSelection",
        "year": 2024,
        "domain": "Efficient view selection",
        "main_takeaway": (
            "Learning to select only the most helpful views can preserve detection or "
            "recognition performance while using just 2 or 3 views."
        ),
        "relation_to_this_project": (
            "Aligns strongly with our finding that most gain is captured already by the "
            "second view, with diminishing returns afterward."
        ),
        "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Learning_to_Select_Views_for_Efficient_Multi-View_Understanding_CVPR_2024_paper.html",
    },
    {
        "paper_key": "Dutta2024_MAVREC",
        "year": 2024,
        "domain": "Aerial multiview dataset",
        "main_takeaway": (
            "Synchronized aerial and ground views can improve aerial perception, and "
            "multi-view data is useful for stronger aerial detection training."
        ),
        "relation_to_this_project": (
            "Supports our separation between training-time view diversity and "
            "inference-time multi-view selection."
        ),
        "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Dutta_Multiview_Aerial_Visual_RECognition_MAVREC_Can_Multi-view_Improve_Aerial_Visual_CVPR_2024_paper.html",
    },
    {
        "paper_key": "Daryani2025_CaMuViD",
        "year": 2025,
        "domain": "Calibration-free multi-view detection",
        "main_takeaway": (
            "Calibration-free multi-view fusion can still improve detection and handle "
            "occlusion across views."
        ),
        "relation_to_this_project": (
            "Matches our box-fusion comparison, where even conservative late fusion adds "
            "value beyond best-view selection."
        ),
        "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Daryani_CaMuViD_Calibration-Free_Multi-View_Detection_CVPR_2025_paper.html",
    },
]


def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def annotate_bars(values: Iterable[float], xpos: Iterable[float], fmt: str = "{:.3f}") -> None:
    for x, y in zip(xpos, values):
        plt.text(x, y + 0.002, fmt.format(y), ha="center", va="bottom", fontsize=8)


def write_literature_summary() -> pd.DataFrame:
    literature_df = pd.DataFrame(LITERATURE_ROWS)
    literature_df.to_csv(OUTPUT_DIR / "literature_summary.csv", index=False)
    return literature_df


def build_operational_curve(protocol_df: pd.DataFrame, gain_df: pd.DataFrame) -> pd.DataFrame:
    curve = (
        protocol_df.loc[protocol_df["protocol_id"].isin(["n1_any1", "n2_any1", "n3_any1"])]
        .sort_values("drone_count")
        .copy()
    )
    merged = curve.merge(
        gain_df[
            [
                "drone_count",
                "fraction_of_total_ap50_95_gain_captured",
                "fraction_of_total_strict_quality_gain_captured",
            ]
        ],
        on="drone_count",
        how="left",
    )
    merged.to_csv(OUTPUT_DIR / "operational_gain_curve.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    x = merged["drone_count"].to_numpy()

    axes[0].plot(x, merged["expected_target_threshold_ap50_95"], marker="o", linewidth=2.3, color="#1f77b4")
    axes[0].set_title("Target AP50-95")
    axes[0].set_xlabel("Number of drones/views")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x)
    for xi, yi in zip(x, merged["expected_target_threshold_ap50_95"]):
        axes[0].text(xi, yi + 0.003, f"{yi:.3f}", ha="center", fontsize=8)

    axes[1].plot(
        x,
        merged["expected_target_threshold_strict_quality_iou50"],
        marker="o",
        linewidth=2.3,
        color="#d62728",
    )
    axes[1].set_title("Target strict quality")
    axes[1].set_xlabel("Number of drones/views")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(x)
    for xi, yi in zip(x, merged["expected_target_threshold_strict_quality_iou50"]):
        axes[1].text(xi, yi + 0.003, f"{yi:.3f}", ha="center", fontsize=8)

    axes[2].plot(x, merged["expected_target_found_rate"], marker="o", linewidth=2.3, color="#2ca02c")
    axes[2].set_title("Target found rate")
    axes[2].set_xlabel("Number of drones/views")
    axes[2].set_ylabel("Rate")
    axes[2].set_xticks(x)
    for xi, yi in zip(x, merged["expected_target_found_rate"]):
        axes[2].text(xi, yi + 0.0008, f"{yi:.4f}", ha="center", fontsize=8)

    fig.suptitle("Operational gain from 1 to 3 views", fontsize=13)
    save_plot(OUTPUT_DIR / "operational_gain_curve.png")
    return merged


def build_class_gain_plot(class_gain_df: pd.DataFrame) -> pd.DataFrame:
    class_gain = class_gain_df.sort_values("delta_target_ap50_95_1_to_2", ascending=True).copy()
    class_gain.to_csv(OUTPUT_DIR / "class_gain_one_to_two.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    y = np.arange(len(class_gain))

    axes[0].barh(y, class_gain["delta_target_ap50_95_1_to_2"], color="#4c78a8")
    axes[0].set_title("AP50-95 gain: 1 -> 2 views")
    axes[0].set_xlabel("Delta target AP50-95")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(class_gain["target_class"])

    axes[1].barh(y, class_gain["delta_target_strict_quality_iou50_1_to_2"], color="#e45756")
    axes[1].set_title("Strict-quality gain: 1 -> 2 views")
    axes[1].set_xlabel("Delta target strict quality")

    fig.suptitle("Per-class benefit of adding the second drone", fontsize=13)
    save_plot(OUTPUT_DIR / "class_gain_one_to_two.png")
    return class_gain


def build_pair_lift_distribution(pair_vs_single_df: pd.DataFrame) -> pd.DataFrame:
    lift = pair_vs_single_df["pair_minus_best_single"].dropna().copy()
    summary = pd.DataFrame(
        [
            {
                "pair_count": int(lift.shape[0]),
                "mean_pair_lift": float(lift.mean()),
                "median_pair_lift": float(lift.median()),
                "pairs_beating_best_single_fraction": float((lift > 0).mean()),
                "pairs_beating_best_single_count": int((lift > 0).sum()),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "pair_training_lift_summary.csv", index=False)

    plt.figure(figsize=(8.6, 4.8))
    plt.hist(lift, bins=36, color="#72b7b2", edgecolor="white")
    plt.axvline(lift.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"mean = {lift.mean():.3f}")
    plt.axvline(lift.median(), color="#1f77b4", linestyle=":", linewidth=2, label=f"median = {lift.median():.3f}")
    plt.xlabel("Pair mAP50-95 minus best constituent single")
    plt.ylabel("Number of viewpoint pairs")
    plt.title("Training-side pair lift distribution")
    plt.legend()
    save_plot(OUTPUT_DIR / "pair_training_lift_distribution.png")
    return summary


def build_training_regime_plot(
    single_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    matched_controls_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = [
        {
            "group": "Best single",
            "kind": "Restricted views",
            "mAP50_95": float(single_df["mAP50-95"].max()),
            "train_images": int(single_df.loc[single_df["mAP50-95"].idxmax(), "number_of_train_images"]),
        },
        {
            "group": "Best single",
            "kind": "Matched M4 control",
            "mAP50_95": float(
                matched_controls_df.loc[matched_controls_df["control_id"] == "mc_single_best_s00", "mAP50-95"].iloc[0]
            ),
            "train_images": int(
                matched_controls_df.loc[
                    matched_controls_df["control_id"] == "mc_single_best_s00", "number_of_train_images"
                ].iloc[0]
            ),
        },
        {
            "group": "Mean single",
            "kind": "Restricted views",
            "mAP50_95": float(single_df["mAP50-95"].mean()),
            "train_images": int(round(single_df["number_of_train_images"].mean())),
        },
        {
            "group": "Mean single",
            "kind": "Matched M4 control",
            "mAP50_95": float(
                matched_controls_df.loc[matched_controls_df["control_id"] == "mc_single_mean_s00", "mAP50-95"].iloc[0]
            ),
            "train_images": int(
                matched_controls_df.loc[
                    matched_controls_df["control_id"] == "mc_single_mean_s00", "number_of_train_images"
                ].iloc[0]
            ),
        },
        {
            "group": "Best pair",
            "kind": "Restricted views",
            "mAP50_95": float(pair_df["mAP50-95"].max()),
            "train_images": int(pair_df.loc[pair_df["mAP50-95"].idxmax(), "number_of_train_images"]),
        },
        {
            "group": "Best pair",
            "kind": "Matched M4 control",
            "mAP50_95": float(
                matched_controls_df.loc[matched_controls_df["control_id"] == "mc_pair_best_s00", "mAP50-95"].iloc[0]
            ),
            "train_images": int(
                matched_controls_df.loc[
                    matched_controls_df["control_id"] == "mc_pair_best_s00", "number_of_train_images"
                ].iloc[0]
            ),
        },
        {
            "group": "Mean pair",
            "kind": "Restricted views",
            "mAP50_95": float(pair_df["mAP50-95"].mean()),
            "train_images": int(round(pair_df["number_of_train_images"].mean())),
        },
        {
            "group": "Mean pair",
            "kind": "Matched M4 control",
            "mAP50_95": float(
                matched_controls_df.loc[matched_controls_df["control_id"] == "mc_pair_mean_s00", "mAP50-95"].iloc[0]
            ),
            "train_images": int(
                matched_controls_df.loc[
                    matched_controls_df["control_id"] == "mc_pair_mean_s00", "number_of_train_images"
                ].iloc[0]
            ),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "training_regime_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    groups = ["Best single", "Mean single", "Best pair", "Mean pair"]
    kinds = ["Restricted views", "Matched M4 control"]
    width = 0.35
    x = np.arange(len(groups))
    colors = ["#4c78a8", "#f58518"]

    for offset, kind in enumerate(kinds):
        subset = summary.loc[summary["kind"] == kind].set_index("group").loc[groups]
        xpos = x + (offset - 0.5) * width
        bars = ax.bar(xpos, subset["mAP50_95"], width=width, label=kind, color=colors[offset])
        for bar, train_images in zip(bars, subset["train_images"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.3f}\n{train_images} img",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("mAP50-95 on full M4 test split")
    ax.set_title("Training-side effect of viewpoint restriction versus matched-count M4 control")
    ax.legend()
    save_plot(OUTPUT_DIR / "training_regime_comparison.png")
    return summary


def build_relationship_plot(robust_df: pd.DataFrame) -> pd.DataFrame:
    k2 = robust_df.loc[robust_df["drone_count"] == 2].copy()
    axis_best = (
        k2.sort_values(["relationship_axis", "mean_AP50_95"], ascending=[True, False])
        .groupby("relationship_axis", as_index=False)
        .first()
    )
    mixed = (
        k2.loc[k2["relationship_axis"] == "mixed_diversity"]
        .sort_values("mean_AP50_95", ascending=True)
        .copy()
    )

    axis_best.to_csv(OUTPUT_DIR / "relationship_axis_best_k2.csv", index=False)
    mixed.to_csv(OUTPUT_DIR / "mixed_diversity_k2.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].errorbar(
        axis_best["mean_AP50_95"],
        np.arange(len(axis_best)),
        xerr=[
            axis_best["mean_AP50_95"] - axis_best["ci95_low"],
            axis_best["ci95_high"] - axis_best["mean_AP50_95"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        capsize=4,
    )
    axes[0].set_yticks(np.arange(len(axis_best)))
    axes[0].set_yticklabels(axis_best["relationship_axis"] + ": " + axis_best["relationship_type"])
    axes[0].set_xlabel("Mean AP50-95")
    axes[0].set_title("Best k=2 relationship type per axis")

    axes[1].errorbar(
        mixed["mean_AP50_95"],
        np.arange(len(mixed)),
        xerr=[
            mixed["mean_AP50_95"] - mixed["ci95_low"],
            mixed["ci95_high"] - mixed["mean_AP50_95"],
        ],
        fmt="o",
        color="#e45756",
        ecolor="#f2a3a3",
        capsize=4,
    )
    axes[1].set_yticks(np.arange(len(mixed)))
    axes[1].set_yticklabels(mixed["relationship_type"])
    axes[1].set_xlabel("Mean AP50-95")
    axes[1].set_title("k=2 mixed-diversity patterns")

    fig.suptitle("Robust pair-design patterns after scene-normalized validation", fontsize=13)
    save_plot(OUTPUT_DIR / "relationship_pattern_robustness_k2.png")
    return axis_best


def build_fusion_plot(fusion_df: pd.DataFrame) -> pd.DataFrame:
    fusion = fusion_df.sort_values("drone_count").copy()
    fusion.to_csv(OUTPUT_DIR / "fusion_policy_comparison.csv", index=False)

    x = np.arange(len(fusion))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    bars1 = ax.bar(x - width, fusion["oracle_best_available_quality"], width, label="Best-view oracle", color="#4c78a8")
    bars2 = ax.bar(
        x,
        fusion["fusion_support_weighted_or_quality"],
        width,
        label="Support-weighted OR",
        color="#72b7b2",
    )
    bars3 = ax.bar(
        x + width,
        fusion["fusion_noisy_or_max_iou_quality"],
        width,
        label="Noisy-OR + best IoU",
        color="#f58518",
    )

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} views" for k in fusion["drone_count"]])
    ax.set_ylabel("Strict-quality style fusion score")
    ax.set_title("Late-fusion policies on top of multi-view selection")
    ax.legend()
    save_plot(OUTPUT_DIR / "fusion_policy_comparison.png")
    return fusion


def write_headline_metrics(
    operational_curve: pd.DataFrame,
    pair_lift_summary: pd.DataFrame,
    fusion_df: pd.DataFrame,
) -> pd.DataFrame:
    k1 = operational_curve.loc[operational_curve["drone_count"] == 1].iloc[0]
    k2 = operational_curve.loc[operational_curve["drone_count"] == 2].iloc[0]
    k3 = operational_curve.loc[operational_curve["drone_count"] == 3].iloc[0]
    lift = pair_lift_summary.iloc[0]
    fusion2 = fusion_df.loc[fusion_df["drone_count"] == 2].iloc[0]

    metrics = pd.DataFrame(
        [
            {"metric": "one_to_two_target_ap50_95_gain", "value": k2["expected_target_threshold_ap50_95"] - k1["expected_target_threshold_ap50_95"]},
            {"metric": "one_to_two_target_strict_quality_gain", "value": k2["expected_target_threshold_strict_quality_iou50"] - k1["expected_target_threshold_strict_quality_iou50"]},
            {"metric": "one_to_two_target_found_rate_gain", "value": k2["expected_target_found_rate"] - k1["expected_target_found_rate"]},
            {"metric": "two_views_fraction_total_ap_gain_captured", "value": k2["fraction_of_total_ap50_95_gain_captured"]},
            {"metric": "two_views_fraction_total_strict_gain_captured", "value": k2["fraction_of_total_strict_quality_gain_captured"]},
            {"metric": "two_to_three_target_ap50_95_gain", "value": k3["expected_target_threshold_ap50_95"] - k2["expected_target_threshold_ap50_95"]},
            {"metric": "mean_pair_training_lift_over_best_single", "value": lift["mean_pair_lift"]},
            {"metric": "median_pair_training_lift_over_best_single", "value": lift["median_pair_lift"]},
            {"metric": "fraction_pairs_beating_best_single", "value": lift["pairs_beating_best_single_fraction"]},
            {"metric": "two_view_support_weighted_fusion_gain_over_oracle", "value": fusion2["gap_support_weighted_vs_oracle"]},
            {"metric": "two_view_noisy_or_gain_over_oracle", "value": fusion2["gap_noisy_or_vs_oracle"]},
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "headline_metrics.csv", index=False)
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    protocol_df = load_csv("m4_two_drone_operational_analysis/thesis_swarm_outputs/protocol_overall_summary.csv")
    gain_df = load_csv("m4_marginal_viewpoint_value_analysis/outputs/multi_view_gain_summary.csv")
    class_gain_df = load_csv("m4_marginal_viewpoint_value_analysis/outputs/class_multi_view_gain_summary.csv")
    pair_vs_single_df = load_csv(
        "viewpoint_data_separated/single_vs_pair_comparison______pairtrained_vs_singleviewbaselines/outputs/pair_vs_single_enriched.csv"
    )
    single_df = load_csv("viewpoint_data_separated/72_trained_models/reports/master_results.csv")
    pair_df = load_csv("viewpoint_data_separated/m4_pair_results/snapshot/reports/master_results.csv")
    matched_controls_df = load_csv("outputs/m4_matched_control_experiment/reports/master_results.csv")
    robust_df = load_csv(
        "m4_viewpoint_selection_analysis/outputs/robustness/robust_relationship_recommendations.csv"
    )
    fusion_df = load_csv("m4_oracle_vs_box_fusion_comparison/outputs/overall_policy_comparison.csv")

    write_literature_summary()
    operational_curve = build_operational_curve(protocol_df, gain_df)
    build_class_gain_plot(class_gain_df)
    pair_lift_summary = build_pair_lift_distribution(pair_vs_single_df)
    build_training_regime_plot(single_df, pair_df, matched_controls_df)
    build_relationship_plot(robust_df)
    fusion_summary = build_fusion_plot(fusion_df)
    write_headline_metrics(operational_curve, pair_lift_summary, fusion_summary)

    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
