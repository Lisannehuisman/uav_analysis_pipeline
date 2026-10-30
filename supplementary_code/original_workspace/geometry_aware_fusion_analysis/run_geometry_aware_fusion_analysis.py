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
DEFAULT_GEOMETRY_CSV = ROOT / "geometry_ground_truth_analysis" / "outputs" / "view_geometry_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "geometry_aware_fusion_analysis" / "outputs"
VIEWPOINT_PRIOR_SHRINKAGE = 4.0

METHODS = [
    "best_box",
    "noisy_or_best_iou",
    "support_weighted_or",
    "geometry_prior_selector",
    "geometry_calibrated_selector",
    "viewpoint_cell_prior_selector",
    "geometry_weighted_or_best_iou",
    "geometry_weighted_or_mean_iou",
    "viewpoint_cell_weighted_or_best_iou",
    "hybrid_geometry_cell_weighted_or_best_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate geometry-aware target-centric fusion baselines by combining "
            "scene-view target metrics with manifest-derived camera-target geometry."
        )
    )
    parser.add_argument("--scene-view-csv", default=str(DEFAULT_SCENE_VIEW_CSV), help="CSV with per-scene, per-view target metrics.")
    parser.add_argument("--geometry-csv", default=str(DEFAULT_GEOMETRY_CSV), help="CSV with per-image geometry metadata.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs.")
    parser.add_argument(
        "--min-added-view-support",
        type=int,
        default=8,
        help="Minimum support for added-viewpoint headline rows.",
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


def method_label(method_id: str) -> str:
    return {
        "best_box": "Best box (max)",
        "noisy_or_best_iou": "Noisy-OR + best IoU",
        "support_weighted_or": "Support-weighted OR",
        "geometry_prior_selector": "Geometry prior selector",
        "geometry_calibrated_selector": "Geometry-calibrated selector",
        "viewpoint_cell_prior_selector": "Viewpoint-cell prior selector",
        "geometry_weighted_or_best_iou": "Geometry-weighted OR + best IoU",
        "geometry_weighted_or_mean_iou": "Geometry-weighted OR + mean IoU",
        "viewpoint_cell_weighted_or_best_iou": "Viewpoint-cell OR + best IoU",
        "hybrid_geometry_cell_weighted_or_best_iou": "Hybrid geometry+cell OR + best IoU",
    }[method_id]


def method_family(method_id: str) -> str:
    return {
        "best_box": "Best-view rescue",
        "noisy_or_best_iou": "Probabilistic accumulation",
        "support_weighted_or": "Conservative corroboration",
        "geometry_prior_selector": "Geometry-aware selection",
        "geometry_calibrated_selector": "Geometry-aware selection",
        "viewpoint_cell_prior_selector": "Geometry-aware selection",
        "geometry_weighted_or_best_iou": "Geometry-aware accumulation",
        "geometry_weighted_or_mean_iou": "Geometry-aware accumulation",
        "viewpoint_cell_weighted_or_best_iou": "Geometry-aware accumulation",
        "hybrid_geometry_cell_weighted_or_best_iou": "Geometry-aware accumulation",
    }[method_id]


def load_joined_scene_records(scene_view_path: Path, geometry_path: Path) -> pd.DataFrame:
    scene_df = pd.read_csv(scene_view_path)
    geometry_df = pd.read_csv(geometry_path)

    numeric_scene_columns = [
        "azimuth",
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
        "target_detected",
        "target_visible",
    ]
    for column in numeric_scene_columns:
        scene_df[column] = pd.to_numeric(scene_df[column], errors="coerce").fillna(0.0)

    keep_geometry_columns = [
        "file_name",
        "focus_class_name",
        "camera_to_target_distance_m",
        "radius_name",
        "elevation_name",
        "azimuth_deg",
        "yaw_error_deg",
        "pitch_error_deg",
        "bbox_area_norm",
        "bbox_center_distance_norm",
    ]
    joined_df = scene_df.merge(geometry_df[keep_geometry_columns], on="file_name", how="left")
    if joined_df["camera_to_target_distance_m"].isna().any():
        missing = int(joined_df["camera_to_target_distance_m"].isna().sum())
        raise ValueError(f"Geometry join failed for {missing} rows.")

    joined_df["azimuth_deg"] = pd.to_numeric(joined_df["azimuth_deg"], errors="coerce").fillna(joined_df["azimuth"])
    return joined_df


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    distance = df["camera_to_target_distance_m"].to_numpy(dtype=float) / 10.0
    azimuth_rad = np.deg2rad(df["azimuth_deg"].to_numpy(dtype=float))
    elev_mid = (df["elevation_name"].to_numpy(dtype=str) == "mid").astype(float)
    elev_high = (df["elevation_name"].to_numpy(dtype=str) == "high").astype(float)
    rad_mid = (df["radius_name"].to_numpy(dtype=str) == "mid").astype(float)
    rad_far = (df["radius_name"].to_numpy(dtype=str) == "far").astype(float)

    return np.column_stack(
        [
            np.ones(len(df)),
            distance,
            distance**2,
            np.sin(azimuth_rad),
            np.cos(azimuth_rad),
            elev_mid,
            elev_high,
            rad_mid,
            rad_far,
        ]
    )


def fit_predict_geometry_priors(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    global_mean = float(scene_df["target_strict_quality_iou50"].mean())
    for scene_key, holdout_df in scene_df.groupby("scene_key", sort=True):
        target_class = holdout_df["target_class"].iloc[0]
        train_df = scene_df.loc[
            (scene_df["target_class"] == target_class) & (scene_df["scene_key"] != scene_key)
        ].copy()

        class_mean = float(train_df["target_strict_quality_iou50"].mean()) if not train_df.empty else 0.0
        holdout = holdout_df.copy()
        holdout["class_train_scene_count"] = train_df["scene_key"].nunique()
        holdout["class_train_row_count"] = len(train_df)
        holdout["class_train_mean_quality"] = class_mean

        if len(train_df) >= 12 and train_df["scene_key"].nunique() >= 4:
            x_train = design_matrix(train_df)
            y_train = train_df["target_strict_quality_iou50"].to_numpy(dtype=float)
            beta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
            predicted_geometry = design_matrix(holdout) @ beta
            holdout["geometry_prior_source"] = "class_regression_leave_one_scene_out"
        elif not train_df.empty:
            predicted_geometry = np.full(len(holdout), class_mean, dtype=float)
            holdout["geometry_prior_source"] = "class_mean_fallback"
        else:
            predicted_geometry = np.full(len(holdout), global_mean, dtype=float)
            holdout["geometry_prior_source"] = "global_mean_fallback"

        safe_mean = class_mean if class_mean > 1e-6 else max(global_mean, 1e-6)
        predicted_geometry = np.clip(predicted_geometry, 0.0, 1.0)
        geometry_weight = np.clip(predicted_geometry / safe_mean, 0.5, 1.5)

        viewpoint_stats = (
            train_df.groupby("viewpoint")["target_strict_quality_iou50"]
            .agg(["mean", "count"])
            .reset_index()
        )
        viewpoint_mean_map = dict(zip(viewpoint_stats["viewpoint"], viewpoint_stats["mean"]))
        viewpoint_count_map = dict(zip(viewpoint_stats["viewpoint"], viewpoint_stats["count"]))

        raw_cell_prior = holdout["viewpoint"].map(viewpoint_mean_map).astype(float)
        cell_support = holdout["viewpoint"].map(viewpoint_count_map).fillna(0.0).astype(float)
        fallback_prior = class_mean if class_mean > 1e-6 else global_mean
        raw_cell_prior = raw_cell_prior.fillna(fallback_prior)
        shrink = VIEWPOINT_PRIOR_SHRINKAGE
        predicted_viewpoint_cell_prior = (
            (cell_support * raw_cell_prior.to_numpy(dtype=float)) + (shrink * safe_mean)
        ) / (cell_support.to_numpy(dtype=float) + shrink)
        predicted_viewpoint_cell_prior = np.clip(predicted_viewpoint_cell_prior, 0.0, 1.0)
        viewpoint_cell_weight = np.clip(predicted_viewpoint_cell_prior / safe_mean, 0.5, 1.5)

        predicted_hybrid_prior = 0.5 * predicted_geometry + 0.5 * predicted_viewpoint_cell_prior
        predicted_hybrid_prior = np.clip(predicted_hybrid_prior, 0.0, 1.0)
        hybrid_weight = np.clip(predicted_hybrid_prior / safe_mean, 0.5, 1.5)

        holdout["predicted_geometry_prior"] = predicted_geometry
        holdout["geometry_weight"] = geometry_weight
        holdout["predicted_viewpoint_cell_prior"] = predicted_viewpoint_cell_prior
        holdout["viewpoint_cell_prior_support"] = cell_support
        holdout["viewpoint_cell_weight"] = viewpoint_cell_weight
        holdout["predicted_hybrid_prior"] = predicted_hybrid_prior
        holdout["hybrid_weight"] = hybrid_weight
        rows.append(holdout)

    prior_df = pd.concat(rows, ignore_index=True)
    return prior_df


def build_prior_diagnostics(prior_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_class, class_df in prior_df.groupby("target_class", sort=True):
        actual = class_df["target_strict_quality_iou50"].to_numpy(dtype=float)
        geometry_prior = class_df["predicted_geometry_prior"].to_numpy(dtype=float)
        cell_prior = class_df["predicted_viewpoint_cell_prior"].to_numpy(dtype=float)
        hybrid_prior = class_df["predicted_hybrid_prior"].to_numpy(dtype=float)
        rows.append(
            {
                "target_class": target_class,
                "scene_count": int(class_df["scene_key"].nunique()),
                "row_count": int(len(class_df)),
                "actual_mean_quality": float(np.mean(actual)),
                "predicted_geometry_mean_prior": float(np.mean(geometry_prior)),
                "predicted_cell_mean_prior": float(np.mean(cell_prior)),
                "predicted_hybrid_mean_prior": float(np.mean(hybrid_prior)),
                "geometry_prior_actual_correlation": float(np.corrcoef(actual, geometry_prior)[0, 1]) if len(class_df) > 1 else np.nan,
                "cell_prior_actual_correlation": float(np.corrcoef(actual, cell_prior)[0, 1]) if len(class_df) > 1 else np.nan,
                "hybrid_prior_actual_correlation": float(np.corrcoef(actual, hybrid_prior)[0, 1]) if len(class_df) > 1 else np.nan,
                "geometry_prior_mae": float(np.mean(np.abs(actual - geometry_prior))),
                "cell_prior_mae": float(np.mean(np.abs(actual - cell_prior))),
                "hybrid_prior_mae": float(np.mean(np.abs(actual - hybrid_prior))),
                "mean_geometry_weight": float(class_df["geometry_weight"].mean()),
                "mean_viewpoint_cell_weight": float(class_df["viewpoint_cell_weight"].mean()),
                "mean_hybrid_weight": float(class_df["hybrid_weight"].mean()),
            }
        )
    diagnostics_df = pd.DataFrame(rows).sort_values("target_class").reset_index(drop=True)
    return diagnostics_df


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
    weighted_confidences = [
        min(1.0, float(record["target_match_confidence_iou50"]) * float(record["geometry_weight"]))
        for record in records
        if float(record["target_match_confidence_iou50"]) > 0.0
    ]
    cell_weighted_confidences = [
        min(1.0, float(record["target_match_confidence_iou50"]) * float(record["viewpoint_cell_weight"]))
        for record in records
        if float(record["target_match_confidence_iou50"]) > 0.0
    ]
    hybrid_weighted_confidences = [
        min(1.0, float(record["target_match_confidence_iou50"]) * float(record["hybrid_weight"]))
        for record in records
        if float(record["target_match_confidence_iou50"]) > 0.0
    ]
    geometry_priors = [float(record["predicted_geometry_prior"]) for record in records]
    cell_priors = [float(record["predicted_viewpoint_cell_prior"]) for record in records]
    geometry_selection_scores = [
        float(record["target_match_confidence_iou50"]) * float(record["geometry_weight"])
        for record in records
    ]
    cell_selection_scores = [
        float(record["target_match_confidence_iou50"]) * float(record["viewpoint_cell_weight"])
        for record in records
    ]

    if method_id == "best_box":
        return max(qualities) if qualities else 0.0
    if method_id == "noisy_or_best_iou":
        return noisy_or(confidences) * (max(ious) if ious else 0.0)
    if method_id == "support_weighted_or":
        return noisy_or(confidences) * (float(np.mean(ious)) if ious else 0.0) * support_ratio
    if method_id == "geometry_prior_selector":
        selected_index = int(np.argmax(geometry_priors))
        return qualities[selected_index]
    if method_id == "geometry_calibrated_selector":
        selected_index = int(np.argmax(geometry_selection_scores))
        return qualities[selected_index]
    if method_id == "viewpoint_cell_prior_selector":
        selected_index = int(np.argmax(cell_priors))
        return qualities[selected_index]
    if method_id == "geometry_weighted_or_best_iou":
        return noisy_or(weighted_confidences) * (max(ious) if ious else 0.0)
    if method_id == "geometry_weighted_or_mean_iou":
        return noisy_or(weighted_confidences) * (float(np.mean(ious)) if ious else 0.0)
    if method_id == "viewpoint_cell_weighted_or_best_iou":
        return noisy_or(cell_weighted_confidences) * (max(ious) if ious else 0.0)
    if method_id == "hybrid_geometry_cell_weighted_or_best_iou":
        return noisy_or(hybrid_weighted_confidences) * (max(ious) if ious else 0.0)
    raise KeyError(f"Unknown method_id: {method_id}")


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
            for method_id in METHODS:
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

    for method_id in METHODS:
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

            for method_id in METHODS:
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


def build_method_value_summary(overall_df: pd.DataFrame, ordered_rows: pd.DataFrame) -> pd.DataFrame:
    pair_only = overall_df.loc[overall_df["drone_count"] == 2].copy()
    summary = (
        ordered_rows.groupby(["method_id", "method_label", "method_family"], as_index=False)
        .agg(
            mean_lift_vs_primary=("lift_vs_primary", "mean"),
            mean_lift_vs_best_constituent=("lift_vs_best_constituent", "mean"),
            positive_vs_primary_rate=("lift_vs_primary", lambda values: float((pd.Series(values) > 0.0).mean())),
            positive_vs_best_constituent_rate=(
                "lift_vs_best_constituent",
                lambda values: float((pd.Series(values) > 0.0).mean()),
            ),
            primary_miss_count=("primary_miss", "sum"),
            rescue_count=("rescue_when_primary_miss", "sum"),
        )
    )
    summary["rescue_rate_given_primary_miss"] = np.where(
        summary["primary_miss_count"] > 0,
        summary["rescue_count"] / summary["primary_miss_count"],
        np.nan,
    )

    summary = summary.merge(
        pair_only[["method_id", "mean_scene_expected_quality", "gain_vs_single_reference"]],
        on="method_id",
        how="left",
    )
    best_box_score = float(pair_only.loc[pair_only["method_id"] == "best_box", "mean_scene_expected_quality"].iloc[0])
    noisy_or_score = float(
        pair_only.loc[pair_only["method_id"] == "noisy_or_best_iou", "mean_scene_expected_quality"].iloc[0]
    )
    summary["gap_vs_best_box"] = summary["mean_scene_expected_quality"] - best_box_score
    summary["gap_vs_noisy_or_best_iou"] = summary["mean_scene_expected_quality"] - noisy_or_score
    return summary.sort_values("mean_scene_expected_quality", ascending=False).reset_index(drop=True)


def build_pairwise_method_comparison(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_method in METHODS:
        left_scores = pair_df[f"{left_method}_score"].to_numpy(dtype=float)
        for right_method in METHODS:
            right_scores = pair_df[f"{right_method}_score"].to_numpy(dtype=float)
            rows.append(
                {
                    "left_method_id": left_method,
                    "left_method_label": method_label(left_method),
                    "right_method_id": right_method,
                    "right_method_label": method_label(right_method),
                    "win_rate_left_over_right": float(np.mean(left_scores > right_scores)),
                    "tie_rate": float(np.mean(np.isclose(left_scores, right_scores))),
                    "mean_score_gap_left_minus_right": float(np.mean(left_scores - right_scores)),
                }
            )
    return pd.DataFrame(rows)


def plot_overall_method_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    pair_only = summary_df.loc[summary_df["drone_count"] == 2].copy()
    pair_only = pair_only.sort_values("mean_scene_expected_quality", ascending=False)

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    bars = ax.bar(pair_only["method_label"], pair_only["mean_scene_expected_quality"], color="#4c78a8")
    for bar, gain in zip(bars, pair_only["gain_vs_single_reference"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}\n({gain:+.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Scene-balanced expected target quality")
    ax.set_title("Geometry-aware fusion baselines versus the single-view reference")
    ax.tick_params(axis="x", rotation=24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_tradeoff(value_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    colors = {
        "Best-view rescue": "#4c78a8",
        "Probabilistic accumulation": "#f58518",
        "Conservative corroboration": "#54a24b",
        "Geometry-aware selection": "#e45756",
        "Geometry-aware accumulation": "#72b7b2",
    }
    sizes = 900 * np.maximum(value_df["gain_vs_single_reference"].to_numpy(dtype=float), 0.005)
    ax.scatter(
        value_df["mean_lift_vs_best_constituent"],
        value_df["rescue_rate_given_primary_miss"],
        s=sizes,
        c=[colors.get(family, "#999999") for family in value_df["method_family"]],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in value_df.iterrows():
        ax.text(
            float(row["mean_lift_vs_best_constituent"]) + 0.001,
            float(row["rescue_rate_given_primary_miss"]) + 0.005,
            row["method_label"],
            fontsize=8,
        )
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Mean lift versus best constituent single")
    ax.set_ylabel("Rescue rate given primary miss")
    ax.set_title("What each fusion technique adds: corroboration versus rescue")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_actual(prior_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.scatter(
        prior_df["predicted_geometry_prior"],
        prior_df["target_strict_quality_iou50"],
        s=18,
        alpha=0.45,
        color="#72b7b2",
    )
    ax.set_xlabel("Predicted geometry prior")
    ax.set_ylabel("Actual single-view strict quality")
    ax.set_title("Geometry prior versus actual single-view quality")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_gap_heatmap(pairwise_df: pd.DataFrame, output_path: Path) -> None:
    labels = [method_label(method_id) for method_id in METHODS]
    matrix = np.zeros((len(METHODS), len(METHODS)), dtype=float)
    for i, left_method in enumerate(METHODS):
        for j, right_method in enumerate(METHODS):
            value = pairwise_df.loc[
                (pairwise_df["left_method_id"] == left_method) & (pairwise_df["right_method_id"] == right_method),
                "mean_score_gap_left_minus_right",
            ].iloc[0]
            matrix[i, j] = float(value)

    fig, ax = plt.subplots(figsize=(9.8, 7.6))
    image = ax.imshow(matrix, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Mean pair-score gap: row method minus column method")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    prior_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    value_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    headline_df: pd.DataFrame,
) -> None:
    single_reference = overall_df.loc[overall_df["method_id"] == "single_view_reference"].iloc[0]
    pair_best = overall_df.loc[overall_df["drone_count"] == 2].sort_values(
        "mean_scene_expected_quality", ascending=False
    )
    best_method = pair_best.iloc[0]
    noisy_or_score = float(
        pair_best.loc[pair_best["method_id"] == "noisy_or_best_iou", "mean_scene_expected_quality"].iloc[0]
    )
    best_geometry_method = overall_df.loc[
        overall_df["method_id"].isin(
            [
                "geometry_prior_selector",
                "geometry_calibrated_selector",
                "viewpoint_cell_prior_selector",
                "geometry_weighted_or_best_iou",
                "geometry_weighted_or_mean_iou",
                "viewpoint_cell_weighted_or_best_iou",
                "hybrid_geometry_cell_weighted_or_best_iou",
            ]
        )
    ].sort_values("mean_scene_expected_quality", ascending=False).iloc[0]

    overall_geometry_corr = float(
        np.corrcoef(
            prior_df["predicted_geometry_prior"].to_numpy(dtype=float),
            prior_df["target_strict_quality_iou50"].to_numpy(dtype=float),
        )[0, 1]
    )
    overall_cell_corr = float(
        np.corrcoef(
            prior_df["predicted_viewpoint_cell_prior"].to_numpy(dtype=float),
            prior_df["target_strict_quality_iou50"].to_numpy(dtype=float),
        )[0, 1]
    )
    overall_hybrid_corr = float(
        np.corrcoef(
            prior_df["predicted_hybrid_prior"].to_numpy(dtype=float),
            prior_df["target_strict_quality_iou50"].to_numpy(dtype=float),
        )[0, 1]
    )
    best_geometry_value_row = value_df.loc[value_df["method_id"] == best_geometry_method["method_id"]].iloc[0]
    noisy_or_value_row = value_df.loc[value_df["method_id"] == "noisy_or_best_iou"].iloc[0]
    pairwise_against_noisy_or = pairwise_df.loc[
        (pairwise_df["left_method_id"] == best_geometry_method["method_id"])
        & (pairwise_df["right_method_id"] == "noisy_or_best_iou")
    ].iloc[0]

    lines = [
        "# Geometry-Aware Fusion Report",
        "",
        "## Purpose",
        "",
        "This analysis tests geometry-aware target fusion baselines on the same scene-view evaluation base used by the earlier multiview comparisons.",
        "It uses manifest-derived camera-target geometry to predict a per-view reliability prior, then injects that prior into view selection and late fusion.",
        "",
        "## Important boundary",
        "",
        "This is a geometry-aware reliability baseline, not full 3D outline reprojection.",
        "It uses camera-target pose metadata but does not yet use camera intrinsics, a 3D object mesh, or explicit shared-plane box reprojection.",
        "",
        "## Geometry prior quality",
        "",
        f"- Geometry-regression prior correlation: `{overall_geometry_corr:.4f}`",
        f"- Viewpoint-cell prior correlation: `{overall_cell_corr:.4f}`",
        f"- Hybrid prior correlation: `{overall_hybrid_corr:.4f}`",
        f"- Mean geometry weight: `{prior_df['geometry_weight'].mean():.4f}`",
        f"- Mean viewpoint-cell weight: `{prior_df['viewpoint_cell_weight'].mean():.4f}`",
        f"- Mean hybrid weight: `{prior_df['hybrid_weight'].mean():.4f}`",
        "",
        "## Single-view reference",
        "",
        f"- Scene-balanced single-view reference quality: `{single_reference['mean_scene_expected_quality']:.4f}`",
        "",
        "## Best overall two-view method",
        "",
        f"- Best method: `{best_method['method_label']}`",
        f"- Method family: `{best_method['method_family']}`",
        f"- Scene-balanced two-view quality: `{best_method['mean_scene_expected_quality']:.4f}`",
        f"- Gain versus single-view reference: `{best_method['gain_vs_single_reference']:+.4f}`",
        "",
        "## Best geometry-aware two-view method",
        "",
        f"- Best geometry-aware method: `{best_geometry_method['method_label']}`",
        f"- Scene-balanced two-view quality: `{best_geometry_method['mean_scene_expected_quality']:.4f}`",
        f"- Gain versus single-view reference: `{best_geometry_method['gain_vs_single_reference']:+.4f}`",
        f"- Gap versus `Noisy-OR + best IoU`: `{best_geometry_method['mean_scene_expected_quality'] - noisy_or_score:+.4f}`",
        f"- Pair-row win rate versus `Noisy-OR + best IoU`: `{float(pairwise_against_noisy_or['win_rate_left_over_right']):.4f}`",
        f"- Mean pair-score gap versus `Noisy-OR + best IoU`: `{float(pairwise_against_noisy_or['mean_score_gap_left_minus_right']):+.4f}`",
        "",
        "## Interpreting the geometry-aware methods",
        "",
        "- `Geometry prior selector` chooses the view with the strongest predicted geometry reliability prior and keeps that view's actual target quality.",
        "- `Geometry-calibrated selector` chooses the view with the strongest confidence after geometry reweighting.",
        "- `Viewpoint-cell prior selector` uses leave-one-scene-out class-specific evidence for the exact lattice cell rather than a smooth regression surface.",
        "- `Geometry-weighted OR` keeps the noisy-OR late-fusion logic, but calibrates each view's confidence with a leave-one-scene-out geometry prior.",
        "- `Viewpoint-cell OR` and `Hybrid geometry+cell OR` test whether a discrete lattice prior helps more than a smooth geometric prior.",
        "",
        "## How to show the value of one technique over another",
        "",
        f"- Use `overall_method_summary.csv` and `overall_geometry_method_comparison.png` to show end-to-end quality. Right now the best deployable geometry-aware method is `{best_geometry_method['method_label']}` at `{best_geometry_method['mean_scene_expected_quality']:.4f}`, but its gap to `Noisy-OR + best IoU` is essentially zero.",
        f"- Use `method_value_summary.csv` and `method_tradeoff_scatter.png` to separate rescue from true corroboration. For the best geometry-aware method, mean lift versus best constituent is `{best_geometry_value_row['mean_lift_vs_best_constituent']:+.4f}` and rescue rate is `{best_geometry_value_row['rescue_rate_given_primary_miss']:.4f}`.",
        f"- Use `pairwise_method_comparison.csv` and `pairwise_method_gap_heatmap.png` to answer the direct question 'which method beats which'. Here the key nuance is that `{best_geometry_method['method_label']}` wins on `{float(pairwise_against_noisy_or['win_rate_left_over_right']):.4f}` of pair rows against `Noisy-OR + best IoU`, yet its mean pair-score gap stays `{float(pairwise_against_noisy_or['mean_score_gap_left_minus_right']):+.4f}`, so the result should be presented as a near-tie rather than a decisive win.",
        "",
        "## Strong added viewpoints by method",
        "",
    ]

    if headline_df.empty:
        lines.append("No added-view headline rows met the support threshold.")
    else:
        selected_methods = [
            "geometry_prior_selector",
            "geometry_calibrated_selector",
            "viewpoint_cell_prior_selector",
            "geometry_weighted_or_best_iou",
            "geometry_weighted_or_mean_iou",
            "viewpoint_cell_weighted_or_best_iou",
            "hybrid_geometry_cell_weighted_or_best_iou",
        ]
        for method_id in selected_methods:
            subset = headline_df.loc[headline_df["method_id"] == method_id].nlargest(5, "mean_lift_vs_primary")
            if subset.empty:
                continue
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
            "- `geometry_priors.csv`",
            "- `geometry_prior_diagnostics.csv`",
            "- `overall_method_summary.csv`",
            "- `method_value_summary.csv`",
            "- `pairwise_method_comparison.csv`",
            "- `ordered_pair_gain_summary.csv`",
            "- `added_viewpoint_summary.csv`",
            "- `added_viewpoint_headlines.csv`",
            "- `overall_geometry_method_comparison.png`",
            "- `predicted_prior_vs_actual.png`",
            "- `method_tradeoff_scatter.png`",
            "- `pairwise_method_gap_heatmap.png`",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_view_path = Path(args.scene_view_csv)
    geometry_path = Path(args.geometry_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    scene_df = load_joined_scene_records(scene_view_path, geometry_path)
    prior_df = fit_predict_geometry_priors(scene_df)
    diagnostics_df = build_prior_diagnostics(prior_df)
    pair_df = build_pair_rows(prior_df)
    overall_df = build_scene_balanced_summary(prior_df, pair_df)
    ordered_rows = build_ordered_pair_gain_rows(prior_df)
    pair_gain_summary, added_view_summary, headline_df = summarize_added_viewpoints(
        ordered_rows, args.min_added_view_support
    )
    value_df = build_method_value_summary(overall_df, ordered_rows)
    pairwise_df = build_pairwise_method_comparison(pair_df)

    prior_df.to_csv(output_dir / "geometry_priors.csv", index=False)
    diagnostics_df.to_csv(output_dir / "geometry_prior_diagnostics.csv", index=False)
    overall_df.to_csv(output_dir / "overall_method_summary.csv", index=False)
    value_df.to_csv(output_dir / "method_value_summary.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_method_comparison.csv", index=False)
    pair_df.to_csv(output_dir / "pair_method_rows.csv", index=False)
    ordered_rows.to_csv(output_dir / "ordered_pair_method_rows.csv", index=False)
    pair_gain_summary.to_csv(output_dir / "ordered_pair_gain_summary.csv", index=False)
    added_view_summary.to_csv(output_dir / "added_viewpoint_summary.csv", index=False)
    headline_df.to_csv(output_dir / "added_viewpoint_headlines.csv", index=False)

    plot_overall_method_summary(overall_df, output_dir / "overall_geometry_method_comparison.png")
    plot_prior_vs_actual(prior_df, output_dir / "predicted_prior_vs_actual.png")
    plot_method_tradeoff(value_df, output_dir / "method_tradeoff_scatter.png")
    plot_pairwise_gap_heatmap(pairwise_df, output_dir / "pairwise_method_gap_heatmap.png")
    write_report(
        output_dir / "geometry_aware_fusion_report.md",
        prior_df,
        diagnostics_df,
        overall_df,
        value_df,
        pairwise_df,
        headline_df,
    )

    print(f"Wrote geometry-aware fusion outputs to: {output_dir}")


if __name__ == "__main__":
    main()
