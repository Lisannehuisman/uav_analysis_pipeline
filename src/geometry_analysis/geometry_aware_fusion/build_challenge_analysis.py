from __future__ import annotations

import argparse
import math
from itertools import permutations
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIOR_CSV = ROOT / "geometry_aware_fusion_analysis" / "outputs" / "geometry_priors.csv"
OUTPUT_DIR = ROOT / "geometry_aware_fusion_analysis" / "outputs"

METHODS = [
    "primary_single_reference",
    "mean_quality",
    "best_box",
    "geometry_prior_selector",
    "geometry_calibrated_selector",
    "viewpoint_cell_prior_selector",
    "noisy_or_best_iou",
    "support_weighted_or",
    "geometry_weighted_or_best_iou",
    "viewpoint_cell_weighted_or_best_iou",
    "hybrid_geometry_cell_weighted_or_best_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build challenge-slice analyses for target-centric multiview comparison."
    )
    parser.add_argument("--input-csv", default=str(PRIOR_CSV), help="Joined geometry prior CSV.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
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
        "primary_single_reference": "Primary single reference",
        "mean_quality": "Mean quality",
        "best_box": "Best box (max)",
        "geometry_prior_selector": "Geometry prior selector",
        "geometry_calibrated_selector": "Geometry-calibrated selector",
        "viewpoint_cell_prior_selector": "Viewpoint-cell selector",
        "noisy_or_best_iou": "Noisy-OR + best IoU",
        "support_weighted_or": "Support-weighted OR",
        "geometry_weighted_or_best_iou": "Geometry-weighted OR + best IoU",
        "viewpoint_cell_weighted_or_best_iou": "Viewpoint-cell OR + best IoU",
        "hybrid_geometry_cell_weighted_or_best_iou": "Hybrid geometry+cell OR",
    }[method_id]


def method_family(method_id: str) -> str:
    return {
        "primary_single_reference": "Reference",
        "mean_quality": "Naive averaging",
        "best_box": "Selection",
        "geometry_prior_selector": "Geometry-aware selection",
        "geometry_calibrated_selector": "Geometry-aware selection",
        "viewpoint_cell_prior_selector": "Geometry-aware selection",
        "noisy_or_best_iou": "Evidence accumulation",
        "support_weighted_or": "Evidence accumulation",
        "geometry_weighted_or_best_iou": "Geometry-aware accumulation",
        "viewpoint_cell_weighted_or_best_iou": "Geometry-aware accumulation",
        "hybrid_geometry_cell_weighted_or_best_iou": "Geometry-aware accumulation",
    }[method_id]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
        "predicted_geometry_prior",
        "geometry_weight",
        "predicted_viewpoint_cell_prior",
        "viewpoint_cell_weight",
        "predicted_hybrid_prior",
        "hybrid_weight",
        "bbox_area_norm",
        "target_detected",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


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

    if method_id == "primary_single_reference":
        return qualities[0]
    if method_id == "mean_quality":
        return float(np.mean(qualities)) if qualities else 0.0
    if method_id == "best_box":
        return max(qualities) if qualities else 0.0
    if method_id == "geometry_prior_selector":
        selected_index = int(np.argmax(geometry_priors))
        return qualities[selected_index]
    if method_id == "geometry_calibrated_selector":
        selected_index = int(np.argmax(geometry_selection_scores))
        return qualities[selected_index]
    if method_id == "viewpoint_cell_prior_selector":
        selected_index = int(np.argmax(cell_priors))
        return qualities[selected_index]
    if method_id == "noisy_or_best_iou":
        return noisy_or(confidences) * (max(ious) if ious else 0.0)
    if method_id == "support_weighted_or":
        return noisy_or(confidences) * (float(np.mean(ious)) if ious else 0.0) * support_ratio
    if method_id == "geometry_weighted_or_best_iou":
        return noisy_or(weighted_confidences) * (max(ious) if ious else 0.0)
    if method_id == "viewpoint_cell_weighted_or_best_iou":
        return noisy_or(cell_weighted_confidences) * (max(ious) if ious else 0.0)
    if method_id == "hybrid_geometry_cell_weighted_or_best_iou":
        return noisy_or(hybrid_weighted_confidences) * (max(ious) if ious else 0.0)
    raise KeyError(method_id)


def build_ordered_pair_rows(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scene_mean_map = scene_df.groupby("scene_key")["target_strict_quality_iou50"].mean().to_dict()

    for scene_key, scene_group in scene_df.groupby("scene_key", sort=True):
        records = scene_group.to_dict("records")
        scene_mean_quality = float(scene_mean_map[scene_key])
        for primary, secondary in permutations(records, 2):
            pair_records = [primary, secondary]
            row = {
                "scene_key": scene_key,
                "target_class": primary["target_class"],
                "primary_file_name": primary["file_name"],
                "secondary_file_name": secondary["file_name"],
                "primary_viewpoint": primary["viewpoint"],
                "secondary_viewpoint": secondary["viewpoint"],
                "primary_radius_name": primary["radius_name"],
                "primary_elevation_name": primary["elevation_name"],
                "primary_bbox_area_norm": float(primary["bbox_area_norm"]),
                "primary_single_quality": float(primary["target_strict_quality_iou50"]),
                "primary_confidence": float(primary["target_match_confidence_iou50"]),
                "primary_iou": float(primary["target_match_iou_at_confidence_iou50"]),
                "primary_detected": int(float(primary["target_detected"]) > 0.0),
                "scene_mean_quality": scene_mean_quality,
                "best_constituent_single": max(
                    float(primary["target_strict_quality_iou50"]),
                    float(secondary["target_strict_quality_iou50"]),
                ),
            }
            for method_id in METHODS:
                row[f"{method_id}_score"] = coalition_score(pair_records, method_id)
            rows.append(row)
    return pd.DataFrame(rows)


def build_slice_config(view_df: pd.DataFrame) -> list[dict[str, object]]:
    quality_q25 = float(view_df["target_strict_quality_iou50"].quantile(0.25))
    iou_q25 = float(view_df["target_match_iou_at_confidence_iou50"].quantile(0.25))
    bbox_q25 = float(view_df["bbox_area_norm"].quantile(0.25))
    scene_mean_q25 = float(view_df.groupby("scene_key")["target_strict_quality_iou50"].mean().quantile(0.25))

    return [
        {
            "slice_id": "all_primary_views",
            "slice_label": "All primary views",
            "slice_description": "All ordered primary->secondary pairs in the current cache.",
            "filter_expr": lambda df: pd.Series(True, index=df.index),
        },
        {
            "slice_id": "weak_primary_quality_q25",
            "slice_label": "Weak primary quality (Q1)",
            "slice_description": f"Primary view target_strict_quality_iou50 <= dataset Q1 ({quality_q25:.4f}).",
            "filter_expr": lambda df, q=quality_q25: df["primary_single_quality"] <= q,
        },
        {
            "slice_id": "small_target_bbox_q25",
            "slice_label": "Small target box (Q1)",
            "slice_description": f"Primary bbox_area_norm <= dataset Q1 ({bbox_q25:.4f}).",
            "filter_expr": lambda df, q=bbox_q25: df["primary_bbox_area_norm"] <= q,
        },
        {
            "slice_id": "low_primary_iou_q25",
            "slice_label": "Low primary IoU (Q1)",
            "slice_description": f"Primary matched IoU <= dataset Q1 ({iou_q25:.4f}).",
            "filter_expr": lambda df, q=iou_q25: df["primary_iou"] <= q,
        },
        {
            "slice_id": "far_primary_views",
            "slice_label": "Far primary views",
            "slice_description": "Primary radius_name == far.",
            "filter_expr": lambda df: df["primary_radius_name"].astype(str) == "far",
        },
        {
            "slice_id": "hard_scene_mean_q25",
            "slice_label": "Hard scenes by mean quality (Q1)",
            "slice_description": f"Scene mean single-view quality <= dataset Q1 ({scene_mean_q25:.4f}).",
            "filter_expr": lambda df, q=scene_mean_q25: df["scene_mean_quality"] <= q,
        },
        {
            "slice_id": "primary_miss_only",
            "slice_label": "Primary miss only",
            "slice_description": "Primary target_detected == 0, so any positive pair score is a rescue case.",
            "filter_expr": lambda df: df["primary_detected"] == 0,
        },
    ]


def summarize_slices(pair_df: pd.DataFrame, slice_config: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    headline_rows: list[dict[str, object]] = []

    for config in slice_config:
        mask = config["filter_expr"](pair_df)
        slice_df = pair_df.loc[mask].copy()
        if slice_df.empty:
            continue

        for method_id in METHODS:
            score_col = f"{method_id}_score"
            mean_score = float(slice_df[score_col].mean())
            primary_miss_rows = int((slice_df["primary_detected"] == 0).sum())
            rescue_rate = float(
                (slice_df.loc[slice_df["primary_detected"] == 0, score_col] > 0.0).mean()
            ) if primary_miss_rows > 0 else math.nan
            summary_rows.append(
                {
                    "slice_id": config["slice_id"],
                    "slice_label": config["slice_label"],
                    "slice_description": config["slice_description"],
                    "method_id": method_id,
                    "method_label": method_label(method_id),
                    "method_family": method_family(method_id),
                    "pair_row_count": int(len(slice_df)),
                    "scene_count": int(slice_df["scene_key"].nunique()),
                    "primary_miss_row_count": primary_miss_rows,
                    "mean_primary_single_quality": float(slice_df["primary_single_quality"].mean()),
                    "mean_best_constituent_single": float(slice_df["best_constituent_single"].mean()),
                    "mean_pair_score": mean_score,
                    "gain_vs_primary": mean_score - float(slice_df["primary_single_quality"].mean()),
                    "gain_vs_best_constituent": mean_score - float(slice_df["best_constituent_single"].mean()),
                    "rescue_rate_given_primary_miss": rescue_rate,
                }
            )

        slice_summary = pd.DataFrame([row for row in summary_rows if row["slice_id"] == config["slice_id"]]).copy()
        slice_summary = slice_summary.sort_values("mean_pair_score", ascending=False).reset_index(drop=True)
        slice_summary["rank_desc"] = np.arange(1, len(slice_summary) + 1)

        best_row = slice_summary.iloc[0]
        best_box_row = slice_summary.loc[slice_summary["method_id"] == "best_box"].iloc[0]
        noisy_or_row = slice_summary.loc[slice_summary["method_id"] == "noisy_or_best_iou"].iloc[0]
        best_selector_row = slice_summary.loc[
            slice_summary["method_family"].isin(["Selection", "Geometry-aware selection"])
        ].sort_values("mean_pair_score", ascending=False).iloc[0]
        best_accum_row = slice_summary.loc[
            slice_summary["method_family"].isin(["Evidence accumulation", "Geometry-aware accumulation"])
        ].sort_values("mean_pair_score", ascending=False).iloc[0]

        headline_rows.append(
            {
                "slice_id": config["slice_id"],
                "slice_label": config["slice_label"],
                "slice_description": config["slice_description"],
                "pair_row_count": int(len(slice_df)),
                "scene_count": int(slice_df["scene_key"].nunique()),
                "primary_miss_row_count": int((slice_df["primary_detected"] == 0).sum()),
                "mean_primary_single_quality": float(slice_df["primary_single_quality"].mean()),
                "mean_best_constituent_single": float(slice_df["best_constituent_single"].mean()),
                "best_method_id": best_row["method_id"],
                "best_method_label": best_row["method_label"],
                "best_method_score": best_row["mean_pair_score"],
                "best_selector_method_id": best_selector_row["method_id"],
                "best_selector_score": best_selector_row["mean_pair_score"],
                "best_accum_method_id": best_accum_row["method_id"],
                "best_accum_score": best_accum_row["mean_pair_score"],
                "accum_minus_best_box": float(best_accum_row["mean_pair_score"] - best_box_row["mean_pair_score"]),
                "accum_minus_noisy_or": float(best_accum_row["mean_pair_score"] - noisy_or_row["mean_pair_score"]),
                "best_box_score": float(best_box_row["mean_pair_score"]),
                "noisy_or_score": float(noisy_or_row["mean_pair_score"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["rank_desc_within_slice"] = (
            summary_df.groupby("slice_id")["mean_pair_score"].rank(method="min", ascending=False).astype(int)
        )
    headline_df = pd.DataFrame(headline_rows)
    return summary_df, headline_df


def font_candidates(bold: bool = False) -> list[str]:
    if bold:
        return [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
        ]
    return [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in font_candidates(bold=bold):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def lerp_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round((1.0 - t) * a + t * b)) for a, b in zip(color_a, color_b))


def build_heatmap_png(summary_df: pd.DataFrame, output_path: Path) -> None:
    method_order = [
        "primary_single_reference",
        "mean_quality",
        "best_box",
        "geometry_prior_selector",
        "geometry_calibrated_selector",
        "viewpoint_cell_prior_selector",
        "noisy_or_best_iou",
        "support_weighted_or",
        "geometry_weighted_or_best_iou",
        "viewpoint_cell_weighted_or_best_iou",
        "hybrid_geometry_cell_weighted_or_best_iou",
    ]
    slice_order = [
        "all_primary_views",
        "weak_primary_quality_q25",
        "small_target_bbox_q25",
        "low_primary_iou_q25",
        "far_primary_views",
        "hard_scene_mean_q25",
        "primary_miss_only",
    ]
    label_map = dict(zip(summary_df["slice_id"], summary_df["slice_label"]))
    pivot = summary_df.pivot_table(index="method_id", columns="slice_id", values="mean_pair_score")
    gain_pivot = summary_df.pivot_table(index="method_id", columns="slice_id", values="gain_vs_primary")

    width = 2100
    height = 980
    left_margin = 470
    top_margin = 180
    cell_w = 220
    cell_h = 58

    image = Image.new("RGB", (width, height), hex_rgb("#f7f7f4"))
    draw = ImageDraw.Draw(image)
    title_font = load_font(36, bold=True)
    header_font = load_font(20, bold=True)
    cell_font = load_font(17)

    draw.text((40, 35), "Challenge-Slice Method Comparison", font=title_font, fill=(17, 17, 17))
    subtitle = (
        "Each column isolates a harder subset of the target-centric benchmark. Cell text shows mean pair score and gain versus "
        "the primary single view. Darker cells indicate higher mean pair score within this figure."
    )
    draw.multiline_text((40, 88), "\n".join(textwrap.wrap(subtitle, width=120)), font=load_font(18), fill=(60, 60, 60), spacing=5)

    value_min = float(summary_df["mean_pair_score"].min())
    value_max = float(summary_df["mean_pair_score"].max())
    low_color = hex_rgb("#f8e8e8")
    high_color = hex_rgb("#dfeee0")

    for col_idx, slice_id in enumerate(slice_order):
        if slice_id not in pivot.columns:
            continue
        x = left_margin + (col_idx * cell_w)
        header = label_map[slice_id]
        wrapped = "\n".join(textwrap.wrap(header, width=16))
        draw.multiline_text((x + 8, top_margin - 90), wrapped, font=header_font, fill=(17, 17, 17), spacing=4)

    for row_idx, method_id in enumerate(method_order):
        if method_id not in pivot.index:
            continue
        y = top_margin + (row_idx * cell_h)
        method_text = method_label(method_id)
        draw.text((40, y + 16), method_text, font=cell_font, fill=(17, 17, 17))
        draw.text((260, y + 16), method_family(method_id), font=cell_font, fill=(110, 110, 110))

        for col_idx, slice_id in enumerate(slice_order):
            if slice_id not in pivot.columns:
                continue
            x = left_margin + (col_idx * cell_w)
            value = float(pivot.loc[method_id, slice_id])
            gain = float(gain_pivot.loc[method_id, slice_id])
            t = 0.5 if value_max <= value_min else ((value - value_min) / (value_max - value_min))
            fill = lerp_color(low_color, high_color, t)
            draw.rounded_rectangle((x, y, x + cell_w - 10, y + cell_h - 8), radius=10, fill=fill, outline=(180, 180, 180), width=1)
            draw.text((x + 10, y + 7), f"{value:.4f}", font=cell_font, fill=(17, 17, 17))
            draw.text((x + 10, y + 30), f"dP {gain:+.4f}", font=cell_font, fill=(70, 70, 70))

    image.save(output_path)


def build_report(summary_df: pd.DataFrame, headline_df: pd.DataFrame, output_path: Path) -> None:
    all_slice = headline_df.loc[headline_df["slice_id"] == "all_primary_views"].iloc[0]
    weak_slice = headline_df.loc[headline_df["slice_id"] == "weak_primary_quality_q25"].iloc[0]
    small_slice = headline_df.loc[headline_df["slice_id"] == "small_target_bbox_q25"].iloc[0]
    miss_slice = headline_df.loc[headline_df["slice_id"] == "primary_miss_only"].iloc[0]
    far_slice = headline_df.loc[headline_df["slice_id"] == "far_primary_views"].iloc[0]

    lines = [
        "# Challenge Analysis Report",
        "",
        "## Purpose",
        "",
        "This analysis tests whether the centered-target benchmark becomes more discriminative when evaluation is restricted to the harder tail of the current cache.",
        "Instead of claiming general scene-wide detection difficulty, it isolates target-centric challenge slices that are still honest to the current dataset design.",
        "All numbers in this report are based on ordered primary->secondary pair rows, so they are challenge-slice robustness probes rather than replacements for the scene-balanced summaries used elsewhere.",
        "",
        "## Slice Design",
        "",
        "- `All primary views`: the full ordered-pair analysis baseline.",
        "- `Weak primary quality (Q1)`: primary views already in the weakest quartile by target strict quality.",
        "- `Small target box (Q1)`: primary views where the projected target box is in the smallest quartile.",
        "- `Low primary IoU (Q1)`: primary views where matched localization is in the weakest quartile.",
        "- `Far primary views`: primary radius is `far`.",
        "- `Hard scenes by mean quality (Q1)`: scenes whose mean single-view quality is in the weakest quartile.",
        "- `Primary miss only`: rescue-only rows where the primary view failed to detect the target.",
        "",
        "## Main Reading Rule",
        "",
        "The main question is not only which method wins overall, but whether the gap between accumulation-based methods and selection-based methods widens when the primary view is harder.",
        "",
        "## Headline Findings",
        "",
        f"- On `All primary views`, the best accumulation method is `{all_slice['best_accum_method_id']}` at `{all_slice['best_accum_score']:.4f}`. Its gap over `best_box` is `{all_slice['accum_minus_best_box']:+.4f}`.",
        f"- On `Weak primary quality (Q1)`, the best accumulation method is `{weak_slice['best_accum_method_id']}` at `{weak_slice['best_accum_score']:.4f}`. Its gap over `best_box` is `{weak_slice['accum_minus_best_box']:+.4f}`.",
        f"- On `Small target box (Q1)`, the best accumulation method is `{small_slice['best_accum_method_id']}` at `{small_slice['best_accum_score']:.4f}`. Its gap over `best_box` is `{small_slice['accum_minus_best_box']:+.4f}`.",
        f"- On `Far primary views`, the best accumulation method is `{far_slice['best_accum_method_id']}` at `{far_slice['best_accum_score']:.4f}`. Its gap over `best_box` is `{far_slice['accum_minus_best_box']:+.4f}`.",
        f"- On `Primary miss only`, the best accumulation method is `{miss_slice['best_accum_method_id']}` at `{miss_slice['best_accum_score']:.4f}`. In this slice every positive pair score is a rescue event, so several methods collapse to the same secondary-view success ceiling.",
        "",
        "## Interpretation",
        "",
        "If the accumulation-versus-selection gap grows on the harder slices, then the challenge analysis is doing useful work: it shows where a centered-target benchmark still differentiates multiview policies.",
        "If the top accumulation methods remain almost tied with each other, that should still be interpreted as a small within-family difference rather than evidence that the challenge slices fully remove the ceiling effect.",
        "",
        "## Files",
        "",
        "- `challenge_slice_method_summary.csv`",
        "- `challenge_slice_headlines.csv`",
        "- `challenge_slice_score_table.csv`",
        "- `challenge_slice_method_heatmap.png`",
        "",
        "## Recommended thesis use",
        "",
        "Use this analysis as a robustness probe: the benchmark remains target-centric and center-biased, but these slices show whether the main conclusions still hold when the primary observation is less favorable than average.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    view_df = load_data(input_path)
    pair_df = build_ordered_pair_rows(view_df)
    slice_config = build_slice_config(view_df)
    summary_df, headline_df = summarize_slices(pair_df, slice_config)

    summary_csv = output_dir / "challenge_slice_method_summary.csv"
    headline_csv = output_dir / "challenge_slice_headlines.csv"
    pivot_csv = output_dir / "challenge_slice_score_table.csv"
    report_md = output_dir / "challenge_analysis_report.md"
    heatmap_png = output_dir / "challenge_slice_method_heatmap.png"

    summary_df.sort_values(["slice_id", "mean_pair_score"], ascending=[True, False]).to_csv(summary_csv, index=False)
    headline_df.sort_values("slice_id").to_csv(headline_csv, index=False)
    pivot_df = summary_df.pivot_table(index="method_label", columns="slice_label", values="mean_pair_score")
    pivot_df.to_csv(pivot_csv)
    build_heatmap_png(summary_df, heatmap_png)
    build_report(summary_df, headline_df, report_md)


if __name__ == "__main__":
    main()
