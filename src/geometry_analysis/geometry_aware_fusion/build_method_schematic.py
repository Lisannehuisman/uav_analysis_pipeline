from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "geometry_aware_fusion_analysis" / "outputs"
OVERALL_CSV = OUTPUT_DIR / "overall_method_summary.csv"
VALUE_CSV = OUTPUT_DIR / "method_value_summary.csv"
PAIRWISE_CSV = OUTPUT_DIR / "pairwise_method_comparison.csv"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_box(ax, xy, wh, title, body, fc="#ffffff", ec="#444444", title_color="#111111", body_size=10):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.015, y + h - 0.04, title, fontsize=13, fontweight="bold", va="top", color=title_color)
    ax.text(x + 0.015, y + h - 0.08, body, fontsize=body_size, va="top", color="#222222", family="DejaVu Sans Mono")


def add_arrow(ax, start, end, color="#666666"):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=color)
    ax.add_patch(arrow)


def build_schematic_png(output_path: Path, overall_df: pd.DataFrame, value_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    pair_df = overall_df.loc[overall_df["drone_count"] == 2].copy()
    best_row = pair_df.sort_values("mean_scene_expected_quality", ascending=False).iloc[0]
    noisy_or_row = pair_df.loc[pair_df["method_id"] == "noisy_or_best_iou"].iloc[0]
    best_box_row = pair_df.loc[pair_df["method_id"] == "best_box"].iloc[0]
    cell_or_row = pair_df.loc[pair_df["method_id"] == "viewpoint_cell_weighted_or_best_iou"].iloc[0]
    pairwise_cell_vs_noisy = pairwise_df.loc[
        (pairwise_df["left_method_id"] == "viewpoint_cell_weighted_or_best_iou")
        & (pairwise_df["right_method_id"] == "noisy_or_best_iou")
    ].iloc[0]

    fig = plt.figure(figsize=(18, 13), dpi=200)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f7f4")

    ax.text(0.03, 0.97, "How The Multiview Methods Actually Combine Two Views", fontsize=24, fontweight="bold", va="top")
    ax.text(
        0.03,
        0.94,
        "Notation: q = target strict quality, p = target-match confidence, i = target-match IoU, g = smooth geometry prior, c = viewpoint-cell prior.",
        fontsize=12,
        va="top",
        color="#333333",
    )

    add_box(
        ax,
        (0.03, 0.78),
        (0.2, 0.12),
        "Input View 1",
        "view 1 provides:\nq1 = quality\np1 = confidence\ni1 = IoU\ng1 = geometry prior\nc1 = cell prior",
        fc="#eef6ff",
        ec="#5a8fcf",
    )
    add_box(
        ax,
        (0.26, 0.78),
        (0.2, 0.12),
        "Input View 2",
        "view 2 provides:\nq2 = quality\np2 = confidence\ni2 = IoU\ng2 = geometry prior\nc2 = cell prior",
        fc="#eef6ff",
        ec="#5a8fcf",
    )
    add_box(
        ax,
        (0.50, 0.77),
        (0.44, 0.14),
        "Big Conceptual Split",
        "Selection methods choose one view and keep only that view's score.\n\nAccumulation methods keep evidence from both views and combine it.\n\nThat is the main reason the OR-based methods are much stronger than the selector methods.",
        fc="#fff7e8",
        ec="#d39b2d",
    )
    add_arrow(ax, (0.23, 0.84), (0.50, 0.84))
    add_arrow(ax, (0.46, 0.84), (0.50, 0.84))

    add_box(
        ax,
        (0.03, 0.53),
        (0.29, 0.21),
        "Selection Family",
        wrap(
            "single-view reference: output = q1\n\n"
            "best_box: output = max(q1, q2)\n"
            "Interpretation: second view only rescues if it is better.\n\n"
            "geometry_prior_selector: choose argmax(g1, g2), output that q\n"
            "geometry_calibrated_selector: choose argmax(p1*g1, p2*g2), output that q\n"
            "viewpoint_cell_prior_selector: choose argmax(c1, c2), output that q\n\n"
            "Key limitation: a selector still throws one view away.", 47
        ),
        fc="#fef2f2",
        ec="#d66a6a",
    )
    add_box(
        ax,
        (0.35, 0.53),
        (0.29, 0.21),
        "Accumulation Family",
        wrap(
            "mean_quality: output = (q1 + q2) / 2\n"
            "Usually weak because a bad view drags a good one down.\n\n"
            "noisy_or_best_iou: output = [1-(1-p1)(1-p2)] * max(i1, i2)\n"
            "This accumulates evidence instead of discarding one view.\n\n"
            "noisy_or_mean_iou: same OR confidence term, but uses mean IoU instead of best IoU.", 48
        ),
        fc="#eef8ee",
        ec="#5d9f68",
    )
    add_box(
        ax,
        (0.67, 0.53),
        (0.28, 0.21),
        "Geometry-Aware Accumulation",
        wrap(
            "support_weighted_or: noisy-OR * mean(IoU) * support_ratio\n\n"
            "geometry_weighted_or_best_iou: use p1*g1 and p2*g2 inside noisy-OR\n\n"
            "viewpoint_cell_or_best_iou: use p1*c1 and p2*c2 inside noisy-OR\n\n"
            "hybrid_geometry+cell_or: use p1*h1 and p2*h2 with h = 0.5*g + 0.5*c", 42
        ),
        fc="#eef4f7",
        ec="#5d8899",
    )

    add_box(
        ax,
        (0.03, 0.28),
        (0.29, 0.19),
        "What The Scores Mean",
        wrap(
            "gain_vs_single_reference > 0:\nmethod improves the average mission result.\n\n"
            "mean_lift_vs_best_constituent > 0:\nmethod does more than pick the better single drone; it gets corroboration value from combining views.\n\n"
            "rescue_rate_given_primary_miss:\nif the first view misses, how often can the pair still recover?", 46
        ),
        fc="#ffffff",
        ec="#8b8b8b",
    )
    add_box(
        ax,
        (0.35, 0.28),
        (0.29, 0.19),
        "How To Read Your Current Results",
        wrap(
            f"best_box = {best_box_row['mean_scene_expected_quality']:.4f}\n"
            f"noisy_or_best_iou = {noisy_or_row['mean_scene_expected_quality']:.4f}\n"
            f"viewpoint_cell_or_best_iou = {cell_or_row['mean_scene_expected_quality']:.4f}\n\n"
            "So the big jump is from selection to OR-based evidence accumulation.\n"
            "Geometry refines the OR rule, but does not radically change the picture.", 43
        ),
        fc="#ffffff",
        ec="#8b8b8b",
    )
    add_box(
        ax,
        (0.67, 0.28),
        (0.28, 0.19),
        "Near-Tie Nuance",
        wrap(
            f"best overall method now: {best_row['method_label']}\n"
            f"scene-balanced score = {best_row['mean_scene_expected_quality']:.4f}\n\n"
            f"cell OR beats plain noisy-OR on {pairwise_cell_vs_noisy['win_rate_left_over_right']:.3f} of pair rows,\n"
            f"but mean pair-score gap is {pairwise_cell_vs_noisy['mean_score_gap_left_minus_right']:+.4f}.\n\n"
            "Interpretation: often locally better, but globally almost tied.", 41
        ),
        fc="#fff7e8",
        ec="#d39b2d",
    )

    add_box(
        ax,
        (0.03, 0.05),
        (0.92, 0.17),
        "Bottom-Line Interpretation",
        wrap(
            "1. The weak methods are the ones that either average too naively (`mean`) or choose only one view (`selectors`).\n"
            "2. The strong methods are the ones that keep both views alive and accumulate evidence (`Noisy-OR`, `Viewpoint-cell OR`, `Hybrid OR`).\n"
            "3. A geometry prior is useful as a reliability modifier, not as a replacement for fusion.\n"
            "4. In your current dataset, the main value of the second drone is still: rescue + corroboration through OR-style evidence accumulation.\n"
            "5. Geometry-aware methods mainly sharpen that accumulation rule instead of changing the fundamental story.", 140
        ),
        fc="#edf6ed",
        ec="#6ea36e",
        title_color="#103d10",
        body_size=11,
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_explainer_markdown(output_path: Path, overall_df: pd.DataFrame, value_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    pair_df = overall_df.loc[overall_df["drone_count"] == 2].copy().sort_values(
        "mean_scene_expected_quality", ascending=False
    )
    best_row = pair_df.iloc[0]
    noisy_or_row = pair_df.loc[pair_df["method_id"] == "noisy_or_best_iou"].iloc[0]
    best_box_row = pair_df.loc[pair_df["method_id"] == "best_box"].iloc[0]
    cell_or_row = pair_df.loc[pair_df["method_id"] == "viewpoint_cell_weighted_or_best_iou"].iloc[0]
    pairwise_cell_vs_noisy = pairwise_df.loc[
        (pairwise_df["left_method_id"] == "viewpoint_cell_weighted_or_best_iou")
        & (pairwise_df["right_method_id"] == "noisy_or_best_iou")
    ].iloc[0]

    lines = [
        "# Method Combination Schematic",
        "",
        f"![Method schematic]({(OUTPUT_DIR / 'method_combination_schematic.png').as_posix()})",
        "",
        "## Plain-Language Summary",
        "",
        "Every method starts with two views. The real difference is whether the method:",
        "",
        "- keeps only one view,",
        "- averages views naively, or",
        "- accumulates evidence from both views.",
        "",
        "That is the central interpretation key for your results.",
        "",
        "## What Each Family Really Does",
        "",
        "- `Single-view reference`: uses one view only. No multiview benefit is possible.",
        f"- `Best box (max)`: keeps the better of the two single-view target qualities. This is mostly a rescue rule. Current score: `{best_box_row['mean_scene_expected_quality']:.4f}`.",
        "- `Mean quality`: averages the two view qualities. This is too pessimistic when one view is weak.",
        "- `Noisy-OR + best IoU`: accumulates evidence from both views, then keeps the best localization term. This is the main strong non-geometry baseline.",
        "- `Support-weighted OR`: similar to noisy-OR, but more conservative because it rewards agreement/support.",
        "- `Geometry prior selector`: chooses the view that geometry predicts will be best. It still throws away one view.",
        "- `Geometry calibrated selector`: chooses the view with the strongest geometry-adjusted confidence. Still a selector.",
        "- `Viewpoint-cell prior selector`: chooses the view from the best exact lattice cell for that class. Still a selector.",
        "- `Geometry-weighted OR`: keeps both views, but geometry reweights their confidence before OR fusion.",
        "- `Viewpoint-cell OR`: keeps both views and uses class-specific lattice-cell reliability as the reweighting term.",
        "- `Hybrid geometry+cell OR`: combines the smooth geometry prior and the discrete cell prior before OR fusion.",
        "",
        "## How To Interpret The Current Results",
        "",
        f"- `Noisy-OR + best IoU` = `{noisy_or_row['mean_scene_expected_quality']:.4f}`",
        f"- `Viewpoint-cell OR + best IoU` = `{cell_or_row['mean_scene_expected_quality']:.4f}`",
        f"- Best overall method right now = `{best_row['method_label']}` at `{best_row['mean_scene_expected_quality']:.4f}`",
        "",
        "So the main performance jump comes from moving away from selection/averaging and toward evidence accumulation.",
        "",
        "## The Important Nuance",
        "",
        f"`Viewpoint-cell OR + best IoU` wins against plain `Noisy-OR + best IoU` on `{pairwise_cell_vs_noisy['win_rate_left_over_right']:.4f}` of pair rows, but its mean pair-score gap is only `{pairwise_cell_vs_noisy['mean_score_gap_left_minus_right']:+.4f}`.",
        "",
        "That means you should present it as a near-tie with slightly better local behavior, not as a dramatic breakthrough.",
        "",
        "## Best Way To Show This In Results",
        "",
        "- Use `overall_method_summary.csv` for the end-to-end ranking.",
        "- Use `method_value_summary.csv` to separate rescue from true corroboration.",
        "- Use `pairwise_method_comparison.csv` to show direct method-versus-method comparisons.",
        "- Use `added_viewpoint_headlines.csv` to show which specific second viewpoints help most.",
        "",
        "## Practical Reading Rule",
        "",
        "- If `mean_lift_vs_best_constituent` is near `0`, the method mainly acts like a smart selector/rescue rule.",
        "- If `mean_lift_vs_best_constituent` is clearly positive, the method gets real cross-view corroboration value.",
        "- If `rescue_rate_given_primary_miss` is high, the second drone is especially useful when the first drone fails.",
        "- If `win_rate` is high but `mean_score_gap` is near zero, the comparison is practically a tie.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    overall_df = pd.read_csv(OVERALL_CSV)
    value_df = pd.read_csv(VALUE_CSV)
    pairwise_df = pd.read_csv(PAIRWISE_CSV)

    build_schematic_png(OUTPUT_DIR / "method_combination_schematic.png", overall_df, value_df, pairwise_df)
    build_explainer_markdown(OUTPUT_DIR / "method_combination_explainer.md", overall_df, value_df, pairwise_df)

    print(f"Wrote schematic outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
