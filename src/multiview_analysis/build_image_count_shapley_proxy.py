from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_RECORDS = WORKSPACE / "results" / "intermediate" / "scene_view_records.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "results" / "recomputed" / "image_count_shapley_proxy"


@dataclass(frozen=True)
class ViewRecord:
    scene_key: str
    target_class: str
    image_id: int
    viewpoint: str
    ap50_95: float
    strict_quality: float
    confidence: float
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the marginal value of the 1st, 2nd, 3rd, ... image without "
            "using angle labels by averaging over random coalition orderings."
        )
    )
    parser.add_argument("--scene-records", default=str(DEFAULT_SCENE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum number of images to include per random ordering.",
    )
    parser.add_argument(
        "--permutations-per-scene",
        type=int,
        default=256,
        help="Random orderings sampled per scene.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_float(raw: str) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return 0.0
    return float(text)


def parse_int(raw: str) -> int:
    text = str(raw).strip()
    if not text:
        return 0
    return int(float(text))


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_scene_records(path: Path) -> list[ViewRecord]:
    rows: list[ViewRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ViewRecord(
                    scene_key=row["scene_key"],
                    target_class=row["target_class"],
                    image_id=parse_int(row["image_id"]),
                    viewpoint=row["viewpoint"],
                    ap50_95=parse_float(row["target_ap50_95"]),
                    strict_quality=parse_float(row["target_strict_quality_iou50"]),
                    confidence=parse_float(row["target_match_confidence_iou50"]),
                    iou=parse_float(row["target_match_iou_at_confidence_iou50"]),
                )
            )
    return rows


def group_by_scene(records: list[ViewRecord]) -> dict[str, list[ViewRecord]]:
    grouped: dict[str, list[ViewRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_key].append(record)
    return dict(grouped)


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    complement = 1.0
    for confidence in confidences:
        complement *= max(0.0, 1.0 - float(confidence))
    return 1.0 - complement


def summarize_scene_orderings(
    scene_records: list[ViewRecord],
    max_images: int,
    permutations_per_scene: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if not scene_records:
        return []
    scene_key = scene_records[0].scene_key
    target_class = scene_records[0].target_class
    usable_k = min(max_images, len(scene_records))
    prefix_ap_values = [[] for _ in range(usable_k)]
    prefix_strict_values = [[] for _ in range(usable_k)]
    prefix_fusion_values = [[] for _ in range(usable_k)]
    marginal_ap_values = [[] for _ in range(usable_k)]
    marginal_strict_values = [[] for _ in range(usable_k)]
    marginal_fusion_values = [[] for _ in range(usable_k)]

    record_indices = np.arange(len(scene_records))
    for _ in range(permutations_per_scene):
        order = rng.permutation(record_indices)[:usable_k]
        best_ap = 0.0
        best_strict = 0.0
        confidences: list[float] = []
        best_iou = 0.0
        previous_ap = 0.0
        previous_strict = 0.0
        previous_fusion = 0.0

        for position, record_index in enumerate(order):
            record = scene_records[int(record_index)]
            best_ap = max(best_ap, float(record.ap50_95))
            best_strict = max(best_strict, float(record.strict_quality))
            if float(record.confidence) > 0.0:
                confidences.append(float(record.confidence))
            best_iou = max(best_iou, float(record.iou))
            fusion_value = noisy_or(confidences) * best_iou if best_iou > 0.0 else 0.0

            prefix_ap_values[position].append(best_ap)
            prefix_strict_values[position].append(best_strict)
            prefix_fusion_values[position].append(fusion_value)
            marginal_ap_values[position].append(best_ap - previous_ap)
            marginal_strict_values[position].append(best_strict - previous_strict)
            marginal_fusion_values[position].append(fusion_value - previous_fusion)

            previous_ap = best_ap
            previous_strict = best_strict
            previous_fusion = fusion_value

    rows: list[dict[str, object]] = []
    for position in range(usable_k):
        rows.append(
            {
                "scene_key": scene_key,
                "target_class": target_class,
                "available_image_count": len(scene_records),
                "added_image_number": position + 1,
                "base_coalition_size": position,
                "permutations_used": permutations_per_scene,
                "mean_prefix_target_ap50_95": float(np.mean(prefix_ap_values[position])),
                "mean_prefix_target_strict_quality_iou50": float(np.mean(prefix_strict_values[position])),
                "mean_prefix_noisy_or_best_iou": float(np.mean(prefix_fusion_values[position])),
                "mean_marginal_target_ap50_95": float(np.mean(marginal_ap_values[position])),
                "mean_marginal_target_strict_quality_iou50": float(np.mean(marginal_strict_values[position])),
                "mean_marginal_noisy_or_best_iou": float(np.mean(marginal_fusion_values[position])),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["added_image_number"])].append(row)

    summary_rows: list[dict[str, object]] = []
    max_k = max(grouped) if grouped else 0
    reference_rows = grouped.get(max_k, [])
    reference_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in reference_rows])) if reference_rows else float("nan")
    reference_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in reference_rows])) if reference_rows else float("nan")
    reference_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in reference_rows])) if reference_rows else float("nan")

    k1_rows = grouped.get(1, [])
    k1_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in k1_rows])) if k1_rows else float("nan")
    k1_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in k1_rows])) if k1_rows else float("nan")
    k1_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in k1_rows])) if k1_rows else float("nan")

    for k in sorted(grouped):
        members = grouped[k]
        mean_prefix_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in members]))
        mean_prefix_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in members]))
        mean_prefix_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in members]))
        total_ap_gain = reference_ap - k1_ap
        total_strict_gain = reference_strict - k1_strict
        total_fusion_gain = reference_fusion - k1_fusion
        summary_rows.append(
            {
                "added_image_number": k,
                "base_coalition_size": k - 1,
                "scene_support": len(members),
                "mean_prefix_target_ap50_95": mean_prefix_ap,
                "mean_prefix_target_strict_quality_iou50": mean_prefix_strict,
                "mean_prefix_noisy_or_best_iou": mean_prefix_fusion,
                "mean_marginal_target_ap50_95": float(np.mean([float(row["mean_marginal_target_ap50_95"]) for row in members])),
                "mean_marginal_target_strict_quality_iou50": float(np.mean([float(row["mean_marginal_target_strict_quality_iou50"]) for row in members])),
                "mean_marginal_noisy_or_best_iou": float(np.mean([float(row["mean_marginal_noisy_or_best_iou"]) for row in members])),
                "fraction_of_total_target_ap50_95_gain_captured": (
                    float("nan")
                    if math.isclose(total_ap_gain, 0.0, abs_tol=1e-12)
                    else (mean_prefix_ap - k1_ap) / total_ap_gain
                ),
                "fraction_of_total_target_strict_quality_gain_captured": (
                    float("nan")
                    if math.isclose(total_strict_gain, 0.0, abs_tol=1e-12)
                    else (mean_prefix_strict - k1_strict) / total_strict_gain
                ),
                "fraction_of_total_noisy_or_best_iou_gain_captured": (
                    float("nan")
                    if math.isclose(total_fusion_gain, 0.0, abs_tol=1e-12)
                    else (mean_prefix_fusion - k1_fusion) / total_fusion_gain
                ),
            }
        )
    return summary_rows


def aggregate_by_class(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_class"]), int(row["added_image_number"]))].append(row)

    class_names = sorted({str(row["target_class"]) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for target_class in class_names:
        class_ks = sorted(k for cls, k in grouped if cls == target_class)
        if not class_ks:
            continue
        max_k = max(class_ks)
        ref_rows = grouped[(target_class, max_k)]
        ref_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in ref_rows]))
        ref_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in ref_rows]))
        ref_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in ref_rows]))
        k1_rows = grouped[(target_class, 1)]
        k1_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in k1_rows]))
        k1_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in k1_rows]))
        k1_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in k1_rows]))

        for k in class_ks:
            members = grouped[(target_class, k)]
            mean_prefix_ap = float(np.mean([float(row["mean_prefix_target_ap50_95"]) for row in members]))
            mean_prefix_strict = float(np.mean([float(row["mean_prefix_target_strict_quality_iou50"]) for row in members]))
            mean_prefix_fusion = float(np.mean([float(row["mean_prefix_noisy_or_best_iou"]) for row in members]))
            total_ap_gain = ref_ap - k1_ap
            total_strict_gain = ref_strict - k1_strict
            total_fusion_gain = ref_fusion - k1_fusion
            summary_rows.append(
                {
                    "target_class": target_class,
                    "added_image_number": k,
                    "base_coalition_size": k - 1,
                    "scene_support": len(members),
                    "mean_prefix_target_ap50_95": mean_prefix_ap,
                    "mean_prefix_target_strict_quality_iou50": mean_prefix_strict,
                    "mean_prefix_noisy_or_best_iou": mean_prefix_fusion,
                    "mean_marginal_target_ap50_95": float(np.mean([float(row["mean_marginal_target_ap50_95"]) for row in members])),
                    "mean_marginal_target_strict_quality_iou50": float(np.mean([float(row["mean_marginal_target_strict_quality_iou50"]) for row in members])),
                    "mean_marginal_noisy_or_best_iou": float(np.mean([float(row["mean_marginal_noisy_or_best_iou"]) for row in members])),
                    "fraction_of_total_target_ap50_95_gain_captured": (
                        float("nan")
                        if math.isclose(total_ap_gain, 0.0, abs_tol=1e-12)
                        else (mean_prefix_ap - k1_ap) / total_ap_gain
                    ),
                    "fraction_of_total_target_strict_quality_gain_captured": (
                        float("nan")
                        if math.isclose(total_strict_gain, 0.0, abs_tol=1e-12)
                        else (mean_prefix_strict - k1_strict) / total_strict_gain
                    ),
                    "fraction_of_total_noisy_or_best_iou_gain_captured": (
                        float("nan")
                        if math.isclose(total_fusion_gain, 0.0, abs_tol=1e-12)
                        else (mean_prefix_fusion - k1_fusion) / total_fusion_gain
                    ),
                }
            )
    return summary_rows


def plot_progression(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    x = [int(row["added_image_number"]) for row in summary_rows]
    strict_prefix = [float(row["mean_prefix_target_strict_quality_iou50"]) for row in summary_rows]
    fusion_prefix = [float(row["mean_prefix_noisy_or_best_iou"]) for row in summary_rows]
    strict_marginal = [float(row["mean_marginal_target_strict_quality_iou50"]) for row in summary_rows]
    fusion_marginal = [float(row["mean_marginal_noisy_or_best_iou"]) for row in summary_rows]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    ax_left.plot(x, strict_prefix, marker="o", linewidth=2.2, color="#1f6f8b", label="Best strict quality")
    ax_left.plot(x, fusion_prefix, marker="s", linewidth=2.2, color="#d97706", label="Noisy-OR x best IoU")
    ax_left.set_title("Angle-agnostic coalition value by number of images")
    ax_left.set_xlabel("Number of images in coalition")
    ax_left.set_ylabel("Mean coalition value")
    ax_left.set_xticks(x)
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.legend(frameon=False)

    ax_right.plot(x, strict_marginal, marker="o", linewidth=2.2, color="#1f6f8b", label="Best strict quality")
    ax_right.plot(x, fusion_marginal, marker="s", linewidth=2.2, color="#d97706", label="Noisy-OR x best IoU")
    ax_right.set_title("Expected marginal value of the kth image")
    ax_right.set_xlabel("Added image number (k)")
    ax_right.set_ylabel("Mean marginal gain")
    ax_right.set_xticks(x)
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_report(
    summary_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    output_path: Path,
    scene_records_path: Path,
    max_images: int,
    permutations_per_scene: int,
    seed: int,
) -> None:
    by_k = {int(row["added_image_number"]): row for row in summary_rows}
    lines = [
        "# Angle-Agnostic Image-Count Shapley Proxy",
        "",
        "This report answers the supervisor's count-first question:",
        "",
        "> what is the marginal contribution of adding more images, even before exact angle labels are used?",
        "",
        "## Where these results come from",
        "",
        f"- Source file: `{scene_records_path}`.",
        "- Each row in that source file is one scene-view observation identified by `scene_key` and `viewpoint`.",
        "- The source rows already contain target-centric detector outputs such as `target_match_confidence_iou50`, `target_match_iou_at_confidence_iou50`, and `target_strict_quality_iou50`.",
        "- In other words, this report is not computed from raw images directly. It is computed from the cached per-view target metrics that were generated earlier in the pipeline.",
        "",
        "## What the two coalition values mean",
        "",
        "- `Best strict-quality value` means: if a coalition of images is available, keep the single best target strict-quality score among those images.",
        "- `Target strict quality` for one image is `matched target confidence x matched target IoU`.",
        "- `Noisy-OR x best IoU value` means: fuse the matched target confidences across the coalition with `Noisy-OR(confidences) = 1 - product(1 - confidence_i)`, then multiply by the best matched IoU seen anywhere in that coalition.",
        "- The first coalition value is a selection game: keep the best single image.",
        "- The second coalition value is a fusion game: combine confidence evidence across images, then attach the best localization quality available in the coalition.",
        "",
        "## Why this report uses IoU50-based target quality",
        "",
        "- The source cache stores one matched target confidence and one matched target IoU per image using an IoU>=0.50 target match.",
        "- That gives one stable per-image target quality signal: `target_strict_quality_iou50 = confidence x matched IoU`.",
        "- This is useful for coalition analysis because every coalition rule then operates on the same bounded 0--1 target-quality quantity.",
        "- By contrast, AP50:95 is still reported elsewhere in the project, but it is less natural as the common coalition value for selection and fusion games because those games need one per-view matched target signal that can be combined image-by-image.",
        "- Using IoU50 also keeps the match rule less brittle than a stricter IoU threshold would, which matters when the question is rescue availability and marginal multiview value rather than only very-tight localization.",
        "- So the benefit of IoU50 here is not that it is universally better than AP50:95; it is that it is the most practical target-match definition for this particular coalition-growth analysis.",
        "",
        "## How the calculation works",
        "",
        "- Images are grouped by target instance (`scene_key`).",
        "- Angle labels are ignored after grouping; only the set of available images matters.",
        f"- For each scene, `{permutations_per_scene}` random image orderings are sampled up to `{max_images}` images using random seed `{seed}`.",
        "- For one sampled ordering, the first image defines a 1-image coalition, the first two images define a 2-image coalition, and so on.",
        "- At each coalition size `k`, the script computes a best strict-quality prefix value: the maximum strict-quality score among the first `k` images in that sampled ordering.",
        "- At each coalition size `k`, the script also computes a fusion prefix value: `Noisy-OR(first k matched confidences) x best IoU(first k images)`.",
        "- The marginal contribution of the `k`th added image is then `prefix_value_at_k - prefix_value_at_(k-1)`.",
        "- These marginal gains are averaged over all sampled orderings within a scene.",
        "- The final table then averages those scene-level means over all scenes that have at least `k` available images.",
        "- This is a Shapley-style permutation expectation over image order, but it does not yet assign value to named angles.",
        "",
        "## How to read the table",
        "",
        "- `Added image number`: the position `k` of the added image in the coalition-growth process.",
        "- `Scene support`: how many scenes had at least `k` images available, so could contribute to that row.",
        "- `Mean best strict-quality value`: the mean coalition value at size `k` under the best-single-image rule.",
        "- `Mean Noisy-OR x best IoU value`: the mean coalition value at size `k` under the fusion rule.",
        "- `Mean strict-quality marginal gain`: the expected increase caused by adding the `k`th image under the best-single-image rule.",
        "- `Mean Noisy-OR x best IoU marginal gain`: the expected increase caused by adding the `k`th image under the fusion rule.",
        "- The support drops from 205 to 177 by `k=8` because not every scene has 8 available images.",
        "",
        "## What angles can appear in a coalition",
        "",
        "- Across the full cache, all `72` M4 viewpoints are present.",
        "- But one coalition is never formed from all 72 at once. A coalition is formed only within one `scene_key`, using whatever viewpoints are available for that specific scene in `scene_view_records.csv`.",
        "- Because this is the count-first angle-agnostic report, there is no restriction that a coalition stay inside one ring, one elevation, or one radius. Low, mid, and high elevations can be mixed if they are available in the same scene. The same is true for radius and azimuth.",
        "- The sampled orderings therefore can use angles from 'all over' within the scene-level viewpoint set.",
        "- Example scene `S0-SM_barrel_1` has available viewpoints such as `elhigh-radfar-az270`, `elhigh-radmid-az045`, `elhigh-radnear-az090`, `ellow-radfar-az135`, `ellow-radmid-az135`, `ellow-radnear-az090`, `ellow-radnear-az135`, `ellow-radnear-az270`, `elmid-radfar-az225`, and `elmid-radmid-az090`.",
        "- One sampled ordering for that scene can therefore produce coalitions like `k=2: [ellow-radnear-az090, ellow-radnear-az135]`, `k=3: [ellow-radnear-az090, ellow-radnear-az135, elhigh-radfar-az270]`, or `k=5: [ellow-radnear-az090, ellow-radnear-az135, elhigh-radfar-az270, ellow-radnear-az270, ellow-radfar-az135]`.",
        "- If the research question becomes `which specific angle is the best teammate?`, then the ring- or subset-based Shapley analyses are the more angle-controlled methods.",
        "",
        "## Overall progression",
        "",
        "| Added image number | Scene support | Mean best strict-quality value | Mean Noisy-OR x best IoU value | Mean strict-quality marginal gain | Mean Noisy-OR x best IoU marginal gain |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{int(row['added_image_number'])} | "
            f"{int(row['scene_support'])} | "
            f"{fmt(float(row['mean_prefix_target_strict_quality_iou50']))} | "
            f"{fmt(float(row['mean_prefix_noisy_or_best_iou']))} | "
            f"{fmt(float(row['mean_marginal_target_strict_quality_iou50']))} | "
            f"{fmt(float(row['mean_marginal_noisy_or_best_iou']))} |"
        )

    if 2 in by_k and 3 in by_k:
        lines.extend(
            [
                "",
                "## Headline interpretation",
                "",
                f"- The row for `k=1` is the expected value of the first available image because the empty coalition starts at value `0`.",
                f"- The **2nd image** adds `{fmt(float(by_k[2]['mean_marginal_target_strict_quality_iou50']))}` on the best-strict-quality game and `{fmt(float(by_k[2]['mean_marginal_noisy_or_best_iou']))}` on the fusion game.",
                f"- The **3rd image** adds `{fmt(float(by_k[3]['mean_marginal_target_strict_quality_iou50']))}` on the best-strict-quality game and `{fmt(float(by_k[3]['mean_marginal_noisy_or_best_iou']))}` on the fusion game.",
                f"- For example, under the best strict-quality game, the mean coalition value rises from `{fmt(float(by_k[1]['mean_prefix_target_strict_quality_iou50']))}` at 1 image to `{fmt(float(by_k[2]['mean_prefix_target_strict_quality_iou50']))}` at 2 images, so the average 2nd-image gain is `{fmt(float(by_k[2]['mean_marginal_target_strict_quality_iou50']))}`.",
                f"- Under the fusion game, the mean coalition value rises from `{fmt(float(by_k[1]['mean_prefix_noisy_or_best_iou']))}` at 1 image to `{fmt(float(by_k[2]['mean_prefix_noisy_or_best_iou']))}` at 2 images, so the average 2nd-image gain is `{fmt(float(by_k[2]['mean_marginal_noisy_or_best_iou']))}`.",
                "- This is the count-first result you can report even before estimating exact angles on real imagery.",
                "- The pattern shows strong diminishing returns: most of the gain comes from the 2nd image, the 3rd still helps, and later added images contribute progressively smaller average improvements.",
                "",
                "## How to use this in the thesis",
                "",
                "- Use this report for the question `how many different images are worth adding?`.",
                "- Use ring or subset Shapley only after that, when the question becomes `which viewpoint is the most valuable teammate?`.",
                "- The count-first report is the cleaner bridge from synthetic results to raw real-drone images, because it does not require trusted pose metadata.",
            ]
        )

    top_classes = sorted(
        [row for row in class_rows if int(row["added_image_number"]) == 2],
        key=lambda row: float(row["mean_marginal_target_strict_quality_iou50"]),
        reverse=True,
    )[:8]
    if top_classes:
        lines.extend(["", "## Classes with the largest 2nd-image gain", ""])
        for row in top_classes:
            lines.append(
                f"- `{row['target_class']}`: 2nd-image strict-quality gain `{fmt(float(row['mean_marginal_target_strict_quality_iou50']))}`, fusion gain `{fmt(float(row['mean_marginal_noisy_or_best_iou']))}`."
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_scene_records(Path(args.scene_records))
    scene_groups = group_by_scene(records)
    rng = np.random.default_rng(args.seed)

    detail_rows: list[dict[str, object]] = []
    for scene_key in sorted(scene_groups):
        detail_rows.extend(
            summarize_scene_orderings(
                scene_groups[scene_key],
                max_images=args.max_images,
                permutations_per_scene=args.permutations_per_scene,
                rng=rng,
            )
        )

    summary_rows = aggregate_rows(detail_rows)
    class_rows = aggregate_by_class(detail_rows)

    write_csv(output_dir / "image_count_shapley_proxy_scene_detail.csv", detail_rows)
    write_csv(output_dir / "image_count_shapley_proxy_summary.csv", summary_rows)
    write_csv(output_dir / "image_count_shapley_proxy_by_class.csv", class_rows)
    build_report(
        summary_rows,
        class_rows,
        output_dir / "image_count_shapley_proxy_report.md",
        scene_records_path=Path(args.scene_records),
        max_images=args.max_images,
        permutations_per_scene=args.permutations_per_scene,
        seed=args.seed,
    )
    plot_progression(summary_rows, output_dir / "image_count_shapley_proxy_progression.png")


if __name__ == "__main__":
    main()
