from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[2]
TABLES = WORKSPACE / "results" / "tables"
OUTPUTS = WORKSPACE / "results" / "recomputed" / "viewpoint_training"
PLOTS = OUTPUTS / "plots"

FULL_M4_CSV = TABLES / "detector_family_standardized_test_summary.csv"
SINGLE_CSV = TABLES / "single_view_sweep_master_results.csv"
PAIR_CSV = TABLES / "pair_view_sweep_master_results.csv"
MATCHED_CONTROL_CSV = TABLES / "equal_budget_control_table.csv"
PROTOCOL_CSV = TABLES / "one_vs_two_summary.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    return float(value)


def training_row(
    variant_id: str,
    label: str,
    group: str,
    precision: float,
    recall: float,
    f1: float,
    map50: float,
    map50_95: float,
    support_count: int,
    note: str,
) -> dict[str, object]:
    return {
        "panel": "training_side",
        "variant_id": variant_id,
        "label": label,
        "group": group,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "support_count": support_count,
        "note": note,
    }


def operational_row(
    short_label: str,
    label: str,
    target_found_rate: float,
    target_ap50_95: float,
    strict_quality: float,
    best_available_ap50_95: float,
    note: str,
) -> dict[str, object]:
    return {
        "panel": "operational_side",
        "variant_id": short_label,
        "label": label,
        "target_found_rate": target_found_rate,
        "target_threshold_ap50_95": target_ap50_95,
        "target_threshold_strict_quality_iou50": strict_quality,
        "best_available_ap50_95": best_available_ap50_95,
        "note": note,
    }


def fair_comparison_row(
    comparison_group: str,
    variant_id: str,
    label: str,
    train_images: int | str,
    val_images: int | str,
    precision: float,
    recall: float,
    f1: float,
    map50: float,
    map50_95: float,
    delta_map50_95_vs_matched_m4: float | str,
    delta_map50_vs_matched_m4: float | str,
    delta_f1_vs_matched_m4: float | str,
    note: str,
) -> dict[str, object]:
    return {
        "comparison_group": comparison_group,
        "variant_id": variant_id,
        "label": label,
        "train_images": train_images,
        "val_images": val_images,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "delta_map50_95_vs_matched_m4": delta_map50_95_vs_matched_m4,
        "delta_map50_vs_matched_m4": delta_map50_vs_matched_m4,
        "delta_f1_vs_matched_m4": delta_f1_vs_matched_m4,
        "note": note,
    }


def load_training_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    full_rows = read_csv(FULL_M4_CSV)
    single_rows = read_csv(SINGLE_CSV)
    pair_rows = read_csv(PAIR_CSV)
    matched_control_rows = read_csv(MATCHED_CONTROL_CSV) if MATCHED_CONTROL_CSV.exists() else []

    full_m4 = next(
        row for row in full_rows if row["detector"] == "YOLOv8l" and row["regime"] == "M4"
    )

    completed_singles = [row for row in single_rows if row.get("evaluation_status") == "completed"]
    completed_pairs = [row for row in pair_rows if row.get("option_a_status") == "completed"]

    best_single = max(completed_singles, key=lambda row: to_float(row["mAP50-95"]))
    best_pair = max(completed_pairs, key=lambda row: to_float(row["mAP50-95"]))
    matched_single_controls = [
        row
        for row in matched_control_rows
        if row.get("evaluation_status") == "completed"
        and row.get("source_group") == "single"
        and row.get("source_id") == best_single["single_id"]
    ]
    matched_single_mean_controls = [
        row
        for row in matched_control_rows
        if row.get("evaluation_status") == "completed"
        and row.get("source_group") == "single_mean"
        and row.get("source_id") == "single_mean"
    ]
    matched_pair_controls = [
        row
        for row in matched_control_rows
        if row.get("evaluation_status") == "completed"
        and row.get("source_group") == "pair"
        and row.get("source_id") == best_pair["pair_id"]
    ]
    matched_pair_mean_controls = [
        row
        for row in matched_control_rows
        if row.get("evaluation_status") == "completed"
        and row.get("source_group") == "pair_mean"
        and row.get("source_id") == "pair_mean"
    ]

    single_mean = {
        "precision": mean(to_float(row["precision"]) for row in completed_singles),
        "recall": mean(to_float(row["recall"]) for row in completed_singles),
        "f1": mean(to_float(row["F1"]) for row in completed_singles),
        "map50": mean(to_float(row["mAP50"]) for row in completed_singles),
        "map50_95": mean(to_float(row["mAP50-95"]) for row in completed_singles),
    }
    pair_mean = {
        "precision": mean(to_float(row["precision"]) for row in completed_pairs),
        "recall": mean(to_float(row["recall"]) for row in completed_pairs),
        "f1": mean(to_float(row["F1"]) for row in completed_pairs),
        "map50": mean(to_float(row["mAP50"]) for row in completed_pairs),
        "map50_95": mean(to_float(row["mAP50-95"]) for row in completed_pairs),
    }

    rows = [
        training_row(
            variant_id="full_m4",
            label="Full M4 baseline",
            group="full_m4",
            precision=to_float(full_m4["precision"]),
            recall=to_float(full_m4["recall"]),
            f1=to_float(full_m4["f1"]),
            map50=to_float(full_m4["map50"]),
            map50_95=to_float(full_m4["map50_95"]),
            support_count=1,
            note="YOLOv8l trained on full M4 diversity; evaluated on standardized full M4 test split.",
        ),
        training_row(
            variant_id="single_mean",
            label="Single-view mean",
            group="single",
            precision=single_mean["precision"],
            recall=single_mean["recall"],
            f1=single_mean["f1"],
            map50=single_mean["map50"],
            map50_95=single_mean["map50_95"],
            support_count=len(completed_singles),
            note="Mean across all completed single-view trained models.",
        ),
        training_row(
            variant_id="single_best",
            label="Single-view best",
            group="single",
            precision=to_float(best_single["precision"]),
            recall=to_float(best_single["recall"]),
            f1=to_float(best_single["F1"]),
            map50=to_float(best_single["mAP50"]),
            map50_95=to_float(best_single["mAP50-95"]),
            support_count=1,
            note=f"Best single-view model: {best_single['single_id']} ({best_single['viewpoint']}).",
        ),
    ]
    matched_single_meta = None
    if matched_single_controls:
        matched_single_row = training_row(
            variant_id="matched_single_best",
            label="Equal-budget M4 (best single)",
            group="matched_m4",
            precision=mean(to_float(row["precision"]) for row in matched_single_controls),
            recall=mean(to_float(row["recall"]) for row in matched_single_controls),
            f1=mean(to_float(row["F1"]) for row in matched_single_controls),
            map50=mean(to_float(row["mAP50"]) for row in matched_single_controls),
            map50_95=mean(to_float(row["mAP50-95"]) for row in matched_single_controls),
            support_count=len(matched_single_controls),
            note=(
                "Mean matched-control performance across "
                f"{len(matched_single_controls)} full-M4 run(s) with the same "
                f"train/val image counts as the best single-view model "
                f"({matched_single_controls[0]['number_of_train_images']} / {matched_single_controls[0]['number_of_val_images']})."
            ),
        )
        rows.append(matched_single_row)
        matched_single_meta = {
            "row": matched_single_row,
            "train_count": int(matched_single_controls[0]["number_of_train_images"]),
            "val_count": int(matched_single_controls[0]["number_of_val_images"]),
            "runs": len(matched_single_controls),
        }

    rows.extend(
        [
        training_row(
            variant_id="pair_mean",
            label="Pair-view mean",
            group="pair",
            precision=pair_mean["precision"],
            recall=pair_mean["recall"],
            f1=pair_mean["f1"],
            map50=pair_mean["map50"],
            map50_95=pair_mean["map50_95"],
            support_count=len(completed_pairs),
            note="Mean across completed pair-trained models in the current snapshot.",
        ),
        training_row(
            variant_id="pair_best",
            label="Pair-view best",
            group="pair",
            precision=to_float(best_pair["precision"]),
            recall=to_float(best_pair["recall"]),
            f1=to_float(best_pair["F1"]),
            map50=to_float(best_pair["mAP50"]),
            map50_95=to_float(best_pair["mAP50-95"]),
            support_count=1,
            note=(
                f"Best completed pair model: {best_pair['pair_id']} "
                f"({best_pair['viewpoint_1']} + {best_pair['viewpoint_2']})."
            ),
        ),
    ]
    )
    matched_pair_meta = None
    if matched_pair_controls:
        matched_pair_row = training_row(
            variant_id="matched_pair_best",
            label="Equal-budget M4 (best pair)",
            group="matched_m4",
            precision=mean(to_float(row["precision"]) for row in matched_pair_controls),
            recall=mean(to_float(row["recall"]) for row in matched_pair_controls),
            f1=mean(to_float(row["F1"]) for row in matched_pair_controls),
            map50=mean(to_float(row["mAP50"]) for row in matched_pair_controls),
            map50_95=mean(to_float(row["mAP50-95"]) for row in matched_pair_controls),
            support_count=len(matched_pair_controls),
            note=(
                "Mean matched-control performance across "
                f"{len(matched_pair_controls)} full-M4 run(s) with the same "
                f"train/val image counts as the best pair-view model "
                f"({matched_pair_controls[0]['number_of_train_images']} / {matched_pair_controls[0]['number_of_val_images']})."
            ),
        )
        rows.append(matched_pair_row)
        matched_pair_meta = {
            "row": matched_pair_row,
            "train_count": int(matched_pair_controls[0]["number_of_train_images"]),
            "val_count": int(matched_pair_controls[0]["number_of_val_images"]),
            "runs": len(matched_pair_controls),
        }

    metadata = {
        "full_m4": rows[0],
        "single_mean": rows[1],
        "single_best": rows[2],
        "matched_single_best": matched_single_meta,
        "matched_single_mean": (
            None
            if not matched_single_mean_controls
            else {
                "row": training_row(
                    variant_id="matched_single_mean",
                    label="Equal-budget M4 (single mean)",
                    group="matched_m4",
                    precision=mean(to_float(row["precision"]) for row in matched_single_mean_controls),
                    recall=mean(to_float(row["recall"]) for row in matched_single_mean_controls),
                    f1=mean(to_float(row["F1"]) for row in matched_single_mean_controls),
                    map50=mean(to_float(row["mAP50"]) for row in matched_single_mean_controls),
                    map50_95=mean(to_float(row["mAP50-95"]) for row in matched_single_mean_controls),
                    support_count=len(matched_single_mean_controls),
                    note=(
                        "Mean matched-control performance across "
                        f"{len(matched_single_mean_controls)} full-M4 run(s) with rounded mean "
                        f"single-view train/val counts "
                        f"({matched_single_mean_controls[0]['number_of_train_images']} / {matched_single_mean_controls[0]['number_of_val_images']})."
                    ),
                ),
                "train_count": int(matched_single_mean_controls[0]["number_of_train_images"]),
                "val_count": int(matched_single_mean_controls[0]["number_of_val_images"]),
                "runs": len(matched_single_mean_controls),
            }
        ),
        "pair_mean": next(row for row in rows if row["variant_id"] == "pair_mean"),
        "pair_best": next(row for row in rows if row["variant_id"] == "pair_best"),
        "matched_pair_best": matched_pair_meta,
        "matched_pair_mean": (
            None
            if not matched_pair_mean_controls
            else {
                "row": training_row(
                    variant_id="matched_pair_mean",
                    label="Equal-budget M4 (pair mean)",
                    group="matched_m4",
                    precision=mean(to_float(row["precision"]) for row in matched_pair_mean_controls),
                    recall=mean(to_float(row["recall"]) for row in matched_pair_mean_controls),
                    f1=mean(to_float(row["F1"]) for row in matched_pair_mean_controls),
                    map50=mean(to_float(row["mAP50"]) for row in matched_pair_mean_controls),
                    map50_95=mean(to_float(row["mAP50-95"]) for row in matched_pair_mean_controls),
                    support_count=len(matched_pair_mean_controls),
                    note=(
                        "Mean matched-control performance across "
                        f"{len(matched_pair_mean_controls)} full-M4 run(s) with rounded mean "
                        f"pair-view train/val counts "
                        f"({matched_pair_mean_controls[0]['number_of_train_images']} / {matched_pair_mean_controls[0]['number_of_val_images']})."
                    ),
                ),
                "train_count": int(matched_pair_mean_controls[0]["number_of_train_images"]),
                "val_count": int(matched_pair_mean_controls[0]["number_of_val_images"]),
                "runs": len(matched_pair_mean_controls),
            }
        ),
        "single_completed": len(completed_singles),
        "pair_completed": len(completed_pairs),
        "pair_total_defined": len(pair_rows),
        "best_single_viewpoint": best_single["viewpoint"],
        "best_single_id": best_single["single_id"],
        "best_pair_viewpoint_1": best_pair["viewpoint_1"],
        "best_pair_viewpoint_2": best_pair["viewpoint_2"],
        "best_pair_id": best_pair["pair_id"],
    }
    return rows, metadata


def build_fair_comparison_rows(training_meta: dict[str, object]) -> list[dict[str, object]]:
    full_m4 = training_meta["full_m4"]
    single_mean = training_meta["single_mean"]
    matched_single_mean = training_meta.get("matched_single_mean")
    single_best = training_meta["single_best"]
    matched_single = training_meta.get("matched_single_best")
    pair_mean = training_meta["pair_mean"]
    matched_pair_mean = training_meta.get("matched_pair_mean")
    pair_best = training_meta["pair_best"]
    matched_pair = training_meta.get("matched_pair_best")

    rows: list[dict[str, object]] = [
        fair_comparison_row(
            comparison_group="reference",
            variant_id="full_m4_reference",
            label="Full M4 baseline",
            train_images="all",
            val_images="all",
            precision=float(full_m4["precision"]),
            recall=float(full_m4["recall"]),
            f1=float(full_m4["f1"]),
            map50=float(full_m4["map50"]),
            map50_95=float(full_m4["map50_95"]),
            delta_map50_95_vs_matched_m4="",
            delta_map50_vs_matched_m4="",
            delta_f1_vs_matched_m4="",
            note="Reference model trained on the full M4 regime.",
        )
    ]

    rows.append(
        fair_comparison_row(
            comparison_group="single_context",
            variant_id="single_mean_context",
            label="Single-view mean",
            train_images="varies" if matched_single_mean is None else matched_single_mean["train_count"],
            val_images="varies" if matched_single_mean is None else matched_single_mean["val_count"],
            precision=float(single_mean["precision"]),
            recall=float(single_mean["recall"]),
            f1=float(single_mean["f1"]),
            map50=float(single_mean["map50"]),
            map50_95=float(single_mean["map50_95"]),
            delta_map50_95_vs_matched_m4=(
                ""
                if matched_single_mean is None
                else float(single_mean["map50_95"]) - float(matched_single_mean["row"]["map50_95"])
            ),
            delta_map50_vs_matched_m4=(
                ""
                if matched_single_mean is None
                else float(single_mean["map50"]) - float(matched_single_mean["row"]["map50"])
            ),
            delta_f1_vs_matched_m4=(
                ""
                if matched_single_mean is None
                else float(single_mean["f1"]) - float(matched_single_mean["row"]["f1"])
            ),
            note="Mean across all 72 completed single-view models; compared against an M4 control trained with rounded mean train/val counts.",
        )
    )

    if matched_single_mean is not None:
        matched_single_mean_row = matched_single_mean["row"]
        rows.append(
            fair_comparison_row(
                comparison_group="single_context",
                variant_id="matched_single_mean",
                label="Equal-budget M4 (single mean)",
                train_images=matched_single_mean["train_count"],
                val_images=matched_single_mean["val_count"],
                precision=float(matched_single_mean_row["precision"]),
                recall=float(matched_single_mean_row["recall"]),
                f1=float(matched_single_mean_row["f1"]),
                map50=float(matched_single_mean_row["map50"]),
                map50_95=float(matched_single_mean_row["map50_95"]),
                delta_map50_95_vs_matched_m4=0.0,
                delta_map50_vs_matched_m4=0.0,
                delta_f1_vs_matched_m4=0.0,
                note="Full-M4 control with rounded mean single-view train/val image counts.",
            )
        )

    if matched_single is not None:
        matched_single_row = matched_single["row"]
        rows.extend(
            [
                fair_comparison_row(
                    comparison_group="single_budget",
                    variant_id="single_best",
                    label="Single-view best",
                    train_images=matched_single["train_count"],
                    val_images=matched_single["val_count"],
                    precision=float(single_best["precision"]),
                    recall=float(single_best["recall"]),
                    f1=float(single_best["f1"]),
                    map50=float(single_best["map50"]),
                    map50_95=float(single_best["map50_95"]),
                    delta_map50_95_vs_matched_m4=float(single_best["map50_95"]) - float(matched_single_row["map50_95"]),
                    delta_map50_vs_matched_m4=float(single_best["map50"]) - float(matched_single_row["map50"]),
                    delta_f1_vs_matched_m4=float(single_best["f1"]) - float(matched_single_row["f1"]),
                    note=f"Best single-view model: {training_meta['best_single_viewpoint']}.",
                ),
                fair_comparison_row(
                    comparison_group="single_budget",
                    variant_id="matched_single_best",
                    label="Equal-budget M4 (best single)",
                    train_images=matched_single["train_count"],
                    val_images=matched_single["val_count"],
                    precision=float(matched_single_row["precision"]),
                    recall=float(matched_single_row["recall"]),
                    f1=float(matched_single_row["f1"]),
                    map50=float(matched_single_row["map50"]),
                    map50_95=float(matched_single_row["map50_95"]),
                    delta_map50_95_vs_matched_m4=0.0,
                    delta_map50_vs_matched_m4=0.0,
                    delta_f1_vs_matched_m4=0.0,
                    note="Full-M4 control with the same train/val image counts as the best single-view model.",
                ),
            ]
        )

    rows.append(
        fair_comparison_row(
            comparison_group="pair_context",
            variant_id="pair_mean_context",
            label="Pair-view mean",
            train_images="varies" if matched_pair_mean is None else matched_pair_mean["train_count"],
            val_images="varies" if matched_pair_mean is None else matched_pair_mean["val_count"],
            precision=float(pair_mean["precision"]),
            recall=float(pair_mean["recall"]),
            f1=float(pair_mean["f1"]),
            map50=float(pair_mean["map50"]),
            map50_95=float(pair_mean["map50_95"]),
            delta_map50_95_vs_matched_m4=(
                ""
                if matched_pair_mean is None
                else float(pair_mean["map50_95"]) - float(matched_pair_mean["row"]["map50_95"])
            ),
            delta_map50_vs_matched_m4=(
                ""
                if matched_pair_mean is None
                else float(pair_mean["map50"]) - float(matched_pair_mean["row"]["map50"])
            ),
            delta_f1_vs_matched_m4=(
                ""
                if matched_pair_mean is None
                else float(pair_mean["f1"]) - float(matched_pair_mean["row"]["f1"])
            ),
            note="Mean across all completed pair-view models; compared against an M4 control trained with rounded mean train/val counts.",
        )
    )

    if matched_pair_mean is not None:
        matched_pair_mean_row = matched_pair_mean["row"]
        rows.append(
            fair_comparison_row(
                comparison_group="pair_context",
                variant_id="matched_pair_mean",
                label="Equal-budget M4 (pair mean)",
                train_images=matched_pair_mean["train_count"],
                val_images=matched_pair_mean["val_count"],
                precision=float(matched_pair_mean_row["precision"]),
                recall=float(matched_pair_mean_row["recall"]),
                f1=float(matched_pair_mean_row["f1"]),
                map50=float(matched_pair_mean_row["map50"]),
                map50_95=float(matched_pair_mean_row["map50_95"]),
                delta_map50_95_vs_matched_m4=0.0,
                delta_map50_vs_matched_m4=0.0,
                delta_f1_vs_matched_m4=0.0,
                note="Full-M4 control with rounded mean pair-view train/val image counts.",
            )
        )

    if matched_pair is not None:
        matched_pair_row = matched_pair["row"]
        rows.extend(
            [
                fair_comparison_row(
                    comparison_group="pair_budget",
                    variant_id="pair_best",
                    label="Pair-view best",
                    train_images=matched_pair["train_count"],
                    val_images=matched_pair["val_count"],
                    precision=float(pair_best["precision"]),
                    recall=float(pair_best["recall"]),
                    f1=float(pair_best["f1"]),
                    map50=float(pair_best["map50"]),
                    map50_95=float(pair_best["map50_95"]),
                    delta_map50_95_vs_matched_m4=float(pair_best["map50_95"]) - float(matched_pair_row["map50_95"]),
                    delta_map50_vs_matched_m4=float(pair_best["map50"]) - float(matched_pair_row["map50"]),
                    delta_f1_vs_matched_m4=float(pair_best["f1"]) - float(matched_pair_row["f1"]),
                    note=(
                        f"Best pair-view model: {training_meta['best_pair_viewpoint_1']} + "
                        f"{training_meta['best_pair_viewpoint_2']}."
                    ),
                ),
                fair_comparison_row(
                    comparison_group="pair_budget",
                    variant_id="matched_pair_best",
                    label="Equal-budget M4 (best pair)",
                    train_images=matched_pair["train_count"],
                    val_images=matched_pair["val_count"],
                    precision=float(matched_pair_row["precision"]),
                    recall=float(matched_pair_row["recall"]),
                    f1=float(matched_pair_row["f1"]),
                    map50=float(matched_pair_row["map50"]),
                    map50_95=float(matched_pair_row["map50_95"]),
                    delta_map50_95_vs_matched_m4=0.0,
                    delta_map50_vs_matched_m4=0.0,
                    delta_f1_vs_matched_m4=0.0,
                    note="Full-M4 control with the same train/val image counts as the best pair-view model.",
                ),
            ]
        )

    return rows


def load_operational_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    protocol_rows = read_csv(PROTOCOL_CSV)
    kept = {
        "1-of-1": "1-of-1 (full M4 model)",
        "1-of-2": "1-of-2 OR",
        "1-of-3": "1-of-3 OR",
    }

    selected = [row for row in protocol_rows if row["short_label"] in kept]
    selected.sort(key=lambda row: {"1-of-1": 0, "1-of-2": 1, "1-of-3": 2}[row["short_label"]])

    rows = [
        operational_row(
            short_label=row["short_label"],
            label=kept[row["short_label"]],
            target_found_rate=to_float(row["expected_target_found_rate"]),
            target_ap50_95=to_float(row["expected_target_threshold_ap50_95"]),
            strict_quality=to_float(row["expected_target_threshold_strict_quality_iou50"]),
            best_available_ap50_95=to_float(row["expected_best_available_ap50_95"]),
            note=(
                "Target-centric multiview protocol with fixed YOLOv8l_M4 predictions; "
                "not a retraining experiment."
            ),
        )
        for row in selected
    ]

    metadata = {row["variant_id"]: row for row in rows}
    return rows, metadata


def plot_training_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    ordered = list(reversed(rows))
    labels = [str(row["label"]) for row in ordered]
    colors = {
        "full_m4": "#1f77b4",
        "single": "#2a9d8f",
        "pair": "#f4a261",
        "matched_m4": "#6c757d",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.5), sharey=True, constrained_layout=True)
    metric_specs = [
        ("map50_95", "mAP50-95"),
        ("map50", "mAP50"),
        ("f1", "F1"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        values = [float(row[metric_key]) for row in ordered]
        bar_colors = [colors[str(row["group"])] for row in ordered]
        bars = ax.barh(labels, values, color=bar_colors)
        ax.set_title(metric_label)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, max(values) * 1.16)
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + (bar.get_height() / 2), f"{value:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Training-side comparison on the full fixed M4 test split\n"
        "Full M4, viewpoint-subset models, and equal-count M4 controls",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_operational_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    ordered = list(reversed(rows))
    labels = [str(row["label"]) for row in ordered]
    color = "#bc6c25"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True, constrained_layout=True)
    metric_specs = [
        ("target_found_rate", "Target found rate"),
        ("target_threshold_ap50_95", "Target AP50-95"),
        ("target_threshold_strict_quality_iou50", "Strict quality"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        values = [float(row[metric_key]) for row in ordered]
        bars = ax.barh(labels, values, color=color)
        ax.set_title(metric_label)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, 1.05)
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + (bar.get_height() / 2), f"{value:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Operational comparison with the fixed YOLOv8l_M4 model\n"
        "1-of-1 vs 1-of-2 OR vs 1-of-3 OR",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_headline_dashboard(
    training_rows: list[dict[str, object]],
    operational_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    training_labels = [str(row["label"]) for row in training_rows]
    training_values = [float(row["map50_95"]) for row in training_rows]
    training_color_map = {
        "full_m4": "#1f77b4",
        "single": "#2a9d8f",
        "pair": "#f4a261",
        "matched_m4": "#6c757d",
    }
    training_colors = [training_color_map[str(row["group"])] for row in training_rows]

    operational_labels = [str(row["label"]) for row in operational_rows]
    operational_values = [float(row["target_threshold_ap50_95"]) for row in operational_rows]
    operational_colors = ["#8d99ae", "#ef8354", "#d62828"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)

    axes[0].bar(training_labels, training_values, color=training_colors)
    axes[0].set_title("Training-side headline metric")
    axes[0].set_ylabel("mAP50-95 on full fixed M4 test split")
    axes[0].set_ylim(0, max(training_values) * 1.18)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=28)
    for idx, value in enumerate(training_values):
        axes[0].text(idx, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(operational_labels, operational_values, color=operational_colors)
    axes[1].set_title("Operational headline metric")
    axes[1].set_ylabel("Expected target AP50-95")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=18)
    for idx, value in enumerate(operational_values):
        axes[1].text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Full M4, matched controls, viewpoint-subset models, and 1-of-2/1-of-3\n"
        "Panels use different evaluation definitions, so compare trends within each panel.",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fair_training_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    preferred_order = [
        "full_m4_reference",
        "single_mean_context",
        "single_best",
        "matched_single_best",
        "pair_mean_context",
        "pair_best",
        "matched_pair_best",
    ]
    row_by_id = {str(row["variant_id"]): row for row in rows}
    ordered = [row_by_id[variant_id] for variant_id in preferred_order if variant_id in row_by_id]
    labels = [str(row["label"]) for row in ordered]
    colors = {
        "single_context": "#7bc8c0",
        "single_budget": "#2a9d8f",
        "pair_context": "#f7c59f",
        "pair_budget": "#f4a261",
        "reference": "#1f77b4",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.5), sharey=True, constrained_layout=True)
    metric_specs = [
        ("map50_95", "mAP50-95"),
        ("map50", "mAP50"),
        ("f1", "F1"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        values = [float(row[metric_key]) for row in ordered]
        bar_colors = [colors[str(row["comparison_group"])] for row in ordered]
        bars = ax.barh(labels, values, color=bar_colors)
        ax.set_title(metric_label)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, max(values) * 1.16)
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + (bar.get_height() / 2), f"{value:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Fair training-side comparison on the full fixed M4 test split\n"
        "Restricted-view means and bests vs equal-count M4 controls",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_fair_comparison_report(rows: list[dict[str, object]], output_path: Path) -> None:
    full_m4 = next(row for row in rows if row["variant_id"] == "full_m4_reference")
    single_context_rows = [row for row in rows if row["comparison_group"] == "single_context"]
    single_budget_rows = [row for row in rows if row["comparison_group"] == "single_budget"]
    pair_context_rows = [row for row in rows if row["comparison_group"] == "pair_context"]
    pair_budget_rows = [row for row in rows if row["comparison_group"] == "pair_budget"]

    lines = [
        "# Fair Training-Side Comparison",
        "",
        "## What This Table Shows",
        "",
        "- This table compares restricted-view mean and best results against full-M4 controls trained with the same train/val image counts.",
        "- All models are evaluated on the same full fixed M4 test split.",
        "",
        "## Reference",
        "",
        (
            f"- Full M4 baseline: train=`all`, val=`all`, "
            f"`mAP50-95 = {float(full_m4['map50_95']):.4f}`, "
            f"`mAP50 = {float(full_m4['map50']):.4f}`, "
            f"`F1 = {float(full_m4['f1']):.4f}`"
        ),
        "",
        "## Single-View Mean Budget",
        "",
    ]

    if single_context_rows:
        for row in single_context_rows:
            gap = row["delta_map50_95_vs_matched_m4"]
            gap_text = "" if gap == "" else f", `delta vs equal-budget M4 = {float(gap):+.4f}`"
            lines.append(
                (
                    f"- {row['label']}: train=`{row['train_images']}`, val=`{row['val_images']}`, "
                    f"`mAP50-95 = {float(row['map50_95']):.4f}`, "
                    f"`mAP50 = {float(row['map50']):.4f}`, `F1 = {float(row['f1']):.4f}`{gap_text}"
                )
            )
    else:
        lines.append("- Single-view mean-budget fair comparison is not available yet.")

    lines.extend(["", "## Single-View Best Budget", ""])

    if single_budget_rows:
        for row in single_budget_rows:
            gap = row["delta_map50_95_vs_matched_m4"]
            gap_text = "" if gap == "" else f", `delta vs equal-budget M4 = {float(gap):+.4f}`"
            lines.append(
                (
                    f"- {row['label']}: train=`{row['train_images']}`, val=`{row['val_images']}`, "
                    f"`mAP50-95 = {float(row['map50_95']):.4f}`, "
                    f"`mAP50 = {float(row['map50']):.4f}`, `F1 = {float(row['f1']):.4f}`{gap_text}"
                )
            )
    else:
        lines.append("- Single-view best-budget fair comparison is not available yet.")

    lines.extend(["", "## Pair-View Mean Budget", ""])

    if pair_context_rows:
        for row in pair_context_rows:
            gap = row["delta_map50_95_vs_matched_m4"]
            gap_text = "" if gap == "" else f", `delta vs equal-budget M4 = {float(gap):+.4f}`"
            lines.append(
                (
                    f"- {row['label']}: train=`{row['train_images']}`, val=`{row['val_images']}`, "
                    f"`mAP50-95 = {float(row['map50_95']):.4f}`, "
                    f"`mAP50 = {float(row['map50']):.4f}`, `F1 = {float(row['f1']):.4f}`{gap_text}"
                )
            )
    else:
        lines.append("- Pair-view mean-budget fair comparison is not available yet.")

    lines.extend(["", "## Pair-View Best Budget", ""])

    if pair_budget_rows:
        for row in pair_budget_rows:
            gap = row["delta_map50_95_vs_matched_m4"]
            gap_text = "" if gap == "" else f", `delta vs equal-budget M4 = {float(gap):+.4f}`"
            lines.append(
                (
                    f"- {row['label']}: train=`{row['train_images']}`, val=`{row['val_images']}`, "
                    f"`mAP50-95 = {float(row['map50_95']):.4f}`, "
                    f"`mAP50 = {float(row['map50']):.4f}`, `F1 = {float(row['f1']):.4f}`{gap_text}"
                )
            )
    else:
        lines.append("- Pair-view best-budget fair comparison is not available yet.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    training_rows: list[dict[str, object]],
    training_meta: dict[str, object],
    operational_rows: list[dict[str, object]],
    operational_meta: dict[str, object],
    output_path: Path,
) -> None:
    full_m4 = training_meta["full_m4"]
    single_mean = training_meta["single_mean"]
    matched_single_mean = training_meta.get("matched_single_mean")
    single_best = training_meta["single_best"]
    matched_single = training_meta.get("matched_single_best")
    pair_mean = training_meta["pair_mean"]
    matched_pair_mean = training_meta.get("matched_pair_mean")
    pair_best = training_meta["pair_best"]
    matched_pair = training_meta.get("matched_pair_best")
    one_of_one = operational_meta["1-of-1"]
    one_of_two = operational_meta["1-of-2"]
    one_of_three = operational_meta["1-of-3"]

    lines = [
        "# Full M4 vs Single-View vs Pair-View vs 1-of-2/1-of-3",
        "",
        "## What This Comparison Does",
        "",
        "- This analysis places the main training-side baselines and the operational multiview protocols in one summary.",
        "- Left side of the story: detector generalization after training on different viewpoint subsets.",
        "- Right side of the story: target-centric multiview performance when the detector is held fixed and extra views are made available at inference time.",
        "",
        "## Important Interpretation Boundary",
        "",
        "- `Full M4`, `single-view`, and `pair-view` are per-image detector evaluations on the full fixed M4 test split.",
        "- `1-of-1`, `1-of-2`, and `1-of-3` are operational target-centric protocols built on fixed YOLOv8l_M4 predictions.",
        "- These two panels should therefore be compared for trend and takeaway, not as identical metrics.",
        "",
        "## Training-Side Comparison",
        "",
        f"- Full M4 baseline: `mAP50-95 = {float(full_m4['map50_95']):.4f}`, `mAP50 = {float(full_m4['map50']):.4f}`, `F1 = {float(full_m4['f1']):.4f}`",
        (
            f"- Mean single-view model ({training_meta['single_completed']} runs): "
            f"`mAP50-95 = {float(single_mean['map50_95']):.4f}`, "
            f"gap to full M4 `{float(single_mean['map50_95']) - float(full_m4['map50_95']):+.4f}`"
        ),
        (
            f"- Best single-view model (`{training_meta['best_single_viewpoint']}`): "
            f"`mAP50-95 = {float(single_best['map50_95']):.4f}`, "
            f"gap to full M4 `{float(single_best['map50_95']) - float(full_m4['map50_95']):+.4f}`"
        ),
    ]
    if matched_single_mean is not None:
        matched_single_mean_row = matched_single_mean["row"]
        lines.append(
            (
                f"- Equal-budget M4 for single-view mean ({matched_single_mean['runs']} run(s), "
                f"`train={matched_single_mean['train_count']}`, `val={matched_single_mean['val_count']}`): "
                f"`mAP50-95 = {float(matched_single_mean_row['map50_95']):.4f}`, "
                f"gap to single-view mean `{float(single_mean['map50_95']) - float(matched_single_mean_row['map50_95']):+.4f}`"
            )
        )
    if matched_single is not None:
        matched_single_row = matched_single["row"]
        lines.append(
            (
                f"- Equal-budget M4 for best single ({matched_single['runs']} run(s), "
                f"`train={matched_single['train_count']}`, `val={matched_single['val_count']}`): "
                f"`mAP50-95 = {float(matched_single_row['map50_95']):.4f}`, "
                f"gap to best single `{float(single_best['map50_95']) - float(matched_single_row['map50_95']):+.4f}`"
            )
        )
    lines.extend(
        [
        (
            f"- Mean pair-view model ({training_meta['pair_completed']} / {training_meta['pair_total_defined']} completed): "
            f"`mAP50-95 = {float(pair_mean['map50_95']):.4f}`, "
            f"gap to full M4 `{float(pair_mean['map50_95']) - float(full_m4['map50_95']):+.4f}`"
        ),
        (
            f"- Best pair-view model (`{training_meta['best_pair_viewpoint_1']}` + `{training_meta['best_pair_viewpoint_2']}`): "
            f"`mAP50-95 = {float(pair_best['map50_95']):.4f}`, "
            f"gap to full M4 `{float(pair_best['map50_95']) - float(full_m4['map50_95']):+.4f}`, "
            f"lift over best single `{float(pair_best['map50_95']) - float(single_best['map50_95']):+.4f}`"
        ),
    ]
    )
    if matched_pair_mean is not None:
        matched_pair_mean_row = matched_pair_mean["row"]
        lines.append(
            (
                f"- Equal-budget M4 for pair-view mean ({matched_pair_mean['runs']} run(s), "
                f"`train={matched_pair_mean['train_count']}`, `val={matched_pair_mean['val_count']}`): "
                f"`mAP50-95 = {float(matched_pair_mean_row['map50_95']):.4f}`, "
                f"gap to pair-view mean `{float(pair_mean['map50_95']) - float(matched_pair_mean_row['map50_95']):+.4f}`"
            )
        )
    if matched_pair is not None:
        matched_pair_row = matched_pair["row"]
        lines.append(
            (
                f"- Equal-budget M4 for best pair ({matched_pair['runs']} run(s), "
                f"`train={matched_pair['train_count']}`, `val={matched_pair['val_count']}`): "
                f"`mAP50-95 = {float(matched_pair_row['map50_95']):.4f}`, "
                f"gap to best pair `{float(pair_best['map50_95']) - float(matched_pair_row['map50_95']):+.4f}`"
            )
        )
    lines.extend(
        [
        "",
        "## Operational Comparison",
        "",
        (
            f"- 1-of-1: target found `{float(one_of_one['target_found_rate']):.4f}`, "
            f"target AP50-95 `{float(one_of_one['target_threshold_ap50_95']):.4f}`, "
            f"strict quality `{float(one_of_one['target_threshold_strict_quality_iou50']):.4f}`"
        ),
        (
            f"- 1-of-2 OR: target found `{float(one_of_two['target_found_rate']):.4f}`, "
            f"target AP50-95 `{float(one_of_two['target_threshold_ap50_95']):.4f}` "
            f"(`{float(one_of_two['target_threshold_ap50_95']) - float(one_of_one['target_threshold_ap50_95']):+.4f}` vs 1-of-1), "
            f"strict quality `{float(one_of_two['target_threshold_strict_quality_iou50']):.4f}` "
            f"(`{float(one_of_two['target_threshold_strict_quality_iou50']) - float(one_of_one['target_threshold_strict_quality_iou50']):+.4f}`)"
        ),
        (
            f"- 1-of-3 OR: target found `{float(one_of_three['target_found_rate']):.4f}`, "
            f"target AP50-95 `{float(one_of_three['target_threshold_ap50_95']):.4f}` "
            f"(`{float(one_of_three['target_threshold_ap50_95']) - float(one_of_one['target_threshold_ap50_95']):+.4f}` vs 1-of-1), "
            f"strict quality `{float(one_of_three['target_threshold_strict_quality_iou50']):.4f}` "
            f"(`{float(one_of_three['target_threshold_strict_quality_iou50']) - float(one_of_one['target_threshold_strict_quality_iou50']):+.4f}`)"
        ),
        "",
        "## Main Takeaway",
        "",
    ]
    )
    if matched_single is not None or matched_pair is not None:
        lines.extend(
            [
                "- The equal-count M4 controls are the fairer headline comparison when the question is about viewpoint diversity at fixed image budget.",
                "- Use the equal-budget M4 gap to judge whether restricted-view training is still weaker than an equally sized diverse-view subset.",
            ]
        )
    lines.extend(
        [
        "- Full M4 remains the strongest training-side detector baseline.",
        "- Pair-view training clearly improves over single-view training, but it still does not close the gap to the strongest diverse-view baseline.",
        "- Extra views at inference time help a lot under 1-of-2 and 1-of-3 OR protocols, but that is a different kind of gain than retraining on richer viewpoint subsets.",
        "- The clean story is therefore: viewpoint diversity helps both in training and at inference, but those gains appear through different mechanisms.",
        "",
        "## Generated Files",
        "",
        "- `training_side_summary.csv`",
        "- `operational_side_summary.csv`",
        "- `comparison_summary.md`",
        "- `plots/training_side_comparison.png`",
        "- `fair_training_side_table.csv`",
        "- `fair_training_side_table.md`",
        "- `plots/fair_training_side_comparison.png`",
        "- `plots/operational_side_comparison.png`",
        "- `plots/headline_comparison_dashboard.png`",
    ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUTS)
    ensure_dir(PLOTS)

    training_rows, training_meta = load_training_rows()
    fair_rows = build_fair_comparison_rows(training_meta)
    operational_rows, operational_meta = load_operational_rows()

    write_csv(
        OUTPUTS / "training_side_summary.csv",
        [
            "panel",
            "variant_id",
            "label",
            "group",
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95",
            "support_count",
            "note",
        ],
        training_rows,
    )
    write_csv(
        OUTPUTS / "operational_side_summary.csv",
        [
            "panel",
            "variant_id",
            "label",
            "target_found_rate",
            "target_threshold_ap50_95",
            "target_threshold_strict_quality_iou50",
            "best_available_ap50_95",
            "note",
        ],
        operational_rows,
    )
    write_csv(
        OUTPUTS / "fair_training_side_table.csv",
        [
            "comparison_group",
            "variant_id",
            "label",
            "train_images",
            "val_images",
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95",
            "delta_map50_95_vs_matched_m4",
            "delta_map50_vs_matched_m4",
            "delta_f1_vs_matched_m4",
            "note",
        ],
        fair_rows,
    )

    plot_training_comparison(training_rows, PLOTS / "training_side_comparison.png")
    plot_fair_training_comparison(fair_rows, PLOTS / "fair_training_side_comparison.png")
    plot_operational_comparison(operational_rows, PLOTS / "operational_side_comparison.png")
    plot_headline_dashboard(training_rows, operational_rows, PLOTS / "headline_comparison_dashboard.png")
    write_fair_comparison_report(fair_rows, OUTPUTS / "fair_training_side_table.md")
    write_summary(training_rows, training_meta, operational_rows, operational_meta, OUTPUTS / "comparison_summary.md")

    print(f"Wrote outputs under: {OUTPUTS}")


if __name__ == "__main__":
    main()
