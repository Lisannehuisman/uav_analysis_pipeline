from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from m4_two_drone_operational_analysis.analyze_two_drone_operational import build_view_records, scene_view_rows
from multiview_transformer.common import parse_scene_view_metadata, read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset")
TRAIN_IMAGES_DIR = DATASET_ROOT / "images" / "train_M4"
VAL_IMAGES_DIR = DATASET_ROOT / "images" / "val"
TEST_IMAGES_DIR = DATASET_ROOT / "images" / "test"

TRAIN_GT_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "ground_truth" / "M4_train_gt.json"
VAL_SCENE_VIEW_CSV = ROOT / "m4_two_drone_operational_analysis" / "outputs_val" / "scene_view_records.csv"
TEST_SCENE_VIEW_CSV = ROOT / "m4_two_drone_operational_analysis" / "outputs_test" / "scene_view_records.csv"

TRAIN_PRED_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "predictions" / "YOLOv8l_M4_train_predictions.json"
VAL_PRED_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_val_eval" / "predictions" / "YOLOv8l_M4_val_predictions.json"
TEST_PRED_JSON = ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "predictions" / "YOLOv8l_M4_test_predictions.json"

TRAIN_SOURCE_GT = ROOT / "outputs" / "detector_family_comparison" / "standardized_train_eval" / "ground_truth" / "M4_train_gt.json"
VAL_SOURCE_GT = DATASET_ROOT / "coco_annotations" / "coco_instances_val_M4_fixed.json"
TEST_SOURCE_GT = ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json"

VIEWPOINT_MANIFEST = ROOT / "viewpoint_data_separated" / "72_trained_models" / "manifests" / "full_single_viewpoints.csv"
DEFAULT_OUTPUT_DIR = ROOT / "probability_fusion" / "outputs" / "full72_manifest_aligned_cache"
IMPORTED_MODEL = ROOT / "outputs" / "imported_runs" / "yolov8l_m4" / "M4_clean_yolov8l_run1" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact 72-view manifest-aligned cache for the synthetic M4 dataset by "
            "combining train_M4 metadata with cached val/test scene-view records."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the manifest, aligned cache CSV, and summary files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into an existing output directory instead of creating a suffixed one.",
    )
    return parser.parse_args()


def ensure_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def unique_output_dir(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_viewpoint_inventory(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(ensure_file(path, "72-viewpoint inventory CSV"))


def known_classes_from_inventory(rows: list[dict[str, str]]) -> list[str]:
    # The class is inferred from scene file names, so we only need the synthetic class names.
    return [
        "barrel",
        "container",
        "male",
        "rock",
        "suv",
        "tank",
        "tent",
        "tower",
        "tree",
        "whitevan",
    ]


def image_path_for_split(split: str, file_name: str) -> str:
    if split == "train":
        return str((TRAIN_IMAGES_DIR / file_name).resolve())
    if split == "val":
        return str((VAL_IMAGES_DIR / file_name).resolve())
    if split == "test":
        return str((TEST_IMAGES_DIR / file_name).resolve())
    raise KeyError(f"Unknown split: {split}")


def build_train_manifest_rows(known_class_names: list[str]) -> list[dict[str, object]]:
    gt = load_json(ensure_file(TRAIN_GT_JSON, "standardized train GT JSON"))
    rows: list[dict[str, object]] = []
    for image in gt["images"]:
        file_name = str(image["file_name"])
        instance_id, viewpoint, elevation, radius, azimuth, class_name = parse_scene_view_metadata(
            file_name=file_name,
            known_class_names=known_class_names,
        )
        rows.append(
            {
                "split": "train",
                "instance_id": instance_id,
                "class_name": class_name,
                "base_class": class_name,
                "file_name": file_name,
                "image_id": int(image["id"]),
                "image_path": image_path_for_split("train", file_name),
                "viewpoint": viewpoint,
                "elevation": elevation,
                "radius": radius,
                "azimuth": int(azimuth),
                "manifest_source": str(TRAIN_GT_JSON),
                "score_available": 0,
                "prediction_cache_status": "missing_train_predictions",
                "source_prediction_json": str(TRAIN_PRED_JSON),
                "source_ground_truth_json": str(TRAIN_SOURCE_GT),
                "raw_view_score": "",
                "view_score": "",
                "view_found_label": "",
                "best_iou": "",
                "match_iou_at_score": "",
                "target_strict_quality_iou50": "",
                "source_scene_view_cache": "",
            }
        )
    return rows


def build_train_scored_rows_if_available() -> list[dict[str, object]] | None:
    if not TRAIN_PRED_JSON.is_file():
        return None
    records = build_view_records(
        gt_json=ensure_file(TRAIN_GT_JSON, "standardized train GT JSON"),
        pred_json=TRAIN_PRED_JSON,
        score_threshold=0.001,
    )
    scene_rows = scene_view_rows(records)
    rows: list[dict[str, object]] = []
    for row in scene_rows:
        file_name = str(row["file_name"])
        rows.append(
            {
                "split": "train",
                "instance_id": str(row["scene_key"]),
                "class_name": str(row["target_class"]),
                "base_class": str(row["target_class"]),
                "file_name": file_name,
                "image_id": int(float(row["image_id"])),
                "image_path": image_path_for_split("train", file_name),
                "viewpoint": str(row["viewpoint"]),
                "elevation": str(row["elevation"]),
                "radius": str(row["radius"]),
                "azimuth": int(float(row["azimuth"])),
                "manifest_source": str(TRAIN_GT_JSON),
                "score_available": 1,
                "prediction_cache_status": "cached_available",
                "source_prediction_json": str(TRAIN_PRED_JSON),
                "source_ground_truth_json": str(TRAIN_SOURCE_GT),
                "raw_view_score": str(row["target_match_confidence_iou50"]),
                "view_score": str(row["target_match_confidence_iou50"]),
                "view_found_label": str(int(float(row["target_detected"]))),
                "best_iou": str(row["target_best_iou"]),
                "match_iou_at_score": str(row["target_match_iou_at_confidence_iou50"]),
                "target_strict_quality_iou50": str(row["target_strict_quality_iou50"]),
                "source_scene_view_cache": "",
            }
        )
    return rows


def build_cached_split_rows(
    split: str,
    scene_view_csv: Path,
    source_pred_json: Path,
    source_gt_json: Path,
) -> list[dict[str, object]]:
    cache_rows = read_csv_rows(ensure_file(scene_view_csv, f"{split} scene-view cache"))
    rows: list[dict[str, object]] = []
    for row in cache_rows:
        file_name = str(row["file_name"])
        rows.append(
            {
                "split": split,
                "instance_id": str(row["scene_key"]),
                "class_name": str(row["target_class"]),
                "base_class": str(row["target_class"]),
                "file_name": file_name,
                "image_id": int(float(row["image_id"])),
                "image_path": image_path_for_split(split, file_name),
                "viewpoint": str(row["viewpoint"]),
                "elevation": str(row["elevation"]),
                "radius": str(row["radius"]),
                "azimuth": int(float(row["azimuth"])),
                "manifest_source": str(scene_view_csv),
                "score_available": 1,
                "prediction_cache_status": "cached_available",
                "source_prediction_json": str(source_pred_json),
                "source_ground_truth_json": str(source_gt_json),
                "raw_view_score": str(row["target_match_confidence_iou50"]),
                "view_score": str(row["target_match_confidence_iou50"]),
                "view_found_label": str(int(float(row["target_detected"]))),
                "best_iou": str(row["target_best_iou"]),
                "match_iou_at_score": str(row["target_match_iou_at_confidence_iou50"]),
                "target_strict_quality_iou50": str(row["target_strict_quality_iou50"]),
                "source_scene_view_cache": str(scene_view_csv),
            }
        )
    return rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    write_csv_rows(path, fieldnames=fieldnames, rows=rows)


def build_instance_coverage_summary(
    aligned_rows: list[dict[str, object]],
    expected_viewpoints: list[str],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aligned_rows:
        grouped[str(row["instance_id"])].append(row)

    expected_set = set(expected_viewpoints)
    summary_rows: list[dict[str, object]] = []
    for instance_id in sorted(grouped):
        rows = grouped[instance_id]
        observed_viewpoints = [str(row["viewpoint"]) for row in rows]
        observed_set = set(observed_viewpoints)
        split_counts = Counter(str(row["split"]) for row in rows)
        duplicate_count = len(observed_viewpoints) - len(observed_set)
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        score_available_count = sum(int(row["score_available"]) for row in rows)
        summary_rows.append(
            {
                "instance_id": instance_id,
                "class_name": str(rows[0]["class_name"]),
                "total_rows": len(rows),
                "unique_viewpoint_count": len(observed_set),
                "expected_viewpoint_count": len(expected_viewpoints),
                "missing_viewpoint_count": len(missing),
                "duplicate_viewpoint_count": duplicate_count,
                "unexpected_viewpoint_count": len(extra),
                "train_rows": split_counts.get("train", 0),
                "val_rows": split_counts.get("val", 0),
                "test_rows": split_counts.get("test", 0),
                "score_available_count": score_available_count,
                "score_missing_count": len(rows) - score_available_count,
                "missing_viewpoints": "|".join(missing),
                "unexpected_viewpoints": "|".join(extra),
            }
        )
    return summary_rows


def build_viewpoint_split_check(
    aligned_rows: list[dict[str, object]],
    viewpoint_inventory_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped_by_viewpoint: dict[str, Counter[str]] = defaultdict(Counter)
    for row in aligned_rows:
        grouped_by_viewpoint[str(row["viewpoint"])][str(row["split"])] += 1

    check_rows: list[dict[str, object]] = []
    for inv_row in viewpoint_inventory_rows:
        viewpoint = str(inv_row["viewpoint"])
        counts = grouped_by_viewpoint.get(viewpoint, Counter())
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
        test_count = counts.get("test", 0)
        total_count = train_count + val_count + test_count
        expected_train = int(inv_row["expected_train_images"])
        expected_val = int(inv_row["expected_val_images"])
        expected_test = int(inv_row["expected_test_view_images"])
        check_rows.append(
            {
                "viewpoint": viewpoint,
                "single_id": str(inv_row["single_id"]),
                "expected_train_images": expected_train,
                "observed_train_images": train_count,
                "expected_val_images": expected_val,
                "observed_val_images": val_count,
                "expected_test_view_images": expected_test,
                "observed_test_images": test_count,
                "expected_total_images": expected_train + expected_val + expected_test,
                "observed_total_images": total_count,
                "train_match": int(train_count == expected_train),
                "val_match": int(val_count == expected_val),
                "test_match": int(test_count == expected_test),
                "total_match": int(total_count == expected_train + expected_val + expected_test),
            }
        )
    return check_rows


def build_summary_markdown(
    output_dir: Path,
    aligned_rows: list[dict[str, object]],
    instance_summary_rows: list[dict[str, object]],
    viewpoint_check_rows: list[dict[str, object]],
) -> str:
    total_rows = len(aligned_rows)
    score_available = sum(int(row["score_available"]) for row in aligned_rows)
    instance_count = len(instance_summary_rows)
    exact_72_count = sum(1 for row in instance_summary_rows if int(row["unique_viewpoint_count"]) == 72 and int(row["missing_viewpoint_count"]) == 0)
    perfect_viewpoint_checks = sum(1 for row in viewpoint_check_rows if int(row["total_match"]) == 1)

    lines = [
        "# Full 72-View Manifest-Aligned Cache",
        "",
        "## What this cache is",
        "",
        "- An exact synthetic object-instance manifest over `train_M4 + val + test`.",
        "- Each row is aligned to one `(instance_id, viewpoint)` cell in the intended 72-view grid.",
        "- Cached target-match scores are reused for `val` and `test` from the existing M4 scene-view caches.",
        "- `train` rows are included in the manifest now, but their detector scores remain pending until a train prediction JSON is generated.",
        "",
        "## Coverage",
        "",
        f"- Total rows: `{total_rows}`.",
        f"- Instance count: `{instance_count}`.",
        f"- Instances with a complete 72-view union: `{exact_72_count}` / `{instance_count}`.",
        f"- Rows with cached scores available now: `{score_available}`.",
        f"- Rows still missing scores now: `{total_rows - score_available}`.",
        "",
        "## Split structure",
        "",
        "- The full 72-view grid is distributed across splits, not stored inside one split.",
        "- In this dataset, `train` carries the majority of viewpoints per instance, while `val` and `test` provide the remaining viewpoint cells.",
        "",
        "## Viewpoint inventory check",
        "",
        f"- Viewpoint rows matching expected train+val+test counts: `{perfect_viewpoint_checks}` / `{len(viewpoint_check_rows)}`.",
        "",
        "## Current score sources",
        "",
        f"- Train predictions expected at: `{TRAIN_PRED_JSON}`.",
        f"- Val predictions reused from: `{VAL_PRED_JSON}`.",
        f"- Test predictions reused from: `{TEST_PRED_JSON}`.",
        f"- Imported M4 model weights stored at: `{IMPORTED_MODEL}`.",
        "",
        "## Files",
        "",
        "- `full72_manifest.csv`",
        "- `full72_manifest_aligned_cache.csv`",
        "- `instance_coverage_summary.csv`",
        "- `viewpoint_split_check.csv`",
        "",
        f"Generated in `{output_dir}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = unique_output_dir(Path(args.output_dir).resolve(), overwrite=args.overwrite)

    viewpoint_inventory_rows = load_viewpoint_inventory(VIEWPOINT_MANIFEST)
    expected_viewpoints = [str(row["viewpoint"]) for row in viewpoint_inventory_rows]
    known_class_names = known_classes_from_inventory(viewpoint_inventory_rows)

    train_rows = build_train_scored_rows_if_available()
    if train_rows is None:
        train_rows = build_train_manifest_rows(known_class_names=known_class_names)
    val_rows = build_cached_split_rows(
        split="val",
        scene_view_csv=VAL_SCENE_VIEW_CSV,
        source_pred_json=VAL_PRED_JSON,
        source_gt_json=VAL_SOURCE_GT,
    )
    test_rows = build_cached_split_rows(
        split="test",
        scene_view_csv=TEST_SCENE_VIEW_CSV,
        source_pred_json=TEST_PRED_JSON,
        source_gt_json=TEST_SOURCE_GT,
    )

    aligned_rows = train_rows + val_rows + test_rows
    aligned_rows.sort(key=lambda row: (str(row["instance_id"]), str(row["split"]), str(row["viewpoint"])))

    manifest_fieldnames = [
        "split",
        "instance_id",
        "class_name",
        "base_class",
        "file_name",
        "image_id",
        "image_path",
        "viewpoint",
        "elevation",
        "radius",
        "azimuth",
        "manifest_source",
    ]
    aligned_cache_fieldnames = manifest_fieldnames + [
        "score_available",
        "prediction_cache_status",
        "source_prediction_json",
        "source_ground_truth_json",
        "raw_view_score",
        "view_score",
        "view_found_label",
        "best_iou",
        "match_iou_at_score",
        "target_strict_quality_iou50",
        "source_scene_view_cache",
    ]

    full_manifest_rows = [{field: row[field] for field in manifest_fieldnames} for row in aligned_rows]
    instance_summary_rows = build_instance_coverage_summary(
        aligned_rows=aligned_rows,
        expected_viewpoints=expected_viewpoints,
    )
    viewpoint_check_rows = build_viewpoint_split_check(
        aligned_rows=aligned_rows,
        viewpoint_inventory_rows=viewpoint_inventory_rows,
    )

    write_rows(output_dir / "full72_manifest.csv", manifest_fieldnames, full_manifest_rows)
    write_rows(output_dir / "full72_manifest_aligned_cache.csv", aligned_cache_fieldnames, aligned_rows)
    write_rows(output_dir / "instance_coverage_summary.csv", list(instance_summary_rows[0].keys()), instance_summary_rows)
    write_rows(output_dir / "viewpoint_split_check.csv", list(viewpoint_check_rows[0].keys()), viewpoint_check_rows)
    (output_dir / "README.md").write_text(
        build_summary_markdown(
            output_dir=output_dir,
            aligned_rows=aligned_rows,
            instance_summary_rows=instance_summary_rows,
            viewpoint_check_rows=viewpoint_check_rows,
        ),
        encoding="utf-8",
    )

    exact_72_count = sum(1 for row in instance_summary_rows if int(row["missing_viewpoint_count"]) == 0 and int(row["unique_viewpoint_count"]) == 72)
    print("Built full 72-view manifest-aligned cache")
    print(f"Output directory: {output_dir}")
    print(f"Rows: {len(aligned_rows)}")
    print(f"Instances with full 72-view union: {exact_72_count}/{len(instance_summary_rows)}")
    print(f"Cached score rows currently available: {sum(int(row['score_available']) for row in aligned_rows)}")


if __name__ == "__main__":
    main()
