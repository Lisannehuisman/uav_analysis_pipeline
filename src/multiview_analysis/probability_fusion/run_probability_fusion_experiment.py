from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from m4_two_drone_operational_analysis.analyze_two_drone_operational import (
    build_view_records,
    load_json,
    match_detections,
    parse_viewpoint_metadata,
    scene_view_rows,
    xywh_to_xyxy,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VIEW_COUNT = 72
EPSILON = 1e-12

DEFAULT_OUTPUT_DIR = ROOT / "probability_fusion" / "outputs" / "m4_test_probability_fusion"

DEFAULT_SPLIT_CONFIG = {
    "val": {
        "scene_view_csv": ROOT / "m4_two_drone_operational_analysis" / "outputs_val" / "scene_view_records.csv",
        "gt_json": ROOT / "outputs" / "detector_family_comparison" / "standardized_val_eval" / "ground_truth" / "M4_val_gt.json",
        "pred_json": ROOT / "outputs" / "detector_family_comparison" / "standardized_val_eval" / "predictions" / "YOLOv8l_M4_val_predictions.json",
    },
    "test": {
        "scene_view_csv": ROOT / "m4_two_drone_operational_analysis" / "outputs_test" / "scene_view_records.csv",
        "gt_json": ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "ground_truth" / "M4_test_gt.json",
        "pred_json": ROOT / "outputs" / "detector_family_comparison" / "standardized_test_eval" / "predictions" / "YOLOv8l_M4_test_predictions.json",
    },
}

PROBABILITY_METHODS = [
    ("max_score", "Max"),
    ("mean_score", "Mean"),
    ("noisy_or_score", "Noisy-OR"),
    ("product_score", "Product"),
    ("log_product_score", "Log-product"),
]

PAIR_FOCUS_GROUPS = [
    "same_elevation_different_azimuth",
    "same_azimuth_different_elevation",
    "different_azimuth_and_different_elevation",
    "nearby_45_pair",
    "opposing_180_pair",
    "same_radius",
    "different_radius",
]


@dataclass
class CalibrationBundle:
    method: str
    note: str
    model: object | None
    fit_sample_count: int
    train_brier: float
    train_ece: float
    eval_brier: float
    eval_ece: float
    eval_auc: float
    eval_ap: float

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        array = np.asarray(scores, dtype=float)
        if self.method == "none" or self.model is None:
            return np.clip(array, 0.0, 1.0)

        positives = array > 0.0
        calibrated = np.zeros_like(array, dtype=float)
        if not np.any(positives):
            return calibrated

        positive_scores = array[positives].reshape(-1, 1)
        if self.method == "platt":
            calibrated[positives] = self.model.predict_proba(positive_scores)[:, 1]
        elif self.method == "isotonic":
            calibrated[positives] = self.model.predict(array[positives])
        else:
            raise KeyError(f"Unsupported calibration method: {self.method}")
        return np.clip(calibrated, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic-only multi-perspective probability fusion experiment. "
            "Reuses cached M4 scene-view records and detector prediction JSONs where available."
        )
    )
    parser.add_argument(
        "--aligned-cache-csv",
        default="",
        help=(
            "Optional full manifest-aligned cache CSV. When provided, the experiment reads per-view rows from this "
            "cache instead of rebuilding from the older split-specific scene_view_records.csv files."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=["test", "val", "combined", "full72"],
        default="test",
        help=(
            "Which synthetic split to evaluate. 'combined' concatenates val and test. "
            "'full72' uses all rows from an aligned cache and is intended for the manifest-complete 72-view setup."
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "isotonic", "platt"],
        default="none",
        help="Optional post-hoc calibration for matched single-view target scores.",
    )
    parser.add_argument(
        "--calibration-split",
        choices=["val", "test"],
        default="val",
        help="Split used to fit the optional calibration model.",
    )
    parser.add_argument(
        "--coalition-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Coalition sizes to evaluate.",
    )
    parser.add_argument(
        "--max-combinations-per-k",
        type=int,
        default=5000,
        help="For larger k, sample at most this many combinations per instance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible coalition sampling.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="Prediction score threshold used if a scene-view cache must be rebuilt from JSON.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to define a valid target match.",
    )
    parser.add_argument(
        "--expected-view-count",
        type=int,
        default=EXPECTED_VIEW_COUNT,
        help="Expected full synthetic view-grid size used in coverage diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for CSVs, plots, and the summary README.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing directly into an existing output directory.",
    )
    parser.add_argument(
        "--rebuild-scene-view-cache",
        action="store_true",
        help="Ignore existing scene_view_records.csv files and rebuild them from GT/prediction JSON if possible.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def safe_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def to_serialized_list(values: Sequence[object], digits: int = 6) -> str:
    serialized: list[str] = []
    for value in values:
        if isinstance(value, float):
            serialized.append(f"{value:.{digits}f}")
        else:
            serialized.append(str(value))
    return "|".join(serialized)


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def azimuth_gap(first: int, second: int) -> int:
    delta = abs(int(first) - int(second)) % 360
    return min(delta, 360 - delta)


def compute_pairwise_azimuth_gaps(azimuths: Sequence[int]) -> list[int]:
    gaps: list[int] = []
    for index, first in enumerate(azimuths):
        for second in azimuths[index + 1 :]:
            gaps.append(azimuth_gap(int(first), int(second)))
    return gaps


def noisy_or(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    complement = 1.0
    for score in scores:
        complement *= max(0.0, 1.0 - float(score))
    return float(1.0 - complement)


def product_score(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return float(np.prod(np.asarray(scores, dtype=float)))


def log_product_score(scores: Sequence[float], epsilon: float = EPSILON) -> float:
    if not scores:
        return float("-inf")
    return float(np.sum(np.log(np.asarray(scores, dtype=float) + epsilon)))


def expected_calibration_error(
    scores: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> float:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    if score_array.size == 0:
        return float("nan")
    if np.min(score_array) < 0.0 or np.max(score_array) > 1.0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = float(len(score_array))
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if right == 1.0:
            mask = (score_array >= left) & (score_array <= right)
        else:
            mask = (score_array >= left) & (score_array < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(score_array[mask]))
        accuracy = float(np.mean(label_array[mask]))
        weight = float(np.sum(mask)) / total
        ece += weight * abs(confidence - accuracy)
    return float(ece)


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {
        "average_precision": float("nan"),
        "auc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "best_threshold": float("nan"),
        "ece_10bin": float("nan"),
        "brier_score": float("nan"),
    }
    if scores.size == 0 or labels.size == 0:
        return metrics

    positive_count = int(np.sum(labels))
    negative_count = int(len(labels) - positive_count)

    if positive_count > 0 and negative_count > 0:
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        metrics["auc"] = float(roc_auc_score(labels, scores))
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        if thresholds.size:
            f1_scores = 2.0 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], EPSILON, None)
            best_index = int(np.argmax(f1_scores))
            metrics["precision"] = float(precision[:-1][best_index])
            metrics["recall"] = float(recall[:-1][best_index])
            metrics["f1"] = float(f1_scores[best_index])
            metrics["best_threshold"] = float(thresholds[best_index])

    if np.min(scores) >= 0.0 and np.max(scores) <= 1.0:
        metrics["ece_10bin"] = expected_calibration_error(scores=scores, labels=labels, num_bins=10)
        metrics["brier_score"] = float(np.mean((scores - labels.astype(float)) ** 2))
    return metrics


def deterministic_rng(seed: int, *tokens: object) -> np.random.Generator:
    digest_input = "::".join([str(seed), *[str(token) for token in tokens]])
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def sample_combinations(
    indices: Sequence[int],
    coalition_size: int,
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    index_array = np.asarray(indices, dtype=int)
    if sample_size <= 0:
        return []

    sampled: set[tuple[int, ...]] = set()
    max_attempts = max(sample_size * 20, 1000)
    attempts = 0
    while len(sampled) < sample_size and attempts < max_attempts:
        draw = tuple(sorted(int(value) for value in rng.choice(index_array, size=coalition_size, replace=False)))
        sampled.add(draw)
        attempts += 1
    return sorted(sampled)


def select_coalitions(
    indices: Sequence[int],
    instance_id: str,
    coalition_size: int,
    max_combinations_per_k: int,
    seed: int,
) -> tuple[list[tuple[int, ...]], str]:
    total = math.comb(len(indices), coalition_size)
    if coalition_size <= 2:
        return list(combinations(indices, coalition_size)), "full_enumeration"
    if total <= max_combinations_per_k:
        return list(combinations(indices, coalition_size)), "full_enumeration"

    rng = deterministic_rng(seed, instance_id, coalition_size)
    sampled = sample_combinations(
        indices=indices,
        coalition_size=coalition_size,
        sample_size=min(max_combinations_per_k, total),
        rng=rng,
    )
    return sampled, "sampled"


def load_scene_view_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected scene-view cache not found: {path}")
    dataframe = pd.read_csv(path)
    numeric_columns = [
        "image_id",
        "azimuth",
        "target_visible",
        "target_detected",
        "target_match_confidence_iou50",
        "target_match_iou_at_confidence_iou50",
        "target_strict_quality_iou50",
        "target_tp",
        "target_fp",
        "target_fn",
    ]
    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)
    return dataframe


def build_scene_view_cache_from_json(
    split_name: str,
    gt_json: Path,
    pred_json: Path,
    score_threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    for path, description in [
        (gt_json, f"{split_name} ground-truth JSON"),
        (pred_json, f"{split_name} prediction JSON"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    records = build_view_records(
        gt_json=gt_json,
        pred_json=pred_json,
        score_threshold=score_threshold,
    )
    rows = scene_view_rows(records)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"scene_view_records_{split_name}.csv"
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    return pd.DataFrame(rows)


def load_or_build_scene_view_cache(
    split_name: str,
    split_config: dict[str, Path],
    score_threshold: float,
    output_dir: Path,
    rebuild_cache: bool,
) -> pd.DataFrame:
    cache_path = split_config["scene_view_csv"]
    if cache_path.is_file() and not rebuild_cache:
        dataframe = load_scene_view_dataframe(cache_path)
        dataframe["source_cache"] = str(cache_path)
        dataframe["source_split"] = split_name
        return dataframe

    dataframe = build_scene_view_cache_from_json(
        split_name=split_name,
        gt_json=split_config["gt_json"],
        pred_json=split_config["pred_json"],
        score_threshold=score_threshold,
        output_dir=output_dir,
    )
    dataframe["source_cache"] = str(output_dir / "cache" / f"scene_view_records_{split_name}.csv")
    dataframe["source_split"] = split_name
    return dataframe


def build_manifest_from_scene_views(scene_df: pd.DataFrame) -> pd.DataFrame:
    manifest = scene_df[
        [
            "source_split",
            "scene_key",
            "target_class",
            "file_name",
            "image_id",
            "viewpoint",
            "elevation",
            "radius",
            "azimuth",
        ]
    ].drop_duplicates()
    manifest = manifest.rename(
        columns={
            "source_split": "split",
            "scene_key": "instance_id",
            "target_class": "class_name",
        }
    ).copy()
    manifest["base_class"] = manifest["class_name"]
    manifest["image_path"] = manifest["file_name"]
    return manifest.sort_values(["split", "instance_id", "viewpoint"]).reset_index(drop=True)


def build_manifest_from_aligned_cache(aligned_df: pd.DataFrame) -> pd.DataFrame:
    manifest = aligned_df[
        [
            "split",
            "instance_id",
            "class_name",
            "base_class",
            "file_name",
            "image_path",
            "viewpoint",
            "elevation",
            "radius",
            "azimuth",
        ]
    ].drop_duplicates()
    return manifest.sort_values(["split", "instance_id", "viewpoint"]).reset_index(drop=True)


def summarize_manifest_coverage(manifest_df: pd.DataFrame, expected_view_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for instance_id, group in manifest_df.groupby("instance_id", sort=True):
        viewpoints = sorted(str(value) for value in group["viewpoint"].unique())
        rows.append(
            {
                "instance_id": instance_id,
                "class_name": str(group["class_name"].iloc[0]),
                "base_class": str(group["base_class"].iloc[0]),
                "split_membership": to_serialized_list(sorted(group["split"].astype(str).unique())),
                "available_view_count": int(group["viewpoint"].nunique()),
                "missing_view_count_vs_expected": max(0, int(expected_view_count - group["viewpoint"].nunique())),
                "available_viewpoints": to_serialized_list(viewpoints),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_level_calibration_rows(
    gt_json: Path,
    pred_json: Path,
    score_threshold: float,
    iou_threshold: float,
) -> pd.DataFrame:
    if not gt_json.is_file():
        raise FileNotFoundError(f"Missing calibration ground-truth JSON: {gt_json}")
    if not pred_json.is_file():
        raise FileNotFoundError(f"Missing calibration prediction JSON: {pred_json}")

    gt = load_json(gt_json)
    preds = load_json(pred_json)

    class_id_to_name = {int(row["id"]): str(row["name"]) for row in gt["categories"]}
    class_name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}
    known_class_names = list(class_name_to_id.keys())

    image_meta = {int(row["id"]): row for row in gt["images"]}
    gt_by_image: dict[int, list[dict[str, object]]] = {}
    pred_by_image: dict[int, list[dict[str, object]]] = {}

    for row in gt["annotations"]:
        gt_by_image.setdefault(int(row["image_id"]), []).append(row)
    for row in preds:
        if float(row["score"]) >= score_threshold:
            pred_by_image.setdefault(int(row["image_id"]), []).append(row)

    rows: list[dict[str, object]] = []
    for image_id in sorted(image_meta):
        meta = image_meta[image_id]
        scene_key, viewpoint, elevation, radius, azimuth, target_class = parse_viewpoint_metadata(
            str(meta["file_name"]),
            known_class_names,
        )
        target_class_id = class_name_to_id[target_class]

        gt_target_annotations = [
            row for row in gt_by_image.get(image_id, []) if int(row["category_id"]) == target_class_id
        ]
        pred_target_annotations = [
            row for row in pred_by_image.get(image_id, []) if int(row["category_id"]) == target_class_id
        ]
        if not pred_target_annotations:
            continue

        gt_boxes = np.array([xywh_to_xyxy(row["bbox"]) for row in gt_target_annotations], dtype=np.float32)
        gt_classes = np.full((len(gt_target_annotations),), target_class_id, dtype=np.int32)
        pred_boxes = np.array([xywh_to_xyxy(row["bbox"]) for row in pred_target_annotations], dtype=np.float32)
        pred_scores = np.array([float(row["score"]) for row in pred_target_annotations], dtype=np.float32)
        pred_classes = np.full((len(pred_target_annotations),), target_class_id, dtype=np.int32)

        if gt_boxes.size == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int32)

        tp_flags, tp_count, _, _ = match_detections(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            iou_threshold=iou_threshold,
        )
        sorted_scores = pred_scores[np.argsort(-pred_scores)]
        for rank_index, (score, flag) in enumerate(zip(sorted_scores, tp_flags, strict=True)):
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": str(meta["file_name"]),
                    "instance_id": scene_key,
                    "target_class": target_class,
                    "viewpoint": viewpoint,
                    "elevation": elevation,
                    "radius": radius,
                    "azimuth": azimuth,
                    "prediction_rank_within_image": rank_index + 1,
                    "raw_score": float(score),
                    "is_correct_target_prediction": int(flag),
                    "num_target_gt_boxes": int(len(gt_target_annotations)),
                    "num_target_tp_predictions": int(tp_count),
                }
            )
    return pd.DataFrame(rows)


def fit_calibration_bundle(
    method: str,
    calibration_df: pd.DataFrame | None,
    evaluation_df: pd.DataFrame | None,
    fallback_note: str = "",
) -> CalibrationBundle:
    if method == "none":
        return CalibrationBundle(
            method="none",
            note="Calibration disabled by configuration." if not fallback_note else fallback_note,
            model=None,
            fit_sample_count=0,
            train_brier=float("nan"),
            train_ece=float("nan"),
            eval_brier=float("nan"),
            eval_ece=float("nan"),
            eval_auc=float("nan"),
            eval_ap=float("nan"),
        )

    if calibration_df is None or calibration_df.empty:
        return CalibrationBundle(
            method="none",
            note=(
                "Calibration requested, but no calibration prediction rows were available. Falling back to raw scores."
                if not fallback_note
                else fallback_note
            ),
            model=None,
            fit_sample_count=0,
            train_brier=float("nan"),
            train_ece=float("nan"),
            eval_brier=float("nan"),
            eval_ece=float("nan"),
            eval_auc=float("nan"),
            eval_ap=float("nan"),
        )

    labels = calibration_df["is_correct_target_prediction"].to_numpy(dtype=int)
    scores = calibration_df["raw_score"].to_numpy(dtype=float)
    if len(np.unique(labels)) < 2:
        return CalibrationBundle(
            method="none",
            note=(
                "Calibration requested, but the calibration split did not contain both positive and negative target-class predictions. Falling back to raw scores."
                if not fallback_note
                else fallback_note
            ),
            model=None,
            fit_sample_count=int(len(calibration_df)),
            train_brier=float("nan"),
            train_ece=float("nan"),
            eval_brier=float("nan"),
            eval_ece=float("nan"),
            eval_auc=float("nan"),
            eval_ap=float("nan"),
        )

    if method == "platt":
        model = LogisticRegression(random_state=0, max_iter=1000)
        model.fit(scores.reshape(-1, 1), labels)
        train_probs = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores, labels)
        train_probs = model.predict(scores)
    else:
        raise KeyError(f"Unknown calibration method: {method}")

    train_metrics = compute_binary_metrics(np.asarray(train_probs, dtype=float), labels)
    if evaluation_df is not None and not evaluation_df.empty and len(np.unique(evaluation_df["is_correct_target_prediction"])) >= 2:
        eval_scores = evaluation_df["raw_score"].to_numpy(dtype=float)
        eval_labels = evaluation_df["is_correct_target_prediction"].to_numpy(dtype=int)
        if method == "platt":
            eval_probs = model.predict_proba(eval_scores.reshape(-1, 1))[:, 1]
        else:
            eval_probs = model.predict(eval_scores)
        eval_metrics = compute_binary_metrics(np.asarray(eval_probs, dtype=float), eval_labels)
    else:
        eval_metrics = {
            "brier_score": float("nan"),
            "ece_10bin": float("nan"),
            "auc": float("nan"),
            "average_precision": float("nan"),
        }

    note = (
        "Calibration was fit on target-class prediction correctness and then applied only to positive matched per-view scores. "
        "Views without a valid IoU>=0.5 target match remain at score 0."
    )
    return CalibrationBundle(
        method=method,
        note=note,
        model=model,
        fit_sample_count=int(len(calibration_df)),
        train_brier=float(train_metrics["brier_score"]),
        train_ece=float(train_metrics["ece_10bin"]),
        eval_brier=float(eval_metrics["brier_score"]),
        eval_ece=float(eval_metrics["ece_10bin"]),
        eval_auc=float(eval_metrics["auc"]),
        eval_ap=float(eval_metrics["average_precision"]),
    )


def build_per_view_scores(
    scene_df: pd.DataFrame,
    calibrator: CalibrationBundle,
) -> pd.DataFrame:
    per_view = scene_df.copy()
    per_view = per_view.rename(columns={"scene_key": "instance_id", "target_class": "class_name"})
    per_view["base_class"] = per_view["class_name"]
    per_view["raw_view_score"] = pd.to_numeric(per_view["target_match_confidence_iou50"], errors="coerce").fillna(0.0)
    per_view["view_found_label"] = pd.to_numeric(per_view["target_detected"], errors="coerce").fillna(0).astype(int)
    per_view["view_score"] = calibrator.transform(per_view["raw_view_score"].to_numpy(dtype=float))
    per_view["best_iou"] = pd.to_numeric(per_view["target_best_iou"], errors="coerce").fillna(0.0)
    per_view["match_iou_at_score"] = pd.to_numeric(
        per_view["target_match_iou_at_confidence_iou50"], errors="coerce"
    ).fillna(0.0)
    per_view["split"] = per_view["source_split"].astype(str)
    keep_columns = [
        "split",
        "instance_id",
        "class_name",
        "base_class",
        "file_name",
        "image_id",
        "viewpoint",
        "elevation",
        "radius",
        "azimuth",
        "raw_view_score",
        "view_score",
        "view_found_label",
        "best_iou",
        "match_iou_at_score",
        "source_cache",
    ]
    per_view = per_view[keep_columns].sort_values(
        ["instance_id", "viewpoint", "view_score", "raw_view_score", "split"],
        ascending=[True, True, False, False, True],
    )
    per_view = per_view.drop_duplicates(subset=["instance_id", "viewpoint"], keep="first")
    return per_view.sort_values(["split", "instance_id", "viewpoint"]).reset_index(drop=True)


def build_per_view_scores_from_aligned_cache(
    aligned_df: pd.DataFrame,
    calibrator: CalibrationBundle,
) -> pd.DataFrame:
    available = aligned_df.copy()
    if "score_available" in available.columns:
        available["score_available"] = pd.to_numeric(available["score_available"], errors="coerce").fillna(0).astype(int)
        available = available.loc[available["score_available"] == 1].copy()
    if available.empty:
        return available

    available["raw_view_score"] = pd.to_numeric(available["raw_view_score"], errors="coerce").fillna(0.0)
    available["view_found_label"] = pd.to_numeric(available["view_found_label"], errors="coerce").fillna(0).astype(int)
    available["best_iou"] = pd.to_numeric(available["best_iou"], errors="coerce").fillna(0.0)
    available["match_iou_at_score"] = pd.to_numeric(available["match_iou_at_score"], errors="coerce").fillna(0.0)
    available["view_score"] = calibrator.transform(available["raw_view_score"].to_numpy(dtype=float))
    available["source_cache"] = available.get("source_scene_view_cache", "")

    keep_columns = [
        "split",
        "instance_id",
        "class_name",
        "base_class",
        "file_name",
        "image_path",
        "image_id",
        "viewpoint",
        "elevation",
        "radius",
        "azimuth",
        "raw_view_score",
        "view_score",
        "view_found_label",
        "best_iou",
        "match_iou_at_score",
        "source_cache",
    ]
    for column in keep_columns:
        if column not in available.columns:
            available[column] = ""
    return available[keep_columns].sort_values(["split", "instance_id", "viewpoint"]).reset_index(drop=True)


def classify_geometry(coalition_size: int, elevations: Sequence[str], radii: Sequence[str], azimuths: Sequence[int]) -> dict[str, object]:
    unique_elevations = sorted(set(elevations))
    unique_radii = sorted(set(radii))
    unique_azimuths = sorted(set(int(value) for value in azimuths))
    gaps = compute_pairwise_azimuth_gaps(unique_azimuths)

    geometry_group = "other"
    if coalition_size == 1:
        geometry_group = "single_view"
    elif len(unique_elevations) == 1 and len(unique_azimuths) > 1:
        geometry_group = "same_elevation_different_azimuth"
    elif len(unique_azimuths) == 1 and len(unique_elevations) > 1:
        geometry_group = "same_azimuth_different_elevation"
    elif len(unique_elevations) > 1 and len(unique_azimuths) > 1:
        geometry_group = "different_azimuth_and_different_elevation"

    if coalition_size == 1:
        azimuth_relation = "single_view"
    elif len(unique_azimuths) == 1:
        azimuth_relation = "same_azimuth"
    elif coalition_size == 2 and gaps and gaps[0] == 45:
        azimuth_relation = "nearby_45_pair"
    elif coalition_size == 2 and gaps and gaps[0] == 180:
        azimuth_relation = "opposing_180_pair"
    elif coalition_size == 2:
        azimuth_relation = "other_pair_gap"
    else:
        azimuth_relation = "mixed_multi_view"

    elevation_relation = "same_elevation" if len(unique_elevations) == 1 else "different_elevation"
    radius_relation = "same_radius" if len(unique_radii) == 1 else "different_radius"

    return {
        "unique_elevation_count": len(unique_elevations),
        "unique_radius_count": len(unique_radii),
        "unique_azimuth_count": len(unique_azimuths),
        "elevation_relation": elevation_relation,
        "radius_relation": radius_relation,
        "geometry_group": geometry_group,
        "azimuth_relation": azimuth_relation,
        "azimuth_gap_min": int(min(gaps)) if gaps else 0,
        "azimuth_gap_max": int(max(gaps)) if gaps else 0,
    }


def build_coalition_rows(
    per_view_df: pd.DataFrame,
    coalition_sizes: Sequence[int],
    max_combinations_per_k: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for instance_id, group in per_view_df.groupby("instance_id", sort=True):
        group = group.sort_values(["split", "viewpoint", "image_id"]).reset_index(drop=True)
        available_count = int(len(group))
        indices = list(range(available_count))
        if available_count == 0:
            continue

        for coalition_size in sorted(set(int(value) for value in coalition_sizes)):
            if coalition_size <= 0 or available_count < coalition_size:
                continue
            selected, selection_mode = select_coalitions(
                indices=indices,
                instance_id=str(instance_id),
                coalition_size=coalition_size,
                max_combinations_per_k=max_combinations_per_k,
                seed=seed,
            )
            for coalition_index, selected_indices in enumerate(selected, start=1):
                members = group.iloc[list(selected_indices)].copy()
                scores = members["view_score"].to_numpy(dtype=float)
                raw_scores = members["raw_view_score"].to_numpy(dtype=float)
                found_flags = members["view_found_label"].to_numpy(dtype=int)
                best_single = float(np.max(scores)) if scores.size else 0.0
                coalition_found = int(np.max(found_flags)) if found_flags.size else 0

                max_value = float(np.max(scores)) if scores.size else 0.0
                mean_value = float(np.mean(scores)) if scores.size else 0.0
                noisy_or_value = noisy_or(scores.tolist())
                product_value = product_score(scores.tolist())
                log_product_value = log_product_score(scores.tolist())

                geometry = classify_geometry(
                    coalition_size=coalition_size,
                    elevations=members["elevation"].astype(str).tolist(),
                    radii=members["radius"].astype(str).tolist(),
                    azimuths=members["azimuth"].astype(int).tolist(),
                )

                row = {
                    "instance_id": str(instance_id),
                    "class_name": str(members["class_name"].iloc[0]),
                    "base_class": str(members["base_class"].iloc[0]),
                    "split_membership": to_serialized_list(sorted(members["split"].astype(str).unique())),
                    "available_views_for_instance": available_count,
                    "coalition_index": coalition_index,
                    "coalition_size": coalition_size,
                    "selection_mode": selection_mode,
                    "views_included": to_serialized_list(members["viewpoint"].astype(str).tolist()),
                    "file_names": to_serialized_list(members["file_name"].astype(str).tolist()),
                    "image_ids": to_serialized_list(members["image_id"].astype(int).tolist(), digits=0),
                    "azimuths": to_serialized_list(members["azimuth"].astype(int).tolist(), digits=0),
                    "elevations": to_serialized_list(members["elevation"].astype(str).tolist()),
                    "radii": to_serialized_list(members["radius"].astype(str).tolist()),
                    "individual_raw_scores": to_serialized_list(raw_scores.tolist()),
                    "individual_scores": to_serialized_list(scores.tolist()),
                    "individual_found_labels": to_serialized_list(found_flags.astype(int).tolist(), digits=0),
                    "best_single_score": best_single,
                    "coalition_found_label": coalition_found,
                    "target_present_label": 1,
                    "supporting_view_count": int(np.sum(found_flags)),
                    "max_score": max_value,
                    "mean_score": mean_value,
                    "noisy_or_score": noisy_or_value,
                    "product_score": product_value,
                    "log_product_score": log_product_value,
                    "delta_max_vs_best_single": max_value - best_single,
                    "delta_mean_vs_best_single": mean_value - best_single,
                    "delta_noisy_or_vs_best_single": noisy_or_value - best_single,
                    "delta_product_vs_best_single": product_value - best_single,
                    "delta_log_product_vs_best_single": log_product_value - best_single,
                }
                row.update(geometry)
                rows.append(row)

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    sort_columns = ["coalition_size", "instance_id", "coalition_index"]
    return dataframe.sort_values(sort_columns).reset_index(drop=True)


def summarize_methods(coalition_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for coalition_size in sorted(coalition_df["coalition_size"].unique()):
        subset = coalition_df.loc[coalition_df["coalition_size"] == coalition_size].copy()
        labels = subset["coalition_found_label"].to_numpy(dtype=int)
        found_rate = float(np.mean(labels)) if labels.size else float("nan")
        for method_key, method_label in PROBABILITY_METHODS:
            scores = subset[method_key].to_numpy(dtype=float)
            metrics = compute_binary_metrics(scores=scores, labels=labels)
            summary_rows.append(
                {
                    "fusion_method": method_key,
                    "fusion_label": method_label,
                    "coalition_size": int(coalition_size),
                    "num_coalitions": int(len(subset)),
                    "mean_score": float(np.mean(scores)) if scores.size else float("nan"),
                    "found_rate": found_rate,
                    "average_precision": float(metrics["average_precision"]),
                    "auc": float(metrics["auc"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "best_threshold": float(metrics["best_threshold"]),
                    "calibration_error_ece10": float(metrics["ece_10bin"]),
                    "brier_score": float(metrics["brier_score"]),
                    "mean_delta_vs_best_single": float(np.mean(scores - subset["best_single_score"].to_numpy(dtype=float))),
                    "improved_over_best_single_rate": float(
                        np.mean(scores > (subset["best_single_score"].to_numpy(dtype=float) + EPSILON))
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(["fusion_method", "coalition_size"]).reset_index(drop=True)
    summary_df["delta_mean_score_vs_previous_k"] = np.nan
    summary_df["delta_found_rate_vs_previous_k"] = np.nan
    summary_df["delta_ap_vs_previous_k"] = np.nan
    summary_df["delta_mean_score_vs_k1"] = np.nan
    summary_df["delta_found_rate_vs_k1"] = np.nan
    summary_df["delta_ap_vs_k1"] = np.nan

    for method_key in summary_df["fusion_method"].unique():
        method_mask = summary_df["fusion_method"] == method_key
        method_rows = summary_df.loc[method_mask].sort_values("coalition_size")
        if method_rows.empty:
            continue
        base_row = method_rows.iloc[0]
        previous_row = None
        for row_index, row in method_rows.iterrows():
            if previous_row is not None:
                summary_df.loc[row_index, "delta_mean_score_vs_previous_k"] = float(row["mean_score"] - previous_row["mean_score"])
                summary_df.loc[row_index, "delta_found_rate_vs_previous_k"] = float(row["found_rate"] - previous_row["found_rate"])
                if not math.isnan(float(row["average_precision"])) and not math.isnan(float(previous_row["average_precision"])):
                    summary_df.loc[row_index, "delta_ap_vs_previous_k"] = float(row["average_precision"] - previous_row["average_precision"])
            summary_df.loc[row_index, "delta_mean_score_vs_k1"] = float(row["mean_score"] - base_row["mean_score"])
            summary_df.loc[row_index, "delta_found_rate_vs_k1"] = float(row["found_rate"] - base_row["found_rate"])
            if not math.isnan(float(row["average_precision"])) and not math.isnan(float(base_row["average_precision"])):
                summary_df.loc[row_index, "delta_ap_vs_k1"] = float(row["average_precision"] - base_row["average_precision"])
            previous_row = row
    return summary_df


def summarize_geometry(coalition_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    relation_columns = [
        "geometry_group",
        "azimuth_relation",
        "elevation_relation",
        "radius_relation",
    ]
    for relation_column in relation_columns:
        for (coalition_size, relation_value), subset in coalition_df.groupby(["coalition_size", relation_column], sort=True):
            labels = subset["coalition_found_label"].to_numpy(dtype=int)
            for method_key, method_label in PROBABILITY_METHODS:
                scores = subset[method_key].to_numpy(dtype=float)
                metrics = compute_binary_metrics(scores=scores, labels=labels)
                rows.append(
                    {
                        "relation_type": relation_column,
                        "relation_group": relation_value,
                        "coalition_size": int(coalition_size),
                        "fusion_method": method_key,
                        "fusion_label": method_label,
                        "num_coalitions": int(len(subset)),
                        "mean_score": float(np.mean(scores)) if scores.size else float("nan"),
                        "found_rate": float(np.mean(labels)) if labels.size else float("nan"),
                        "average_precision": float(metrics["average_precision"]),
                        "auc": float(metrics["auc"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "best_threshold": float(metrics["best_threshold"]),
                        "mean_delta_vs_best_single": float(np.mean(scores - subset["best_single_score"].to_numpy(dtype=float))),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pair_geometry_focus(coalition_df: pd.DataFrame) -> pd.DataFrame:
    pair_df = coalition_df.loc[coalition_df["coalition_size"] == 2].copy()
    rows: list[dict[str, object]] = []
    if pair_df.empty:
        return pd.DataFrame(rows)

    filters = {
        "same_elevation_different_azimuth": pair_df["geometry_group"] == "same_elevation_different_azimuth",
        "same_azimuth_different_elevation": pair_df["geometry_group"] == "same_azimuth_different_elevation",
        "different_azimuth_and_different_elevation": pair_df["geometry_group"] == "different_azimuth_and_different_elevation",
        "nearby_45_pair": pair_df["azimuth_relation"] == "nearby_45_pair",
        "opposing_180_pair": pair_df["azimuth_relation"] == "opposing_180_pair",
        "same_radius": pair_df["radius_relation"] == "same_radius",
        "different_radius": pair_df["radius_relation"] == "different_radius",
    }

    for group_name, mask in filters.items():
        subset = pair_df.loc[mask].copy()
        if subset.empty:
            continue
        labels = subset["coalition_found_label"].to_numpy(dtype=int)
        for method_key, method_label in PROBABILITY_METHODS:
            scores = subset[method_key].to_numpy(dtype=float)
            metrics = compute_binary_metrics(scores=scores, labels=labels)
            rows.append(
                {
                    "geometry_focus_group": group_name,
                    "fusion_method": method_key,
                    "fusion_label": method_label,
                    "num_coalitions": int(len(subset)),
                    "mean_score": float(np.mean(scores)) if scores.size else float("nan"),
                    "found_rate": float(np.mean(labels)) if labels.size else float("nan"),
                    "average_precision": float(metrics["average_precision"]),
                    "auc": float(metrics["auc"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "best_threshold": float(metrics["best_threshold"]),
                    "mean_delta_vs_best_single": float(np.mean(scores - subset["best_single_score"].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def plot_method_performance(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method_key, method_label in PROBABILITY_METHODS:
        subset = summary_df.loc[summary_df["fusion_method"] == method_key].sort_values("coalition_size")
        if subset.empty:
            continue
        axes[0].plot(subset["coalition_size"], subset["average_precision"], marker="o", label=method_label)
        axes[1].plot(subset["coalition_size"], subset["f1"], marker="o", label=method_label)

    axes[0].set_title("Average Precision by coalition size")
    axes[0].set_xlabel("Coalition size k")
    axes[0].set_ylabel("Average precision")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Best-threshold F1 by coalition size")
    axes[1].set_xlabel("Coalition size k")
    axes[1].set_ylabel("F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.savefig(output_dir / "fusion_method_performance_by_coalition_size.png", dpi=200)
    plt.close(fig)


def plot_marginal_gain(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method_key, method_label in PROBABILITY_METHODS:
        subset = summary_df.loc[summary_df["fusion_method"] == method_key].sort_values("coalition_size")
        if subset.empty:
            continue
        axes[0].plot(subset["coalition_size"], subset["delta_mean_score_vs_previous_k"], marker="o", label=method_label)
        axes[1].plot(subset["coalition_size"], subset["delta_ap_vs_previous_k"], marker="o", label=method_label)

    axes[0].set_title("Marginal mean-score gain vs previous k")
    axes[0].set_xlabel("Coalition size k")
    axes[0].set_ylabel("Delta mean score")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Marginal AP gain vs previous k")
    axes[1].set_xlabel("Coalition size k")
    axes[1].set_ylabel("Delta average precision")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.savefig(output_dir / "marginal_gain_from_adding_views.png", dpi=200)
    plt.close(fig)


def plot_max_noisy_or_product(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    focus_methods = ["max_score", "noisy_or_score", "product_score"]
    label_lookup = dict(PROBABILITY_METHODS)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method_key in focus_methods:
        subset = summary_df.loc[summary_df["fusion_method"] == method_key].sort_values("coalition_size")
        if subset.empty:
            continue
        ax.plot(
            subset["coalition_size"],
            subset["mean_delta_vs_best_single"],
            marker="o",
            label=label_lookup[method_key],
        )
    ax.set_title("Max vs Noisy-OR vs Product")
    ax.set_xlabel("Coalition size k")
    ax.set_ylabel("Mean uplift vs best constituent single view")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "max_vs_noisy_or_vs_product.png", dpi=200)
    plt.close(fig)


def plot_geometry_focus(pair_geometry_df: pd.DataFrame, output_dir: Path) -> None:
    if pair_geometry_df.empty:
        return
    focus = pair_geometry_df.loc[
        pair_geometry_df["fusion_method"].isin(["max_score", "noisy_or_score", "product_score"])
    ].copy()
    if focus.empty:
        return

    pivot = focus.pivot(index="geometry_focus_group", columns="fusion_label", values="mean_delta_vs_best_single").fillna(0.0)
    ordered_index = [group for group in PAIR_FOCUS_GROUPS if group in pivot.index]
    pivot = pivot.loc[ordered_index]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Pair-geometry comparison by fusion rule")
    ax.set_xlabel("Geometry focus group")
    ax.set_ylabel("Mean uplift vs best constituent single view")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "geometry_group_comparison.png", dpi=200)
    plt.close(fig)


def plot_reliability(
    calibration_rows: pd.DataFrame,
    calibrator: CalibrationBundle,
    output_dir: Path,
    title_prefix: str,
) -> None:
    if calibration_rows.empty or calibrator.method == "none" or calibrator.model is None:
        return
    labels = calibration_rows["is_correct_target_prediction"].to_numpy(dtype=int)
    raw_scores = calibration_rows["raw_score"].to_numpy(dtype=float)
    calibrated_scores = calibrator.transform(raw_scores)
    if len(np.unique(labels)) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    raw_fraction, raw_mean = calibration_curve(labels, raw_scores, n_bins=10, strategy="uniform")
    cal_fraction, cal_mean = calibration_curve(labels, calibrated_scores, n_bins=10, strategy="uniform")

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Ideal")
    ax.plot(raw_mean, raw_fraction, marker="o", label="Raw score")
    ax.plot(cal_mean, cal_fraction, marker="o", label=f"{calibrator.method} calibrated")
    ax.set_title(f"{title_prefix} reliability")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed correctness")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "reliability_plot.png", dpi=200)
    plt.close(fig)


def save_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    dataframe.to_csv(path, index=False)


def save_json(path: Path, payload: dict[str, object]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def headline_metric(row: pd.Series) -> float:
    ap = float(row.get("average_precision", float("nan")))
    if not math.isnan(ap):
        return ap
    f1 = float(row.get("f1", float("nan")))
    if not math.isnan(f1):
        return f1
    return float(row.get("mean_score", float("-inf")))


def sort_summary_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    sortable = dataframe.copy()
    sortable["average_precision_rank"] = pd.to_numeric(sortable["average_precision"], errors="coerce").fillna(-1.0)
    sortable["mean_delta_rank"] = pd.to_numeric(sortable["mean_delta_vs_best_single"], errors="coerce").fillna(float("-inf"))
    sortable["f1_rank"] = pd.to_numeric(sortable["f1"], errors="coerce").fillna(-1.0)
    sortable["mean_score_rank"] = pd.to_numeric(sortable["mean_score"], errors="coerce").fillna(float("-inf"))
    return sortable.sort_values(
        [
            "average_precision_rank",
            "mean_delta_rank",
            "f1_rank",
            "mean_score_rank",
            "coalition_size",
        ],
        ascending=[False, False, False, False, False],
    )


def select_summary_row(
    summary_df: pd.DataFrame,
    *,
    multi_view_only: bool = False,
    methods: Sequence[str] | None = None,
) -> pd.Series | None:
    subset = summary_df.copy()
    if methods is not None:
        subset = subset.loc[subset["fusion_method"].isin(methods)].copy()
    if multi_view_only and (subset["coalition_size"] > 1).any():
        subset = subset.loc[subset["coalition_size"] > 1].copy()
    if subset.empty:
        return None
    return sort_summary_rows(subset).iloc[0]


def select_pair_geometry_row(pair_geometry_df: pd.DataFrame) -> pd.Series | None:
    if pair_geometry_df.empty:
        return None
    sortable = pair_geometry_df.copy()
    sortable["average_precision_rank"] = pd.to_numeric(sortable["average_precision"], errors="coerce").fillna(-1.0)
    sortable["mean_delta_rank"] = pd.to_numeric(sortable["mean_delta_vs_best_single"], errors="coerce").fillna(float("-inf"))
    sortable["f1_rank"] = pd.to_numeric(sortable["f1"], errors="coerce").fillna(-1.0)
    sortable["mean_score_rank"] = pd.to_numeric(sortable["mean_score"], errors="coerce").fillna(float("-inf"))
    return sortable.sort_values(
        ["average_precision_rank", "mean_delta_rank", "f1_rank", "mean_score_rank"],
        ascending=[False, False, False, False],
    ).iloc[0]


def build_run_summary_markdown(
    args: argparse.Namespace,
    output_dir: Path,
    manifest_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    per_view_df: pd.DataFrame,
    coalition_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pair_geometry_df: pd.DataFrame,
    calibrator: CalibrationBundle,
) -> str:
    coverage_counts = coverage_df["available_view_count"].tolist() if not coverage_df.empty else []
    coverage_min = int(min(coverage_counts)) if coverage_counts else 0
    coverage_max = int(max(coverage_counts)) if coverage_counts else 0
    coverage_mean = float(np.mean(coverage_counts)) if coverage_counts else 0.0
    full_grid_count = int(np.sum(coverage_df["available_view_count"] == args.expected_view_count)) if not coverage_df.empty else 0

    best_row = select_summary_row(summary_df, multi_view_only=True)
    best_rescue = select_summary_row(
        summary_df,
        multi_view_only=True,
        methods=["max_score", "noisy_or_score"],
    )
    best_product = select_summary_row(
        summary_df,
        multi_view_only=True,
        methods=["product_score"],
    )

    interpretation_lines = [
        "This experiment tests multi-perspective probability fusion only in the synthetic setting, where repeated observations of the same object are available under known viewpoint geometry.",
        "Product-based fusion is interpreted as an agreement model, while max and noisy-OR are interpreted as rescue models.",
    ]
    if best_rescue is not None and best_product is not None:
        rescue_ap = float(best_rescue["average_precision"])
        product_ap = float(best_product["average_precision"])
        rescue_delta = float(best_rescue["mean_delta_vs_best_single"])
        product_delta = float(best_product["mean_delta_vs_best_single"])
        if not math.isnan(rescue_delta) and not math.isnan(product_delta):
            if rescue_delta > product_delta + 1e-6:
                interpretation_lines.append(
                    "In this cache, rescue-style fusion improved over the best constituent single view more than product fusion did, which suggests that multi-view benefit mainly comes from increasing the chance that at least one viewpoint sees the target clearly."
                )
            elif product_delta > rescue_delta + 1e-6:
                interpretation_lines.append(
                    "In this cache, product fusion matched or exceeded the rescue-style uplift over the best constituent single view, so there is some support for an agreement-style interpretation."
                )
            else:
                interpretation_lines.append(
                    "In this cache, product fusion and rescue-style fusion showed similar uplift over the best constituent single view, so the evidence does not strongly favor one interpretation."
                )
        elif not math.isnan(rescue_ap) and not math.isnan(product_ap):
            if rescue_ap > product_ap + 1e-6:
                interpretation_lines.append(
                    "In this cache, rescue-style fusion also outperformed product fusion on average precision, which points in the same rescue-oriented direction."
                )
            elif product_ap > rescue_ap + 1e-6:
                interpretation_lines.append(
                    "In this cache, product fusion exceeded the best rescue rule on average precision, which points more toward agreement."
                )
        else:
            interpretation_lines.append(
                "Average-precision comparison between rescue and agreement rules was not available, so the interpretation should rely on the thresholded summaries and mean-score trends."
            )

    lines = [
        "# Synthetic Multi-Perspective Probability Fusion",
        "",
        "## What was tested",
        "",
        "- Synthetic-only target-level fusion using cached YOLOv8l M4 detections.",
        f"- Evaluation split: `{args.evaluation_split}`.",
        f"- Coalition sizes: `{', '.join(str(value) for value in sorted(set(args.coalition_sizes)))}`.",
        "- Fusion rules: `max`, `mean`, `noisy_or`, `product`, and `log_product`.",
        "",
        "## Why this is synthetic-only",
        "",
        "- The experiment groups repeated views of the same synthetic object instance (`scene_key`) under known azimuth, elevation, and radius geometry.",
        "- No real-image files are used anywhere in this analysis.",
        "",
        "## How scores were extracted",
        "",
        "- Per-view target score `p_{i,v}` is the best correct-class detection confidence with IoU >= 0.5 against a target-class ground-truth box.",
        "- If no valid target match exists in a view, that view score is set to `0` and the view-level found label is `0`.",
        "- Coalition labels are operational target-recovery labels: a coalition is labeled `found=1` if at least one constituent view contains a valid target match.",
        "",
        "## Coverage note",
        "",
        f"- Expected full grid size: `{args.expected_view_count}` views per instance.",
        f"- Observed cached coverage: min `{coverage_min}`, max `{coverage_max}`, mean `{coverage_mean:.2f}` views per instance.",
        f"- Instances with the full expected grid present in the current cache: `{full_grid_count}` / `{len(coverage_df)}`.",
        "- The script therefore reports the actual cached view coverage and proceeds without pretending that a full 72-view reconstruction exists when it does not.",
        "",
        "## Calibration",
        "",
        f"- Requested calibration mode: `{args.calibration}`.",
        f"- Effective calibration mode: `{calibrator.method}`.",
        f"- Note: {calibrator.note}",
        f"- Calibration fit samples: `{calibrator.fit_sample_count}`.",
        "",
        "## Main result",
        "",
    ]

    if best_row is not None:
        lines.extend(
            [
                f"- Best multi-view summary row: `{best_row['fusion_label']}` at `k={int(best_row['coalition_size'])}`.",
                f"- Mean fused score: `{float(best_row['mean_score']):.4f}`.",
                f"- Average precision: `{float(best_row['average_precision']):.4f}`.",
                f"- Best-threshold F1: `{float(best_row['f1']):.4f}` at threshold `{float(best_row['best_threshold']):.4f}`.",
                f"- Mean uplift vs best constituent single view: `{float(best_row['mean_delta_vs_best_single']):+.4f}`.",
                f"- Rate of beating the best constituent single view: `{float(best_row['improved_over_best_single_rate']):.4f}`.",
            ]
        )
    else:
        lines.append("- No coalition rows were generated, so there is no headline result.")
    if best_rescue is not None and best_product is not None:
        lines.extend(
            [
                f"- Best rescue row: `{best_rescue['fusion_label']}` at `k={int(best_rescue['coalition_size'])}` with uplift `{float(best_rescue['mean_delta_vs_best_single']):+.4f}`.",
                f"- Best product row: `{best_product['fusion_label']}` at `k={int(best_product['coalition_size'])}` with uplift `{float(best_product['mean_delta_vs_best_single']):+.4f}`.",
            ]
        )
    lines.extend(
        [
            "- The found-label AP and F1 metrics are near-ceiling for `k=1` because the per-view score is itself defined from valid target matches; the multi-view interpretation should therefore focus mainly on coalition-size trends and uplift over the best constituent single view.",
        ]
    )

    lines.extend(
        [
            "",
            "## Rescue vs agreement interpretation",
            "",
        ]
    )
    lines.extend([f"- {line}" for line in interpretation_lines])

    best_pair_geom = select_pair_geometry_row(pair_geometry_df)
    if best_pair_geom is not None:
        lines.extend(
            [
                "",
                "## Geometry signal",
                "",
                f"- Strongest pairwise geometry summary row: `{best_pair_geom['geometry_focus_group']}` with `{best_pair_geom['fusion_label']}`.",
                f"- Pair-group mean uplift vs best constituent single view: `{float(best_pair_geom['mean_delta_vs_best_single']):+.4f}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Important limitations",
            "",
            "- The current cached prediction files do not provide persistent 3D object IDs inside the annotation JSON, so IoU validation is target-class based rather than explicit world-instance-ID based.",
            "- If the requested calibration mode is enabled, it is fit at prediction level and then applied to matched per-view scores; zero-score views remain zero.",
            "- When `evaluation_split=combined`, view density improves, but the same object instances appear across val and test, so split purity is weaker than in `test`-only evaluation.",
            "",
            "## Output files",
            "",
            "- `manifest_proxy.csv`",
            "- `view_coverage_summary.csv`",
            "- `per_view_scores.csv`",
            "- `per_coalition_scores.csv`",
            "- `fusion_summary.csv`",
            "- `geometry_group_summary.csv`",
            "- `pair_geometry_focus_summary.csv`",
            "- `fusion_method_performance_by_coalition_size.png`",
            "- `marginal_gain_from_adding_views.png`",
            "- `max_vs_noisy_or_vs_product.png`",
            "- `geometry_group_comparison.png`",
        ]
    )
    if calibrator.method != "none":
        lines.append("- `reliability_plot.png`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This experiment tests multi-perspective probability fusion only in the synthetic setting, where repeated observations of the same object are available under known viewpoint geometry. Product-based fusion is interpreted as an agreement model, while max and noisy-OR fusion are interpreted as rescue models. If max or noisy-OR outperform product fusion, this suggests that multi-view benefit in object detection arises mainly from increasing the probability that at least one viewpoint observes the target clearly, rather than from multiplying mutually reinforcing probabilities across all views."
    )
    lines.append("")
    lines.append(f"Generated in `{output_dir}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = unique_output_dir(Path(args.output_dir).resolve(), overwrite=args.overwrite)

    save_json(
        output_dir / "run_config.json",
        {
            "evaluation_split": args.evaluation_split,
            "calibration": args.calibration,
            "calibration_split": args.calibration_split,
            "coalition_sizes": list(sorted(set(args.coalition_sizes))),
            "max_combinations_per_k": args.max_combinations_per_k,
            "seed": args.seed,
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "expected_view_count": args.expected_view_count,
            "output_dir": str(output_dir),
            "rebuild_scene_view_cache": bool(args.rebuild_scene_view_cache),
            "aligned_cache_csv": args.aligned_cache_csv,
        },
    )

    aligned_cache_df: pd.DataFrame | None = None
    evaluation_scene_df: pd.DataFrame | None = None
    manifest_df: pd.DataFrame
    coverage_df: pd.DataFrame

    if args.aligned_cache_csv:
        aligned_cache_path = Path(args.aligned_cache_csv).resolve()
        if not aligned_cache_path.is_file():
            raise FileNotFoundError(f"Aligned cache CSV not found: {aligned_cache_path}")
        aligned_cache_df = pd.read_csv(aligned_cache_path)
        if args.evaluation_split == "full72":
            evaluation_aligned_df = aligned_cache_df.copy()
        elif args.evaluation_split == "combined":
            evaluation_aligned_df = aligned_cache_df.loc[aligned_cache_df["split"].astype(str).isin(["val", "test"])].copy()
        else:
            evaluation_aligned_df = aligned_cache_df.loc[aligned_cache_df["split"].astype(str) == args.evaluation_split].copy()
        manifest_df = build_manifest_from_aligned_cache(aligned_cache_df)
        coverage_df = summarize_manifest_coverage(manifest_df=manifest_df, expected_view_count=args.expected_view_count)
    else:
        if args.evaluation_split == "full72":
            raise SystemExit("--evaluation-split full72 requires --aligned-cache-csv.")
        needed_splits = {args.evaluation_split}
        if args.evaluation_split == "combined":
            needed_splits.update({"val", "test"})
        if args.calibration != "none":
            needed_splits.add(args.calibration_split)
        split_frames: dict[str, pd.DataFrame] = {}
        for split_name in sorted(split for split in needed_splits if split in {"val", "test"}):
            split_frames[split_name] = load_or_build_scene_view_cache(
                split_name=split_name,
                split_config=DEFAULT_SPLIT_CONFIG[split_name],
                score_threshold=args.score_threshold,
                output_dir=output_dir,
                rebuild_cache=args.rebuild_scene_view_cache,
            )

        if args.evaluation_split == "combined":
            evaluation_scene_df = pd.concat([split_frames["val"], split_frames["test"]], ignore_index=True)
        else:
            evaluation_scene_df = split_frames[args.evaluation_split].copy()

        manifest_df = build_manifest_from_scene_views(evaluation_scene_df)
        coverage_df = summarize_manifest_coverage(manifest_df=manifest_df, expected_view_count=args.expected_view_count)

    calibration_rows_train: pd.DataFrame | None = None
    calibration_rows_eval: pd.DataFrame | None = None
    calibration_fallback_note = ""
    if args.calibration != "none":
        try:
            calibration_rows_train = build_prediction_level_calibration_rows(
                gt_json=DEFAULT_SPLIT_CONFIG[args.calibration_split]["gt_json"],
                pred_json=DEFAULT_SPLIT_CONFIG[args.calibration_split]["pred_json"],
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
            evaluation_for_calibration = "test" if args.calibration_split == "val" else "val"
            if args.evaluation_split in {"val", "test"}:
                evaluation_for_calibration = args.evaluation_split
            calibration_rows_eval = build_prediction_level_calibration_rows(
                gt_json=DEFAULT_SPLIT_CONFIG[evaluation_for_calibration]["gt_json"],
                pred_json=DEFAULT_SPLIT_CONFIG[evaluation_for_calibration]["pred_json"],
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
        except FileNotFoundError as error:
            calibration_fallback_note = (
                "Calibration requested, but the required GT/prediction JSON files were not all available "
                f"for prediction-level fitting. Falling back to raw scores. Missing input: {error}"
            )

    calibrator = fit_calibration_bundle(
        method=args.calibration,
        calibration_df=calibration_rows_train,
        evaluation_df=calibration_rows_eval,
        fallback_note=calibration_fallback_note,
    )

    if aligned_cache_df is not None:
        per_view_df = build_per_view_scores_from_aligned_cache(
            aligned_df=evaluation_aligned_df,
            calibrator=calibrator,
        )
    else:
        per_view_df = build_per_view_scores(scene_df=evaluation_scene_df, calibrator=calibrator)
    coalition_df = build_coalition_rows(
        per_view_df=per_view_df,
        coalition_sizes=args.coalition_sizes,
        max_combinations_per_k=args.max_combinations_per_k,
        seed=args.seed,
    )
    if coalition_df.empty:
        raise SystemExit(
            "No coalition rows were generated. Check that the selected evaluation split has enough repeated synthetic views for the requested coalition sizes."
        )

    summary_df = summarize_methods(coalition_df)
    geometry_df = summarize_geometry(coalition_df)
    pair_geometry_df = summarize_pair_geometry_focus(coalition_df)

    save_dataframe(output_dir / "manifest_proxy.csv", manifest_df)
    save_dataframe(output_dir / "view_coverage_summary.csv", coverage_df)
    save_dataframe(output_dir / "per_view_scores.csv", per_view_df)
    save_dataframe(output_dir / "per_coalition_scores.csv", coalition_df)
    save_dataframe(output_dir / "fusion_summary.csv", summary_df)
    save_dataframe(output_dir / "geometry_group_summary.csv", geometry_df)
    save_dataframe(output_dir / "pair_geometry_focus_summary.csv", pair_geometry_df)

    calibration_summary = pd.DataFrame(
        [
            {
                "requested_method": args.calibration,
                "effective_method": calibrator.method,
                "note": calibrator.note,
                "fit_sample_count": calibrator.fit_sample_count,
                "train_brier": calibrator.train_brier,
                "train_ece": calibrator.train_ece,
                "eval_brier": calibrator.eval_brier,
                "eval_ece": calibrator.eval_ece,
                "eval_auc": calibrator.eval_auc,
                "eval_ap": calibrator.eval_ap,
            }
        ]
    )
    save_dataframe(output_dir / "calibration_summary.csv", calibration_summary)

    if calibrator.method != "none" and calibrator.model is not None:
        with (output_dir / "calibrator.pkl").open("wb") as handle:
            pickle.dump(calibrator.model, handle)

    plot_method_performance(summary_df=summary_df, output_dir=output_dir)
    plot_marginal_gain(summary_df=summary_df, output_dir=output_dir)
    plot_max_noisy_or_product(summary_df=summary_df, output_dir=output_dir)
    plot_geometry_focus(pair_geometry_df=pair_geometry_df, output_dir=output_dir)
    if calibration_rows_eval is not None and not calibration_rows_eval.empty:
        plot_reliability(
            calibration_rows=calibration_rows_eval,
            calibrator=calibrator,
            output_dir=output_dir,
            title_prefix=f"{args.evaluation_split} target-prediction",
        )

    readme_text = build_run_summary_markdown(
        args=args,
        output_dir=output_dir,
        manifest_df=manifest_df,
        coverage_df=coverage_df,
        per_view_df=per_view_df,
        coalition_df=coalition_df,
        summary_df=summary_df,
        pair_geometry_df=pair_geometry_df,
        calibrator=calibrator,
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    if summary_df.empty:
        print("No summary rows were produced.")
        return

    best_row = summary_df.iloc[summary_df.apply(headline_metric, axis=1).astype(float).argmax()]
    best_row = select_summary_row(summary_df, multi_view_only=True)
    if best_row is None:
        best_row = sort_summary_rows(summary_df).iloc[0]
    print("Synthetic Multi-Perspective Probability Fusion")
    print(f"Output directory: {output_dir}")
    print(
        "Coverage note: "
        f"{int(coverage_df['available_view_count'].min())}-{int(coverage_df['available_view_count'].max())} cached views per instance "
        f"(mean {float(coverage_df['available_view_count'].mean()):.2f}; expected {args.expected_view_count})."
    )
    print(
        "Best multi-view fusion summary: "
        f"{best_row['fusion_label']} at k={int(best_row['coalition_size'])} "
        f"(AP={float(best_row['average_precision']):.4f}, F1={float(best_row['f1']):.4f}, "
        f"mean score={float(best_row['mean_score']):.4f}, uplift={float(best_row['mean_delta_vs_best_single']):+.4f})."
    )
    print(
        "Thesis interpretation: "
        "Product fusion is the agreement model; max and noisy-OR are rescue models. "
        "If the rescue rules outperform product fusion, the multi-view benefit is more consistent with at-least-one-good-view recovery than with all-view agreement."
    )


if __name__ == "__main__":
    main()

