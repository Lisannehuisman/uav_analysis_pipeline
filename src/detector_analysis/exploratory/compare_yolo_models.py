from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


@dataclass
class AggregateMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


@dataclass
class OfficialValidation:
    aggregate: AggregateMetrics
    per_class_rows: list[dict[str, float | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two YOLO detection models on the same test split."
    )
    parser.add_argument("--model-a", required=True, help="Path to the first .pt model.")
    parser.add_argument("--model-b", required=True, help="Path to the second .pt model.")
    parser.add_argument(
        "--label-a",
        default="Model A",
        help="Display label for the first model in plots and tables.",
    )
    parser.add_argument(
        "--label-b",
        default="Model B",
        help="Display label for the second model in plots and tables.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to your YOLO dataset YAML file that defines the test split.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, 0, 0,1. Defaults to Ultralytics auto selection.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_comparison_output",
        help="Folder where CSVs and plots will be written.",
    )
    return parser.parse_args()


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    configured_root = data_dict.get("path")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data_dict = yaml.safe_load(handle)

    split_value = data_dict.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}.")

    root = resolve_dataset_root(data_yaml, data_dict)
    candidates: list[Path]
    if isinstance(split_value, str):
        candidates = [Path(split_value)]
    elif isinstance(split_value, list):
        candidates = [Path(item) for item in split_value]
    else:
        raise ValueError(f"Unsupported '{split}' entry in {data_yaml}: {split_value!r}")

    images: list[Path] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (root / candidate).resolve()
        if resolved.is_dir():
            for path in resolved.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
        elif resolved.is_file() and resolved.suffix.lower() == ".txt":
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw_path = line.strip()
                    if not raw_path:
                        continue
                    listed_path = Path(raw_path)
                    if not listed_path.is_absolute():
                        listed_path = (resolved.parent / listed_path).resolve()
                    if listed_path.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(listed_path)
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(resolved)
        else:
            raise FileNotFoundError(
                f"Could not resolve image path '{candidate}' from split '{split}'."
            )

    if not images:
        raise FileNotFoundError(f"No images found for split '{split}' in {data_yaml}.")
    return sorted(images)


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(
        f"Expected '{image_path}' to live under an 'images' directory so the label path can be derived."
    )


def load_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    boxes: list[list[float]] = []
    classes: list[int] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls_id = int(float(cls))
            xc = float(xc) * width
            yc = float(yc) * height
            w = float(w) * width
            h = float(h) * height
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    intersection = inter_w * inter_h

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0, a_max=None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], a_min=0, a_max=None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0, a_max=None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], a_min=0, a_max=None
    )
    union = area1[:, None] + area2[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def compute_ap(tp_flags: np.ndarray, confidences: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if tp_flags.size == 0:
        return 0.0

    order = np.argsort(-confidences)
    tp = tp_flags[order].astype(np.float32)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapezoid(mpre, mrec))


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, int, int, int]:
    if len(pred_boxes) == 0:
        return np.zeros((0,), dtype=np.float32), 0, 0, len(gt_boxes)

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_classes = pred_classes[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    tp_flags = np.zeros((len(pred_boxes),), dtype=np.float32)

    for pred_idx, pred_cls in enumerate(pred_classes):
        best_iou = -1.0
        best_gt = -1
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx in matched_gt or gt_cls != pred_cls:
                continue
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = float(iou)
                best_gt = gt_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp_flags[pred_idx] = 1.0

    tp = int(tp_flags.sum())
    fp = int(len(pred_boxes) - tp)
    fn = int(len(gt_boxes) - tp)
    return tp_flags, tp, fp, fn


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_single_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> dict[str, float]:
    tp_flags_50, tp_50, fp_50, fn_50 = match_detections(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, 0.5
    )
    precision = safe_divide(tp_50, tp_50 + fp_50)
    recall = safe_divide(tp_50, tp_50 + fn_50)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    ap_values: list[float] = []
    for threshold in IOU_THRESHOLDS:
        tp_flags, _, _, _ = match_detections(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, float(threshold)
        )
        ap_values.append(compute_ap(tp_flags, pred_scores, len(gt_boxes)))

    valid_ap_values = [value for value in ap_values if not math.isnan(value)]
    ap50 = ap_values[0] if not math.isnan(ap_values[0]) else float("nan")
    ap50_95 = float(np.mean(valid_ap_values)) if valid_ap_values else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50,
        "ap50_95": ap50_95,
        "tp": float(tp_50),
        "fp": float(fp_50),
        "fn": float(fn_50),
        "num_gt": float(len(gt_boxes)),
        "num_pred": float(len(pred_boxes)),
    }


def extract_aggregate_metrics(validation_result) -> AggregateMetrics:
    box_metrics = validation_result.box
    precision = float(getattr(box_metrics, "mp", 0.0))
    recall = float(getattr(box_metrics, "mr", 0.0))
    map50 = float(getattr(box_metrics, "map50", 0.0))
    map50_95 = float(getattr(box_metrics, "map", 0.0))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return AggregateMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map50_95=map50_95,
    )


def extract_per_class_rows(validation_result, model_label: str) -> list[dict[str, float | str]]:
    names = getattr(validation_result, "names", {}) or {}
    maps = getattr(validation_result.box, "maps", [])
    rows: list[dict[str, float | str]] = []
    for cls_id, cls_name in names.items():
        cls_index = int(cls_id)
        ap_value = float(maps[cls_index]) if cls_index < len(maps) else float("nan")
        rows.append(
            {
                "model": model_label,
                "class_id": cls_index,
                "class_name": str(cls_name),
                "ap50_95": ap_value,
            }
        )
    return rows


def run_official_validation(
    model_path: Path,
    model_label: str,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    batch: int,
    device: str | None,
) -> OfficialValidation:
    model = YOLO(str(model_path))
    result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )
    return OfficialValidation(
        aggregate=extract_aggregate_metrics(result),
        per_class_rows=extract_per_class_rows(result, model_label=model_label),
    )


def iter_chunks(values: list[Path], chunk_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def run_per_image_evaluation(
    model_path: Path,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    batch: int,
    device: str | None,
) -> list[dict[str, float | str]]:
    model = YOLO(str(model_path))
    rows: list[dict[str, float | str]] = []

    for batch_paths in iter_chunks(image_paths, batch):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            height, width = result.orig_shape
            gt_boxes, gt_classes = load_yolo_labels(
                label_path_from_image(image_path), width=width, height=height
            )

            if result.boxes is None or len(result.boxes) == 0:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
                pred_classes = np.zeros((0,), dtype=np.int32)
            else:
                pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                pred_scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                pred_classes = result.boxes.cls.cpu().numpy().astype(np.int32)

            metrics = evaluate_single_image(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_classes=pred_classes,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )
            row = {"image": str(image_path)}
            row.update(metrics)
            rows.append(row)

    return rows


def write_per_image_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "image",
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap50_95",
        "tp",
        "fp",
        "fn",
        "num_gt",
        "num_pred",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    output_path: Path,
    label_a: str,
    label_b: str,
    metrics_a: AggregateMetrics,
    metrics_b: AggregateMetrics,
) -> None:
    rows = [
        {
            "model": label_a,
            "precision": metrics_a.precision,
            "recall": metrics_a.recall,
            "f1": metrics_a.f1,
            "map50": metrics_a.map50,
            "map50_95": metrics_a.map50_95,
        },
        {
            "model": label_b,
            "precision": metrics_b.precision,
            "recall": metrics_b.recall,
            "f1": metrics_b.f1,
            "map50": metrics_b.map50,
            "map50_95": metrics_b.map50_95,
        },
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "precision", "recall", "f1", "map50", "map50_95"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "class_id", "class_name", "ap50_95"],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_aggregate_plot(
    output_path: Path,
    label_a: str,
    label_b: str,
    metrics_a: AggregateMetrics,
    metrics_b: AggregateMetrics,
) -> None:
    metric_names = ["precision", "recall", "f1", "map50_95"]
    values_a = [getattr(metrics_a, name) for name in metric_names]
    values_b = [getattr(metrics_b, name) for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, values_a, width, label=label_a, color="#1f77b4")
    ax.bar(x + width / 2, values_b, width, label=label_b, color="#ff7f0e")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Dataset-level YOLO validation metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "mAP50-95"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_per_class_plot(
    output_path: Path,
    label_a: str,
    label_b: str,
    rows_a: list[dict[str, float | str]],
    rows_b: list[dict[str, float | str]],
) -> None:
    class_names = [str(row["class_name"]) for row in sorted(rows_a, key=lambda row: int(row["class_id"]))]
    values_a = [float(row["ap50_95"]) for row in sorted(rows_a, key=lambda row: int(row["class_id"]))]

    rows_b_sorted = sorted(rows_b, key=lambda row: int(row["class_id"]))
    values_b_map = {str(row["class_name"]): float(row["ap50_95"]) for row in rows_b_sorted}
    values_b = [values_b_map.get(class_name, float("nan")) for class_name in class_names]

    x = np.arange(len(class_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 6))
    ax.bar(x - width / 2, values_a, width, label=label_a, color="#1f77b4")
    ax.bar(x + width / 2, values_b, width, label=label_b, color="#ff7f0e")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AP50-95")
    ax.set_title("Per-class AP50-95 comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_boxplots(
    output_path: Path,
    label_a: str,
    label_b: str,
    rows_a: list[dict[str, float | str]],
    rows_b: list[dict[str, float | str]],
) -> None:
    metrics = [
        ("precision", "Per-image Precision @ IoU=0.50"),
        ("recall", "Per-image Recall @ IoU=0.50"),
        ("f1", "Per-image F1 @ IoU=0.50"),
        ("ap50", "Per-image AP50"),
        ("ap50_95", "Per-image AP50-95"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (metric_name, title) in enumerate(metrics):
        data_a = [float(row[metric_name]) for row in rows_a if not math.isnan(float(row[metric_name]))]
        data_b = [float(row[metric_name]) for row in rows_b if not math.isnan(float(row[metric_name]))]
        axes[idx].boxplot(
            [data_a, data_b],
            labels=[label_a, label_b],
            patch_artist=True,
            boxprops={"facecolor": "#cfe2f3"},
            medianprops={"color": "#d62728", "linewidth": 2},
        )
        axes[idx].scatter(
            np.random.normal(1, 0.04, size=len(data_a)),
            data_a,
            s=10,
            alpha=0.22,
            color="#1f77b4",
        )
        axes[idx].scatter(
            np.random.normal(2, 0.04, size=len(data_b)),
            data_b,
            s=10,
            alpha=0.22,
            color="#ff7f0e",
        )
        axes[idx].set_title(title)
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Image-by-image performance distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(
    label_a: str,
    label_b: str,
    metrics_a: AggregateMetrics,
    metrics_b: AggregateMetrics,
    output_dir: Path,
) -> None:
    print("\nDataset-level comparison")
    print("-" * 72)
    print(
        f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP50':>10} {'mAP50-95':>12}"
    )
    for label, metrics in ((label_a, metrics_a), (label_b, metrics_b)):
        print(
            f"{label:<20} {metrics.precision:>10.4f} {metrics.recall:>10.4f} {metrics.f1:>10.4f} "
            f"{metrics.map50:>10.4f} {metrics.map50_95:>12.4f}"
        )
    print(f"\nSaved comparison files to: {output_dir}")


def main() -> None:
    args = parse_args()

    model_a = Path(args.model_a).resolve()
    model_b = Path(args.model_b).resolve()
    data_yaml = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = resolve_split_images(data_yaml, args.split)

    official_a = run_official_validation(
        model_path=model_a,
        model_label=args.label_a,
        data_yaml=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        device=args.device,
    )
    official_b = run_official_validation(
        model_path=model_b,
        model_label=args.label_b,
        data_yaml=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        device=args.device,
    )

    rows_a = run_per_image_evaluation(
        model_path=model_a,
        image_paths=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        device=args.device,
    )
    rows_b = run_per_image_evaluation(
        model_path=model_b,
        image_paths=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        device=args.device,
    )

    write_summary_csv(
        output_dir / "aggregate_metrics.csv",
        args.label_a,
        args.label_b,
        official_a.aggregate,
        official_b.aggregate,
    )
    write_per_class_csv(
        output_dir / "per_class_ap50_95.csv",
        official_a.per_class_rows + official_b.per_class_rows,
    )
    write_per_image_csv(output_dir / "per_image_metrics_model_a.csv", rows_a)
    write_per_image_csv(output_dir / "per_image_metrics_model_b.csv", rows_b)
    create_aggregate_plot(
        output_dir / "aggregate_metrics.png",
        args.label_a,
        args.label_b,
        official_a.aggregate,
        official_b.aggregate,
    )
    create_per_class_plot(
        output_dir / "per_class_ap50_95.png",
        args.label_a,
        args.label_b,
        official_a.per_class_rows,
        official_b.per_class_rows,
    )
    create_boxplots(output_dir / "per_image_boxplots.png", args.label_a, args.label_b, rows_a, rows_b)
    print_summary(args.label_a, args.label_b, official_a.aggregate, official_b.aggregate, output_dir)


if __name__ == "__main__":
    main()


# python compare_yolo_models.py `
# >>   --model-a "C:\DATA\airsim\thesis\results\yolov8l_m4\M4_clean_yolov8l_run1\weights\best.pt" `
# >>   --model-b "C:\DATA\airsim\thesis\results\yolov8l\yolov8l_results\S0_M4_yolov8l\weights\best.pt" `
# >>   --label-a "M4_clean" `
# >>   --label-b "S0_M4" `
# >>   --data "C:\DATA\airsim\thesis\captures\S0_20251219_164144\dataset\M4_fixed.yaml" `
# >>   --split test `
# >>   --imgsz 640 `
# >>   --batch 16 `
# >>   --output-dir "comparison_output"