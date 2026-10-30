from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results" / "intermediate" / "real_uav_prediction_labels"
GT_LABEL_DIR = ROOT / "data_collection" / "raw_data" / "self_collected_uav_validation" / "labels"

GT_CLASS_NAMES = [
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

# This is the class order visible in the saved YOLO confusion matrices for
# these two runs. The root data.yaml uses the same class names in a different
# order, so ground truth must be translated before comparing predictions.
EVAL_CLASS_NAMES = [
    "tent",
    "tank",
    "tower",
    "container",
    "whitevan",
    "suv",
    "male",
    "rock",
    "barrel",
    "tree",
]

GT_TO_EVAL_CLASS = {
    gt_idx: EVAL_CLASS_NAMES.index(name)
    for gt_idx, name in enumerate(GT_CLASS_NAMES)
}

RUNS = {
    "synthetic_only": RUN_ROOT / "real_uav_synthetic_only_eval" / "labels",
    "finetuned": RUN_ROOT / "real_uav_finetuned_test_eval" / "labels",
}

STABLE_SUPPORT_MIN = 10


@dataclass(frozen=True)
class Box:
    image_key: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


def normalize_stem(stem: str) -> str:
    base = stem.split("_jpg.rf.", 1)[0]
    return base.replace(" ", "-")


def read_ground_truth() -> dict[str, list[Box]]:
    by_key: dict[str, list[Box]] = defaultdict(list)
    for path in GT_LABEL_DIR.glob("*.txt"):
        key = normalize_stem(path.stem)
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = GT_TO_EVAL_CLASS[int(float(parts[0]))]
            coords = [float(v) for v in parts[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
            else:
                xs = coords[0::2]
                ys = coords[1::2]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
            by_key[key].append(Box(key, cls, (x1, y1, x2, y2)))
    return by_key


def read_predictions(label_dir: Path) -> dict[str, list[Box]]:
    by_key: dict[str, list[Box]] = defaultdict(list)
    for path in label_dir.glob("*.txt"):
        key = normalize_stem(path.stem)
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h, conf = [float(v) for v in parts[1:6]]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            by_key[key].append(Box(key, cls, (x1, y1, x2, y2), conf))
    return by_key


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def match_counts(
    gt_by_key: dict[str, list[Box]],
    pred_by_key: dict[str, list[Box]],
    image_keys: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[int, dict[str, int]]:
    counts = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in range(len(EVAL_CLASS_NAMES))}

    for key in image_keys:
        gt_boxes = gt_by_key.get(key, [])
        pred_boxes = [p for p in pred_by_key.get(key, []) if p.conf >= conf_threshold]
        for cls in range(len(EVAL_CLASS_NAMES)):
            gt_cls = [g for g in gt_boxes if g.cls == cls]
            pred_cls = sorted((p for p in pred_boxes if p.cls == cls), key=lambda p: p.conf, reverse=True)
            matched_gt: set[int] = set()
            for pred in pred_cls:
                best_idx = -1
                best_iou = 0.0
                for idx, gt in enumerate(gt_cls):
                    if idx in matched_gt:
                        continue
                    overlap = iou(pred.xyxy, gt.xyxy)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    counts[cls]["tp"] += 1
                    matched_gt.add(best_idx)
                else:
                    counts[cls]["fp"] += 1
            counts[cls]["fn"] += len(gt_cls) - len(matched_gt)
    return counts


def average_precision(
    gt_by_key: dict[str, list[Box]],
    pred_by_key: dict[str, list[Box]],
    image_keys: list[str],
    cls: int,
    iou_threshold: float = 0.50,
) -> float | None:
    gt_count = sum(1 for key in image_keys for g in gt_by_key.get(key, []) if g.cls == cls)
    if gt_count == 0:
        return None

    predictions = sorted(
        [p for key in image_keys for p in pred_by_key.get(key, []) if p.cls == cls],
        key=lambda p: p.conf,
        reverse=True,
    )
    matched: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []

    for pred in predictions:
        gt_cls = [g for g in gt_by_key.get(pred.image_key, []) if g.cls == cls]
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_cls):
            if idx in matched[pred.image_key]:
                continue
            overlap = iou(pred.xyxy, gt.xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched[pred.image_key].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not predictions:
        return 0.0

    cum_tp = []
    cum_fp = []
    running_tp = 0
    running_fp = 0
    for t, f in zip(tp, fp):
        running_tp += t
        running_fp += f
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    recalls = [t / gt_count for t in cum_tp]
    precisions = [t / max(t + f, 1) for t, f in zip(cum_tp, cum_fp)]

    # COCO/VOC-style interpolated precision envelope.
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    return precision, recall, f1


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.3f}"


def main() -> None:
    out_dir = ROOT / "results" / "recomputed" / "real_uav_finetune"
    out_dir.mkdir(exist_ok=True)

    gt_by_key = read_ground_truth()
    pred_by_run = {name: read_predictions(path) for name, path in RUNS.items()}
    image_keys = sorted(set().union(*(set(preds) for preds in pred_by_run.values())))

    missing_gt = [key for key in image_keys if key not in gt_by_key]
    if missing_gt:
        raise SystemExit(f"Missing ground truth for {len(missing_gt)} images: {missing_gt[:5]}")

    gt_support = Counter(g.cls for key in image_keys for g in gt_by_key.get(key, []))
    supported_classes = sorted(gt_support)

    summary_rows = []
    stable_rows = []
    per_class_rows = []
    conf_threshold = 0.25
    iou_threshold = 0.50

    for run_name, preds in pred_by_run.items():
        counts = match_counts(gt_by_key, preds, image_keys, conf_threshold, iou_threshold)
        total_tp = sum(counts[cls]["tp"] for cls in supported_classes)
        total_fp = sum(counts[cls]["fp"] for cls in supported_classes)
        total_fn = sum(counts[cls]["fn"] for cls in supported_classes)
        precision, recall, f1 = prf(total_tp, total_fp, total_fn)
        ap_values = {
            cls: average_precision(gt_by_key, preds, image_keys, cls, iou_threshold)
            for cls in supported_classes
        }
        map50 = sum(v for v in ap_values.values() if v is not None) / len(ap_values)
        pred_count_025 = sum(1 for key in image_keys for p in preds.get(key, []) if p.conf >= conf_threshold)
        pred_count_all = sum(len(preds.get(key, [])) for key in image_keys)

        summary_rows.append(
            {
                "run": run_name,
                "images": len(image_keys),
                "gt_objects_supported_classes": sum(gt_support.values()),
                "predictions_conf_ge_0.25": pred_count_025,
                "predictions_all_confidences": pred_count_all,
                "tp_at_0.25_iou50": total_tp,
                "fp_at_0.25_iou50": total_fp,
                "fn_at_0.25_iou50": total_fn,
                "precision_at_0.25_iou50": precision,
                "recall_at_0.25_iou50": recall,
                "f1_at_0.25_iou50": f1,
                "mAP50_supported_classes": map50,
            }
        )

        stable_classes = [cls for cls in supported_classes if gt_support[cls] >= STABLE_SUPPORT_MIN]
        stable_tp = sum(counts[cls]["tp"] for cls in stable_classes)
        stable_fp = sum(counts[cls]["fp"] for cls in stable_classes)
        stable_fn = sum(counts[cls]["fn"] for cls in stable_classes)
        stable_precision, stable_recall, stable_f1 = prf(stable_tp, stable_fp, stable_fn)
        stable_map50 = sum(ap_values[cls] for cls in stable_classes if ap_values[cls] is not None) / len(stable_classes)
        stable_rows.append(
            {
                "run": run_name,
                "classes": ",".join(EVAL_CLASS_NAMES[cls] for cls in stable_classes),
                "gt_objects": sum(gt_support[cls] for cls in stable_classes),
                "tp_at_0.25_iou50": stable_tp,
                "fp_at_0.25_iou50": stable_fp,
                "fn_at_0.25_iou50": stable_fn,
                "precision_at_0.25_iou50": stable_precision,
                "recall_at_0.25_iou50": stable_recall,
                "f1_at_0.25_iou50": stable_f1,
                "mAP50": stable_map50,
            }
        )

        for cls in range(len(EVAL_CLASS_NAMES)):
            tp = counts[cls]["tp"]
            fp = counts[cls]["fp"]
            fn = counts[cls]["fn"]
            precision_c, recall_c, f1_c = prf(tp, fp, fn)
            per_class_rows.append(
                {
                    "run": run_name,
                    "class_id": cls,
                    "class_name": EVAL_CLASS_NAMES[cls],
                    "gt_support": gt_support.get(cls, 0),
                    "evaluated": cls in supported_classes,
                    "tp_at_0.25_iou50": tp,
                    "fp_at_0.25_iou50": fp,
                    "fn_at_0.25_iou50": fn,
                    "precision_at_0.25_iou50": precision_c,
                    "recall_at_0.25_iou50": recall_c,
                    "f1_at_0.25_iou50": f1_c,
                    "ap50": ap_values.get(cls),
                }
            )

    with (out_dir / "supported_class_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_dir / "stable_class_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stable_rows[0]))
        writer.writeheader()
        writer.writerows(stable_rows)

    with (out_dir / "per_class_supported_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    class_support_rows = [
        {"class_id": cls, "class_name": EVAL_CLASS_NAMES[cls], "gt_support": gt_support.get(cls, 0)}
        for cls in range(len(EVAL_CLASS_NAMES))
    ]
    with (out_dir / "class_support.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_support_rows[0]))
        writer.writeheader()
        writer.writerows(class_support_rows)

    report = [
        "# Real UAV Finetune Evaluation",
        "",
        f"Images evaluated: {len(image_keys)}",
        f"Supported classes in this 39-image subset: {', '.join(EVAL_CLASS_NAMES[c] for c in supported_classes)}",
        f"Stable classes with at least {STABLE_SUPPORT_MIN} objects: {', '.join(row['classes'] for row in stable_rows[:1])}",
        "",
        "The class order used by these runs is the synthetic/review-set order, not the copied validation-set YAML order.",
        "Classes with zero ground-truth support should be reported as not applicable, not as precision=1.",
        "",
        "## Supported-Class Summary",
        "",
        "| run | P@0.25/IoU50 | R@0.25/IoU50 | F1@0.25/IoU50 | mAP50 | TP | FP | FN | preds >=0.25 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report.append(
            "| {run} | {p} | {r} | {f1} | {map50} | {tp} | {fp} | {fn} | {preds} |".format(
                run=row["run"],
                p=format_float(row["precision_at_0.25_iou50"]),
                r=format_float(row["recall_at_0.25_iou50"]),
                f1=format_float(row["f1_at_0.25_iou50"]),
                map50=format_float(row["mAP50_supported_classes"]),
                tp=row["tp_at_0.25_iou50"],
                fp=row["fp_at_0.25_iou50"],
                fn=row["fn_at_0.25_iou50"],
                preds=row["predictions_conf_ge_0.25"],
            )
        )
    report.extend(
        [
            "",
            f"## Stable-Class Summary (support >= {STABLE_SUPPORT_MIN})",
            "",
            "| run | classes | P@0.25/IoU50 | R@0.25/IoU50 | F1@0.25/IoU50 | mAP50 | TP | FP | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stable_rows:
        report.append(
            "| {run} | {classes} | {p} | {r} | {f1} | {map50} | {tp} | {fp} | {fn} |".format(
                run=row["run"],
                classes=row["classes"],
                p=format_float(row["precision_at_0.25_iou50"]),
                r=format_float(row["recall_at_0.25_iou50"]),
                f1=format_float(row["f1_at_0.25_iou50"]),
                map50=format_float(row["mAP50"]),
                tp=row["tp_at_0.25_iou50"],
                fp=row["fp_at_0.25_iou50"],
                fn=row["fn_at_0.25_iou50"],
            )
        )
    report.extend(["", "## Class Support", ""])
    for cls in range(len(EVAL_CLASS_NAMES)):
        report.append(f"- {EVAL_CLASS_NAMES[cls]}: {gt_support.get(cls, 0)}")

    report.extend(["", "## Per-Class Metrics", ""])
    report.append("| run | class | support | P | R | F1 | AP50 | TP | FP | FN |")
    report.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_class_rows:
        if not row["evaluated"]:
            continue
        report.append(
            "| {run} | {cls} | {support} | {p} | {r} | {f1} | {ap} | {tp} | {fp} | {fn} |".format(
                run=row["run"],
                cls=row["class_name"],
                support=row["gt_support"],
                p=format_float(row["precision_at_0.25_iou50"]),
                r=format_float(row["recall_at_0.25_iou50"]),
                f1=format_float(row["f1_at_0.25_iou50"]),
                ap=format_float(row["ap50"]),
                tp=row["tp_at_0.25_iou50"],
                fp=row["fp_at_0.25_iou50"],
                fn=row["fn_at_0.25_iou50"],
            )
        )

    (out_dir / "real_uav_finetune_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary_rows, indent=2))
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
