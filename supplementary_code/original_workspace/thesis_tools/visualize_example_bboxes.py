from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_OUTPUT_DIR = Path("outputs") / "thesis_tools" / "example_bounding_boxes"
VALID_METRICS = {"precision", "recall", "f1", "ap50", "ap50_95"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready example images with ground-truth and optional predicted bounding boxes."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Per-image metrics CSV used to select example images.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to map class IDs to names.",
    )
    parser.add_argument(
        "--metric",
        default="ap50_95",
        choices=sorted(VALID_METRICS),
        help="Metric used to rank images for selection.",
    )
    parser.add_argument(
        "--top-k",
        default=6,
        type=int,
        help="Number of images to select.",
    )
    parser.add_argument(
        "--selection",
        choices=["best", "worst", "mixed"],
        default="mixed",
        help="How to select examples from the per-image CSV.",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated list of object classes to include.",
    )
    parser.add_argument(
        "--unique-classes",
        action="store_true",
        help="Prefer at most one selected image per object class.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional YOLO model path for prediction overlay.",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Prediction confidence threshold when --model-path is provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where annotated images and the summary grid will be written.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_class_map(path: Path) -> dict[int, str]:
    rows = read_csv_rows(path)
    class_map: dict[int, str] = {}
    for row in rows:
        class_id = row.get("class_id")
        class_name = row.get("class_name")
        if class_id not in (None, "") and class_name:
            class_map[int(class_id)] = class_name
    return class_map


def infer_object_class(image_path: str, known_names: list[str]) -> str | None:
    stem = Path(image_path).stem
    object_token_match = re.search(r"^S0-SM_([^-]+)-", stem, re.IGNORECASE)
    if not object_token_match:
        return None
    object_token = object_token_match.group(1).lower()
    for candidate in sorted(known_names, key=len, reverse=True):
        if object_token.startswith(candidate.lower()):
            return candidate
    fallback = re.match(r"([a-zA-Z]+)", object_token)
    return fallback.group(1).lower() if fallback else None


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Could not derive label path from image path: {image_path}")


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    if not label_path.exists():
        return []

    labels: list[tuple[int, list[float]]] = []
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
            labels.append((cls_id, [x1, y1, x2, y2]))
    return labels


def load_label_class_names(label_path: Path, class_map: dict[int, str]) -> list[str]:
    if not label_path.exists():
        return []

    label_names: list[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            label_names.append(class_map.get(cls_id, str(cls_id)))
    return label_names


def select_examples(
    rows: list[dict[str, str]],
    metric: str,
    top_k: int,
    selection: str,
    allowed_classes: set[str],
    class_names: list[str],
    class_map: dict[int, str],
) -> list[dict[str, str]]:
    filtered = build_ranked_pool(rows, metric, allowed_classes, class_names, class_map)
    if selection == "best":
        return filtered[:top_k]
    if selection == "worst":
        return list(reversed(filtered[-top_k:]))

    best_count = math.ceil(top_k / 2)
    worst_count = top_k - best_count
    chosen = filtered[:best_count]
    chosen.extend(list(reversed(filtered[-worst_count:])))
    return chosen


def build_ranked_pool(
    rows: list[dict[str, str]],
    metric: str,
    allowed_classes: set[str],
    class_names: list[str],
    class_map: dict[int, str],
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        object_class = infer_object_class(row["image"], class_names)
        if object_class is None:
            continue
        if allowed_classes and object_class not in allowed_classes:
            continue
        if row.get(metric) in (None, ""):
            continue
        label_names = load_label_class_names(label_path_from_image(Path(row["image"])), class_map)
        if label_names and object_class not in label_names:
            continue
        row = dict(row)
        row["object_class"] = object_class
        row["gt_classes"] = ",".join(label_names)
        filtered.append(row)

    filtered.sort(key=lambda row: float(row[metric]), reverse=True)
    return filtered


def enforce_unique_classes(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_classes: set[str] = set()
    for row in rows:
        object_class = str(row["object_class"])
        if object_class in seen_classes:
            continue
        selected.append(row)
        seen_classes.add(object_class)
        if len(selected) >= top_k:
            break
    return selected if selected else rows[:top_k]


def draw_boxes(
    image_path: Path,
    labels: list[tuple[int, list[float]]],
    class_map: dict[int, str],
    predictions: list[tuple[str, float, list[float]]] | None,
    title: str,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for cls_id, box in labels:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
        draw.text((x1 + 2, max(y1 - 14, 2)), f"GT: {class_map.get(cls_id, str(cls_id))}", fill=(0, 220, 0))

    if predictions:
        for class_name, confidence, box in predictions:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(220, 40, 40), width=2)
            draw.text((x1 + 2, min(y2 + 2, image.height - 14)), f"Pred: {class_name} {confidence:.2f}", fill=(220, 40, 40))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def maybe_predict(model_path: str, image_paths: list[Path], conf: float, class_map: dict[int, str]) -> dict[str, list[tuple[str, float, list[float]]]]:
    if not model_path:
        return {}

    config_dir = Path("Ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    from ultralytics import YOLO

    model = YOLO(model_path)
    predictions: dict[str, list[tuple[str, float, list[float]]]] = {}
    for image_path in image_paths:
        result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
        rows: list[tuple[str, float, list[float]]] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, confs, classes, strict=True):
                rows.append((class_map.get(cls_id, str(cls_id)), float(score), [float(v) for v in box]))
        predictions[str(image_path)] = rows
    return predictions


def make_grid(image_paths: list[Path], output_path: Path, title: str) -> None:
    ncols = 2
    nrows = math.ceil(len(image_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for idx, image_path in enumerate(image_paths):
        ax = axes[idx]
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(image_path.stem)
        ax.axis("off")

    for idx in range(len(image_paths), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(Path(args.per_image).resolve())
    class_map = load_class_map(Path(args.per_class_csv).resolve())
    class_names = list(class_map.values())
    allowed_classes = {name.strip() for name in args.classes.split(",") if name.strip()}

    ranked_pool = build_ranked_pool(rows, args.metric, allowed_classes, class_names, class_map)
    selected = select_examples(rows, args.metric, args.top_k, args.selection, allowed_classes, class_names, class_map)
    if args.unique_classes:
        selected = enforce_unique_classes(selected, args.top_k)
        seen_images = {str(row["image"]) for row in selected}
        seen_classes = {str(row["object_class"]) for row in selected}
        for row in ranked_pool:
            image_key = str(row["image"])
            object_class = str(row["object_class"])
            if image_key in seen_images or object_class in seen_classes:
                continue
            selected.append(row)
            seen_images.add(image_key)
            seen_classes.add(object_class)
            if len(selected) >= args.top_k:
                break
        if len(selected) < args.top_k:
            for row in ranked_pool:
                image_key = str(row["image"])
                if image_key in seen_images:
                    continue
                selected.append(row)
                seen_images.add(image_key)
                if len(selected) >= args.top_k:
                    break
    image_paths = [Path(row["image"]) for row in selected]
    predictions = maybe_predict(args.model_path, image_paths, args.conf, class_map)

    rendered_paths: list[Path] = []
    selection_rows: list[dict[str, object]] = []
    for idx, row in enumerate(selected, start=1):
        image_path = Path(row["image"])
        image = Image.open(image_path)
        labels = load_yolo_labels(label_path_from_image(image_path), image.width, image.height)
        output_path = output_dir / f"{idx:02d}_{image_path.stem}.png"
        draw_boxes(
            image_path=image_path,
            labels=labels,
            class_map=class_map,
            predictions=predictions.get(str(image_path)),
            title=f"{row['object_class']} | {args.metric}={float(row[args.metric]):.3f}",
            output_path=output_path,
        )
        rendered_paths.append(output_path)
        selection_rows.append(
            {
                "rank": idx,
                "object_class": row["object_class"],
                "image": str(image_path),
                args.metric: float(row[args.metric]),
                "n_gt_boxes": len(labels),
                "gt_classes": row.get("gt_classes", ""),
                "prediction_overlay": bool(args.model_path),
            }
        )

    with (output_dir / "selected_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)

    make_grid(
        rendered_paths,
        output_dir / "selected_examples_grid.png",
        title="Thesis example images with bounding boxes",
    )
    print(f"Saved example bounding box visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
