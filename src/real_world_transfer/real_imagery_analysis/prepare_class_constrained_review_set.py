from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALIAS_TO_CANONICAL = {
    "barrel": "barrel",
    "container": "container",
    "containter": "container",
    "male": "male",
    "man": "male",
    "person": "male",
    "rock": "rock",
    "suv": "suv",
    "tank": "tank",
    "tent": "tent",
    "tower": "tower",
    "tree": "tree",
    "whitevan": "whitevan",
    "white_van": "whitevan",
    "white-van": "whitevan",
    "whitetruck": "whitevan",
    "white_truck": "whitevan",
    "white-truck": "whitevan",
}


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    staged_stem: str
    suggested_class_name: str | None
    suggested_class_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a folder of real-world images into a YOLO review set that keeps the "
            "existing synthetic ontology fixed."
        )
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Folder with raw images to review.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Workspace-local directory where the staged review set will be created.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("my_data.yaml"),
        help="YOLO data YAML that defines the canonical class names.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to stage images into the review set.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional YOLO weights path used to draft auto-labels.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Minimum confidence to keep draft detections when --model-path is used.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for draft auto-labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output directory before rebuilding it.",
    )
    parser.add_argument(
        "--restrict-to-suggested-class",
        action="store_true",
        help=(
            "When draft auto-labels are enabled, keep only detections whose class matches the "
            "class inferred from the filename."
        ),
    )
    return parser.parse_args()


def parse_names_from_data_yaml(path: Path) -> dict[int, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names: dict[int, str] = {}
    in_names = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not raw_line.startswith("  "):
            break
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key_text, value_text = stripped.split(":", 1)
        key = int(key_text.strip())
        value = value_text.strip().strip("'\"")
        names[key] = value
    if not names:
        raise ValueError(f"Could not parse class names from '{path}'.")
    return names


def normalize_token(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def infer_suggested_class(stem: str, class_name_to_id: dict[str, int]) -> tuple[str | None, int | None]:
    prefix = stem.split("(")[0].strip()
    normalized = normalize_token(prefix)
    canonical = ALIAS_TO_CANONICAL.get(normalized)
    if canonical is None:
        return None, None
    class_id = class_name_to_id.get(canonical)
    if class_id is None:
        return canonical, None
    return canonical, class_id


def build_staged_stem(index: int, source_path: Path) -> str:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in source_path.stem).strip("_")
    safe_name = safe_name or "image"
    return f"{index:04d}_{safe_name.lower()}"


def iter_image_paths(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def stage_image(source_path: Path, target_path: Path, copy_mode: str) -> None:
    if copy_mode == "hardlink":
        try:
            target_path.hardlink_to(source_path)
            return
        except OSError:
            pass
    shutil.copy2(source_path, target_path)


def write_output_yaml(output_dir: Path, names: dict[int, str]) -> None:
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images",
        "val: images",
        "test: images",
        "names:",
    ]
    for class_id, class_name in sorted(names.items()):
        lines.append(f"  {class_id}: {class_name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, records: list[ImageRecord]) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "staged_image",
                "staged_label",
                "suggested_class_name",
                "suggested_class_id",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": str(record.source_path),
                    "staged_image": str((output_dir / "images" / f"{record.staged_stem}{record.source_path.suffix.lower()}").resolve()),
                    "staged_label": str((output_dir / "labels" / f"{record.staged_stem}.txt").resolve()),
                    "suggested_class_name": record.suggested_class_name or "",
                    "suggested_class_id": "" if record.suggested_class_id is None else record.suggested_class_id,
                }
            )


def write_empty_labels(output_dir: Path, records: list[ImageRecord]) -> None:
    for record in records:
        (output_dir / "labels" / f"{record.staged_stem}.txt").write_text("", encoding="utf-8")


def write_draft_labels(
    output_dir: Path,
    records: list[ImageRecord],
    model_path: Path,
    conf_threshold: float,
    imgsz: int,
    restrict_to_suggested_class: bool,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the current environment, so draft auto-labels "
            "cannot be generated."
        ) from exc

    image_paths = [
        output_dir / "images" / f"{record.staged_stem}{record.source_path.suffix.lower()}"
        for record in records
    ]
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf_threshold,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )

    draft_csv_path = output_dir / "draft_predictions.csv"
    with draft_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "staged_image",
                "class_id",
                "class_name",
                "x_center",
                "y_center",
                "width",
                "height",
                "confidence",
            ],
        )
        writer.writeheader()

        for record, image_path, result in zip(records, image_paths, results):
            label_path = output_dir / "labels" / f"{record.staged_stem}.txt"
            lines: list[str] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xywhn = result.boxes.xywhn.tolist()
                class_ids = result.boxes.cls.tolist()
                confidences = result.boxes.conf.tolist()
                for box, class_id_float, confidence in zip(xywhn, class_ids, confidences):
                    class_id = int(class_id_float)
                    if restrict_to_suggested_class and record.suggested_class_id is not None:
                        if class_id != record.suggested_class_id:
                            continue
                    x_center, y_center, width, height = box
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                    writer.writerow(
                        {
                            "staged_image": str(image_path.resolve()),
                            "class_id": class_id,
                            "class_name": result.names.get(class_id, str(class_id)),
                            "x_center": f"{x_center:.6f}",
                            "y_center": f"{y_center:.6f}",
                            "width": f"{width:.6f}",
                            "height": f"{height:.6f}",
                            "confidence": f"{confidence:.6f}",
                        }
                    )
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_yaml = args.data_yaml.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: '{source_dir}'")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: '{source_dir}'")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find data YAML: '{data_yaml}'")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: '{output_dir}'. Re-run with --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    names = parse_names_from_data_yaml(data_yaml)
    class_name_to_id = {class_name: class_id for class_id, class_name in names.items()}

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    for index, source_path in enumerate(iter_image_paths(source_dir), start=1):
        suggested_name, suggested_id = infer_suggested_class(source_path.stem, class_name_to_id)
        staged_stem = build_staged_stem(index=index, source_path=source_path)
        staged_image_path = images_dir / f"{staged_stem}{source_path.suffix.lower()}"
        stage_image(source_path=source_path, target_path=staged_image_path, copy_mode=args.copy_mode)
        records.append(
            ImageRecord(
                source_path=source_path,
                staged_stem=staged_stem,
                suggested_class_name=suggested_name,
                suggested_class_id=suggested_id,
            )
        )

    if not records:
        raise RuntimeError(f"No supported image files found in '{source_dir}'.")

    write_output_yaml(output_dir=output_dir, names=names)
    write_manifest(output_dir=output_dir, records=records)

    if args.model_path is not None:
        write_draft_labels(
            output_dir=output_dir,
            records=records,
            model_path=args.model_path.resolve(),
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            restrict_to_suggested_class=args.restrict_to_suggested_class,
        )
        mode = "draft auto-labels"
    else:
        write_empty_labels(output_dir=output_dir, records=records)
        mode = "empty label files"

    print(f"Staged {len(records)} images into '{output_dir}'.")
    print(f"Prepared {mode} under '{labels_dir}'.")
    print(f"Review manifest written to '{output_dir / 'manifest.csv'}'.")


if __name__ == "__main__":
    main()
