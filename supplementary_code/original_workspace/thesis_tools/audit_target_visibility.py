from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("outputs") / "thesis_tools" / "target_visibility_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the filename target object is actually present in the ground-truth labels."
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Per-image metrics CSV used to enumerate test images.",
    )
    parser.add_argument(
        "--per-class-csv",
        default="comparison_output/per_class_ap50_95.csv",
        help="Per-class CSV used to map class IDs to names.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the audit CSVs and summary.",
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


def load_label_class_names(label_path: Path, class_map: dict[int, str]) -> list[str]:
    if not label_path.exists():
        return []
    names: list[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            names.append(class_map.get(cls_id, str(cls_id)))
    return names


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(Path(args.per_image).resolve())
    class_map = load_class_map(Path(args.per_class_csv).resolve())
    class_names = list(class_map.values())

    mismatch_rows: list[dict[str, object]] = []
    per_object_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        image_path = Path(row["image"])
        target_class = infer_object_class(str(image_path), class_names)
        if target_class is None:
            continue
        gt_classes = load_label_class_names(label_path_from_image(image_path), class_map)
        stats = per_object_counts.setdefault(target_class, {"total_images": 0, "target_present": 0, "target_absent": 0})
        stats["total_images"] += 1
        if target_class in gt_classes:
            stats["target_present"] += 1
        else:
            stats["target_absent"] += 1
            mismatch_rows.append(
                {
                    "image": str(image_path),
                    "target_class_from_filename": target_class,
                    "gt_classes": ",".join(gt_classes),
                    "ap50_95": row.get("ap50_95", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "f1": row.get("f1", ""),
                }
            )

    summary_rows: list[dict[str, object]] = []
    total_images = 0
    total_absent = 0
    for object_class in sorted(per_object_counts):
        stats = per_object_counts[object_class]
        absent = int(stats["target_absent"])
        total = int(stats["total_images"])
        total_images += total
        total_absent += absent
        summary_rows.append(
            {
                "object_class": object_class,
                "total_images": total,
                "target_present": int(stats["target_present"]),
                "target_absent": absent,
                "absent_fraction": absent / total if total else 0.0,
            }
        )

    write_csv(output_dir / "target_visibility_summary.csv", summary_rows)
    write_csv(output_dir / "target_visibility_mismatches.csv", mismatch_rows)

    summary_text = "\n".join(
        [
            "# Target visibility audit",
            "",
            f"- Total images checked: {total_images}",
            f"- Images where filename target is absent from GT labels: {total_absent}",
            f"- Overall absent fraction: {total_absent / total_images:.4f}" if total_images else "- Overall absent fraction: n/a",
            "",
            "Interpretation:",
            "- These images should not be treated as clean positive evidence for the filename target object.",
            "- For viewpoint analysis, they are better interpreted as target-absent or fully occluded target-view samples.",
            "- The detector-family benchmark itself is unaffected, because that evaluation uses the actual GT labels.",
        ]
    )
    (output_dir / "target_visibility_summary.md").write_text(summary_text, encoding="utf-8")

    print(f"Saved target visibility audit to: {output_dir}")


if __name__ == "__main__":
    main()
