from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a CVAT XML export against the staged review-set images and optionally write a corrected copy."
    )
    parser.add_argument("--xml", required=True, type=Path, help="Path to the CVAT annotations XML.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory containing the referenced images.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest.csv from the staged review set so filename-implied classes can be checked.",
    )
    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
        help="Path to write a JSON verification report.",
    )
    parser.add_argument(
        "--write-corrected-xml",
        type=Path,
        help="Optional path to write a corrected XML copy.",
    )
    return parser.parse_args()


def load_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {Path(row["staged_image"]).name: row["suggested_class_name"] for row in reader}


def main() -> None:
    args = parse_args()
    root = ET.parse(args.xml).getroot()
    images = root.findall("image")
    image_files = {path.name: path for path in args.images_dir.iterdir() if path.is_file()}
    manifest = load_manifest(args.manifest)

    missing_images: list[str] = []
    dimension_mismatches: list[dict[str, object]] = []
    suspicious_images: list[dict[str, object]] = []
    fixed_images: list[str] = []
    label_counter: Counter[str] = Counter()

    for image_node in images:
        image_name = image_node.attrib["name"]
        xml_width = int(image_node.attrib["width"])
        xml_height = int(image_node.attrib["height"])
        image_path = image_files.get(image_name)

        if image_path is None:
            missing_images.append(image_name)
            continue

        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (xml_width, xml_height):
            dimension_mismatches.append(
                {
                    "image": image_name,
                    "xml_size": [xml_width, xml_height],
                    "actual_size": [actual_width, actual_height],
                }
            )

        boxes = image_node.findall("box")
        labels = [box.attrib["label"] for box in boxes]
        label_counter.update(labels)
        suggested_class = manifest.get(image_name)

        if suggested_class:
            unique_labels = sorted(set(labels))
            if labels and any(label != suggested_class for label in labels):
                suspicious_images.append(
                    {
                        "image": image_name,
                        "suggested_class": suggested_class,
                        "labels_present": unique_labels,
                        "box_count": len(labels),
                    }
                )

            # Fix the confirmed container/barrel mixup only when every box on the image is barrel.
            if suggested_class == "container" and labels and set(labels) == {"barrel"}:
                for box in boxes:
                    box.set("label", "container")
                fixed_images.append(image_name)

    report = {
        "xml_image_count": len(images),
        "image_dir_count": len(image_files),
        "missing_images": missing_images,
        "dimension_mismatches": dimension_mismatches,
        "label_counts_before_fix": dict(sorted(label_counter.items())),
        "suspicious_images": suspicious_images,
        "fixed_images": fixed_images,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.write_corrected_xml is not None:
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(args.write_corrected_xml, encoding="utf-8", xml_declaration=True)

    print(f"Verified {len(images)} XML image entries against '{args.images_dir}'.")
    print(f"Missing images: {len(missing_images)}")
    print(f"Dimension mismatches: {len(dimension_mismatches)}")
    print(f"Suspicious images: {len(suspicious_images)}")
    print(f"Fixed confirmed container mixups: {len(fixed_images)}")
    print(f"Report written to '{args.output_report}'.")
    if args.write_corrected_xml is not None:
        print(f"Corrected XML written to '{args.write_corrected_xml}'.")


if __name__ == "__main__":
    main()
