from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from multiview_transformer.common import (
    class_name_lookup,
    compute_target_box_stats,
    infer_label_path,
    load_image_size,
    load_yolo_labels,
    parse_scene_view_metadata,
    read_csv_rows,
    read_yaml,
    resolve_split_images,
    viewpoint_sort_key,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a scene-level manifest for the multiview transformer baseline.",
    )
    parser.add_argument("--data-yaml", required=True, help="YOLO dataset YAML for the base M4 dataset.")
    parser.add_argument(
        "--output-csv",
        default="outputs/multiview_transformer/manifests/m4_scene_manifest.csv",
        help="Where to write the combined scene manifest.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to include in the manifest.",
    )
    parser.add_argument(
        "--quality-csv",
        default="",
        help="Optional CSV with extra supervision columns keyed by file name.",
    )
    parser.add_argument(
        "--quality-key",
        default="file_name",
        help="Column in the quality CSV used to join rows onto the manifest.",
    )
    parser.add_argument(
        "--quality-columns",
        default="",
        help="Optional comma-separated list of quality columns to merge. Defaults to every non-key column.",
    )
    parser.add_argument(
        "--skip-image-size",
        action="store_true",
        help="Skip reading image dimensions if you only need a lightweight manifest.",
    )
    return parser.parse_args()


def load_quality_lookup(
    quality_csv: str,
    quality_key: str,
    quality_columns_arg: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not quality_csv:
        return {}, []
    quality_rows = read_csv_rows(Path(quality_csv))
    if not quality_rows:
        return {}, []
    if quality_columns_arg.strip():
        quality_columns = [column.strip() for column in quality_columns_arg.split(",") if column.strip()]
    else:
        quality_columns = [column for column in quality_rows[0] if column != quality_key]

    lookup: dict[str, dict[str, str]] = {}
    for row in quality_rows:
        lookup[str(row[quality_key])] = {column: str(row.get(column, "")) for column in quality_columns}
    return lookup, quality_columns


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    split_counts = Counter(str(row["split"]) for row in rows)
    scene_counts: dict[str, set[str]] = defaultdict(set)
    viewpoint_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_name = str(row["split"])
        scene_counts[split_name].add(str(row["scene_key"]))
        viewpoint_counts[split_name].add(str(row["viewpoint"]))

    lines = [
        "# Multiview manifest summary",
        "",
        f"- Total rows: {len(rows)}",
    ]
    for split_name in sorted(split_counts):
        lines.extend(
            [
                f"- {split_name}: {split_counts[split_name]} images",
                f"- {split_name}: {len(scene_counts[split_name])} scenes",
                f"- {split_name}: {len(viewpoint_counts[split_name])} unique absolute viewpoints",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data_yaml).resolve()
    data_dict = read_yaml(data_yaml)
    _, name_to_id = class_name_lookup(data_dict)
    known_class_names = list(name_to_id.keys())
    quality_lookup, quality_columns = load_quality_lookup(
        quality_csv=args.quality_csv,
        quality_key=args.quality_key,
        quality_columns_arg=args.quality_columns,
    )

    rows: list[dict[str, object]] = []
    for split_name in args.splits:
        for image_path in resolve_split_images(data_yaml, split_name):
            file_name = image_path.name
            scene_key, viewpoint, elevation, radius, azimuth, target_class = parse_scene_view_metadata(
                file_name=file_name,
                known_class_names=known_class_names,
            )
            target_class_id = name_to_id[target_class.lower()]
            label_path = infer_label_path(image_path)
            labels = load_yolo_labels(label_path)
            (
                num_objects_total,
                num_target_boxes,
                target_visible,
                target_max_area_norm,
                target_total_area_norm,
            ) = compute_target_box_stats(labels=labels, target_class_id=target_class_id)
            if args.skip_image_size:
                image_width, image_height = 0, 0
            else:
                image_width, image_height = load_image_size(image_path)

            row: dict[str, object] = {
                "split": split_name,
                "scene_key": scene_key,
                "file_name": file_name,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "target_class": target_class,
                "target_class_id": target_class_id,
                "viewpoint": viewpoint,
                "elevation": elevation,
                "radius": radius,
                "azimuth": azimuth,
                "elevation_id": {"low": 0, "mid": 1, "high": 2}[elevation],
                "radius_id": {"near": 0, "mid": 1, "far": 2}[radius],
                "azimuth_sin": round(math.sin(math.radians(azimuth)), 8),
                "azimuth_cos": round(math.cos(math.radians(azimuth)), 8),
                "image_width": image_width,
                "image_height": image_height,
                "num_objects_total": num_objects_total,
                "num_target_boxes": num_target_boxes,
                "target_visible": target_visible,
                "target_max_area_norm": round(target_max_area_norm, 8),
                "target_total_area_norm": round(target_total_area_norm, 8),
            }
            if file_name in quality_lookup:
                for column in quality_columns:
                    row[column] = quality_lookup[file_name][column]
            rows.append(row)

    rows.sort(key=lambda row: (str(row["split"]), str(row["scene_key"]), viewpoint_sort_key(str(row["viewpoint"]))))
    output_csv = Path(args.output_csv).resolve()

    base_fieldnames = [
        "split",
        "scene_key",
        "file_name",
        "image_path",
        "label_path",
        "target_class",
        "target_class_id",
        "viewpoint",
        "elevation",
        "radius",
        "azimuth",
        "elevation_id",
        "radius_id",
        "azimuth_sin",
        "azimuth_cos",
        "image_width",
        "image_height",
        "num_objects_total",
        "num_target_boxes",
        "target_visible",
        "target_max_area_norm",
        "target_total_area_norm",
    ]
    fieldnames = base_fieldnames + [column for column in quality_columns if column not in base_fieldnames]
    write_csv_rows(output_csv, fieldnames=fieldnames, rows=rows)
    write_summary(output_csv.with_suffix(".summary.md"), rows=rows)
    print(f"Wrote {len(rows)} scene-view rows to {output_csv}")


if __name__ == "__main__":
    main()
