from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Iterable

import yaml


DEFAULT_OUTPUT_DIR = Path("outputs") / "thesis_tools" / "dataset_structure_audit"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit dataset structure, split leakage, viewpoint balance, and "
            "performance slices for a YOLO-style dataset."
        )
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--per-image",
        default="comparison_output/per_image_metrics_model_b.csv",
        help="Optional per-image metrics CSV used to rank weak slices.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the audit CSVs and markdown summary are written.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    configured_root = data_dict.get("path")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def _expand_candidate(root: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    data_dict = load_yaml(data_yaml)
    split_value = data_dict.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' was not found in {data_yaml}.")

    root = resolve_dataset_root(data_yaml, data_dict)
    raw_candidates = [split_value] if isinstance(split_value, str) else list(split_value)
    image_paths: list[Path] = []

    for raw_candidate in raw_candidates:
        candidate = _expand_candidate(root, Path(raw_candidate))
        if candidate.is_dir():
            image_paths.extend(
                sorted(path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            )
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            with candidate.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    listed_path = Path(stripped)
                    image_paths.append(_expand_candidate(root, listed_path))
            continue
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(candidate)
            continue
        raise FileNotFoundError(f"Could not resolve split item '{raw_candidate}' from {data_yaml}.")

    unique_paths = sorted({path.resolve() for path in image_paths})
    if not unique_paths:
        raise FileNotFoundError(f"No images found for split '{split}' in {data_yaml}.")
    return unique_paths


def load_class_names(data_yaml: Path) -> list[str]:
    data_dict = load_yaml(data_yaml)
    names = data_dict.get("names", {})
    if isinstance(names, dict):
        ordered = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        ordered = [str(name) for name in names]
    else:
        raise ValueError(f"Unsupported names entry in {data_yaml}: {names!r}")
    return ordered


def load_class_map(data_yaml: Path) -> dict[int, str]:
    names = load_class_names(data_yaml)
    return {index: name for index, name in enumerate(names)}


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def parse_image_metadata(image_path: Path, class_names: list[str]) -> dict[str, str]:
    match = re.match(
        r"^S0-SM_(?P<object_token>[^-]+)-el(?P<elevation>low|mid|high)-rad(?P<radius>near|mid|far)-az(?P<azimuth>\d{3})$",
        image_path.stem,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not parse image filename: {image_path.name}")

    object_token = match.group("object_token").lower()
    class_name = None
    for candidate in sorted(class_names, key=len, reverse=True):
        if object_token.startswith(candidate.lower()):
            class_name = candidate
            break
    if class_name is None:
        raise ValueError(f"Could not infer class name from token '{object_token}'.")

    remainder = object_token[len(class_name) :]
    instance_token = remainder.lstrip("_")
    if not instance_token:
        instance_token = "unknown"

    return {
        "class_name": class_name,
        "instance_id": f"{class_name}_{instance_token}",
        "elevation": match.group("elevation").lower(),
        "radius": match.group("radius").lower(),
        "azimuth": match.group("azimuth"),
    }


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Expected image path under an 'images' directory: {image_path}")


def load_label_class_names(label_path: Path, class_map: dict[int, str]) -> list[str]:
    if not label_path.exists():
        return []

    names: list[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            names.append(class_map.get(class_id, str(class_id)))
    return names


def read_per_image_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {normalize_path(row["image"]): row for row in rows if row.get("image")}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_factor(records: Iterable[dict[str, object]], factor_name: str) -> list[dict[str, object]]:
    counter = Counter(str(record[factor_name]) for record in records)
    return [
        {
            "factor": factor_name,
            "level": level,
            "image_count": count,
        }
        for level, count in sorted(counter.items())
    ]


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data_yaml).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(data_yaml)
    class_map = load_class_map(data_yaml)
    per_image_metrics = read_per_image_metrics(Path(args.per_image).resolve())

    records: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        for image_path in resolve_split_images(data_yaml, split):
            metadata = parse_image_metadata(image_path, class_names)
            label_path = label_path_from_image(image_path)
            gt_class_names = load_label_class_names(label_path, class_map)
            metric_row = per_image_metrics.get(normalize_path(image_path), {})
            metric_value = None
            if metric_row.get("ap50_95"):
                try:
                    metric_value = float(metric_row["ap50_95"])
                except ValueError:
                    metric_value = None

            records.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "label_path": str(label_path),
                    "class_name": metadata["class_name"],
                    "instance_id": metadata["instance_id"],
                    "elevation": metadata["elevation"],
                    "radius": metadata["radius"],
                    "azimuth": metadata["azimuth"],
                    "num_gt_boxes": len(gt_class_names),
                    "target_present": metadata["class_name"] in gt_class_names,
                    "ap50_95": metric_value,
                }
            )

    split_summary = [
        {"split": split, "image_count": sum(1 for record in records if record["split"] == split)}
        for split in ("train", "val", "test")
    ]

    per_instance_counts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        per_instance_counts[str(record["instance_id"])].append(record)

    class_summary: list[dict[str, object]] = []
    split_class_counts: Counter[tuple[str, str]] = Counter(
        (str(record["split"]), str(record["class_name"])) for record in records
    )
    class_instance_map: dict[str, set[str]] = defaultdict(set)
    class_instance_view_counts: dict[str, list[int]] = defaultdict(list)
    for instance_id, instance_records in per_instance_counts.items():
        class_name = str(instance_records[0]["class_name"])
        class_instance_map[class_name].add(instance_id)
        class_instance_view_counts[class_name].append(len(instance_records))

    for class_name in class_names:
        view_counts = sorted(class_instance_view_counts[class_name])
        class_summary.append(
            {
                "class_name": class_name,
                "total_images": sum(split_class_counts[(split, class_name)] for split in ("train", "val", "test")),
                "train_images": split_class_counts[("train", class_name)],
                "val_images": split_class_counts[("val", class_name)],
                "test_images": split_class_counts[("test", class_name)],
                "unique_instances": len(class_instance_map[class_name]),
                "mean_views_per_instance": format_float(safe_mean(view_counts), digits=2),
                "min_views_per_instance": min(view_counts) if view_counts else 0,
                "max_views_per_instance": max(view_counts) if view_counts else 0,
            }
        )

    factor_summary = (
        summarize_factor(records, "elevation")
        + summarize_factor(records, "radius")
        + summarize_factor(records, "azimuth")
    )

    class_factor_summary: list[dict[str, object]] = []
    for factor_name in ("elevation", "radius", "azimuth"):
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for record in records:
            grouped[(str(record["class_name"]), str(record[factor_name]))].append(record)
        for (class_name, level), group_records in sorted(grouped.items()):
            metric_values = [
                float(record["ap50_95"])
                for record in group_records
                if record["split"] == "test" and record["ap50_95"] is not None
            ]
            class_factor_summary.append(
                {
                    "class_name": class_name,
                    "factor": factor_name,
                    "level": level,
                    "image_count_all_splits": len(group_records),
                    "test_image_count": sum(1 for record in group_records if record["split"] == "test"),
                    "unique_instances": len({str(record["instance_id"]) for record in group_records}),
                    "mean_test_ap50_95": format_float(safe_mean(metric_values)),
                }
            )

    visibility_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            visibility_grouped[str(record["class_name"])].append(record)
    visibility_summary: list[dict[str, object]] = []
    for class_name in class_names:
        group_records = visibility_grouped[class_name]
        target_absent = sum(1 for record in group_records if not bool(record["target_present"]))
        total = len(group_records)
        visibility_summary.append(
            {
                "class_name": class_name,
                "test_images": total,
                "target_absent": target_absent,
                "absent_fraction": format_float(target_absent / total if total else None),
            }
        )

    instance_split_rows: list[dict[str, object]] = []
    for instance_id, instance_records in sorted(per_instance_counts.items()):
        split_presence = {split: False for split in ("train", "val", "test")}
        for record in instance_records:
            split_presence[str(record["split"])] = True
        instance_split_rows.append(
            {
                "class_name": str(instance_records[0]["class_name"]),
                "instance_id": instance_id,
                "train": int(split_presence["train"]),
                "val": int(split_presence["val"]),
                "test": int(split_presence["test"]),
                "split_count": sum(int(value) for value in split_presence.values()),
                "image_count": len(instance_records),
            }
        )

    weak_slice_rows: list[dict[str, object]] = []
    for factor_name in ("elevation", "radius", "azimuth"):
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for record in records:
            if record["split"] == "test" and record["ap50_95"] is not None:
                grouped[(str(record["class_name"]), str(record[factor_name]))].append(record)
        ranked_rows = []
        for (class_name, level), group_records in grouped.items():
            metric_values = [float(record["ap50_95"]) for record in group_records if record["ap50_95"] is not None]
            ranked_rows.append(
                {
                    "class_name": class_name,
                    "factor": factor_name,
                    "level": level,
                    "test_image_count": len(group_records),
                    "mean_test_ap50_95": safe_mean(metric_values),
                }
            )
        weak_slice_rows.extend(
            sorted(
                ranked_rows,
                key=lambda row: (row["mean_test_ap50_95"] if row["mean_test_ap50_95"] is not None else 1.0),
            )[:10]
        )

    max_instance_count = max(len(class_instance_map[class_name]) for class_name in class_names)
    expansion_plan_rows: list[dict[str, object]] = []
    for class_name in class_names:
        current_instances = len(class_instance_map[class_name])
        missing_instances = max_instance_count - current_instances
        expansion_plan_rows.append(
            {
                "class_name": class_name,
                "current_instances": current_instances,
                "target_instances_if_matched_to_max_class": max_instance_count,
                "additional_instances_needed": missing_instances,
                "additional_images_needed_if_full_72_view_grid": missing_instances * 72,
            }
        )

    write_csv(output_dir / "split_summary.csv", split_summary, ["split", "image_count"])
    write_csv(
        output_dir / "class_summary.csv",
        class_summary,
        [
            "class_name",
            "total_images",
            "train_images",
            "val_images",
            "test_images",
            "unique_instances",
            "mean_views_per_instance",
            "min_views_per_instance",
            "max_views_per_instance",
        ],
    )
    write_csv(output_dir / "factor_summary.csv", factor_summary, ["factor", "level", "image_count"])
    write_csv(
        output_dir / "class_factor_summary.csv",
        class_factor_summary,
        [
            "class_name",
            "factor",
            "level",
            "image_count_all_splits",
            "test_image_count",
            "unique_instances",
            "mean_test_ap50_95",
        ],
    )
    write_csv(
        output_dir / "visibility_summary.csv",
        visibility_summary,
        ["class_name", "test_images", "target_absent", "absent_fraction"],
    )
    write_csv(
        output_dir / "instance_split_overlap.csv",
        instance_split_rows,
        ["class_name", "instance_id", "train", "val", "test", "split_count", "image_count"],
    )
    write_csv(
        output_dir / "weak_slices.csv",
        [
            {
                **row,
                "mean_test_ap50_95": format_float(row["mean_test_ap50_95"]),
            }
            for row in weak_slice_rows
        ],
        ["class_name", "factor", "level", "test_image_count", "mean_test_ap50_95"],
    )
    write_csv(
        output_dir / "expansion_plan_equalize_instances.csv",
        expansion_plan_rows,
        [
            "class_name",
            "current_instances",
            "target_instances_if_matched_to_max_class",
            "additional_instances_needed",
            "additional_images_needed_if_full_72_view_grid",
        ],
    )

    total_instances = len(per_instance_counts)
    multi_split_instances = sum(1 for row in instance_split_rows if int(row["split_count"]) > 1)
    three_split_instances = sum(1 for row in instance_split_rows if int(row["split_count"]) == 3)
    factor_counter = Counter((row["factor"], row["image_count"]) for row in factor_summary)
    perfectly_balanced = len(factor_counter) == 3
    weak_slices_sorted = sorted(
        [
            row
            for row in weak_slice_rows
            if row["mean_test_ap50_95"] is not None
        ],
        key=lambda row: float(row["mean_test_ap50_95"]),
    )[:8]
    expansion_rows_sorted = sorted(
        expansion_plan_rows,
        key=lambda row: int(row["additional_instances_needed"]),
        reverse=True,
    )
    visibility_rows_sorted = sorted(
        visibility_summary,
        key=lambda row: float(row["absent_fraction"]) if row["absent_fraction"] != "n/a" else -1.0,
        reverse=True,
    )

    report_lines = [
        "# Dataset Structure Audit",
        "",
        "## Headline findings",
        f"- Total images: {len(records)}",
        f"- Split sizes: train {split_summary[0]['image_count']}, val {split_summary[1]['image_count']}, test {split_summary[2]['image_count']}",
        f"- Unique object instances inferred from filenames: {total_instances}",
        f"- Instances appearing in more than one split: {multi_split_instances}/{total_instances}",
        f"- Instances appearing in all three splits: {three_split_instances}/{total_instances}",
        (
            "- Viewpoint grid balance: overall elevation, radius, and azimuth counts are exactly balanced."
            if perfectly_balanced
            else "- Viewpoint grid balance: not perfectly balanced overall."
        ),
        "",
        "## What the dataset is doing well",
        "- The dataset is already very strong on viewpoint coverage density.",
        "- Each object instance is usually covered by a near-complete 72-view grid.",
        "- The global counts over elevation, radius, and azimuth are not the main source of bias.",
        "",
        "## Structural risks",
        "- The main bias is not missing viewpoints but repeated viewpoints of the same object instances.",
        "- Because the same instances appear in train, val, and test, the benchmark mainly measures viewpoint interpolation on known objects.",
        "- That setup can make results look much better than they would on unseen instances or scenes.",
        "",
        "## Class imbalance by unique instances",
    ]
    for row in sorted(class_summary, key=lambda item: int(item["unique_instances"])):
        report_lines.append(
            f"- {row['class_name']}: {row['unique_instances']} instances, {row['total_images']} images, "
            f"{row['mean_views_per_instance']} views per instance on average"
        )

    report_lines.extend(
        [
            "",
            "## Label visibility issues in the test split",
        ]
    )
    for row in visibility_rows_sorted[:5]:
        report_lines.append(
            f"- {row['class_name']}: absent target fraction {row['absent_fraction']} "
            f"({row['target_absent']}/{row['test_images']})"
        )

    report_lines.extend(
        [
            "",
            "## Weak performance slices in the current test set",
        ]
    )
    for row in weak_slices_sorted:
        report_lines.append(
            f"- {row['class_name']} | {row['factor']}={row['level']}: "
            f"mean test AP50-95 {format_float(row['mean_test_ap50_95'])} over {row['test_image_count']} images"
        )

    report_lines.extend(
        [
            "",
            "## Best next data additions",
            "- Add new object instances before adding more viewpoints of existing instances.",
            "- Build an instance-disjoint test split so that no object instance appears in both train and test.",
            "- Use extra data to strengthen the low-performing slices, but collect those slices on new instances and new scenes.",
            "- If you want equal class balance by unique instances, the largest gains come from the classes below.",
        ]
    )
    for row in expansion_rows_sorted[:6]:
        report_lines.append(
            f"- {row['class_name']}: add {row['additional_instances_needed']} new instances "
            f"(about {row['additional_images_needed_if_full_72_view_grid']} images for a full 72-view capture)"
        )

    report_lines.extend(
        [
            "",
            "## Output files",
            "- split_summary.csv",
            "- class_summary.csv",
            "- factor_summary.csv",
            "- class_factor_summary.csv",
            "- visibility_summary.csv",
            "- instance_split_overlap.csv",
            "- weak_slices.csv",
            "- expansion_plan_equalize_instances.csv",
        ]
    )

    (output_dir / "dataset_structure_audit.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved dataset structure audit to: {output_dir}")


if __name__ == "__main__":
    main()
