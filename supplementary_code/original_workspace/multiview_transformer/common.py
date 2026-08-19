from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCENE_VIEW_RE = re.compile(
    r"^(?P<scene>.+)-(?P<viewpoint>el[a-z]+-rad[a-z]+-az(?P<azimuth>\d+))$",
    re.IGNORECASE,
)
TARGET_RE = re.compile(r"^S0-SM_([^-]+)-", re.IGNORECASE)
VIEWPOINT_RE = re.compile(r"^(el[a-z]+)-(rad[a-z]+)-az(\d+)$", re.IGNORECASE)
ELEVATION_SORT = {"low": 0, "mid": 1, "high": 2}
RADIUS_SORT = {"near": 0, "mid": 1, "far": 2}
ELEVATION_TOKENS = {"ellow": "low", "elmid": "mid", "elhigh": "high"}
RADIUS_TOKENS = {"radnear": "near", "radmid": "mid", "radfar": "far"}
MANIFEST_NUMERIC_COLUMNS = {
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
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def class_name_lookup(data_dict: dict) -> tuple[dict[int, str], dict[str, int]]:
    raw_names = data_dict.get("names", {})
    if isinstance(raw_names, list):
        id_to_name = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        id_to_name = {int(key): str(value) for key, value in raw_names.items()}
    name_to_id = {name.lower(): class_id for class_id, name in id_to_name.items()}
    return id_to_name, name_to_id


def resolve_dataset_root(data_yaml: Path, data_dict: dict) -> Path:
    raw_root = str(data_dict.get("path", "")).strip()
    if not raw_root:
        return data_yaml.parent.resolve()
    root_path = Path(raw_root)
    if not root_path.is_absolute():
        root_path = (data_yaml.parent / root_path).resolve()
    return root_path


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def image_paths_from_entry(dataset_root: Path, entry_value: str) -> list[Path]:
    resolved = resolve_path(dataset_root, entry_value)
    if resolved.is_dir():
        return sorted(
            [
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    if resolved.is_file() and resolved.suffix.lower() in {".txt", ".lst"}:
        image_paths: list[Path] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                image_path = resolve_path(resolved.parent, line)
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(image_path)
        return sorted(image_paths)

    if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
        return [resolved]

    raise FileNotFoundError(f"Could not resolve dataset entry '{entry_value}' under {dataset_root}")


def resolve_split_images(data_yaml: Path, split_name: str) -> list[Path]:
    data_dict = read_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml, data_dict)
    entry = data_dict.get(split_name)
    if entry is None:
        return []
    if isinstance(entry, (list, tuple)):
        images: list[Path] = []
        for value in entry:
            images.extend(image_paths_from_entry(dataset_root, str(value)))
        return sorted(set(images))
    return image_paths_from_entry(dataset_root, str(entry))


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        image_index = parts.index("images")
        parts[image_index] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = image_path.with_suffix(".txt")
    return label_path


def parse_viewpoint(viewpoint: str) -> tuple[str, str, int]:
    match = VIEWPOINT_RE.match(viewpoint.strip().lower())
    if match is None:
        raise ValueError(f"Could not parse viewpoint '{viewpoint}'")
    elevation_token, radius_token, azimuth_text = match.groups()
    elevation = ELEVATION_TOKENS.get(elevation_token.lower(), elevation_token.lower())
    radius = RADIUS_TOKENS.get(radius_token.lower(), radius_token.lower())
    return elevation, radius, int(azimuth_text)


def viewpoint_sort_key(viewpoint: str) -> tuple[int, int, int]:
    elevation, radius, azimuth = parse_viewpoint(viewpoint)
    return (
        ELEVATION_SORT.get(elevation, 99),
        RADIUS_SORT.get(radius, 99),
        azimuth,
    )


def human_viewpoint_label(viewpoint: str) -> str:
    elevation, radius, azimuth = parse_viewpoint(viewpoint)
    return f"{elevation} | {radius} | az{azimuth:03d}"


def parse_scene_view_metadata(file_name: str, known_class_names: Sequence[str]) -> tuple[str, str, str, str, int, str]:
    stem = Path(file_name).stem
    match = SCENE_VIEW_RE.match(stem)
    if match is None:
        raise ValueError(f"Could not parse scene/viewpoint from '{file_name}'")

    scene_key = match.group("scene")
    viewpoint = match.group("viewpoint").lower()
    elevation, radius, azimuth = parse_viewpoint(viewpoint)

    target_match = TARGET_RE.search(stem)
    if target_match is None:
        raise ValueError(f"Could not infer target class from '{file_name}'")

    object_token = target_match.group(1).lower()
    target_class = ""
    for candidate in sorted((name.lower() for name in known_class_names), key=len, reverse=True):
        if object_token.startswith(candidate):
            target_class = candidate
            break
    if not target_class:
        prefix_match = re.match(r"([a-zA-Z]+)", object_token)
        if prefix_match is None:
            raise ValueError(f"Could not map '{object_token}' to a target class")
        target_class = prefix_match.group(1).lower()

    return scene_key, viewpoint, elevation, radius, azimuth, target_class


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append(
                (
                    int(float(parts[0])),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            )
    return rows


def compute_target_box_stats(
    labels: Sequence[tuple[int, float, float, float, float]],
    target_class_id: int,
) -> tuple[int, int, int, float, float]:
    num_objects_total = len(labels)
    target_boxes = [label for label in labels if label[0] == target_class_id]
    num_target_boxes = len(target_boxes)
    target_visible = int(num_target_boxes > 0)
    target_areas = [max(0.0, width) * max(0.0, height) for _, _, _, width, height in target_boxes]
    target_max_area_norm = max(target_areas) if target_areas else 0.0
    target_total_area_norm = sum(target_areas)
    return (
        num_objects_total,
        num_target_boxes,
        target_visible,
        float(target_max_area_norm),
        float(target_total_area_norm),
    )


def to_float(value: str | float | int, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str | float | int, default: int = 0) -> int:
    return int(round(to_float(value, default=default)))


def group_rows_by_scene(rows: Sequence[dict[str, str]], split_name: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != split_name:
            continue
        grouped[str(row["scene_key"])].append(row)
    for scene_key, members in grouped.items():
        grouped[scene_key] = sorted(members, key=lambda item: viewpoint_sort_key(str(item["viewpoint"])))
    return dict(sorted(grouped.items()))


def azimuth_gap(first: int, second: int) -> int:
    delta = abs(int(first) - int(second)) % 360
    return min(delta, 360 - delta)


def max_pairwise_azimuth_gap(viewpoints: Sequence[str]) -> int:
    azimuths = [parse_viewpoint(viewpoint)[2] for viewpoint in viewpoints]
    if len(azimuths) < 2:
        return 0
    best_gap = 0
    for index, first in enumerate(azimuths):
        for second in azimuths[index + 1 :]:
            best_gap = max(best_gap, azimuth_gap(first, second))
    return best_gap


def unique_count_by_factor(viewpoints: Sequence[str]) -> tuple[int, int]:
    elevations = {parse_viewpoint(viewpoint)[0] for viewpoint in viewpoints}
    radii = {parse_viewpoint(viewpoint)[1] for viewpoint in viewpoints}
    return len(elevations), len(radii)


def spaced_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [0]
    return [int(round(index * (length - 1) / (count - 1))) for index in range(count)]


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    numerator = sum(a * b for a, b in zip(centered_first, centered_second, strict=True))
    denom_first = math.sqrt(sum(a * a for a in centered_first))
    denom_second = math.sqrt(sum(b * b for b in centered_second))
    denominator = denom_first * denom_second
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator

