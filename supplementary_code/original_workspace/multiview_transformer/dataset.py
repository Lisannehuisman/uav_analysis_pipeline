from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .common import (
    group_rows_by_scene,
    read_csv_rows,
    spaced_indices,
    to_float,
    to_int,
    viewpoint_sort_key,
)


def make_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_image_tensor(
    image_path: Path,
    transform: Callable,
    image_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    cache_key = str(image_path)
    if image_cache is not None and cache_key in image_cache:
        return image_cache[cache_key].clone()

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB"))

    if image_cache is not None:
        image_cache[cache_key] = tensor.clone()
    return tensor


def build_scene_example(
    rows: Sequence[dict[str, str]],
    transform: Callable,
    max_views: int,
    score_column: str,
    visible_column: str,
    image_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("build_scene_example requires at least one row")

    image_tensors: list[torch.Tensor] = []
    view_mask = torch.zeros(max_views, dtype=torch.float32)
    elevation_ids = torch.zeros(max_views, dtype=torch.long)
    radius_ids = torch.zeros(max_views, dtype=torch.long)
    azimuth_features = torch.zeros(max_views, 2, dtype=torch.float32)
    view_score_targets = torch.zeros(max_views, dtype=torch.float32)
    view_visible_targets = torch.zeros(max_views, dtype=torch.float32)
    selected_viewpoints: list[str] = []

    for index, row in enumerate(rows[:max_views]):
        tensor = load_image_tensor(Path(row["image_path"]), transform=transform, image_cache=image_cache)
        image_tensors.append(tensor)
        view_mask[index] = 1.0
        elevation_ids[index] = to_int(row.get("elevation_id", 0))
        radius_ids[index] = to_int(row.get("radius_id", 0))
        azimuth_features[index, 0] = to_float(row.get("azimuth_sin", 0.0))
        azimuth_features[index, 1] = to_float(row.get("azimuth_cos", 1.0))
        view_score_targets[index] = to_float(row.get(score_column, 0.0))
        view_visible_targets[index] = to_float(row.get(visible_column, 0.0))
        selected_viewpoints.append(str(row["viewpoint"]))

    if not image_tensors:
        raise ValueError("No image tensors were loaded for the scene example")

    template = image_tensors[0]
    while len(image_tensors) < max_views:
        image_tensors.append(torch.zeros_like(template))

    scene_rows = list(rows[:max_views])
    set_score_target = max(to_float(row.get(score_column, 0.0)) for row in scene_rows)
    set_visible_target = max(to_float(row.get(visible_column, 0.0)) for row in scene_rows)

    return {
        "images": torch.stack(image_tensors, dim=0),
        "view_mask": view_mask,
        "elevation_ids": elevation_ids,
        "radius_ids": radius_ids,
        "azimuth_features": azimuth_features,
        "view_score_targets": view_score_targets,
        "view_visible_targets": view_visible_targets,
        "set_score_target": torch.tensor(float(set_score_target), dtype=torch.float32),
        "set_visible_target": torch.tensor(float(set_visible_target), dtype=torch.float32),
        "class_id": torch.tensor(to_int(rows[0].get("target_class_id", 0)), dtype=torch.long),
        "scene_key": str(rows[0]["scene_key"]),
        "target_class": str(rows[0]["target_class"]),
        "selected_viewpoints": selected_viewpoints,
    }


class SceneSetDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: int,
        score_column: str,
        visible_column: str = "target_visible",
        min_views: int = 1,
        max_views: int = 3,
        random_subset: bool = True,
        deterministic_sampling: bool = False,
        seed: int = 0,
        max_scenes: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_csv_rows(self.manifest_path)
        self.scene_groups = list(group_rows_by_scene(self.rows, split_name=split).items())
        if max_scenes > 0:
            self.scene_groups = self.scene_groups[:max_scenes]
        self.image_transform = make_image_transform(image_size=image_size)
        self.score_column = score_column
        self.visible_column = visible_column
        self.min_views = max(1, min_views)
        self.max_views = max(1, max_views)
        self.random_subset = random_subset
        self.deterministic_sampling = deterministic_sampling
        self.seed = int(seed)
        self.class_names = sorted({str(row["target_class"]) for row in self.rows})
        self.num_classes = len({to_int(row.get("target_class_id", 0)) for row in self.rows})

    def __len__(self) -> int:
        return len(self.scene_groups)

    def _rng_for_index(self, index: int) -> random.Random:
        if self.deterministic_sampling:
            return random.Random(self.seed + index)
        return random

    def _select_rows(self, rows: Sequence[dict[str, str]], index: int) -> list[dict[str, str]]:
        available = len(rows)
        upper = min(self.max_views, available)
        lower = min(self.min_views, upper)
        if self.random_subset:
            rng = self._rng_for_index(index)
            count = rng.randint(lower, upper)
            selected = rng.sample(list(rows), count)
            return sorted(selected, key=lambda item: viewpoint_sort_key(str(item["viewpoint"])))

        count = upper
        positions = spaced_indices(available, count)
        return [rows[position] for position in positions]

    def __getitem__(self, index: int) -> dict[str, object]:
        _, rows = self.scene_groups[index]
        selected_rows = self._select_rows(rows, index=index)
        return build_scene_example(
            rows=selected_rows,
            transform=self.image_transform,
            max_views=self.max_views,
            score_column=self.score_column,
            visible_column=self.visible_column,
        )


def collate_scene_batches(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    tensor_fields = {
        "images",
        "view_mask",
        "elevation_ids",
        "radius_ids",
        "azimuth_features",
        "view_score_targets",
        "view_visible_targets",
        "set_score_target",
        "set_visible_target",
        "class_id",
    }
    output: dict[str, object] = {}
    for field in tensor_fields:
        output[field] = torch.stack([item[field] for item in batch], dim=0)

    output["scene_keys"] = [str(item["scene_key"]) for item in batch]
    output["target_classes"] = [str(item["target_class"]) for item in batch]
    output["selected_viewpoints"] = [list(item["selected_viewpoints"]) for item in batch]
    return output
