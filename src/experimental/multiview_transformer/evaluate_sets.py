from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from multiview_transformer.common import (
    max_pairwise_azimuth_gap,
    read_csv_rows,
    unique_count_by_factor,
    viewpoint_sort_key,
    write_csv_rows,
)
from multiview_transformer.dataset import build_scene_example, collate_scene_batches, make_image_transform
from multiview_transformer.model import MultiViewAngleTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed viewpoint sets with a trained multiview transformer.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by multiview_transformer/train.py")
    parser.add_argument("--manifest-path", default="", help="Optional manifest override.")
    parser.add_argument("--split", default="test", help="Manifest split to evaluate.")
    parser.add_argument("--combo-size", type=int, default=2, help="Number of viewpoints in each candidate set.")
    parser.add_argument("--score-column", default="", help="Optional score-column override.")
    parser.add_argument("--visible-column", default="", help="Optional visible-column override.")
    parser.add_argument("--require-complete", action="store_true", help="Require every candidate viewpoint to exist in a scene.")
    parser.add_argument(
        "--shortlist-from",
        default="",
        help="Optional CSV containing a viewpoint column to shortlist before generating combinations.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-K shortlist limit.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Optional cap on evaluated scenes.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of scene-combinations per inference batch.")
    parser.add_argument("--device", default="auto", help="Optional device override.")
    parser.add_argument("--output-dir", default="", help="Optional output directory override.")
    return parser.parse_args()


def select_device(candidate: str) -> torch.device:
    candidate = candidate.strip().lower()
    if not candidate or candidate == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if candidate.isdigit():
        return torch.device(f"cuda:{candidate}")
    return torch.device(candidate)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MultiViewAngleTransformer, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    model = MultiViewAngleTransformer(
        num_classes=max(1, len(payload.get("class_names", []))),
        backbone_name=str(config["model"].get("backbone_name", "resnet18")),
        backbone_pretrained=False,
        backbone_checkpoint="",
        embed_dim=int(config["model"].get("embed_dim", 256)),
        num_heads=int(config["model"].get("num_heads", 4)),
        num_layers=int(config["model"].get("num_layers", 4)),
        mlp_ratio=float(config["model"].get("mlp_ratio", 2.0)),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, config


def grouped_rows(rows: list[dict[str, str]], split_name: str, max_scenes: int) -> list[tuple[str, dict[str, dict[str, str]]]]:
    scene_lookup: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("split") != split_name:
            continue
        scene_lookup.setdefault(str(row["scene_key"]), {})[str(row["viewpoint"])] = row
    ordered = [
        (
            scene_key,
            dict(sorted(view_lookup.items(), key=lambda item: viewpoint_sort_key(item[0]))),
        )
        for scene_key, view_lookup in sorted(scene_lookup.items())
    ]
    if max_scenes > 0:
        ordered = ordered[:max_scenes]
    return ordered


def shortlist_viewpoints(rows: list[dict[str, str]], shortlist_from: str, top_k: int) -> list[str]:
    if shortlist_from:
        shortlist_rows = read_csv_rows(Path(shortlist_from))
        viewpoints: list[str] = []
        for row in shortlist_rows:
            if row.get("viewpoint"):
                viewpoints.append(str(row["viewpoint"]))
            elif row.get("viewpoint_1") and not row.get("viewpoint_2"):
                viewpoints.append(str(row["viewpoint_1"]))
        if top_k > 0:
            viewpoints = viewpoints[:top_k]
        return sorted(set(viewpoints), key=viewpoint_sort_key)

    return sorted({str(row["viewpoint"]) for row in rows}, key=viewpoint_sort_key)


def batch_inference(
    model: MultiViewAngleTransformer,
    device: torch.device,
    examples: list[dict[str, object]],
) -> list[dict[str, object]]:
    batch = collate_scene_batches(examples)
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        outputs = model(
            images=tensor_batch["images"],
            view_mask=tensor_batch["view_mask"],
            elevation_ids=tensor_batch["elevation_ids"],
            radius_ids=tensor_batch["radius_ids"],
            azimuth_features=tensor_batch["azimuth_features"],
            class_ids=tensor_batch["class_id"],
        )
    predicted_set_scores = torch.sigmoid(outputs["set_score_logits"]).cpu().tolist()

    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "scene_key": str(example["scene_key"]),
                "target_class": str(example["target_class"]),
                "predicted_set_score": float(predicted_set_scores[index]),
                "actual_set_score": float(example["set_score_target"].item()),
                "actual_visible_any": float(example["set_visible_target"].item()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    device = select_device(args.device)
    model, config = load_model(checkpoint_path=checkpoint_path, device=device)

    manifest_path = Path(args.manifest_path or config["data"]["manifest_path"]).resolve()
    rows = read_csv_rows(manifest_path)
    score_column = args.score_column or str(config["data"].get("score_column", "target_max_area_norm"))
    visible_column = args.visible_column or str(config["data"].get("visible_column", "target_visible"))
    max_views = int(config["data"].get("max_views", 3))
    if args.combo_size > max_views:
        raise SystemExit(
            f"combo-size={args.combo_size} exceeds max_views={max_views} in the training config. "
            "Increase data.max_views and retrain before evaluating larger sets."
        )
    blank_rows = [
        row
        for row in rows
        if row.get("split") == args.split and (score_column not in row or str(row.get(score_column, "")).strip() == "")
    ]
    if blank_rows:
        raise SystemExit(
            f"Manifest column '{score_column}' is blank for {len(blank_rows)} rows in split '{args.split}'."
        )
    image_size = int(config["data"].get("image_size", 224))
    transform = make_image_transform(image_size=image_size)
    scenes = grouped_rows(rows=rows, split_name=args.split, max_scenes=args.max_scenes)

    shortlist = shortlist_viewpoints(
        rows=[row for row in rows if row.get("split") == args.split],
        shortlist_from=args.shortlist_from,
        top_k=args.top_k,
    )
    candidates = list(itertools.combinations(shortlist, args.combo_size))
    output_dir = Path(args.output_dir or Path(config["experiment"]["output_dir"]) / "eval" / args.split).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    scene_prediction_rows: list[dict[str, object]] = []
    image_cache: dict[str, torch.Tensor] = {}

    for candidate in candidates:
        examples: list[dict[str, object]] = []
        for _, view_lookup in scenes:
            selected_rows = [view_lookup[viewpoint] for viewpoint in candidate if viewpoint in view_lookup]
            if args.require_complete and len(selected_rows) != len(candidate):
                continue
            if not selected_rows:
                continue
            selected_rows = sorted(selected_rows, key=lambda row: viewpoint_sort_key(str(row["viewpoint"])))
            example = build_scene_example(
                rows=selected_rows,
                transform=transform,
                max_views=max_views,
                score_column=score_column,
                visible_column=visible_column,
                image_cache=image_cache,
            )
            examples.append(example)

        if not examples:
            continue

        per_scene_rows: list[dict[str, object]] = []
        for start in range(0, len(examples), args.batch_size):
            per_scene_rows.extend(batch_inference(model=model, device=device, examples=examples[start : start + args.batch_size]))

        combo_label = " + ".join(candidate)
        predicted_scores = [float(row["predicted_set_score"]) for row in per_scene_rows]
        actual_scores = [float(row["actual_set_score"]) for row in per_scene_rows]
        visible_scores = [float(row["actual_visible_any"]) for row in per_scene_rows]
        unique_elevation_count, unique_radius_count = unique_count_by_factor(candidate)
        summary_row: dict[str, object] = {
            "combination_label": combo_label,
            "combo_size": args.combo_size,
            "scene_count": len(per_scene_rows),
            "mean_predicted_set_score": sum(predicted_scores) / len(predicted_scores),
            "mean_actual_set_score": sum(actual_scores) / len(actual_scores),
            "mean_actual_visible_any": sum(visible_scores) / len(visible_scores),
            "max_pairwise_azimuth_gap": max_pairwise_azimuth_gap(candidate),
            "unique_elevation_count": unique_elevation_count,
            "unique_radius_count": unique_radius_count,
        }
        for index, viewpoint in enumerate(candidate, start=1):
            summary_row[f"viewpoint_{index}"] = viewpoint
        summary_rows.append(summary_row)

        for row in per_scene_rows:
            scene_prediction_row = {
                "combination_label": combo_label,
                "combo_size": args.combo_size,
                "scene_key": row["scene_key"],
                "target_class": row["target_class"],
                "predicted_set_score": row["predicted_set_score"],
                "actual_set_score": row["actual_set_score"],
                "actual_visible_any": row["actual_visible_any"],
            }
            for index, viewpoint in enumerate(candidate, start=1):
                scene_prediction_row[f"viewpoint_{index}"] = viewpoint
            scene_prediction_rows.append(scene_prediction_row)

        print(
            f"Evaluated combo {combo_label} | "
            f"scenes {len(per_scene_rows)} | "
            f"pred {summary_row['mean_predicted_set_score']:.4f} | "
            f"actual {summary_row['mean_actual_set_score']:.4f}"
        )

    summary_rows.sort(key=lambda row: (-float(row["mean_predicted_set_score"]), -float(row["mean_actual_set_score"])))
    summary_path = output_dir / f"combo_{args.combo_size}_summary.csv"
    scene_path = output_dir / f"combo_{args.combo_size}_scene_predictions.csv"

    summary_fields = [
        "combination_label",
        "combo_size",
        *[f"viewpoint_{index}" for index in range(1, args.combo_size + 1)],
        "scene_count",
        "mean_predicted_set_score",
        "mean_actual_set_score",
        "mean_actual_visible_any",
        "max_pairwise_azimuth_gap",
        "unique_elevation_count",
        "unique_radius_count",
    ]
    scene_fields = [
        "combination_label",
        "combo_size",
        *[f"viewpoint_{index}" for index in range(1, args.combo_size + 1)],
        "scene_key",
        "target_class",
        "predicted_set_score",
        "actual_set_score",
        "actual_visible_any",
    ]
    write_csv_rows(summary_path, fieldnames=summary_fields, rows=summary_rows)
    write_csv_rows(scene_path, fieldnames=scene_fields, rows=scene_prediction_rows)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "split": args.split,
        "combo_size": args.combo_size,
        "candidate_count": len(candidates),
        "evaluated_combo_count": len(summary_rows),
        "score_column": score_column,
        "visible_column": visible_column,
        "device": str(device),
    }
    (output_dir / f"combo_{args.combo_size}_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
