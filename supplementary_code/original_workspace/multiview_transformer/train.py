from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from multiview_transformer.common import pearson_correlation, read_csv_rows
from multiview_transformer.dataset import SceneSetDataset, collate_scene_batches
from multiview_transformer.losses import compute_multiview_losses
from multiview_transformer.model import MultiViewAngleTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multiview transformer viewpoint-ranking baseline.")
    parser.add_argument("--config", required=True, help="YAML config for the experiment.")
    parser.add_argument("--manifest-path", default="", help="Optional manifest override.")
    parser.add_argument("--output-dir", default="", help="Optional output directory override.")
    parser.add_argument("--device", default="", help="Optional device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--epochs", type=int, default=0, help="Optional epoch override.")
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = deepcopy(config)
    if args.manifest_path:
        updated["data"]["manifest_path"] = args.manifest_path
    if args.output_dir:
        updated["experiment"]["output_dir"] = args.output_dir
    if args.device:
        updated["train"]["device"] = args.device
    if args.epochs > 0:
        updated["train"]["epochs"] = args.epochs
    return updated


def select_device(device_value: str) -> torch.device:
    candidate = device_value.strip().lower()
    if not candidate or candidate == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if candidate.isdigit():
        return torch.device(f"cuda:{candidate}")
    return torch.device(candidate)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(row[key] for row in rows) / len(rows))


def run_epoch(
    model: MultiViewAngleTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler,
    loss_weights: dict[str, float],
    use_amp: bool,
    grad_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_rows: list[dict[str, float]] = []
    predicted_scores: list[float] = []
    actual_scores: list[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(
                images=batch["images"],
                view_mask=batch["view_mask"],
                elevation_ids=batch["elevation_ids"],
                radius_ids=batch["radius_ids"],
                azimuth_features=batch["azimuth_features"],
                class_ids=batch["class_id"],
            )
            loss, metrics = compute_multiview_losses(outputs=outputs, batch=batch, weights=loss_weights)

        if training:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        metric_rows.append(metrics)
        predicted_scores.extend(torch.sigmoid(outputs["set_score_logits"]).detach().cpu().tolist())
        actual_scores.extend(batch["set_score_target"].detach().cpu().tolist())

    return {
        "loss": mean_metric(metric_rows, "loss"),
        "set_score_loss": mean_metric(metric_rows, "set_score_loss"),
        "view_score_loss": mean_metric(metric_rows, "view_score_loss"),
        "visible_loss": mean_metric(metric_rows, "visible_loss"),
        "class_loss": mean_metric(metric_rows, "class_loss"),
        "set_mae": mean_metric(metric_rows, "set_mae"),
        "view_mae": mean_metric(metric_rows, "view_mae"),
        "visible_accuracy": mean_metric(metric_rows, "visible_accuracy"),
        "class_accuracy": mean_metric(metric_rows, "class_accuracy"),
        "set_corr": pearson_correlation(predicted_scores, actual_scores),
    }


def save_checkpoint(
    path: Path,
    model: MultiViewAngleTransformer,
    config: dict,
    epoch: int,
    metrics: dict[str, float],
    class_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "class_names": class_names,
        },
        path,
    )


def append_history(csv_path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def validate_supervision_columns(
    manifest_rows: list[dict[str, str]],
    splits: list[str],
    score_column: str,
    visible_column: str,
) -> None:
    missing_score = 0
    missing_visible = 0
    relevant = [row for row in manifest_rows if row.get("split") in splits]
    if not relevant:
        raise SystemExit(f"No manifest rows found for splits {splits}")
    for row in relevant:
        if score_column not in row or str(row.get(score_column, "")).strip() == "":
            missing_score += 1
        if visible_column not in row or str(row.get(visible_column, "")).strip() == "":
            missing_visible += 1
    if missing_score:
        raise SystemExit(
            f"Manifest column '{score_column}' is blank for {missing_score} rows across splits {splits}. "
            "If you switched to detector-quality supervision, rebuild the manifest with train/val quality rows too."
        )
    if missing_visible:
        raise SystemExit(
            f"Manifest column '{visible_column}' is blank for {missing_visible} rows across splits {splits}."
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = apply_cli_overrides(read_config(config_path), args)
    seed = int(config["train"].get("seed", 0))
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"]).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts" / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    device = select_device(str(config["train"].get("device", "auto")))
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    manifest_path = Path(config["data"]["manifest_path"]).resolve()
    score_column = str(config["data"].get("score_column", "target_max_area_norm"))
    visible_column = str(config["data"].get("visible_column", "target_visible"))
    manifest_rows = read_csv_rows(manifest_path)
    validate_supervision_columns(
        manifest_rows=manifest_rows,
        splits=[str(config["data"].get("train_split", "train")), str(config["data"].get("val_split", "val"))],
        score_column=score_column,
        visible_column=visible_column,
    )
    image_size = int(config["data"].get("image_size", 224))
    max_views = int(config["data"].get("max_views", 3))
    min_views = int(config["data"].get("min_views", 1))
    val_view_count = int(config["evaluation"].get("val_view_count", max_views))

    train_dataset = SceneSetDataset(
        manifest_path=manifest_path,
        split=str(config["data"].get("train_split", "train")),
        image_size=image_size,
        score_column=score_column,
        visible_column=visible_column,
        min_views=min_views,
        max_views=max_views,
        random_subset=True,
        deterministic_sampling=False,
        seed=seed,
        max_scenes=int(config["data"].get("max_train_scenes", 0)),
    )
    val_dataset = SceneSetDataset(
        manifest_path=manifest_path,
        split=str(config["data"].get("val_split", "val")),
        image_size=image_size,
        score_column=score_column,
        visible_column=visible_column,
        min_views=val_view_count,
        max_views=val_view_count,
        random_subset=False,
        deterministic_sampling=True,
        seed=seed,
        max_scenes=int(config["data"].get("max_val_scenes", 0)),
    )

    batch_size = int(config["train"].get("batch_size", 8))
    num_workers = int(config["train"].get("num_workers", 4))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_scene_batches,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_scene_batches,
    )

    model = MultiViewAngleTransformer(
        num_classes=max(1, train_dataset.num_classes),
        backbone_name=str(config["model"].get("backbone_name", "resnet18")),
        backbone_pretrained=bool(config["model"].get("backbone_pretrained", False)),
        backbone_checkpoint=str(config["model"].get("backbone_checkpoint", "")),
        embed_dim=int(config["model"].get("embed_dim", 256)),
        num_heads=int(config["model"].get("num_heads", 4)),
        num_layers=int(config["model"].get("num_layers", 4)),
        mlp_ratio=float(config["model"].get("mlp_ratio", 2.0)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)

    learning_rate = float(config["train"].get("lr", 3e-4))
    weight_decay = float(config["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"].get("epochs", 40))),
    )
    scaler = GradScaler(enabled=use_amp)

    freeze_backbone_epochs = int(config["train"].get("freeze_backbone_epochs", 0))
    loss_weights = dict(config.get("loss", {}))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))
    epochs = int(config["train"].get("epochs", 40))
    patience = int(config["train"].get("patience", 8))
    metric_name = str(config["train"].get("metric", "val_set_mae"))
    maximize_metric = bool(config["train"].get("maximize_metric", False))
    history_path = output_dir / "logs" / "history.csv"

    best_metric = float("-inf") if maximize_metric else float("inf")
    best_epoch = -1
    stale_epochs = 0
    history_fields = [
        "epoch",
        "train_loss",
        "train_set_mae",
        "train_view_mae",
        "train_set_corr",
        "val_loss",
        "val_set_mae",
        "val_view_mae",
        "val_set_corr",
        "learning_rate",
    ]

    for epoch in range(1, epochs + 1):
        model.set_backbone_trainable(epoch > freeze_backbone_epochs)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            loss_weights=loss_weights,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            scaler=scaler,
            loss_weights=loss_weights,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_set_mae": round(train_metrics["set_mae"], 6),
            "train_view_mae": round(train_metrics["view_mae"], 6),
            "train_set_corr": round(train_metrics["set_corr"], 6) if not np.isnan(train_metrics["set_corr"]) else "",
            "val_loss": round(val_metrics["loss"], 6),
            "val_set_mae": round(val_metrics["set_mae"], 6),
            "val_view_mae": round(val_metrics["view_mae"], 6),
            "val_set_corr": round(val_metrics["set_corr"], 6) if not np.isnan(val_metrics["set_corr"]) else "",
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_history(history_path, fieldnames=history_fields, row=epoch_row)

        monitored_value = val_metrics["set_mae"] if metric_name == "val_set_mae" else val_metrics["loss"]
        improved = monitored_value > best_metric if maximize_metric else monitored_value < best_metric
        if improved:
            best_metric = monitored_value
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                config=config,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                class_names=train_dataset.class_names,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            config=config,
            epoch=epoch,
            metrics={"train": train_metrics, "val": val_metrics},
            class_names=train_dataset.class_names,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val set MAE {val_metrics['set_mae']:.4f}"
        )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch} after {stale_epochs} stale epochs.")
            break

    summary = {
        "device": str(device),
        "use_amp": use_amp,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "train_scene_count": len(train_dataset),
        "val_scene_count": len(val_dataset),
        "score_column": score_column,
        "visible_column": visible_column,
        "manifest_path": str(manifest_path),
    }
    (output_dir / "artifacts" / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
