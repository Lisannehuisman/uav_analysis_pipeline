from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def compute_multiview_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    view_mask = batch["view_mask"].float()
    view_score_targets = batch["view_score_targets"].float()
    view_visible_targets = batch["view_visible_targets"].float()
    set_score_target = batch["set_score_target"].float()
    class_id = batch["class_id"].long()

    set_score_prediction = torch.sigmoid(outputs["set_score_logits"])
    view_score_prediction = torch.sigmoid(outputs["view_score_logits"])

    set_score_loss = F.smooth_l1_loss(set_score_prediction, set_score_target, reduction="mean")
    view_score_error = F.smooth_l1_loss(view_score_prediction, view_score_targets, reduction="none")
    view_score_loss = masked_mean(view_score_error, view_mask)

    view_visible_error = F.binary_cross_entropy_with_logits(
        outputs["view_visible_logits"],
        view_visible_targets,
        reduction="none",
    )
    visible_loss = masked_mean(view_visible_error, view_mask)

    class_loss = F.cross_entropy(outputs["class_logits"], class_id, reduction="mean")

    total_loss = (
        weights.get("set_weight", 1.0) * set_score_loss
        + weights.get("view_weight", 0.5) * view_score_loss
        + weights.get("visible_weight", 0.25) * visible_loss
        + weights.get("class_weight", 0.1) * class_loss
    )

    with torch.no_grad():
        set_mae = torch.mean(torch.abs(set_score_prediction - set_score_target))
        view_mae = masked_mean(torch.abs(view_score_prediction - view_score_targets), view_mask)
        visible_accuracy = masked_mean(
            ((torch.sigmoid(outputs["view_visible_logits"]) >= 0.5) == (view_visible_targets >= 0.5)).float(),
            view_mask,
        )
        class_accuracy = (outputs["class_logits"].argmax(dim=1) == class_id).float().mean()

    metrics = {
        "loss": float(total_loss.detach().item()),
        "set_score_loss": float(set_score_loss.detach().item()),
        "view_score_loss": float(view_score_loss.detach().item()),
        "visible_loss": float(visible_loss.detach().item()),
        "class_loss": float(class_loss.detach().item()),
        "set_mae": float(set_mae.detach().item()),
        "view_mae": float(view_mae.detach().item()),
        "visible_accuracy": float(visible_accuracy.detach().item()),
        "class_accuracy": float(class_accuracy.detach().item()),
    }
    return total_loss, metrics
