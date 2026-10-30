from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def build_resnet_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    backbone_key = backbone_name.lower()
    if backbone_key == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        feature_dim = 512
    elif backbone_key == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
        feature_dim = 512
    elif backbone_key == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")

    encoder = nn.Sequential(*list(model.children())[:-2])
    return encoder, feature_dim


class MultiViewAngleTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        backbone_pretrained: bool = False,
        backbone_checkpoint: str = "",
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_resnet_backbone(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained,
        )
        if backbone_checkpoint:
            checkpoint_path = Path(backbone_checkpoint)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.elevation_embedding = nn.Embedding(3, embed_dim)
        self.radius_embedding = nn.Embedding(3, embed_dim)
        self.class_embedding = nn.Embedding(max(1, num_classes), embed_dim)
        self.azimuth_projection = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.view_dropout = nn.Dropout(dropout)
        self.set_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.view_score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.view_visible_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.set_score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.class_head = nn.Linear(embed_dim, max(1, num_classes))
        nn.init.trunc_normal_(self.set_token, std=0.02)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, view_count, channels, height, width = images.shape
        flattened = images.view(batch_size * view_count, channels, height, width)
        feature_map = self.backbone(flattened)
        pooled = self.pool(feature_map).flatten(1)
        projected = self.image_projection(pooled)
        return projected.view(batch_size, view_count, -1)

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        elevation_ids: torch.Tensor,
        radius_ids: torch.Tensor,
        azimuth_features: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_tokens = self.encode_images(images)
        class_tokens = self.class_embedding(class_ids).unsqueeze(1)
        view_tokens = (
            image_tokens
            + self.elevation_embedding(elevation_ids)
            + self.radius_embedding(radius_ids)
            + self.azimuth_projection(azimuth_features)
            + class_tokens
        )
        view_tokens = self.view_dropout(view_tokens)

        set_token = self.set_token.expand(images.shape[0], 1, -1) + class_tokens
        transformer_input = torch.cat([set_token, view_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((images.shape[0], 1), dtype=torch.bool, device=images.device),
                ~view_mask.bool(),
            ],
            dim=1,
        )
        encoded = self.transformer(transformer_input, src_key_padding_mask=padding_mask)
        set_representation = encoded[:, 0]
        view_representations = encoded[:, 1:]

        return {
            "set_score_logits": self.set_score_head(set_representation).squeeze(-1),
            "view_score_logits": self.view_score_head(view_representations).squeeze(-1),
            "view_visible_logits": self.view_visible_head(view_representations).squeeze(-1),
            "class_logits": self.class_head(set_representation),
        }

