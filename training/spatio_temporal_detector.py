from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from training.fusion_head import FusionHead
from training.spatial_resnet50 import SpatialResNet50
from training.temporal_diff_cnn import TemporalDiffCNN


@dataclass
class ModelConfig:
    num_classes: int = 1
    temporal_in_channels: int = 3
    temporal_num_frames: int = 7
    temporal_feature_dim: int = 256
    fusion_hidden_dim: int = 512
    dropout: float = 0.3
    pretrained: bool = True
    freeze_spatial_backbone: bool = False
    temporal_pool: Literal["mean", "attention"] = "mean"
    use_spatial_attention: bool = True
    use_texture_enhancement: bool = True


class SpatioTemporalDeepfakeDetector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.spatial_branch = SpatialResNet50(
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_spatial_backbone,
            use_spatial_attention=config.use_spatial_attention,
            use_texture_enhancement=config.use_texture_enhancement,
        )
        self.temporal_branch = TemporalDiffCNN(
            in_channels=config.temporal_in_channels,
            feature_dim=config.temporal_feature_dim,
            pool_mode=config.temporal_pool,
            dropout=config.dropout,
        )
        self.fusion_head = FusionHead(
            spatial_dim=self.spatial_branch.out_dim,
            temporal_dim=config.temporal_feature_dim,
            hidden_dim=config.fusion_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    def freeze_spatial(self) -> None:
        self.spatial_branch.freeze()

    def unfreeze_spatial(self) -> None:
        self.spatial_branch.unfreeze()

    def _validate_temporal_input(self, temporal: torch.Tensor) -> None:
        if temporal.ndim != 5:
            raise ValueError(
                f"Expected temporal input with shape [B, T, C, H, W], got ndim={temporal.ndim}"
            )
        if temporal.shape[1] != self.config.temporal_num_frames:
            raise ValueError(
                f"Expected temporal_num_frames={self.config.temporal_num_frames}, got {temporal.shape[1]}"
            )

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_temporal_input(temporal)

        if return_features:
            spatial_feat, spatial_extras = self.spatial_branch(
                spatial,
                return_attention=True,
                return_feature_maps=True,
            )
            temporal_feat, temporal_attn = self.temporal_branch(
                temporal,
                return_attention=True,
            )
        else:
            spatial_feat = self.spatial_branch(spatial)
            temporal_feat = self.temporal_branch(temporal)
            spatial_extras = {}
            temporal_attn = None

        logits = self.fusion_head(spatial_feat, temporal_feat)

        if self.config.num_classes == 1:
            logits = logits.squeeze(1)

        if not return_features:
            return logits

        features = {
            "spatial_feat": spatial_feat,
            "temporal_feat": temporal_feat,
        }
        features.update(spatial_extras)
        if temporal_attn is not None:
            features["temporal_attn"] = temporal_attn
        return logits, features


if __name__ == "__main__":
    cfg = ModelConfig(
        temporal_pool="attention",
        use_spatial_attention=True,
        use_texture_enhancement=True,
    )
    model = SpatioTemporalDeepfakeDetector(cfg)

    spatial = torch.randn(2, 3, 224, 224)
    temporal = torch.randn(2, 7, 3, 224, 224)

    logits, features = model(spatial, temporal, return_features=True)
    print("logits:", tuple(logits.shape))
    print("spatial_feat:", tuple(features["spatial_feat"].shape))
    print("temporal_feat:", tuple(features["temporal_feat"].shape))
    print("spatial_attn:", tuple(features["spatial_attn"].shape))
    print("temporal_attn:", tuple(features["temporal_attn"].shape))
