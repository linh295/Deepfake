from __future__ import annotations

from dataclasses import InitVar, dataclass
import torch
import torch.nn as nn

from training.fusion_head import FusionHead, WeightedProbabilityFusionHead
from training.spatial_resnet50 import SpatialResNet50
from training.temporal_diff_cnn import TemporalDiffCNN, TemporalPoolMode


@dataclass
class ModelConfig:
    num_classes: int = 1
    temporal_in_channels: int = 3
    temporal_num_frames: int = 7
    temporal_feature_dim: int = 512
    fusion_hidden_dim: int = 1024
    dropout: float = 0.5
    pretrained: bool = True
    freeze_spatial_backbone: bool = False
    temporal_pool: TemporalPoolMode = "mean"
    use_spatial_attention: bool = True
    use_texture_enhancement: bool = True
    # Accepted only so legacy checkpoints still deserialize; intentionally ignored.
    use_cross_branch_attention: InitVar[bool | None] = None
    use_feature_delta: bool = False
    spatial_only: bool = False
    temporal_only: bool = False
    fusion_mode: str = "concat"
    fusion_spatial_weight: float = 0.65
    learnable_fusion_weight: bool = False

    def __post_init__(self, use_cross_branch_attention: bool | None) -> None:
        if self.spatial_only and self.temporal_only:
            raise ValueError("spatial_only and temporal_only cannot both be enabled")
        if self.fusion_mode not in {"concat", "weighted_prob"}:
            raise ValueError(f"Unsupported fusion_mode={self.fusion_mode}")
        if self.fusion_mode == "weighted_prob" and (self.spatial_only or self.temporal_only):
            raise ValueError("weighted_prob fusion requires both spatial and temporal branches")


class SpatioTemporalDeepfakeDetector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.temporal_only:
            self.spatial_branch: nn.Module = nn.Identity()
            spatial_dim = 0
        else:
            self.spatial_branch = SpatialResNet50(
                pretrained=config.pretrained,
                freeze_backbone=config.freeze_spatial_backbone,
                use_spatial_attention=config.use_spatial_attention,
                use_texture_enhancement=config.use_texture_enhancement,
            )
            spatial_dim = self.spatial_branch.out_dim
        if config.spatial_only:
            self.temporal_branch: nn.Module = nn.Identity()
        else:
            self.temporal_branch = TemporalDiffCNN(
                in_channels=config.temporal_in_channels,
                feature_dim=config.temporal_feature_dim,
                pool_mode=config.temporal_pool,
                dropout=config.dropout,
                use_feature_delta=config.use_feature_delta,
            )
        if config.fusion_mode == "weighted_prob":
            if config.num_classes != 1:
                raise ValueError("weighted_prob fusion currently supports binary classification only")
            self.fusion_head: nn.Module = WeightedProbabilityFusionHead(
                spatial_dim=spatial_dim,
                temporal_dim=config.temporal_feature_dim,
                hidden_dim=config.fusion_hidden_dim,
                spatial_weight=config.fusion_spatial_weight,
                learnable_weight=config.learnable_fusion_weight,
                dropout=config.dropout,
            )
        else:
            self.fusion_head = FusionHead(
                spatial_dim=spatial_dim,
                temporal_dim=0 if config.spatial_only else config.temporal_feature_dim,
                hidden_dim=config.fusion_hidden_dim,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )

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
        if not self.config.spatial_only:
            self._validate_temporal_input(temporal)

        if self.config.temporal_only:
            spatial_feat = None
            spatial_extras: dict[str, torch.Tensor] = {}
        else:
            spatial_feat, spatial_extras = self.spatial_branch(
                spatial,
                return_attention=True,
                return_feature_maps=return_features,
            )
        if self.config.spatial_only:
            temporal_feat = None
            temporal_attn = None
        elif return_features:
            temporal_feat, temporal_attn = self.temporal_branch(
                temporal,
                return_attention=True,
            )
        else:
            temporal_feat = self.temporal_branch(temporal)
            temporal_attn = None

        fusion_extras: dict[str, torch.Tensor] = {}
        if self.config.fusion_mode == "weighted_prob":
            if spatial_feat is None or temporal_feat is None:
                raise RuntimeError("weighted_prob fusion requires both branch features.")
            fusion_output = self.fusion_head(
                spatial_feat,
                temporal_feat,
                return_branch_outputs=return_features,
            )
            if return_features:
                logits, fusion_extras = fusion_output
            else:
                logits = fusion_output
        else:
            classifier_feat = temporal_feat if self.config.temporal_only else spatial_feat
            if classifier_feat is None:
                raise RuntimeError("No feature branch is available for classification.")
            logits = self.fusion_head(classifier_feat, None if self.config.temporal_only else temporal_feat)

        if self.config.num_classes == 1:
            logits = logits.squeeze(1)

        if not return_features:
            return logits

        features: dict[str, torch.Tensor] = {}
        if spatial_feat is not None:
            features["spatial_feat"] = spatial_feat
        if temporal_feat is not None:
            features["temporal_feat"] = temporal_feat
        features.update(spatial_extras)
        features.update(fusion_extras)
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
