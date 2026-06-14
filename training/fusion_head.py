from __future__ import annotations

import torch
import torch.nn as nn


def _build_classifier(
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, num_classes),
    )


class FusionHead(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = spatial_dim + temporal_dim
        self.net = _build_classifier(in_dim, hidden_dim, num_classes, dropout)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = spatial_feat if temporal_feat is None else torch.cat([spatial_feat, temporal_feat], dim=1)
        return self.net(fused)


class WeightedProbabilityFusionHead(nn.Module):
    """Late fusion that preserves independently useful spatial and temporal decisions."""

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        spatial_weight: float = 0.65,
        learnable_weight: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 < spatial_weight < 1.0:
            raise ValueError("spatial_weight must be strictly between 0 and 1")

        self.spatial_head = _build_classifier(spatial_dim, hidden_dim, 1, dropout)
        self.temporal_head = _build_classifier(temporal_dim, hidden_dim, 1, dropout)
        initial_logits = torch.log(torch.tensor([spatial_weight, 1.0 - spatial_weight]))
        if learnable_weight:
            self.weight_logits = nn.Parameter(initial_logits)
        else:
            self.register_buffer("weight_logits", initial_logits)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.weight_logits, dim=0)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
        return_branch_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spatial_logit = self.spatial_head(spatial_feat)
        temporal_logit = self.temporal_head(temporal_feat)
        spatial_prob = torch.sigmoid(spatial_logit)
        temporal_prob = torch.sigmoid(temporal_logit)
        weights = self.weights()
        fused_prob = weights[0] * spatial_prob + weights[1] * temporal_prob
        dtype_info = torch.finfo(fused_prob.dtype)
        fused_logit = torch.logit(fused_prob.clamp(min=dtype_info.tiny, max=1.0 - dtype_info.eps))

        if not return_branch_outputs:
            return fused_logit
        return fused_logit, {
            "spatial_logit": spatial_logit,
            "temporal_logit": temporal_logit,
            "spatial_prob": spatial_prob,
            "temporal_prob": temporal_prob,
            "fusion_weights": weights,
        }
