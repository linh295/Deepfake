from __future__ import annotations

import torch
import torch.nn as nn


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
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, spatial_feat: torch.Tensor, temporal_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([spatial_feat, temporal_feat], dim=1)
        return self.net(fused)
