from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class TemporalDiffCNN(nn.Module):
    """
    Temporal branch for frame differences.

    Input shape:
        [B, T, C, H, W]
    Output shape:
        [B, feature_dim]
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        pool_mode: Literal["mean", "attention"] = "mean",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if pool_mode not in {"mean", "attention"}:
            raise ValueError(f"Unsupported pool_mode={pool_mode}")

        self.pool_mode = pool_mode
        self.feature_dim = feature_dim

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.pool_mode == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.Tanh(),
                nn.Linear(feature_dim // 2, 1),
            )
        else:
            self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = self.proj(x)
        x = x.reshape(b, t, self.feature_dim)

        if self.pool_mode == "mean":
            return x.mean(dim=1)

        attn_logits = self.attention(x)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (x * attn_weights).sum(dim=1)
