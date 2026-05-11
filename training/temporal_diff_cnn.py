from typing import Literal

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

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
        feature_dim: int = 512,
        pool_mode: Literal["mean", "attention"] = "mean",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if pool_mode not in {"mean", "attention"}:
            raise ValueError(f"Unsupported pool_mode={pool_mode}")

        self.pool_mode = pool_mode
        self.feature_dim = feature_dim

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2), # Output channel là 512
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim), # Khớp với output của encoder
            nn.BatchNorm1d(feature_dim),
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

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = self.proj(x)
        x = x.reshape(b, t, self.feature_dim)

        if self.pool_mode == "mean":
            pooled = x.mean(dim=1)
            if return_attention:
                uniform_attn = torch.full(
                    (b, t, 1),
                    fill_value=1.0 / max(1, t),
                    dtype=x.dtype,
                    device=x.device,
                )
                return pooled, uniform_attn
            return pooled

        attn_logits = self.attention(x)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        if return_attention:
            return pooled, attn_weights
        return pooled
