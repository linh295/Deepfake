from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Temporal branch for frame differences optimized with Bidirectional GRU 
    to capture long-range temporal logic under low FPS (5 FPS).

    Input shape:
        [B, T, C, H, W]
    Output shape:
        [B, feature_dim]
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        pool_mode: Literal["mean", "attention", "gru"] = "gru",  # Đặt mặc định là gru
        dropout: float = 0.3,
        gru_hidden_dim: int = 256,
        gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.freeze_branch = False
        # Mở rộng hỗ trợ pool_mode để tránh lỗi tương thích nếu config truyền vào
        if pool_mode not in {"mean", "attention", "gru"}:
            raise ValueError(f"Unsupported pool_mode={pool_mode}")

        self.pool_mode = pool_mode
        self.feature_dim = feature_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers

        # Backbone trích xuất đặc trưng từng khung hình (Giữ nguyên cấu trúc của bạn)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Bộ chiếu đặc trưng không gian (Giữ nguyên cấu trúc của bạn)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Cấu hình nhánh tuần tự thời gian dựa trên chế độ pool_mode
        if self.pool_mode == "gru":
            # Khởi tạo Bidirectional GRU đa tầng
            self.gru = nn.GRU(
                input_size=feature_dim,
                hidden_size=gru_hidden_dim,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            # Lớp tuyến tính gộp output 2 chiều (gru_hidden_dim * 2) về lại kích thước feature_dim cho Fusion
            self.gru_proj = nn.Sequential(
                nn.Linear(gru_hidden_dim * 2, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.attention = None
        elif self.pool_mode == "attention":
            self.gru = None
            self.gru_proj = None
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.Tanh(),
                nn.Linear(feature_dim // 2, 1),
            )
        else:
            self.gru = None
            self.gru_proj = None
            self.attention = None

    def set_trainable(self, trainable: bool) -> None:
        self.freeze_branch = not trainable
        for param in self.parameters():
            param.requires_grad = trainable
        if self.freeze_branch:
            self.eval()

    def freeze(self) -> None:
        self.set_trainable(False)

    def unfreeze(self) -> None:
        self.set_trainable(True)

    def train(self, mode: bool = True) -> "TemporalDiffCNN":
        if self.freeze_branch:
            super().train(False)
            return self
        super().train(mode)
        return self

    def _apply_spatial_attention_gate(
        self,
        x: torch.Tensor,
        spatial_attention: torch.Tensor | None,
        *,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        if spatial_attention is None:
            return x
        if spatial_attention.ndim != 4 or spatial_attention.shape[0] != batch_size:
            raise ValueError(
                "Expected spatial_attention with shape [B, 1, H, W], "
                f"got {tuple(spatial_attention.shape)}"
            )

        attn = spatial_attention.to(device=x.device, dtype=x.dtype)
        attn = F.interpolate(
            attn,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attn = attn.repeat_interleave(num_frames, dim=0)
        return x * attn

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        spatial_attention: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        
        # 1. Trích xuất đặc trưng không gian trên từng frame độc lập
        x = x.reshape(b * t, c, h, w)
        for layer in self.frame_encoder[:-1]:
            x = layer(x)
        x = self._apply_spatial_attention_gate(
            x,
            spatial_attention,
            batch_size=b,
            num_frames=t,
        )
        x = self.frame_encoder[-1](x)
        x = self.proj(x)
        x = x.reshape(b, t, self.feature_dim) # Shape: [B, T, feature_dim]

        # ---- CHẾ ĐỘ GRU (MỚI TỐI ƯU HÓA) ----
        if self.pool_mode == "gru":
            # gru_out shape: [B, T, gru_hidden_dim * 2]
            # hn shape: [num_layers * 2, B, gru_hidden_dim]
            gru_out, hn = self.gru(x)
            
            # Trích xuất trạng thái ẩn cuối cùng của cả 2 chiều:
            # Chiều xuôi (Forward): Lấy bước thời gian cuối cùng của nửa đầu hidden_dim -> gru_out[:, -1, :gru_hidden_dim]
            # Chiều ngược (Backward): Lấy bước thời gian đầu tiên của nửa sau hidden_dim -> gru_out[:, 0, gru_hidden_dim:]
            # Thay vì trích xuất thủ công phức tạp từ gru_out, cấu trúc hn cho phép ta lấy trực tiếp:
            # hn[-2] là tầng cuối cùng của chiều xuôi, hn[-1] là tầng cuối cùng của chiều ngược.
            
            feat_forward = hn[-2]   # Shape: [B, gru_hidden_dim]
            feat_backward = hn[-1]  # Shape: [B, gru_hidden_dim]
            
            # Khôi phục liên kết không gian bằng cách ghép cặp đặc trưng 2 chiều
            pooled = torch.cat([feat_forward, feat_backward], dim=-1) # Shape: [B, gru_hidden_dim * 2]
            pooled = self.gru_proj(pooled) # Hạ chiều về lại [B, feature_dim]

            if return_attention:
                # Trả về ma trận trọng số đồng nhất làm dummy data để không gây lỗi crash luồng code cũ
                dummy_attn = torch.full((b, t, 1), fill_value=1.0 / max(1, t), dtype=x.dtype, device=x.device)
                return pooled, dummy_attn
            return pooled

        # ---- CÁC CHẾ ĐỘ CŨ (ĐỂ BẢO TOÀN COMPATIBILITY) ----
        if self.pool_mode == "mean":
            pooled = x.mean(dim=1)
            if return_attention:
                uniform_attn = torch.full((b, t, 1), fill_value=1.0 / max(1, t), dtype=x.dtype, device=x.device)
                return pooled, uniform_attn
            return pooled

        # Chế độ Attention cũ
        attn_logits = self.attention(x)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        if return_attention:
            return pooled, attn_weights
        return pooled
