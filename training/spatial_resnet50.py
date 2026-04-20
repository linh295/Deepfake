from __future__ import annotations

import torch
import torch.nn as nn


def _load_torchvision_models():
    try:
        import torchvision.models as tvm
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for SpatialResNet50. "
            "Add torchvision to the project dependencies before using this model."
        ) from exc
    return tvm


class SpatialResNet50(nn.Module):
    """
    ResNet50 spatial branch with:
    1) texture enhancement from shallow features
    2) spatial attention on the fused feature map

    Output:
        pooled feature vector [B, 2048]
    Optional:
        return attention maps / feature maps for visualization and debugging
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_spatial_attention: bool = True,
        use_texture_enhancement: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_backbone = False
        self.use_spatial_attention = use_spatial_attention
        self.use_texture_enhancement = use_texture_enhancement

        tvm = _load_torchvision_models()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Option 3: texture enhancement via shallow feature projection + fusion
        self.texture_proj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048

        self.set_trainable(not freeze_backbone)

    def _set_frozen_batchnorm_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def set_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for param in self.parameters():
            param.requires_grad = trainable
        if self.freeze_backbone:
            self._set_frozen_batchnorm_eval()

    def freeze(self) -> None:
        self.set_trainable(False)

    def unfreeze(self) -> None:
        self.set_trainable(True)

    def train(self, mode: bool = True) -> "SpatialResNet50":
        super().train(mode)
        if self.freeze_backbone:
            self._set_frozen_batchnorm_eval()
        return self

    def _fuse_texture(
        self,
        shallow_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_texture_enhancement:
            return semantic_feat, None

        texture_feat = self.texture_proj(shallow_feat)
        texture_feat = torch.nn.functional.interpolate(
            texture_feat,
            size=semantic_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([semantic_feat, texture_feat], dim=1)
        fused = self.fusion_conv(fused)
        return fused, texture_feat

    def _apply_spatial_attention(
        self,
        feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_spatial_attention:
            return feat, None

        attn = torch.sigmoid(self.attn_conv(feat))
        attended = feat * attn
        return attended, attn

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_feature_maps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.stem(x)
        shallow_feat = self.layer1(x)
        x = self.layer2(shallow_feat)
        x = self.layer3(x)
        semantic_feat = self.layer4(x)

        fused_feat, texture_feat = self._fuse_texture(shallow_feat, semantic_feat)
        attended_feat, spatial_attn = self._apply_spatial_attention(fused_feat)

        pooled = self.pool(attended_feat)
        pooled = torch.flatten(pooled, 1)

        if not (return_attention or return_feature_maps):
            return pooled

        extras: dict[str, torch.Tensor] = {}
        if spatial_attn is not None:
            extras["spatial_attn"] = spatial_attn
        if return_feature_maps:
            extras["shallow_feat"] = shallow_feat
            extras["semantic_feat"] = semantic_feat
            extras["fused_feat"] = fused_feat
            if texture_feat is not None:
                extras["texture_feat"] = texture_feat
            extras["attended_feat"] = attended_feat

        return pooled, extras
