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
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone

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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
            self._set_frozen_batchnorm_eval()

    def _set_frozen_batchnorm_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "SpatialResNet50":
        super().train(mode)
        if self.freeze_backbone:
            self._set_frozen_batchnorm_eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
