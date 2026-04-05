from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from training import (
    FusionHead,
    ModelConfig,
    SpatialResNet50,
    SpatioTemporalDeepfakeDetector,
    TemporalDiffCNN,
)


class _FakeResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )


def _fake_torchvision_models():
    return types.SimpleNamespace(
        ResNet50_Weights=types.SimpleNamespace(IMAGENET1K_V2=object()),
        resnet50=lambda weights=None: _FakeResNet(),
    )


class SpatioTemporalDetectorTestCase(unittest.TestCase):
    def test_public_exports_are_available(self) -> None:
        self.assertTrue(issubclass(SpatialResNet50, nn.Module))
        self.assertTrue(issubclass(TemporalDiffCNN, nn.Module))
        self.assertTrue(issubclass(FusionHead, nn.Module))
        self.assertTrue(issubclass(SpatioTemporalDeepfakeDetector, nn.Module))

    def test_binary_forward_shapes_and_features(self) -> None:
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(ModelConfig(pretrained=False))

        spatial = torch.randn(2, 3, 224, 224)
        temporal = torch.randn(2, 7, 3, 224, 224)

        logits, features = model(spatial, temporal, return_features=True)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(features["spatial_feat"].shape), (2, 2048))
        self.assertEqual(tuple(features["temporal_feat"].shape), (2, 256))

    def test_multiclass_forward_shape(self) -> None:
        cfg = ModelConfig(num_classes=3, pretrained=False)
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 7, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_attention_pool_runs(self) -> None:
        cfg = ModelConfig(pretrained=False, temporal_pool="attention")
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 7, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2,))

    def test_temporal_frame_count_validation_raises(self) -> None:
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(ModelConfig(pretrained=False, temporal_num_frames=7))

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 224, 224), torch.randn(2, 6, 3, 224, 224))

    def test_frozen_spatial_backbone_disables_grad_and_keeps_batchnorm_eval(self) -> None:
        cfg = ModelConfig(pretrained=False, freeze_spatial_backbone=True)
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        self.assertTrue(all(not param.requires_grad for param in model.spatial_branch.parameters()))
        model.train()
        batch_norms = [module for module in model.spatial_branch.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))

    def test_temporal_branch_accepts_noncontiguous_input(self) -> None:
        branch = TemporalDiffCNN(pool_mode="mean")
        x = torch.randn(2, 3, 7, 32, 32).permute(0, 2, 1, 3, 4)

        output = branch(x)

        self.assertEqual(tuple(output.shape), (2, 256))


if __name__ == "__main__":
    unittest.main()
