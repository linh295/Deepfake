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
from training.fusion_head import WeightedProbabilityFusionHead


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

    def test_spatial_only_skips_temporal_branch(self) -> None:
        cfg = ModelConfig(pretrained=False, spatial_only=True, use_texture_enhancement=False)
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        with mock.patch.object(model.temporal_branch, "forward", wraps=model.temporal_branch.forward) as temporal_forward:
            logits, features = model(
                torch.randn(2, 3, 224, 224),
                torch.empty(2, 0, 3, 1, 1),
                return_features=True,
            )

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertIn("spatial_feat", features)
        self.assertNotIn("temporal_feat", features)
        temporal_forward.assert_not_called()

    def test_temporal_only_skips_spatial_branch(self) -> None:
        model = SpatioTemporalDeepfakeDetector(
            ModelConfig(pretrained=False, temporal_only=True, temporal_pool="gru_attn")
        )

        with mock.patch.object(model.spatial_branch, "forward", wraps=model.spatial_branch.forward) as spatial_forward:
            logits, features = model(
                torch.empty(2, 3, 1, 1),
                torch.randn(2, 7, 3, 32, 32),
                return_features=True,
            )

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertNotIn("spatial_feat", features)
        self.assertIn("temporal_feat", features)
        self.assertIn("temporal_attn", features)
        spatial_forward.assert_not_called()

    def test_single_branch_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(spatial_only=True, temporal_only=True)

    def test_weighted_probability_fusion_matches_probability_ensemble(self) -> None:
        head = WeightedProbabilityFusionHead(
            spatial_dim=2,
            temporal_dim=2,
            hidden_dim=4,
            spatial_weight=0.65,
            dropout=0.0,
        ).eval()
        spatial = torch.randn(3, 2)
        temporal = torch.randn(3, 2)

        with torch.no_grad():
            fused_logit, outputs = head(spatial, temporal, return_branch_outputs=True)

        expected_prob = 0.65 * outputs["spatial_prob"] + 0.35 * outputs["temporal_prob"]
        self.assertTrue(torch.allclose(torch.sigmoid(fused_logit), expected_prob, atol=1e-6))
        self.assertTrue(torch.allclose(outputs["fusion_weights"], torch.tensor([0.65, 0.35]), atol=1e-6))

    def test_detector_weighted_probability_fusion_returns_branch_outputs(self) -> None:
        cfg = ModelConfig(
            pretrained=False,
            use_texture_enhancement=False,
            temporal_pool="mean",
            fusion_mode="weighted_prob",
            fusion_spatial_weight=0.65,
        )
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        logits, features = model(
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 7, 3, 32, 32),
            return_features=True,
        )

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertIn("spatial_prob", features)
        self.assertIn("temporal_prob", features)
        self.assertIn("fusion_weights", features)

    def test_attention_pool_runs(self) -> None:
        cfg = ModelConfig(pretrained=False, temporal_pool="attention")
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 7, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2,))

    def test_detector_does_not_pass_spatial_attention_to_temporal_branch(self) -> None:
        cfg = ModelConfig(
            pretrained=False,
            temporal_pool="mean",
            use_texture_enhancement=False,
            use_cross_branch_attention=True,
        )
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            model = SpatioTemporalDeepfakeDetector(cfg)

        with mock.patch.object(model.temporal_branch, "forward", wraps=model.temporal_branch.forward) as temporal_forward:
            logits, features = model(
                torch.randn(2, 3, 224, 224),
                torch.randn(2, 7, 3, 224, 224),
                return_features=True,
            )

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertIn("spatial_attn", features)
        self.assertNotIn("spatial_attention", temporal_forward.call_args.kwargs)

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

    def test_spatial_backbone_can_freeze_and_unfreeze_at_runtime(self) -> None:
        with mock.patch("training.spatial_resnet50._load_torchvision_models", return_value=_fake_torchvision_models()):
            branch = SpatialResNet50(pretrained=False, freeze_backbone=False)

        self.assertTrue(all(param.requires_grad for param in branch.parameters()))

        branch.set_trainable(False)
        branch.train()
        batch_norms = [module for module in branch.modules() if isinstance(module, nn.BatchNorm2d)]

        self.assertTrue(all(not param.requires_grad for param in branch.parameters()))
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))

        branch.set_trainable(True)
        branch.train()

        self.assertTrue(all(param.requires_grad for param in branch.parameters()))
        self.assertTrue(all(module.training for module in batch_norms))

    def test_temporal_branch_accepts_noncontiguous_input(self) -> None:
        branch = TemporalDiffCNN(pool_mode="mean")
        x = torch.randn(2, 3, 7, 32, 32).permute(0, 2, 1, 3, 4)

        output = branch(x)

        self.assertEqual(tuple(output.shape), (2, 256))

    def test_gru_pooling_variants_return_feature_dim(self) -> None:
        x = torch.randn(2, 7, 3, 32, 32)

        for pool_mode in ("gru", "gru_mean", "gru_max", "gru_mean_max", "gru_attn"):
            with self.subTest(pool_mode=pool_mode):
                branch = TemporalDiffCNN(pool_mode=pool_mode, feature_dim=128, gru_hidden_dim=64)
                output = branch(x)

                self.assertEqual(tuple(output.shape), (2, 128))

    def test_gru_attention_pooling_returns_learned_attention_weights(self) -> None:
        branch = TemporalDiffCNN(pool_mode="gru_attn", feature_dim=128, gru_hidden_dim=64)
        x = torch.randn(2, 7, 3, 32, 32)

        output, attn = branch(x, return_attention=True)

        self.assertEqual(tuple(output.shape), (2, 128))
        self.assertEqual(tuple(attn.shape), (2, 7, 1))
        self.assertTrue(torch.allclose(attn.sum(dim=1), torch.ones(2, 1), atol=1e-5))

    def test_feature_delta_keeps_gru_output_shape(self) -> None:
        x = torch.randn(1, 7, 3, 224, 224)

        for use_feature_delta in (False, True):
            with self.subTest(use_feature_delta=use_feature_delta):
                branch = TemporalDiffCNN(
                    pool_mode="gru_mean",
                    feature_dim=64,
                    gru_hidden_dim=32,
                    use_feature_delta=use_feature_delta,
                )
                output = branch(x)

                self.assertEqual(tuple(output.shape), (1, 64))


if __name__ == "__main__":
    unittest.main()
