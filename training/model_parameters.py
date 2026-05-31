from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn

from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector


def format_count(value: int) -> str:
    return f"{value:,}"


def format_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def load_checkpoint_if_requested(
    model: SpatioTemporalDeepfakeDetector,
    checkpoint: object | None,
) -> None:
    if checkpoint is None:
        return

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def load_checkpoint(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_config(args: argparse.Namespace, checkpoint: object | None) -> ModelConfig:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        saved_config = dict(checkpoint["model_config"])
        saved_config["pretrained"] = args.pretrained
        saved_config["freeze_spatial_backbone"] = args.freeze_spatial_backbone
        return ModelConfig(**saved_config)

    return ModelConfig(
        num_classes=args.num_classes,
        temporal_in_channels=3,
        temporal_num_frames=args.clip_len - 1,
        temporal_feature_dim=args.temporal_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_spatial_backbone=args.freeze_spatial_backbone,
        temporal_pool=args.temporal_pool,
        use_spatial_attention=not args.disable_spatial_attention,
        use_texture_enhancement=not args.disable_texture_enhancement,
        use_cross_branch_attention=not args.disable_cross_branch_attention,
    )


def summarize_module(name: str, module: nn.Module) -> dict[str, int | str]:
    total = count_parameters(module.parameters())
    trainable = count_parameters(param for param in module.parameters() if param.requires_grad)
    return {
        "module": name,
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def print_parameter_summary(model: SpatioTemporalDeepfakeDetector) -> None:
    rows = [
        summarize_module("spatial_branch", model.spatial_branch),
        summarize_module("temporal_branch", model.temporal_branch),
        summarize_module("fusion_head", model.fusion_head),
    ]

    total = count_parameters(model.parameters())
    trainable = count_parameters(param for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    buffers = sum(buffer.numel() for buffer in model.buffers())
    estimated_param_mb = total * 4 / (1024**2)

    headers = ("Module", "Total", "Trainable", "Frozen")
    print(f"{headers[0]:<18} {headers[1]:>16} {headers[2]:>16} {headers[3]:>16}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['module']:<18} "
            f"{format_count(int(row['total'])):>16} "
            f"{format_count(int(row['trainable'])):>16} "
            f"{format_count(int(row['frozen'])):>16}"
        )
    print("-" * 70)
    print(f"{'TOTAL':<18} {format_count(total):>16} {format_count(trainable):>16} {format_count(frozen):>16}")
    print()
    print(f"Total parameters:     {format_count(total)} ({format_millions(total)})")
    print(f"Trainable parameters: {format_count(trainable)} ({format_millions(trainable)})")
    print(f"Frozen parameters:    {format_count(frozen)} ({format_millions(frozen)})")
    print(f"Buffers:              {format_count(buffers)}")
    print(f"Estimated param size: {estimated_param_mb:.2f} MB (float32 only)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count parameters in the spatio-temporal detector.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path, e.g. artifacts/.../best.pt")
    parser.add_argument("--clip-len", type=int, default=8, help="Clip length used by training. Temporal frames = clip_len - 1.")
    parser.add_argument("--temporal-pool", choices=["mean", "attention", "gru"], default="gru")
    parser.add_argument("--temporal-feature-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true", help="Load torchvision ImageNet weights before counting.")
    parser.add_argument("--freeze-spatial-backbone", action="store_true")
    parser.add_argument("--disable-spatial-attention", action="store_true")
    parser.add_argument("--disable-texture-enhancement", action="store_true")
    parser.add_argument("--disable-cross-branch-attention", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_len < 2:
        raise ValueError("--clip-len must be at least 2 because temporal input uses frame differences.")
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint is not None else None
    config = build_model_config(args, checkpoint)
    model = SpatioTemporalDeepfakeDetector(config)
    load_checkpoint_if_requested(model, checkpoint)
    print_parameter_summary(model)


if __name__ == "__main__":
    main()
