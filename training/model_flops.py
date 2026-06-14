from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.temporal_diff_cnn import TEMPORAL_POOL_CHOICES


def format_count(value: float) -> str:
    return f"{int(round(value)):,}"


def format_giga(value: float) -> str:
    return f"{value / 1_000_000_000:.3f}G"


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def module_branch(name: str) -> str:
    if name.startswith("spatial_branch"):
        return "spatial_branch"
    if name.startswith("temporal_branch"):
        return "temporal_branch"
    if name.startswith("fusion_head"):
        return "fusion_head"
    return "other"


@dataclass
class FlopEntry:
    module: str
    op: str
    macs: float
    flops: float


class FlopCounter:
    def __init__(self, *, include_bn: bool = False, include_pooling: bool = False) -> None:
        self.include_bn = include_bn
        self.include_pooling = include_pooling
        self.entries: list[FlopEntry] = []
        self.handles: list[Any] = []

    def add(self, module_name: str, op: str, macs: float, flops: float | None = None) -> None:
        if flops is None:
            flops = 2.0 * macs
        self.entries.append(FlopEntry(module=module_name, op=op, macs=float(macs), flops=float(flops)))

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv2d_hook(name)))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook(name)))
            elif isinstance(module, nn.GRU):
                self.handles.append(module.register_forward_hook(self._gru_hook(name)))
            elif self.include_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.handles.append(module.register_forward_hook(self._batchnorm_hook(name)))
            elif self.include_pooling and isinstance(
                module,
                (
                    nn.AdaptiveAvgPool2d,
                    nn.AvgPool2d,
                    nn.MaxPool2d,
                ),
            ):
                self.handles.append(module.register_forward_hook(self._pooling_hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv2d_hook(self, name: str):
        def hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            batch_size = int(output.shape[0])
            out_channels = int(output.shape[1])
            out_h = int(output.shape[2])
            out_w = int(output.shape[3])
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels
            groups = module.groups
            kernel_mul = (in_channels // groups) * kernel_h * kernel_w
            macs = batch_size * out_channels * out_h * out_w * kernel_mul
            if module.bias is not None:
                macs += batch_size * out_channels * out_h * out_w
            self.add(name, "Conv2d", macs)

        return hook

    def _linear_hook(self, name: str):
        def hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            output_elements = int(output.numel())
            macs = output_elements * module.in_features
            if module.bias is not None:
                macs += output_elements
            self.add(name, "Linear", macs)

        return hook

    def _gru_hook(self, name: str):
        def hook(module: nn.GRU, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                return
            batch_size = int(x.shape[0] if module.batch_first else x.shape[1])
            seq_len = int(x.shape[1] if module.batch_first else x.shape[0])
            hidden_size = int(module.hidden_size)
            num_directions = 2 if module.bidirectional else 1

            macs = 0
            layer_input_size = int(module.input_size)
            for layer_idx in range(module.num_layers):
                current_input_size = layer_input_size if layer_idx == 0 else hidden_size * num_directions
                # GRU has 3 gates. Each gate applies input-hidden and hidden-hidden matrix products.
                macs += batch_size * seq_len * num_directions * 3 * hidden_size * (
                    current_input_size + hidden_size
                )
            self.add(name, "GRU", macs)

        return hook

    def _batchnorm_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.add(name, type(module).__name__, macs=0.0, flops=float(output.numel() * 2))

        return hook

    def _pooling_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.add(name, type(module).__name__, macs=0.0, flops=float(output.numel()))

        return hook

    def totals_by_branch(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = defaultdict(lambda: {"macs": 0.0, "flops": 0.0})
        for entry in self.entries:
            branch = module_branch(entry.module)
            totals[branch]["macs"] += entry.macs
            totals[branch]["flops"] += entry.flops
        return dict(totals)

    def totals_by_op(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = defaultdict(lambda: {"macs": 0.0, "flops": 0.0})
        for entry in self.entries:
            totals[entry.op]["macs"] += entry.macs
            totals[entry.op]["flops"] += entry.flops
        return dict(totals)

    def total_macs(self) -> float:
        return sum(entry.macs for entry in self.entries)

    def total_flops(self) -> float:
        return sum(entry.flops for entry in self.entries)


def load_checkpoint(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_config(args: argparse.Namespace, checkpoint: object | None) -> ModelConfig:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        saved_config = dict(checkpoint["model_config"])
        saved_config["pretrained"] = args.pretrained
        saved_config["freeze_spatial_backbone"] = False
        return ModelConfig(**saved_config)

    return ModelConfig(
        num_classes=args.num_classes,
        temporal_in_channels=3,
        temporal_num_frames=args.clip_len - 1,
        temporal_feature_dim=args.temporal_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_spatial_backbone=False,
        temporal_pool=args.temporal_pool,
        use_spatial_attention=not args.disable_spatial_attention,
        use_texture_enhancement=not args.disable_texture_enhancement,
        use_feature_delta=args.use_feature_delta,
    )


def load_checkpoint_if_requested(model: SpatioTemporalDeepfakeDetector, checkpoint: object | None) -> None:
    if checkpoint is None:
        return
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state.")
    model.load_state_dict(state_dict, strict=True)


def print_totals(title: str, totals: dict[str, dict[str, float]]) -> None:
    print(title)
    print(f"{'Name':<24} {'MACs':>18} {'FLOPs':>18}")
    print("-" * 64)
    for name, values in sorted(totals.items()):
        print(
            f"{name:<24} "
            f"{format_giga(values['macs']):>18} "
            f"{format_giga(values['flops']):>18}"
        )
    print()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FLOPs/MACs for the spatio-temporal detector.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path, e.g. artifacts/.../best.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-len", type=int, default=8, help="Clip length. Temporal frames = clip_len - 1.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input resolution for spatial and temporal tensors.")
    parser.add_argument("--temporal-pool", choices=TEMPORAL_POOL_CHOICES, default="gru")
    parser.add_argument("--temporal-feature-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true", help="Load torchvision ImageNet weights before counting.")
    parser.add_argument("--disable-spatial-attention", action="store_true")
    parser.add_argument("--disable-texture-enhancement", action="store_true")
    parser.add_argument("--use-feature-delta", action="store_true")
    parser.add_argument("--include-bn", action="store_true", help="Include BatchNorm elementwise operations.")
    parser.add_argument("--include-pooling", action="store_true", help="Include pooling elementwise operations.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_len < 2:
        raise ValueError("--clip-len must be at least 2 because temporal input uses frame differences.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size < 32:
        raise ValueError("--image-size should be at least 32 for the ResNet/temporal encoders.")
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint is not None else None
    config = build_model_config(args, checkpoint)
    device = torch.device(args.device)
    model = SpatioTemporalDeepfakeDetector(config).to(device)
    load_checkpoint_if_requested(model, checkpoint)
    model.eval()

    spatial = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    temporal = torch.randn(
        args.batch_size,
        config.temporal_num_frames,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    counter = FlopCounter(include_bn=args.include_bn, include_pooling=args.include_pooling)
    counter.register(model)
    try:
        with torch.no_grad():
            output = model(spatial, temporal)
    finally:
        counter.close()

    params = count_parameters(model)
    print("FLOPs/MACs estimate")
    print("-" * 64)
    print(f"Input spatial:       {tuple(spatial.shape)}")
    print(f"Input temporal:      {tuple(temporal.shape)}")
    print(f"Output:              {tuple(output.shape) if isinstance(output, torch.Tensor) else type(output)}")
    print(f"Temporal pool:       {config.temporal_pool}")
    print(f"Use feature delta:   {config.use_feature_delta}")
    print(f"Parameters:          {format_count(params)} ({params / 1_000_000:.3f}M)")
    print(f"Total MACs:          {format_giga(counter.total_macs())}")
    print(f"Total FLOPs:         {format_giga(counter.total_flops())}")
    print()
    print("Convention: Conv/Linear/GRU FLOPs count multiply and add as 2 operations.")
    print("Functional ops such as cat, interpolate, sigmoid, softmax, ReLU, and tensor multiply are not counted.")
    print()

    print_totals("By branch", counter.totals_by_branch())
    print_totals("By op", counter.totals_by_op())

    if args.output_json is not None:
        payload = {
            "input": {
                "batch_size": args.batch_size,
                "clip_len": args.clip_len,
                "temporal_num_frames": config.temporal_num_frames,
                "image_size": args.image_size,
            },
            "model_config": {
                "temporal_pool": config.temporal_pool,
                "temporal_feature_dim": config.temporal_feature_dim,
                "fusion_hidden_dim": config.fusion_hidden_dim,
                "use_spatial_attention": config.use_spatial_attention,
                "use_texture_enhancement": config.use_texture_enhancement,
                "use_feature_delta": config.use_feature_delta,
            },
            "parameters": params,
            "total_macs": counter.total_macs(),
            "total_flops": counter.total_flops(),
            "by_branch": counter.totals_by_branch(),
            "by_op": counter.totals_by_op(),
            "entries": [entry.__dict__ for entry in counter.entries],
        }
        write_json(Path(args.output_json), payload)
        print(f"Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
