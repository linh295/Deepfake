from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.model_flops import FlopCounter, count_parameters, format_count, format_giga, write_json
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    config_overrides: dict[str, Any]
    checkpoint_name: str | None = None


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        key="spatial_resnet50_baseline",
        label="Spatial branch (ResNet50 baseline)",
        config_overrides={
            "spatial_only": True,
            "temporal_only": False,
            "fusion_mode": "concat",
            "temporal_pool": "mean",
        },
        checkpoint_name="spatial_only_best.pt",
    ),
    AblationSpec(
        key="temporal_frame_diff_mean",
        label="Temporal branch (Frame Difference + Mean Pooling)",
        config_overrides={
            "spatial_only": False,
            "temporal_only": True,
            "fusion_mode": "concat",
            "temporal_pool": "mean",
        },
        checkpoint_name="temporal_mean_only_best.pt",
    ),
    AblationSpec(
        key="temporal_frame_diff_bigru_mean",
        label="Temporal branch (Frame Difference + BiGRU + Mean Pooling)",
        config_overrides={
            "spatial_only": False,
            "temporal_only": True,
            "fusion_mode": "concat",
            "temporal_pool": "gru_mean",
        },
        checkpoint_name="temporal_gru_mean_only_best.pt",
    ),
    AblationSpec(
        key="decision_level_spatio_temporal",
        label="Decision-level Spatio-Temporal Model",
        config_overrides={
            "spatial_only": False,
            "temporal_only": False,
            "fusion_mode": "weighted_prob",
            "fusion_spatial_weight": 0.5,
            "learnable_fusion_weight": False,
            "temporal_pool": "gru_mean",
        },
        checkpoint_name="decision_level.pt",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FLOPs/MACs for each model ablation.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-len", type=int, default=8, help="Clip length. Temporal frames = clip_len - 1.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--temporal-feature-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true", help="Load torchvision ImageNet weights before counting.")
    parser.add_argument(
        "--enable-spatial-attention",
        action="store_true",
        help="Enable the spatial attention block. Default keeps the ResNet50 baseline plain.",
    )
    parser.add_argument(
        "--enable-texture-enhancement",
        action="store_true",
        help="Enable shallow texture enhancement. Default keeps the ResNet50 baseline plain.",
    )
    parser.add_argument("--use-feature-delta", action="store_true")
    parser.add_argument("--include-bn", action="store_true", help="Include BatchNorm elementwise operations.")
    parser.add_argument("--include-pooling", action="store_true", help="Include pooling elementwise operations.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory with ablation checkpoints. Used only with --load-checkpoints.",
    )
    parser.add_argument(
        "--load-checkpoints",
        action="store_true",
        help="Load matching checkpoint weights before counting. FLOPs are normally independent of weights.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_classes": args.num_classes,
        "temporal_in_channels": 3,
        "temporal_num_frames": args.clip_len - 1,
        "temporal_feature_dim": args.temporal_feature_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "dropout": args.dropout,
        "pretrained": args.pretrained,
        "freeze_spatial_backbone": False,
        "use_spatial_attention": args.enable_spatial_attention,
        "use_texture_enhancement": args.enable_texture_enhancement,
        "use_feature_delta": args.use_feature_delta,
    }


def load_checkpoint_if_requested(
    model: SpatioTemporalDeepfakeDetector,
    spec: AblationSpec,
    args: argparse.Namespace,
) -> str | None:
    if not args.load_checkpoints:
        return None
    if args.checkpoint_dir is None:
        raise ValueError("--load-checkpoints requires --checkpoint-dir.")
    if spec.checkpoint_name is None:
        raise ValueError(f"No checkpoint name configured for ablation: {spec.key}")

    path = Path(args.checkpoint_dir) / spec.checkpoint_name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for {spec.key}: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain model_state: {path}")
    model.load_state_dict(state_dict, strict=True)
    return str(path)


def estimate_ablation(spec: AblationSpec, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config_values = base_config(args)
    config_values.update(spec.config_overrides)
    config = ModelConfig(**config_values)
    model = SpatioTemporalDeepfakeDetector(config).to(device)
    checkpoint_path = load_checkpoint_if_requested(model, spec, args)
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

    return {
        "key": spec.key,
        "label": spec.label,
        "checkpoint": checkpoint_path,
        "input": {
            "batch_size": args.batch_size,
            "clip_len": args.clip_len,
            "temporal_num_frames": config.temporal_num_frames,
            "image_size": args.image_size,
        },
        "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else str(type(output)),
        "model_config": asdict(config),
        "parameters": count_parameters(model),
        "total_macs": counter.total_macs(),
        "total_flops": counter.total_flops(),
        "by_branch": counter.totals_by_branch(),
        "by_op": counter.totals_by_op(),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print("Ablation FLOPs/MACs estimate")
    print("-" * 118)
    print(
        f"{'Ablation':<58} {'Params':>12} {'MACs':>12} {'FLOPs':>12} "
        f"{'Spatial':>12} {'Temporal':>12} {'Fusion':>12}"
    )
    print("-" * 118)
    for row in rows:
        by_branch = row["by_branch"]
        spatial_flops = by_branch.get("spatial_branch", {}).get("flops", 0.0)
        temporal_flops = by_branch.get("temporal_branch", {}).get("flops", 0.0)
        fusion_flops = by_branch.get("fusion_head", {}).get("flops", 0.0)
        print(
            f"{row['label']:<58} "
            f"{format_count(row['parameters']):>12} "
            f"{format_giga(row['total_macs']):>12} "
            f"{format_giga(row['total_flops']):>12} "
            f"{format_giga(spatial_flops):>12} "
            f"{format_giga(temporal_flops):>12} "
            f"{format_giga(fusion_flops):>12}"
        )
    print()
    print("Convention: Conv/Linear/GRU FLOPs count multiply and add as 2 operations.")
    print("Functional ops such as cat, mean/max pooling tensors, sigmoid, softmax, ReLU, and multiply are not counted.")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "label",
        "parameters",
        "total_macs",
        "total_flops",
        "spatial_branch_flops",
        "temporal_branch_flops",
        "fusion_head_flops",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            by_branch = row["by_branch"]
            writer.writerow(
                {
                    "key": row["key"],
                    "label": row["label"],
                    "parameters": row["parameters"],
                    "total_macs": row["total_macs"],
                    "total_flops": row["total_flops"],
                    "spatial_branch_flops": by_branch.get("spatial_branch", {}).get("flops", 0.0),
                    "temporal_branch_flops": by_branch.get("temporal_branch", {}).get("flops", 0.0),
                    "fusion_head_flops": by_branch.get("fusion_head", {}).get("flops", 0.0),
                }
            )


def main() -> None:
    args = parse_args()
    if args.clip_len < 2:
        raise ValueError("--clip-len must be at least 2 because temporal input uses frame differences.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size < 32:
        raise ValueError("--image-size should be at least 32 for the encoders.")

    device = torch.device(args.device)
    rows = [estimate_ablation(spec, args, device) for spec in ABLATIONS]
    print_table(rows)

    payload = {
        "input": {
            "batch_size": args.batch_size,
            "clip_len": args.clip_len,
            "temporal_num_frames": args.clip_len - 1,
            "image_size": args.image_size,
        },
        "include_bn": args.include_bn,
        "include_pooling": args.include_pooling,
        "ablations": rows,
    }
    if args.output_json is not None:
        write_json(Path(args.output_json), payload)
        print(f"Saved JSON report to: {args.output_json}")
    if args.output_csv is not None:
        write_csv(Path(args.output_csv), rows)
        print(f"Saved CSV report to: {args.output_csv}")


if __name__ == "__main__":
    main()
