from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.checkpointing import (
    initialize_weighted_fusion_from_branch_checkpoints,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine spatial-only and temporal-only checkpoints using fixed probability fusion."
    )
    parser.add_argument("--spatial-checkpoint", type=str, required=True)
    parser.add_argument("--temporal-checkpoint", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, required=True)
    parser.add_argument("--spatial-weight", type=float, default=0.65)
    args = parser.parse_args()
    if not 0.0 < args.spatial_weight < 1.0:
        parser.error("--spatial-weight must be strictly between 0 and 1")
    return args


def build_combined_config(
    spatial_payload: dict[str, Any],
    temporal_payload: dict[str, Any],
    spatial_weight: float,
) -> ModelConfig:
    spatial_cfg = spatial_payload["model_config"]
    temporal_cfg = temporal_payload["model_config"]
    if spatial_cfg["fusion_hidden_dim"] != temporal_cfg["fusion_hidden_dim"]:
        raise ValueError("Spatial and temporal checkpoints use different fusion_hidden_dim values")

    return ModelConfig(
        num_classes=1,
        temporal_in_channels=temporal_cfg["temporal_in_channels"],
        temporal_num_frames=temporal_cfg["temporal_num_frames"],
        temporal_feature_dim=temporal_cfg["temporal_feature_dim"],
        fusion_hidden_dim=spatial_cfg["fusion_hidden_dim"],
        dropout=spatial_cfg["dropout"],
        pretrained=False,
        freeze_spatial_backbone=False,
        temporal_pool=temporal_cfg["temporal_pool"],
        use_spatial_attention=spatial_cfg["use_spatial_attention"],
        use_texture_enhancement=spatial_cfg["use_texture_enhancement"],
        use_feature_delta=temporal_cfg.get("use_feature_delta", False),
        fusion_mode="weighted_prob",
        fusion_spatial_weight=spatial_weight,
        learnable_fusion_weight=False,
    )


def main() -> None:
    args = parse_args()
    spatial_path = Path(args.spatial_checkpoint)
    temporal_path = Path(args.temporal_checkpoint)
    output_path = Path(args.output_checkpoint)

    spatial_payload = load_checkpoint(spatial_path)
    temporal_payload = load_checkpoint(temporal_path)
    model_cfg = build_combined_config(spatial_payload, temporal_payload, args.spatial_weight)
    model = SpatioTemporalDeepfakeDetector(model_cfg)
    init_stats = initialize_weighted_fusion_from_branch_checkpoints(
        model=model,
        spatial_checkpoint=spatial_path,
        temporal_checkpoint=temporal_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "fusion_sources": {
                "spatial_checkpoint": str(spatial_path),
                "temporal_checkpoint": str(temporal_path),
                "spatial_weight": float(args.spatial_weight),
                "temporal_weight": float(1.0 - args.spatial_weight),
                **init_stats,
            },
        },
        output_path,
    )
    print(
        f"Saved weighted fusion checkpoint: {output_path} "
        f"(spatial={args.spatial_weight:.2f}, temporal={1.0 - args.spatial_weight:.2f})"
    )


if __name__ == "__main__":
    main()
