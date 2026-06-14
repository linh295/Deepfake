from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from torch.optim import AdamW

from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.runtime import capture_rng_state


def save_checkpoint(
    path: Path,
    model: SpatioTemporalDeepfakeDetector,
    optimizer: AdamW,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    best_val_auc: float,
    best_selection_metric: float,
    cfg: Any,
    model_cfg: ModelConfig,
    class_balance_info: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_auc": best_val_auc,
            "best_selection_metric": best_selection_metric,
            "train_config": asdict(cfg),
            "model_config": asdict(model_cfg),
            "class_balance": class_balance_info,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def initialize_weighted_fusion_from_branch_checkpoints(
    model: SpatioTemporalDeepfakeDetector,
    spatial_checkpoint: Path | str,
    temporal_checkpoint: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, int]:
    if model.config.fusion_mode != "weighted_prob":
        raise ValueError("Branch checkpoint initialization requires fusion_mode=weighted_prob")

    spatial_payload = load_checkpoint(spatial_checkpoint, map_location=map_location)
    temporal_payload = load_checkpoint(temporal_checkpoint, map_location=map_location)
    spatial_config = spatial_payload.get("model_config", {})
    temporal_config = temporal_payload.get("model_config", {})
    if not spatial_config.get("spatial_only", False):
        raise ValueError(f"Expected a spatial-only checkpoint: {spatial_checkpoint}")
    if not temporal_config.get("temporal_only", False):
        raise ValueError(f"Expected a temporal-only checkpoint: {temporal_checkpoint}")

    target_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    mappings = (
        (spatial_payload["model_state"], "spatial_branch.", "spatial_branch."),
        (spatial_payload["model_state"], "fusion_head.net.", "fusion_head.spatial_head."),
        (temporal_payload["model_state"], "temporal_branch.", "temporal_branch."),
        (temporal_payload["model_state"], "fusion_head.net.", "fusion_head.temporal_head."),
    )
    for source_state, source_prefix, target_prefix in mappings:
        for source_key, value in source_state.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key not in target_state:
                raise KeyError(f"Checkpoint parameter has no matching target parameter: {target_key}")
            if target_state[target_key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {target_key}: "
                    f"target={tuple(target_state[target_key].shape)} source={tuple(value.shape)}"
                )
            loaded[target_key] = value

    if not loaded:
        raise RuntimeError("No branch parameters were loaded from the supplied checkpoints")
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    unexpected = list(unexpected)
    if unexpected:
        raise RuntimeError(f"Unexpected parameters while initializing branches: {unexpected}")
    return {
        "loaded_parameters": len(loaded),
        "uninitialized_parameters": len(missing),
    }


def read_history(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {"train": [], "val": []}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("train", [])
    payload.setdefault("val", [])
    return payload


def write_history(path: Path, history: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
