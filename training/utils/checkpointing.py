from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.runtime import capture_rng_state


def save_checkpoint(
    path: Path,
    model: SpatioTemporalDeepfakeDetector,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    epoch: int,
    best_val_auc: float,
    best_selection_metric: float,
    cfg: Any,
    model_cfg: ModelConfig,
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
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


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
