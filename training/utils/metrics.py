from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "spatial": batch["spatial"].to(device, non_blocking=True),
        "temporal": batch["temporal"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "spatial_index": batch["spatial_index"].to(device, non_blocking=True),
        "meta": batch["meta"],
    }


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    unique = np.unique(labels)
    if unique.size >= 2:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auc"] = float("nan")
    return metrics


def finalize_epoch_metrics(
    *,
    running_loss: float,
    num_steps: int,
    all_probs: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    stage_name: str,
) -> dict[str, float]:
    if num_steps == 0 or not all_probs or not all_labels:
        raise RuntimeError(f"No valid {stage_name} batches were produced for this epoch.")

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(np.int64)
    metrics = compute_binary_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / num_steps
    return metrics


def select_checkpoint_metric(val_metrics: dict[str, float]) -> tuple[float, str]:
    val_auc = float(val_metrics["auc"])
    if not math.isnan(val_auc):
        return val_auc, "val_auc"
    return -float(val_metrics["loss"]), "neg_val_loss"


def get_current_lrs(optimizer: AdamW) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]

