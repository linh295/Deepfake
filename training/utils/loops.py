from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from training.spatio_temporal_detector import SpatioTemporalDeepfakeDetector
from training.utils.metrics import (
    ValidationDiagnostics,
    build_validation_diagnostics,
    compute_binary_metrics,
    finalize_epoch_metrics,
    move_batch_to_device,
)

if TYPE_CHECKING:
    from training.train import TrainConfig


_TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def _forward_training_loss(
    model: SpatioTemporalDeepfakeDetector,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    branch_aux_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if branch_aux_loss_weight <= 0.0:
        logits = model(spatial, temporal)
        loss = criterion(logits, labels)
        return logits, loss, {"fused_loss": float(loss.detach())}

    logits, features = model(spatial, temporal, return_features=True)
    spatial_loss = criterion(features["spatial_logit"].squeeze(1), labels)
    temporal_loss = criterion(features["temporal_logit"].squeeze(1), labels)
    fused_loss = criterion(logits, labels)
    auxiliary_loss = 0.5 * (spatial_loss + temporal_loss)
    loss = fused_loss + branch_aux_loss_weight * auxiliary_loss
    return logits, loss, {
        "fused_loss": float(fused_loss.detach()),
        "spatial_aux_loss": float(spatial_loss.detach()),
        "temporal_aux_loss": float(temporal_loss.detach()),
    }


def train_one_epoch(
    model: SpatioTemporalDeepfakeDetector,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: "TrainConfig",
    total_batches: int | None = None,
    stage_label: str | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    num_steps = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    component_loss_sums: dict[str, float] = {}

    pbar = tqdm(
        loader,
        total=total_batches,
        desc=stage_label or f"Train {epoch}",
        dynamic_ncols=True,
        bar_format=_TQDM_BAR_FORMAT,
    )
    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
            logits, loss, component_losses = _forward_training_loss(
                model=model,
                spatial=batch["spatial"],
                temporal=batch["temporal"],
                labels=batch["label"],
                criterion=criterion,
                branch_aux_loss_weight=float(getattr(cfg, "branch_aux_loss_weight", 0.0)),
            )

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits.detach())
        running_loss += loss.item()
        num_steps += 1
        all_probs.append(probs.cpu())
        all_labels.append(batch["label"].detach().cpu())
        for name, value in component_losses.items():
            component_loss_sums[name] = component_loss_sums.get(name, 0.0) + value

        if step % cfg.log_every == 0 or step == 1:
            pbar.set_postfix(
                loss=f"{running_loss / num_steps:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    metrics = finalize_epoch_metrics(
        running_loss=running_loss,
        num_steps=num_steps,
        all_probs=all_probs,
        all_labels=all_labels,
        stage_name="training",
    )
    metrics.update({name: value / num_steps for name, value in component_loss_sums.items()})
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: SpatioTemporalDeepfakeDetector,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: "TrainConfig",
    total_batches: int | None = None,
    stage_label: str | None = None,
) -> tuple[dict[str, float], ValidationDiagnostics]:
    model.eval()
    running_loss = 0.0
    num_steps = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_spatial_probs: list[torch.Tensor] = []
    all_temporal_probs: list[torch.Tensor] = []

    pbar = tqdm(
        loader,
        total=total_batches,
        desc=stage_label or f"Val {epoch}",
        dynamic_ncols=True,
        bar_format=_TQDM_BAR_FORMAT,
    )
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        with autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
            if getattr(getattr(model, "config", None), "fusion_mode", "concat") == "weighted_prob":
                logits, features = model(batch["spatial"], batch["temporal"], return_features=True)
                all_spatial_probs.append(features["spatial_prob"].detach().cpu())
                all_temporal_probs.append(features["temporal_prob"].detach().cpu())
            else:
                logits = model(batch["spatial"], batch["temporal"])
            loss = criterion(logits, batch["label"])

        probs = torch.sigmoid(logits)
        running_loss += loss.item()
        num_steps += 1
        all_probs.append(probs.cpu())
        all_labels.append(batch["label"].cpu())

    metrics = finalize_epoch_metrics(
        running_loss=running_loss,
        num_steps=num_steps,
        all_probs=all_probs,
        all_labels=all_labels,
        stage_name="validation",
    )
    if all_spatial_probs and all_temporal_probs:
        labels_np = torch.cat(all_labels).numpy().astype(np.int64).reshape(-1)
        spatial_probs_np = torch.cat(all_spatial_probs).numpy().astype(np.float32).reshape(-1)
        temporal_probs_np = torch.cat(all_temporal_probs).numpy().astype(np.float32).reshape(-1)
        metrics.update(
            {
                f"spatial_{name}": value
                for name, value in compute_binary_metrics(labels_np, spatial_probs_np).items()
            }
        )
        metrics.update(
            {
                f"temporal_{name}": value
                for name, value in compute_binary_metrics(labels_np, temporal_probs_np).items()
            }
        )
    diagnostics = build_validation_diagnostics(
        all_probs=all_probs,
        all_labels=all_labels,
        threshold=0.5,
    )
    return metrics, diagnostics
