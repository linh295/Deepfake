from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from training.spatio_temporal_detector import SpatioTemporalDeepfakeDetector
from training.utils.metrics import (
    ValidationDiagnostics,
    build_validation_diagnostics,
    finalize_epoch_metrics,
    move_batch_to_device,
)

if TYPE_CHECKING:
    from training.train import TrainConfig


_TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


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
            logits = model(batch["spatial"], batch["temporal"])
            loss = criterion(logits, batch["label"])

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

        if step % cfg.log_every == 0 or step == 1:
            pbar.set_postfix(
                loss=f"{running_loss / num_steps:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return finalize_epoch_metrics(
        running_loss=running_loss,
        num_steps=num_steps,
        all_probs=all_probs,
        all_labels=all_labels,
        stage_name="training",
    )


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
    diagnostics = build_validation_diagnostics(
        all_probs=all_probs,
        all_labels=all_labels,
        threshold=0.5,
    )
    return metrics, diagnostics
