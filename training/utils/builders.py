from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.dataset import ClipDatasetConfig, build_clip_dataloader
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.class_balance import ClassBalanceInfo

if TYPE_CHECKING:
    from training.train import TrainConfig


def build_dataloaders(cfg: "TrainConfig") -> tuple[Any, Any]:
    train_ds_cfg = ClipDatasetConfig(
        shard_pattern=cfg.train_shards,
        clip_len=cfg.clip_len,
        invert_binary_labels=cfg.invert_binary_labels,
        training=True,
        shuffle_buffer=cfg.train_shuffle_buffer,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=False,
    )
    val_ds_cfg = ClipDatasetConfig(
        shard_pattern=cfg.val_shards,
        clip_len=cfg.clip_len,
        invert_binary_labels=cfg.invert_binary_labels,
        training=False,
        shuffle_buffer=0,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=False,
    )
    train_loader = build_clip_dataloader(train_ds_cfg)
    val_loader = build_clip_dataloader(val_ds_cfg)
    return train_loader, val_loader


def build_model_config(cfg: "TrainConfig") -> ModelConfig:
    return ModelConfig(
        num_classes=1,
        temporal_in_channels=3,
        temporal_num_frames=cfg.clip_len - 1,
        temporal_feature_dim=256,
        fusion_hidden_dim=512,
        dropout=cfg.model_dropout,
        pretrained=True,
        freeze_spatial_backbone=False,
        temporal_pool=cfg.temporal_pool,
        use_spatial_attention=cfg.use_spatial_attention,
        use_texture_enhancement=cfg.use_texture_enhancement,
    )


def build_model(
    cfg: "TrainConfig",
    device: torch.device,
) -> tuple[SpatioTemporalDeepfakeDetector, ModelConfig]:
    model_cfg = build_model_config(cfg)
    model = SpatioTemporalDeepfakeDetector(model_cfg)
    model.to(device)
    return model, model_cfg


def build_loss(
    cfg: "TrainConfig",
    device: torch.device,
    class_balance_info: ClassBalanceInfo | None = None,
) -> nn.Module:
    if (
        cfg.use_pos_weight
        and cfg.auto_pos_weight
        and class_balance_info is not None
        and class_balance_info.effective_pos_weight is not None
    ):
        pos_weight = torch.tensor(
            [class_balance_info.effective_pos_weight],
            dtype=torch.float32,
            device=device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def build_optimizer(model: SpatioTemporalDeepfakeDetector, cfg: "TrainConfig") -> AdamW:
    return AdamW(
        [
            {"params": model.spatial_branch.parameters(), "lr": cfg.lr_spatial},
            {"params": model.temporal_branch.parameters(), "lr": cfg.lr_temporal},
            {"params": model.fusion_head.parameters(), "lr": cfg.lr_fusion},
        ],
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer: AdamW, cfg: "TrainConfig") -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.scheduler_threshold,
        min_lr=cfg.scheduler_min_lr,
    )
