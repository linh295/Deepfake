from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from dataloader.dataset import ClipDatasetConfig, build_clip_dataloader
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.class_balance import ClassBalanceInfo
from training.utils.losses import BinaryFocalLossWithLogits

if TYPE_CHECKING:
    from training.train import TrainConfig


def linear_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


@dataclass
class TrainingSchedulers:
    plateau_scheduler: ReduceLROnPlateau
    warmup_scheduler: LambdaLR | None
    warmup_epochs: int

    def step(self, metric: float, epoch: int) -> str:
        if self.warmup_scheduler is not None and epoch <= self.warmup_epochs:
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            return "linear_warmup"

        self.plateau_scheduler.step(metric)
        return "plateau"

    def state_dict(self) -> dict[str, Any]:
        return {
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict() if self.warmup_scheduler is not None else None,
            "warmup_epochs": self.warmup_epochs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "plateau_scheduler" not in state_dict:
            self.plateau_scheduler.load_state_dict(state_dict)
            return

        self.plateau_scheduler.load_state_dict(state_dict["plateau_scheduler"])
        warmup_state = state_dict.get("warmup_scheduler")
        if self.warmup_scheduler is not None and warmup_state is not None:
            self.warmup_scheduler.load_state_dict(warmup_state)


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
        use_augmentation=cfg.use_augmentation,
        augment_recompute_diff=cfg.augment_recompute_diff,
        hflip_prob=cfg.hflip_prob,
        brightness=cfg.brightness,
        contrast=cfg.contrast,
        jpeg_prob=cfg.jpeg_prob,
        jpeg_quality_min=cfg.jpeg_quality_min,
        jpeg_quality_max=cfg.jpeg_quality_max,
        blur_prob=cfg.blur_prob,
        blur_sigma_min=cfg.blur_sigma_min,
        blur_sigma_max=cfg.blur_sigma_max,
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
        use_augmentation=False,
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
        use_feature_delta=cfg.use_feature_delta,
        spatial_only=cfg.spatial_only,
        temporal_only=cfg.temporal_only,
        fusion_mode=cfg.fusion_mode,
        fusion_spatial_weight=cfg.fusion_spatial_weight,
        learnable_fusion_weight=cfg.learnable_fusion_weight,
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
    if cfg.loss_type == "focal":
        return BinaryFocalLossWithLogits(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            reduction="mean",
        )

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
    parameter_groups = []
    if not model.config.temporal_only:
        parameter_groups.append({"params": model.spatial_branch.parameters(), "lr": cfg.lr_spatial})
    if not model.config.spatial_only:
        parameter_groups.append({"params": model.temporal_branch.parameters(), "lr": cfg.lr_temporal})
    parameter_groups.append({"params": model.fusion_head.parameters(), "lr": cfg.lr_fusion})
    return AdamW(parameter_groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: AdamW, cfg: "TrainConfig") -> TrainingSchedulers:
    warmup_epochs = max(0, int(cfg.warmup_epochs))
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: linear_warmup_factor(epoch, warmup_epochs),
        )

    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.scheduler_threshold,
        min_lr=cfg.scheduler_min_lr,
    )
    return TrainingSchedulers(
        plateau_scheduler=plateau_scheduler,
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
    )
