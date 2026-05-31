from __future__ import annotations

from dataclasses import dataclass

from training.spatio_temporal_detector import SpatioTemporalDeepfakeDetector


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    spatial_trainable: bool
    temporal_trainable: bool
    fusion_trainable: bool = True


def set_spatial_branch_trainable(
    model: SpatioTemporalDeepfakeDetector,
    *,
    trainable: bool,
) -> None:
    model.spatial_branch.set_trainable(trainable)


def set_temporal_branch_trainable(
    model: SpatioTemporalDeepfakeDetector,
    *,
    trainable: bool,
) -> None:
    model.temporal_branch.set_trainable(trainable)


def set_fusion_head_trainable(
    model: SpatioTemporalDeepfakeDetector,
    *,
    trainable: bool,
) -> None:
    for param in model.fusion_head.parameters():
        param.requires_grad = trainable


def resolve_alternate_freezing_phase(
    epoch: int,
    *,
    spatial_freeze_warmup_epochs: int,
    temporal_freeze_epochs: int,
) -> TrainingPhase:
    if epoch <= spatial_freeze_warmup_epochs:
        return TrainingPhase(
            name="temporal_warmup",
            spatial_trainable=False,
            temporal_trainable=True,
        )

    spatial_refine_until = spatial_freeze_warmup_epochs + temporal_freeze_epochs
    if epoch <= spatial_refine_until:
        return TrainingPhase(
            name="spatial_refine_temporal_frozen",
            spatial_trainable=True,
            temporal_trainable=False,
        )

    return TrainingPhase(
        name="full_finetune",
        spatial_trainable=True,
        temporal_trainable=True,
    )


def apply_training_phase(
    model: SpatioTemporalDeepfakeDetector,
    phase: TrainingPhase,
) -> None:
    set_spatial_branch_trainable(model, trainable=phase.spatial_trainable)
    set_temporal_branch_trainable(model, trainable=phase.temporal_trainable)
    set_fusion_head_trainable(model, trainable=phase.fusion_trainable)


def is_spatial_branch_trainable(model: SpatioTemporalDeepfakeDetector) -> bool:
    return any(param.requires_grad for param in model.spatial_branch.parameters())


def is_temporal_branch_trainable(model: SpatioTemporalDeepfakeDetector) -> bool:
    return any(param.requires_grad for param in model.temporal_branch.parameters())
