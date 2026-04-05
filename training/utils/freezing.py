from __future__ import annotations

from training.spatio_temporal_detector import SpatioTemporalDeepfakeDetector


def set_spatial_branch_trainable(
    model: SpatioTemporalDeepfakeDetector,
    *,
    trainable: bool,
) -> None:
    model.spatial_branch.set_trainable(trainable)


def is_spatial_branch_trainable(model: SpatioTemporalDeepfakeDetector) -> bool:
    return any(param.requires_grad for param in model.spatial_branch.parameters())
