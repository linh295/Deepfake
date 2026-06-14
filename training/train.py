from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler

from configs.loggings import logger, setup_logging
from configs.settings import settings
from training.utils.builders import (
    build_dataloaders,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)
from training.utils.class_balance import build_class_balance_info
from training.utils.checkpointing import (
    initialize_weighted_fusion_from_branch_checkpoints,
    load_checkpoint,
    read_history,
    save_checkpoint,
    write_history,
)
from training.utils.figures import render_training_figures, resolve_figure_output_dirs
from training.utils.loops import train_one_epoch, validate_one_epoch
from training.utils.metrics import get_current_lrs, select_checkpoint_metric
from training.utils.progress import build_progress_totals
from training.utils.runtime import resolve_device, restore_rng_state, set_seed
from training.temporal_diff_cnn import TEMPORAL_POOL_CHOICES


@dataclass
class TrainConfig:
    train_shards: str
    val_shards: str
    output_dir: str = "artifacts/experiments/st_detector"
    resume_checkpoint: str | None = None
    init_spatial_checkpoint: str | None = None
    init_temporal_checkpoint: str | None = None
    epochs: int = 30
    batch_size: int = 12
    num_workers: int = 4

    lr_spatial: float = 1e-5
    lr_temporal: float = 1e-4
    lr_fusion: float = 1e-4
    weight_decay: float = 1e-4

    clip_len: int = 8
    invert_binary_labels: bool = False
    use_pos_weight: bool = True
    auto_pos_weight: bool = True
    max_pos_weight: float | None = None

    model_dropout: float = 0.3
    temporal_pool: str = "gru"
    use_spatial_attention: bool = True
    use_texture_enhancement: bool = True
    use_feature_delta: bool = False
    spatial_only: bool = False
    temporal_only: bool = False
    fusion_mode: str = "concat"
    fusion_spatial_weight: float = 0.65
    learnable_fusion_weight: bool = False
    branch_aux_loss_weight: float = 0.0

    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    save_every: int = 1
    log_every: int = 20
    pin_memory: bool = True
    persistent_workers: bool = True
    train_shuffle_buffer: int = 1000
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.1
    scheduler_patience: int = 2
    scheduler_threshold: float = 1e-4
    scheduler_min_lr: float = 0.0
    spatial_freeze_warmup_epochs: int = 0
    warmup_epochs: int = 0
    use_augmentation: bool = False
    augment_recompute_diff: bool = True
    hflip_prob: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    
    loss_type: str = "bce"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    
    jpeg_prob: float = 0.0
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 95

    blur_prob: float = 0.0
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train spatio-temporal deepfake detector")
    parser.add_argument("--train-shards", type=str, required=True, help='e.g. "clip_data/train/shard-*.tar"')
    parser.add_argument("--val-shards", type=str, required=True, help='e.g. "clip_data/val/shard-*.tar"')
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments/st_detector")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--init-spatial-checkpoint", type=str, default=None)
    parser.add_argument("--init-temporal-checkpoint", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr-spatial", type=float, default=1e-5)
    parser.add_argument("--lr-temporal", type=float, default=5e-5)
    parser.add_argument("--lr-fusion", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--invert-binary-labels", action="store_true")
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--disable-auto-pos-weight", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=None)

    parser.add_argument("--model-dropout", type=float, default=0.3)
    parser.add_argument("--temporal-pool", type=str, choices=TEMPORAL_POOL_CHOICES, default="gru")
    parser.add_argument("--disable-spatial-attention", action="store_true")
    parser.add_argument("--disable-texture-enhancement", action="store_true")
    parser.add_argument("--use-feature-delta", action="store_true")
    parser.add_argument(
        "--spatial-only",
        action="store_true",
        help="Run a spatial-only ablation: skip temporal_branch and classify from spatial features only.",
    )
    parser.add_argument(
        "--temporal-only",
        action="store_true",
        help="Run a temporal-only ablation: skip spatial_branch and classify from temporal features only.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["concat", "weighted_prob"],
        default="concat",
        help="Feature concatenation or weighted late fusion of branch probabilities.",
    )
    parser.add_argument(
        "--fusion-spatial-weight",
        type=float,
        default=0.65,
        help="Initial/fixed spatial probability weight used by weighted_prob fusion.",
    )
    parser.add_argument(
        "--learnable-fusion-weight",
        action="store_true",
        help="Allow weighted_prob fusion weights to be optimized instead of remaining fixed.",
    )
    parser.add_argument(
        "--branch-aux-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the mean spatial/temporal auxiliary classification loss.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--scheduler-min-lr", type=float, default=0.0)
    parser.add_argument(
        "--spatial-freeze-warmup-epochs",
        type=int,
        default=0,
        help="Freeze spatial_branch for the first N epochs so temporal_branch and fusion_head train first.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm all optimizer parameter-group LRs from 0 to target over the first N epochs.",
    )
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--disable-augment-recompute-diff", action="store_true")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.10)
    parser.add_argument("--contrast", type=float, default=0.10)
    parser.add_argument("--loss-type", type=str, choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--jpeg-prob", type=float, default=0.0)
    parser.add_argument("--jpeg-quality-min", type=int, default=70)
    parser.add_argument("--jpeg-quality-max", type=int, default=95)

    parser.add_argument("--blur-prob", type=float, default=0.0)
    parser.add_argument("--blur-sigma-min", type=float, default=0.1)
    parser.add_argument("--blur-sigma-max", type=float, default=1.0)
    args = parser.parse_args()
    if args.spatial_only and args.temporal_only:
        parser.error("--spatial-only and --temporal-only cannot be used together")
    if args.fusion_mode == "weighted_prob" and (args.spatial_only or args.temporal_only):
        parser.error("--fusion-mode weighted_prob requires both spatial and temporal branches")
    if not 0.0 < args.fusion_spatial_weight < 1.0:
        parser.error("--fusion-spatial-weight must be strictly between 0 and 1")
    if args.branch_aux_loss_weight < 0.0:
        parser.error("--branch-aux-loss-weight must be greater than or equal to 0")
    if args.branch_aux_loss_weight > 0.0 and args.fusion_mode != "weighted_prob":
        parser.error("--branch-aux-loss-weight requires --fusion-mode weighted_prob")
    if bool(args.init_spatial_checkpoint) != bool(args.init_temporal_checkpoint):
        parser.error("--init-spatial-checkpoint and --init-temporal-checkpoint must be provided together")
    if args.init_spatial_checkpoint and args.fusion_mode != "weighted_prob":
        parser.error("Branch checkpoint initialization requires --fusion-mode weighted_prob")
    if args.resume_checkpoint and args.init_spatial_checkpoint:
        parser.error("--resume-checkpoint cannot be combined with branch checkpoint initialization")

    return TrainConfig(
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        init_spatial_checkpoint=args.init_spatial_checkpoint,
        init_temporal_checkpoint=args.init_temporal_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_spatial=args.lr_spatial,
        lr_temporal=args.lr_temporal,
        lr_fusion=args.lr_fusion,
        weight_decay=args.weight_decay,
        clip_len=args.clip_len,
        invert_binary_labels=args.invert_binary_labels,
        use_pos_weight=not args.disable_pos_weight,
        auto_pos_weight=not args.disable_auto_pos_weight,
        max_pos_weight=args.max_pos_weight,
        model_dropout=args.model_dropout,
        temporal_pool=args.temporal_pool,
        use_spatial_attention=not args.disable_spatial_attention,
        use_texture_enhancement=not args.disable_texture_enhancement,
        use_feature_delta=args.use_feature_delta,
        spatial_only=args.spatial_only,
        temporal_only=args.temporal_only,
        fusion_mode=args.fusion_mode,
        fusion_spatial_weight=args.fusion_spatial_weight,
        learnable_fusion_weight=args.learnable_fusion_weight,
        branch_aux_loss_weight=args.branch_aux_loss_weight,
        seed=args.seed,
        device=args.device,
        use_amp=not args.disable_amp,
        grad_clip_norm=args.grad_clip_norm,
        save_every=args.save_every,
        log_every=args.log_every,
        pin_memory=not args.disable_pin_memory,
        persistent_workers=not args.disable_persistent_workers,
        train_shuffle_buffer=args.train_shuffle_buffer,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_threshold=args.scheduler_threshold,
        scheduler_min_lr=args.scheduler_min_lr,
        spatial_freeze_warmup_epochs=args.spatial_freeze_warmup_epochs,
        warmup_epochs=args.warmup_epochs,
        use_augmentation=args.use_augmentation,
        augment_recompute_diff=not args.disable_augment_recompute_diff,
        hflip_prob=args.hflip_prob,
        brightness=args.brightness,
        contrast=args.contrast,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        jpeg_prob=args.jpeg_prob,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        blur_prob=args.blur_prob,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
    )


def apply_spatial_warmup_freeze(model: torch.nn.Module, epoch: int, warmup_epochs: int) -> str:
    spatial_branch = getattr(model, "spatial_branch", None)
    if spatial_branch is None or not hasattr(spatial_branch, "freeze") or not hasattr(spatial_branch, "unfreeze"):
        return "full_train"

    if warmup_epochs <= 0:
        spatial_branch.unfreeze()
        return "full_train"

    if epoch <= warmup_epochs:
        spatial_branch.freeze()
        return "temporal_warmup_spatial_frozen"

    spatial_branch.unfreeze()
    return "full_train"


def main() -> None:
    setup_logging()
    cfg = parse_args()
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dirs = resolve_figure_output_dirs(output_dir=output_dir, figure_root=settings.FIGURE_DIR)

    logger.info("Device: {}", device)
    logger.info("Output dir: {}", output_dir)
    logger.info("Figure dir: {}", figure_dirs.run_dir)
    logger.info("Train shards: {}", cfg.train_shards)
    logger.info("Val shards: {}", cfg.val_shards)
    logger.info("Invert binary labels: {}", cfg.invert_binary_labels)
    logger.info(
        "Model options | dropout={} | temporal_pool={} | spatial_attention={} | texture_enhancement={} | feature_delta={} | spatial_only={} | temporal_only={} | fusion_mode={} | fusion_spatial_weight={} | learnable_fusion_weight={} | branch_aux_loss_weight={} | spatial_freeze_warmup_epochs={} | warmup_epochs={}",
        cfg.model_dropout,
        cfg.temporal_pool,
        cfg.use_spatial_attention,
        cfg.use_texture_enhancement,
        cfg.use_feature_delta,
        cfg.spatial_only,
        cfg.temporal_only,
        cfg.fusion_mode,
        cfg.fusion_spatial_weight,
        cfg.learnable_fusion_weight,
        cfg.branch_aux_loss_weight,
        cfg.spatial_freeze_warmup_epochs,
        cfg.warmup_epochs,
    )

    train_loader, val_loader = build_dataloaders(cfg)
    progress_totals = build_progress_totals(
        train_shards=cfg.train_shards,
        val_shards=cfg.val_shards,
        batch_size=cfg.batch_size,
    )

    class_balance_info = None
    if cfg.use_pos_weight and cfg.auto_pos_weight:
        class_balance_info = build_class_balance_info(
            shard_pattern=cfg.train_shards,
            invert_binary_labels=cfg.invert_binary_labels,
            max_pos_weight=cfg.max_pos_weight,
        )

    model, model_cfg = build_model(cfg, device)
    if cfg.init_spatial_checkpoint is not None and cfg.init_temporal_checkpoint is not None:
        init_stats = initialize_weighted_fusion_from_branch_checkpoints(
            model=model,
            spatial_checkpoint=cfg.init_spatial_checkpoint,
            temporal_checkpoint=cfg.init_temporal_checkpoint,
            map_location=device,
        )
        logger.info(
            "Initialized weighted fusion from branch checkpoints | spatial={} | temporal={} | loaded_parameters={} | uninitialized_parameters={}",
            cfg.init_spatial_checkpoint,
            cfg.init_temporal_checkpoint,
            init_stats["loaded_parameters"],
            init_stats["uninitialized_parameters"],
        )
    criterion = build_loss(cfg, device, class_balance_info)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    best_val_auc = float("nan")
    best_selection_metric = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    start_epoch = 1
    history: dict[str, Any] = {
        "run": {
            "class_balance": class_balance_info.as_dict() if class_balance_info is not None else None,
            "use_pos_weight": cfg.use_pos_weight,
            "auto_pos_weight": cfg.auto_pos_weight,
            "model_dropout": cfg.model_dropout,
            "temporal_pool": cfg.temporal_pool,
            "use_spatial_attention": cfg.use_spatial_attention,
            "use_texture_enhancement": cfg.use_texture_enhancement,
            "use_feature_delta": cfg.use_feature_delta,
            "spatial_only": cfg.spatial_only,
            "temporal_only": cfg.temporal_only,
            "fusion_mode": cfg.fusion_mode,
            "fusion_spatial_weight": cfg.fusion_spatial_weight,
            "learnable_fusion_weight": cfg.learnable_fusion_weight,
            "branch_aux_loss_weight": cfg.branch_aux_loss_weight,
            "init_spatial_checkpoint": cfg.init_spatial_checkpoint,
            "init_temporal_checkpoint": cfg.init_temporal_checkpoint,
            "spatial_freeze_warmup_epochs": cfg.spatial_freeze_warmup_epochs,
            "warmup_epochs": cfg.warmup_epochs,
        },
        "train": [],
        "val": [],
    }

    if cfg.resume_checkpoint is not None:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_auc = float(checkpoint.get("best_val_auc", best_val_auc))
        best_selection_metric = float(checkpoint.get("best_selection_metric", best_selection_metric))
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = read_history(history_path)
            history.setdefault("run", {})
            history.setdefault("train", [])
            history.setdefault("val", [])
            history["run"].update(
                {
                    "class_balance": class_balance_info.as_dict() if class_balance_info is not None else None,
                    "use_pos_weight": cfg.use_pos_weight,
                    "auto_pos_weight": cfg.auto_pos_weight,
                    "model_dropout": cfg.model_dropout,
                    "temporal_pool": cfg.temporal_pool,
                    "use_spatial_attention": cfg.use_spatial_attention,
                    "use_texture_enhancement": cfg.use_texture_enhancement,
                    "use_feature_delta": cfg.use_feature_delta,
                    "spatial_only": cfg.spatial_only,
                    "temporal_only": cfg.temporal_only,
                    "fusion_mode": cfg.fusion_mode,
                    "fusion_spatial_weight": cfg.fusion_spatial_weight,
                    "learnable_fusion_weight": cfg.learnable_fusion_weight,
                    "branch_aux_loss_weight": cfg.branch_aux_loss_weight,
                    "init_spatial_checkpoint": cfg.init_spatial_checkpoint,
                    "init_temporal_checkpoint": cfg.init_temporal_checkpoint,
                    "spatial_freeze_warmup_epochs": cfg.spatial_freeze_warmup_epochs,
                    "warmup_epochs": cfg.warmup_epochs,
                }
            )
            if history["val"]:
                best_epoch = int(max(history["val"], key=lambda row: float(row.get("selection_metric", -math.inf))).get("epoch", -1))
        logger.info(
            "Resumed checkpoint {} | start_epoch={} | best_val_auc={} | best_selection_metric={:.4f}",
            resume_path,
            start_epoch,
            best_val_auc,
            best_selection_metric,
        )

    if class_balance_info is not None:
        logger.info(
            "Class balance | positive_class={} | positive_count={} | negative_count={} | raw_pos_weight={} | effective_pos_weight={}",
            class_balance_info.positive_class_name,
            class_balance_info.positive_count,
            class_balance_info.negative_count,
            class_balance_info.raw_pos_weight,
            class_balance_info.effective_pos_weight,
        )
        if class_balance_info.effective_pos_weight is None:
            logger.warning(
                "Auto pos_weight could not be derived safely from train shards; falling back to unweighted BCEWithLogitsLoss."
            )

    for epoch in range(start_epoch, cfg.epochs + 1):
        phase_name = apply_spatial_warmup_freeze(
            model,
            epoch=epoch,
            warmup_epochs=cfg.spatial_freeze_warmup_epochs,
        )
        logger.info("Epoch {} phase: {}", epoch, phase_name)
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            cfg,
            total_batches=progress_totals["train"],
            stage_label=f"Train {epoch}/{cfg.epochs}",
        )
        val_metrics, val_diagnostics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            cfg,
            total_batches=progress_totals["val"],
            stage_label=f"Val {epoch}/{cfg.epochs}",
        )

        current_selection_metric, selection_metric_name = select_checkpoint_metric(val_metrics)
        current_lrs = get_current_lrs(optimizer)
        scheduler_phase = scheduler.step(current_selection_metric, epoch=epoch)

        current_val_auc = float(val_metrics["auc"])
        if not math.isnan(current_val_auc):
            if math.isnan(best_val_auc) or current_val_auc > best_val_auc:
                best_val_auc = current_val_auc

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append(
            {
                "epoch": epoch,
                **val_metrics,
                "selection_metric": current_selection_metric,
                "selection_metric_name": selection_metric_name,
                "learning_rates": current_lrs,
                "phase": phase_name,
                "scheduler_phase": scheduler_phase,
            }
        )

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch {} | time={:.1f}s | train_loss={:.4f} train_auc={:.4f} train_acc={:.4f} | "
            "val_loss={:.4f} val_auc={:.4f} val_acc={:.4f} | phase={} | scheduler={} | {}={:.4f} | lrs={}",
            epoch,
            elapsed,
            train_metrics["loss"],
            train_metrics["auc"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
            phase_name,
            scheduler_phase,
            selection_metric_name,
            current_selection_metric,
            current_lrs,
        )
        if cfg.fusion_mode == "weighted_prob":
            logger.info(
                "Epoch {} branch validation | spatial_auc={:.4f} | temporal_auc={:.4f}",
                epoch,
                val_metrics["spatial_auc"],
                val_metrics["temporal_auc"],
            )

        is_new_best = current_selection_metric > best_selection_metric
        should_stop = False

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_auc,
                best_selection_metric,
                cfg,
                model_cfg,
                class_balance_info.as_dict() if class_balance_info is not None else None,
            )

        if is_new_best:
            best_selection_metric = current_selection_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_auc,
                best_selection_metric,
                cfg,
                model_cfg,
                class_balance_info.as_dict() if class_balance_info is not None else None,
            )
            logger.info(
                "Saved new best checkpoint at epoch {} with {}={:.4f}",
                epoch,
                selection_metric_name,
                best_selection_metric,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                should_stop = True
                logger.info(
                    "Early stopping triggered after {} epochs without improvement. "
                    "Best epoch={} best_selection_metric={:.4f}",
                    epochs_without_improvement,
                    best_epoch,
                    best_selection_metric,
                )

        write_history(output_dir / "history.json", history)

        try:
            render_training_figures(
                history=history,
                diagnostics=val_diagnostics,
                class_balance_info=class_balance_info.as_dict() if class_balance_info is not None else None,
                current_epoch=epoch,
                best_epoch=best_epoch,
                selection_metric_name=selection_metric_name,
                figure_dir=figure_dirs.latest_dir,
                warmup_epochs=cfg.spatial_freeze_warmup_epochs,
                latest_bundle=True,
                invert_binary_labels=cfg.invert_binary_labels,
            )
        except Exception as exc:
            logger.warning("Failed to render latest training figures at epoch {} ({}).", epoch, exc)

        if is_new_best:
            try:
                render_training_figures(
                    history=history,
                    diagnostics=val_diagnostics,
                    class_balance_info=class_balance_info.as_dict() if class_balance_info is not None else None,
                    current_epoch=epoch,
                    best_epoch=best_epoch,
                    selection_metric_name=selection_metric_name,
                    figure_dir=figure_dirs.best_root_dir / f"epoch_{epoch:03d}",
                    warmup_epochs=cfg.spatial_freeze_warmup_epochs,
                    latest_bundle=False,
                    invert_binary_labels=cfg.invert_binary_labels,
                )
            except Exception as exc:
                logger.warning("Failed to render best-epoch training figures at epoch {} ({}).", epoch, exc)

        if should_stop:
            break

    logger.info(
        "Training finished. Best epoch={} | best_val_auc={} | best_selection_metric={:.4f}",
        best_epoch,
        best_val_auc,
        best_selection_metric,
    )


if __name__ == "__main__":
    main()
