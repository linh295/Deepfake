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
from training.utils.checkpointing import save_checkpoint, write_history
from training.utils.figures import render_training_figures, resolve_figure_output_dirs
from training.utils.freezing import set_spatial_branch_trainable
from training.utils.loops import train_one_epoch, validate_one_epoch
from training.utils.metrics import get_current_lrs, select_checkpoint_metric
from training.utils.progress import build_progress_totals
from training.utils.runtime import resolve_device, set_seed


@dataclass
class TrainConfig:
    train_shards: str
    val_shards: str
    output_dir: str = "artifacts/experiments/st_detector"
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
    temporal_pool: str = "mean"
    use_spatial_attention: bool = True
    use_texture_enhancement: bool = True

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
    spatial_freeze_warmup_epochs: int = 3


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train spatio-temporal deepfake detector")
    parser.add_argument("--train-shards", type=str, required=True, help='e.g. "clip_data/train/shard-*.tar"')
    parser.add_argument("--val-shards", type=str, required=True, help='e.g. "clip_data/val/shard-*.tar"')
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments/st_detector")

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
    parser.add_argument("--temporal-pool", type=str, choices=["mean", "attention"], default="mean")
    parser.add_argument("--disable-spatial-attention", action="store_true")
    parser.add_argument("--disable-texture-enhancement", action="store_true")

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
    parser.add_argument("--spatial-freeze-warmup-epochs", type=int, default=3)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        output_dir=args.output_dir,
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
    )


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
        "Model options | dropout={} | temporal_pool={} | spatial_attention={} | texture_enhancement={}",
        cfg.model_dropout,
        cfg.temporal_pool,
        cfg.use_spatial_attention,
        cfg.use_texture_enhancement,
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
    criterion = build_loss(cfg, device, class_balance_info)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    best_val_auc = float("nan")
    best_selection_metric = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: dict[str, Any] = {
        "run": {
            "class_balance": class_balance_info.as_dict() if class_balance_info is not None else None,
            "use_pos_weight": cfg.use_pos_weight,
            "auto_pos_weight": cfg.auto_pos_weight,
            "model_dropout": cfg.model_dropout,
            "temporal_pool": cfg.temporal_pool,
            "use_spatial_attention": cfg.use_spatial_attention,
            "use_texture_enhancement": cfg.use_texture_enhancement,
        },
        "train": [],
        "val": [],
    }

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

    for epoch in range(1, cfg.epochs + 1):
        freeze_spatial = epoch <= cfg.spatial_freeze_warmup_epochs
        phase_name = "temporal_warmup" if freeze_spatial else "full_finetune"
        set_spatial_branch_trainable(model, trainable=not freeze_spatial)

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
        scheduler.step(current_selection_metric)

        current_val_auc = float(val_metrics["auc"])
        if not math.isnan(current_val_auc):
            if math.isnan(best_val_auc) or current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
        current_lrs = get_current_lrs(optimizer)

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append(
            {
                "epoch": epoch,
                **val_metrics,
                "selection_metric": current_selection_metric,
                "selection_metric_name": selection_metric_name,
                "learning_rates": current_lrs,
                "phase": phase_name,
            }
        )

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch {} | time={:.1f}s | train_loss={:.4f} train_auc={:.4f} train_acc={:.4f} | "
            "val_loss={:.4f} val_auc={:.4f} val_acc={:.4f} | phase={} | {}={:.4f} | lrs={}",
            epoch,
            elapsed,
            train_metrics["loss"],
            train_metrics["auc"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
            phase_name,
            selection_metric_name,
            current_selection_metric,
            current_lrs,
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
