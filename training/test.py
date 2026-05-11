from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import autocast
from tqdm import tqdm

from configs.loggings import logger, setup_logging
from dataloader.dataset import ClipDatasetConfig, build_clip_dataloader
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.checkpointing import load_checkpoint
from training.utils.runtime import resolve_device, set_seed


def _build_model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> SpatioTemporalDeepfakeDetector:
    model_config_payload = ckpt.get("model_config")
    if model_config_payload is None:
        raise KeyError(
            "Checkpoint does not contain 'model_config'. "
            "Please use a checkpoint saved by the current training script."
        )

    model_cfg = ModelConfig(**model_config_payload)
    model = SpatioTemporalDeepfakeDetector(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "auc": _safe_auc(labels, probs),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "label_distribution": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
        "prediction_distribution": {
            "negative": int((preds == 0).sum()),
            "positive": int((preds == 1).sum()),
        },
        "probability_summary": {
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
            "mean": float(np.mean(probs)),
            "std": float(np.std(probs)),
        },
    }


def _extract_meta_value(meta: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return str(value)
    return ""


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="Test", dynamic_ncols=True):
        spatial = batch["spatial"].to(device, non_blocking=True)
        temporal = batch["temporal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(spatial, temporal)

        probs = torch.sigmoid(logits.detach())

        all_probs.append(probs.cpu())
        all_labels.append(labels.detach().cpu())

        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(np.int64)

        metas = batch.get("meta", [{} for _ in range(len(labels_np))])
        spatial_indices = batch.get("spatial_index")
        if spatial_indices is not None:
            spatial_indices_np = spatial_indices.detach().cpu().numpy()
        else:
            spatial_indices_np = np.full(len(labels_np), -1)

        for i, meta in enumerate(metas):
            rows.append(
                {
                    "prob_positive": float(probs_np[i]),
                    "label": int(labels_np[i]),
                    "spatial_index": int(spatial_indices_np[i]),
                    "key": _extract_meta_value(meta, "key", "__key__", "sample_key"),
                    "video_id": _extract_meta_value(meta, "video_id", "video", "source_video"),
                    "category": _extract_meta_value(meta, "category", "method", "manipulation"),
                    "label_name": _extract_meta_value(meta, "label", "class_name"),
                }
            )

    if not all_probs or not all_labels:
        raise RuntimeError("No test samples were produced. Please check test shard path.")

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(np.int64)
    return labels, probs, rows


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "key",
        "video_id",
        "category",
        "label_name",
        "label",
        "prob_positive",
        "pred",
        "spatial_index",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            prob = float(row["prob_positive"])
            row_to_write = dict(row)
            row_to_write["pred"] = int(prob >= threshold)
            writer.writerow({key: row_to_write.get(key, "") for key in fieldnames})


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate spatio-temporal deepfake detector on test shards")

    parser.add_argument("--test-shards", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/test_results")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--invert-binary-labels", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: {}", device)
    logger.info("Test shards: {}", args.test_shards)
    logger.info("Checkpoint: {}", args.checkpoint)
    logger.info("Output dir: {}", output_dir)
    logger.info("Invert binary labels: {}", args.invert_binary_labels)
    logger.info("Threshold: {}", args.threshold)

    test_cfg = ClipDatasetConfig(
        shard_pattern=args.test_shards,
        clip_len=args.clip_len,
        invert_binary_labels=args.invert_binary_labels,
        training=False,
        shuffle_buffer=0,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.disable_pin_memory,
        persistent_workers=not args.disable_persistent_workers,
        drop_last=False,
        use_augmentation=False,
        augment_recompute_diff=False,
    )
    test_loader = build_clip_dataloader(test_cfg)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = _build_model_from_checkpoint(ckpt, device)

    checkpoint_meta = {
        "epoch": ckpt.get("epoch"),
        "best_val_auc": ckpt.get("best_val_auc"),
        "best_selection_metric": ckpt.get("best_selection_metric"),
        "train_config": ckpt.get("train_config"),
        "model_config": ckpt.get("model_config"),
        "class_balance": ckpt.get("class_balance"),
    }

    labels, probs, rows = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=not args.disable_amp,
    )

    metrics = compute_metrics(labels, probs, threshold=args.threshold)
    payload = {
        "checkpoint": checkpoint_meta,
        "test": metrics,
    }

    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"

    write_metrics_json(metrics_path, payload)
    write_predictions_csv(predictions_path, rows, threshold=args.threshold)

    logger.info("===== TEST RESULT =====")
    logger.info("AUC: {:.6f}", metrics["auc"])
    logger.info("ACC: {:.6f}", metrics["accuracy"])
    logger.info("Balanced ACC: {:.6f}", metrics["balanced_accuracy"])
    logger.info("F1: {:.6f}", metrics["f1"])
    logger.info("Precision: {:.6f}", metrics["precision"])
    logger.info("Recall: {:.6f}", metrics["recall"])
    logger.info("Confusion matrix: {}", metrics["confusion_matrix"])
    logger.info("Label distribution: {}", metrics["label_distribution"])
    logger.info("Prediction distribution: {}", metrics["prediction_distribution"])
    logger.info("Saved metrics to: {}", metrics_path)
    logger.info("Saved predictions to: {}", predictions_path)


if __name__ == "__main__":
    main()
