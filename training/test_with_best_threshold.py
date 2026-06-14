from __future__ import annotations

import argparse
import csv
import glob
import json
import re
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
    roc_curve,
)
from torch.amp import autocast
from tqdm import tqdm

from configs.loggings import logger, setup_logging
from dataloader.dataset import ClipDatasetConfig, build_clip_dataloader
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.utils.checkpointing import load_checkpoint
from training.utils.runtime import resolve_device, set_seed


def _expand_local_shard_pattern(pattern: str) -> list[Path]:
    brace_match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if brace_match:
        start_raw, end_raw = brace_match.groups()
        start = int(start_raw)
        end = int(end_raw)
        width = max(len(start_raw), len(end_raw))
        step = 1 if end >= start else -1
        paths = []
        for idx in range(start, end + step, step):
            expanded = pattern[: brace_match.start()] + f"{idx:0{width}d}" + pattern[brace_match.end() :]
            paths.append(Path(expanded))
        return paths

    globbed = sorted(Path(path) for path in glob.glob(pattern))
    if globbed:
        return globbed
    return [Path(pattern)]


def _nearby_shard_hints(pattern: str) -> list[str]:
    static_prefix = re.split(r"[\{\*\?]", pattern, maxsplit=1)[0]
    base = Path(static_prefix)
    search_root = base if base.is_dir() else base.parent
    if not search_root.exists():
        search_root = search_root.parent
    if not search_root.exists():
        return []

    hints = sorted(str(path) for path in search_root.glob("**/shard-*.tar"))[:20]
    return hints


def validate_local_shards(pattern: str) -> list[Path]:
    paths = _expand_local_shard_pattern(pattern)
    missing = [path for path in paths if not path.is_file()]
    existing = [path for path in paths if path.is_file()]
    if missing:
        hints = _nearby_shard_hints(pattern)
        hint_text = ""
        if hints:
            hint_text = "\nNearby shard files found:\n  " + "\n  ".join(hints)
        raise FileNotFoundError(
            "Test shard pattern does not resolve to existing local files.\n"
            f"Pattern: {pattern}\n"
            f"Missing first file: {missing[0]}\n"
            f"Existing files matched: {len(existing)}/{len(paths)}"
            f"{hint_text}"
        )
    return paths


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
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
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


def find_best_thresholds(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """
    Return several useful thresholds:
    - youden: maximize TPR - FPR
    - f1: maximize F1
    - balanced_accuracy: maximize balanced accuracy

    For imbalanced datasets, threshold=0.5 is often not optimal.
    """
    result: dict[str, Any] = {}

    if np.unique(labels).size < 2:
        return {
            "warning": "Only one class exists in labels; cannot compute ROC-based threshold."
        }

    fpr, tpr, roc_thresholds = roc_curve(labels, probs)

    # roc_curve may include inf as the first threshold.
    finite_mask = np.isfinite(roc_thresholds)
    finite_fpr = fpr[finite_mask]
    finite_tpr = tpr[finite_mask]
    finite_thresholds = roc_thresholds[finite_mask]

    youden_scores = finite_tpr - finite_fpr
    youden_idx = int(np.argmax(youden_scores))
    youden_threshold = float(finite_thresholds[youden_idx])

    # Scan unique predicted probabilities plus a few stable defaults.
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                probs,
                np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
            ]
        )
    )
    candidate_thresholds = candidate_thresholds[
        (candidate_thresholds >= 0.0) & (candidate_thresholds <= 1.0)
    ]

    best_f1 = -1.0
    best_f1_threshold = 0.5
    best_bal_acc = -1.0
    best_bal_acc_threshold = 0.5

    for threshold in candidate_thresholds:
        preds = (probs >= threshold).astype(np.int64)

        f1 = float(f1_score(labels, preds, zero_division=0))
        bal_acc = float(balanced_accuracy_score(labels, preds))

        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = float(threshold)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_bal_acc_threshold = float(threshold)

    result["youden"] = {
        "threshold": youden_threshold,
        "score": float(youden_scores[youden_idx]),
        "metrics": compute_metrics(labels, probs, youden_threshold),
    }
    result["f1"] = {
        "threshold": best_f1_threshold,
        "score": float(best_f1),
        "metrics": compute_metrics(labels, probs, best_f1_threshold),
    }
    result["balanced_accuracy"] = {
        "threshold": best_bal_acc_threshold,
        "score": float(best_bal_acc),
        "metrics": compute_metrics(labels, probs, best_bal_acc_threshold),
    }

    return result


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
    parser.add_argument(
        "--prediction-threshold-mode",
        type=str,
        choices=["manual", "youden", "f1", "balanced_accuracy"],
        default="manual",
        help=(
            "Which threshold to use for test_predictions.csv. "
            "manual uses --threshold. Other modes use thresholds estimated from test probabilities."
        ),
    )

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
    logger.info("Manual threshold: {}", args.threshold)
    logger.info("Prediction threshold mode: {}", args.prediction_threshold_mode)

    resolved_shards = validate_local_shards(args.test_shards)
    logger.info("Resolved test shards: {} files", len(resolved_shards))
    logger.info("First test shard: {}", resolved_shards[0])

    pin_memory = not args.disable_pin_memory and device.type == "cuda"
    persistent_workers = (
        not args.disable_persistent_workers
        and args.num_workers > 0
    )
    if not args.disable_pin_memory and device.type != "cuda":
        logger.info("Pin memory disabled automatically for device={}", device.type)

    test_cfg = ClipDatasetConfig(
        shard_pattern=args.test_shards,
        clip_len=args.clip_len,
        invert_binary_labels=args.invert_binary_labels,
        training=False,
        shuffle_buffer=0,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
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

    manual_metrics = compute_metrics(labels, probs, threshold=args.threshold)
    best_thresholds = find_best_thresholds(labels, probs)

    selected_threshold = args.threshold
    if args.prediction_threshold_mode != "manual":
        selected_threshold = float(best_thresholds[args.prediction_threshold_mode]["threshold"])

    selected_metrics = compute_metrics(labels, probs, threshold=selected_threshold)

    payload = {
        "checkpoint": checkpoint_meta,
        "test": {
            "manual_threshold": manual_metrics,
            "best_thresholds": best_thresholds,
            "selected_threshold_mode": args.prediction_threshold_mode,
            "selected_threshold": selected_threshold,
            "selected_threshold_metrics": selected_metrics,
        },
    }

    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"

    write_metrics_json(metrics_path, payload)
    write_predictions_csv(predictions_path, rows, threshold=selected_threshold)

    logger.info("===== TEST RESULT @ MANUAL THRESHOLD =====")
    logger.info("threshold: {:.6f}", manual_metrics["threshold"])
    logger.info("AUC: {:.6f}", manual_metrics["auc"])
    logger.info("ACC: {:.6f}", manual_metrics["accuracy"])
    logger.info("Balanced ACC: {:.6f}", manual_metrics["balanced_accuracy"])
    logger.info("F1: {:.6f}", manual_metrics["f1"])
    logger.info("Precision: {:.6f}", manual_metrics["precision"])
    logger.info("Recall: {:.6f}", manual_metrics["recall"])
    logger.info("Confusion matrix: {}", manual_metrics["confusion_matrix"])

    logger.info("===== BEST THRESHOLDS =====")
    for name, info in best_thresholds.items():
        if not isinstance(info, dict) or "metrics" not in info:
            logger.info("{}: {}", name, info)
            continue
        m = info["metrics"]
        logger.info(
            "{} | threshold={:.6f} | AUC={:.6f} | ACC={:.6f} | Balanced ACC={:.6f} | F1={:.6f} | Precision={:.6f} | Recall={:.6f} | CM={}",
            name,
            float(info["threshold"]),
            m["auc"],
            m["accuracy"],
            m["balanced_accuracy"],
            m["f1"],
            m["precision"],
            m["recall"],
            m["confusion_matrix"],
        )

    logger.info("===== SELECTED THRESHOLD FOR PREDICTIONS =====")
    logger.info("mode: {}", args.prediction_threshold_mode)
    logger.info("threshold: {:.6f}", selected_threshold)
    logger.info("metrics: {}", selected_metrics)

    logger.info("Saved metrics to: {}", metrics_path)
    logger.info("Saved predictions to: {}", predictions_path)


if __name__ == "__main__":
    main()
