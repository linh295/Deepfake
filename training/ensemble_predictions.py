from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Late-fuse spatial-only and temporal-only prediction CSV files."
    )
    parser.add_argument("--spatial-predictions", type=str, required=True)
    parser.add_argument("--temporal-predictions", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight-start", type=float, default=0.5)
    parser.add_argument("--weight-end", type=float, default=0.95)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        choices=["auc", "accuracy", "f1"],
        default="auc",
        help="Metric used to identify the best spatial weight in this sweep.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.weight_start <= 1.0:
        parser.error("--weight-start must be in [0, 1]")
    if not 0.0 <= args.weight_end <= 1.0:
        parser.error("--weight-end must be in [0, 1]")
    if args.weight_end < args.weight_start:
        parser.error("--weight-end must be greater than or equal to --weight-start")
    if args.weight_step <= 0.0:
        parser.error("--weight-step must be greater than 0")
    return args


def read_predictions(path: Path, branch_name: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{branch_name} prediction file not found: {path}")

    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"key", "label", "prob_positive"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            key = str(row["key"]).strip()
            if not key:
                raise ValueError(f"{path}:{row_number} has an empty key")
            if key in rows:
                raise ValueError(f"{path} contains duplicate key: {key}")
            rows[key] = {
                **row,
                "label": int(row["label"]),
                "prob_positive": float(row["prob_positive"]),
            }

    if not rows:
        raise RuntimeError(f"No predictions found in {path}")
    return rows


def align_predictions(
    spatial_rows: dict[str, dict[str, Any]],
    temporal_rows: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    spatial_keys = set(spatial_rows)
    temporal_keys = set(temporal_rows)
    if spatial_keys != temporal_keys:
        spatial_only = sorted(spatial_keys - temporal_keys)[:5]
        temporal_only = sorted(temporal_keys - spatial_keys)[:5]
        raise ValueError(
            "Prediction files do not contain identical sample keys. "
            f"Only spatial examples={spatial_only}; only temporal examples={temporal_only}"
        )

    keys = sorted(spatial_keys)
    labels = np.asarray([spatial_rows[key]["label"] for key in keys], dtype=np.int64)
    temporal_labels = np.asarray([temporal_rows[key]["label"] for key in keys], dtype=np.int64)
    if not np.array_equal(labels, temporal_labels):
        mismatch_index = int(np.flatnonzero(labels != temporal_labels)[0])
        raise ValueError(f"Label mismatch for key={keys[mismatch_index]}")

    spatial_probs = np.asarray([spatial_rows[key]["prob_positive"] for key in keys], dtype=np.float64)
    temporal_probs = np.asarray([temporal_rows[key]["prob_positive"] for key in keys], dtype=np.float64)
    return keys, labels, spatial_probs, temporal_probs


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    auc = float(roc_auc_score(labels, probs)) if np.unique(labels).size >= 2 else float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def build_weights(start: float, end: float, step: float) -> np.ndarray:
    count = int(np.floor((end - start) / step + 1e-9)) + 1
    weights = start + np.arange(count, dtype=np.float64) * step
    if weights[-1] < end - 1e-9:
        weights = np.append(weights, end)
    return np.clip(weights, 0.0, 1.0)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spatial_path = Path(args.spatial_predictions)
    temporal_path = Path(args.temporal_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_rows = read_predictions(spatial_path, "spatial")
    temporal_rows = read_predictions(temporal_path, "temporal")
    keys, labels, spatial_probs, temporal_probs = align_predictions(spatial_rows, temporal_rows)

    sweep_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_probs: np.ndarray | None = None
    for spatial_weight in build_weights(args.weight_start, args.weight_end, args.weight_step):
        temporal_weight = 1.0 - spatial_weight
        ensemble_probs = spatial_weight * spatial_probs + temporal_weight * temporal_probs
        metrics = compute_metrics(labels, ensemble_probs, args.threshold)
        row = {
            "spatial_weight": float(spatial_weight),
            "temporal_weight": float(temporal_weight),
            "threshold": float(args.threshold),
            **metrics,
        }
        sweep_rows.append(row)
        if best_row is None or float(row[args.selection_metric]) > float(best_row[args.selection_metric]):
            best_row = row
            best_probs = ensemble_probs

    assert best_row is not None and best_probs is not None
    best_predictions: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        source = spatial_rows[key]
        best_predictions.append(
            {
                "key": key,
                "video_id": source.get("video_id", ""),
                "category": source.get("category", ""),
                "label": int(labels[index]),
                "spatial_prob": float(spatial_probs[index]),
                "temporal_prob": float(temporal_probs[index]),
                "ensemble_prob": float(best_probs[index]),
                "ensemble_pred": int(best_probs[index] >= args.threshold),
            }
        )

    write_csv(
        output_dir / "ensemble_sweep.csv",
        ["spatial_weight", "temporal_weight", "threshold", "auc", "accuracy", "f1"],
        sweep_rows,
    )
    write_csv(
        output_dir / "best_ensemble_predictions.csv",
        [
            "key",
            "video_id",
            "category",
            "label",
            "spatial_prob",
            "temporal_prob",
            "ensemble_prob",
            "ensemble_pred",
        ],
        best_predictions,
    )

    summary = {
        "spatial_predictions": str(spatial_path),
        "temporal_predictions": str(temporal_path),
        "num_samples": len(keys),
        "selection_metric": args.selection_metric,
        "threshold": float(args.threshold),
        "spatial_only_metrics": compute_metrics(labels, spatial_probs, args.threshold),
        "temporal_only_metrics": compute_metrics(labels, temporal_probs, args.threshold),
        "best_ensemble": best_row,
        "warning": (
            "Choose the spatial weight on a validation set, then apply that fixed weight to the test set. "
            "Selecting the best weight directly on a test set leaks test labels."
        ),
    }
    with (output_dir / "ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        "Best ensemble | spatial_weight={:.2f} | temporal_weight={:.2f} | "
        "AUC={:.6f} | ACC={:.6f} | F1={:.6f}".format(
            best_row["spatial_weight"],
            best_row["temporal_weight"],
            best_row["auc"],
            best_row["accuracy"],
            best_row["f1"],
        )
    )
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
