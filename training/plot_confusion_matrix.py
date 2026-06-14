from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a confusion matrix from a test metrics JSON file.")
    parser.add_argument("--metrics-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--threshold-mode",
        choices=["manual", "youden", "f1", "balanced_accuracy", "selected"],
        default="f1",
    )
    parser.add_argument("--negative-label", type=str, default="Real")
    parser.add_argument("--positive-label", type=str, default="Fake")
    parser.add_argument("--title", type=str, default="Cross-domain Confusion Matrix")
    return parser.parse_args()


def resolve_metrics(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    evaluation = payload.get("evaluation") or payload.get("test") or payload
    if mode == "manual":
        return evaluation["manual_threshold"]
    if mode == "selected":
        return evaluation["selected_threshold_metrics"]
    return evaluation["best_thresholds"][mode]["metrics"]


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = resolve_metrics(payload, args.threshold_mode)
    cm = metrics["confusion_matrix"]
    matrix = np.asarray([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    image = ax.imshow(row_percent, cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within actual class")

    labels = [args.positive_label, args.negative_label]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(
        f"{args.title}\n",
        fontsize=13,
    )

    for row_idx in range(2):
        for col_idx in range(2):
            percent = row_percent[row_idx, col_idx]
            text_color = "white" if percent >= 0.55 else "#0f172a"
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:,}\n{percent:.1%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="semibold",
            )

    stem = f"confusion_matrix_{args.threshold_mode}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved confusion matrix: {png_path}")
    print(f"Saved confusion matrix: {pdf_path}")


if __name__ == "__main__":
    main()
