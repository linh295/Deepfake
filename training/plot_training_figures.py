from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render individual training figures from history.json.")
    parser.add_argument("--history-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--title-prefix", type=str, default="")
    return parser.parse_args()


def metric_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, float("nan"))))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def save_figure(fig: Any, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def style_axis(ax: Any, title: str, ylabel: str, best_epoch: int, warmup_epochs: int) -> None:
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d9dee6", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if warmup_epochs > 0:
        ax.axvline(warmup_epochs + 0.5, color="#94a3b8", linestyle="--", linewidth=1.2, label="warmup end")
    if best_epoch > 0:
        ax.axvline(best_epoch, color="#ef4444", linestyle=":", linewidth=1.5, label="best epoch")


def infer_best_epoch(val_rows: list[dict[str, Any]]) -> int:
    valid_rows = [
        row
        for row in val_rows
        if math.isfinite(float(row.get("selection_metric", float("nan"))))
    ]
    if not valid_rows:
        return -1
    return int(max(valid_rows, key=lambda row: float(row["selection_metric"]))["epoch"])


def render_train_val_metric(
    *,
    plt: Any,
    epochs: list[int],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    best_epoch: int,
    warmup_epochs: int,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    train_values = metric_series(train_rows, key)
    val_values = metric_series(val_rows, key)
    ax.plot(epochs[: len(train_values)], train_values, marker="o", color="#2563eb", label="train")
    ax.plot(epochs[: len(val_values)], val_values, marker="o", color="#ea580c", label="val")
    style_axis(ax, f"{title_prefix}{title}", ylabel, best_epoch, warmup_epochs)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / key)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_json)
    output_dir = Path(args.output_dir) if args.output_dir else history_path.parent / "individual_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    train_rows = history.get("train", [])
    val_rows = history.get("val", [])
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(val_rows or train_rows)]
    if not epochs:
        raise RuntimeError(f"No epoch history found in {history_path}")

    run_info = history.get("run", {})
    warmup_epochs = int(run_info.get("warmup_epochs", run_info.get("spatial_freeze_warmup_epochs", 0)) or 0)
    best_epoch = infer_best_epoch(val_rows)
    selection_name = str(val_rows[-1].get("selection_metric_name", "selection_metric")) if val_rows else "selection_metric"
    title_prefix = f"{args.title_prefix} | " if args.title_prefix else ""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for key, title, ylabel in (
        ("loss", "Loss", "Loss"),
        ("auc", "ROC AUC", "AUC"),
        ("accuracy", "Accuracy", "Accuracy"),
        ("f1", "F1 Score", "F1"),
    ):
        render_train_val_metric(
            plt=plt,
            epochs=epochs,
            train_rows=train_rows,
            val_rows=val_rows,
            key=key,
            title=title,
            ylabel=ylabel,
            output_dir=output_dir,
            best_epoch=best_epoch,
            warmup_epochs=warmup_epochs,
            title_prefix=title_prefix,
        )

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    selection_values = metric_series(val_rows, "selection_metric")
    ax.plot(epochs[: len(selection_values)], selection_values, marker="o", color="#7c3aed")
    style_axis(
        ax,
        f"{title_prefix}Selection Metric ({selection_name})",
        "Value",
        best_epoch,
        warmup_epochs,
    )
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "selection_metric")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    max_groups = max((len(row.get("learning_rates", [])) for row in val_rows), default=0)
    for group_idx in range(max_groups):
        series = [
            float(row["learning_rates"][group_idx])
            if group_idx < len(row.get("learning_rates", []))
            else float("nan")
            for row in val_rows
        ]
        ax.plot(epochs[: len(series)], series, marker="o", linewidth=1.8, label=f"group_{group_idx}")
    style_axis(ax, f"{title_prefix}Learning Rates", "LR", best_epoch, warmup_epochs)
    if max_groups:
        ax.set_yscale("log")
        ax.legend(frameon=False)
    save_figure(fig, output_dir / "learning_rates")
    plt.close(fig)

    print(f"Saved individual training figures to: {output_dir}")


if __name__ == "__main__":
    main()
