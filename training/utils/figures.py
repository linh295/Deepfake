from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

from configs.settings import settings
from training.utils.metrics import ValidationDiagnostics


_PYPLOT = None


@dataclass(frozen=True)
class FigureOutputDirs:
    run_dir: Path
    latest_dir: Path
    best_root_dir: Path


def _load_pyplot():
    global _PYPLOT
    if _PYPLOT is not None:
        return _PYPLOT
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for training figure rendering. Install project dependencies before training."
        ) from exc
    _PYPLOT = plt
    return _PYPLOT


def resolve_figure_output_dirs(
    *,
    output_dir: Path | str,
    figure_root: Path | str | None = None,
) -> FigureOutputDirs:
    root = Path(figure_root) if figure_root is not None else Path(settings.FIGURE_DIR)
    run_name = Path(output_dir).name or "default_run"
    run_dir = root / run_name
    latest_dir = run_dir / "latest"
    best_root_dir = run_dir / "best"

    latest_dir.mkdir(parents=True, exist_ok=True)
    best_root_dir.mkdir(parents=True, exist_ok=True)
    return FigureOutputDirs(run_dir=run_dir, latest_dir=latest_dir, best_root_dir=best_root_dir)


def render_training_figures(
    *,
    history: dict[str, Any],
    diagnostics: ValidationDiagnostics,
    class_balance_info: dict[str, Any] | None,
    current_epoch: int,
    best_epoch: int,
    selection_metric_name: str,
    figure_dir: Path | str,
    warmup_epochs: int,
    latest_bundle: bool,
) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _render_training_dashboard(
        history=history,
        class_balance_info=class_balance_info,
        current_epoch=current_epoch,
        best_epoch=best_epoch,
        selection_metric_name=selection_metric_name,
        warmup_epochs=warmup_epochs,
        figure_dir=figure_dir,
    )
    _render_validation_curves(
        diagnostics=diagnostics,
        figure_dir=figure_dir,
        latest_bundle=latest_bundle,
        current_epoch=current_epoch,
    )
    _render_validation_confusion(
        diagnostics=diagnostics,
        figure_dir=figure_dir,
        latest_bundle=latest_bundle,
        current_epoch=current_epoch,
    )
    _render_validation_score_distribution(
        diagnostics=diagnostics,
        figure_dir=figure_dir,
        latest_bundle=latest_bundle,
        current_epoch=current_epoch,
    )


def _figure_summary_text(
    *,
    class_balance_info: dict[str, Any] | None,
    selection_metric_name: str,
    best_epoch: int,
    warmup_epochs: int,
) -> str:
    fragments = [
        f"selection={selection_metric_name}",
        f"best_epoch={best_epoch if best_epoch > 0 else 'n/a'}",
        f"warmup_epochs={warmup_epochs}",
    ]
    if class_balance_info is not None:
        fragments.append(
            "positive_class={positive_class} pos={positive_count} neg={negative_count} pos_weight={pos_weight}".format(
                positive_class=class_balance_info.get("positive_class_name", "unknown"),
                positive_count=class_balance_info.get("positive_count", "n/a"),
                negative_count=class_balance_info.get("negative_count", "n/a"),
                pos_weight=class_balance_info.get("effective_pos_weight", "n/a"),
            )
        )
    return " | ".join(fragments)


def _style_axes(ax, *, title: str, xlabel: str = "Epoch", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#d9dee6", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_phase_markers(ax, *, best_epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs > 0:
        ax.axvline(
            warmup_epochs + 0.5,
            color="#94a3b8",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
    if best_epoch > 0:
        ax.axvline(
            best_epoch,
            color="#ef4444",
            linestyle=":",
            linewidth=1.3,
            alpha=0.95,
        )


def _save_figure(fig, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.clf()


def _metric_series(history_rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in history_rows:
        value = row.get(key, float("nan"))
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def _render_training_dashboard(
    *,
    history: dict[str, Any],
    class_balance_info: dict[str, Any] | None,
    current_epoch: int,
    best_epoch: int,
    selection_metric_name: str,
    warmup_epochs: int,
    figure_dir: Path,
) -> None:
    plt = _load_pyplot()
    plt.style.use("default")

    train_rows = history.get("train", [])
    val_rows = history.get("val", [])
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(val_rows or train_rows)]
    if not epochs:
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    fig.patch.set_facecolor("white")
    colors = {
        "train": "#2563eb",
        "val": "#ea580c",
        "selection": "#7c3aed",
    }

    train_loss = _metric_series(train_rows, "loss")
    val_loss = _metric_series(val_rows, "loss")
    axes[0, 0].plot(epochs[: len(train_loss)], train_loss, marker="o", color=colors["train"], label="train")
    axes[0, 0].plot(epochs[: len(val_loss)], val_loss, marker="o", color=colors["val"], label="val")
    _style_axes(axes[0, 0], title="Loss", ylabel="Loss")

    train_auc = _metric_series(train_rows, "auc")
    val_auc = _metric_series(val_rows, "auc")
    axes[0, 1].plot(epochs[: len(train_auc)], train_auc, marker="o", color=colors["train"], label="train")
    axes[0, 1].plot(epochs[: len(val_auc)], val_auc, marker="o", color=colors["val"], label="val")
    _style_axes(axes[0, 1], title="ROC AUC", ylabel="AUC")

    train_acc = _metric_series(train_rows, "accuracy")
    val_acc = _metric_series(val_rows, "accuracy")
    axes[1, 0].plot(epochs[: len(train_acc)], train_acc, marker="o", color=colors["train"], label="train")
    axes[1, 0].plot(epochs[: len(val_acc)], val_acc, marker="o", color=colors["val"], label="val")
    _style_axes(axes[1, 0], title="Accuracy", ylabel="Accuracy")

    train_f1 = _metric_series(train_rows, "f1")
    val_f1 = _metric_series(val_rows, "f1")
    axes[1, 1].plot(epochs[: len(train_f1)], train_f1, marker="o", color=colors["train"], label="train")
    axes[1, 1].plot(epochs[: len(val_f1)], val_f1, marker="o", color=colors["val"], label="val")
    _style_axes(axes[1, 1], title="F1 Score", ylabel="F1")

    selection_values = _metric_series(val_rows, "selection_metric")
    axes[2, 0].plot(epochs[: len(selection_values)], selection_values, marker="o", color=colors["selection"])
    _style_axes(axes[2, 0], title=f"Selection Metric ({selection_metric_name})", ylabel="Value")

    lr_axes = axes[2, 1]
    max_groups = max((len(row.get("learning_rates", [])) for row in val_rows), default=0)
    if max_groups == 0:
        lr_axes.text(0.5, 0.5, "Learning-rate history unavailable", ha="center", va="center", fontsize=11)
        lr_axes.set_xticks([])
        lr_axes.set_yticks([])
        _style_axes(lr_axes, title="Learning Rates", ylabel="LR")
    else:
        lr_values_flat: list[float] = []
        for group_idx in range(max_groups):
            series = []
            for row in val_rows:
                lrs = row.get("learning_rates", [])
                if group_idx < len(lrs):
                    series.append(float(lrs[group_idx]))
                else:
                    series.append(float("nan"))
            lr_values_flat.extend(value for value in series if math.isfinite(value) and value > 0)
            lr_axes.plot(
                epochs[: len(series)],
                series,
                marker="o",
                linewidth=1.8,
                label=f"group_{group_idx}",
            )
        _style_axes(lr_axes, title="Learning Rates", ylabel="LR")
        if lr_values_flat:
            lr_axes.set_yscale("log")

    for ax in axes.flat:
        _annotate_phase_markers(ax, best_epoch=best_epoch, warmup_epochs=warmup_epochs)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Training Monitor | epoch {current_epoch}", fontsize=18, y=1.02)
    fig.text(
        0.5,
        0.005,
        _figure_summary_text(
            class_balance_info=class_balance_info,
            selection_metric_name=selection_metric_name,
            best_epoch=best_epoch,
            warmup_epochs=warmup_epochs,
        ),
        ha="center",
        fontsize=10,
        color="#475569",
    )
    _save_figure(fig, figure_dir / "training_dashboard")
    plt.close(fig)


def _render_validation_curves(
    *,
    diagnostics: ValidationDiagnostics,
    figure_dir: Path,
    latest_bundle: bool,
    current_epoch: int,
) -> None:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    if diagnostics.has_both_classes:
        fpr, tpr, _ = roc_curve(diagnostics.labels, diagnostics.probs)
        precision, recall, _ = precision_recall_curve(diagnostics.labels, diagnostics.probs)
        avg_precision = average_precision_score(diagnostics.labels, diagnostics.probs)

        axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2.2)
        axes[0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.0)
        _style_axes(axes[0], title="Validation ROC", xlabel="False Positive Rate", ylabel="True Positive Rate")

        axes[1].plot(recall, precision, color="#ea580c", linewidth=2.2)
        _style_axes(axes[1], title=f"Validation PR | AP={avg_precision:.3f}", xlabel="Recall", ylabel="Precision")
    else:
        for ax, title in zip(axes, ("Validation ROC", "Validation PR"), strict=False):
            ax.text(
                0.5,
                0.5,
                "Unavailable for single-class validation",
                ha="center",
                va="center",
                fontsize=12,
                color="#475569",
            )
            _style_axes(ax, title=title, xlabel="", ylabel="")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Validation Curves | epoch {current_epoch}", fontsize=16)
    stem_name = "validation_curves_latest" if latest_bundle else "validation_curves"
    _save_figure(fig, figure_dir / stem_name)
    plt.close(fig)


def _render_validation_confusion(
    *,
    diagnostics: ValidationDiagnostics,
    figure_dir: Path,
    latest_bundle: bool,
    current_epoch: int,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(6.3, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    matrix = confusion_matrix(diagnostics.labels, diagnostics.preds, labels=[0, 1])
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["real", "fake"])
    ax.set_yticks([0, 1], labels=["real", "fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Validation Confusion Matrix | epoch {current_epoch}", fontsize=14)

    max_value = int(matrix.max()) if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            count = int(matrix[row_idx, col_idx])
            text_color = "white" if count > max_value / 2 else "#0f172a"
            ax.text(col_idx, row_idx, str(count), ha="center", va="center", color=text_color, fontsize=12)

    stem_name = "validation_confusion_latest" if latest_bundle else "validation_confusion"
    _save_figure(fig, figure_dir / stem_name)
    plt.close(fig)


def _render_validation_score_distribution(
    *,
    diagnostics: ValidationDiagnostics,
    figure_dir: Path,
    latest_bundle: bool,
    current_epoch: int,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    bins = np.linspace(0.0, 1.0, 21)
    class_labels = {0: ("real", "#2563eb"), 1: ("fake", "#ea580c")}
    has_any_hist = False
    for class_id, (label_name, color) in class_labels.items():
        class_probs = diagnostics.probs[diagnostics.labels == class_id]
        if class_probs.size == 0:
            continue
        has_any_hist = True
        ax.hist(
            class_probs,
            bins=bins,
            alpha=0.65,
            density=True,
            label=f"{label_name} (n={class_probs.size})",
            color=color,
            edgecolor="white",
        )

    if not has_any_hist:
        ax.text(0.5, 0.5, "No validation scores available", ha="center", va="center", fontsize=12)
    _style_axes(ax, title=f"Validation Score Distribution | epoch {current_epoch}", xlabel="Predicted probability", ylabel="Density")
    if has_any_hist:
        ax.legend(frameon=False)
    if math.isfinite(float(np.mean(diagnostics.probs))):
        ax.axvline(0.5, color="#7c3aed", linestyle="--", linewidth=1.1, alpha=0.9)

    stem_name = "validation_score_distribution_latest" if latest_bundle else "validation_score_distribution"
    _save_figure(fig, figure_dir / stem_name)
    plt.close(fig)
