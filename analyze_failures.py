from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.amp import autocast
from tqdm import tqdm

from training.fusion_head import WeightedProbabilityFusionHead
from training.spatio_temporal_detector import ModelConfig, SpatioTemporalDeepfakeDetector
from training.train import TrainConfig
from training.utils.builders import build_dataloaders, build_model
from training.utils.checkpointing import load_checkpoint
from training.utils.metrics import move_batch_to_device
from training.utils.runtime import resolve_device, set_seed


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and visualize FP/FN failures for the spatio-temporal detector."
    )
    parser.add_argument("--test-shards", type=str, required=True, help='Celeb-DF v2 shard glob, e.g. "clip_data/test/*.tar"')
    parser.add_argument("--checkpoint", type=str, default="artifacts3/experiments/st_detector/best.pt")
    parser.add_argument("--output-dir", type=str, default="artifacts3/experiments/analysis_failures")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clip-len", type=int, default=None, help="Defaults to checkpoint train_config.clip_len or 8.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--invert-binary-labels", action="store_true")
    parser.add_argument("--max-failures", type=int, default=None, help="Optional cap for visualization/debug runs.")
    parser.add_argument(
        "--max-correct-examples",
        type=int,
        default=12,
        help="Number of correctly classified samples to visualize for comparison. Set 0 to disable.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sanitize_filename(value: Any, default: str = "unknown") -> str:
    text = str(value or default)
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or default


def first_meta_value(meta: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return default


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[SpatioTemporalDeepfakeDetector, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_config"), dict):
        model_cfg = ModelConfig(**ckpt["model_config"])
        model = SpatioTemporalDeepfakeDetector(model_cfg).to(device)
    else:
        model, _ = build_model(cfg, device)

    state_dict = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain model_state: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={list(missing)} unexpected={list(unexpected)}")

    for param in model.spatial_branch.parameters():
        param.requires_grad_(True)

    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def build_eval_config(args: argparse.Namespace, ckpt: dict[str, Any]) -> TrainConfig:
    train_cfg = ckpt.get("train_config") if isinstance(ckpt.get("train_config"), dict) else {}
    model_cfg = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
    clip_len = args.clip_len or int(train_cfg.get("clip_len") or model_cfg.get("temporal_num_frames", 7) + 1)

    return TrainConfig(
        train_shards=args.test_shards,
        val_shards=args.test_shards,
        output_dir=args.output_dir,
        epochs=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_len=clip_len,
        invert_binary_labels=args.invert_binary_labels or bool(train_cfg.get("invert_binary_labels", False)),
        model_dropout=float(train_cfg.get("model_dropout", model_cfg.get("dropout", 0.3))),
        temporal_pool=str(train_cfg.get("temporal_pool", model_cfg.get("temporal_pool", "gru"))),
        use_spatial_attention=bool(train_cfg.get("use_spatial_attention", model_cfg.get("use_spatial_attention", True))),
        use_texture_enhancement=bool(
            train_cfg.get("use_texture_enhancement", model_cfg.get("use_texture_enhancement", True))
        ),
        seed=args.seed,
        device=args.device,
        use_amp=not args.disable_amp,
        pin_memory=device_supports_pin_memory(args.device),
        persistent_workers=args.num_workers > 0,
        train_shuffle_buffer=0,
    )


def device_supports_pin_memory(device_name: str) -> bool:
    return str(device_name).startswith("cuda")


def denormalize_spatial_frame(frame: torch.Tensor) -> np.ndarray:
    image = (frame.detach().cpu() * RGB_STD + RGB_MEAN).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def fusion_embeddings(
    model: SpatioTemporalDeepfakeDetector,
    spatial_feat: torch.Tensor,
    temporal_feat: torch.Tensor,
) -> torch.Tensor:
    if isinstance(model.fusion_head, WeightedProbabilityFusionHead):
        spatial_embedding = model.fusion_head.spatial_head[:-1](spatial_feat)
        temporal_embedding = model.fusion_head.temporal_head[:-1](temporal_feat)
        return torch.cat([spatial_embedding, temporal_embedding], dim=1)

    fused = torch.cat([spatial_feat, temporal_feat], dim=1)
    return model.fusion_head.net[:-1](fused)


def forward_with_embeddings(
    model: SpatioTemporalDeepfakeDetector,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_feat, _ = model.spatial_branch(
        spatial,
        return_attention=True,
        return_feature_maps=False,
    )
    temporal_feat = model.temporal_branch(temporal)
    embeddings = fusion_embeddings(model, spatial_feat, temporal_feat)
    if isinstance(model.fusion_head, WeightedProbabilityFusionHead):
        logits = model.fusion_head(spatial_feat, temporal_feat)
    else:
        logits = model.fusion_head.net[-1](embeddings)
    if model.config.num_classes == 1:
        logits = logits.squeeze(1)
    return logits, embeddings


class GradCAM:
    def __init__(self, model: SpatioTemporalDeepfakeDetector) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = self._find_last_conv2d(model.spatial_branch.layer4)
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _find_last_conv2d(self, module: nn.Module) -> nn.Conv2d:
        conv_layers = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
        if not conv_layers:
            raise ValueError("Could not find a Conv2d layer under spatial_branch.layer4 for Grad-CAM.")
        return conv_layers[-1]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activation(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def __call__(self, spatial: torch.Tensor, temporal: torch.Tensor, sample_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(spatial, temporal)
        target_logit = logits.reshape(-1)[sample_index]
        target_logit.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activation = self.activations[sample_index : sample_index + 1]
        gradient = self.gradients[sample_index : sample_index + 1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def save_gradcam_overlay(image: np.ndarray, cam: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    ax.imshow(image)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close("all")


@torch.no_grad()
def compute_temporal_frame_probabilities(
    model: SpatioTemporalDeepfakeDetector,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    *,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_feat, _ = model.spatial_branch(
        spatial,
        return_attention=True,
        return_feature_maps=False,
    )

    frame_probs: list[float] = []
    motion_magnitude: list[float] = []
    _, num_frames, _, _, _ = temporal.shape
    for frame_idx in range(num_frames + 1):
        single_frame_clip = torch.zeros_like(temporal)
        if frame_idx > 0:
            single_frame_clip[:, frame_idx - 1] = temporal[:, frame_idx - 1]
            motion_magnitude.append(float(temporal[:, frame_idx - 1].abs().mean().detach().cpu()))
        else:
            motion_magnitude.append(0.0)

        with autocast(device_type=spatial.device.type, enabled=use_amp and spatial.device.type == "cuda"):
            temporal_feat = model.temporal_branch(single_frame_clip)
            if isinstance(model.fusion_head, WeightedProbabilityFusionHead):
                logit = model.fusion_head(spatial_feat, temporal_feat)
            else:
                embedding = fusion_embeddings(model, spatial_feat, temporal_feat)
                logit = model.fusion_head.net[-1](embedding)
            if model.config.num_classes == 1:
                logit = logit.squeeze(1)
            prob = torch.sigmoid(logit.reshape(-1)[0])
        frame_probs.append(float(prob.detach().cpu()))

    return np.asarray(frame_probs, dtype=np.float32), np.asarray(motion_magnitude, dtype=np.float32)


def save_timeline_chart(
    frame_probs: np.ndarray,
    motion_magnitude: np.ndarray,
    *,
    full_clip_probability: float,
    true_label: int,
    predicted_label: int,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(frame_probs) + 1)

    fig, (ax, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(8.8, 5.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )
    ax.plot(
        x,
        frame_probs,
        color="#111827",
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="Frame fake probability",
    )
    ax.fill_between(x, frame_probs, 0.0, color="#60A5FA", alpha=0.12)
    ax.axhline(0.5, color="#B91C1C", linestyle="--", linewidth=1.2, label="Decision threshold")
    ax.axhline(full_clip_probability, color="#047857", linestyle="-.", linewidth=1.2, label="Full clip prob")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fake probability")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.01,
        0.98,
        f"true={true_label} pred={predicted_label} full_clip={full_clip_probability:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 4},
    )

    bar_values = motion_magnitude.astype(np.float32)
    if float(bar_values.max(initial=0.0)) > 0.0:
        bar_values = bar_values / (bar_values.max() + 1e-8)
    ax_bar.bar(x, bar_values, color="#6B7280", alpha=0.68, width=0.72)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("Abs diff\nnorm.", fontsize=9)
    ax_bar.set_xlabel("Frame index in 8-frame clip")
    ax_bar.set_xticks(x)
    ax_bar.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close("all")


def save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    *,
    output_path: Path,
    perplexity: float,
    random_state: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(embeddings.shape[0])
    if n_samples < 3:
        return {"skipped": True, "reason": "Need at least 3 samples for t-SNE.", "num_samples": n_samples}

    effective_perplexity = min(float(perplexity), float(max(1, n_samples - 1)) / 3.0)
    effective_perplexity = max(1.0, effective_perplexity)
    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)

    correct = labels == preds
    real_correct = correct & (labels == 0)
    fake_correct = correct & (labels == 1)
    fp = (~correct) & (labels == 0) & (preds == 1)
    fn = (~correct) & (labels == 1) & (preds == 0)

    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=180)
    ax.scatter(coords[real_correct, 0], coords[real_correct, 1], s=18, c="#93C5FD", alpha=0.45, label="Correct Real")
    ax.scatter(coords[fake_correct, 0], coords[fake_correct, 1], s=18, c="#FCA5A5", alpha=0.45, label="Correct Fake")
    ax.scatter(coords[fp, 0], coords[fp, 1], s=92, c="#7C2D12", marker="X", linewidths=0.8, edgecolors="white", label="False Positive")
    ax.scatter(coords[fn, 0], coords[fn, 1], s=100, c="#1D4ED8", marker="^", linewidths=0.8, edgecolors="white", label="False Negative")
    ax.set_title("Fusion Embedding Manifold (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.16)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close("all")
    return {
        "skipped": False,
        "num_samples": n_samples,
        "perplexity": effective_perplexity,
        "output_path": str(output_path),
    }


def build_example_row(
    *,
    meta: dict[str, Any],
    spatial_index: int,
    true_label: int,
    predicted_label: int,
    probability: float,
    example_type: str,
    batch_index: int,
) -> dict[str, Any]:
    video_name = first_meta_value(
        meta,
        "video_name",
        "video_id",
        "video",
        "source_video",
        "target_video",
        "key",
        "__key__",
        default=f"sample_{batch_index:06d}",
    )
    frame_index = first_meta_value(
        meta,
        "frame_index",
        "frame_number",
        "start_frame",
        "clip_start_frame",
        "original_frame_index",
        default=spatial_index,
    )
    return {
        "video_name": str(video_name),
        "frame_index": as_int(frame_index, spatial_index),
        "spatial_index": int(spatial_index),
        "overlay_frame_source": "batch['spatial']; this is the model-selected spatial frame, usually default_center_index in eval mode",
        "true_label": int(true_label),
        "predicted_label": int(predicted_label),
        "confidence_probability": float(probability),
        "example_type": example_type,
        "failure_type": example_type if example_type in {"FP", "FN"} else "",
        "key": str(first_meta_value(meta, "key", "__key__", "sample_key", default="")),
        "label_name": str(first_meta_value(meta, "label", "class_name", default="")),
    }


def example_filename(row: dict[str, Any], suffix: str) -> str:
    example_type = sanitize_filename(row.get("example_type") or row.get("failure_type"))
    video_name = sanitize_filename(row["video_name"])
    return f"{example_type}_{video_name}_{suffix}.png"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique output path for {path}")


def save_visual_diagnostics(
    *,
    model: SpatioTemporalDeepfakeDetector,
    grad_cam: GradCAM,
    row: dict[str, Any],
    sample_spatial: torch.Tensor,
    sample_temporal: torch.Tensor,
    spatial_dir: Path,
    temporal_dir: Path,
    use_amp: bool,
) -> None:
    true_label = int(row["true_label"])
    predicted_label = int(row["predicted_label"])
    probability = float(row["confidence_probability"])
    title = f"{row['example_type']} | {row['video_name']} | frame {row['frame_index']}"

    try:
        cam = grad_cam(sample_spatial, sample_temporal, sample_index=0)
        image = denormalize_spatial_frame(sample_spatial[0])
        spatial_path = unique_path(spatial_dir / example_filename(row, "gradcam"))
        save_gradcam_overlay(image, cam, spatial_path)
        row["spatial_gradcam_path"] = str(spatial_path)
    except Exception as exc:
        row["spatial_gradcam_error"] = str(exc)

    try:
        frame_probs, motion_magnitude = compute_temporal_frame_probabilities(
            model,
            sample_spatial,
            sample_temporal,
            use_amp=use_amp,
        )
        timeline_path = unique_path(temporal_dir / example_filename(row, "timeline"))
        save_timeline_chart(
            frame_probs,
            motion_magnitude,
            full_clip_probability=probability,
            true_label=true_label,
            predicted_label=predicted_label,
            title=title,
            output_path=timeline_path,
        )
        row["temporal_timeline_path"] = str(timeline_path)
        row["frame_fake_probabilities"] = [float(value) for value in frame_probs]
        row["frame_absdiff_magnitude"] = [float(value) for value in motion_magnitude]
    except Exception as exc:
        row["temporal_timeline_error"] = str(exc)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    spatial_dir = output_dir / "spatial_attention"
    temporal_dir = output_dir / "temporal_sequence"
    correct_spatial_dir = output_dir / "correct_examples" / "spatial_attention"
    correct_temporal_dir = output_dir / "correct_examples" / "temporal_sequence"

    initial_ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    initial_ckpt = initial_ckpt if isinstance(initial_ckpt, dict) else {}
    cfg = build_eval_config(args, initial_ckpt)
    device = resolve_device(args.device)
    cfg.device = str(device)
    cfg.use_amp = not args.disable_amp

    _, test_loader = build_dataloaders(cfg)
    model, ckpt = load_model_from_checkpoint(checkpoint_path, cfg, device)
    grad_cam = GradCAM(model)

    failures: list[dict[str, Any]] = []
    correct_examples: list[dict[str, Any]] = []
    counters = {"total": 0, "fp": 0, "fn": 0}
    correct_label_counts = {0: 0, 1: 0}
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    try:
        for batch in tqdm(test_loader, desc="Analyze failures", dynamic_ncols=True):
            batch = move_batch_to_device(batch, device)
            labels = batch["label"].detach().to(torch.int64)

            with torch.no_grad(), autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
                logits, embeddings = forward_with_embeddings(model, batch["spatial"], batch["temporal"])
                probs = torch.sigmoid(logits).reshape(-1)
                preds = (probs >= args.threshold).to(torch.int64)

            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            all_preds.append(preds.detach().cpu().numpy().astype(np.int64, copy=False))

            failed_indices = torch.nonzero(preds != labels, as_tuple=False).reshape(-1).tolist()
            correct_indices = torch.nonzero(preds == labels, as_tuple=False).reshape(-1).tolist()
            counters["total"] += int(labels.numel())

            for sample_idx in correct_indices:
                if len(correct_examples) >= args.max_correct_examples:
                    break

                true_label = int(labels[sample_idx].detach().cpu())
                per_class_target = max(1, args.max_correct_examples // 2)
                if correct_label_counts[true_label] >= per_class_target and len(correct_examples) < args.max_correct_examples - 1:
                    other_label = 1 - true_label
                    if correct_label_counts[other_label] < per_class_target:
                        continue

                predicted_label = int(preds[sample_idx].detach().cpu())
                probability = float(probs[sample_idx].detach().cpu())
                example_type = "CORRECT_FAKE" if true_label == 1 else "CORRECT_REAL"

                meta = batch["meta"][sample_idx] if sample_idx < len(batch["meta"]) else {}
                meta = meta if isinstance(meta, dict) else {}
                spatial_index = int(batch["spatial_index"][sample_idx].detach().cpu())
                row = build_example_row(
                    meta=meta,
                    spatial_index=spatial_index,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    probability=probability,
                    example_type=example_type,
                    batch_index=len(correct_examples),
                )

                save_visual_diagnostics(
                    model=model,
                    grad_cam=grad_cam,
                    row=row,
                    sample_spatial=batch["spatial"][sample_idx : sample_idx + 1],
                    sample_temporal=batch["temporal"][sample_idx : sample_idx + 1],
                    spatial_dir=correct_spatial_dir,
                    temporal_dir=correct_temporal_dir,
                    use_amp=cfg.use_amp,
                )
                correct_examples.append(row)
                correct_label_counts[true_label] += 1

            for sample_idx in failed_indices:
                true_label = int(labels[sample_idx].detach().cpu())
                predicted_label = int(preds[sample_idx].detach().cpu())
                probability = float(probs[sample_idx].detach().cpu())
                failure_type = "FP" if predicted_label == 1 and true_label == 0 else "FN"
                counters["fp" if failure_type == "FP" else "fn"] += 1

                meta = batch["meta"][sample_idx] if sample_idx < len(batch["meta"]) else {}
                meta = meta if isinstance(meta, dict) else {}
                spatial_index = int(batch["spatial_index"][sample_idx].detach().cpu())
                row = build_example_row(
                    meta=meta,
                    spatial_index=spatial_index,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    probability=probability,
                    example_type=failure_type,
                    batch_index=len(failures),
                )

                save_visual_diagnostics(
                    model=model,
                    grad_cam=grad_cam,
                    row=row,
                    sample_spatial=batch["spatial"][sample_idx : sample_idx + 1],
                    sample_temporal=batch["temporal"][sample_idx : sample_idx + 1],
                    spatial_dir=spatial_dir,
                    temporal_dir=temporal_dir,
                    use_amp=cfg.use_amp,
                )

                failures.append(row)
                if args.max_failures is not None and len(failures) >= args.max_failures:
                    break

            if args.max_failures is not None and len(failures) >= args.max_failures:
                break
    finally:
        grad_cam.close()

    tsne_result: dict[str, Any]
    if all_embeddings:
        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        preds_np = np.concatenate(all_preds, axis=0)
        try:
            tsne_result = save_tsne_plot(
                embeddings_np,
                labels_np,
                preds_np,
                output_path=output_dir / "feature_space_tsne.png",
                perplexity=args.tsne_perplexity,
                random_state=args.seed,
            )
        except Exception as exc:
            tsne_result = {"skipped": True, "reason": str(exc), "num_samples": int(embeddings_np.shape[0])}
    else:
        tsne_result = {"skipped": True, "reason": "No embeddings were collected.", "num_samples": 0}

    report = {
        "checkpoint": str(checkpoint_path),
        "test_shards": args.test_shards,
        "threshold": float(args.threshold),
        "summary": {
            "total_scanned": counters["total"],
            "false_positives": counters["fp"],
            "false_negatives": counters["fn"],
            "failures_saved": len(failures),
            "correct_examples_saved": len(correct_examples),
            "correct_real_examples_saved": correct_label_counts[0],
            "correct_fake_examples_saved": correct_label_counts[1],
        },
        "feature_space_tsne": tsne_result,
        "notes": {
            "spatial_heatmap": "Overlay uses batch['spatial'], the normalized face crop actually consumed by the model. The current dataloader does not expose the full raw RGB clip.",
            "temporal_sequence": "The model consumes 7 frame-difference tensors for each 8-frame clip. Timeline has 8 positions: frame 1 is a zero-motion baseline, frames 2-8 isolate each corresponding frame-difference tensor.",
            "feature_space_tsne": "Embeddings are fusion_head.net[:-1] outputs, detached to CPU batch-by-batch before t-SNE.",
        },
        "config": asdict(cfg),
        "failures": failures,
        "correct_examples": correct_examples,
    }
    write_json(output_dir / "failure_report.json", report)
    print(
        "Analysis complete | scanned={} | FP={} | FN={} | failures_saved={} | correct_saved={} | report={}".format(
            counters["total"],
            counters["fp"],
            counters["fn"],
            len(failures),
            len(correct_examples),
            output_dir / "failure_report.json",
        )
    )


if __name__ == "__main__":
    main()
