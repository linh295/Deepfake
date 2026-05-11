from __future__ import annotations

import io
import json
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter


@dataclass
class ClipDatasetConfig:
    shard_pattern: str
    clip_len: int = 8
    diff_len: int | None = None
    invert_binary_labels: bool = False
    spatial_candidate_indices: tuple[int, ...] = (2, 3, 4, 5)
    training: bool = True

    # Augmentation config. Applied only when training=True.
    use_augmentation: bool = False
    augment_recompute_diff: bool = True
    hflip_prob: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    
    jpeg_prob: float = 0.0
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 95

    blur_prob: float = 0.0
    blur_kernel: int = 3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.0

    shuffle_buffer: int = 1000
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    bad_sample_log_limit: int = 20

    def __post_init__(self) -> None:
        expected_diff_len = self.clip_len - 1
        if self.diff_len is None:
            self.diff_len = expected_diff_len
        elif self.diff_len != expected_diff_len:
            raise ValueError(
                f"diff_len must equal clip_len - 1 ({expected_diff_len}), got {self.diff_len}"
            )

        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError(f"hflip_prob must be in [0, 1], got {self.hflip_prob}")
        if self.brightness < 0:
            raise ValueError(f"brightness must be >= 0, got {self.brightness}")
        if self.contrast < 0:
            raise ValueError(f"contrast must be >= 0, got {self.contrast}")


class ClipWebDataset:
    """
    WebDataset reader for clip-level shards.

    Expected sample fields:
    - rgb.npy  : uint8 array with shape [T, 3, H, W]
    - diff.npy : uint8 array with shape [T-1, 3, H, W]
    - json     : metadata containing clip info
    - cls      : binary label as bytes/text (optional if json has binary_label)

    Output keys per sample:
    - spatial  : float32 tensor [3, H, W]
    - temporal : float32 tensor [T-1, 3, H, W]
    - label    : float32 tensor []
    - meta     : dict
    """

    def __init__(self, config: ClipDatasetConfig) -> None:
        self.config = config
        self._bad_sample_count = 0

        # Avoid recreating these tensors for every sample.
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _load_npy_bytes(self, payload: bytes) -> np.ndarray:
        with io.BytesIO(payload) as buffer:
            return np.load(buffer, allow_pickle=False)

    def _metadata_int(self, metadata: Dict[str, Any], key: str) -> int | None:
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _candidate_indices_from_metadata(self, metadata: Dict[str, Any], clip_len: int) -> List[int]:
        raw_candidates = metadata.get("center_candidate_indices")
        if not isinstance(raw_candidates, list):
            return []

        candidates: List[int] = []
        for raw_idx in raw_candidates:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < clip_len:
                candidates.append(idx)
        return candidates

    def _choose_spatial_index(self, metadata: Dict[str, Any]) -> int:
        clip_len = int(metadata.get("clip_length", self.config.clip_len))
        valid_candidates = self._candidate_indices_from_metadata(metadata, clip_len)
        if not valid_candidates:
            valid_candidates = [
                idx for idx in self.config.spatial_candidate_indices if 0 <= idx < clip_len
            ]
        if not valid_candidates:
            valid_candidates = [max(0, clip_len // 2)]

        if self.config.training:
            return random.choice(valid_candidates)

        default_idx = metadata.get("default_center_index")
        if default_idx is not None:
            try:
                default_idx = int(default_idx)
                if 0 <= default_idx < clip_len:
                    return default_idx
            except Exception:
                pass
        return valid_candidates[len(valid_candidates) // 2]

    def _extract_label(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        label_value: float | None = None
        if "cls" in sample and sample["cls"] is not None:
            raw = sample["cls"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            label_value = float(int(str(raw).strip()))

        if label_value is None:
            binary_label = metadata.get("binary_label")
            if binary_label in {0, 1, "0", "1"}:
                label_value = float(int(binary_label))

        if label_value is None:
            label = str(metadata.get("label", "")).strip().lower()
            if label in {"real", "original"}:
                label_value = 0.0
            elif label == "fake":
                label_value = 1.0

        if label_value is None:
            raise ValueError("Could not infer label from sample.")

        if self.config.invert_binary_labels:
            return 1.0 - label_value
        return label_value

    def _sample_clip_augmentation_params(self) -> dict[str, float | bool]:
        """
        Sample one augmentation parameter set for the whole clip.

        Critical for temporal modeling:
        all 8 RGB frames receive exactly the same transform.
        """
        do_hflip = random.random() < self.config.hflip_prob

        brightness_factor = 1.0
        if self.config.brightness > 0:
            brightness_factor = random.uniform(
                1.0 - self.config.brightness,
                1.0 + self.config.brightness,
            )

        contrast_factor = 1.0
        if self.config.contrast > 0:
            contrast_factor = random.uniform(
                1.0 - self.config.contrast,
                1.0 + self.config.contrast,
            )
        
        do_jpeg = random.random() < self.config.jpeg_prob
        jpeg_quality = random.randint(
            self.config.jpeg_quality_min,
            self.config.jpeg_quality_max,
        )

        do_blur = random.random() < self.config.blur_prob
        blur_sigma = random.uniform(
            self.config.blur_sigma_min,
            self.config.blur_sigma_max,
        )

        return {
            "do_hflip": do_hflip,
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "do_jpeg": do_jpeg,
            "jpeg_quality": jpeg_quality,
            "do_blur": do_blur,
            "blur_sigma": blur_sigma,
        }

    def _apply_clip_consistent_augmentation(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Apply safe clip-consistent augmentation.

        Input:
            rgb: [T, 3, H, W], value range [0, 255]
        Output:
            rgb: [T, 3, H, W], float32, value range [0, 255]
        """
        rgb = rgb.float()

        if not (self.config.training and self.config.use_augmentation):
            return rgb

        params = self._sample_clip_augmentation_params()

        if bool(params["do_hflip"]):
            rgb = torch.flip(rgb, dims=[-1])

        contrast_factor = float(params["contrast_factor"])
        if contrast_factor != 1.0:
            mean = rgb.mean(dim=(-2, -1), keepdim=True)
            rgb = (rgb - mean) * contrast_factor + mean

        brightness_factor = float(params["brightness_factor"])
        if brightness_factor != 1.0:
            rgb = rgb * brightness_factor
            
        if bool(params["do_blur"]):
            rgb = torch.stack(
                [
                    self._apply_gaussian_blur_to_frame(frame, sigma=float(params["blur_sigma"]))
                    for frame in rgb
                ],
                dim=0,
            )

        if bool(params["do_jpeg"]):
            rgb = torch.stack(
                [
                    self._apply_jpeg_compression_to_frame(frame, quality=int(params["jpeg_quality"]))
                    for frame in rgb
                ],
                dim=0,
            )

        return rgb.clamp(0.0, 255.0)

    def _recompute_diff_from_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Recompute absolute frame difference from RGB clip.

        Input:
            rgb: [T, 3, H, W], value range [0, 255]
        Output:
            diff: [T-1, 3, H, W], value range [0, 255]
        """
        return torch.abs(rgb[1:] - rgb[:-1]).clamp(0.0, 255.0)

    def _normalize_rgb(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float().div(255.0)
        return (tensor - self.rgb_mean) / self.rgb_std

    def _normalize_diff(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().div(255.0)

    def _sample_key(self, sample: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> str:
        sample_key = sample.get("__key__")
        if sample_key is not None:
            return str(sample_key)

        if metadata is not None:
            metadata_key = metadata.get("key")
            if metadata_key is not None:
                return str(metadata_key)

        return "<unknown>"

    def _apply_jpeg_compression_to_frame(self, frame: torch.Tensor, quality: int) -> torch.Tensor:
        """
        frame: [3, H, W], float tensor in [0, 255]
        """
        frame_uint8 = frame.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        image = Image.fromarray(frame_uint8)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")
        arr = np.asarray(compressed).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).float()
    
    def _apply_gaussian_blur_to_frame(self, frame: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        frame: [3, H, W], float tensor in [0, 255]
        """
        frame_uint8 = frame.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        image = Image.fromarray(frame_uint8)
        image = image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

        arr = np.asarray(image).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).float()
    
    def _wrap_sample_error(
        self,
        sample: Dict[str, Any],
        exc: Exception,
        metadata: Dict[str, Any] | None = None,
    ) -> "ClipSampleDecodeError":
        if isinstance(exc, ClipSampleDecodeError):
            return exc

        wrapped = ClipSampleDecodeError(self._sample_key(sample, metadata), str(exc))
        wrapped.__cause__ = exc
        return wrapped

    def _handle_sample_error(self, exn: Exception) -> bool:
        if not isinstance(exn, ClipSampleDecodeError):
            raise exn

        self._bad_sample_count += 1
        if self._bad_sample_count <= self.config.bad_sample_log_limit:
            warnings.warn(f"Skipping malformed clip sample: {exn}", stacklevel=2)
        elif self._bad_sample_count == self.config.bad_sample_log_limit + 1:
            warnings.warn(
                "Reached bad sample log limit; suppressing further malformed-sample warnings.",
                stacklevel=2,
            )
        return True

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] | None = None
        try:
            if "rgb.npy" not in sample:
                raise ValueError("Sample missing rgb.npy")
            if "diff.npy" not in sample:
                raise ValueError("Sample missing diff.npy")
            if "json" not in sample:
                raise ValueError("Sample missing json metadata")

            metadata = json.loads(sample["json"].decode("utf-8"))
            rgb = self._load_npy_bytes(sample["rgb.npy"])
            diff = self._load_npy_bytes(sample["diff.npy"])

            if rgb.ndim != 4:
                raise ValueError(f"Expected rgb ndim=4, got shape={rgb.shape}")
            if diff.ndim != 4:
                raise ValueError(f"Expected diff ndim=4, got shape={diff.shape}")

            metadata_clip_len = self._metadata_int(metadata, "clip_length")
            if metadata_clip_len is not None and rgb.shape[0] != metadata_clip_len:
                raise ValueError(
                    f"Expected rgb clip_length from metadata={metadata_clip_len}, got {rgb.shape[0]}"
                )
            if rgb.shape[0] != self.config.clip_len:
                raise ValueError(
                    f"Expected rgb clip_len={self.config.clip_len}, got {rgb.shape[0]}"
                )

            expected_diff_len = self._metadata_int(metadata, "num_differences")
            if expected_diff_len is None:
                expected_diff_len = rgb.shape[0] - 1
            if diff.shape[0] != expected_diff_len:
                raise ValueError(
                    f"Expected diff num_differences={expected_diff_len}, got {diff.shape[0]}"
                )
            if diff.shape[0] != self.config.diff_len:
                raise ValueError(
                    f"Expected diff_len={self.config.diff_len}, got {diff.shape[0]}"
                )

            spatial_idx = self._choose_spatial_index(metadata)

            rgb_tensor = torch.from_numpy(rgb.copy())  # [T, 3, H, W]
            rgb_tensor = self._apply_clip_consistent_augmentation(rgb_tensor)

            if self.config.training and self.config.use_augmentation and self.config.augment_recompute_diff:
                diff_tensor = self._recompute_diff_from_rgb(rgb_tensor)
            else:
                diff_tensor = torch.from_numpy(diff.copy()).float()

            if diff_tensor.shape[0] != self.config.diff_len:
                raise ValueError(
                    f"Expected processed diff_len={self.config.diff_len}, got {diff_tensor.shape[0]}"
                )

            spatial = rgb_tensor[spatial_idx]
            temporal = diff_tensor
            label = torch.tensor(self._extract_label(sample, metadata), dtype=torch.float32)

            spatial = self._normalize_rgb(spatial)
            temporal = self._normalize_diff(temporal)

            return {
                "spatial": spatial,
                "temporal": temporal,
                "label": label,
                "spatial_index": torch.tensor(spatial_idx, dtype=torch.long),
                "meta": metadata,
            }
        except Exception as exc:
            raise self._wrap_sample_error(sample, exc, metadata) from exc

    def build_dataset(self) -> wds.WebDataset:
        shardshuffle = 100 if self.config.training else False
        dataset = wds.WebDataset(
            self.config.shard_pattern,
            shardshuffle=shardshuffle,
            seed=self.config.seed,
        )

        if self.config.training:
            dataset = dataset.shuffle(self.config.shuffle_buffer, rng=random.Random(self.config.seed))

        map_handler = self._handle_sample_error if self.config.training else wds.handlers.reraise_exception
        dataset = dataset.map(self._process_sample, handler=map_handler)
        return dataset


class ClipSampleDecodeError(ValueError):
    def __init__(self, sample_key: str, message: str) -> None:
        super().__init__(f"{message} [sample_key={sample_key}]")
        self.sample_key = sample_key


def collate_clip_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    spatial = torch.stack([item["spatial"] for item in batch], dim=0)
    temporal = torch.stack([item["temporal"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    spatial_indices = torch.stack([item["spatial_index"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]

    return {
        "spatial": spatial,
        "temporal": temporal,
        "label": labels,
        "spatial_index": spatial_indices,
        "meta": metas,
    }


def build_clip_dataloader(config: ClipDatasetConfig) -> DataLoader:
    dataset_builder = ClipWebDataset(config)
    dataset = dataset_builder.build_dataset()

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=config.drop_last,
        collate_fn=collate_clip_batch,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clip WebDataset smoke test")
    parser.add_argument("--shards", type=str, required=True, help="Shard pattern, e.g. clip_data/train/shard-*.tar")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    cfg = ClipDatasetConfig(
        shard_pattern=args.shards,
        training=not args.eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.augment,
    )
    loader = build_clip_dataloader(cfg)

    batch = next(iter(loader))
    print("spatial:", tuple(batch["spatial"].shape))
    print("temporal:", tuple(batch["temporal"].shape))
    print("label:", tuple(batch["label"].shape))
    print("spatial_index:", batch["spatial_index"].tolist())
    print("first meta key:", batch["meta"][0].get("key"))
