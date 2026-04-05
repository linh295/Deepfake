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


@dataclass
class ClipDatasetConfig:
    shard_pattern: str
    clip_len: int = 8
    diff_len: int | None = None
    invert_binary_labels: bool = False
    spatial_candidate_indices: tuple[int, ...] = (2, 3, 4, 5)
    training: bool = True
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
            return
        if self.diff_len != expected_diff_len:
            raise ValueError(
                f"diff_len must equal clip_len - 1 ({expected_diff_len}), got {self.diff_len}"
            )


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

    def _normalize_rgb(self, tensor: torch.Tensor) -> torch.Tensor:
        # ImageNet normalization for spatial RGB branch.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = tensor.float().div(255.0)
        return (tensor - mean) / std

    def _normalize_diff(self, tensor: torch.Tensor) -> torch.Tensor:
        # diff.npy stores absolute frame-difference magnitudes, so keep a simple
        # [0, 1] scaling until temporal-model expectations are defined.
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

    def _wrap_sample_error(
        self,
        sample: Dict[str, Any],
        exc: Exception,
        metadata: Dict[str, Any] | None = None,
    ) -> ClipSampleDecodeError:
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
            spatial = torch.from_numpy(rgb[spatial_idx].copy())
            temporal = torch.from_numpy(diff.copy())
            label = torch.tensor(self._extract_label(sample, metadata), dtype=torch.float32)

            spatial = self._normalize_rgb(spatial)
            temporal = self._normalize_diff(temporal)

            output = {
                "spatial": spatial,                  # [3, H, W]
                "temporal": temporal,                # [T-1, 3, H, W]
                "label": label,                      # []
                "spatial_index": torch.tensor(spatial_idx, dtype=torch.long),
                "meta": metadata,
            }
            return output
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
        "spatial": spatial,              # [B, 3, H, W]
        "temporal": temporal,            # [B, T-1, 3, H, W]
        "label": labels,                 # [B]
        "spatial_index": spatial_indices,
        "meta": metas,
    }



def build_clip_dataloader(config: ClipDatasetConfig) -> DataLoader:
    dataset_builder = ClipWebDataset(config)
    dataset = dataset_builder.build_dataset()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=config.drop_last,
        collate_fn=collate_clip_batch,
    )
    return loader


if __name__ == "__main__":
    # Minimal smoke-test example:
    # python dataset.py --shards "clip_data/train/shard-*.tar"
    import argparse

    parser = argparse.ArgumentParser(description="Clip WebDataset smoke test")
    parser.add_argument("--shards", type=str, required=True, help="Shard pattern, e.g. clip_data/train/shard-*.tar")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="Use deterministic center frame instead of random sampling")
    args = parser.parse_args()

    cfg = ClipDatasetConfig(
        shard_pattern=args.shards,
        training=not args.eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = build_clip_dataloader(cfg)

    batch = next(iter(loader))
    print("spatial:", tuple(batch["spatial"].shape))
    print("temporal:", tuple(batch["temporal"].shape))
    print("label:", tuple(batch["label"].shape))
    print("spatial_index:", batch["spatial_index"].tolist())
    print("first meta key:", batch["meta"][0].get("key"))
