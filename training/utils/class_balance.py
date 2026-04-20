from __future__ import annotations

import json
import tarfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from training.utils.progress import resolve_shard_paths


@dataclass(frozen=True)
class ClassBalanceInfo:
    negative_count: int
    positive_count: int
    positive_class_name: str
    raw_pos_weight: float | None
    effective_pos_weight: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_base_key(member_name: str) -> str | None:
    normalized = member_name.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return None
    known_suffixes = (
        ".rgb.npy",
        ".diff.npy",
        ".json",
        ".cls",
        ".jpg",
        ".jpeg",
        ".png",
        ".npy",
    )
    for suffix in known_suffixes:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    if "." not in normalized:
        return None
    return normalized.rsplit(".", 1)[0]


def _extract_label_from_parts(parts: dict[str, bytes], invert_binary_labels: bool) -> int:
    label_value: int | None = None

    cls_payload = parts.get("cls")
    if cls_payload is not None:
        raw = cls_payload.decode("utf-8") if isinstance(cls_payload, bytes) else str(cls_payload)
        label_value = int(str(raw).strip())

    if label_value is None:
        json_payload = parts.get("json")
        if json_payload is None:
            raise ValueError("Sample missing both cls and json payloads for label extraction.")

        metadata = json.loads(json_payload.decode("utf-8"))
        binary_label = metadata.get("binary_label")
        if binary_label in {0, 1, "0", "1"}:
            label_value = int(binary_label)
        else:
            label = str(metadata.get("label", "")).strip().lower()
            if label in {"real", "original"}:
                label_value = 0
            elif label == "fake":
                label_value = 1

    if label_value not in {0, 1}:
        raise ValueError("Could not infer binary label from shard sample.")

    if invert_binary_labels:
        return 1 - label_value
    return label_value


@lru_cache(maxsize=32)
def count_effective_labels_in_shards(
    shard_pattern: str,
    invert_binary_labels: bool,
) -> tuple[int, int]:
    negative_count = 0
    positive_count = 0

    for shard_path in resolve_shard_paths(shard_pattern):
        current_key: str | None = None
        current_parts: dict[str, bytes] = {}

        with tarfile.open(shard_path, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue

                base_key = _sample_base_key(member.name)
                if base_key is None:
                    continue

                if current_key is not None and base_key != current_key:
                    label_value = _extract_label_from_parts(current_parts, invert_binary_labels)
                    if label_value == 1:
                        positive_count += 1
                    else:
                        negative_count += 1
                    current_parts = {}

                current_key = base_key
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                suffix = member.name.rsplit(".", 1)[-1].lower()
                if suffix in {"json", "cls"}:
                    current_parts[suffix] = extracted.read()

            if current_key is not None and current_parts:
                label_value = _extract_label_from_parts(current_parts, invert_binary_labels)
                if label_value == 1:
                    positive_count += 1
                else:
                    negative_count += 1

    return negative_count, positive_count


def build_class_balance_info(
    *,
    shard_pattern: str,
    invert_binary_labels: bool,
    max_pos_weight: float | None = None,
) -> ClassBalanceInfo:
    negative_count, positive_count = count_effective_labels_in_shards(
        shard_pattern,
        invert_binary_labels,
    )

    raw_pos_weight: float | None = None
    effective_pos_weight: float | None = None
    if negative_count > 0 and positive_count > 0:
        raw_pos_weight = negative_count / positive_count
        effective_pos_weight = raw_pos_weight
        if max_pos_weight is not None:
            effective_pos_weight = min(effective_pos_weight, max_pos_weight)

    return ClassBalanceInfo(
        negative_count=negative_count,
        positive_count=positive_count,
        positive_class_name="real" if invert_binary_labels else "fake",
        raw_pos_weight=raw_pos_weight,
        effective_pos_weight=effective_pos_weight,
    )
