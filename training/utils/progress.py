from __future__ import annotations

import glob
import tarfile
from functools import lru_cache
from pathlib import Path

from webdataset.shardlists import expand_urls

from configs.loggings import logger


def _has_glob_magic(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


@lru_cache(maxsize=32)
def resolve_shard_paths(shard_pattern: str) -> tuple[str, ...]:
    normalized_pattern = shard_pattern.replace("\\", "/")
    expanded_patterns = expand_urls(normalized_pattern)
    resolved_paths: list[str] = []

    for pattern in expanded_patterns:
        if _has_glob_magic(pattern):
            resolved_paths.extend(sorted(glob.glob(pattern)))
            continue

        if Path(pattern).exists():
            resolved_paths.append(pattern)

    deduped_paths = tuple(dict.fromkeys(resolved_paths))
    if not deduped_paths:
        raise FileNotFoundError(f"No shard files matched pattern: {shard_pattern}")
    return deduped_paths


@lru_cache(maxsize=32)
def count_samples_in_shards(shard_pattern: str) -> int:
    total_samples = 0

    for shard_path in resolve_shard_paths(shard_pattern):
        sample_keys: set[str] = set()
        with tarfile.open(shard_path, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                member_path = member.name.replace("\\", "/")
                if not member_path.endswith(".json"):
                    continue
                sample_keys.add(member_path[:-5])
        total_samples += len(sample_keys)

    return total_samples


def estimate_total_batches(
    *,
    shard_pattern: str,
    batch_size: int,
    drop_last: bool = False,
) -> int:
    sample_count = count_samples_in_shards(shard_pattern)
    if sample_count == 0:
        return 0
    if drop_last:
        return sample_count // batch_size
    return (sample_count + batch_size - 1) // batch_size


def build_progress_totals(
    *,
    train_shards: str,
    val_shards: str,
    batch_size: int,
    drop_last: bool = False,
) -> dict[str, int | None]:
    totals: dict[str, int | None] = {"train": None, "val": None}

    for split_name, shard_pattern in (("train", train_shards), ("val", val_shards)):
        try:
            totals[split_name] = estimate_total_batches(
                shard_pattern=shard_pattern,
                batch_size=batch_size,
                drop_last=drop_last,
            )
        except Exception as exc:
            logger.warning(
                "Could not infer total batches for {} shards ({}). Falling back to unknown-length progress.",
                split_name,
                exc,
            )

    return totals
