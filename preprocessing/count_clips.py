from __future__ import annotations

import argparse
from pathlib import Path

from training.utils.progress import count_samples_in_shards, resolve_shard_paths


def count_split_clips(split_dir: Path) -> tuple[int, int]:
    shard_pattern = str(split_dir / "shard-*.tar")
    shard_paths = resolve_shard_paths(shard_pattern)
    clip_count = count_samples_in_shards(shard_pattern)
    return len(shard_paths), clip_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count clip samples inside clip_data shards")
    parser.add_argument(
        "--clip-root",
        type=Path,
        default=Path("clip_data"),
        help="Root directory that contains train/ and val/ shard folders",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split names to count under clip_root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_root: Path = args.clip_root
    total_clips = 0

    for split in args.splits:
        split_dir = clip_root / split
        shard_count, clip_count = count_split_clips(split_dir)
        total_clips += clip_count
        print(f"{split}: shards={shard_count} clips={clip_count} path={split_dir}")

    print(f"total_clips={total_clips}")


if __name__ == "__main__":
    main()
