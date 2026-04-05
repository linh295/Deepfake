from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from webdataset import ShardWriter

from configs.loggings import logger, setup_logging
from configs.settings import settings


_VALID_SPLITS = {"train", "val", "test"}


@dataclass
class FrameSample:
    key: str
    split: str
    category: str
    video_id: str
    video_name: str
    frame_number: int
    original_frame_index: int
    timestamp: float
    binary_label: int
    extraction_fps: float
    video_fps: float
    image_rgb: np.ndarray
    metadata: Dict[str, Any]


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def _label_from_metadata(meta: Dict[str, Any]) -> int:
    raw = _normalize_text(meta.get("binary_label"))
    if raw in {"0", "1"}:
        return int(raw)
    label = _normalize_text(meta.get("label")).lower()
    if label in {"real", "original"}:
        return 0
    if label == "fake":
        return 1
    return -1


def _split_from_metadata(meta: Dict[str, Any], fallback_split: Optional[str]) -> str:
    split = _normalize_text(meta.get("split")).lower()
    if split in _VALID_SPLITS:
        return split
    return fallback_split or ""


def _base_key_from_tar_member(member_name: str) -> Optional[str]:
    path = member_name.replace("\\", "/")
    if path.endswith("/"):
        return None
    basename = path.rsplit("/", 1)[-1]
    if "." not in basename:
        return None
    base, _ = basename.rsplit(".", 1)
    if not base:
        return None
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    return f"{parent}/{base}" if parent else base


def _flush_sample_parts(
    current_key: Optional[str],
    current_parts: Dict[str, bytes],
    fallback_split: Optional[str],
) -> Optional[FrameSample]:
    if not current_key or "json" not in current_parts:
        return None

    metadata = json.loads(current_parts["json"].decode("utf-8"))
    img_bytes = current_parts.get("jpg") or current_parts.get("jpeg") or current_parts.get("png")
    if img_bytes is None:
        return None

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    split = _split_from_metadata(metadata, fallback_split=fallback_split)
    category = _normalize_text(metadata.get("category"))
    video_id = _normalize_text(metadata.get("video_id")) or _normalize_text(metadata.get("video_name"))
    video_name = _normalize_text(metadata.get("video_name")) or video_id

    return FrameSample(
        key=_normalize_text(metadata.get("key")) or current_key,
        split=split,
        category=category,
        video_id=video_id,
        video_name=video_name,
        frame_number=_safe_int(metadata.get("frame_number"), default=0),
        original_frame_index=_safe_int(metadata.get("original_frame_index"), default=0),
        timestamp=_safe_float(metadata.get("timestamp"), default=0.0),
        binary_label=_label_from_metadata(metadata),
        extraction_fps=_safe_float(metadata.get("extraction_fps"), default=0.0),
        video_fps=_safe_float(metadata.get("video_fps"), default=0.0),
        image_rgb=image_rgb,
        metadata=metadata,
    )


def _canonical_video_id(frame: FrameSample) -> str:
    video_id = _normalize_text(frame.video_id)
    if video_id:
        return video_id.replace("\\", "/")

    category = _normalize_text(frame.category)
    video_name = _normalize_text(frame.video_name)
    if category and video_name:
        return f"{category}/{video_name}".replace("\\", "/")
    return video_name or "unknown_video"


def _clip_key_video_path(frame: FrameSample) -> str:
    return _canonical_video_id(frame)


def iter_frame_samples_from_shards(split_input_dir: Path, fallback_split: Optional[str]) -> Iterator[FrameSample]:
    shard_paths = sorted(split_input_dir.glob("shard-*.tar"))
    for shard_path in tqdm(shard_paths, desc=f"Read shards {split_input_dir.name}", leave=False):
        with tarfile.open(shard_path, "r") as archive:
            current_key: Optional[str] = None
            current_parts: Dict[str, bytes] = {}

            for member in archive.getmembers():
                if not member.isfile():
                    continue
                base_key = _base_key_from_tar_member(member.name)
                if base_key is None:
                    continue

                if current_key is not None and base_key != current_key:
                    sample = _flush_sample_parts(current_key, current_parts, fallback_split=fallback_split)
                    if sample is not None:
                        yield sample
                    current_parts = {}

                current_key = base_key
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                suffix = member.name.rsplit(".", 1)[-1].lower()
                current_parts[suffix] = extracted.read()

            if current_key is not None:
                sample = _flush_sample_parts(current_key, current_parts, fallback_split=fallback_split)
                if sample is not None:
                    yield sample


def _frame_continuity_ok(frames: List[FrameSample], frame_stride: int) -> bool:
    if len(frames) <= 1:
        return True
    expected = max(1, int(frame_stride))
    for a, b in zip(frames[:-1], frames[1:]):
        if (b.frame_number - a.frame_number) != expected:
            return False
    return True


def _stack_clip_rgb(frames: List[FrameSample]) -> np.ndarray:
    rgb = np.stack([np.transpose(frame.image_rgb, (2, 0, 1)) for frame in frames], axis=0)
    return rgb.astype(np.uint8, copy=False)


def _make_frame_difference(rgb_clip: np.ndarray) -> np.ndarray:
    diffs = np.abs(rgb_clip[1:].astype(np.int16) - rgb_clip[:-1].astype(np.int16))
    return diffs.astype(np.uint8)


def _npy_bytes(array: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, array, allow_pickle=False)
    return bio.getvalue()


def _video_group_key(frame: FrameSample) -> Tuple[str, str, str]:
    return (frame.split, frame.category, _canonical_video_id(frame))


def build_clips_for_video(
    frames: List[FrameSample],
    clip_len: int,
    frame_stride: int,
    clip_stride: int,
) -> List[Dict[str, Any]]:
    if not frames:
        return []

    frames = sorted(frames, key=lambda x: (x.frame_number, x.original_frame_index, x.key))
    needed = (clip_len - 1) * frame_stride + 1
    if len(frames) < needed:
        return []

    samples: List[Dict[str, Any]] = []
    clip_id = 0

    for start in range(0, len(frames) - needed + 1, clip_stride):
        idxs = [start + i * frame_stride for i in range(clip_len)]
        clip_frames = [frames[i] for i in idxs]

        if not _frame_continuity_ok(clip_frames, frame_stride=frame_stride):
            continue

        rgb_clip = _stack_clip_rgb(clip_frames)
        diff_clip = _make_frame_difference(rgb_clip)

        first = clip_frames[0]
        split, category, _ = _video_group_key(first)
        clip_video_path = _clip_key_video_path(first)
        clip_key = f"{clip_video_path}/clip_{clip_id:06d}"

        center_candidates = [2, 3, 4, 5] if clip_len == 8 else list(
            range(max(0, clip_len // 2 - 1), min(clip_len, clip_len // 2 + 2))
        )

        meta = {
            "key": clip_key,
            "split": split,
            "category": category,
            "video_id": first.video_id,
            "video_name": first.video_name,
            "binary_label": first.binary_label,
            "label": "fake" if first.binary_label == 1 else "real",
            "clip_length": clip_len,
            "frame_stride": frame_stride,
            "clip_stride": clip_stride,
            "num_differences": clip_len - 1,
            "frame_numbers": [f.frame_number for f in clip_frames],
            "original_frame_indices": [f.original_frame_index for f in clip_frames],
            "timestamps": [f.timestamp for f in clip_frames],
            "source_keys": [f.key for f in clip_frames],
            "center_candidate_indices": center_candidates,
            "default_center_index": center_candidates[len(center_candidates) // 2] if center_candidates else 0,
            "spatial_sampling_note": "Choose a random center frame from candidate indices during training.",
            "extraction_fps": first.extraction_fps,
            "video_fps": first.video_fps,
            "height": int(rgb_clip.shape[2]),
            "width": int(rgb_clip.shape[3]),
            "rgb_dtype": str(rgb_clip.dtype),
            "diff_dtype": str(diff_clip.dtype),
        }

        sample: Dict[str, Any] = {
            "__key__": clip_key,
            "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            "rgb.npy": _npy_bytes(rgb_clip),
            "diff.npy": _npy_bytes(diff_clip),
        }
        if first.binary_label in {0, 1}:
            sample["cls"] = str(first.binary_label).encode("utf-8")

        samples.append(sample)
        clip_id += 1

    return samples

def _assert_split_output_is_writable(split_output_dir: Path, overwrite: bool) -> None:
    shard_paths = sorted(split_output_dir.glob("shard-*.tar"))
    if not shard_paths:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        return

    if not overwrite:
        raise RuntimeError(
            f"Output already contains shards for split at {split_output_dir}. "
            "Delete the split output or rerun with --overwrite."
        )

    shutil.rmtree(split_output_dir)
    split_output_dir.mkdir(parents=True, exist_ok=True)


def _flush_video_frames(
    *,
    sink: Any,
    frames: List[FrameSample],
    clip_len: int,
    frame_stride: int,
    clip_stride: int,
) -> int:
    clip_samples = build_clips_for_video(
        frames,
        clip_len=clip_len,
        frame_stride=frame_stride,
        clip_stride=clip_stride,
    )
    for sample in clip_samples:
        sink.write(sample)
    return len(clip_samples)


def process_split(
    split: str,
    split_input_dir: Path,
    split_output_dir: Path,
    shard_maxcount: int,
    shard_maxsize: int,
    clip_len: int,
    frame_stride: int,
    clip_stride: int,
    overwrite: bool,
) -> None:
    _assert_split_output_is_writable(split_output_dir, overwrite=overwrite)

    total_frames = 0
    total_videos = 0
    total_clips = 0
    dropped_videos = 0

    shard_pattern = str(split_output_dir / "shard-%06d.tar")
    with ShardWriter(shard_pattern, maxcount=shard_maxcount, maxsize=shard_maxsize) as sink:
        current_video_key: Optional[Tuple[str, str, str]] = None
        current_video_frames: List[FrameSample] = []
        completed_video_keys: set[Tuple[str, str, str]] = set()

        for frame in iter_frame_samples_from_shards(split_input_dir, fallback_split=split):
            total_frames += 1
            group_key = _video_group_key(frame)

            if current_video_key is None:
                current_video_key = group_key
            elif group_key != current_video_key:
                total_videos += 1
                clip_count = _flush_video_frames(
                    sink=sink,
                    frames=current_video_frames,
                    clip_len=clip_len,
                    frame_stride=frame_stride,
                    clip_stride=clip_stride,
                )
                if clip_count == 0:
                    dropped_videos += 1
                total_clips += clip_count
                completed_video_keys.add(current_video_key)
                current_video_frames = []
                current_video_key = group_key

            if group_key in completed_video_keys:
                raise RuntimeError(
                    f"Input shards are not contiguous by video for split={split}: "
                    f"{group_key[2]} reappeared after it was already flushed. "
                    "Rebuild the frame shards from face_detection or keep shard ordering grouped by video."
                )

            current_video_frames.append(frame)

        if current_video_key is not None:
            total_videos += 1
            clip_count = _flush_video_frames(
                sink=sink,
                frames=current_video_frames,
                clip_len=clip_len,
                frame_stride=frame_stride,
                clip_stride=clip_stride,
            )
            if clip_count == 0:
                dropped_videos += 1
            total_clips += clip_count

    logger.info(
        "Split={} | frames={} | videos={} | clips={} | dropped_videos={}",
        split,
        total_frames,
        total_videos,
        total_clips,
        dropped_videos,
    )


def discover_splits(input_dir: Path, split: Optional[str]) -> List[Tuple[str, Path]]:
    if split:
        split_dir = input_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        return [(split, split_dir)]

    found: List[Tuple[str, Path]] = []
    for name in ("train", "val", "test"):
        split_dir = input_dir / name
        if split_dir.exists():
            found.append((name, split_dir))

    if found:
        return found

    shard_paths = sorted(input_dir.glob("shard-*.tar"))
    if shard_paths:
        return [("unspecified", input_dir)]

    raise FileNotFoundError(f"No split directories or shard files found under: {input_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clip-level WebDataset shards (8-frame clips + frame differences) from aligned frame shards"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(settings.CROP_DATA_DIR),
        help="Root directory containing aligned frame shards, typically split subdirectories train/val/test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.ROOT_DIR / "clip_data"),
        help="Output directory for clip-level shards",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=sorted(_VALID_SPLITS),
        help="Process only one split",
    )
    parser.add_argument("--clip-len", type=int, default=8, help="Number of frames per clip")
    parser.add_argument("--frame-stride", type=int, default=1, help="Frame step inside each clip")
    parser.add_argument("--clip-stride", type=int, default=4, help="Clip sliding-window stride")
    parser.add_argument("--shard-maxcount", type=int, default=2000, help="Maximum clips per output shard")
    parser.add_argument("--shard-maxsize", type=int, default=2_000_000_000, help="Maximum bytes per output shard")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output shards for the target split before rebuilding")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    splits = discover_splits(input_dir=input_dir, split=args.split)
    logger.info("Input dir: {}", input_dir)
    logger.info("Output dir: {}", output_dir)
    logger.info("Clip len: {}", args.clip_len)
    logger.info("Frame stride: {}", args.frame_stride)
    logger.info("Clip stride: {}", args.clip_stride)
    logger.info("Overwrite: {}", args.overwrite)

    for split_name, split_input_dir in splits:
        split_output_dir = output_dir / split_name
        process_split(
            split=split_name,
            split_input_dir=split_input_dir,
            split_output_dir=split_output_dir,
            shard_maxcount=args.shard_maxcount,
            shard_maxsize=args.shard_maxsize,
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            clip_stride=args.clip_stride,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
