"""
Rebuild frame_extraction_metadata.csv from existing extracted frame files.

Scans  frame_data/<category>/*.jpg,  reads video properties from the original
FaceForensics++ dataset, and writes a fresh CSV.  Face-detection columns are
left empty — run  face_detection.py  afterwards to populate them.

Usage
-----
    # Rebuild all categories
    uv run -m preprocessing.rebuild_metadata

    # Rebuild a single category only
    uv run -m preprocessing.rebuild_metadata --category original

    # Write to a custom CSV path
    uv run -m preprocessing.rebuild_metadata --output-csv frame_data/meta_new.csv

    # Skip backing up the existing CSV
    uv run -m preprocessing.rebuild_metadata --no-backup
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
from tqdm import tqdm

from configs.loggings import logger, setup_logging
from configs.settings import settings


# ── column schema ─────────────────────────────────────────────────────────────

BASE_COLUMNS = [
    "frame_path",
    "video_name",
    "category",
    "label",
    "frame_number",
    "original_frame_index",
    "timestamp",
    "video_fps",
    "extraction_fps",
    "width",
    "height",
    "video_duration",
    "total_video_frames",
    "extraction_date",
]

FACE_COLUMNS = [
    "face_detected",
    "num_faces",
    "face_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_source",
]

ALL_COLUMNS = BASE_COLUMNS + FACE_COLUMNS


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_video_props(video_path: Path) -> Dict:
    """Return basic properties of a video file, or {} if unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0
    return {
        "video_fps": round(fps, 2),
        "total_video_frames": total_frames,
        "width": width,
        "height": height,
        "video_duration": round(duration, 2),
    }


def _index_video_props(dataset_root: Path, category: str) -> Dict[str, Dict]:
    """Build a {video_stem: props} mapping for all .mp4 files in one category."""
    category_path = dataset_root / category
    if not category_path.exists():
        logger.warning("Dataset category path not found, video props will be empty: {}", category_path)
        return {}

    video_files = sorted(category_path.glob("*.mp4"))
    if not video_files:
        logger.warning("No .mp4 files found in {}", category_path)
        return {}

    props: Dict[str, Dict] = {}
    for vf in tqdm(video_files, desc=f"  indexing {category} videos", leave=False):
        p = _read_video_props(vf)
        if p:
            props[vf.stem] = p

    return props


# ── core rebuild ──────────────────────────────────────────────────────────────

def rebuild_metadata(
    dataset_root: Path,
    frame_root: Path,
    output_csv: Path,
    target_fps: int,
    categories: Dict[str, str],
    category_filter: Optional[str],
    backup: bool,
) -> None:
    """
    Scan extracted frames and write a fresh metadata CSV.

    Parameters
    ----------
    dataset_root:    Root of the original FaceForensics++ dataset (for video props).
    frame_root:      Root of the extracted frame folders.
    output_csv:      Destination CSV path.
    target_fps:      FPS used during extraction – used to recompute original_frame_index.
    categories:      {category_name: label} mapping from settings.
    category_filter: If set, only process this one category.
    backup:          If True and output_csv already exists, rename it to *.bak.csv first.
    """
    if not frame_root.exists():
        raise FileNotFoundError(f"frame_root not found: {frame_root}")

    # ── optional backup ───────────────────────────────────────────────────────
    if backup and output_csv.exists():
        backup_path = output_csv.with_suffix(settings.FACE_DETECTION_BACKUP_SUFFIX)
        if backup_path.exists():
            backup_path.unlink()
        output_csv.rename(backup_path)
        logger.info("Backed up existing CSV → {}", backup_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    count_by_category: Dict[str, int] = {}

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for category, label in categories.items():
            if category_filter and category != category_filter:
                continue

            frame_dir = frame_root / category
            if not frame_dir.exists():
                logger.warning("Frame directory not found, skipping: {}", frame_dir)
                continue

            logger.info("Processing category: {}  (label={})", category, label)
            video_props = _index_video_props(dataset_root, category)

            frame_files = sorted(frame_dir.glob("*.jpg"))
            if not frame_files:
                logger.warning("No frames found in {}", frame_dir)
                continue

            count_by_category[category] = 0

            for frame_path in tqdm(frame_files, desc=f"  {category}"):
                stem = frame_path.stem

                # Expected filename pattern: {video_name}_frame_{N:05d}.jpg
                if "_frame_" not in stem:
                    continue

                video_name, frame_idx_str = stem.rsplit("_frame_", 1)
                try:
                    frame_number = int(frame_idx_str)
                except ValueError:
                    continue

                props = video_props.get(video_name, {})
                vfps: float = float(props.get("video_fps", 0))
                frame_interval = max(1, int(vfps / target_fps)) if vfps else 1
                original_frame_index = frame_number * frame_interval
                timestamp = round(original_frame_index / vfps, 2) if vfps else 0

                extraction_date = datetime.fromtimestamp(
                    frame_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")

                row: Dict[str, object] = {
                    # ── base columns ──────────────────────────────────────────
                    "frame_path": str(frame_path.relative_to(frame_root)),
                    "video_name": video_name,
                    "category": category,
                    "label": label,
                    "frame_number": frame_number,
                    "original_frame_index": original_frame_index,
                    "timestamp": timestamp,
                    "video_fps": vfps,
                    "extraction_fps": target_fps,
                    "width": props.get("width", 0),
                    "height": props.get("height", 0),
                    "video_duration": props.get("video_duration", 0),
                    "total_video_frames": props.get("total_video_frames", 0),
                    "extraction_date": extraction_date,
                    # ── face detection columns (blank – run face_detection.py next) ──
                    "face_detected": "",
                    "num_faces": "",
                    "face_confidence": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "bbox_source": "",
                }
                writer.writerow(row)
                total_rows += 1
                count_by_category[category] += 1

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CSV written to: {}", output_csv)
    logger.info("Total rows: {}", total_rows)
    logger.info("Frames by category:")
    for cat in sorted(count_by_category):
        logger.info("  {:25s}: {} frames", cat, count_by_category[cat])
    logger.info("=" * 60)
    logger.info(
        "Face-detection columns are empty. Run face_detection.py to populate them."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild frame_extraction_metadata.csv from existing extracted frames. "
            "Face-detection columns are left empty."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(settings.FRAME_DATA_DIR / settings.FRAME_EXTRACTION_METADATA_CSV),
        help="Destination CSV path (default: frame_data/frame_extraction_metadata.csv)",
    )
    parser.add_argument(
        "--frame-root",
        type=str,
        default=str(settings.FRAME_DATA_DIR),
        help="Root directory that contains the per-category frame folders",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(settings.RAW_DATA_DIR),
        help="Root directory of the original FaceForensics++ dataset (used to read video properties)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only process one category, e.g. 'original' or 'Deepfakes'",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up the existing CSV before overwriting it",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=settings.TARGET_FPS,
        help=f"Target FPS used during extraction (default: {settings.TARGET_FPS})",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Rebuilding metadata CSV")
    logger.info("  frame_root   : {}", args.frame_root)
    logger.info("  dataset_root : {}", args.dataset_root)
    logger.info("  output_csv   : {}", args.output_csv)
    logger.info("  category     : {}", args.category or "(all)")
    logger.info("  target_fps   : {}", args.target_fps)
    logger.info("=" * 60)

    rebuild_metadata(
        dataset_root=Path(args.dataset_root),
        frame_root=Path(args.frame_root),
        output_csv=Path(args.output_csv),
        target_fps=args.target_fps,
        categories=dict(settings.DATASET_CATEGORIES),
        category_filter=args.category,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
