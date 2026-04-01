"""
Optimized Frame Extractor for FaceForensics++ Dataset

Key improvements:
- Sequential decode instead of repeated frame seeking
- Incremental CSV writing (low memory)
- Better error logging
- More accurate FPS sampling using timestamp accumulation
- Per-video timing stats
- Safer multiprocessing
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from configs.loggings import logger, setup_logging
from configs.settings import settings

warnings.filterwarnings("ignore")


CSV_FIELDNAMES = [
    "frame_path",
    "video_id",
    "video_path",
    "video_name",
    "category",
    "label",
    "binary_label",
    "split",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_csv(csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_rows_to_csv(csv_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerows(rows)


def get_video_info(video_path: Path) -> Optional[Tuple[cv2.VideoCapture, float, int, int, int, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    duration = total_frames / video_fps
    return cap, video_fps, total_frames, width, height, duration


def process_video_standalone(
    video_path_str: str,
    category: str,
    output_dir_str: str,
    base_output_path_str: str,
    target_fps: int,
    jpeg_quality: int,
    label_map: Dict[str, str],
    video_id: Optional[str] = None,
    video_rel_path: Optional[str] = None,
    binary_label: Optional[int] = None,
    split: Optional[str] = None,
) -> Dict:
    """
    Process a single video using sequential decode.

    Returns:
        {
            "rows": List[Dict],
            "stats": Dict,
            "error": Optional[str],
        }
    """
    t0 = time.perf_counter()

    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    base_output_path = Path(base_output_path_str)

    rows: List[Dict] = []
    error: Optional[str] = None

    try:
        info = get_video_info(video_path)
        if info is None:
            return {
                "rows": [],
                "stats": {
                    "video_name": video_path.stem,
                    "category": category,
                    "saved_frames": 0,
                    "elapsed_sec": round(time.perf_counter() - t0, 3),
                },
                "error": f"Cannot open or invalid video metadata: {video_path}",
            }

        cap, video_fps, total_frames, width, height, duration = info
        extraction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_name = video_path.stem
        resolved_video_id = video_id or video_name
        resolved_binary_label = binary_label if binary_label is not None else int(label_map[category] != "real")
        resolved_video_rel_path = video_rel_path or str(video_path)
        resolved_split = split or ""
        resolved_label = label_map.get(category, "fake" if resolved_binary_label else "real")

        # More stable sampling than int(video_fps / target_fps)
        sample_period = 1.0 / max(target_fps, 1)
        next_sample_ts = 0.0

        frame_idx = 0
        saved_count = 0

        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = frame_idx / video_fps

            # Save frame when current timestamp reaches next sampling point
            if ts + 1e-9 >= next_sample_ts:
                frame_filename = f"{video_name}_frame_{saved_count:05d}.jpg"
                frame_path = output_dir / frame_filename

                ok = cv2.imwrite(str(frame_path), frame, imwrite_params)
                if ok:
                    rows.append(
                        {
                            "frame_path": str(frame_path.relative_to(base_output_path)),
                            "video_id": resolved_video_id,
                            "video_path": resolved_video_rel_path,
                            "video_name": video_name,
                            "category": category,
                            "label": resolved_label,
                            "binary_label": resolved_binary_label,
                            "split": resolved_split,
                            "frame_number": saved_count,
                            "original_frame_index": frame_idx,
                            "timestamp": round(ts, 4),
                            "video_fps": round(video_fps, 4),
                            "extraction_fps": target_fps,
                            "width": width,
                            "height": height,
                            "video_duration": round(duration, 4),
                            "total_video_frames": total_frames,
                            "extraction_date": extraction_date,
                        }
                    )
                    saved_count += 1
                    next_sample_ts += sample_period

            frame_idx += 1

        cap.release()

        elapsed = time.perf_counter() - t0
        return {
            "rows": rows,
            "stats": {
                "video_name": video_name,
                "category": category,
                "saved_frames": saved_count,
                "elapsed_sec": round(elapsed, 3),
                "video_fps": round(video_fps, 3),
                "duration_sec": round(duration, 3),
                "total_frames": total_frames,
            },
            "error": None,
        }

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    return {
        "rows": [],
        "stats": {
            "video_name": Path(video_path_str).stem,
            "category": category,
            "saved_frames": 0,
            "elapsed_sec": round(elapsed, 3),
        },
        "error": error,
    }


class FrameExtractor:
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        fps: Optional[int] = None,
        metadata_csv: Optional[str] = None,
        manifest_path: Optional[Path] = None,
        jpeg_quality: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        default_dataset_path = settings.RAW_DATA_DIR
        if not default_dataset_path.exists():
            default_dataset_path = settings.ROOT_DIR / "FaceForensics++_C23"

        self.dataset_path = dataset_path or default_dataset_path
        self.output_path = output_path or settings.FRAME_DATA_DIR
        self.target_fps = fps or settings.TARGET_FPS
        self.metadata_csv = metadata_csv or settings.FRAME_EXTRACTION_METADATA_CSV
        self.manifest_path = manifest_path or settings.ROOT_DIR / "artifacts" / "videos_master.csv"
        self.jpeg_quality = jpeg_quality or settings.JPEG_QUALITY
        self.num_workers = num_workers or settings.NUM_WORKERS
        self.categories = settings.DATASET_CATEGORIES
        self.manifest_rows = self._load_video_manifest()
        self.manifest_index = {
            (str(row["category"]), str(row["video_id"])): row for row in self.manifest_rows
        }

    def _load_video_manifest(self) -> List[Dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Video manifest not found: {self.manifest_path}")

        required_columns = {"video_id", "video_path", "category"}
        manifest_rows: List[Dict[str, Any]] = []

        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(required_columns - fieldnames)
            if missing_columns:
                raise ValueError(
                    f"Manifest is missing required columns: {', '.join(missing_columns)}"
                )

            for row in reader:
                category = str(row.get("category", "")).strip()
                video_id = str(row.get("video_id", "")).strip()
                video_rel_path = str(row.get("video_path", "")).strip()
                split = str(row.get("split", "")).strip().lower()

                if not category or not video_id or not video_rel_path:
                    continue

                binary_label_raw = str(row.get("binary_label", "")).strip()
                if binary_label_raw in {"0", "1"}:
                    binary_label = int(binary_label_raw)
                else:
                    binary_label = 0 if self.categories.get(category) == "real" else 1

                label = "real" if binary_label == 0 else "fake"
                absolute_video_path = (self.dataset_path / PurePosixPath(video_rel_path)).resolve()

                manifest_rows.append(
                    {
                        "video_id": video_id,
                        "video_path": video_rel_path,
                        "category": category,
                        "binary_label": binary_label,
                        "label": label,
                        "split": split,
                        "absolute_video_path": absolute_video_path,
                    }
                )

        return manifest_rows

    def _select_manifest_rows(
        self,
        category: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = self.manifest_rows

        if category:
            rows = [row for row in rows if row["category"] == category]

        if split:
            split = split.strip().lower()
            rows = [row for row in rows if row["split"] == split]

        return rows

    def _load_video_properties_for_category(self, category: str) -> Dict[str, Dict]:
        video_props: Dict[str, Dict] = {}
        category_path = self.dataset_path / category

        if not category_path.exists():
            return video_props

        video_files = list(category_path.glob("*.mp4"))
        for video_path in tqdm(video_files, desc=f"Indexing {category}", leave=False):
            info = get_video_info(video_path)
            if info is None:
                continue

            cap, fps, total_frames, width, height, duration = info
            cap.release()

            video_props[video_path.stem] = {
                "video_fps": round(fps, 4),
                "total_video_frames": total_frames,
                "width": width,
                "height": height,
                "video_duration": round(duration, 4),
            }

        return video_props

    def rebuild_metadata_from_frames(self) -> None:
        csv_path = self.output_path / self.metadata_csv
        init_csv(csv_path)

        total_rows = 0
        real_rows = 0
        fake_rows = 0
        category_counts: Dict[str, int] = {}

        for category, label in self.categories.items():
            frame_dir = self.output_path / category
            if not frame_dir.exists():
                logger.warning(f"Frame directory not found: {frame_dir}")
                continue

            video_props = self._load_video_properties_for_category(category)
            frame_files = sorted(frame_dir.glob("*.jpg"))
            category_counts[category] = 0
            batch_rows: List[Dict] = []

            for frame_path in tqdm(frame_files, desc=f"Rebuilding {category}"):
                stem = frame_path.stem
                if "_frame_" not in stem:
                    continue

                video_name, frame_idx_str = stem.rsplit("_frame_", 1)
                try:
                    frame_number = int(frame_idx_str)
                except ValueError:
                    continue

                props = video_props.get(video_name, {})
                manifest_row = self.manifest_index.get((category, video_name), {})
                video_fps = props.get("video_fps", 0.0)
                timestamp = round(frame_number / self.target_fps, 4) if self.target_fps > 0 else 0.0
                original_frame_index = int(round(timestamp * video_fps)) if video_fps > 0 else 0
                extraction_date = datetime.fromtimestamp(frame_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                row = {
                    "frame_path": str(frame_path.relative_to(self.output_path)),
                    "video_id": manifest_row.get("video_id", video_name),
                    "video_path": manifest_row.get("video_path", ""),
                    "video_name": video_name,
                    "category": category,
                    "label": manifest_row.get("label", label),
                    "binary_label": manifest_row.get("binary_label", 0 if label == "real" else 1),
                    "split": manifest_row.get("split", ""),
                    "frame_number": frame_number,
                    "original_frame_index": original_frame_index,
                    "timestamp": timestamp,
                    "video_fps": video_fps,
                    "extraction_fps": self.target_fps,
                    "width": props.get("width", 0),
                    "height": props.get("height", 0),
                    "video_duration": props.get("video_duration", 0),
                    "total_video_frames": props.get("total_video_frames", 0),
                    "extraction_date": extraction_date,
                }
                batch_rows.append(row)

                total_rows += 1
                category_counts[category] += 1
                if label == "real":
                    real_rows += 1
                else:
                    fake_rows += 1

                if len(batch_rows) >= 5000:
                    append_rows_to_csv(csv_path, batch_rows)
                    batch_rows.clear()

            if batch_rows:
                append_rows_to_csv(csv_path, batch_rows)

        logger.info(f"Metadata rebuilt and saved to {csv_path}")
        logger.info(f"Total frames indexed: {total_rows}")
        logger.info(f"Real frames: {real_rows}")
        logger.info(f"Fake frames: {fake_rows}")
        logger.info("Frames by category:")
        for category in sorted(category_counts.keys()):
            logger.info(f"  {category}: {category_counts[category]} frames")

    def process_category(self, category: str, csv_path: Path, split: Optional[str] = None) -> int:
        output_dir = self.output_path / category
        ensure_dir(output_dir)

        manifest_rows = self._select_manifest_rows(category=category, split=split)

        if settings.MAX_VIDEOS_PER_CATEGORY:
            manifest_rows = manifest_rows[: settings.MAX_VIDEOS_PER_CATEGORY]
            logger.info(f"Testing mode: limited to {len(manifest_rows)} videos for {category}")

        if not manifest_rows:
            logger.warning(
                f"No videos found in manifest for category={category}"
                + (f", split={split}" if split else "")
            )
            return 0

        missing_files = [
            str(row["absolute_video_path"]) for row in manifest_rows if not Path(row["absolute_video_path"]).exists()
        ]
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} videos for {category}; they will be skipped")

        manifest_rows = [row for row in manifest_rows if Path(row["absolute_video_path"]).exists()]
        if not manifest_rows:
            logger.warning(f"All manifest videos are missing for category={category}")
            return 0

        logger.info(
            f"Processing {len(manifest_rows)} videos from manifest for {category} "
            f"with {self.num_workers} workers"
            + (f" | split={split}" if split else "")
        )

        total_saved = 0
        errors = 0

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        process_video_standalone,
                        str(row["absolute_video_path"]),
                        category,
                        str(output_dir),
                        str(self.output_path),
                        self.target_fps,
                        self.jpeg_quality,
                        self.categories,
                        str(row["video_id"]),
                        str(row["video_path"]),
                        int(row["binary_label"]),
                        str(row["split"]),
                    ): row
                    for row in manifest_rows
                }

                with tqdm(total=len(manifest_rows), desc=f"Processing {category}") as pbar:
                    for future in as_completed(futures):
                        row = futures[future]
                        try:
                            result = future.result()
                            rows = result["rows"]
                            stats = result["stats"]
                            error = result["error"]

                            append_rows_to_csv(csv_path, rows)
                            total_saved += len(rows)

                            if error:
                                errors += 1
                                logger.warning(f"[{category}] {Path(row['absolute_video_path']).name}: {error}")
                            else:
                                logger.info(
                                    f"[{category}] {stats['video_name']} | "
                                    f"saved={stats['saved_frames']} | "
                                    f"time={stats['elapsed_sec']}s"
                                )
                        except Exception as e:
                            errors += 1
                            logger.error(
                                f"Worker failed for {row['absolute_video_path']}: {type(e).__name__}: {e}"
                            )
                        pbar.update(1)
        else:
            for row in tqdm(manifest_rows, desc=f"Processing {category}"):
                result = process_video_standalone(
                    str(row["absolute_video_path"]),
                    category,
                    str(output_dir),
                    str(self.output_path),
                    self.target_fps,
                    self.jpeg_quality,
                    self.categories,
                    str(row["video_id"]),
                    str(row["video_path"]),
                    int(row["binary_label"]),
                    str(row["split"]),
                )
                append_rows_to_csv(csv_path, result["rows"])
                total_saved += len(result["rows"])

                if result["error"]:
                    errors += 1
                    logger.warning(f"[{category}] {Path(row['absolute_video_path']).name}: {result['error']}")
                else:
                    stats = result["stats"]
                    logger.info(
                        f"[{category}] {stats['video_name']} | "
                        f"saved={stats['saved_frames']} | "
                        f"time={stats['elapsed_sec']}s"
                    )

        logger.info(f"Completed {category}: saved={total_saved}, errors={errors}")
        return total_saved

    def extract_all(self, only_category: Optional[str] = None, only_split: Optional[str] = None) -> None:
        t0 = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Starting optimized frame extraction")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Video manifest: {self.manifest_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"JPEG Quality: {self.jpeg_quality}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info("=" * 60)

        ensure_dir(self.output_path)

        csv_path = self.output_path / self.metadata_csv
        init_csv(csv_path)

        categories = [only_category] if only_category else list(self.categories.keys())
        total_saved = 0

        for category in categories:
            if category not in self.categories:
                logger.warning(f"Unknown category skipped: {category}")
                continue
            total_saved += self.process_category(category, csv_path, split=only_split)

        elapsed = time.perf_counter() - t0
        logger.info("=" * 60)
        logger.info(f"Frame extraction completed | total_saved={total_saved} | elapsed={elapsed:.2f}s")
        logger.info(f"Metadata CSV: {csv_path}")
        logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized frame extractor and metadata builder")
    parser.add_argument("--rebuild-csv-only", action="store_true", help="Rebuild metadata CSV from existing frames")
    parser.add_argument("--category", type=str, default=None, help="Process only one category")
    parser.add_argument("--split", type=str, default=None, help="Process only one split from videos_master.csv")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to video-level manifest CSV, defaults to artifacts/videos_master.csv",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override number of workers")
    parser.add_argument("--fps", type=int, default=None, help="Override target FPS")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="Override JPEG quality (0-100)")
    args = parser.parse_args()

    setup_logging()

    extractor = FrameExtractor(
        manifest_path=args.manifest,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        num_workers=args.workers,
    )

    if args.rebuild_csv_only:
        extractor.rebuild_metadata_from_frames()
    else:
        extractor.extract_all(only_category=args.category, only_split=args.split)


if __name__ == "__main__":
    main()
