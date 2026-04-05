"""
Optimized Frame Extractor for FaceForensics++ Dataset

Key improvements:
- Sequential decode instead of repeated frame seeking
- Incremental CSV writing (low memory)
- Better error logging
- More accurate FPS sampling using timestamp accumulation
- Per-video timing stats
- Safer multiprocessing
- Resume support via audit + existing frame scan
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

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

AUDIT_FIELDNAMES = [
    "category",
    "video_id",
    "split",
    "status",
    "updated_at",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_csv(csv_path: Path, fieldnames: List[str]) -> None:
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def ensure_csv_with_header(csv_path: Path, fieldnames: List[str]) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    init_csv(csv_path, fieldnames)


def append_rows_to_csv(csv_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not rows:
        return
    ensure_csv_with_header(csv_path, fieldnames)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_split(value: Any) -> str:
    return str(value or "").strip().lower()


def build_video_key(category: str, video_id: str, split: Any) -> str:
    return f"{category}|{video_id}|{normalize_split(split)}"


def build_video_key_from_row(row: Dict[str, Any]) -> str:
    return build_video_key(
        str(row.get("category", "")).strip(),
        str(row.get("video_id", "")).strip(),
        row.get("split", ""),
    )


def build_frame_filename(video_name: str, frame_number: int) -> str:
    return f"{video_name}_frame_{frame_number:05d}.jpg"


def parse_frame_filename(frame_path: Path) -> Optional[Tuple[str, int]]:
    stem = frame_path.stem
    if "_frame_" not in stem:
        return None

    video_name, frame_idx_str = stem.rsplit("_frame_", 1)
    try:
        frame_number = int(frame_idx_str)
    except ValueError:
        return None

    return video_name, frame_number


def compute_sample_frame_indices(video_fps: float, total_frames: int, target_fps: int) -> List[int]:
    if video_fps <= 0 or total_frames <= 0:
        return []

    sample_period = 1.0 / max(target_fps, 1)
    next_sample_ts = 0.0
    sample_indices: List[int] = []

    for frame_idx in range(total_frames):
        ts = frame_idx / video_fps
        if ts + 1e-9 >= next_sample_ts:
            sample_indices.append(frame_idx)
            next_sample_ts += sample_period

    return sample_indices


def scan_video_frame_map(output_dir: Path, video_name: str) -> Dict[int, Path]:
    frame_map: Dict[int, Path] = {}
    for frame_path in sorted(output_dir.glob(f"{video_name}_frame_*.jpg")):
        parsed = parse_frame_filename(frame_path)
        if parsed is None:
            continue
        parsed_video_name, frame_number = parsed
        if parsed_video_name != video_name:
            continue
        frame_map[frame_number] = frame_path
    return frame_map


def contiguous_prefix_count(frame_map: Dict[int, Path]) -> int:
    count = 0
    while count in frame_map:
        count += 1
    return count


def build_frame_row(
    *,
    frame_path: Path,
    base_output_path: Path,
    video_id: str,
    video_rel_path: str,
    video_name: str,
    category: str,
    label: str,
    binary_label: int,
    split: str,
    frame_number: int,
    original_frame_index: int,
    timestamp: float,
    video_fps: float,
    target_fps: int,
    width: int,
    height: int,
    duration: float,
    total_frames: int,
    extraction_date: str,
) -> Dict[str, Any]:
    return {
        "frame_path": str(frame_path.relative_to(base_output_path)),
        "video_id": video_id,
        "video_path": video_rel_path,
        "video_name": video_name,
        "category": category,
        "label": label,
        "binary_label": binary_label,
        "split": split,
        "frame_number": frame_number,
        "original_frame_index": original_frame_index,
        "timestamp": round(timestamp, 4),
        "video_fps": round(video_fps, 4),
        "extraction_fps": target_fps,
        "width": width,
        "height": height,
        "video_duration": round(duration, 4),
        "total_video_frames": total_frames,
        "extraction_date": extraction_date,
    }


def build_rows_from_existing_frames(
    *,
    frame_paths: List[Path],
    sample_indices: List[int],
    video_fps: float,
    total_frames: int,
    width: int,
    height: int,
    duration: float,
    base_output_path: Path,
    video_id: str,
    video_rel_path: str,
    video_name: str,
    category: str,
    label: str,
    binary_label: int,
    split: str,
    target_fps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for frame_number, frame_path in enumerate(frame_paths):
        if frame_number >= len(sample_indices):
            break
        original_frame_index = sample_indices[frame_number]
        extraction_date = datetime.fromtimestamp(frame_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            build_frame_row(
                frame_path=frame_path,
                base_output_path=base_output_path,
                video_id=video_id,
                video_rel_path=video_rel_path,
                video_name=video_name,
                category=category,
                label=label,
                binary_label=binary_label,
                split=split,
                frame_number=frame_number,
                original_frame_index=original_frame_index,
                timestamp=original_frame_index / video_fps,
                video_fps=video_fps,
                target_fps=target_fps,
                width=width,
                height=height,
                duration=duration,
                total_frames=total_frames,
                extraction_date=extraction_date,
            )
        )

    return rows


def append_audit_row(audit_csv: Path, category: str, video_id: str, split: str) -> None:
    append_rows_to_csv(
        audit_csv,
        [
            {
                "category": category,
                "video_id": video_id,
                "split": normalize_split(split),
                "status": "complete",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
        AUDIT_FIELDNAMES,
    )


def load_completed_video_keys(audit_csv: Path) -> Set[str]:
    completed_keys: Set[str] = set()
    if not audit_csv.exists() or audit_csv.stat().st_size == 0:
        return completed_keys

    with open(audit_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "complete":
                continue
            completed_keys.add(build_video_key_from_row(row))

    return completed_keys


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
    resume_existing_count: int = 0,
) -> Dict[str, Any]:
    """
    Process a single video using sequential decode and resume from existing frames.

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

    rows: List[Dict[str, Any]] = []
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
                    "new_frames": 0,
                    "resumed_frames": 0,
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
        resolved_split = normalize_split(split)
        resolved_label = label_map.get(category, "fake" if resolved_binary_label else "real")

        sample_indices = compute_sample_frame_indices(video_fps, total_frames, target_fps)
        expected_samples = len(sample_indices)
        existing_frame_map = scan_video_frame_map(output_dir, video_name)
        contiguous_existing = contiguous_prefix_count(existing_frame_map)
        existing_count = min(max(0, int(resume_existing_count)), contiguous_existing, expected_samples)
        existing_frame_paths = [existing_frame_map[idx] for idx in range(existing_count)]
        rows = build_rows_from_existing_frames(
            frame_paths=existing_frame_paths,
            sample_indices=sample_indices,
            video_fps=video_fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration=duration,
            base_output_path=base_output_path,
            video_id=resolved_video_id,
            video_rel_path=resolved_video_rel_path,
            video_name=video_name,
            category=category,
            label=resolved_label,
            binary_label=resolved_binary_label,
            split=resolved_split,
            target_fps=target_fps,
        )

        if existing_count >= expected_samples:
            cap.release()
            elapsed = time.perf_counter() - t0
            return {
                "rows": rows,
                "stats": {
                    "video_name": video_name,
                    "category": category,
                    "saved_frames": len(rows),
                    "new_frames": 0,
                    "resumed_frames": existing_count,
                    "elapsed_sec": round(elapsed, 3),
                    "video_fps": round(video_fps, 3),
                    "duration_sec": round(duration, 3),
                    "total_frames": total_frames,
                },
                "error": None,
            }

        sample_period = 1.0 / max(target_fps, 1)
        next_sample_ts = existing_count * sample_period
        saved_count = existing_count

        if existing_count > 0:
            next_original_frame_index = sample_indices[existing_count]
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_original_frame_index)
            frame_idx = next_original_frame_index
        else:
            frame_idx = 0

        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        while saved_count < expected_samples:
            ret, frame = cap.read()
            if not ret:
                break

            ts = frame_idx / video_fps
            if ts + 1e-9 >= next_sample_ts:
                frame_filename = build_frame_filename(video_name, saved_count)
                frame_path = output_dir / frame_filename
                ok = cv2.imwrite(str(frame_path), frame, imwrite_params)
                if ok:
                    rows.append(
                        build_frame_row(
                            frame_path=frame_path,
                            base_output_path=base_output_path,
                            video_id=resolved_video_id,
                            video_rel_path=resolved_video_rel_path,
                            video_name=video_name,
                            category=category,
                            label=resolved_label,
                            binary_label=resolved_binary_label,
                            split=resolved_split,
                            frame_number=saved_count,
                            original_frame_index=frame_idx,
                            timestamp=ts,
                            video_fps=video_fps,
                            target_fps=target_fps,
                            width=width,
                            height=height,
                            duration=duration,
                            total_frames=total_frames,
                            extraction_date=extraction_date,
                        )
                    )
                    saved_count += 1
                    next_sample_ts += sample_period

            frame_idx += 1

        cap.release()
        if saved_count < expected_samples:
            error = f"Incomplete extraction: expected {expected_samples} frames, got {saved_count}"

        elapsed = time.perf_counter() - t0
        return {
            "rows": rows,
            "stats": {
                "video_name": video_name,
                "category": category,
                "saved_frames": len(rows),
                "new_frames": max(0, len(rows) - existing_count),
                "resumed_frames": existing_count,
                "elapsed_sec": round(elapsed, 3),
                "video_fps": round(video_fps, 3),
                "duration_sec": round(duration, 3),
                "total_frames": total_frames,
            },
            "error": error,
        }

    except Exception as e:
        if "cap" in locals():
            cap.release()
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    return {
        "rows": [],
        "stats": {
            "video_name": Path(video_path_str).stem,
            "category": category,
            "saved_frames": 0,
            "new_frames": 0,
            "resumed_frames": 0,
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
        resume: bool = True,
        audit_csv: Optional[str] = None,
    ):
        default_dataset_path = settings.RAW_DATA_DIR
        if not default_dataset_path.exists():
            default_dataset_path = settings.ROOT_DIR / "FaceForensics++_C23"

        self.dataset_path = dataset_path or default_dataset_path
        self.output_path = output_path or settings.FRAME_DATA_DIR
        self.target_fps = fps or settings.TARGET_FPS
        self.metadata_csv = metadata_csv or settings.FRAME_EXTRACTION_METADATA_CSV
        self.audit_csv = audit_csv or settings.FRAME_EXTRACTION_AUDIT_CSV
        self.manifest_path = manifest_path or settings.ROOT_DIR / "artifacts" / "videos_master.csv"
        self.jpeg_quality = jpeg_quality or settings.JPEG_QUALITY
        self.num_workers = num_workers or settings.NUM_WORKERS
        self.resume = resume
        self.categories = settings.DATASET_CATEGORIES
        self.manifest_rows = self._load_video_manifest()
        self.manifest_index = {
            (str(row["category"]), str(row["video_id"])): row for row in self.manifest_rows
        }
        self.manifest_key_index = {
            build_video_key(str(row["category"]), str(row["video_id"]), row.get("split", "")): row
            for row in self.manifest_rows
        }
        self.expected_sample_count_cache: Dict[str, Optional[int]] = {}
        self.completed_video_keys: Set[str] = set()

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
                split = normalize_split(row.get("split", ""))

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
                        "original_fps": safe_float(row.get("original_fps"), 0.0),
                        "num_frames": safe_int(row.get("num_frames"), 0),
                    }
                )

        return manifest_rows

    def _load_metadata_rows(self, csv_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        rows: List[Dict[str, str]] = []
        grouped: Dict[str, List[Dict[str, str]]] = {}

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return rows, grouped

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = build_video_key_from_row(row)
                rows.append(row)
                grouped.setdefault(key, []).append(row)

        return rows, grouped

    def _expected_sample_count(self, manifest_row: Dict[str, Any]) -> Optional[int]:
        key = build_video_key(str(manifest_row["category"]), str(manifest_row["video_id"]), manifest_row.get("split", ""))
        if key in self.expected_sample_count_cache:
            return self.expected_sample_count_cache[key]

        video_fps = safe_float(manifest_row.get("original_fps"), 0.0)
        total_frames = safe_int(manifest_row.get("num_frames"), 0)
        if video_fps <= 0 or total_frames <= 0:
            info = get_video_info(Path(manifest_row["absolute_video_path"]))
            if info is None:
                self.expected_sample_count_cache[key] = None
                return None
            cap, video_fps, total_frames, _, _, _ = info
            cap.release()

        expected_count = len(compute_sample_frame_indices(video_fps, total_frames, self.target_fps))
        self.expected_sample_count_cache[key] = expected_count
        return expected_count

    def _is_metadata_complete(
        self,
        manifest_row: Dict[str, Any],
        metadata_rows: List[Dict[str, str]],
    ) -> bool:
        expected_count = self._expected_sample_count(manifest_row)
        if expected_count is None or len(metadata_rows) != expected_count:
            return False

        try:
            frame_numbers = sorted(int(row.get("frame_number", "")) for row in metadata_rows)
        except ValueError:
            return False

        if frame_numbers != list(range(expected_count)):
            return False

        output_dir = self.output_path / str(manifest_row["category"])
        frame_map = scan_video_frame_map(output_dir, str(manifest_row["video_id"]))
        if len(frame_map) != expected_count:
            return False

        return contiguous_prefix_count(frame_map) == expected_count

    def _bootstrap_completed_keys_from_metadata(
        self,
        audit_path: Path,
        metadata_groups: Dict[str, List[Dict[str, str]]],
    ) -> Set[str]:
        bootstrapped_keys: Set[str] = set()

        for key, rows in metadata_groups.items():
            manifest_row = self.manifest_key_index.get(key)
            if manifest_row is None:
                continue
            if self._is_metadata_complete(manifest_row, rows):
                bootstrapped_keys.add(key)

        if not bootstrapped_keys:
            return bootstrapped_keys

        init_csv(audit_path, AUDIT_FIELDNAMES)
        for key in sorted(bootstrapped_keys):
            manifest_row = self.manifest_key_index[key]
            append_audit_row(
                audit_path,
                str(manifest_row["category"]),
                str(manifest_row["video_id"]),
                str(manifest_row.get("split", "")),
            )

        logger.info(f"Bootstrapped {len(bootstrapped_keys)} completed videos from existing metadata")
        return bootstrapped_keys

    def _rebuild_completed_video_rows(self, manifest_row: Dict[str, Any]) -> List[Dict[str, Any]]:
        output_dir = self.output_path / str(manifest_row["category"])
        info = get_video_info(Path(manifest_row["absolute_video_path"]))
        if info is None:
            return []

        cap, video_fps, total_frames, width, height, duration = info
        cap.release()

        sample_indices = compute_sample_frame_indices(video_fps, total_frames, self.target_fps)
        expected_count = len(sample_indices)
        frame_map = scan_video_frame_map(output_dir, str(manifest_row["video_id"]))
        if contiguous_prefix_count(frame_map) < expected_count:
            return []

        frame_paths = [frame_map[idx] for idx in range(expected_count)]
        return build_rows_from_existing_frames(
            frame_paths=frame_paths,
            sample_indices=sample_indices,
            video_fps=video_fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration=duration,
            base_output_path=self.output_path,
            video_id=str(manifest_row["video_id"]),
            video_rel_path=str(manifest_row["video_path"]),
            video_name=str(manifest_row["video_id"]),
            category=str(manifest_row["category"]),
            label=str(manifest_row["label"]),
            binary_label=int(manifest_row["binary_label"]),
            split=str(manifest_row.get("split", "")),
            target_fps=self.target_fps,
        )

    def _prepare_resume_state(self, csv_path: Path, audit_path: Path) -> Set[str]:
        ensure_dir(self.output_path)
        completed_keys = load_completed_video_keys(audit_path)
        metadata_rows, metadata_groups = self._load_metadata_rows(csv_path)

        if not completed_keys and metadata_groups:
            completed_keys = self._bootstrap_completed_keys_from_metadata(audit_path, metadata_groups)

        completed_rows = [row for row in metadata_rows if build_video_key_from_row(row) in completed_keys]
        completed_row_keys = {build_video_key_from_row(row) for row in completed_rows}
        missing_completed_keys = sorted(completed_keys - completed_row_keys)
        needs_rewrite = (
            not csv_path.exists()
            or any(key not in completed_keys for key in metadata_groups)
            or bool(missing_completed_keys)
        )

        if needs_rewrite:
            recovered_rows: List[Dict[str, Any]] = []
            for key in missing_completed_keys:
                manifest_row = self.manifest_key_index.get(key)
                if manifest_row is None:
                    continue
                recovered_rows.extend(self._rebuild_completed_video_rows(manifest_row))

            init_csv(csv_path, CSV_FIELDNAMES)
            append_rows_to_csv(csv_path, completed_rows + recovered_rows, CSV_FIELDNAMES)
        else:
            ensure_csv_with_header(csv_path, CSV_FIELDNAMES)

        return completed_keys

    def _select_manifest_rows(
        self,
        category: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = self.manifest_rows

        if category:
            rows = [row for row in rows if row["category"] == category]

        if split:
            split = normalize_split(split)
            rows = [row for row in rows if row["split"] == split]

        return rows

    def _load_video_properties_for_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        video_props: Dict[str, Dict[str, Any]] = {}
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
                "sample_indices": compute_sample_frame_indices(fps, total_frames, self.target_fps),
            }

        return video_props

    def rebuild_metadata_from_frames(self) -> None:
        csv_path = self.output_path / self.metadata_csv
        init_csv(csv_path, CSV_FIELDNAMES)

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
            batch_rows: List[Dict[str, Any]] = []

            for frame_path in tqdm(frame_files, desc=f"Rebuilding {category}"):
                parsed = parse_frame_filename(frame_path)
                if parsed is None:
                    continue

                video_name, frame_number = parsed
                props = video_props.get(video_name, {})
                sample_indices = props.get("sample_indices", [])
                if frame_number >= len(sample_indices):
                    continue

                manifest_row = self.manifest_index.get((category, video_name), {})
                video_fps = safe_float(props.get("video_fps"), 0.0)
                if video_fps <= 0:
                    continue

                original_frame_index = sample_indices[frame_number]
                extraction_date = datetime.fromtimestamp(frame_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                batch_rows.append(
                    build_frame_row(
                        frame_path=frame_path,
                        base_output_path=self.output_path,
                        video_id=str(manifest_row.get("video_id", video_name)),
                        video_rel_path=str(manifest_row.get("video_path", "")),
                        video_name=video_name,
                        category=category,
                        label=str(manifest_row.get("label", label)),
                        binary_label=int(manifest_row.get("binary_label", 0 if label == "real" else 1)),
                        split=str(manifest_row.get("split", "")),
                        frame_number=frame_number,
                        original_frame_index=original_frame_index,
                        timestamp=original_frame_index / video_fps,
                        video_fps=video_fps,
                        target_fps=self.target_fps,
                        width=int(props.get("width", 0)),
                        height=int(props.get("height", 0)),
                        duration=safe_float(props.get("video_duration"), 0.0),
                        total_frames=int(props.get("total_video_frames", 0)),
                        extraction_date=extraction_date,
                    )
                )

                total_rows += 1
                category_counts[category] += 1
                if label == "real":
                    real_rows += 1
                else:
                    fake_rows += 1

                if len(batch_rows) >= 5000:
                    append_rows_to_csv(csv_path, batch_rows, CSV_FIELDNAMES)
                    batch_rows.clear()

            if batch_rows:
                append_rows_to_csv(csv_path, batch_rows, CSV_FIELDNAMES)

        logger.info(f"Metadata rebuilt and saved to {csv_path}")
        logger.info(f"Total frames indexed: {total_rows}")
        logger.info(f"Real frames: {real_rows}")
        logger.info(f"Fake frames: {fake_rows}")
        logger.info("Frames by category:")
        for category in sorted(category_counts.keys()):
            logger.info(f"  {category}: {category_counts[category]} frames")

    def process_category(
        self,
        category: str,
        csv_path: Path,
        audit_path: Path,
        split: Optional[str] = None,
    ) -> int:
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

        pending_rows: List[Dict[str, Any]] = []
        skipped_completed = 0

        for row in manifest_rows:
            video_key = build_video_key(str(row["category"]), str(row["video_id"]), row.get("split", ""))
            if self.resume and video_key in self.completed_video_keys:
                skipped_completed += 1
                continue

            row_copy = dict(row)
            if self.resume:
                frame_map = scan_video_frame_map(output_dir, str(row_copy["video_id"]))
                row_copy["resume_existing_count"] = contiguous_prefix_count(frame_map)
            else:
                row_copy["resume_existing_count"] = 0

            pending_rows.append(row_copy)

        if skipped_completed:
            logger.info(f"Skipping {skipped_completed} already completed videos for {category}")

        if not pending_rows:
            logger.info(f"No pending videos for {category}")
            return 0

        logger.info(
            f"Processing {len(pending_rows)} videos from manifest for {category} "
            f"with {self.num_workers} workers"
            + (f" | split={split}" if split else "")
        )

        total_new_frames = 0
        errors = 0

        def commit_video_result(row: Dict[str, Any], result: Dict[str, Any]) -> None:
            nonlocal total_new_frames, errors

            video_key = build_video_key(str(row["category"]), str(row["video_id"]), row.get("split", ""))
            rows = result["rows"]
            stats = result["stats"]
            error = result["error"]

            try:
                append_rows_to_csv(csv_path, rows, CSV_FIELDNAMES)
                if not error:
                    append_audit_row(audit_path, category, str(row["video_id"]), str(row.get("split", "")))
                    self.completed_video_keys.add(video_key)
            except Exception as exc:
                errors += 1
                logger.error(
                    f"Failed to commit results for {row['absolute_video_path']}: {type(exc).__name__}: {exc}"
                )
                return

            total_new_frames += int(stats.get("new_frames", 0))
            if error:
                errors += 1
                logger.warning(f"[{category}] {Path(row['absolute_video_path']).name}: {error}")
                return

            logger.info(
                f"[{category}] {stats['video_name']} | "
                f"frames={stats['saved_frames']} | "
                f"new={stats['new_frames']} | "
                f"resumed={stats['resumed_frames']} | "
                f"time={stats['elapsed_sec']}s"
            )

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
                        int(row["resume_existing_count"]),
                    ): row
                    for row in pending_rows
                }

                with tqdm(total=len(pending_rows), desc=f"Processing {category}") as pbar:
                    for future in as_completed(futures):
                        row = futures[future]
                        try:
                            result = future.result()
                            commit_video_result(row, result)
                        except Exception as e:
                            errors += 1
                            logger.error(
                                f"Worker failed for {row['absolute_video_path']}: {type(e).__name__}: {e}"
                            )
                        pbar.update(1)
        else:
            for row in tqdm(pending_rows, desc=f"Processing {category}"):
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
                    int(row["resume_existing_count"]),
                )
                commit_video_result(row, result)

        logger.info(f"Completed {category}: new_frames={total_new_frames}, errors={errors}")
        return total_new_frames

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
        logger.info(f"Resume enabled: {self.resume}")
        logger.info("=" * 60)

        ensure_dir(self.output_path)

        csv_path = self.output_path / self.metadata_csv
        audit_path = self.output_path / self.audit_csv

        if self.resume:
            self.completed_video_keys = self._prepare_resume_state(csv_path, audit_path)
            ensure_csv_with_header(audit_path, AUDIT_FIELDNAMES)
        else:
            init_csv(csv_path, CSV_FIELDNAMES)
            init_csv(audit_path, AUDIT_FIELDNAMES)
            self.completed_video_keys = set()

        categories = [only_category] if only_category else list(self.categories.keys())
        total_new_frames = 0

        for category in categories:
            if category not in self.categories:
                logger.warning(f"Unknown category skipped: {category}")
                continue
            total_new_frames += self.process_category(category, csv_path, audit_path, split=only_split)

        elapsed = time.perf_counter() - t0
        logger.info("=" * 60)
        logger.info(f"Frame extraction completed | total_new_frames={total_new_frames} | elapsed={elapsed:.2f}s")
        logger.info(f"Metadata CSV: {csv_path}")
        logger.info(f"Audit CSV: {audit_path}")
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
    parser.add_argument("--no-resume", action="store_true", help="Disable resume and start with fresh metadata/audit")
    args = parser.parse_args()

    setup_logging()

    extractor = FrameExtractor(
        manifest_path=args.manifest,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        num_workers=args.workers,
        resume=not args.no_resume,
    )

    if args.rebuild_csv_only:
        extractor.rebuild_metadata_from_frames()
    else:
        extractor.extract_all(only_category=args.category, only_split=args.split)


if __name__ == "__main__":
    main()
