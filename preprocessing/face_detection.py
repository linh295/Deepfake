from __future__ import annotations

import argparse
import csv
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm

from configs.loggings import logger, setup_logging
from configs.settings import settings


RETINAFACE_CACHE_DIR = Path(settings.RETINAFACE_WEIGHT_DIR)
RETINAFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DEEPFACE_HOME", str(RETINAFACE_CACHE_DIR))

try:
    RetinaFace = importlib.import_module("retinaface.RetinaFace")
    RETINAFACE_IMPORT_ERROR = None
except Exception as import_error:  # pragma: no cover
    RetinaFace = None
    RETINAFACE_IMPORT_ERROR = import_error


NEW_COLUMNS = [
    "face_detected",
    "num_faces",
    "face_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_source",
]


def _ensure_retinaface_available() -> None:
    if RetinaFace is None:
        if RETINAFACE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "RetinaFace import failed: "
                f"{type(RETINAFACE_IMPORT_ERROR).__name__}: {RETINAFACE_IMPORT_ERROR}. "
                "If message mentions tf-keras, run: uv add tf-keras"
            )
        raise RuntimeError("RetinaFace is not installed. Install it with: uv add retina-face")


def _best_face_from_result(result: Any) -> tuple[int, float, Optional[list[int]]]:
    if not result or not isinstance(result, dict):
        return 0, 0.0, None

    best_score = -1.0
    best_bbox: Optional[list[int]] = None
    face_count = 0

    for _, face_data in result.items():
        if not isinstance(face_data, dict):
            continue

        bbox = face_data.get("facial_area")
        score = float(face_data.get("score", 0.0))
        if not bbox or len(bbox) != 4:
            continue

        face_count += 1
        if score > best_score:
            best_score = score
            best_bbox = [int(v) for v in bbox]

    if face_count == 0 or best_bbox is None:
        return 0, 0.0, None

    return face_count, round(best_score, 6), best_bbox


def _resize_for_detection(image: Any, max_side: int) -> tuple[Any, float]:
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return image, 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_face_on_image(
    image_path: Path,
    threshold: float,
    max_side: int,
) -> tuple[int, float, Optional[list[int]], str]:
    """
    Returns:
        num_faces, confidence, bbox, status
    status in {"ok", "missing_file", "read_fail", "detect_error", "no_face"}
    """
    if RetinaFace is None:
        return 0, 0.0, None, "detect_error"

    if not image_path.exists():
        return 0, 0.0, None, "missing_file"

    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0.0, None, "read_fail"

    resized, scale = _resize_for_detection(image, max_side=max_side)

    try:
        result = RetinaFace.detect_faces(resized, threshold=threshold)
    except Exception as e:
        logger.debug("Detection failed on {}: {}", image_path, e)
        return 0, 0.0, None, "detect_error"

    num_faces, confidence, bbox = _best_face_from_result(result)
    if bbox is None:
        return 0, 0.0, None, "no_face"

    if scale != 1.0:
        bbox = [int(round(v / scale)) for v in bbox]

    h, w = image.shape[:2]
    bbox[0] = max(0, min(bbox[0], w - 1))
    bbox[1] = max(0, min(bbox[1], h - 1))
    bbox[2] = max(0, min(bbox[2], w - 1))
    bbox[3] = max(0, min(bbox[3], h - 1))

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return 0, 0.0, None, "no_face"

    return num_faces, confidence, bbox, "ok"


def _parse_existing_bbox(row: Dict[str, str]) -> Optional[list[int]]:
    try:
        x1 = row.get("bbox_x1", "").strip()
        y1 = row.get("bbox_y1", "").strip()
        x2 = row.get("bbox_x2", "").strip()
        y2 = row.get("bbox_y2", "").strip()
        if not (x1 and y1 and x2 and y2):
            return None
        return [int(x1), int(y1), int(x2), int(y2)]
    except Exception:
        return None


def _interpolate_bbox(
    left_idx: int,
    left_bbox: list[int],
    right_idx: int,
    right_bbox: list[int],
    target_idx: int,
) -> list[int]:
    if right_idx == left_idx:
        return left_bbox[:]

    ratio = (target_idx - left_idx) / float(right_idx - left_idx)
    out = []
    for lv, rv in zip(left_bbox, right_bbox):
        value = lv + ratio * (rv - lv)
        out.append(int(round(value)))
    return out


def _group_rows_by_video(
    rows: List[Dict[str, str]],
    category_filter: Optional[str],
) -> Dict[str, List[tuple[int, Dict[str, str]]]]:
    grouped: Dict[str, List[tuple[int, Dict[str, str]]]] = {}

    for idx, row in enumerate(rows):
        if category_filter and row.get("category", "") != category_filter:
            continue

        video_id = (
            row.get("video_id")
            or row.get("source_video_id")
            or row.get("video_rel_path")
            or row.get("video_abs_path")
        )
        if not video_id:
            frame_path = row.get("frame_path", "")
            video_id = str(Path(frame_path).parent)

        grouped.setdefault(video_id, []).append((idx, row))

    return grouped


def _sort_video_rows(video_rows: List[tuple[int, Dict[str, str]]]) -> List[tuple[int, Dict[str, str]]]:
    def _frame_sort_key(item: tuple[int, Dict[str, str]]) -> tuple[int, str]:
        _, row = item

        frame_idx_str = (
            row.get("frame_index")
            or row.get("frame_idx")
            or row.get("frame_number")
            or ""
        ).strip()
        if frame_idx_str.isdigit():
            return int(frame_idx_str), row.get("frame_path", "")

        frame_path = row.get("frame_path", "")
        stem = Path(frame_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits.isdigit():
            return int(digits), frame_path

        return 10**12, frame_path

    return sorted(video_rows, key=_frame_sort_key)


def _build_detect_indices(num_rows: int, detect_every_k: int) -> List[int]:
    if num_rows <= 0:
        return []

    if detect_every_k <= 1:
        return list(range(num_rows))

    indices = list(range(0, num_rows, detect_every_k))
    if indices[-1] != num_rows - 1:
        indices.append(num_rows - 1)
    return indices


def _apply_row_result(
    row: Dict[str, str],
    bbox: Optional[list[int]],
    num_faces: int,
    confidence: float,
    bbox_source: str,
) -> None:
    if bbox is None:
        row["face_detected"] = "0"
        row["num_faces"] = str(num_faces)
        row["face_confidence"] = f"{confidence:.6f}"
        row["bbox_x1"] = ""
        row["bbox_y1"] = ""
        row["bbox_x2"] = ""
        row["bbox_y2"] = ""
        row["bbox_source"] = bbox_source
        return

    row["face_detected"] = "1"
    row["num_faces"] = str(num_faces)
    row["face_confidence"] = f"{confidence:.6f}"
    row["bbox_x1"] = str(bbox[0])
    row["bbox_y1"] = str(bbox[1])
    row["bbox_x2"] = str(bbox[2])
    row["bbox_y2"] = str(bbox[3])
    row["bbox_source"] = bbox_source


def _process_single_video(
    video_id: str,
    indexed_rows: List[tuple[int, Dict[str, str]]],
    frame_root: Path,
    threshold: float,
    max_side: int,
    detect_every_k: int,
    overwrite_existing: bool,
) -> dict:
    local_rows = _sort_video_rows(indexed_rows)
    num_rows = len(local_rows)

    logger.info("Processing video: {} | total_frames={}", video_id, num_rows)

    stats = {
        "video_id": video_id,
        "total_rows": num_rows,
        "processed_detect_frames": 0,
        "detected_frames": 0,
        "interpolated_frames": 0,
        "kept_existing_frames": 0,
        "missing_file": 0,
        "read_fail": 0,
        "detect_error": 0,
        "no_face": 0,
    }

    detect_indices = _build_detect_indices(num_rows=num_rows, detect_every_k=detect_every_k)
    explicit_results: Dict[int, dict] = {}

    detect_pbar = tqdm(
        total=len(detect_indices),
        desc=f"Detect {str(video_id)[:40]}",
        leave=False,
    )

    for local_idx in detect_indices:
        _, row = local_rows[local_idx]

        existing_bbox = _parse_existing_bbox(row)
        if existing_bbox is not None and not overwrite_existing:
            explicit_results[local_idx] = {
                "bbox": existing_bbox,
                "num_faces": int(row.get("num_faces", "1") or 1),
                "confidence": float(row.get("face_confidence", "1.0") or 1.0),
                "source": "existing",
            }
            stats["kept_existing_frames"] += 1
            detect_pbar.update(1)
            continue

        rel_path = row.get("frame_path", "")
        image_path = frame_root / rel_path

        num_faces, confidence, bbox, status = detect_face_on_image(
            image_path=image_path,
            threshold=threshold,
            max_side=max_side,
        )

        stats["processed_detect_frames"] += 1

        if status == "ok":
            stats["detected_frames"] += 1
        elif status == "missing_file":
            stats["missing_file"] += 1
        elif status == "read_fail":
            stats["read_fail"] += 1
        elif status == "detect_error":
            stats["detect_error"] += 1
        elif status == "no_face":
            stats["no_face"] += 1

        explicit_results[local_idx] = {
            "bbox": bbox,
            "num_faces": num_faces,
            "confidence": confidence,
            "source": "detected",
        }
        detect_pbar.update(1)

    detect_pbar.close()

    output_rows: List[tuple[int, Dict[str, str]]] = []

    valid_detect_points = [
        idx for idx, result in explicit_results.items()
        if result["bbox"] is not None
    ]
    valid_detect_points.sort()

    for local_idx, (global_idx, row) in enumerate(local_rows):
        row_copy = dict(row)

        if local_idx in explicit_results:
            result = explicit_results[local_idx]
            _apply_row_result(
                row=row_copy,
                bbox=result["bbox"],
                num_faces=result["num_faces"],
                confidence=result["confidence"],
                bbox_source=result["source"],
            )
            output_rows.append((global_idx, row_copy))
            continue

        left_idx = None
        right_idx = None

        for idx in reversed(valid_detect_points):
            if idx < local_idx:
                left_idx = idx
                break

        for idx in valid_detect_points:
            if idx > local_idx:
                right_idx = idx
                break

        if left_idx is not None and right_idx is not None:
            left = explicit_results[left_idx]
            right = explicit_results[right_idx]
            interp_bbox = _interpolate_bbox(
                left_idx=left_idx,
                left_bbox=left["bbox"],
                right_idx=right_idx,
                right_bbox=right["bbox"],
                target_idx=local_idx,
            )
            interp_conf = min(left["confidence"], right["confidence"])
            _apply_row_result(
                row=row_copy,
                bbox=interp_bbox,
                num_faces=1,
                confidence=interp_conf,
                bbox_source="interpolated",
            )
            stats["interpolated_frames"] += 1
        elif left_idx is not None:
            left = explicit_results[left_idx]
            _apply_row_result(
                row=row_copy,
                bbox=left["bbox"],
                num_faces=1 if left["bbox"] else 0,
                confidence=left["confidence"],
                bbox_source="propagated_left",
            )
            if left["bbox"] is not None:
                stats["interpolated_frames"] += 1
        elif right_idx is not None:
            right = explicit_results[right_idx]
            _apply_row_result(
                row=row_copy,
                bbox=right["bbox"],
                num_faces=1 if right["bbox"] else 0,
                confidence=right["confidence"],
                bbox_source="propagated_right",
            )
            if right["bbox"] is not None:
                stats["interpolated_frames"] += 1
        else:
            _apply_row_result(
                row=row_copy,
                bbox=None,
                num_faces=0,
                confidence=0.0,
                bbox_source="no_valid_detection",
            )

        output_rows.append((global_idx, row_copy))

    logger.info(
        "Done video: {} | total={} | detect_processed={} | detected={} | interpolated={} | no_face={} | detect_error={}",
        video_id,
        stats["total_rows"],
        stats["processed_detect_frames"],
        stats["detected_frames"],
        stats["interpolated_frames"],
        stats["no_face"],
        stats["detect_error"],
    )

    return {
        "video_id": video_id,
        "rows": output_rows,
        "stats": stats,
    }


def update_metadata_with_bboxes_sequential(
    metadata_csv: Path,
    frame_root: Path,
    threshold: float,
    category: Optional[str],
    overwrite_existing: bool,
    limit: Optional[int],
    max_side: int,
    detect_every_k: int,
) -> Path:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    with metadata_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError("Metadata CSV has no header")
        fieldnames = list(reader.fieldnames)

        for col in NEW_COLUMNS:
            if col not in fieldnames:
                fieldnames.append(col)

        all_rows = list(reader)

    # 'rows' is the subset to process; global_idx still maps into all_rows
    rows = all_rows[:limit] if limit is not None else all_rows

    grouped = _group_rows_by_video(rows=rows, category_filter=category)
    video_items = list(grouped.items())

    if not video_items:
        raise RuntimeError("No matching rows/videos found to process.")

    logger.info("Total rows in CSV: {}", len(all_rows))
    if limit is not None:
        logger.info("Processing limited to first {} rows", limit)
    logger.info("Total videos to process: {}", len(video_items))
    logger.info("detect_every_k: {}", detect_every_k)
    logger.info("max_side: {}", max_side)

    # Size by all_rows so unprocessed rows are preserved when writing back
    updated_rows: List[Optional[Dict[str, str]]] = [None] * len(all_rows)

    total_stats = {
        "videos": 0,
        "rows": 0,
        "processed_detect_frames": 0,
        "detected_frames": 0,
        "interpolated_frames": 0,
        "kept_existing_frames": 0,
        "missing_file": 0,
        "read_fail": 0,
        "detect_error": 0,
        "no_face": 0,
    }

    video_pbar = tqdm(total=len(video_items), desc="Processing videos")

    for video_id, indexed_rows in video_items:
        result = _process_single_video(
            video_id=video_id,
            indexed_rows=indexed_rows,
            frame_root=frame_root,
            threshold=threshold,
            max_side=max_side,
            detect_every_k=detect_every_k,
            overwrite_existing=overwrite_existing,
        )

        video_pbar.update(1)

        stats = result["stats"]
        total_stats["videos"] += 1
        total_stats["rows"] += stats["total_rows"]
        total_stats["processed_detect_frames"] += stats["processed_detect_frames"]
        total_stats["detected_frames"] += stats["detected_frames"]
        total_stats["interpolated_frames"] += stats["interpolated_frames"]
        total_stats["kept_existing_frames"] += stats["kept_existing_frames"]
        total_stats["missing_file"] += stats["missing_file"]
        total_stats["read_fail"] += stats["read_fail"]
        total_stats["detect_error"] += stats["detect_error"]
        total_stats["no_face"] += stats["no_face"]

        for global_idx, row in result["rows"]:
            updated_rows[global_idx] = row

    video_pbar.close()

    # Fill every slot that was not touched (other categories or beyond --limit)
    for idx in range(len(all_rows)):
        if updated_rows[idx] is None:
            updated_rows[idx] = dict(all_rows[idx])

    final_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(updated_rows):
        if row is None:
            raise RuntimeError(f"Missing processed row at index {idx}")
        final_rows.append(row)

    temp_csv = metadata_csv.with_suffix(".tmp.csv")
    with temp_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    backup_csv = metadata_csv.with_suffix(settings.FACE_DETECTION_BACKUP_SUFFIX)
    if backup_csv.exists():
        backup_csv.unlink()
    metadata_csv.replace(backup_csv)
    temp_csv.replace(metadata_csv)

    logger.info("Metadata updated: {}", metadata_csv)
    logger.info("Backup created: {}", backup_csv)
    logger.info("Videos processed: {}", total_stats["videos"])
    logger.info("Rows written: {}", total_stats["rows"])
    logger.info("Detect frames actually processed: {}", total_stats["processed_detect_frames"])
    logger.info("Frames with explicit detection: {}", total_stats["detected_frames"])
    logger.info("Frames filled by interpolation/propagation: {}", total_stats["interpolated_frames"])
    logger.info("Frames kept from existing bbox: {}", total_stats["kept_existing_frames"])
    logger.info("Missing file: {}", total_stats["missing_file"])
    logger.info("Read fail: {}", total_stats["read_fail"])
    logger.info("Detect error: {}", total_stats["detect_error"])
    logger.info("No face: {}", total_stats["no_face"])

    return metadata_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RetinaFace detection with detect-every-K and resize-before-detect (sequential version)"
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=str(settings.FRAME_DATA_DIR / settings.FRAME_EXTRACTION_METADATA_CSV),
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--frame-root",
        type=str,
        default=str(settings.FRAME_DATA_DIR),
        help="Root directory containing extracted frame folders",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only process one category (e.g. NeuralTextures)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_DETECTION_THRESHOLD,
        help="RetinaFace confidence threshold",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-run detection even when bbox columns already have data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load/process first N rows from CSV for quick testing",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize image before detection so longest side <= max-side",
    )
    parser.add_argument(
        "--detect-every-k",
        type=int,
        default=5,
        help="Run face detection every K frames within each video, then interpolate bbox for intermediate frames",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger.info("RetinaFace cache dir: {}", RETINAFACE_CACHE_DIR)
    _ensure_retinaface_available()

    args = parse_args()

    update_metadata_with_bboxes_sequential(
        metadata_csv=Path(args.metadata_csv),
        frame_root=Path(args.frame_root),
        threshold=args.threshold,
        category=args.category,
        overwrite_existing=args.overwrite_existing,
        limit=args.limit,
        max_side=args.max_side,
        detect_every_k=args.detect_every_k,
    )


if __name__ == "__main__":
    main()