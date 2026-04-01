from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm
from webdataset import ShardWriter

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


ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left_eye
        [73.5318, 51.5014],  # right_eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # mouth_left
        [70.7299, 92.2041],  # mouth_right
    ],
    dtype=np.float32,
)


def _ensure_retinaface_available() -> None:
    if RetinaFace is None:
        if RETINAFACE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "RetinaFace import failed: "
                f"{type(RETINAFACE_IMPORT_ERROR).__name__}: {RETINAFACE_IMPORT_ERROR}. "
                "If message mentions tf-keras, run: uv add tf-keras"
            )
        raise RuntimeError("RetinaFace is not installed. Install it with: uv add retina-face")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_video_group_key(row: Dict[str, str]) -> str:
    category = _normalize_text(row.get("category"))
    video_name = _normalize_text(row.get("video_name"))

    if video_name:
        return f"{category}/{video_name}" if category else video_name

    video_id = _normalize_text(row.get("video_id"))
    if video_id:
        return f"{category}/{video_id}" if category else video_id

    source_video_id = _normalize_text(row.get("source_video_id"))
    if source_video_id:
        return f"{category}/{source_video_id}" if category else source_video_id

    video_rel_path = _normalize_text(row.get("video_rel_path"))
    if video_rel_path:
        return f"{category}/{video_rel_path}" if category else video_rel_path

    video_abs_path = _normalize_text(row.get("video_abs_path"))
    if video_abs_path:
        return video_abs_path

    frame_path = _normalize_text(row.get("frame_path"))
    if frame_path:
        stem = Path(frame_path).stem
        return f"{category}/{stem}" if category else stem

    return f"{category}/unknown_video" if category else "unknown_video"


def _group_rows_by_video(
    rows: List[Dict[str, str]],
    category_filter: Optional[str],
) -> Dict[str, List[tuple[int, Dict[str, str]]]]:
    grouped: Dict[str, List[tuple[int, Dict[str, str]]]] = {}

    for idx, row in enumerate(rows):
        if category_filter and row.get("category", "") != category_filter:
            continue
        video_key = _extract_video_group_key(row)
        grouped.setdefault(video_key, []).append((idx, row))

    return grouped


def _sort_video_rows(video_rows: List[tuple[int, Dict[str, str]]]) -> List[tuple[int, Dict[str, str]]]:
    def _frame_sort_key(item: tuple[int, Dict[str, str]]) -> tuple[int, str]:
        _, row = item

        frame_idx_str = (
            row.get("frame_number")
            or row.get("frame_index")
            or row.get("frame_idx")
            or row.get("original_frame_index")
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


def load_image(image_path: Path) -> tuple[Optional[Any], str]:
    if not image_path.exists():
        return None, "missing_file"

    image = cv2.imread(str(image_path))
    if image is None:
        return None, "read_fail"

    return image, "ok"


def _parse_landmarks(landmarks: Any) -> Optional[Dict[str, List[float]]]:
    if not isinstance(landmarks, dict):
        return None

    parsed: Dict[str, List[float]] = {}
    for k, v in landmarks.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            parsed[k] = [float(v[0]), float(v[1])]

    required = {"left_eye", "right_eye", "nose", "mouth_left", "mouth_right"}
    if not required.issubset(set(parsed.keys())):
        return None

    return parsed


def _best_face_from_result(
    result: Any,
) -> tuple[int, float, Optional[list[int]], Optional[Dict[str, List[float]]]]:
    if not result or not isinstance(result, dict):
        return 0, 0.0, None, None

    best_score = -1.0
    best_bbox: Optional[list[int]] = None
    best_landmarks: Optional[Dict[str, List[float]]] = None
    face_count = 0

    for _, face_data in result.items():
        if not isinstance(face_data, dict):
            continue

        bbox = face_data.get("facial_area")
        score = float(face_data.get("score", 0.0))
        landmarks = face_data.get("landmarks")

        if not bbox or len(bbox) != 4:
            continue

        face_count += 1
        if score > best_score:
            best_score = score
            best_bbox = [int(v) for v in bbox]
            best_landmarks = _parse_landmarks(landmarks)

    if face_count == 0 or best_bbox is None:
        return 0, 0.0, None, None

    return face_count, round(best_score, 6), best_bbox, best_landmarks


def detect_face_on_array(
    image: Any,
    threshold: float,
    max_side: int,
) -> tuple[int, float, Optional[list[int]], Optional[Dict[str, List[float]]], str]:
    if RetinaFace is None:
        return 0, 0.0, None, None, "detect_error"

    if image is None:
        return 0, 0.0, None, None, "invalid_image"

    resized, scale = _resize_for_detection(image, max_side=max_side)

    try:
        result = RetinaFace.detect_faces(resized, threshold=threshold)
    except Exception as e:
        logger.debug("Detection failed on in-memory image: {}", e)
        return 0, 0.0, None, None, "detect_error"

    num_faces, confidence, bbox, landmarks = _best_face_from_result(result)
    if bbox is None:
        return 0, 0.0, None, None, "no_face"

    if scale != 1.0:
        bbox = [int(round(v / scale)) for v in bbox]
        if landmarks is not None:
            scaled_landmarks: Dict[str, List[float]] = {}
            for k, pt in landmarks.items():
                scaled_landmarks[k] = [float(pt[0] / scale), float(pt[1] / scale)]
            landmarks = scaled_landmarks

    h, w = image.shape[:2]
    bbox[0] = max(0, min(bbox[0], w - 1))
    bbox[1] = max(0, min(bbox[1], h - 1))
    bbox[2] = max(0, min(bbox[2], w - 1))
    bbox[3] = max(0, min(bbox[3], h - 1))

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return 0, 0.0, None, None, "no_face"

    return num_faces, confidence, bbox, landmarks, "ok"


def _retinaface_landmarks_to_template_order(
    landmarks: Dict[str, List[float]],
) -> Optional[np.ndarray]:
    if landmarks is None:
        return None

    required = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    pts = []
    for key in required:
        if key not in landmarks:
            return None
        v = landmarks[key]
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return None
        pts.append([float(v[0]), float(v[1])])

    return np.array(pts, dtype=np.float32)


def align_face_5pts(
    image: Any,
    landmarks: Optional[Dict[str, List[float]]],
    output_size: tuple[int, int],
) -> tuple[Optional[Any], Optional[np.ndarray], str]:
    if image is None:
        return None, None, "read_fail"

    src = _retinaface_landmarks_to_template_order(landmarks)
    if src is None:
        return None, None, "missing_landmarks"

    dst = ARCFACE_112_TEMPLATE.copy()
    out_w, out_h = output_size
    if (out_w, out_h) != (112, 112):
        dst[:, 0] *= out_w / 112.0
        dst[:, 1] *= out_h / 112.0

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None, None, "align_fail"

    aligned = cv2.warpAffine(
        image,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return aligned, M, "ok"


def encode_image_to_bytes(image: Any, image_format: str, jpeg_quality: int) -> bytes:
    ext = image_format.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError("Failed to encode aligned image")
    return buffer.tobytes()


def build_sample_key(row: Dict[str, str]) -> str:
    category = _normalize_text(row.get("category")) or "unknown"
    video_name = _normalize_text(row.get("video_name")) or "unknownvideo"
    frame_number = (
        _normalize_text(row.get("frame_number"))
        or _normalize_text(row.get("frame_index"))
        or _normalize_text(row.get("frame_idx"))
        or _normalize_text(row.get("original_frame_index"))
        or "0"
    )
    frame_path = _normalize_text(row.get("frame_path"))
    stem = Path(frame_path).stem if frame_path else f"{video_name}_{frame_number}"

    safe_key = f"{category}/{video_name}/{stem}"
    return safe_key.replace("\\", "/")


def _safe_label_id(row: Dict[str, str]) -> int:
    label_text = row.get("label") or row.get("binary_label") or ""
    label_text = str(label_text).strip().lower()

    if label_text in {"real", "original", "0"}:
        return 0
    if label_text in {"fake", "1"}:
        return 1
    return -1


def _extract_shard_index(shard_path: Path) -> Optional[int]:
    match = re.fullmatch(r"shard-(\d{6})\.tar", shard_path.name)
    if not match:
        return None
    return int(match.group(1))


def infer_start_shard(output_dir: Path) -> int:
    shard_indices: List[int] = []

    for shard_path in output_dir.glob("shard-*.tar"):
        idx = _extract_shard_index(shard_path)
        if idx is not None:
            shard_indices.append(idx)

    if not shard_indices:
        return 0

    return max(shard_indices) + 1


def load_processed_keys_from_audit(audit_csv: Optional[Path]) -> Set[str]:
    keys: Set[str] = set()

    if audit_csv is None or not audit_csv.exists():
        return keys

    with audit_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _normalize_text(row.get("key"))
            if key:
                keys.add(key)

    return keys


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

    if "/" in path:
        parent = path.rsplit("/", 1)[0]
        return f"{parent}/{base}"
    return base


def load_processed_keys_from_existing_shards(output_dir: Path) -> Set[str]:
    keys: Set[str] = set()
    shard_files = sorted(output_dir.glob("shard-*.tar"))

    for shard_path in tqdm(shard_files, desc="Scan existing shards for resume", leave=False):
        try:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    key = _base_key_from_tar_member(member.name)
                    if key:
                        keys.add(key)
        except Exception as e:
            logger.warning("Failed to scan shard {}: {}", shard_path, e)

    return keys


def append_audit_row(audit_csv: Path, row: Dict[str, Any]) -> None:
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = audit_csv.exists()
    fieldnames = list(row.keys())

    with audit_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or audit_csv.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def build_audit_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(sample["json"].decode("utf-8"))


def process_single_video_to_samples(
    video_id: str,
    indexed_rows: List[tuple[int, Dict[str, str]]],
    frame_root: Path,
    threshold: float,
    max_side: int,
    aligned_size: tuple[int, int],
    image_format: str,
    jpeg_quality: int,
    skip_no_face: bool,
    processed_keys: Set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    local_rows = _sort_video_rows(indexed_rows)
    stats = {
        "total_rows": len(local_rows),
        "processed_frames": 0,
        "detected_frames": 0,
        "missing_file": 0,
        "read_fail": 0,
        "detect_error": 0,
        "no_face": 0,
        "missing_landmarks": 0,
        "align_fail": 0,
        "encode_fail": 0,
        "written_samples": 0,
        "skipped_no_face": 0,
        "skipped_existing": 0,
    }

    samples: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(local_rows), desc=f"Frames {str(video_id)[:40]}", leave=False)

    for _, row in local_rows:
        sample_key = build_sample_key(row)
        if sample_key in processed_keys:
            stats["skipped_existing"] += 1
            pbar.update(1)
            continue

        rel_path = row.get("frame_path", "")
        image_path = frame_root / rel_path

        image, load_status = load_image(image_path)
        if load_status == "missing_file":
            stats["processed_frames"] += 1
            stats["missing_file"] += 1
            pbar.update(1)
            continue

        if load_status == "read_fail":
            stats["processed_frames"] += 1
            stats["read_fail"] += 1
            pbar.update(1)
            continue

        num_faces, confidence, bbox, landmarks, detect_status = detect_face_on_array(
            image=image,
            threshold=threshold,
            max_side=max_side,
        )
        stats["processed_frames"] += 1

        if detect_status == "detect_error":
            stats["detect_error"] += 1
            pbar.update(1)
            continue

        if detect_status == "no_face" or bbox is None:
            stats["no_face"] += 1
            if skip_no_face:
                stats["skipped_no_face"] += 1
                pbar.update(1)
                continue
            pbar.update(1)
            continue

        stats["detected_frames"] += 1

        aligned_img, affine_matrix, align_status = align_face_5pts(
            image=image,
            landmarks=landmarks,
            output_size=aligned_size,
        )

        if align_status == "missing_landmarks":
            stats["missing_landmarks"] += 1
            pbar.update(1)
            continue

        if align_status != "ok" or aligned_img is None or affine_matrix is None:
            stats["align_fail"] += 1
            pbar.update(1)
            continue

        try:
            image_bytes = encode_image_to_bytes(
                image=aligned_img,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
        except Exception as e:
            logger.warning("Encode failed on {}: {}", image_path, e)
            stats["encode_fail"] += 1
            pbar.update(1)
            continue

        out_h, out_w = aligned_img.shape[:2]
        label_id = _safe_label_id(row)

        metadata = {
            "key": sample_key,
            "video_id": video_id,
            "frame_path": rel_path,
            "category": row.get("category", ""),
            "video_name": row.get("video_name", ""),
            "label": row.get("label", ""),
            "binary_label": row.get("binary_label", ""),
            "frame_number": row.get("frame_number", ""),
            "original_frame_index": row.get("original_frame_index", ""),
            "timestamp": row.get("timestamp", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
            "face_detected": 1,
            "num_faces": num_faces,
            "face_confidence": confidence,
            "bbox_x1": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x2": int(bbox[2]),
            "bbox_y2": int(bbox[3]),
            "landmarks": landmarks,
            "alignment_mode": "similarity_5pts",
            "align_status": align_status,
            "aligned_width": int(out_w),
            "aligned_height": int(out_h),
            "aligned_size": [int(aligned_size[0]), int(aligned_size[1])],
            "affine_matrix": affine_matrix.tolist(),
            "image_format": image_format,
        }

        sample: Dict[str, Any] = {
            "__key__": sample_key,
            "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        }

        if image_format.lower() in {".jpg", ".jpeg"}:
            sample["jpg"] = image_bytes
        elif image_format.lower() == ".png":
            sample["png"] = image_bytes

        if label_id in {0, 1}:
            sample["cls"] = str(label_id).encode("utf-8")

        samples.append(sample)
        stats["written_samples"] += 1
        pbar.update(1)

    pbar.close()
    return samples, stats


def run_pipeline(
    metadata_csv: Path,
    frame_root: Path,
    output_dir: Path,
    category: Optional[str],
    threshold: float,
    max_side: int,
    aligned_size: tuple[int, int],
    image_format: str,
    jpeg_quality: int,
    shard_maxcount: int,
    shard_maxsize: int,
    limit: Optional[int],
    skip_no_face: bool,
    audit_csv: Optional[Path],
) -> None:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError("Metadata CSV has no header")
        all_rows = list(reader)

    rows = all_rows[:limit] if limit is not None else all_rows
    grouped = _group_rows_by_video(rows=rows, category_filter=category)
    video_items = list(grouped.items())

    if not video_items:
        raise RuntimeError("No matching rows/videos found to process.")

    start_shard = infer_start_shard(output_dir)

    processed_keys = load_processed_keys_from_audit(audit_csv)
    if processed_keys:
        logger.info("Loaded {} processed keys from audit CSV", len(processed_keys))
    else:
        logger.info("No usable audit CSV found for resume")

    shard_processed_keys = load_processed_keys_from_existing_shards(output_dir)
    if shard_processed_keys:
        logger.info("Loaded {} processed keys from existing shards", len(shard_processed_keys))

    processed_keys |= shard_processed_keys

    logger.info("Total rows loaded: {}", len(rows))
    logger.info("Total videos: {}", len(video_items))
    logger.info("Output dir: {}", output_dir)
    logger.info("max_side: {}", max_side)
    logger.info("aligned_size: {}", aligned_size)
    logger.info("image_format: {}", image_format)
    logger.info("skip_no_face: {}", skip_no_face)
    logger.info("Resume start_shard: {}", start_shard)
    logger.info("Total processed keys for resume: {}", len(processed_keys))

    shard_pattern = str(output_dir / "shard-%06d.tar")

    total_stats = {
        "videos": 0,
        "total_rows": 0,
        "processed_frames": 0,
        "detected_frames": 0,
        "missing_file": 0,
        "read_fail": 0,
        "detect_error": 0,
        "no_face": 0,
        "missing_landmarks": 0,
        "align_fail": 0,
        "encode_fail": 0,
        "written_samples": 0,
        "skipped_no_face": 0,
        "skipped_existing": 0,
    }

    with ShardWriter(
        shard_pattern,
        maxcount=shard_maxcount,
        maxsize=shard_maxsize,
        start_shard=start_shard,
    ) as sink:
        video_pbar = tqdm(total=len(video_items), desc="Videos")

        for video_id, indexed_rows in video_items:
            samples, stats = process_single_video_to_samples(
                video_id=video_id,
                indexed_rows=indexed_rows,
                frame_root=frame_root,
                threshold=threshold,
                max_side=max_side,
                aligned_size=aligned_size,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                skip_no_face=skip_no_face,
                processed_keys=processed_keys,
            )

            for sample in samples:
                sink.write(sample)
                processed_keys.add(sample["__key__"])

                if audit_csv is not None:
                    append_audit_row(audit_csv, build_audit_row(sample))

            total_stats["videos"] += 1
            total_stats["total_rows"] += stats["total_rows"]
            total_stats["processed_frames"] += stats["processed_frames"]
            total_stats["detected_frames"] += stats["detected_frames"]
            total_stats["missing_file"] += stats["missing_file"]
            total_stats["read_fail"] += stats["read_fail"]
            total_stats["detect_error"] += stats["detect_error"]
            total_stats["no_face"] += stats["no_face"]
            total_stats["missing_landmarks"] += stats["missing_landmarks"]
            total_stats["align_fail"] += stats["align_fail"]
            total_stats["encode_fail"] += stats["encode_fail"]
            total_stats["written_samples"] += stats["written_samples"]
            total_stats["skipped_no_face"] += stats["skipped_no_face"]
            total_stats["skipped_existing"] += stats["skipped_existing"]

            logger.info(
                "Done video={} | rows={} | written={} | detected={} | no_face={} | skipped_existing={} | missing_landmarks={} | align_fail={}",
                video_id,
                stats["total_rows"],
                stats["written_samples"],
                stats["detected_frames"],
                stats["no_face"],
                stats["skipped_existing"],
                stats["missing_landmarks"],
                stats["align_fail"],
            )

            video_pbar.update(1)

        video_pbar.close()

    logger.info("=== FINAL SUMMARY ===")
    for k, v in total_stats.items():
        logger.info("{}: {}", k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RetinaFace detect all frames + 5-point align + write WebDataset shards with resume support"
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
        "--output-dir",
        type=str,
        default=str(settings.CROP_DATA_DIR),
        help="Directory to store output .tar shards",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only process one category",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_DETECTION_THRESHOLD,
        help="RetinaFace confidence threshold",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows for testing",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize image before detection so longest side <= max-side",
    )
    parser.add_argument(
        "--aligned-width",
        type=int,
        default=224,
        help="Output aligned face width",
    )
    parser.add_argument(
        "--aligned-height",
        type=int,
        default=224,
        help="Output aligned face height",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
        help="Encoded image format stored inside shard",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if using jpg/jpeg output",
    )
    parser.add_argument(
        "--shard-maxcount",
        type=int,
        default=10000,
        help="Maximum samples per shard",
    )
    parser.add_argument(
        "--shard-maxsize",
        type=int,
        default=2_000_000_000,
        help="Maximum bytes per shard",
    )
    parser.add_argument(
        "--skip-no-face",
        action="store_true",
        help="Skip frames with no detected face",
    )
    parser.add_argument(
        "--audit-csv",
        type=str,
        default=str(settings.AUDIT_FILE),
        help="Optional audit CSV path",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger.info("RetinaFace cache dir: {}", RETINAFACE_CACHE_DIR)
    _ensure_retinaface_available()

    args = parse_args()

    run_pipeline(
        metadata_csv=Path(args.metadata_csv),
        frame_root=Path(args.frame_root),
        output_dir=Path(args.output_dir),
        category=args.category,
        threshold=args.threshold,
        max_side=args.max_side,
        aligned_size=(args.aligned_width, args.aligned_height),
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        shard_maxcount=args.shard_maxcount,
        shard_maxsize=args.shard_maxsize,
        limit=args.limit,
        skip_no_face=args.skip_no_face,
        audit_csv=Path(args.audit_csv) if args.audit_csv else None,
    )


if __name__ == "__main__":
    main()