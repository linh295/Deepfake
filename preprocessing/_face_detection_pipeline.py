from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from webdataset import ShardWriter

from configs.loggings import logger, setup_logging
from configs.settings import settings


if os.environ.get("DEEPFAKE_TEST_WORKSPACE_TEMP"):
    class _WorkspaceTemporaryDirectory:
        def __init__(self) -> None:
            root = Path(os.environ["DEEPFAKE_TEST_WORKSPACE_TEMP"])
            root.mkdir(parents=True, exist_ok=True)
            self.name = str(root / uuid.uuid4().hex)
            Path(self.name).mkdir(parents=True, exist_ok=True)

        def cleanup(self) -> None:
            shutil.rmtree(self.name, ignore_errors=True)

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.cleanup()
            return False

    tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory  # type: ignore[assignment]


ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_REQUIRED_LANDMARK_KEYS = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
_REQUIRED_METADATA_FIELDS = {"frame_path"}
_VALID_SPLITS = {"train", "val", "test"}

_RETINAFACE_MODULE: Any = None
_RETINAFACE_IMPORT_ERROR: Optional[Exception] = None
_RETINAFACE_IMPORT_ATTEMPTED = False


@dataclass
class DetectionRecord:
    status: str
    num_faces: int = 0
    confidence: float = 0.0
    bbox: Optional[List[int]] = None
    landmarks: Optional[Dict[str, List[float]]] = None
    source: str = ""


@dataclass
class FaceDetectionConfig:
    metadata_csv: Path
    frame_root: Path
    output_dir: Path
    category: Optional[str]
    threshold: float
    max_side: int
    aligned_size: Tuple[int, int]
    crop_scale: float
    image_format: str
    jpeg_quality: int
    shard_maxcount: int
    shard_maxsize: int
    limit: Optional[int]
    skip_no_face: bool
    audit_csv: Optional[Path]
    detect_every_k: int
    retinaface_cache_dir: Path
    align_canvas_size: Tuple[int, int] = (0, 0)
    split: Optional[str] = None

    def __post_init__(self) -> None:
        requested_w, requested_h = self.align_canvas_size
        if requested_w <= 0 or requested_h <= 0:
            self.align_canvas_size = _infer_align_canvas_size(self.aligned_size)
            return

        side = int(max(requested_w, requested_h, self.aligned_size[0], self.aligned_size[1]))
        self.align_canvas_size = (side, side)


@dataclass
class VideoProcessingStats:
    total_rows: int = 0
    processed_frames: int = 0
    detected_frames: int = 0
    missing_file: int = 0
    read_fail: int = 0
    detect_error: int = 0
    no_face: int = 0
    missing_landmarks: int = 0
    align_fail: int = 0
    encode_fail: int = 0
    written_samples: int = 0
    skipped_no_face: int = 0
    skipped_existing: int = 0
    interpolated_frames: int = 0
    detected_frames_direct: int = 0

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class PipelineStats:
    videos: int = 0
    total_rows: int = 0
    processed_frames: int = 0
    detected_frames: int = 0
    missing_file: int = 0
    read_fail: int = 0
    detect_error: int = 0
    no_face: int = 0
    missing_landmarks: int = 0
    align_fail: int = 0
    encode_fail: int = 0
    written_samples: int = 0
    skipped_no_face: int = 0
    skipped_existing: int = 0
    interpolated_frames: int = 0
    detected_frames_direct: int = 0

    def merge_video(self, stats: VideoProcessingStats) -> None:
        self.videos += 1
        for key, value in stats.as_dict().items():
            setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _extract_video_group_key(row: Dict[str, str]) -> str:
    category = _normalize_text(row.get("category"))
    video_name = _normalize_text(row.get("video_name"))
    if video_name:
        return f"{category}/{video_name}" if category else video_name

    video_id = _normalize_text(row.get("video_id"))
    if video_id:
        return f"{category}/{video_id}" if category else video_id

    frame_path = _normalize_text(row.get("frame_path"))
    if frame_path:
        stem = Path(frame_path).stem
        return f"{category}/{stem}" if category else stem

    return f"{category}/unknown_video" if category else "unknown_video"


def _group_rows_by_video(
    rows: List[Dict[str, str]],
    category_filter: Optional[str],
) -> Dict[str, List[Tuple[int, Dict[str, str]]]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for idx, row in enumerate(rows):
        if category_filter and row.get("category", "") != category_filter:
            continue
        key = _extract_video_group_key(row)
        grouped.setdefault(key, []).append((idx, row))
    return grouped


def _sort_video_rows(video_rows: List[Tuple[int, Dict[str, str]]]) -> List[Tuple[int, Dict[str, str]]]:
    def _frame_sort_key(item: Tuple[int, Dict[str, str]]) -> Tuple[int, str]:
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


def _resize_for_detection(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def load_image(image_path: Path) -> Tuple[Optional[np.ndarray], str]:
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
    for key in _REQUIRED_LANDMARK_KEYS:
        value = landmarks.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        parsed[key] = [float(value[0]), float(value[1])]
    return parsed


def _face_area(bbox: List[int]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_iou(box1: Optional[List[int]], box2: Optional[List[int]]) -> float:
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = _face_area(box1) + _face_area(box2) - inter
    return inter / union if union > 0 else 0.0


def _clip_bbox(bbox: List[int], width: int, height: int) -> Optional[List[int]]:
    out = [
        int(max(0, min(bbox[0], width - 1))),
        int(max(0, min(bbox[1], height - 1))),
        int(max(0, min(bbox[2], width - 1))),
        int(max(0, min(bbox[3], height - 1))),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _extract_all_faces(result: Any) -> List[Dict[str, Any]]:
    faces: List[Dict[str, Any]] = []
    if not result or not isinstance(result, dict):
        return faces

    for face_data in result.values():
        if not isinstance(face_data, dict):
            continue
        bbox = face_data.get("facial_area")
        if not bbox or len(bbox) != 4:
            continue
        faces.append(
            {
                "bbox": [int(v) for v in bbox],
                "landmarks": _parse_landmarks(face_data.get("landmarks")),
                "score": float(face_data.get("score", 0.0)),
            }
        )
    return faces


def _choose_main_face(
    candidates: List[Dict[str, Any]],
    prev_bbox: Optional[List[int]],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    best_face: Optional[Dict[str, Any]] = None
    best_value = -1.0
    for face in candidates:
        bbox = face["bbox"]
        area = _face_area(bbox)
        confidence = float(face["score"])
        if prev_bbox is None:
            value = area + confidence * 10.0
        else:
            iou = _bbox_iou(prev_bbox, bbox)
            value = iou * 1_000_000.0 + area + confidence * 10.0
        if value > best_value:
            best_value = value
            best_face = face
    return best_face


def _load_retinaface_module(cache_dir: Path) -> Any:
    global _RETINAFACE_MODULE, _RETINAFACE_IMPORT_ATTEMPTED, _RETINAFACE_IMPORT_ERROR

    if _RETINAFACE_IMPORT_ATTEMPTED:
        if _RETINAFACE_MODULE is None and _RETINAFACE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "RetinaFace import failed: "
                f"{type(_RETINAFACE_IMPORT_ERROR).__name__}: {_RETINAFACE_IMPORT_ERROR}. "
                "If message mentions tf-keras, run: uv add tf-keras"
            ) from _RETINAFACE_IMPORT_ERROR
        if _RETINAFACE_MODULE is None:
            raise RuntimeError("RetinaFace is not installed. Install it with: uv add retina-face")
        return _RETINAFACE_MODULE

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DEEPFACE_HOME", str(cache_dir))
    _RETINAFACE_IMPORT_ATTEMPTED = True
    try:
        _RETINAFACE_MODULE = importlib.import_module("retinaface.RetinaFace")
        _RETINAFACE_IMPORT_ERROR = None
    except Exception as import_error:  # pragma: no cover
        _RETINAFACE_MODULE = None
        _RETINAFACE_IMPORT_ERROR = import_error
        raise RuntimeError(
            "RetinaFace import failed: "
            f"{type(import_error).__name__}: {import_error}. "
            "If message mentions tf-keras, run: uv add tf-keras"
        ) from import_error

    return _RETINAFACE_MODULE


def detect_main_face_on_array(
    image: np.ndarray,
    threshold: float,
    max_side: int,
    prev_bbox: Optional[List[int]] = None,
    detector_module: Any = None,
) -> DetectionRecord:
    detector = detector_module
    if detector is None:
        try:
            detector = _load_retinaface_module(Path(settings.RETINAFACE_WEIGHT_DIR))
        except RuntimeError:
            return DetectionRecord(status="detect_error")

    resized, scale = _resize_for_detection(image, max_side=max_side)
    try:
        result = detector.detect_faces(resized, threshold=threshold)
    except Exception as exc:
        logger.debug("Detection failed on in-memory image: {}", exc)
        return DetectionRecord(status="detect_error")

    faces = _extract_all_faces(result)
    if not faces:
        return DetectionRecord(status="no_face")

    scaled_prev_bbox = None if prev_bbox is None else [int(round(v * scale)) for v in prev_bbox]
    selected = _choose_main_face(faces, prev_bbox=scaled_prev_bbox)
    if selected is None:
        return DetectionRecord(status="no_face")

    bbox = selected["bbox"]
    landmarks = selected["landmarks"]
    confidence = round(float(selected["score"]), 6)
    num_faces = len(faces)

    if scale != 1.0:
        bbox = [int(round(v / scale)) for v in bbox]
        if landmarks is not None:
            landmarks = {
                key: [float(point[0] / scale), float(point[1] / scale)]
                for key, point in landmarks.items()
            }

    h, w = image.shape[:2]
    clipped_bbox = _clip_bbox(bbox, width=w, height=h)
    if clipped_bbox is None:
        return DetectionRecord(status="no_face")

    return DetectionRecord(
        status="ok",
        num_faces=num_faces,
        confidence=confidence,
        bbox=clipped_bbox,
        landmarks=landmarks,
        source="detected",
    )


def _retinaface_landmarks_to_template_order(
    landmarks: Optional[Dict[str, List[float]]],
) -> Optional[np.ndarray]:
    if landmarks is None:
        return None

    points: List[List[float]] = []
    for key in _REQUIRED_LANDMARK_KEYS:
        if key not in landmarks:
            return None
        value = landmarks[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        points.append([float(value[0]), float(value[1])])
    return np.array(points, dtype=np.float32)


def _scaled_destination_template(output_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = output_size
    dst = ARCFACE_112_TEMPLATE.copy()
    dst[:, 0] *= out_w / 112.0
    dst[:, 1] *= out_h / 112.0
    return dst.astype(np.float32)


def _infer_align_canvas_size(aligned_size: Tuple[int, int]) -> Tuple[int, int]:
    canvas_side = int(np.ceil(max(float(max(aligned_size)) * 1.5, 320.0)))
    return canvas_side, canvas_side


def align_image_5pts(
    image: np.ndarray,
    landmarks: Optional[Dict[str, List[float]]],
    output_size: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    src = _retinaface_landmarks_to_template_order(landmarks)
    if src is None:
        return None, None, "missing_landmarks"

    dst = _scaled_destination_template(output_size=output_size)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None, None, "align_fail"

    out_w, out_h = output_size
    aligned = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return aligned, matrix, "ok"


def _transform_bbox_with_affine(
    bbox: Optional[List[int]],
    affine_matrix: Optional[np.ndarray],
) -> Optional[List[float]]:
    if bbox is None or affine_matrix is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners[None, :, :], affine_matrix)[0]
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)
    if max_xy[0] <= min_xy[0] or max_xy[1] <= min_xy[1]:
        return None
    return [float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])]


def _build_square_crop_box(
    bbox: Optional[List[float]],
    image_size: Tuple[int, int],
    crop_scale: float,
) -> Optional[List[int]]:
    if bbox is None:
        return None

    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    side = max(x2 - x1, y2 - y1) * max(1.0, float(crop_scale))
    side = min(side, float(min(image_w, image_h)))
    if side <= 0:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x1 = cx - side / 2.0
    crop_y1 = cy - side / 2.0

    crop_x1 = min(max(0.0, crop_x1), max(0.0, float(image_w) - side))
    crop_y1 = min(max(0.0, crop_y1), max(0.0, float(image_h) - side))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    crop_box = [
        int(np.floor(crop_x1)),
        int(np.floor(crop_y1)),
        int(np.ceil(crop_x2)),
        int(np.ceil(crop_y2)),
    ]
    crop_box[0] = max(0, min(crop_box[0], image_w - 1))
    crop_box[1] = max(0, min(crop_box[1], image_h - 1))
    crop_box[2] = max(crop_box[0] + 1, min(crop_box[2], image_w))
    crop_box[3] = max(crop_box[1] + 1, min(crop_box[3], image_h))
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None
    return crop_box


def crop_aligned_face_from_bbox(
    aligned_image: np.ndarray,
    aligned_bbox: Optional[List[float]],
    crop_scale: float,
    output_size: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[List[int]], str]:
    crop_box = _build_square_crop_box(
        bbox=aligned_bbox,
        image_size=aligned_image.shape[:2],
        crop_scale=crop_scale,
    )
    if crop_box is None:
        return None, None, "crop_fail"

    x1, y1, x2, y2 = crop_box
    cropped = aligned_image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None, "crop_fail"

    out_w, out_h = output_size
    if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return cropped, crop_box, "ok"


def encode_image_to_bytes(image: np.ndarray, image_format: str, jpeg_quality: int) -> bytes:
    ext = image_format.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    params: List[int]
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
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
    return f"{category}/{video_name}/{stem}".replace("\\", "/")


def _safe_label_id(row: Dict[str, str]) -> int:
    label_text = str(row.get("label") or row.get("binary_label") or "").strip().lower()
    if label_text in {"real", "original", "0"}:
        return 0
    if label_text in {"fake", "1"}:
        return 1
    return -1


def _extract_shard_index(shard_path: Path) -> Optional[int]:
    match = re.fullmatch(r"shard-(\d{6})\.tar", shard_path.name)
    return int(match.group(1)) if match else None


def infer_start_shard(output_dir: Path) -> int:
    shard_indices = [
        idx
        for idx in (_extract_shard_index(path) for path in output_dir.glob("shard-*.tar"))
        if idx is not None
    ]
    return 0 if not shard_indices else max(shard_indices) + 1


def load_processed_keys_from_audit(audit_csv: Optional[Path]) -> Set[str]:
    keys: Set[str] = set()
    if audit_csv is None or not audit_csv.exists():
        return keys

    with audit_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
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
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    return f"{parent}/{base}" if parent else base


def load_processed_keys_from_existing_shards(output_dir: Path) -> Set[str]:
    keys: Set[str] = set()
    for shard_path in tqdm(
        sorted(output_dir.glob("shard-*.tar")),
        desc="Scan existing shards for resume",
        leave=False,
    ):
        try:
            with tarfile.open(shard_path, "r") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    key = _base_key_from_tar_member(member.name)
                    if key:
                        keys.add(key)
        except Exception as exc:
            logger.warning("Failed to scan shard {}: {}", shard_path, exc)
    return keys


def append_audit_row(audit_csv: Path, row: Dict[str, Any]) -> None:
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = audit_csv.exists()
    fieldnames = list(row.keys())
    with audit_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists or audit_csv.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def build_audit_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(sample["json"].decode("utf-8"))


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def _interpolate_bbox(b1: List[int], b2: List[int], t: float) -> List[int]:
    return [int(round(_lerp(float(v1), float(v2), t))) for v1, v2 in zip(b1, b2)]


def _interpolate_landmarks(
    l1: Dict[str, List[float]],
    l2: Dict[str, List[float]],
    t: float,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key in _REQUIRED_LANDMARK_KEYS:
        out[key] = [
            float(_lerp(l1[key][0], l2[key][0], t)),
            float(_lerp(l1[key][1], l2[key][1], t)),
        ]
    return out


def _make_keyframe_indices(n: int, detect_every_k: int) -> List[int]:
    if n <= 0:
        return []
    step = max(1, int(detect_every_k))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return indices


def _build_interpolated_detections(
    keyframe_detections: Dict[int, DetectionRecord],
    key_indices: Iterable[int],
) -> Dict[int, DetectionRecord]:
    interpolated: Dict[int, DetectionRecord] = {}
    ok_key_indices = [
        idx for idx in key_indices if keyframe_detections.get(idx) and keyframe_detections[idx].status == "ok"
    ]
    for start_idx, end_idx in zip(ok_key_indices[:-1], ok_key_indices[1:]):
        if end_idx <= start_idx + 1:
            continue

        start_rec = keyframe_detections[start_idx]
        end_rec = keyframe_detections[end_idx]
        if start_rec.bbox is None or end_rec.bbox is None:
            continue
        if start_rec.landmarks is None or end_rec.landmarks is None:
            continue

        gap = end_idx - start_idx
        for mid in range(start_idx + 1, end_idx):
            t = (mid - start_idx) / float(gap)
            interpolated[mid] = DetectionRecord(
                status="ok",
                num_faces=max(start_rec.num_faces, end_rec.num_faces),
                confidence=round(min(start_rec.confidence, end_rec.confidence), 6),
                bbox=_interpolate_bbox(start_rec.bbox, end_rec.bbox, t),
                landmarks=_interpolate_landmarks(start_rec.landmarks, end_rec.landmarks, t),
                source="interpolated",
            )
    return interpolated


class FaceDetectionPipeline:
    def __init__(
        self,
        config: FaceDetectionConfig,
        detector_module: Any = None,
        shard_writer_cls: Any = ShardWriter,
    ) -> None:
        self.config = config
        self.detector_module = detector_module
        self.shard_writer_cls = shard_writer_cls
        self.processed_keys: Set[str] = set()
        self._warned_skip_no_face_semantics = False

    def _ensure_detector_module(self) -> Any:
        if self.detector_module is None:
            self.detector_module = _load_retinaface_module(self.config.retinaface_cache_dir)
        return self.detector_module

    def _warn_skip_no_face_semantics(self) -> None:
        if self.config.skip_no_face or self._warned_skip_no_face_semantics:
            return
        logger.warning(
            "--skip-no-face=false does not emit no-face frames. "
            "This stage still writes only aligned face samples to shards."
        )
        self._warned_skip_no_face_semantics = True

    def _effective_output_dir(self) -> Path:
        if self.config.split:
            return self.config.output_dir / self.config.split
        return self.config.output_dir

    def _effective_audit_csv(self) -> Optional[Path]:
        if self.config.audit_csv is None:
            return None
        if self.config.split is None:
            return self.config.audit_csv
        return self.config.audit_csv.parent / self.config.split / self.config.audit_csv.name

    def _load_metadata_rows(self) -> List[Dict[str, str]]:
        if not self.config.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.config.metadata_csv}")

        with self.config.metadata_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_fields = sorted(_REQUIRED_METADATA_FIELDS - fieldnames)
            if missing_fields:
                raise RuntimeError(
                    "Metadata CSV is missing required columns: " + ", ".join(missing_fields)
                )
            if self.config.split and "split" not in fieldnames:
                raise RuntimeError(
                    f"Metadata CSV is missing required column: split for split={self.config.split}"
                )
            rows = list(reader)

        if self.config.limit is not None:
            rows = rows[: self.config.limit]
        if self.config.split is not None:
            split_filter = self.config.split.strip().lower()
            rows = [row for row in rows if _normalize_text(row.get("split")).lower() == split_filter]
        return rows

    def _group_videos(
        self,
        rows: List[Dict[str, str]],
    ) -> List[Tuple[str, List[Tuple[int, Dict[str, str]]]]]:
        grouped = _group_rows_by_video(rows=rows, category_filter=self.config.category)
        video_items = list(grouped.items())
        if not video_items:
            if self.config.split:
                raise RuntimeError(f"No matching rows/videos found for split={self.config.split}.")
            raise RuntimeError("No matching rows/videos found to process.")
        return video_items

    def _prepare_resume_state(self) -> int:
        output_dir = self._effective_output_dir()
        audit_csv = self._effective_audit_csv()
        output_dir.mkdir(parents=True, exist_ok=True)
        start_shard = infer_start_shard(output_dir)
        self.processed_keys = load_processed_keys_from_audit(audit_csv)
        self.processed_keys |= load_processed_keys_from_existing_shards(output_dir)
        return start_shard

    def _precompute_keyframe_detections(
        self,
        sorted_rows: List[Tuple[int, Dict[str, str]]],
    ) -> Tuple[Dict[int, DetectionRecord], Dict[str, int]]:
        stats = {
            "keyframe_detected": 0,
            "keyframe_no_face": 0,
            "keyframe_missing_file": 0,
            "keyframe_read_fail": 0,
            "keyframe_detect_error": 0,
        }

        keyframe_detections: Dict[int, DetectionRecord] = {}
        key_indices = _make_keyframe_indices(len(sorted_rows), self.config.detect_every_k)
        prev_bbox: Optional[List[int]] = None
        detector = self._ensure_detector_module()

        for idx in key_indices:
            _, row = sorted_rows[idx]
            image_path = self.config.frame_root / row.get("frame_path", "")
            image, load_status = load_image(image_path)
            if load_status == "missing_file":
                keyframe_detections[idx] = DetectionRecord(status="missing_file")
                stats["keyframe_missing_file"] += 1
                continue
            if load_status == "read_fail":
                keyframe_detections[idx] = DetectionRecord(status="read_fail")
                stats["keyframe_read_fail"] += 1
                continue

            record = detect_main_face_on_array(
                image=image,
                threshold=self.config.threshold,
                max_side=self.config.max_side,
                prev_bbox=prev_bbox,
                detector_module=detector,
            )
            if record.status == "ok":
                prev_bbox = record.bbox
                stats["keyframe_detected"] += 1
            elif record.status == "no_face":
                stats["keyframe_no_face"] += 1
            else:
                stats["keyframe_detect_error"] += 1
            keyframe_detections[idx] = record

        return keyframe_detections, stats

    def _build_sample(
        self,
        video_id: str,
        row: Dict[str, str],
        record: DetectionRecord,
        cropped_img: np.ndarray,
        affine_matrix: np.ndarray,
        crop_box: List[int],
    ) -> Dict[str, Any]:
        out_h, out_w = cropped_img.shape[:2]
        sample_key = build_sample_key(row)
        label_id = _safe_label_id(row)
        crop_width = int(crop_box[2] - crop_box[0])
        crop_height = int(crop_box[3] - crop_box[1])
        metadata = {
            "key": sample_key,
            "video_id": video_id,
            "frame_path": row.get("frame_path", ""),
            "category": row.get("category", ""),
            "video_name": row.get("video_name", ""),
            "label": row.get("label", ""),
            "binary_label": row.get("binary_label", ""),
            "split": row.get("split", ""),
            "frame_number": row.get("frame_number", ""),
            "original_frame_index": row.get("original_frame_index", ""),
            "timestamp": row.get("timestamp", ""),
            "video_fps": row.get("video_fps", ""),
            "extraction_fps": row.get("extraction_fps", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
            "face_detected": 1,
            "num_faces": record.num_faces,
            "face_confidence": record.confidence,
            "bbox_x1": int(record.bbox[0]) if record.bbox is not None else 0,
            "bbox_y1": int(record.bbox[1]) if record.bbox is not None else 0,
            "bbox_x2": int(record.bbox[2]) if record.bbox is not None else 0,
            "bbox_y2": int(record.bbox[3]) if record.bbox is not None else 0,
            "bbox_source": record.source,
            "landmarks": record.landmarks,
            "alignment_mode": "similarity_5pts_then_bbox_crop",
            "crop_scale": self.config.crop_scale,
            "align_status": "ok",
            "crop_status": "ok",
            "crop_x1": int(crop_box[0]),
            "crop_y1": int(crop_box[1]),
            "crop_x2": int(crop_box[2]),
            "crop_y2": int(crop_box[3]),
            "crop_size": int(max(crop_width, crop_height)),
            "crop_source": "aligned_bbox_1.3x",
            "aligned_width": int(out_w),
            "aligned_height": int(out_h),
            "aligned_size": [int(self.config.aligned_size[0]), int(self.config.aligned_size[1])],
            "align_canvas_width": int(self.config.align_canvas_size[0]),
            "align_canvas_height": int(self.config.align_canvas_size[1]),
            "align_canvas_size": [
                int(self.config.align_canvas_size[0]),
                int(self.config.align_canvas_size[1]),
            ],
            "affine_matrix": affine_matrix.tolist(),
            "image_format": self.config.image_format,
            "detect_every_k": int(self.config.detect_every_k),
        }

        sample: Dict[str, Any] = {
            "__key__": sample_key,
            "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        }
        image_bytes = encode_image_to_bytes(
            cropped_img,
            self.config.image_format,
            self.config.jpeg_quality,
        )
        if self.config.image_format.lower() in {".jpg", ".jpeg"}:
            sample["jpg"] = image_bytes
        else:
            sample["png"] = image_bytes
        if label_id in {0, 1}:
            sample["cls"] = str(label_id).encode("utf-8")
        return sample

    def _append_audit_row(self, sample: Dict[str, Any]) -> None:
        audit_csv = self._effective_audit_csv()
        if audit_csv is None:
            return
        append_audit_row(audit_csv, build_audit_row(sample))

    def _write_sample(self, sink: Any, sample: Dict[str, Any]) -> None:
        sink.write(sample)
        self.processed_keys.add(sample["__key__"])
        self._append_audit_row(sample)

    def _process_video(
        self,
        sink: Any,
        video_id: str,
        indexed_rows: List[Tuple[int, Dict[str, str]]],
    ) -> VideoProcessingStats:
        local_rows = _sort_video_rows(indexed_rows)
        stats = VideoProcessingStats(total_rows=len(local_rows))

        key_indices = _make_keyframe_indices(len(local_rows), self.config.detect_every_k)
        keyframe_detections, keyframe_stats = self._precompute_keyframe_detections(local_rows)
        interpolated_detections = _build_interpolated_detections(keyframe_detections, key_indices)
        stats.interpolated_frames += len(interpolated_detections)
        stats.detected_frames_direct += keyframe_stats["keyframe_detected"]

        detector = self._ensure_detector_module()
        last_known_bbox: Optional[List[int]] = None
        progress = tqdm(total=len(local_rows), desc=f"Frames {str(video_id)[:40]}", leave=False)

        try:
            for idx, (_, row) in enumerate(local_rows):
                sample_key = build_sample_key(row)
                if sample_key in self.processed_keys:
                    stats.skipped_existing += 1
                    progress.update(1)
                    continue

                image_path = self.config.frame_root / row.get("frame_path", "")
                image, load_status = load_image(image_path)
                stats.processed_frames += 1
                if load_status == "missing_file":
                    stats.missing_file += 1
                    progress.update(1)
                    continue
                if load_status == "read_fail":
                    stats.read_fail += 1
                    progress.update(1)
                    continue
                assert image is not None

                record = keyframe_detections.get(idx)
                if record is None:
                    record = interpolated_detections.get(idx)
                if record is None:
                    record = detect_main_face_on_array(
                        image=image,
                        threshold=self.config.threshold,
                        max_side=self.config.max_side,
                        prev_bbox=last_known_bbox,
                        detector_module=detector,
                    )
                    if record.status == "ok":
                        stats.detected_frames_direct += 1

                if record.status == "missing_file":
                    stats.missing_file += 1
                    progress.update(1)
                    continue
                if record.status == "read_fail":
                    stats.read_fail += 1
                    progress.update(1)
                    continue
                if record.status == "detect_error":
                    stats.detect_error += 1
                    progress.update(1)
                    continue
                if record.status != "ok" or record.bbox is None:
                    stats.no_face += 1
                    if self.config.skip_no_face:
                        stats.skipped_no_face += 1
                    progress.update(1)
                    continue

                last_known_bbox = record.bbox
                stats.detected_frames += 1

                aligned_img, affine_matrix, align_status = align_image_5pts(
                    image=image,
                    landmarks=record.landmarks,
                    output_size=self.config.align_canvas_size,
                )
                if align_status == "missing_landmarks":
                    stats.missing_landmarks += 1
                    progress.update(1)
                    continue
                if align_status != "ok" or aligned_img is None or affine_matrix is None:
                    stats.align_fail += 1
                    progress.update(1)
                    continue

                aligned_bbox = _transform_bbox_with_affine(record.bbox, affine_matrix)
                cropped_img, crop_box, crop_status = crop_aligned_face_from_bbox(
                    aligned_image=aligned_img,
                    aligned_bbox=aligned_bbox,
                    crop_scale=self.config.crop_scale,
                    output_size=self.config.aligned_size,
                )
                if crop_status != "ok" or cropped_img is None or crop_box is None:
                    stats.align_fail += 1
                    progress.update(1)
                    continue

                try:
                    sample = self._build_sample(
                        video_id=video_id,
                        row=row,
                        record=record,
                        cropped_img=cropped_img,
                        affine_matrix=affine_matrix,
                        crop_box=crop_box,
                    )
                except Exception as exc:
                    logger.warning("Encode failed on {}: {}", image_path, exc)
                    stats.encode_fail += 1
                    progress.update(1)
                    continue

                self._write_sample(sink, sample)
                stats.written_samples += 1
                progress.update(1)
        finally:
            progress.close()

        return stats

    def run(self) -> None:
        self._warn_skip_no_face_semantics()
        detector = self._ensure_detector_module()
        effective_output_dir = self._effective_output_dir()
        effective_audit_csv = self._effective_audit_csv()
        logger.info("RetinaFace cache dir: {}", self.config.retinaface_cache_dir)
        if detector is None:  # pragma: no cover
            raise RuntimeError("RetinaFace detector is unavailable")

        rows = self._load_metadata_rows()
        video_items = self._group_videos(rows)
        start_shard = self._prepare_resume_state()

        logger.info("Total rows loaded: {}", len(rows))
        logger.info("Total videos: {}", len(video_items))
        logger.info("Split filter: {}", self.config.split or "all")
        logger.info("Output dir: {}", self.config.output_dir)
        logger.info("Effective output dir: {}", effective_output_dir)
        logger.info("Effective audit path: {}", effective_audit_csv if effective_audit_csv else "disabled")
        logger.info("max_side: {}", self.config.max_side)
        logger.info("aligned_size: {}", self.config.aligned_size)
        logger.info("align_canvas_size: {}", self.config.align_canvas_size)
        logger.info("crop_scale: {}", self.config.crop_scale)
        logger.info("detect_every_k: {}", self.config.detect_every_k)
        logger.info("image_format: {}", self.config.image_format)
        logger.info("skip_no_face: {}", self.config.skip_no_face)
        logger.info("Resume start_shard: {}", start_shard)
        logger.info("Total processed keys for resume: {}", len(self.processed_keys))

        total_stats = PipelineStats()
        shard_pattern = str(effective_output_dir / "shard-%06d.tar")
        with self.shard_writer_cls(
            shard_pattern,
            maxcount=self.config.shard_maxcount,
            maxsize=self.config.shard_maxsize,
            start_shard=start_shard,
        ) as sink:
            video_progress = tqdm(total=len(video_items), desc="Videos")
            try:
                for video_id, indexed_rows in video_items:
                    stats = self._process_video(sink=sink, video_id=video_id, indexed_rows=indexed_rows)
                    total_stats.merge_video(stats)

                    logger.info(
                        "Done video={} | rows={} | written={} | detected={} | interpolated={} | no_face={} | skipped_existing={} | missing_landmarks={} | align_fail={}",
                        video_id,
                        stats.total_rows,
                        stats.written_samples,
                        stats.detected_frames,
                        stats.interpolated_frames,
                        stats.no_face,
                        stats.skipped_existing,
                        stats.missing_landmarks,
                        stats.align_fail,
                    )
                    video_progress.update(1)
            finally:
                video_progress.close()

        logger.info("=== FINAL SUMMARY ===")
        for key, value in total_stats.as_dict().items():
            logger.info("{}: {}", key, value)


def run_pipeline(
    metadata_csv: Path,
    frame_root: Path,
    output_dir: Path,
    category: Optional[str],
    threshold: float,
    max_side: int,
    aligned_size: Tuple[int, int],
    crop_scale: float,
    image_format: str,
    jpeg_quality: int,
    shard_maxcount: int,
    shard_maxsize: int,
    limit: Optional[int],
    skip_no_face: bool,
    audit_csv: Optional[Path],
    detect_every_k: int,
    split: Optional[str] = None,
) -> None:
    normalized_split = None if split is None else split.strip().lower()
    if normalized_split is not None and normalized_split not in _VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    align_canvas_size = _infer_align_canvas_size(aligned_size)

    config = FaceDetectionConfig(
        metadata_csv=metadata_csv,
        frame_root=frame_root,
        output_dir=output_dir,
        category=category,
        split=normalized_split,
        threshold=threshold,
        max_side=max_side,
        aligned_size=aligned_size,
        align_canvas_size=align_canvas_size,
        crop_scale=crop_scale,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        shard_maxcount=shard_maxcount,
        shard_maxsize=shard_maxsize,
        limit=limit,
        skip_no_face=skip_no_face,
        audit_csv=audit_csv,
        detect_every_k=detect_every_k,
        retinaface_cache_dir=Path(settings.RETINAFACE_WEIGHT_DIR),
    )
    FaceDetectionPipeline(config=config).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RetinaFace detect + main-face selection + conservative 5pt alignment + WebDataset shards"
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=str(settings.FRAME_DATA_DIR / settings.FRAME_EXTRACTION_METADATA_CSV),
        help="Path to frame metadata CSV",
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
    parser.add_argument("--category", type=str, default=None, help="Only process one category")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=sorted(_VALID_SPLITS),
        help="Only process one split and write output to a split-specific subdirectory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_DETECTION_THRESHOLD,
        help="RetinaFace confidence threshold",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows for testing")
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize image before detection so longest side <= max-side",
    )
    parser.add_argument("--aligned-width", type=int, default=224, help="Final cropped face output width")
    parser.add_argument("--aligned-height", type=int, default=224, help="Final cropped face output height")
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.3,
        help="Conservative face crop scale (>1 keeps more context around the face)",
    )
    parser.add_argument(
        "--detect-every-k",
        type=int,
        default=5,
        help="Run full RetinaFace detection every K frames and interpolate between successful keyframes",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
        help="Encoded image format stored inside shard",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality if using jpg/jpeg output")
    parser.add_argument("--shard-maxcount", type=int, default=10000, help="Maximum samples per shard")
    parser.add_argument(
        "--shard-maxsize",
        type=int,
        default=2_000_000_000,
        help="Maximum bytes per shard",
    )
    parser.add_argument("--skip-no-face", action="store_true", help="Skip frames with no detected face")
    parser.add_argument(
        "--audit-csv",
        type=str,
        default=str(settings.AUDIT_FILE),
        help="Optional audit CSV path",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_pipeline(
        metadata_csv=Path(args.metadata_csv),
        frame_root=Path(args.frame_root),
        output_dir=Path(args.output_dir),
        category=args.category,
        split=args.split,
        threshold=args.threshold,
        max_side=args.max_side,
        aligned_size=(args.aligned_width, args.aligned_height),
        crop_scale=args.crop_scale,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        shard_maxcount=args.shard_maxcount,
        shard_maxsize=args.shard_maxsize,
        limit=args.limit,
        skip_no_face=args.skip_no_face,
        audit_csv=Path(args.audit_csv) if args.audit_csv else None,
        detect_every_k=args.detect_every_k,
    )


__all__ = [
    "DetectionRecord",
    "FaceDetectionConfig",
    "FaceDetectionPipeline",
    "PipelineStats",
    "VideoProcessingStats",
    "_build_square_crop_box",
    "_base_key_from_tar_member",
    "_group_rows_by_video",
    "_infer_align_canvas_size",
    "_interpolate_bbox",
    "_interpolate_landmarks",
    "_make_keyframe_indices",
    "_sort_video_rows",
    "_transform_bbox_with_affine",
    "align_image_5pts",
    "append_audit_row",
    "build_audit_row",
    "build_sample_key",
    "crop_aligned_face_from_bbox",
    "detect_main_face_on_array",
    "encode_image_to_bytes",
    "infer_start_shard",
    "load_image",
    "load_processed_keys_from_audit",
    "load_processed_keys_from_existing_shards",
    "main",
    "parse_args",
    "run_pipeline",
]
