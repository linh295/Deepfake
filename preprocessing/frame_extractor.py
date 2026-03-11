from __future__ import annotations

import argparse

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


REQUIRED_COLUMNS = {"video_abs_path", "video_rel_path", "video_id", "method", "label", "label_str", "num_frames"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _build_frame_indices(video_fps: float, frame_count: int, target_fps: float) -> list[int]:
    if frame_count <= 0:
        return []

    if video_fps <= 0:
        video_fps = target_fps

    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    duration_seconds = frame_count / video_fps
    expected_frames = max(1, int(duration_seconds * target_fps))

    indices: list[int] = []
    for sample_idx in range(expected_frames):
        timestamp_seconds = sample_idx / target_fps
        frame_index = min(int(round(timestamp_seconds * video_fps)), frame_count - 1)
        if not indices or frame_index != indices[-1]:
            indices.append(frame_index)

    if not indices:
        indices = [0]
    return indices


def _normalize_label(row: pd.Series) -> tuple[int, str] | None:
    label_value = str(row.get("label", "")).strip().lower()
    label_text = str(row.get("label_str", "")).strip().lower()

    candidate_values = [label_text, label_value]

    for candidate in candidate_values:
        if candidate in {"real", "0", "false"}:
            return 0, "real"
        if candidate in {"fake", "1", "true"}:
            return 1, "fake"

    return None


def extract_frames_from_metadata(
    clean_csv: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    image_size: tuple[int, int] | None = None,
    max_videos: int = 0,
    image_extension: str = ".jpg",
) -> pd.DataFrame:
    frame = pd.read_csv(clean_csv)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        missing_sorted = ", ".join(sorted(missing_columns))
        raise ValueError(f"clean CSV missing required columns: {missing_sorted}")

    if max_videos > 0:
        frame = frame.head(max_videos)

    ensure_dir(output_dir)
    records: list[dict] = []
    normalized_extension = image_extension if image_extension.startswith(".") else f".{image_extension}"

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Extracting frames"):
        normalized_label = _normalize_label(row)
        if normalized_label is None:
            continue

        label_numeric, label_folder = normalized_label
        video_path = Path(str(row["video_abs_path"]))
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        decoded_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or int(row["num_frames"]))
        if decoded_frame_count <= 0:
            decoded_frame_count = int(row["num_frames"])

        frame_indices = _build_frame_indices(
            video_fps=video_fps,
            frame_count=decoded_frame_count,
            target_fps=target_fps,
        )

        video_output_dir = output_dir / label_folder / str(row["method"]) / str(row["video_id"])
        ensure_dir(video_output_dir)

        saved_paths: list[str] = []
        saved_indices: list[int] = []
        saved_timestamps: list[str] = []

        for saved_idx, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, image = cap.read()
            if not success or image is None:
                continue

            if image_size is not None:
                target_width, target_height = image_size
                image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

            frame_name = f"frame_{saved_idx:05d}{normalized_extension}"
            frame_path = video_output_dir / frame_name
            cv2.imwrite(str(frame_path), image)

            saved_paths.append(_to_relative(frame_path, output_dir))
            saved_indices.append(frame_index)
            timestamp_seconds = frame_index / video_fps if video_fps > 0 else saved_idx / target_fps
            saved_timestamps.append(f"{timestamp_seconds:.6f}")

        cap.release()

        if not saved_paths:
            continue

        records.append(
            {
                "video_id": str(row["video_id"]),
                "method": str(row["method"]),
                "label": label_numeric,
                "label_str": label_folder.upper(),
                "label_folder": label_folder,
                "video_rel_path": str(row["video_rel_path"]),
                "video_abs_path": str(video_path),
                "source_fps": video_fps,
                "target_fps": target_fps,
                "num_saved_frames": len(saved_paths),
                "frame_indices": ";".join(str(index) for index in saved_indices),
                "timestamps_sec": ";".join(saved_timestamps),
                "frame_paths": ";".join(saved_paths),
            }
        )

    manifest = pd.DataFrame.from_records(records)
    manifest.to_csv(output_dir / "frame_manifest.csv", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video frames from cleaned metadata CSV")
    parser.add_argument("--clean-csv", type=Path, default=Path("artifacts/clean_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/frames_5fps"))
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=0)
    parser.add_argument("--image-extension", type=str, default=".jpg")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image_size = None
    if args.resize_width > 0 and args.resize_height > 0:
        image_size = (args.resize_width, args.resize_height)

    manifest = extract_frames_from_metadata(
        clean_csv=args.clean_csv,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        image_size=image_size,
        max_videos=args.max_videos,
        image_extension=args.image_extension,
    )
    print(f"Saved frame manifest -> {args.output_dir / 'frame_manifest.csv'} (videos={len(manifest)})")


if __name__ == "__main__":
    main()
