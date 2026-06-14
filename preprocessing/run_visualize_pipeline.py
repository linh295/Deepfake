from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

from configs.loggings import logger, setup_logging
from configs.settings import settings
from preprocessing._face_detection_pipeline import run_pipeline as run_face_detection_pipeline
from preprocessing.frame_extractor import FrameExtractor


def build_visualize_manifest(video_root: Path, manifest_path: Path, split: str) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["video_id", "video_path", "category", "binary_label", "label", "split"]
    rows = []
    missing_videos = []

    deepfake_dir = video_root / "deepfake"
    deepfake_videos = [f.name for f in deepfake_dir.glob("*.mp4")] if deepfake_dir.exists() else []

    categories = [
        ("deepfake", 0, "fake"),
        ("face2face", 0, "fake"),
        ("faceshifter", 0, "fake"),
        ("faceswap", 0, "fake"),
        ("neural textures", 0, "fake"),
        ("original", 1, "real"),
    ]

    for video_filename in deepfake_videos:
        video_id = Path(video_filename).stem
        orig_id = video_id.split("_")[0]

        for category, binary_label, label in categories:
            if category == "original":
                filename = f"{orig_id}.mp4"
            else:
                filename = video_filename

            video_path = video_root / category / filename
            if not video_path.exists():
                missing_videos.append(str(video_path))
                continue

            rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(Path("visualize") / category / filename),
                    "category": category,
                    "binary_label": binary_label,
                    "label": label,
                    "split": split,
                }
            )

    if missing_videos:
        missing_list = "\n".join(f"  - {item}" for item in missing_videos)
        raise FileNotFoundError(f"Missing visualize videos:\n{missing_list}")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame extraction and face detection for the prepared visualize videos"
    )
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        default=settings.ROOT_DIR / "visualize",
        help="Directory containing the prepared visualize videos",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.ROOT_DIR / "artifacts" / "visualize_master.csv",
        help="Manifest CSV generated for the visualize videos",
    )
    parser.add_argument(
        "--frame-output-dir",
        type=Path,
        default=settings.ROOT_DIR / "frame_data_visualize",
        help="Directory for extracted frames and metadata",
    )
    parser.add_argument(
        "--crop-output-dir",
        type=Path,
        default=settings.ROOT_DIR / "crop_data_visualize",
        help="Directory for face-detection shards",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=settings.ROOT_DIR / "audit_visualize" / "face_detection_audit.csv",
        help="Audit CSV path for face detection",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split value to write into the generated manifest and face-detection output",
    )
    parser.add_argument("--fps", type=int, default=settings.TARGET_FPS, help="Frame extraction FPS")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=settings.JPEG_QUALITY,
        help="JPEG quality used by frame extraction",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.NUM_WORKERS,
        help="Worker count used by frame extraction",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_DETECTION_THRESHOLD,
        help="RetinaFace confidence threshold",
    )
    parser.add_argument("--max-side", type=int, default=640, help="Max side length before detection")
    parser.add_argument("--aligned-width", type=int, default=224, help="Face crop output width")
    parser.add_argument("--aligned-height", type=int, default=224, help="Face crop output height")
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.3,
        help="Crop scale used around the detected face bbox",
    )
    parser.add_argument(
        "--detect-every-k",
        type=int,
        default=5,
        help="Run full detection every K frames",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
        help="Encoded image format stored in the shard",
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
        help="Maximum shard size in bytes",
    )
    parser.add_argument(
        "--skip-no-face",
        action="store_true",
        help="Skip frames where no face is detected",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the face-detection stage to the first N metadata rows",
    )
    return parser.parse_args()


def run_visualize_pipeline(
    *,
    visualize_dir: Path,
    manifest_path: Path,
    frame_output_dir: Path,
    crop_output_dir: Path,
    audit_csv: Path,
    split: str,
    fps: int,
    jpeg_quality: int,
    workers: int,
    threshold: float,
    max_side: int,
    aligned_width: int,
    aligned_height: int,
    crop_scale: float,
    detect_every_k: int,
    image_format: str,
    shard_maxcount: int,
    shard_maxsize: int,
    skip_no_face: bool,
    limit: int | None,
) -> None:
    manifest = build_visualize_manifest(visualize_dir, manifest_path, split)

    logger.info("Visualize manifest written to {}", manifest)
    logger.info("Starting frame extraction for visualize videos")
    extractor = FrameExtractor(
        dataset_path=settings.ROOT_DIR,
        output_path=frame_output_dir,
        metadata_csv=settings.FRAME_EXTRACTION_METADATA_CSV,
        manifest_path=manifest,
        audit_csv=settings.FRAME_EXTRACTION_AUDIT_CSV,
        fps=fps,
        jpeg_quality=jpeg_quality,
        num_workers=workers,
        resume=True,
    )
    extractor.extract_all(only_split=split)

    metadata_csv = frame_output_dir / settings.FRAME_EXTRACTION_METADATA_CSV
    logger.info("Starting face detection for visualize videos")
    run_face_detection_pipeline(
        metadata_csv=metadata_csv,
        frame_root=frame_output_dir,
        output_dir=crop_output_dir,
        category=None,
        threshold=threshold,
        max_side=max_side,
        aligned_size=(aligned_width, aligned_height),
        crop_scale=crop_scale,
        image_format=image_format,
        jpeg_quality=100,
        shard_maxcount=shard_maxcount,
        shard_maxsize=shard_maxsize,
        limit=limit,
        skip_no_face=skip_no_face,
        audit_csv=audit_csv,
        detect_every_k=detect_every_k,
        split=split,
    )

    logger.info("Generating visualization figures from cropped shards")
    import sys
    import os
    sys.path.insert(0, str(settings.ROOT_DIR))
    try:
        from visualize_crops import plot_crops
        shard_file = crop_output_dir / split / "shard-000000.tar"
        if shard_file.exists():
            plot_crops(str(shard_file))
        else:
            logger.warning(f"Shard file not found for visualization: {shard_file}")
    except ImportError as e:
        logger.warning(f"Could not load visualize_crops module: {e}")

def main() -> None:
    setup_logging()
    args = parse_args()
    run_visualize_pipeline(
        visualize_dir=args.visualize_dir,
        manifest_path=args.manifest,
        frame_output_dir=args.frame_output_dir,
        crop_output_dir=args.crop_output_dir,
        audit_csv=args.audit_csv,
        split=args.split,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        threshold=args.threshold,
        max_side=args.max_side,
        aligned_width=args.aligned_width,
        aligned_height=args.aligned_height,
        crop_scale=args.crop_scale,
        detect_every_k=args.detect_every_k,
        image_format=args.image_format,
        shard_maxcount=args.shard_maxcount,
        shard_maxsize=args.shard_maxsize,
        skip_no_face=args.skip_no_face,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
