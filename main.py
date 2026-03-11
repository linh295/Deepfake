from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from preprocessing.data_standardization import CopyMode, OutputMode, standardize_dataset
from preprocessing.frame_extractor import extract_frames_from_metadata
from preprocessing.metadata_cleaner import clean_metadata


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as file:
        return tomllib.load(file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video frames at target FPS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean-metadata", help="Normalize labels and build clean metadata CSV")
    clean_parser.add_argument("--dataset-root", type=Path, default=Path("FaceForensics++_C23 2"))
    clean_parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("FaceForensics++_C23 2/csv/FF++_Metadata_Shuffled.csv"),
    )
    clean_parser.add_argument("--output-csv", type=Path, default=Path("artifacts/clean_metadata.csv"))
    clean_parser.add_argument("--min-frames", type=int, default=16)
    clean_parser.add_argument("--skip-exists-check", action="store_true")

    extract_frames_parser = subparsers.add_parser("extract-frames", help="Extract frames from all videos at a target FPS")
    extract_frames_parser.add_argument("--clean-csv", type=Path, default=Path("artifacts/clean_metadata.csv"))
    extract_frames_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/frames_5fps"))
    extract_frames_parser.add_argument("--target-fps", type=float, default=5.0)
    extract_frames_parser.add_argument("--max-videos", type=int, default=0)
    extract_frames_parser.add_argument("--resize-width", type=int, default=0)
    extract_frames_parser.add_argument("--resize-height", type=int, default=0)
    extract_frames_parser.add_argument("--image-extension", type=str, default=".jpg")

    run_config_parser = subparsers.add_parser("run-config", help="Run clean + extract + standardize from TOML config")
    run_config_parser.add_argument("--config", type=Path, default=Path("preprocess_config.toml"))

    std_parser = subparsers.add_parser("standardize", help="Standardize extracted frames into real/fake folders")
    std_parser.add_argument("--manifest-csv", type=Path, default=Path("artifacts/frames_5fps/frame_manifest.csv"))
    std_parser.add_argument("--frames-root", type=Path, default=Path("artifacts/frames_5fps"))
    std_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_standardized"))
    std_parser.add_argument("--mode", type=str, default="flat", choices=["flat", "by_method"])
    std_parser.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "symlink"])

    return parser


def run_command(args: argparse.Namespace) -> None:
    if args.command == "clean-metadata":
        cleaned = clean_metadata(
            dataset_root=args.dataset_root,
            metadata_csv=args.metadata_csv,
            output_csv=args.output_csv,
            min_frames=args.min_frames,
            check_file_exists=not args.skip_exists_check,
        )
        print(f"Saved cleaned metadata -> {args.output_csv} (rows={len(cleaned)})")
        return

    if args.command == "extract-frames":
        image_size = None
        if args.resize_width > 0 and args.resize_height > 0:
            image_size = (args.resize_width, args.resize_height)

        extract_frames_from_metadata(
            clean_csv=args.clean_csv,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            image_size=image_size,
            max_videos=args.max_videos,
            image_extension=args.image_extension,
        )
        print(f"Saved frame manifest -> {args.output_dir / 'frame_manifest.csv'}")
        return

    if args.command == "standardize":
        standardize_dataset(
            manifest_csv=args.manifest_csv,
            frames_root=args.frames_root,
            output_dir=args.output_dir,
            mode=OutputMode(args.mode),
            copy_mode=CopyMode(args.copy_mode),
        )
        return

    if args.command == "run-config":
        config_path = args.config.resolve()
        config = _load_config(config_path)
        base_dir = config_path.parent

        path_section = config.get("paths", {})
        clean_section = config.get("clean", {})
        extract_section = config.get("extract", {})

        dataset_root = _resolve_path(str(path_section.get("dataset_root", "FaceForensics++_C23 2")), base_dir)
        metadata_csv_value = path_section.get("metadata_csv")
        if metadata_csv_value is None:
            metadata_csv = dataset_root / "csv" / "FF++_Metadata_Shuffled.csv"
        else:
            metadata_csv = _resolve_path(str(metadata_csv_value), base_dir)

        artifact_dir = _resolve_path(str(path_section.get("artifact_dir", "artifacts")), base_dir)
        clean_csv = _resolve_path(str(clean_section.get("output_csv", artifact_dir / "clean_metadata.csv")), base_dir)
        frames_output_dir = _resolve_path(str(extract_section.get("output_dir", artifact_dir / "frames_5fps")), base_dir)

        min_frames = int(clean_section.get("min_frames", 16))
        check_file_exists = bool(clean_section.get("check_file_exists", True))

        resize_width = int(extract_section.get("resize_width", 0))
        resize_height = int(extract_section.get("resize_height", 0))
        image_size = (resize_width, resize_height) if resize_width > 0 and resize_height > 0 else None

        cleaned = clean_metadata(
            dataset_root=dataset_root,
            metadata_csv=metadata_csv,
            output_csv=clean_csv,
            min_frames=min_frames,
            check_file_exists=check_file_exists,
        )

        extract_frames_from_metadata(
            clean_csv=clean_csv,
            output_dir=frames_output_dir,
            target_fps=float(extract_section.get("target_fps", 5.0)),
            image_size=image_size,
            max_videos=int(extract_section.get("max_videos", 0)),
            image_extension=str(extract_section.get("image_extension", ".jpg")),
        )

        std_section = config.get("standardize", {})
        if bool(std_section.get("enabled", True)):
            std_output_dir = _resolve_path(
                str(std_section.get("output_dir", artifact_dir / "dataset_standardized")), base_dir
            )
            standardize_dataset(
                manifest_csv=frames_output_dir / "frame_manifest.csv",
                frames_root=frames_output_dir,
                output_dir=std_output_dir,
                mode=OutputMode(str(std_section.get("mode", "flat"))),
                copy_mode=CopyMode(str(std_section.get("copy_mode", "copy"))),
            )

        print(f"Saved cleaned metadata -> {clean_csv} (rows={len(cleaned)})")
        print(f"Saved frame manifest -> {frames_output_dir / 'frame_manifest.csv'}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
