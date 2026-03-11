from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import pandas as pd

from preprocessing.frame_extractor import extract_frames_from_metadata


def _normalize_label_value(raw_label: str) -> tuple[int, str] | None:
    normalized = str(raw_label).strip().lower()
    if normalized in {"real", "0", "false"}:
        return 0, "REAL"
    if normalized in {"fake", "1", "true"}:
        return 1, "FAKE"
    return None


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


def _clean_metadata(
    dataset_root: Path,
    metadata_csv: Path,
    output_csv: Path,
    min_frames: int,
    check_file_exists: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(metadata_csv)
    frame = frame[[column for column in frame.columns if not str(column).startswith("Unnamed")]].copy()

    required_columns = {"File Path", "Label", "Frame Count"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing_sorted = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in metadata CSV: {missing_sorted}")

    frame["video_rel_path"] = frame["File Path"].astype(str)
    frame["video_abs_path"] = frame["video_rel_path"].map(lambda value: str((dataset_root / value).resolve()))
    frame["video_id"] = frame["video_rel_path"].map(lambda value: Path(value).stem)
    frame["method"] = frame["video_rel_path"].map(
        lambda value: Path(value).parts[0] if len(Path(value).parts) > 0 else "unknown"
    )
    frame["num_frames"] = frame["Frame Count"].astype(int)

    normalized_labels = frame["Label"].map(_normalize_label_value)
    valid_mask = normalized_labels.notna()
    frame = frame[valid_mask].copy()
    normalized_labels = normalized_labels[valid_mask]

    frame["label"] = normalized_labels.map(lambda value: value[0]).astype(int)
    frame["label_str"] = normalized_labels.map(lambda value: value[1])

    frame = frame[frame["num_frames"] >= int(min_frames)]

    if check_file_exists:
        frame = frame[frame["video_abs_path"].map(lambda value: Path(value).exists())]

    output_columns = [
        "video_abs_path",
        "video_rel_path",
        "video_id",
        "method",
        "label",
        "label_str",
        "num_frames",
    ]
    cleaned = frame[output_columns].reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False)
    return cleaned


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

    run_config_parser = subparsers.add_parser("run-config", help="Run clean + extract from TOML config")
    run_config_parser.add_argument("--config", type=Path, default=Path("preprocess_config.toml"))

    return parser


def run_command(args: argparse.Namespace) -> None:
    if args.command == "clean-metadata":
        cleaned = _clean_metadata(
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

        cleaned = _clean_metadata(
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
