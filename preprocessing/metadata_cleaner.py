from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd


def normalize_label_value(raw_label: str) -> tuple[int, str] | None:
    normalized = str(raw_label).strip().lower()
    if normalized in {"real", "0", "false"}:
        return 0, "REAL"
    if normalized in {"fake", "1", "true"}:
        return 1, "FAKE"
    return None


def clean_metadata(
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

    normalized_labels = frame["Label"].map(normalize_label_value)
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
    parser = argparse.ArgumentParser(description="Clean and normalize metadata CSV")
    parser.add_argument("--dataset-root", type=Path, default=Path("FaceForensics++_C23 2"))
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("FaceForensics++_C23 2/csv/FF++_Metadata_Shuffled.csv"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/clean_metadata.csv"))
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--skip-exists-check", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cleaned = clean_metadata(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
        min_frames=args.min_frames,
        check_file_exists=not args.skip_exists_check,
    )
    print(f"Saved cleaned metadata -> {args.output_csv} (rows={len(cleaned)})")


if __name__ == "__main__":
    main()
