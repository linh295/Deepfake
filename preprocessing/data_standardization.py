"""
Chuẩn hoá folder dataset real / fake sau khi extract frames.

Input: frame_manifest.csv từ bước extract-frames
Output structure (2 chế độ):

  Mode "flat":
    output_dir/
      real/<video_id>/frame_*.jpg
      fake/<video_id>/frame_*.jpg

  Mode "by_method":
    output_dir/
      real/<method>/<video_id>/frame_*.jpg
      fake/<method>/<video_id>/frame_*.jpg   (giữ nguyên method subfolder)

Hỗ trợ copy hoặc symlink (nhanh hơn, không tốn dung lượng).
"""
from __future__ import annotations

import argparse

import shutil
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class OutputMode(str, Enum):
    FLAT = "flat"           # real/<video_id>/
    BY_METHOD = "by_method" # real/<method>/<video_id>/


class CopyMode(str, Enum):
    COPY = "copy"
    SYMLINK = "symlink"


def _resolve_frame_paths(row: pd.Series, frames_root: Path) -> list[Path]:
    raw_paths = str(row.get("frame_paths", "")).strip()
    if not raw_paths:
        return []
    return [frames_root / p.strip() for p in raw_paths.split(";") if p.strip()]


def _target_dir(
    output_dir: Path,
    label_folder: str,
    method: str,
    video_id: str,
    mode: OutputMode,
) -> Path:
    if mode == OutputMode.FLAT:
        return output_dir / label_folder / video_id
    return output_dir / label_folder / method / video_id


def _transfer_file(src: Path, dst: Path, copy_mode: CopyMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_mode == CopyMode.SYMLINK:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def standardize_dataset(
    manifest_csv: Path,
    frames_root: Path,
    output_dir: Path,
    mode: OutputMode = OutputMode.FLAT,
    copy_mode: CopyMode = CopyMode.COPY,
) -> pd.DataFrame:
    """
    Chuẩn hoá dataset từ frame_manifest.csv thành cấu trúc real/fake rõ ràng.

    Args:
        manifest_csv:  Đường dẫn tới frame_manifest.csv (output của extract-frames).
        frames_root:   Thư mục gốc chứa frames (thường là artifacts/frames_5fps).
        output_dir:    Thư mục output sau chuẩn hoá.
        mode:          "flat" - gộp method vào video_id level.
                       "by_method" - giữ method subfolder.
        copy_mode:     "copy" - copy file, "symlink" - tạo symlink (nhanh hơn).

    Returns:
        DataFrame manifest của các file sau chuẩn hoá.
    """
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    required_columns = {"video_id", "method", "label_folder", "frame_paths"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"manifest_csv thiếu cột: {', '.join(sorted(missing))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Standardizing [{mode}]"):
        label_folder = str(row["label_folder"]).strip().lower()
        if label_folder not in {"real", "fake"}:
            continue

        video_id = str(row["video_id"])
        method = str(row["method"])
        frame_paths = _resolve_frame_paths(row, frames_root)

        if not frame_paths:
            continue

        target_video_dir = _target_dir(output_dir, label_folder, method, video_id, mode)
        transferred: list[str] = []

        for src in frame_paths:
            if not src.exists():
                continue
            dst = target_video_dir / src.name
            _transfer_file(src, dst, copy_mode)
            transferred.append(str(dst.relative_to(output_dir)))

        if transferred:
            records.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "label": int(row.get("label", -1)),
                    "label_folder": label_folder,
                    "num_frames": len(transferred),
                    "frame_paths": ";".join(transferred),
                    "target_dir": str(target_video_dir.relative_to(output_dir)),
                }
            )

    result_df = pd.DataFrame.from_records(records)
    manifest_out = output_dir / "standardized_manifest.csv"
    result_df.to_csv(manifest_out, index=False)

    _print_summary(result_df, output_dir, mode, copy_mode)
    return result_df


def _print_summary(df: pd.DataFrame, output_dir: Path, mode: OutputMode, copy_mode: CopyMode) -> None:
    if df.empty:
        print("Không có frame nào được chuẩn hoá.")
        return

    real_count = len(df[df["label_folder"] == "real"])
    fake_count = len(df[df["label_folder"] == "fake"])
    total_frames = df["num_frames"].sum()

    print(f"\n--- Standardization complete ---")
    print(f"  Mode       : {mode}")
    print(f"  Copy mode  : {copy_mode}")
    print(f"  Output dir : {output_dir}")
    print(f"  Videos real: {real_count}")
    print(f"  Videos fake: {fake_count}")
    print(f"  Total frames: {total_frames}")
    print(f"  Manifest   : {output_dir / 'standardized_manifest.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardize extracted frames into real/fake dataset structure")
    parser.add_argument("--manifest-csv", type=Path, default=Path("artifacts/frames_5fps/frame_manifest.csv"))
    parser.add_argument("--frames-root", type=Path, default=Path("artifacts/frames_5fps"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_standardized"))
    parser.add_argument("--mode", type=str, default="flat", choices=["flat", "by_method"])
    parser.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "symlink"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    standardize_dataset(
        manifest_csv=args.manifest_csv,
        frames_root=args.frames_root,
        output_dir=args.output_dir,
        mode=OutputMode(args.mode),
        copy_mode=CopyMode(args.copy_mode),
    )


if __name__ == "__main__":
    main()
