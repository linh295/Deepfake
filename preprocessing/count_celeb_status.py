from __future__ import annotations

import argparse
import csv
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_split(value: Any) -> str:
    return str(value or "").strip().lower()


def video_key(row: dict[str, str]) -> str:
    category = str(row.get("category", "")).strip()
    video_name = str(row.get("video_name") or row.get("video_id") or "").strip()
    return f"{category}/{video_name}"


def sample_key(row: dict[str, str]) -> str:
    category = str(row.get("category", "")).strip() or "unknown"
    video_name = str(row.get("video_name") or row.get("video_id") or "unknownvideo").strip()
    frame_path = str(row.get("frame_path", "")).strip()
    if frame_path:
        stem = Path(frame_path).stem
    else:
        frame_number = str(
            row.get("frame_number")
            or row.get("frame_index")
            or row.get("frame_idx")
            or row.get("original_frame_index")
            or "0"
        ).strip()
        stem = f"{video_name}_{frame_number}"
    return f"{category}/{video_name}/{stem}".replace("\\", "/")


def base_key_from_tar_member(member_name: str) -> str | None:
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


def load_detected_keys_from_audit(audit_csv: Path | None) -> set[str]:
    if audit_csv is None or not audit_csv.exists():
        return set()
    rows = read_csv_rows(audit_csv)
    return {str(row.get("key", "")).strip() for row in rows if str(row.get("key", "")).strip()}


def load_detected_keys_from_shards(shard_dir: Path | None) -> set[str]:
    if shard_dir is None or not shard_dir.exists():
        return set()

    keys: set[str] = set()
    for shard_path in sorted(shard_dir.glob("shard-*.tar")):
        try:
            with tarfile.open(shard_path, "r") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    key = base_key_from_tar_member(member.name)
                    if key:
                        keys.add(key)
        except Exception as exc:
            print(f"Warning: cannot scan shard {shard_path}: {type(exc).__name__}: {exc}")
    return keys


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    for key in sorted(counter):
        print(f"  {key}: {counter[key]:,}")


def write_missing_videos(path: Path | None, missing_video_keys: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_key"])
        writer.writeheader()
        for key in missing_video_keys:
            writer.writerow({"video_key": key})
    print(f"Missing video list saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count extracted frames and face-detection progress for Celeb_DC."
    )
    parser.add_argument(
        "--frame-metadata",
        type=Path,
        default=Path("frame_data_celeb_dc/frame_extraction_metadata.csv"),
        help="Frame extraction metadata CSV.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path("frame_data_celeb_dc"),
        help="Root directory containing extracted frame images.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/celeb_dc_videos_master.csv"),
        help="Video-level manifest CSV.",
    )
    parser.add_argument(
        "--face-audit",
        type=Path,
        default=None,
        help="Optional face detection audit CSV. Example: audit/test/test/celeb_dc_face_detection_audit.csv",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("crop_data_celeb_dc/test"),
        help="Face detection shard directory to scan.",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to inspect, or empty string for all.")
    parser.add_argument(
        "--missing-videos-csv",
        type=Path,
        default=Path("artifacts/celeb_dc_missing_face_detection_videos.csv"),
        help="Where to write videos that have extracted frames but no detected face samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_filter = normalize_split(args.split)

    manifest_rows = read_csv_rows(args.manifest)
    frame_rows = read_csv_rows(args.frame_metadata)
    if split_filter:
        manifest_rows = [row for row in manifest_rows if normalize_split(row.get("split")) == split_filter]
        frame_rows = [row for row in frame_rows if normalize_split(row.get("split")) == split_filter]

    manifest_video_keys = {video_key(row) for row in manifest_rows}
    frame_video_keys = {video_key(row) for row in frame_rows}
    frame_sample_keys = {sample_key(row) for row in frame_rows}

    category_video_counts = Counter(str(row.get("category", "")).strip() for row in manifest_rows)
    category_frame_counts = Counter(str(row.get("category", "")).strip() for row in frame_rows)

    frame_rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        frame_rows_by_video[video_key(row)].append(row)

    missing_frame_files = 0
    for row in frame_rows:
        frame_path = args.frame_root / str(row.get("frame_path", "")).strip()
        if not frame_path.exists():
            missing_frame_files += 1

    audit_keys = load_detected_keys_from_audit(args.face_audit)
    shard_keys = load_detected_keys_from_shards(args.shard_dir)
    detected_keys = (audit_keys | shard_keys) & frame_sample_keys
    detected_video_keys = {"/".join(key.split("/")[:2]) for key in detected_keys}

    fully_detected_video_keys: set[str] = set()
    partially_detected_video_keys: set[str] = set()
    for key, rows in frame_rows_by_video.items():
        row_keys = {sample_key(row) for row in rows}
        detected_for_video = row_keys & detected_keys
        if len(detected_for_video) == len(row_keys) and row_keys:
            fully_detected_video_keys.add(key)
        elif detected_for_video:
            partially_detected_video_keys.add(key)

    missing_extracted_video_keys = sorted(manifest_video_keys - frame_video_keys)
    no_detected_sample_video_keys = sorted(frame_video_keys - detected_video_keys)

    detected_frame_category_counts: Counter[str] = Counter()
    for row in frame_rows:
        if sample_key(row) in detected_keys:
            detected_frame_category_counts[str(row.get("category", "")).strip()] += 1

    print("=== Celeb_DC Status ===")
    print(f"Split: {split_filter or 'all'}")
    print(f"Manifest videos: {len(manifest_video_keys):,}")
    print(f"Videos with extracted frames: {len(frame_video_keys):,}")
    print(f"Extracted frame rows: {len(frame_rows):,}")
    print(f"Missing frame files on disk: {missing_frame_files:,}")
    print()
    print(f"Audit detected sample keys: {len(audit_keys):,}" if args.face_audit else "Audit detected sample keys: not provided")
    print(f"Shard detected sample keys: {len(shard_keys):,}" if args.shard_dir else "Shard detected sample keys: not provided")
    print(f"Detected frame samples matched to metadata: {len(detected_keys):,}")
    print(f"Videos with >=1 detected face sample: {len(detected_video_keys):,}")
    print(f"Videos fully detected for every extracted frame: {len(fully_detected_video_keys):,}")
    print(f"Videos partially detected: {len(partially_detected_video_keys):,}")
    print(f"Videos with extracted frames but 0 detected samples: {len(no_detected_sample_video_keys):,}")
    print(f"Manifest videos without extracted frames: {len(missing_extracted_video_keys):,}")
    print()
    print_counter("Manifest videos by category:", category_video_counts)
    print_counter("Extracted frames by category:", category_frame_counts)
    print_counter("Detected frame samples by category:", detected_frame_category_counts)
    print()

    if no_detected_sample_video_keys:
        print("First missing face-detection videos:")
        for key in no_detected_sample_video_keys[:20]:
            print(f"  {key}")
        write_missing_videos(args.missing_videos_csv, no_detected_sample_video_keys)


if __name__ == "__main__":
    main()
