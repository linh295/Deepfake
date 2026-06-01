from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REAL_CATEGORIES = {"Celeb-real", "YouTube-real"}
SYNTHESIS_CATEGORY = "Celeb-synthesis"


def video_key(row: dict[str, str]) -> str:
    category = str(row.get("category", "")).strip()
    video_name = str(row.get("video_name") or row.get("video_id") or "").strip()
    return f"{category}/{video_name}"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Celeb_DC frame metadata subset for visualization/testing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("frame_data_celeb_dc/frame_extraction_metadata.csv"),
        help="Full frame extraction metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/celeb_dc_real_all_synthesis_1000_frame_metadata.csv"),
        help="Subset frame metadata CSV to write.",
    )
    parser.add_argument(
        "--synthesis-videos",
        type=int,
        default=1000,
        help="Number of distinct Celeb-synthesis videos to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)

    synthesis_video_keys = sorted(
        {
            video_key(row)
            for row in rows
            if str(row.get("category", "")).strip() == SYNTHESIS_CATEGORY
        }
    )[: args.synthesis_videos]
    selected_synthesis = set(synthesis_video_keys)

    selected_rows = [
        row
        for row in rows
        if str(row.get("category", "")).strip() in REAL_CATEGORIES
        or video_key(row) in selected_synthesis
    ]
    write_rows(args.output, fieldnames, selected_rows)

    category_frame_counts = Counter(str(row.get("category", "")).strip() for row in selected_rows)
    category_video_counts = Counter()
    for key in {video_key(row) for row in selected_rows}:
        category, _ = key.split("/", 1)
        category_video_counts[category] += 1

    print(f"Saved: {args.output}")
    print(f"Selected videos: {len({video_key(row) for row in selected_rows}):,}")
    print(f"Selected frames: {len(selected_rows):,}")
    print(f"Videos by category: {dict(sorted(category_video_counts.items()))}")
    print(f"Frames by category: {dict(sorted(category_frame_counts.items()))}")


if __name__ == "__main__":
    main()
