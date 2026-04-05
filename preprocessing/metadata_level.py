from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm


CATEGORY_TO_BINARY_LABEL: Dict[str, int] = {
    "original": 0,
    "DeepFakeDetection": 1,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceShifter": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUTPUT_COLUMNS = [
    "video_id",
    "video_path",
    "category",
    "binary_label",
    "compression",
    "split",
    "original_fps",
    "num_frames",
]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build video-level metadata for FaceForensics++ C23.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root_dir / "FaceForensics++_C23",
        help="Path to FaceForensics++_C23 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "artifacts",
        help="Directory to store generated metadata files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="videos_master",
        help="Base name for output files without extension.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) - 1)),
        help="Number of worker threads used to read video metadata.",
    )
    return parser.parse_args()


def iter_video_files(dataset_dir: Path) -> Iterable[Tuple[str, Path]]:
    for category_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        if category == "csv" or category not in CATEGORY_TO_BINARY_LABEL:
            continue

        for video_path in sorted(category_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                yield category, video_path


def probe_video_metadata(video_path: Path) -> Tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    capture.release()

    return round(fps, 6), num_frames


def build_group_key(category: str, video_id: str) -> str:
    if category == "DeepFakeDetection":
        return video_id.split("__", 1)[0]

    if category == "original":
        return video_id

    if "_" in video_id:
        return video_id

    return f"{category}:{video_id}"


def stable_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def assign_splits_for_category(records: List[Dict[str, object]]) -> Dict[str, str]:
    group_weights: Dict[str, int] = {}
    for record in records:
        group_key = str(record["_split_group"])
        group_weights[group_key] = group_weights.get(group_key, 0) + 1

    ordered_groups = sorted(
        group_weights.items(),
        key=lambda item: (-item[1], stable_hash(item[0])),
    )

    total_rows = sum(group_weights.values())
    targets = {
        "train": total_rows * 0.8,
        "val": total_rows * 0.1,
        "test": total_rows * 0.1,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    split_by_group: Dict[str, str] = {}

    for group_key, weight in ordered_groups:
        split = max(
            ("train", "val", "test"),
            key=lambda name: (targets[name] - counts[name], name == "train", name == "val"),
        )
        split_by_group[group_key] = split
        counts[split] += weight

    return split_by_group


def assign_splits(records: List[Dict[str, object]]) -> Dict[str, str]:
    split_by_group: Dict[str, str] = {}
    records_by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for record in records:
        records_by_category[str(record["category"])].append(record)

    for category_records in records_by_category.values():
        split_by_group.update(assign_splits_for_category(category_records))

    return split_by_group


def collect_records(dataset_dir: Path, workers: int) -> List[Dict[str, object]]:
    tasks = list(iter_video_files(dataset_dir))
    if not tasks:
        raise FileNotFoundError(f"No video files found under: {dataset_dir}")

    records: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_item = {
            executor.submit(probe_video_metadata, video_path): (category, video_path)
            for category, video_path in tasks
        }

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Reading videos"):
            category, video_path = future_to_item[future]
            fps, num_frames = future.result()

            video_id = video_path.stem
            relative_path = video_path.relative_to(dataset_dir).as_posix()
            split_group = build_group_key(category, video_id)

            records.append(
                {
                    "video_id": video_id,
                    "video_path": relative_path,
                    "category": category,
                    "binary_label": CATEGORY_TO_BINARY_LABEL[category],
                    "compression": "c23",
                    "original_fps": fps,
                    "num_frames": num_frames,
                    "_split_group": split_group,
                }
            )

    split_by_group = assign_splits(records)
    for record in records:
        record["split"] = split_by_group[str(record["_split_group"])]
        record.pop("_split_group", None)

    records.sort(key=lambda row: (str(row["category"]), str(row["video_id"])))
    return records


def write_outputs(records: List[Dict[str, object]], output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    csv_path = output_dir / f"{output_name}.csv"
    dataframe.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    parquet_path = output_dir / f"{output_name}.parquet"
    try:
        dataframe.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as exc:
        print(f"Skipped Parquet ({type(exc).__name__}): {exc}")

    split_counts = dataframe["split"].value_counts().to_dict()
    category_counts = dataframe["category"].value_counts().sort_index().to_dict()
    print(f"Total videos: {len(dataframe)}")
    print(f"Split counts: {split_counts}")
    print(f"Category counts: {category_counts}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    records = collect_records(dataset_dir=dataset_dir, workers=args.workers)
    write_outputs(records=records, output_dir=output_dir, output_name=args.output_name)


if __name__ == "__main__":
    main()
