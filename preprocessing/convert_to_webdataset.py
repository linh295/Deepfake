from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from webdataset.writer import ShardWriter


def convert_split_csv_to_webdataset(
    csv_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    frame_col: str = "frame_path",
    split_col: str = "split",
    shard_size: int = 5000,
) -> None:
    csv_path = Path(csv_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {frame_col, split_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Loaded {len(df):,} rows from {csv_path}")

    for split in ["train", "val", "test"]:
        split_df = df[df[split_col] == split].reset_index(drop=True)

        if split_df.empty:
            print(f"Skip empty split: {split}")
            continue

        split_out_dir = output_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(split_out_dir / "shard-%06d.tar")
        sink = ShardWriter(pattern, maxcount=shard_size)

        written = 0
        skipped = 0

        for idx, row in tqdm(
            split_df.iterrows(),
            total=len(split_df),
            desc=f"Writing {split}",
        ):
            rel_path = str(row[frame_col]).replace("\\", "/").strip()
            img_path = dataset_root / rel_path

            if not img_path.exists():
                skipped += 1
                continue

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                meta = row.to_dict()
                meta[frame_col] = rel_path

                sample = {
                    "__key__": f"{idx:09d}",
                    "jpg": img_bytes,
                    "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                }
                sink.write(sample)
                written += 1

            except Exception as e:
                skipped += 1
                print(f"[WARN] Skip {img_path}: {e}")

        sink.close()
        print(f"[{split}] written={written:,}, skipped={skipped:,}")


if __name__ == "__main__":
    convert_split_csv_to_webdataset(
        csv_path="artifacts/split_metadata.csv",
        dataset_root="frame_data",
        output_dir="artifacts/wds",
        frame_col="frame_path",
        split_col="split",
        shard_size=5000,
    )