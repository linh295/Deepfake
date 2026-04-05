# Deepfake Preprocessing Pipeline

This repository builds a training-ready deepfake dataset from raw FaceForensics++ style videos.

The current pipeline is organized around four stages:

1. `preprocessing/metadata_level.py`
   Builds `videos_master.csv` with video-level metadata and deterministic `train/val/test` splits balanced within each category.
2. `preprocessing/frame_extractor.py`
   Extracts frames and writes `frame_extraction_metadata.csv`.
3. `preprocessing/face_detection.py`
   Detects the main face, aligns the image, crops around the aligned bounding box, and writes frame-level WebDataset shards.
4. `preprocessing/build_clips.py`
   Groups aligned frames into fixed-length clips and writes clip-level WebDataset shards.

## Repository Layout

```text
configs/         Runtime settings and logging
preprocessing/   Data preparation stages
tests/           Unit and integration tests for preprocessing
docs/            Architecture, pipeline, and data contracts
artifacts/       Generated CSV/parquet and temporary outputs
frame_data/      Extracted frames and frame metadata
crop_data/       Face-aligned frame shards
clip_data/       Clip-level shards
```

## Quickstart

Install dependencies:

```bash
pip install -e .
```

Generate video-level metadata:

```bash
python -m preprocessing.metadata_level --dataset-dir FaceForensics++_C23 --output-dir artifacts --output-name videos_master
```

Extract frames for a single split:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train
```

Resume is enabled by default for frame extraction. Use `--no-resume` to force a clean rerun.

Build aligned face shards for the same split:

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
```

Build clip shards from aligned frame shards:

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train
```

Repeat the frame, face, and clip stages for `val` and `test` when running the full dataset.

## Output Summary

- `artifacts/videos_master.csv`
  Video-level manifest with deterministic split assignment.
- `frame_data/frame_extraction_metadata.csv`
  Frame-level manifest produced by `frame_extractor.py`.
- `frame_data/frame_extraction_audit.csv`
  Resume audit for completed videos in `frame_extractor.py`.
- `crop_data/<split>/shard-*.tar`
  Frame-level WebDataset shards containing aligned face crops.
- `clip_data/<split>/shard-*.tar`
  Clip-level WebDataset shards containing `rgb.npy`, `diff.npy`, and metadata.

## Documentation

- [Pipeline Guide](docs/preprocessing-pipeline.md)
- [Architecture](docs/architecture.md)
- [Data Contracts](docs/data-contracts.md)
- [Contributing](CONTRIBUTING.md)

## Testing

Run the preprocessing test suite:

```bash
python -m unittest tests.test_face_detection tests.test_face_detection_split tests.test_face_detection_crop
```

Compile-check the main modules:

```bash
python -m compileall preprocessing tests
```
