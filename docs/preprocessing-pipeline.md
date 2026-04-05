# Preprocessing Pipeline

## Overview

The recommended execution order is:

1. Build `videos_master.csv`
2. Extract frames
3. Detect faces and write aligned frame shards
4. Build clips from aligned frame shards

## 1. Build Video Metadata

Script:
- [metadata_level.py](../preprocessing/metadata_level.py)

Purpose:
- scan the dataset directory
- collect FPS and frame counts
- assign deterministic `train/val/test` splits balanced within each category

Example:

```bash
python -m preprocessing.metadata_level --dataset-dir FaceForensics++_C23 --output-dir artifacts --output-name videos_master
```

Main output:
- `artifacts/videos_master.csv`

## 2. Extract Frames

Script:
- [frame_extractor.py](../preprocessing/frame_extractor.py)

Purpose:
- read `videos_master.csv`
- extract frames at a target FPS
- keep split information on every frame row
- resume by default using existing frames plus `frame_extraction_audit.csv`

Examples:

Extract only training data:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train
```

Force a clean rerun without resume:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train --no-resume
```

Extract one category only:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --category original
```

Main outputs:
- `frame_data/<category>/<video_name>/*.jpg`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`

## 3. Detect Faces and Build Frame Shards

Script:
- [face_detection.py](../preprocessing/face_detection.py)

Purpose:
- detect the main face per frame
- align the image using 5 landmarks
- crop around the aligned bounding box with configurable context
- write frame-level WebDataset shards

Examples:

Run one split:

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
```

Run three independent jobs:

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split val
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split test
```

Important behavior:

- output is split-specific when `--split` is provided
- audit CSV is also split-specific
- face crops are written in streaming mode, not buffered per video
- alignment runs on a larger square canvas, then the final crop is resized to `--aligned-width/--aligned-height`

Main outputs:

- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`

## 4. Build Clip Shards

Script:
- [build_clips.py](../preprocessing/build_clips.py)

Purpose:
- read aligned frame shards
- build fixed-length clips
- store RGB clips and frame-difference clips
- use canonical frame-shard `video_id` for clip grouping and keys

Example:

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train
```

If the target split output already has clip shards, rerun with `--overwrite` to rebuild it cleanly:

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train --overwrite
```

Main outputs:
- `clip_data/<split>/shard-*.tar`

Important behavior:

- clip grouping/keying uses canonical frame-shard `video_id`
- reruns fail fast on existing split shards unless `--overwrite` is provided

## Recommended Run Pattern

For full preprocessing, run by split after `videos_master.csv` exists:

1. `frame_extractor.py --split train`
2. `face_detection.py --split train`
3. `build_clips.py --split train`
4. Repeat for `val`
5. Repeat for `test`

This keeps outputs isolated and makes parallel jobs simpler to operate.
