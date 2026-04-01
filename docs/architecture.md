# Architecture

## Goal

The repository transforms raw video files into model-ready frame and clip shards for deepfake detection experiments.

## High-Level Flow

```text
Raw videos
  -> video metadata manifest
  -> extracted frames
  -> aligned face frame shards
  -> clip-level shards
```

## Main Components

### `configs/`

- [settings.py](../configs/settings.py)
  Centralizes default paths and runtime settings.
- `loggings.py`
  Shared logging setup for preprocessing stages.

### `preprocessing/`

- [metadata_level.py](../preprocessing/metadata_level.py)
  Builds `videos_master.csv` from the dataset directory and assigns deterministic splits.
- [frame_extractor.py](../preprocessing/frame_extractor.py)
  Extracts frames and emits frame-level metadata.
- [face_detection.py](../preprocessing/face_detection.py)
  Thin entrypoint for the face detection pipeline.
- [_face_detection_pipeline.py](../preprocessing/_face_detection_pipeline.py)
  Main face detection, alignment, crop, resume, and shard-writing logic.
- [build_clips.py](../preprocessing/build_clips.py)
  Builds fixed-length clips from aligned frame shards.

### `tests/`

Stage-level tests for metadata helpers, split routing, crop geometry, and shard behavior.

## Data Boundaries

### Video-Level Manifest

Produced by `metadata_level.py`.

Purpose:
- enumerate videos
- record category and label
- assign `train/val/test`
- provide stable input to frame extraction

### Frame-Level Manifest

Produced by `frame_extractor.py`.

Purpose:
- record extracted frame paths
- preserve original frame indices and timestamps
- propagate split information downstream

### Frame Shards

Produced by `face_detection.py`.

Purpose:
- store aligned face crops and metadata in WebDataset shards
- support resume via existing shards and audit CSV
- isolate outputs by split when `--split` is used

### Clip Shards

Produced by `build_clips.py`.

Purpose:
- aggregate aligned frames into fixed-length clips
- store RGB clips and frame-difference clips as `.npy`

## Split Isolation

The pipeline is designed to run `train`, `val`, and `test` independently after split assignment.

Current split-specific behavior:

- frame metadata includes `split`
- face detection can filter with `--split`
- face shards are written to `crop_data/<split>/`
- face detection audit files are written to split-specific audit paths
- clip builder can process one split or discover all split directories automatically

## Face Detection Design

The face detection stage currently uses:

- RetinaFace for detection
- main-face selection using size and temporal continuity
- 5-point alignment
- alignment to a larger square canvas
- bounding-box crop in aligned coordinates
- final resize to the requested output size

This separation between align canvas and final output helps preserve context before the final crop is resized.
