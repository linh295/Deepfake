# Data Contracts

## 1. `videos_master.csv`

Produced by:
- [metadata_level.py](../preprocessing/metadata_level.py)

Key columns:

- `video_id`
- `video_path`
- `category`
- `binary_label`
- `compression`
- `split`
- `original_fps`
- `num_frames`

Notes:

- `split` is assigned at the video level and should be treated as the source of truth for downstream split routing.
- `binary_label` uses `0` for real and `1` for fake.

## 2. `frame_extraction_metadata.csv`

Produced by:
- [frame_extractor.py](../preprocessing/frame_extractor.py)

Key columns:

- `frame_path`
- `video_id`
- `video_path`
- `video_name`
- `category`
- `label`
- `binary_label`
- `split`
- `frame_number`
- `original_frame_index`
- `timestamp`
- `video_fps`
- `extraction_fps`
- `width`
- `height`
- `video_duration`
- `total_video_frames`
- `extraction_date`

Notes:

- `frame_path` is relative to the frame root directory.
- `split` must remain available because downstream stages can filter on it.

## 3. Face Frame Shards

Produced by:
- [face_detection.py](../preprocessing/face_detection.py)

Typical shard parts per sample:

- `json`
- `jpg` or `png`
- `cls` when label is known

Important metadata fields inside `json`:

- `key`
- `video_id`
- `frame_path`
- `category`
- `video_name`
- `label`
- `binary_label`
- `split`
- `frame_number`
- `original_frame_index`
- `timestamp`
- `face_detected`
- `num_faces`
- `face_confidence`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `bbox_source`
- `landmarks`
- `alignment_mode`
- `crop_scale`
- `align_status`
- `crop_status`
- `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`
- `crop_size`
- `crop_source`
- `aligned_width`, `aligned_height`
- `aligned_size`
- `align_canvas_width`, `align_canvas_height`
- `align_canvas_size`
- `affine_matrix`
- `image_format`
- `detect_every_k`

Notes:

- `bbox_*` refers to the detected bounding box in the original image coordinates.
- `crop_*` refers to the crop window in aligned-canvas coordinates.
- `aligned_width` and `aligned_height` are the final output dimensions after resize.
- `align_canvas_*` describes the larger canvas used before the final crop is resized.

## 4. Clip Shards

Produced by:
- [build_clips.py](../preprocessing/build_clips.py)

Typical shard parts per sample:

- `json`
- `rgb.npy`
- `diff.npy`
- `cls` when label is known

Important metadata fields inside `json`:

- `key`
- `split`
- `category`
- `video_id`
- `video_name`
- `binary_label`
- `label`
- `clip_length`
- `frame_stride`
- `clip_stride`
- `num_differences`
- `frame_numbers`
- `original_frame_indices`
- `timestamps`
- `source_keys`
- `center_candidate_indices`
- `default_center_index`
- `extraction_fps`
- `video_fps`
- `height`
- `width`
- `rgb_dtype`
- `diff_dtype`

Notes:

- `rgb.npy` stores shape `(T, C, H, W)`.
- `diff.npy` stores frame differences across consecutive frames.
- `source_keys` can be used to trace the clip back to frame-level shard samples.
