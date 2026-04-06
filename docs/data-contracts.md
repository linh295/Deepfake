# Data Contract

## 1. `videos_master.csv`

Được sinh bởi:
- [metadata_level.py](../preprocessing/metadata_level.py)

Cột quan trọng:

- `video_id`
- `video_path`
- `category`
- `binary_label`
- `compression`
- `split`
- `original_fps`
- `num_frames`

Ghi chú:

- `split` được gán ở cấp video và nên được xem là nguồn sự thật cho việc định tuyến split ở các stage sau.
- việc gán split nhằm mục tiêu tỷ lệ `8:1:1` trong từng category, trong giới hạn của ràng buộc nhóm nguyên.
- `binary_label` dùng `0` cho real và `1` cho fake.

## 2. `frame_extraction_metadata.csv`

Được sinh bởi:
- [frame_extractor.py](../preprocessing/frame_extractor.py)

Cột quan trọng:

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

Ghi chú:

- `frame_path` là đường dẫn tương đối so với thư mục frame root.
- `split` phải được giữ lại vì các stage sau có thể lọc theo trường này.

## 3. Face Frame Shard

Được sinh bởi:
- [face_detection.py](../preprocessing/face_detection.py)

Thành phần thường có trong mỗi sample shard:

- `json`
- `jpg` hoặc `png`
- `cls` khi nhãn đã biết

Trường metadata quan trọng trong `json`:

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

Ghi chú:

- `bbox_*` là bounding box detect được trong hệ tọa độ của ảnh gốc.
- `crop_*` là cửa sổ crop trong hệ tọa độ của align-canvas.
- `aligned_width` và `aligned_height` là kích thước output cuối sau resize.
- `align_canvas_*` mô tả canvas lớn hơn được dùng trước khi crop cuối cùng bị resize.

## 4. Clip Shard

Được sinh bởi:
- [build_clips.py](../preprocessing/build_clips.py)

Thành phần thường có trong mỗi sample shard:

- `json`
- `rgb.npy`
- `diff.npy`
- `cls` khi nhãn đã biết

Trường metadata quan trọng trong `json`:

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

Ghi chú:

- `rgb.npy` có shape `(T, C, H, W)`.
- `diff.npy` lưu frame difference giữa các frame liên tiếp.
- `source_keys` có thể được dùng để truy vết clip về sample cấp frame.
- `key` của clip được suy ra từ `video_id` canonical của frame shard, không phải `video_name`.

## 5. Hợp Đồng Dữ Liệu Của Batch Từ Dataloader

Được sinh bởi:
- [dataloader/dataset.py](../dataloader/dataset.py)

Key trong batch:

- `spatial`
- `temporal`
- `label`
- `spatial_index`
- `meta`

Tensor shape kỳ vọng:

- `spatial`: `(B, 3, H, W)`
- `temporal`: `(B, T-1, 3, H, W)`
- `label`: `(B,)`
- `spatial_index`: `(B,)`

Ghi chú:

- `spatial` là một frame RGB được chọn từ `rgb.npy`.
- `temporal` đến từ `diff.npy`, vì vậy chiều thời gian của nó phải bằng `clip_length - 1`.
- `spatial` được normalize theo ImageNet mean/std.
- `temporal` được scale về `[0, 1]` mà không dùng ImageNet normalization.
- `meta` giữ lại JSON metadata của từng sample để debug và truy vết.

## 6. Hợp Đồng Dữ Liệu Của Đầu Ra Huấn Luyện

Được sinh bởi:
- [training/train.py](../training/train.py)
- [training/utils/checkpointing.py](../training/utils/checkpointing.py)

### `history.json`

Key cấp cao nhất:

- `run`
- `train`
- `val`

Các trường thường có trong `run`:

- `class_balance`
- `use_pos_weight`
- `auto_pos_weight`

Các trường thường có trong từng dòng `train` và `val`:

- `epoch`
- `loss`
- `accuracy`
- `f1`
- `auc`

Trường bổ sung cho validation:

- `selection_metric`
- `selection_metric_name`
- `learning_rates`

### Checkpoint `.pt`

Key cấp cao nhất:

- `epoch`
- `model_state`
- `optimizer_state`
- `scheduler_state`
- `scaler_state`
- `best_val_auc`
- `best_selection_metric`
- `train_config`
- `model_config`
- `class_balance`
- `rng_state`

Ghi chú:

- `train_config` lưu `TrainConfig` sau khi được serialize.
- `model_config` lưu `ModelConfig` sau khi được serialize.
- `class_balance` là trường tùy chọn và chỉ có khi metadata class-balance được tính toán.
- `rng_state` được lưu để phục vụ khả năng tái lập và resume.
