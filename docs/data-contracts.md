# Data Contract

Tai lieu nay mo ta cac file trung gian va output ma pipeline dang san sinh.

## 1. `videos_master.csv`

Sinh boi:

- [metadata_level.py](../preprocessing/metadata_level.py)

Cot:

- `video_id`
- `video_path`
- `category`
- `binary_label`
- `compression`
- `split`
- `original_fps`
- `num_frames`

Ghi chu:

- `video_path` la duong dan tuong doi so voi dataset root.
- `split` la nguon su that cho dinh tuyen `train/val/test`.
- Split duoc gan xap xi `80/10/10` trong tung category bang grouping on dinh.
- Quy uoc label hien tai: `original=1`, manipulation `=0`.
- Neu can quy uoc nguoc lai o training/evaluation, dung `--invert-binary-labels`.

## 2. `frame_extraction_metadata.csv`

Sinh boi:

- [frame_extractor.py](../preprocessing/frame_extractor.py)

Cot:

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

Ghi chu:

- `frame_path` tuong doi so voi `frame_data`.
- `frame_number` la index cua frame sau khi sample theo target FPS.
- `original_frame_index` la index frame trong video goc.
- `timestamp` tinh theo video FPS goc.
- `split` va `binary_label` duoc truyen tu `videos_master.csv`.

## 3. `frame_extraction_audit.csv`

Sinh boi:

- [frame_extractor.py](../preprocessing/frame_extractor.py)

Cot:

- `category`
- `video_id`
- `split`
- `status`
- `updated_at`

Ghi chu:

- Audit dung de resume theo video.
- Key logic la `category|video_id|split`.

## 4. Face Frame Shard

Sinh boi:

- [face_detection.py](../preprocessing/face_detection.py)

Duong dan:

- `crop_data/<split>/shard-*.tar` khi co `--split`
- `crop_data/shard-*.tar` khi khong tach split

Thanh phan sample:

- `json`
- `jpg` hoac `png`
- `cls` neu label hop le

Truong metadata quan trong trong `json`:

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
- `video_fps`
- `extraction_fps`
- `width`
- `height`
- `face_detected`
- `num_faces`
- `face_confidence`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `bbox_source`
- `landmarks`
- `alignment_mode`
- `crop_scale`
- `crop_status`
- `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`
- `crop_size`
- `crop_source`
- `aligned_width`, `aligned_height`
- `aligned_size`
- `image_format`
- `detect_every_k`

Ghi chu:

- `bbox_*` nam trong toa do anh goc.
- `crop_*` nam trong toa do anh goc.
- `aligned_width/height` la kich thuoc output sau resize.
- `alignment_mode` la `none_bbox_crop`; landmark RetinaFace chi duoc luu metadata, khong bien doi pixel.
- `bbox_source` co the la direct/keyframe/interpolated tuy detection path.
- Audit face detection ghi metadata cua sample da ghi thanh cong.

## 5. Clip Shard

Sinh boi:

- [build_clips.py](../preprocessing/build_clips.py)

Duong dan:

- `clip_data/<split>/shard-*.tar`

Thanh phan sample:

- `json`
- `rgb.npy`
- `diff.npy`
- `cls` neu label hop le

Truong metadata trong `json`:

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
- `clip_start`
- `num_differences`
- `frame_numbers`
- `original_frame_indices`
- `timestamps`
- `source_keys`
- `center_candidate_indices`
- `default_center_index`
- `spatial_sampling_note`
- `extraction_fps`
- `video_fps`
- `height`
- `width`
- `rgb_dtype`
- `diff_dtype`

Array:

- `rgb.npy`: shape `(T, C, H, W)`, dtype `uint8`.
- `diff.npy`: shape `(T-1, C, H, W)`, dtype `uint8`.

Ghi chu:

- `diff.npy` la absolute difference giua frame lien tiep.
- `key` duoc tao tu canonical `video_id` va `clip_id`.
- `center_candidate_indices` mac dinh `[2, 3, 4, 5]` khi `clip_len=8`.
- Truong text `label` trong clip duoc suy theo convention cu `0=real/1=fake`; voi manifest hien tai no co the khong khop semantic category. Doc `binary_label` la nguon su that va dung `--invert-binary-labels` neu can dao quy uoc.

## 6. Dataloader Batch

Sinh boi:

- [dataloader/dataset.py](../dataloader/dataset.py)

Key:

- `spatial`
- `temporal`
- `label`
- `spatial_index`
- `meta`

Shape:

- `spatial`: `(B, 3, H, W)`
- `temporal`: `(B, T-1, 3, H, W)`
- `label`: `(B,)`
- `spatial_index`: `(B,)`

Normalize:

- `spatial`: RGB scale `[0, 1]`, sau do ImageNet mean/std.
- `temporal`: frame difference scale `[0, 1]`.

Selection:

- Train: random trong `center_candidate_indices` neu co.
- Val/test: `default_center_index` neu hop le, fallback ve center candidate.

Augmentation:

- Chi ap dung khi `training=True` va `use_augmentation=True`.
- Hflip/brightness/contrast/blur/JPEG dung cung param cho toan clip.
- Neu `augment_recompute_diff=True`, `temporal` duoc tinh lai tu RGB sau augmentation.

## 7. `history.json`

Sinh boi:

- [training/train.py](../training/train.py)

Key cap cao:

- `run`
- `train`
- `val`

`run` gom:

- `class_balance`
- `use_pos_weight`
- `auto_pos_weight`
- `model_dropout`
- `temporal_pool`
- `use_spatial_attention`
- `use_texture_enhancement`

Moi row trong `train` thuong gom:

- `epoch`
- `loss`
- `accuracy`
- `f1`
- `auc`

Moi row trong `val` gom cac metric tren va:

- `selection_metric`
- `selection_metric_name`
- `learning_rates`

## 8. Checkpoint `.pt`

Sinh boi:

- [training/utils/checkpointing.py](../training/utils/checkpointing.py)

Key:

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

Ghi chu:

- `model_config` duoc `training.test` va `training.test_with_best_threshold` dung de rebuild model.
- Checkpoint nen duoc load bang helper `load_checkpoint` trong `training/utils/checkpointing.py`.

## 9. Test Output

Sinh boi:

- [training/test.py](../training/test.py)
- [training/test_with_best_threshold.py](../training/test_with_best_threshold.py)

File:

- `test_metrics.json`
- `test_predictions.csv`

`test_predictions.csv` cot:

- `key`
- `video_id`
- `category`
- `label_name`
- `label`
- `prob_positive`
- `pred`
- `spatial_index`

`training.test` ghi `test_metrics.json` voi:

- checkpoint metadata
- threshold
- AUC
- accuracy
- balanced accuracy
- F1
- precision
- recall
- confusion matrix
- label distribution
- prediction distribution
- probability summary

`training.test_with_best_threshold` bo sung:

- `manual_threshold`
- `best_thresholds`
- `selected_threshold_mode`
- `selected_threshold`
- `selected_threshold_metrics`
