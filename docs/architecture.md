# Kien Truc

## Muc Tieu

Du an cung cap pipeline end-to-end cho bai toan phat hien deepfake:

1. Quet video FaceForensics++ C23 va tao manifest co split on dinh.
2. Trich xuat frame o target FPS.
3. Detect/can chinh khuon mat va ghi frame WebDataset shard.
4. Gom frame thanh clip WebDataset shard.
5. Huan luyen va danh gia detector spatio-temporal.

## Luong Xu Ly

```text
Video goc
  -> artifacts/videos_master.csv
  -> frame_data/frame_extraction_metadata.csv + frame JPG
  -> crop_data/<split>/shard-*.tar
  -> clip_data/<split>/shard-*.tar
  -> ClipWebDataset/DataLoader
  -> SpatioTemporalDeepfakeDetector
  -> history.json + checkpoint + test report
```

## Thanh Phan Chinh

### `configs/`

- [settings.py](../configs/settings.py): duong dan mac dinh, FPS, JPEG quality, RetinaFace cache va category mapping.
- [loggings.py](../configs/loggings.py): logging dung chung.

`settings.RAW_DATA_DIR` mac dinh la `FaceForensics++_C23 2`; `frame_extractor` fallback sang `FaceForensics++_C23` neu duong dan nay khong ton tai. Khi chay that, nen truyen `--dataset-dir` hoac `--manifest` ro rang.

### `preprocessing/`

- [metadata_level.py](../preprocessing/metadata_level.py): tao manifest video, doc FPS/so frame, gan split.
- [analyze_videos_master.py](../preprocessing/analyze_videos_master.py): in bao cao phan bo split theo category.
- [frame_extractor.py](../preprocessing/frame_extractor.py): trich xuat frame, ghi metadata/audit va ho tro resume.
- [face_detection.py](../preprocessing/face_detection.py): CLI wrapper cho face pipeline.
- [_face_detection_pipeline.py](../preprocessing/_face_detection_pipeline.py): RetinaFace detection, keyframe interpolation, alignment, crop, shard writer va audit.
- [build_clips.py](../preprocessing/build_clips.py): gom frame shard thanh clip shard.
- [count_clips.py](../preprocessing/count_clips.py): dem so shard/clip trong `clip_data`.

### `dataloader/`

- [dataset.py](../dataloader/dataset.py): doc clip WebDataset shard, validate shape, chon spatial frame, normalize tensor, ap dung augmentation khi train va collate batch.

Batch output gom:

- `spatial`: mot frame RGB cho spatial branch
- `temporal`: chuoi frame difference cho temporal branch
- `label`: binary label sau tuy chon invert
- `spatial_index`: index frame RGB duoc chon
- `meta`: metadata JSON cua sample

### `training/`

- [train.py](../training/train.py): entrypoint huan luyen.
- [test.py](../training/test.py): evaluate voi threshold thu cong.
- [test_with_best_threshold.py](../training/test_with_best_threshold.py): evaluate va tinh threshold theo Youden, F1 hoac balanced accuracy.
- [spatio_temporal_detector.py](../training/spatio_temporal_detector.py): module detector cap cao.
- [spatial_resnet50.py](../training/spatial_resnet50.py): spatial branch ResNet50.
- [temporal_diff_cnn.py](../training/temporal_diff_cnn.py): temporal branch tren frame difference, ho tro `mean`, `attention`, `gru`.
- [fusion_head.py](../training/fusion_head.py): MLP fusion classifier.
- `training/utils/`: builder, train/val loop, metric, checkpoint, class balance, freezing, figure va runtime helper.

## Label Va Split

Manifest video la nguon su that cho split va label.

Quy uoc label hien tai trong [metadata_level.py](../preprocessing/metadata_level.py):

- `original = 1`
- `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures` = `0`

Mot so helper va truong text `label` van theo fallback cu `real=0/fake=1`. Voi manifest hien tai, truong text nay co the khong phan anh dung y nghia semantic cua category. Vi vay can xem `binary_label` trong `videos_master.csv` va shard la nguon su that. Training/test co `--invert-binary-labels` de dao label o dataloader khi can.

Split duoc gan o cap video va duoc truyen qua:

- `frame_extraction_metadata.csv`
- frame shard JSON
- clip shard JSON
- dataloader `meta`

## Thiet Ke Preprocessing

### Metadata Video

`metadata_level.py` quet cac category hop le, doc thong tin video bang OpenCV, tao `video_id`, `video_path`, `category`, `binary_label`, `compression`, `split`, `original_fps`, `num_frames`.

Split duoc gan xap xi `80/10/10` trong tung category bang grouping on dinh va hash, giup han che cung mot nhom lien quan bi roi vao nhieu split.

### Frame Extraction

`frame_extractor.py` doc manifest, trich xuat frame theo `settings.TARGET_FPS` mac dinh `5`, ghi JPG vao `frame_data/<category>/` va append metadata theo tung batch. Resume dua tren:

- `frame_extraction_audit.csv`
- metadata CSV da co
- scan frame da ton tai de bootstrap/recover khi can

### Face Detection

Face pipeline:

- detect bang RetinaFace voi nguong mac dinh `0.9`
- chon khuon mat chinh dua tren kich thuoc va lien tuc voi bbox truoc
- detect tren keyframe theo `detect_every_k`
- noi suy bbox/landmark giua keyframe co detection hop le
- can chinh bang 5 landmark len align canvas
- crop vuong quanh bbox da can chinh theo `crop_scale`
- resize ve `aligned_width x aligned_height`, mac dinh `224 x 224`
- ghi sample WebDataset gom `json`, anh `jpg/png`, va `cls` neu co label

Khi co `--split`, output va audit duoc tach thanh `crop_data/<split>/` va `audit/<split>/face_detection_audit.csv`.

### Clip Builder

`build_clips.py` doc frame shard theo split, group theo `(split, category, video_id)`, yeu cau frame cua mot video phai lien tuc trong shard order, sau do tao sliding-window clip.

Mac dinh:

- `clip_len = 8`
- `frame_stride = 1`
- `clip_stride = 4`
- `rgb.npy`: `(T, C, H, W)` uint8
- `diff.npy`: absolute difference giua frame lien tiep, `(T-1, C, H, W)` uint8

## Thiet Ke Mo Hinh

Detector co hai nhanh:

- spatial branch: `SpatialResNet50`
- temporal branch: `TemporalDiffCNN`
- fusion: `FusionHead`

Spatial branch:

- dung `torchvision.models.resnet50` voi `IMAGENET1K_V2` khi `pretrained=True`
- xuat feature 2048 chieu
- texture enhancement lay `layer1`, project len 2048 channel, resize ve kich thuoc semantic feature, concatenate va fuse bang `1x1 conv`
- spatial attention tao map mot channel va nhan voi feature truoc global average pooling

Temporal branch:

- input shape `[B, T-1, 3, H, W]`
- encode tung frame-difference bang CNN residual nhe
- neu `use_cross_branch_attention=True`, spatial attention map `[B, 1, Hs, Ws]` duoc resize va nhan vao temporal feature map cua tung frame truoc global average pooling
- project moi timestep ve `temporal_feature_dim`, CLI train mac dinh `256`
- neu `use_feature_delta=True`, concat `|x_t - x_{t-1}|` vao feature tung timestep truoc GRU
- pooling co the la `mean`, `attention`, hoac `gru`
- CLI train mac dinh `--temporal-pool gru`
- voi `gru`, dung bidirectional GRU final hidden pooling roi project ve lai `feature_dim`
- cac bien the ablation `gru_mean`, `gru_max`, `gru_mean_max`, `gru_attn` pooling tren toan bo `gru_out`

Fusion head:

- concatenate spatial feature va temporal feature
- MLP voi dropout
- xuat mot logit cho binary classification

Khi `return_features=True`, detector tra them `spatial_feat`, `temporal_feat`, attention map/feature map spatial va `temporal_attn` neu temporal branch cung cap.

## Training Va Checkpoint

Training loop:

- `AdamW` voi learning rate rieng cho spatial, temporal va fusion branch
- `BCEWithLogitsLoss` mac dinh, hoac focal loss
- auto `pos_weight` tu train shard neu bat
- AMP tren CUDA theo mac dinh
- gradient clipping
- alternate freezing: freeze spatial trong `spatial_freeze_warmup_epochs` dau, freeze temporal trong `temporal_freeze_epochs` tiep theo, sau do train full model
- `ReduceLROnPlateau(mode="max")` theo selection metric
- early stopping theo selection metric

Selection metric:

- uu tien validation AUC neu tinh duoc
- fallback sang negative validation loss neu validation chi co mot lop

Checkpoint luu:

- model/optimizer/scheduler/scaler state
- `TrainConfig`, `ModelConfig`
- class-balance metadata
- RNG state
- best AUC va best selection metric

## Danh Gia

`training.test` load `best.pt`, rebuild model tu `model_config`, chay test shard va ghi:

- `test_metrics.json`
- `test_predictions.csv`

`training.test_with_best_threshold` bo sung threshold search:

- Youden index
- F1
- balanced accuracy
- threshold thu cong tu `--threshold`

Threshold duoc chon cho predictions bang `--prediction-threshold-mode`.
