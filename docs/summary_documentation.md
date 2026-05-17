# Tong Hop Tai Lieu Pipeline

## Muc Tieu

Du an bien doi video FaceForensics++ C23 thanh clip WebDataset va huan luyen detector deepfake spatio-temporal. Tai lieu chi tiet nam o:

- [Kien Truc](architecture.md)
- [Pipeline Tien Xu Ly](preprocessing-pipeline.md)
- [Quy Trinh Huan Luyen Va Danh Gia](training-workflow.md)
- [Data Contract](data-contracts.md)
- [Thay Doi Tai Lieu](documentation-changes.md)

## Luong Tong The

```text
FaceForensics++ video
  -> videos_master.csv
  -> frame_data/*.jpg + frame_extraction_metadata.csv
  -> crop_data/<split>/shard-*.tar
  -> clip_data/<split>/shard-*.tar
  -> training.train
  -> training.test / training.test_with_best_threshold
```

## Quickstart

### 1. Tao manifest

```bash
python -m preprocessing.metadata_level \
  --dataset-dir FaceForensics++_C23 \
  --output-dir artifacts \
  --output-name videos_master
```

Kiem tra split:

```bash
python -m preprocessing.analyze_videos_master \
  --manifest artifacts/videos_master.csv
```

### 2. Tien xu ly tung split

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --split train

python -m preprocessing.face_detection \
  --metadata-csv frame_data/frame_extraction_metadata.csv \
  --frame-root frame_data \
  --output-dir crop_data \
  --split train

python -m preprocessing.build_clips \
  --input-dir crop_data \
  --output-dir clip_data \
  --split train
```

Lap lai cho `val` va `test`.

Dem clip:

```bash
python -m preprocessing.count_clips \
  --clip-root clip_data \
  --splits train val test
```

### 3. Huan luyen

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --output-dir artifacts/experiments/st_detector
```

Vi du voi augmentation va focal loss:

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --use-augmentation \
  --jpeg-prob 0.2 \
  --blur-prob 0.1 \
  --loss-type focal
```

### 4. Danh gia

```bash
python -m training.test \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --output-dir artifacts/test_results
```

Hoac tim threshold theo F1:

```bash
python -m training.test_with_best_threshold \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --prediction-threshold-mode f1
```

## Output Chinh

Tien xu ly:

- `artifacts/videos_master.csv`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`
- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`
- `clip_data/<split>/shard-*.tar`

Training:

- `artifacts/experiments/st_detector/history.json`
- `artifacts/experiments/st_detector/best.pt`
- `artifacts/experiments/st_detector/checkpoint_epoch_*.pt`
- `figures/<run>/latest/`
- `figures/<run>/best/epoch_*/`

Test:

- `artifacts/test_results/test_metrics.json`
- `artifacts/test_results/test_predictions.csv`

## Diem Can Chu Y

Label:

- `metadata_level.py` hien gan `original=1`, manipulation `=0`.
- Truong text `label` o cac stage sau co the theo convention cu `0=real/1=fake`; khi can tinh toan, uu tien `binary_label`.
- Dung `--invert-binary-labels` trong train/test neu can dao quy uoc.

Clip:

- `rgb.npy` co shape `(clip_len, 3, H, W)`.
- `diff.npy` co shape `(clip_len - 1, 3, H, W)`.
- `training.train --clip-len` phai khop voi clip shard.

Model:

- Spatial branch la ResNet50 ImageNet voi texture enhancement va spatial attention mac dinh bat.
- Temporal branch doc `diff.npy`; CLI train mac dinh `--temporal-pool gru`.
- Fusion head tra mot binary logit.

Resume:

- `frame_extractor` resume bang audit/metadata/frame da co.
- `face_detection` resume bang audit va scan shard da co.
- `build_clips` fail neu output split da co shard, tru khi dung `--overwrite`.

## Kiem Thu

```bash
python -m unittest discover -s tests
python -m compileall preprocessing training dataloader tests
```
