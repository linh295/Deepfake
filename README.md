# Pipeline Du Lieu Deepfake Va Huan Luyen

Kho ma nay bien doi video FaceForensics++ C23 thanh WebDataset shard va huan luyen mo hinh phat hien deepfake spatio-temporal tren clip da tien xu ly.

Codebase gom hai phan chinh:

1. `preprocessing/`: tao manifest, trich xuat frame, can chinh khuon mat, tao frame shard va clip shard.
2. `dataloader/` + `training/`: doc clip shard, chuan hoa batch, huan luyen, danh gia va xuat checkpoint/metric.

## Cau Truc

```text
configs/         Cau hinh runtime, duong dan mac dinh va logging
preprocessing/   CLI tien xu ly du lieu
dataloader/      WebDataset loader cho clip-level shard
training/        Mo hinh, train/test entrypoint va utility
tests/           Unit test cho preprocessing, dataloader va training
docs/            Tai lieu kien truc, pipeline, data contract va workflow
artifacts/       Manifest, experiment output va test output
frame_data/      Frame da trich xuat
crop_data/       Frame shard da can chinh khuon mat
clip_data/       Clip shard dung cho train/val/test
figures/         Bieu do training neu render thanh cong
```

## Cai Dat

Du an yeu cau Python `>=3.12`.

```bash
pip install -e .
```

Neu dung `uv`:

```bash
uv sync
```

## Quy Uoc Label

Can chu y: `preprocessing.metadata_level` hien gan:

- `original = 1`
- cac manipulation nhu `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures` = `0`

`frame_extractor`, `face_detection`, `build_clips`, dataloader va training se truyen tiep `binary_label` nay. Neu can train/evaluate theo quy uoc nguoc lai, dung `--invert-binary-labels` trong `training.train`, `training.test` hoac `training.test_with_best_threshold`.

Luu y: truong text `label` o mot so stage sau van co the theo convention cu `0=real/1=fake`; khi can tinh toan hoac bao cao chinh xac, uu tien `binary_label`.

## Pipeline Tien Xu Ly

Thu tu khuyen nghi:

1. Tao manifest cap video.
2. Trich xuat frame.
3. Phat hien va can chinh khuon mat thanh frame shard.
4. Tao clip shard co do dai co dinh.

### 1. Tao `videos_master.csv`

```bash
python -m preprocessing.metadata_level \
  --dataset-dir FaceForensics++_C23 \
  --output-dir artifacts \
  --output-name videos_master
```

Dau ra:

- `artifacts/videos_master.csv`
- `artifacts/videos_master.parquet` neu moi truong ho tro parquet

Kiem tra phan bo split:

```bash
python -m preprocessing.analyze_videos_master --manifest artifacts/videos_master.csv
```

### 2. Trich Xuat Frame

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --split train
```

Resume duoc bat mac dinh thong qua audit CSV va frame da ton tai. Dung `--no-resume` de tao lai metadata/audit sach. Dung `--rebuild-csv-only` de tao lai metadata CSV tu frame da co.

Dau ra:

- `frame_data/<category>/<video_name>_frame_*.jpg`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`

### 3. Tao Frame Shard Da Can Chinh Khuon Mat

```bash
python -m preprocessing.face_detection \
  --metadata-csv frame_data/frame_extraction_metadata.csv \
  --frame-root frame_data \
  --output-dir crop_data \
  --split train
```

Stage nay dung RetinaFace, detect tren keyframe, noi suy bbox/landmark giua keyframe khi co the, can chinh 5 landmark, crop tren align canvas va resize ve output cuoi.

Dau ra:

- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`

### 4. Tao Clip Shard

```bash
python -m preprocessing.build_clips \
  --input-dir crop_data \
  --output-dir clip_data \
  --split train
```

Mac dinh moi clip co `clip_len=8`, `frame_stride=1`, `clip_stride=4`. Neu output split da ton tai, dung `--overwrite` de rebuild.

Dau ra:

- `clip_data/<split>/shard-*.tar`

Dem so clip:

```bash
python -m preprocessing.count_clips --clip-root clip_data --splits train val test
```

Lap lai cac buoc frame, face va clip cho `val` va `test`.

## Huan Luyen

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --output-dir artifacts/experiments/st_detector
```

Vi du bat augmentation, focal loss va temporal attention:

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --use-augmentation \
  --temporal-pool attention \
  --loss-type focal
```

Mo hinh gom:

- spatial branch: `ResNet50` ImageNet, mac dinh co texture enhancement va spatial attention
- temporal branch: CNN tren `diff.npy`, ho tro `mean`, `attention`, `gru`; CLI mac dinh la `gru`
- fusion head: MLP tren feature khong gian va thoi gian, xuat mot binary logit

Kiem tra tong so tham so cua mo hinh:

```bash
python -m training.model_parameters
```

Neu dung `uv`:

```bash
uv run python -m training.model_parameters
```

Dau ra training:

- `artifacts/experiments/st_detector/history.json`
- `artifacts/experiments/st_detector/best.pt`
- `artifacts/experiments/st_detector/checkpoint_epoch_*.pt`
- `figures/<run>/latest/` va `figures/<run>/best/epoch_*/` neu render figure thanh cong

## Danh Gia

Danh gia voi threshold thu cong:

```bash
python -m training.test \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --output-dir artifacts/test_results
```

Danh gia va tu tim threshold tot nhat:

```bash
python -m training.test_with_best_threshold \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --prediction-threshold-mode f1
```

Dau ra:

- `artifacts/test_results/test_metrics.json`
- `artifacts/test_results/test_predictions.csv`

## Kiem Thu

```bash
python -m unittest discover -s tests
python -m compileall preprocessing training dataloader tests
```

## Tai Lieu

- [Tong Hop Tai Lieu](docs/summary_documentation.md)
- [Kien Truc](docs/architecture.md)
- [Pipeline Tien Xu Ly](docs/preprocessing-pipeline.md)
- [Quy Trinh Huan Luyen Va Danh Gia](docs/training-workflow.md)
- [Data Contract](docs/data-contracts.md)
- [Thay Doi Tai Lieu](docs/documentation-changes.md)
- [Dong Gop](CONTRIBUTING.md)
