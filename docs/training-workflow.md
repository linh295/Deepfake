# Quy Trinh Huan Luyen Va Danh Gia

## Tong Quan

Training tieu thu clip-level WebDataset shard tu `preprocessing.build_clips` va huan luyen detector nhi phan spatio-temporal.

Entrypoint:

- [training/train.py](../training/train.py): train/validate/checkpoint.
- [training/test.py](../training/test.py): evaluate test set voi threshold thu cong.
- [training/test_with_best_threshold.py](../training/test_with_best_threshold.py): evaluate va tim threshold theo metric.

## Train Input

Bat buoc:

- `--train-shards`
- `--val-shards`

Lenh co ban:

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --output-dir artifacts/experiments/st_detector
```

## Tham So CLI Chinh

Du lieu va runtime:

- `--output-dir`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--clip-len`
- `--device`
- `--seed`
- `--disable-amp`
- `--disable-pin-memory`
- `--disable-persistent-workers`
- `--train-shuffle-buffer`
- `--log-every`
- `--save-every`

Label va class balance:

- `--invert-binary-labels`
- `--disable-pos-weight`
- `--disable-auto-pos-weight`
- `--max-pos-weight`

Optimizer/scheduler:

- `--lr-spatial`
- `--lr-temporal`
- `--lr-fusion`
- `--weight-decay`
- `--grad-clip-norm`
- `--scheduler-factor`
- `--scheduler-patience`
- `--scheduler-threshold`
- `--scheduler-min-lr`

Model:

- `--model-dropout`
- `--temporal-pool {mean,attention,gru}`
- `--disable-spatial-attention`
- `--disable-texture-enhancement`
- `--spatial-freeze-warmup-epochs`

Augmentation:

- `--use-augmentation`
- `--disable-augment-recompute-diff`
- `--hflip-prob`
- `--brightness`
- `--contrast`
- `--jpeg-prob`
- `--jpeg-quality-min`
- `--jpeg-quality-max`
- `--blur-prob`
- `--blur-sigma-min`
- `--blur-sigma-max`

Loss va early stopping:

- `--loss-type {bce,focal}`
- `--focal-alpha`
- `--focal-gamma`
- `--early-stopping-patience`

## Label

Manifest hien tai gan `original=1`, manipulation `=0`. Training su dung label trong shard. Neu muon doi positive class theo quy uoc nguoc lai, chay train/test voi:

```bash
--invert-binary-labels
```

Nen dung cung mot quy uoc invert cho train, validation va test.

## Batch Contract

Moi batch tu dataloader gom:

- `spatial`: tensor `(B, 3, H, W)`, mot frame RGB da normalize ImageNet.
- `temporal`: tensor `(B, T-1, 3, H, W)`, frame difference scale ve `[0, 1]`.
- `label`: tensor `(B,)`.
- `spatial_index`: index frame RGB duoc chon tu clip.
- `meta`: metadata JSON cua clip.

Train mode chon ngau nhien mot index trong `center_candidate_indices`; validation/test dung `default_center_index` neu co.

## Mo Hinh

Detector:

```text
spatial RGB frame -> SpatialResNet50 -> spatial feature 2048
diff sequence     -> TemporalDiffCNN -> temporal feature 256
concat features   -> FusionHead      -> binary logit
```

Spatial branch:

- ResNet50 ImageNet.
- Texture enhancement mac dinh bat.
- Spatial attention mac dinh bat.
- Co the freeze trong warmup bang `--spatial-freeze-warmup-epochs`.

Temporal branch:

- Input la `diff.npy`, do dai `clip_len - 1`.
- CNN residual encode tung timestep.
- `--temporal-pool mean`: average theo thoi gian.
- `--temporal-pool attention`: learned attention theo timestep.
- `--temporal-pool gru`: bidirectional GRU tren chuoi feature; day la mac dinh cua CLI train hien tai.

Fusion head:

- MLP hai hidden layer voi dropout.
- Tra ve mot logit; xac suat duoc tinh bang sigmoid khi evaluate.

## Loss, Optimizer Va Scheduler

Mac dinh:

- optimizer: `AdamW`
- loss: `BCEWithLogitsLoss`
- scheduler: `ReduceLROnPlateau(mode="max")`
- AMP: bat tren CUDA
- gradient clipping: bat theo `--grad-clip-norm`

Learning rate tach theo nhanh:

- `--lr-spatial`
- `--lr-temporal`
- `--lr-fusion`

Class balance:

- Neu `use_pos_weight` va `auto_pos_weight` bat, training scan train shard de tinh `pos_weight`.
- `--max-pos-weight` co the gioi han trong so positive.
- Focal loss khong dung `pos_weight`.

## Augmentation

Augmentation chi ap dung cho train loader khi co `--use-augmentation`.

Nguyen tac:

- Mot bo tham so transform duoc sample cho ca clip, giu nhat quan thoi gian.
- Horizontal flip, brightness va contrast chay tren tensor clip.
- Gaussian blur va JPEG compression co the bat bang probability rieng.
- Mac dinh recompute `diff.npy` tu RGB da augment de temporal input khop voi spatial input.
- Dung `--disable-augment-recompute-diff` neu muon giu diff goc.

Vi du:

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar" \
  --use-augmentation \
  --hflip-prob 0.5 \
  --brightness 0.1 \
  --contrast 0.1 \
  --jpeg-prob 0.2 \
  --blur-prob 0.1
```

## Checkpoint Selection

Moi epoch:

1. Train mot epoch.
2. Validate mot epoch.
3. Tinh selection metric.
4. Step scheduler.
5. Ghi checkpoint theo `--save-every`.
6. Cap nhat `best.pt` neu metric tot hon.
7. Ghi `history.json`.
8. Render figures neu moi truong ho tro.

Selection metric:

- `val_auc` neu validation co ca hai lop.
- `-val_loss` neu AUC khong tinh duoc.

Early stopping dung cung selection metric va `--early-stopping-patience`.

## Train Output

Thu muc mac dinh:

- `artifacts/experiments/st_detector`

File:

- `history.json`
- `best.pt`
- `checkpoint_epoch_*.pt`

Figure:

- `figures/<run>/latest/`
- `figures/<run>/best/epoch_*/`

`history.json` gom:

- `run`: class balance, pos_weight option, model option.
- `train`: metric train theo epoch.
- `val`: metric val, selection metric, learning rates va phase.

Checkpoint gom:

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

## Danh Gia Test

Evaluate voi threshold thu cong:

```bash
python -m training.test \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --output-dir artifacts/test_results \
  --threshold 0.5
```

Evaluate va tim threshold:

```bash
python -m training.test_with_best_threshold \
  --test-shards "clip_data/test/shard-*.tar" \
  --checkpoint artifacts/experiments/st_detector/best.pt \
  --output-dir artifacts/test_results \
  --prediction-threshold-mode f1
```

`--prediction-threshold-mode` ho tro:

- `manual`
- `youden`
- `f1`
- `balanced_accuracy`

Test output:

- `test_metrics.json`: checkpoint metadata, metric, confusion matrix, label/pred distribution, threshold info.
- `test_predictions.csv`: key, video_id, category, label, probability, pred, spatial_index.

## Kiem Tra Va Debug

Test lien quan:

```bash
python -m unittest tests.test_dataset_loader tests.test_spatio_temporal_detector tests.test_training_train
python -m compileall training dataloader tests
```

Neu train/test loi som, kiem tra:

- shard glob co match file `.tar` khong
- `--clip-len` co khop voi clip shard khong
- label convention co can `--invert-binary-labels` khong
- device co san sang khong
- `torchvision` co cai dat dung voi `torch` khong
- checkpoint co `model_config` tu training script hien tai khong
