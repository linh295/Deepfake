# Quy Trình Huấn Luyện

## Tổng Quan

Huấn luyện tiêu thụ clip-level WebDataset shard được tạo bởi `preprocessing.build_clips` và huấn luyện một detector nhị phân spatio-temporal.

Entrypoint chính:

- [training/train.py](../training/train.py)

Các stage huấn luyện chính:

1. tạo train dataloader và validation dataloader từ clip shard
2. suy ra metadata class-balance nếu cần từ training shard
3. tạo detector hai nhánh
4. train và validate theo từng epoch
5. ghi history và checkpoint

## Đầu Vào

CLI argument bắt buộc:

- `--train-shards`
- `--val-shards`

Lệnh sử dụng thông thường:

```bash
python -m training.train --train-shards "clip_data/train/shard-*.tar" --val-shards "clip_data/val/shard-*.tar"
```

Các argument tùy chọn thường dùng:

- `--output-dir`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--clip-len`
- `--device`
- `--disable-amp`
- `--invert-binary-labels`
- `--disable-pos-weight`
- `--disable-auto-pos-weight`
- `--max-pos-weight`
- `--model-dropout`
- `--temporal-pool {mean,attention}`
- `--disable-spatial-attention`
- `--disable-texture-enhancement`
- `--spatial-freeze-warmup-epochs`
- `--early-stopping-patience`
- `--use-augmentation`
- `--disable-augment-recompute-diff`
- `--hflip-prob`
- `--brightness`
- `--contrast`
- `--loss-type {bce,focal}`
- `--focal-alpha`
- `--focal-gamma`

## Ngữ Nghĩa Của Mô Hình Và Batch

Detector gồm:

- một nhánh không gian trên một frame RGB của mỗi clip
- một nhánh thời gian trên frame difference
- một fusion head để sinh binary logits

Giả định quan trọng:

- `rgb.npy` phải có `clip_len` frame
- `diff.npy` phải có `clip_len - 1` frame
- validation chọn center frame một cách xác định
- training lấy mẫu spatial frame từ tập candidate center index

Nhánh không gian:

- dùng torchvision `ResNet50` với weight ImageNet khi train qua CLI
- xuất vector đặc trưng 2048 chiều
- mặc định bật texture enhancement từ feature nông và spatial attention trên feature map đã fuse
- có thể tắt từng phần bằng `--disable-texture-enhancement` và `--disable-spatial-attention`

Nhánh thời gian:

- nhận tensor shape `[B, T-1, 3, H, W]`
- encode từng frame-difference bằng CNN residual nhẹ
- xuất vector đặc trưng 256 chiều khi được build từ `training.train`
- pooling mặc định là mean; `--temporal-pool attention` bật attention theo thời gian

Fusion head:

- nối đặc trưng không gian và thời gian
- dùng MLP với dropout để sinh một binary logit cho mỗi sample

## Hành Vi Tối Ưu

Huấn luyện hiện tại sử dụng:

- `AdamW`
- `BCEWithLogitsLoss` mặc định hoặc focal loss khi `--loss-type focal`
- `pos_weight` tùy chọn được suy ra từ class balance
- `ReduceLROnPlateau` với `mode="max"`
- gradient clipping
- AMP trên CUDA theo mặc định
- early stopping theo selection metric

Optimizer dùng learning rate tách theo nhánh:

- `--lr-spatial`
- `--lr-temporal`
- `--lr-fusion`

Chọn checkpoint:

- dùng validation AUC khi có thể tính được
- ngược lại dùng negative validation loss

Hành vi warmup:

- nhánh không gian bị đóng băng trong `spatial_freeze_warmup_epochs` đầu tiên
- sau warmup, toàn bộ detector được fine-tune

Augmentation:

- chỉ áp dụng cho train loader khi bật `--use-augmentation`
- cùng một bộ transform được áp dụng cho toàn clip để giữ tính nhất quán thời gian
- hỗ trợ horizontal flip, brightness, và contrast
- mặc định recompute frame-difference từ RGB đã augment; có thể tắt bằng `--disable-augment-recompute-diff`

## Đầu Ra

Thư mục output mặc định:

- `artifacts/experiments/st_detector`

Tệp được tạo:

- `history.json`
- `best.pt`
- `checkpoint_epoch_*.pt`
- figure summary trong thư mục `figures/` khi render thành công

`history.json` lưu metric theo epoch, learning rate hiện tại, phase huấn luyện, tùy chọn model, và thông tin selection metric. Mỗi checkpoint cũng bao gồm optimizer, scheduler, scaler, config, class-balance, và RNG state.

## Xác Thực Và Debug

Kiểm tra khuyến nghị:

```bash
python -m unittest tests.test_dataset_loader tests.test_spatio_temporal_detector tests.test_training_train
python -m compileall training dataloader tests
```

Nếu training lỗi sớm, hãy kiểm tra:

- glob của train shard và validation shard có resolve ra tệp `.tar` thực tế hay không
- `clip_len` có khớp với clip shard đang được load hay không
- device được chọn có sẵn sàng hay không
- `torchvision` đã được cài đặt cho nhánh `ResNet50` hay chưa
