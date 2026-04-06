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
- `--spatial-freeze-warmup-epochs`

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

## Hành Vi Tối Ưu

Huấn luyện hiện tại sử dụng:

- `AdamW`
- `BCEWithLogitsLoss`
- `pos_weight` tùy chọn được suy ra từ class balance
- `ReduceLROnPlateau` với `mode="max"`
- gradient clipping
- AMP trên CUDA theo mặc định

Chọn checkpoint:

- dùng validation AUC khi có thể tính được
- ngược lại dùng negative validation loss

Hành vi warmup:

- nhánh không gian bị đóng băng trong `spatial_freeze_warmup_epochs` đầu tiên
- sau warmup, toàn bộ detector được fine-tune

## Đầu Ra

Thư mục output mặc định:

- `artifacts/experiments/st_detector`

Tệp được tạo:

- `history.json`
- `best.pt`
- `checkpoint_epoch_*.pt`

`history.json` lưu metric theo epoch và thông tin selection metric. Mỗi checkpoint cũng bao gồm optimizer, scheduler, scaler, config, class-balance, và RNG state.

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
