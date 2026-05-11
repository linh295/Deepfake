# So Sánh Tài Liệu Cũ Và Tài Liệu Mới

## Mục Đích

Tài liệu này ghi lại những điểm khác nhau giữa phiên bản tài liệu trước đó và phiên bản vừa được cập nhật cho pipeline training deepfake detector.

Phạm vi so sánh gồm:

- [README.md](../README.md)
- [training-workflow.md](training-workflow.md)
- [architecture.md](architecture.md)
- [data-contracts.md](data-contracts.md)

## Tổng Quan Thay Đổi

Phiên bản cũ đã mô tả đúng luồng chính của dự án: preprocessing, dataloader, training, checkpoint, và data contract. Tuy nhiên phần training/model còn ở mức tổng quan, chưa phản ánh đầy đủ các tùy chọn hiện có trong code.

Phiên bản mới bổ sung chi tiết về:

- spatial attention
- texture enhancement
- temporal attention pooling
- augmentation trong training
- focal loss
- early stopping
- learning rate riêng theo từng nhánh
- output figure
- các trường mới trong `history.json`

## So Sánh Theo File

### `README.md`

Trước khi cập nhật:

- mô tả detector gồm `ResNet50`, CNN thời gian trên `diff.npy`, và fusion head
- có một lệnh train cơ bản
- liệt kê output training gồm `history.json`, `best.pt`, và checkpoint theo epoch

Sau khi cập nhật:

- nói rõ nhánh không gian có spatial attention và texture enhancement mặc định
- nói rõ nhánh thời gian hỗ trợ mean pooling hoặc attention pooling
- thêm ví dụ train với `--use-augmentation`, `--temporal-pool attention`, và `--loss-type focal`
- bổ sung ghi chú về augmentation nhất quán trên toàn clip
- bổ sung focal loss và early stopping vào hành vi runtime
- bổ sung output figure trong `figures/<run>/latest/` và `figures/<run>/best/`

Ý nghĩa:

- README giờ phản ánh tốt hơn các khả năng train thực tế.
- Người dùng mới có thể thấy ngay các tùy chọn nâng cao mà không cần đọc code.

### `docs/training-workflow.md`

Trước khi cập nhật:

- liệt kê các argument train cơ bản như shard, epoch, batch size, device, AMP, pos weight, và warmup
- mô tả detector ở mức ba thành phần: spatial branch, temporal branch, fusion head
- mô tả optimizer, loss, scheduler, gradient clipping, AMP, checkpoint selection, và warmup

Sau khi cập nhật:

- bổ sung CLI argument cho model option:
  - `--model-dropout`
  - `--temporal-pool {mean,attention}`
  - `--disable-spatial-attention`
  - `--disable-texture-enhancement`
- bổ sung CLI argument cho training behavior:
  - `--early-stopping-patience`
  - `--use-augmentation`
  - `--disable-augment-recompute-diff`
  - `--hflip-prob`
  - `--brightness`
  - `--contrast`
  - `--loss-type {bce,focal}`
  - `--focal-alpha`
  - `--focal-gamma`
- bổ sung mô tả chi tiết từng nhánh model
- bổ sung learning rate riêng cho spatial, temporal, và fusion branch
- bổ sung phần augmentation
- bổ sung output figure
- mô tả `history.json` có thêm learning rate, phase, tùy chọn model, và selection metric

Ý nghĩa:

- File này chuyển từ hướng dẫn training tổng quan thành tài liệu workflow sát với CLI hiện tại.
- Các tham số quan trọng trong `training/train.py` và `training/utils/builders.py` được đưa vào tài liệu.

### `docs/architecture.md`

Trước khi cập nhật:

- mô tả kiến trúc training là detector hai nhánh
- nhánh không gian dùng `ResNet50`
- nhánh thời gian dùng CNN nhẹ trên frame difference
- fusion head nối đặc trưng rồi phân loại
- phần checkpointing mô tả metric và state được lưu

Sau khi cập nhật:

- làm rõ temporal branch là CNN residual nhẹ
- thêm chi tiết về `SpatialResNet50`:
  - output 2048 chiều
  - texture enhancement từ feature nông `layer1`
  - fuse feature bằng `1x1 conv`
  - spatial attention trước global average pooling
- thêm chi tiết về `TemporalDiffCNN`:
  - encode từng frame-difference độc lập
  - project về 256 chiều trong cấu hình train mặc định
  - pool theo mean hoặc attention
- thêm chi tiết về `FusionHead`
- ghi rõ `return_features=True` trả thêm tensor debug/visualization
- bổ sung behavior của training loop:
  - freeze warmup
  - `ReduceLROnPlateau`
  - early stopping
  - render figure summary

Ý nghĩa:

- Tài liệu kiến trúc giờ khớp hơn với implementation trong `spatial_resnet50.py`, `temporal_diff_cnn.py`, `fusion_head.py`, và `spatio_temporal_detector.py`.
- Người đọc có thể hiểu model đang làm gì mà không phải mở từng file Python.

### `docs/data-contracts.md`

Trước khi cập nhật:

- mô tả `history.json` có `run`, `train`, và `val`
- `run` gồm `class_balance`, `use_pos_weight`, và `auto_pos_weight`
- validation có `selection_metric`, `selection_metric_name`, và `learning_rates`

Sau khi cập nhật:

- bổ sung các trường trong `run`:
  - `model_dropout`
  - `temporal_pool`
  - `use_spatial_attention`
  - `use_texture_enhancement`
- bổ sung trường validation:
  - `phase`

Ý nghĩa:

- Data contract giờ phản ánh đúng metadata training được ghi trong `history.json`.
- Việc phân biệt phase như `temporal_warmup` và `full_finetune` được tài liệu hóa.

## Bảng Tóm Tắt Khác Biệt

| Hạng mục | Docs cũ | Docs mới |
| --- | --- | --- |
| Spatial branch | Chỉ nói dùng `ResNet50` | Bổ sung spatial attention, texture enhancement, output 2048 chiều |
| Temporal branch | CNN trên `diff.npy` | Bổ sung CNN residual, feature 256 chiều, mean/attention pooling |
| Fusion head | Nối đặc trưng rồi phân loại | Bổ sung MLP với dropout và binary logit |
| CLI training | Chủ yếu argument cơ bản | Bổ sung model option, augmentation, focal loss, early stopping |
| Augmentation | Chưa mô tả | Mô tả clip-consistent augmentation và recompute diff |
| Loss | `BCEWithLogitsLoss` | Thêm focal loss |
| Checkpoint selection | AUC hoặc negative loss | Giữ nguyên, bổ sung liên hệ với scheduler và early stopping |
| Output | History và checkpoint | Thêm figure output |
| `history.json` | Trường cơ bản | Thêm model option và phase |

## Kết Luận

Tài liệu cũ phù hợp với pipeline cơ bản, nhưng thiếu nhiều chi tiết đã có trong code training hiện tại. Tài liệu mới không thay đổi thiết kế hệ thống, mà cập nhật mô tả để khớp với implementation hiện có và giúp người dùng cấu hình training chính xác hơn.
