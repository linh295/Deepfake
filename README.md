# Pipeline Dữ Liệu Deepfake Và Huấn Luyện

Kho mã này biến đổi video kiểu FaceForensics++ thành các WebDataset shard sẵn sàng cho huấn luyện, sau đó huấn luyện mô hình phát hiện deepfake spatio-temporal trên các clip đã tạo ra.

Codebase hiện tại có hai lớp chính:

1. `preprocessing/`
   Tạo manifest, trích xuất frame, căn chỉnh khuôn mặt, và ghi frame shard cùng clip shard.
2. `dataloader/` + `training/`
   Đọc clip shard, chuẩn bị input cho mô hình, và huấn luyện detector kết hợp đặc trưng không gian từ RGB với đặc trưng thời gian từ frame difference.

## Cấu Trúc Kho Mã

```text
configs/         Cấu hình runtime và logging
preprocessing/   Các bước tiền xử lý và CLI entrypoint
dataloader/      Bộ đọc WebDataset cho clip
training/        Mô hình, vòng lặp train, checkpointing, và utility
tests/           Unit test và integration test cho preprocessing và training
docs/            Tài liệu kiến trúc, pipeline, data contract, và hướng dẫn training
artifacts/       Manifest sinh ra và output thử nghiệm
frame_data/      Frame đã trích xuất và metadata frame
crop_data/       Frame shard đã căn chỉnh khuôn mặt
clip_data/       Clip shard dùng cho huấn luyện
```

## Cài Đặt

Dự án yêu cầu Python `>=3.12`.

Cài dependency ở chế độ editable:

```bash
pip install -e .
```

## Quy Trình Tiền Xử Lý

Thứ tự tiền xử lý khuyến nghị:

1. Tạo `videos_master.csv`
2. Trích xuất frame
3. Phát hiện và căn chỉnh khuôn mặt thành frame shard
4. Tạo clip shard có độ dài cố định

### 1. Tạo Metadata Cấp Video

```bash
python -m preprocessing.metadata_level --dataset-dir FaceForensics++_C23 --output-dir artifacts --output-name videos_master
```

Đầu ra:

- `artifacts/videos_master.csv`

### 2. Trích Xuất Frame

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train
```

Chế độ resume được bật mặc định. Dùng `--no-resume` nếu muốn chạy lại sạch.

Đầu ra:

- `frame_data/<category>/<video_name>/*.jpg`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`

### 3. Tạo Frame Shard Đã Căn Chỉnh Khuôn Mặt

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
```

Đầu ra:

- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`

### 4. Tạo Clip Shard

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train
```

Đầu ra:

- `clip_data/<split>/shard-*.tar`

Lặp lại các bước frame, face, và clip cho `val` và `test` khi chuẩn bị toàn bộ dataset.

## Quy Trình Huấn Luyện

Huấn luyện sử dụng clip shard và xây dựng detector gồm:

- nhánh không gian dựa trên `ResNet50`
- CNN thời gian trên `diff.npy`
- fusion head cho bài toán phân loại nhị phân

Lệnh train ví dụ:

```bash
python -m training.train --train-shards "clip_data/train/shard-*.tar" --val-shards "clip_data/val/shard-*.tar" --output-dir artifacts/experiments/st_detector
```

Hành vi runtime quan trọng:

- nhánh thời gian kỳ vọng `clip_len - 1` frame thời gian từ `diff.npy`
- nhánh không gian có thể bị đóng băng trong warmup thông qua `--spatial-freeze-warmup-epochs`
- độ lệch lớp có thể được ước tính từ training shard và chuyển thành `pos_weight`
- việc chọn checkpoint ưu tiên validation AUC, và fallback sang negative validation loss khi AUC không xác định

Đầu ra huấn luyện:

- `artifacts/experiments/st_detector/history.json`
- `artifacts/experiments/st_detector/best.pt`
- `artifacts/experiments/st_detector/checkpoint_epoch_*.pt`

## Tổng Quan Đầu Ra

- `artifacts/videos_master.csv`
  Manifest cấp video với split được gán một cách xác định.
- `frame_data/frame_extraction_metadata.csv`
  Manifest cấp frame do `frame_extractor.py` sinh ra.
- `frame_data/frame_extraction_audit.csv`
  Tệp audit phục vụ resume cho các video đã xử lý trong `frame_extractor.py`.
- `crop_data/<split>/shard-*.tar`
  WebDataset shard cấp frame chứa ảnh khuôn mặt đã căn chỉnh.
- `clip_data/<split>/shard-*.tar`
  WebDataset shard cấp clip chứa `rgb.npy`, `diff.npy`, và metadata.
- `artifacts/experiments/st_detector/`
  Lịch sử train và checkpoint.

## Kiểm Thử

Chạy toàn bộ test suite:

```bash
python -m unittest discover -s tests
```

Chạy nhóm test preprocessing:

```bash
python -m unittest tests.test_face_detection tests.test_face_detection_split tests.test_face_detection_crop
```

Chạy compile check:

```bash
python -m compileall preprocessing training dataloader tests
```

## Tài Liệu

- [Kiến Trúc](docs/architecture.md)
- [Pipeline Tiền Xử Lý](docs/preprocessing-pipeline.md)
- [Quy Trình Huấn Luyện](docs/training-workflow.md)
- [Data Contract](docs/data-contracts.md)
- [Đóng Góp](CONTRIBUTING.md)
