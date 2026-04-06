# Kiến Trúc

## Mục Tiêu

Kho mã này hỗ trợ một quy trình deepfake end-to-end:

1. biến đổi video gốc thành frame shard và clip shard có thể tái sử dụng
2. huấn luyện detector spatio-temporal trên input WebDataset cấp clip

## Luồng Xử Lý Tổng Quan

```text
Video gốc
  -> manifest metadata cấp video
  -> frame đã trích xuất
  -> frame shard đã căn chỉnh khuôn mặt
  -> clip shard
  -> batch từ dataloader
  -> mô hình spatio-temporal
  -> checkpoint + history
```

## Các Thành Phần Chính

### `configs/`

- [settings.py](../configs/settings.py)
  Tập trung hóa đường dẫn mặc định và runtime setting.
- [loggings.py](../configs/loggings.py)
  Cấu hình logging dùng chung cho preprocessing và training.

`frame_extractor.py` cũng có fallback sang `FaceForensics++_C23` khi thư mục raw-data được cấu hình không tồn tại, vì vậy CLI argument vẫn là nguồn sự thật an toàn nhất cho dataset path.

### `preprocessing/`

- [metadata_level.py](../preprocessing/metadata_level.py)
  Tạo `videos_master.csv` từ thư mục dataset và gán split một cách xác định.
- [frame_extractor.py](../preprocessing/frame_extractor.py)
  Trích xuất frame, giữ lại split metadata, và hỗ trợ resume.
- [face_detection.py](../preprocessing/face_detection.py)
  CLI entrypoint mỏng cho pipeline face detection.
- [_face_detection_pipeline.py](../preprocessing/_face_detection_pipeline.py)
  Chứa logic chính cho detect khuôn mặt, căn chỉnh, crop, nội suy, resume, và ghi shard.
- [build_clips.py](../preprocessing/build_clips.py)
  Tạo clip có độ dài cố định từ frame shard đã căn chỉnh.

### `dataloader/`

- [dataset.py](../dataloader/dataset.py)
  Đọc clip-level WebDataset shard, kiểm tra shape của sample, normalize input không gian và thời gian, và collate thành batch cho training.

Dataloader sinh ra:

- một frame RGB cho nhánh không gian
- một chuỗi tensor frame-difference cho nhánh thời gian
- một nhãn nhị phân
- clip metadata để truy vết

### `training/`

- [train.py](../training/train.py)
  CLI chính cho việc huấn luyện detector spatio-temporal.
- [spatio_temporal_detector.py](../training/spatio_temporal_detector.py)
  Nối dây cấp cao nhất cho mô hình.
- [spatial_resnet50.py](../training/spatial_resnet50.py)
  Nhánh không gian dựa trên torchvision `ResNet50`.
- [temporal_diff_cnn.py](../training/temporal_diff_cnn.py)
  Nhánh thời gian trên chuỗi `diff.npy`.
- [fusion_head.py](../training/fusion_head.py)
  Ghép đặc trưng không gian và thời gian thành bộ phận phân loại nhị phân.
- `training/utils/`
  Gồm builder, vòng lặp train, metric, checkpointing, ước tính class-balance, freezing, progress estimation, và runtime helper.

## Biên Giới Dữ Liệu

### Manifest Cấp Video

Được sinh bởi `metadata_level.py`.

Mục đích:

- liệt kê các video
- ghi lại category và nhãn
- gán `train/val/test`
- cung cấp input ổn định cho bước trích xuất frame

### Manifest Cấp Frame

Được sinh bởi `frame_extractor.py`.

Mục đích:

- ghi lại đường dẫn frame đã trích xuất
- giữ lại frame index gốc và timestamp
- chuyển tiếp thông tin split xuống các stage sau

### Frame Shard

Được sinh bởi `face_detection.py`.

Mục đích:

- lưu ảnh khuôn mặt đã căn chỉnh và metadata trong WebDataset shard
- hỗ trợ resume thông qua shard đã tồn tại và audit CSV
- tách riêng output theo split khi dùng `--split`

### Clip Shard

Được sinh bởi `build_clips.py`.

Mục đích:

- gom các frame đã căn chỉnh thành clip có độ dài cố định
- lưu RGB clip và frame-difference clip dưới dạng `.npy`
- giữ lại đủ metadata để phục vụ training và truy vết

### Đầu Ra Huấn Luyện

Được sinh bởi `training/train.py`.

Mục đích:

- ghi metric theo từng epoch vào `history.json`
- lưu checkpoint có thể resume
- lưu mô hình tốt nhất theo logic chọn validation

## Cô Lập Theo Split

Pipeline preprocessing được thiết kế để chạy `train`, `val`, và `test` một cách độc lập sau khi split đã được gán.

Hành vi hiện tại theo từng split:

- frame metadata có trường `split`
- face detection có thể lọc bằng `--split`
- face shard được ghi vào `crop_data/<split>/`
- file audit của face detection được ghi theo đường dẫn riêng cho từng split
- clip builder có thể xử lý một split hoặc tự động tìm tất cả thư mục split
- training sử dụng shard glob tách riêng theo split, ví dụ `clip_data/train/shard-*.tar`

## Thiết Kế Face Detection

Stage face detection hiện tại sử dụng:

- RetinaFace để detect
- chọn khuôn mặt chính dựa trên kích thước và tính liên tục theo thời gian
- detect ở keyframe và nội suy giữa các lần detect
- căn chỉnh 5 điểm mốc
- căn chỉnh lên một canvas vuông lớn hơn
- crop theo bounding box trong hệ tọa độ đã căn chỉnh
- resize cuối cùng về kích thước output yêu cầu

Sự tách biệt giữa align canvas và output cuối giúp giữ thêm bối cảnh trước khi crop được resize.

## Thiết Kế Mô Hình

Stack training sử dụng detector hai nhánh:

- nhánh không gian: frame RGB đã normalize đưa qua `ResNet50`
- nhánh thời gian: chuỗi frame-difference đã normalize đưa qua CNN nhẹ
- fusion head: nối đặc trưng rồi đưa qua MLP classifier

Nhánh thời gian kỳ vọng `clip_len - 1` bước vì `diff.npy` lưu hiệu của các cặp frame liên tiếp.

## Đánh Giá Và Checkpointing

Huấn luyện tính:

- loss
- accuracy
- F1
- ROC AUC khi cả hai lớp cùng xuất hiện

Việc chọn checkpoint sử dụng validation AUC nếu có, và fallback sang negative validation loss nếu không. Checkpoint cũng bao gồm:

- train config
- model config
- metadata class-balance
- optimizer, scheduler, scaler, và RNG state
