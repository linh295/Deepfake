# Pipeline Tiền Xử Lý

## Tổng Quan

Thứ tự thực thi khuyến nghị:

1. Tạo `videos_master.csv`
2. Trích xuất frame
3. Phát hiện khuôn mặt và ghi aligned frame shard
4. Tạo clip từ aligned frame shard

## 1. Tạo Metadata Cấp Video

Script:
- [metadata_level.py](../preprocessing/metadata_level.py)

Mục đích:
- quét thư mục dataset
- lấy FPS và số frame
- gán `train/val/test` một cách xác định, cân bằng trong từng category

Ví dụ:

```bash
python -m preprocessing.metadata_level --dataset-dir FaceForensics++_C23 --output-dir artifacts --output-name videos_master
```

Đầu ra chính:
- `artifacts/videos_master.csv`

## 2. Trích Xuất Frame

Script:
- [frame_extractor.py](../preprocessing/frame_extractor.py)

Mục đích:
- đọc `videos_master.csv`
- trích xuất frame theo target FPS
- giữ thông tin split trên mỗi dòng frame
- resume mặc định bằng cách dùng frame đã có cùng `frame_extraction_audit.csv`

Ví dụ:

Chỉ trích xuất dữ liệu train:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train
```

Ép chạy lại sạch không dùng resume:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train --no-resume
```

Chỉ trích xuất một category:

```bash
python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --category original
```

Đầu ra chính:
- `frame_data/<category>/<video_name>/*.jpg`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`

## 3. Phát Hiện Khuôn Mặt Và Tạo Frame Shard

Script:
- [face_detection.py](../preprocessing/face_detection.py)

Mục đích:
- detect khuôn mặt chính trên từng frame
- căn chỉnh ảnh bằng 5 landmark
- crop quanh bounding box đã căn chỉnh với mức context có thể cấu hình
- ghi WebDataset shard cấp frame

Ví dụ:

Chạy cho một split:

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
```

Chạy ba job độc lập:

```bash
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split val
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split test
```

Hành vi quan trọng:

- output tách riêng theo split khi có `--split`
- audit CSV cũng tách riêng theo split
- face crop được ghi theo streaming mode, không buffer theo từng video
- alignment chạy trên một canvas vuông lớn hơn, sau đó crop cuối cùng mới được resize về `--aligned-width/--aligned-height`

Đầu ra chính:

- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`

## 4. Tạo Clip Shard

Script:
- [build_clips.py](../preprocessing/build_clips.py)

Mục đích:
- đọc aligned frame shard
- tạo clip có độ dài cố định
- lưu RGB clip và frame-difference clip
- dùng `video_id` canonical của frame shard để group clip và đặt key

Ví dụ:

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train
```

Nếu output của split đó đã tồn tại clip shard, hãy chạy lại với `--overwrite` để rebuild sạch:

```bash
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train --overwrite
```

Đầu ra chính:
- `clip_data/<split>/shard-*.tar`

Hành vi quan trọng:

- việc group clip và đặt key dùng `video_id` canonical của frame shard
- rerun sẽ fail sớm nếu split shard đã tồn tại, trừ khi có `--overwrite`

## Mẫu Chạy Khuyến Nghị

Khi preprocessing toàn bộ dữ liệu, hãy chạy theo từng split sau khi `videos_master.csv` đã tồn tại:

1. `frame_extractor.py --split train`
2. `face_detection.py --split train`
3. `build_clips.py --split train`
4. Lặp lại cho `val`
5. Lặp lại cho `test`

Điều này giữ output tách bạch và giúp vận hành job song song đơn giản hơn.

## Bàn Giao Sang Huấn Luyện

Khi `clip_data/train/` và `clip_data/val/` đã tồn tại, stage training có thể tiêu thụ trực tiếp:

```bash
python -m training.train --train-shards "clip_data/train/shard-*.tar" --val-shards "clip_data/val/shard-*.tar"
```

Xem [Quy Trình Huấn Luyện](training-workflow.md) để biết chi tiết về mô hình, checkpoint, và runtime.
