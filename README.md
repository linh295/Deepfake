# Preprocessing Pipeline

Pipeline được tách theo đúng chức năng:

- `preprocessing/metadata_cleaner.py`: clean metadata đầu vào.
- `preprocessing/frame_extractor.py`: tách frame từ video.
- `preprocessing/data_standardization.py`: chuẩn hoá dữ liệu frame theo cấu trúc train.

Lệnh điều phối nằm ở `main.py`.

## 1) Cài dependencies

```bash
pip install -e .
```

## 2) Clean là gì?

`clean-metadata` là bước làm sạch và chuẩn hoá metadata video trước khi extract frame, gồm:

- Chuẩn hoá nhãn `Label` về `label` (0/1) và `label_str` (REAL/FAKE).
- Tạo cột dùng chung cho các bước sau: `video_abs_path`, `video_rel_path`, `video_id`, `method`, `num_frames`.
- Lọc video không đủ số frame tối thiểu (`min_frames`).
- (Tuỳ chọn) kiểm tra file video có tồn tại thật trên đĩa.

Output của bước này là `artifacts/clean_metadata.csv`.

## 3) Input yêu cầu cho extract

File `--clean-csv` phải có các cột:

- `video_abs_path`
- `video_rel_path`
- `video_id`
- `method`
- `label`
- `label_str`
- `num_frames`

## 4) Chạy clean + extract (khuyến nghị)

```bash
python main.py clean-metadata \
  --dataset-root "FaceForensics++_C23 2" \
  --metadata-csv "FaceForensics++_C23 2/csv/FF++_Metadata_Shuffled.csv" \
  --output-csv artifacts/clean_metadata.csv

python main.py extract-frames \
  --clean-csv artifacts/clean_metadata.csv \
  --output-dir artifacts/frames_5fps \
  --target-fps 5
```

## 5) Chạy extract 5fps

```bash
python main.py extract-frames \
  --clean-csv artifacts/clean_metadata.csv \
  --output-dir artifacts/frames_5fps \
  --target-fps 5
```

Tuỳ chọn resize:

```bash
python main.py extract-frames \
  --clean-csv artifacts/clean_metadata.csv \
  --output-dir artifacts/frames_5fps_224 \
  --target-fps 5 \
  --resize-width 224 \
  --resize-height 224
```

## 6) Output

- Frame được lưu theo nhãn đã chuẩn hoá:
  - `artifacts/frames_5fps/real/<method>/<video_id>/frame_*.jpg`
  - `artifacts/frames_5fps/fake/<method>/<video_id>/frame_*.jpg`
- Manifest: `artifacts/frames_5fps/frame_manifest.csv`

## 7) Chuẩn hoá sau khi extract (tuỳ chọn)

```bash
python main.py standardize \
  --manifest-csv artifacts/frames_5fps/frame_manifest.csv \
  --frames-root artifacts/frames_5fps \
  --output-dir artifacts/dataset_standardized \
  --mode flat \
  --copy-mode copy
```
