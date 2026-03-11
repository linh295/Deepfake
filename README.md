# Frame Extraction (5 FPS)

Repo hiện dùng **một cách chạy duy nhất** để tách frame từ toàn bộ dataset:

- `main.py extract-frames`

## 1) Cài dependencies

```bash
pip install -e .
```

## 2) Input yêu cầu

File `--clean-csv` phải có các cột:

- `video_abs_path`
- `video_rel_path`
- `video_id`
- `method`
- `label`
- `label_str`
- `num_frames`

## 3) Chạy extract 5fps

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

## 4) Output

- Frame được lưu theo nhãn đã chuẩn hoá:
  - `artifacts/frames_5fps/real/<method>/<video_id>/frame_*.jpg`
  - `artifacts/frames_5fps/fake/<method>/<video_id>/frame_*.jpg`
- Manifest: `artifacts/frames_5fps/frame_manifest.csv`
