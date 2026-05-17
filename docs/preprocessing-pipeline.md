# Pipeline Tien Xu Ly

## Tong Quan

Pipeline tien xu ly tao du lieu theo tung tang:

```text
videos_master.csv
  -> frame_extraction_metadata.csv + frame JPG
  -> crop_data/<split>/shard-*.tar
  -> clip_data/<split>/shard-*.tar
```

Nen chay theo split (`train`, `val`, `test`) sau khi manifest video da co de output tach bach va de resume de kiem soat.

## 1. Tao Metadata Cap Video

Script:

- [metadata_level.py](../preprocessing/metadata_level.py)

Lenh:

```bash
python -m preprocessing.metadata_level \
  --dataset-dir FaceForensics++_C23 \
  --output-dir artifacts \
  --output-name videos_master
```

Tham so chinh:

- `--dataset-dir`: thu muc dataset FaceForensics++ C23.
- `--output-dir`: noi ghi manifest.
- `--output-name`: ten file khong gom extension.
- `--workers`: so worker doc metadata video.

Dau ra:

- `artifacts/videos_master.csv`
- `artifacts/videos_master.parquet` neu co engine parquet phu hop

Cot chinh:

- `video_id`
- `video_path`
- `category`
- `binary_label`
- `compression`
- `split`
- `original_fps`
- `num_frames`

Quy uoc label hien tai:

- `original = 1`
- manipulation category = `0`

Kiem tra split:

```bash
python -m preprocessing.analyze_videos_master --manifest artifacts/videos_master.csv
```

## 2. Trich Xuat Frame

Script:

- [frame_extractor.py](../preprocessing/frame_extractor.py)

Lenh train split:

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --split train
```

Lenh chi mot category:

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --category original
```

Rebuild metadata CSV tu frame da co:

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --rebuild-csv-only
```

Chay lai sach khong resume:

```bash
python -m preprocessing.frame_extractor \
  --manifest artifacts/videos_master.csv \
  --split train \
  --no-resume
```

Tham so thuong dung:

- `--manifest`: `videos_master.csv`.
- `--category`: loc category.
- `--split`: loc split.
- `--workers`: override so process.
- `--fps`: override target FPS, mac dinh tu `settings.TARGET_FPS = 5`.
- `--jpeg-quality`: override quality JPG.
- `--no-resume`: reset metadata/audit cho lan chay moi.
- `--rebuild-csv-only`: khong decode video, chi index frame san co.

Dau ra:

- `frame_data/<category>/<video_name>_frame_*.jpg`
- `frame_data/frame_extraction_metadata.csv`
- `frame_data/frame_extraction_audit.csv`

Resume:

- Mac dinh bat.
- Audit key gom `category|video_id|split`.
- Neu audit thieu nhung metadata/frame day du, script co the bootstrap lai audit.
- Neu metadata thieu cho video da complete, script co the recover row tu frame da co.

## 3. Phat Hien Va Can Chinh Khuon Mat

Script:

- [face_detection.py](../preprocessing/face_detection.py)
- [_face_detection_pipeline.py](../preprocessing/_face_detection_pipeline.py)

Lenh:

```bash
python -m preprocessing.face_detection \
  --metadata-csv frame_data/frame_extraction_metadata.csv \
  --frame-root frame_data \
  --output-dir crop_data \
  --split train
```

Tham so quan trong:

- `--metadata-csv`: metadata frame tu stage truoc.
- `--frame-root`: root de resolve `frame_path`.
- `--output-dir`: root frame shard output.
- `--category`: loc category.
- `--split`: loc split va ghi output vao `output-dir/split`.
- `--limit`: gioi han so row de test nhanh.
- `--threshold`: nguong RetinaFace.
- `--max-side`: resize anh truoc detection de canh dai nhat khong vuot qua gioi han.
- `--aligned-width`, `--aligned-height`: kich thuoc crop output, mac dinh `224`.
- `--crop-scale`: mo rong crop quanh bbox.
- `--detect-every-k`: tan suat detect keyframe.
- `--image-format`: `.jpg`, `.jpeg` hoac `.png`.
- `--jpeg-quality`: quality neu ghi JPG.
- `--shard-maxcount`, `--shard-maxsize`: nguong xoay shard.
- `--skip-no-face`: giu semantic CLI cu; stage hien van chi ghi sample da align duoc.
- `--audit-csv`: audit CSV goc; khi co `--split`, file audit se duoc ghi vao thu muc split.

Dau ra:

- `crop_data/<split>/shard-*.tar`
- `audit/<split>/face_detection_audit.csv`

Moi sample shard gom:

- `json`: metadata detection/alignment/crop
- `jpg` hoac `png`: anh mat da can chinh
- `cls`: label neu hop le

Luu y van hanh:

- Output duoc resume bang audit va scan shard da co.
- Frame khong doc duoc hoac khong align duoc se khong tao sample.
- `build_clips.py` yeu cau frame shard giu thu tu contiguous theo video.

## 4. Tao Clip Shard

Script:

- [build_clips.py](../preprocessing/build_clips.py)

Lenh:

```bash
python -m preprocessing.build_clips \
  --input-dir crop_data \
  --output-dir clip_data \
  --split train
```

Rebuild split da ton tai:

```bash
python -m preprocessing.build_clips \
  --input-dir crop_data \
  --output-dir clip_data \
  --split train \
  --overwrite
```

Tham so:

- `--input-dir`: root frame shard, thuong la `crop_data`.
- `--output-dir`: root clip shard, thuong la `clip_data`.
- `--split`: neu bo trong, script tu tim `train/val/test`.
- `--clip-len`: so frame moi clip, mac dinh `8`.
- `--frame-stride`: buoc frame trong clip, mac dinh `1`.
- `--clip-stride`: sliding-window stride, mac dinh `4`.
- `--shard-maxcount`: so clip toi da moi shard.
- `--shard-maxsize`: kich thuoc toi da moi shard.
- `--overwrite`: xoa output split va build lai.

Dau ra:

- `clip_data/<split>/shard-*.tar`

Moi sample clip gom:

- `json`
- `rgb.npy`
- `diff.npy`
- `cls` neu co label

`diff.npy` la absolute frame difference giua cac frame lien tiep, vi vay co `clip_len - 1` timestep.

## Mau Chay Day Du

```bash
python -m preprocessing.metadata_level --dataset-dir FaceForensics++_C23 --output-dir artifacts --output-name videos_master
python -m preprocessing.analyze_videos_master --manifest artifacts/videos_master.csv

python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split train
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split train
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split train

python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split val
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split val
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split val

python -m preprocessing.frame_extractor --manifest artifacts/videos_master.csv --split test
python -m preprocessing.face_detection --metadata-csv frame_data/frame_extraction_metadata.csv --frame-root frame_data --output-dir crop_data --split test
python -m preprocessing.build_clips --input-dir crop_data --output-dir clip_data --split test
```

Dem clip:

```bash
python -m preprocessing.count_clips --clip-root clip_data --splits train val test
```

Sau khi co `clip_data/train` va `clip_data/val`, co the huan luyen:

```bash
python -m training.train \
  --train-shards "clip_data/train/shard-*.tar" \
  --val-shards "clip_data/val/shard-*.tar"
```
