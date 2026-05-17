# Thay Doi Tai Lieu

## Muc Dich

File nay ghi lai nhung diem da duoc dong bo trong lan cap nhat tai lieu hien tai. Pham vi gom:

- [README.md](../README.md)
- [summary_documentation.md](summary_documentation.md)
- [architecture.md](architecture.md)
- [preprocessing-pipeline.md](preprocessing-pipeline.md)
- [training-workflow.md](training-workflow.md)
- [data-contracts.md](data-contracts.md)

## Diem Cap Nhat Chinh

Tai lieu da duoc cap nhat de khop voi code hien tai thay vi mo ta pipeline cu o muc tong quan.

Nhung thay doi quan trong:

- Bo sung `docs/summary_documentation.md`, vi IDE dang mo file nay nhung repo chua co file Markdown tuong ung.
- Lam ro quy uoc label hien tai: `original=1`, manipulation `=0`.
- Them canh bao dung `--invert-binary-labels` neu can dao quy uoc label khi train/test.
- Cap nhat temporal branch: `--temporal-pool` ho tro `mean`, `attention`, `gru`; CLI train mac dinh `gru`.
- Bo sung augmentation moi: JPEG compression va Gaussian blur.
- Bo sung workflow danh gia bang `training.test` va `training.test_with_best_threshold`.
- Bo sung output `test_metrics.json` va `test_predictions.csv`.
- Cap nhat data contract cho frame audit, clip shard, batch, history, checkpoint va test output.
- Cap nhat preprocessing docs theo CLI hien tai cua `frame_extractor`, `face_detection`, `build_clips`, `analyze_videos_master`, `count_clips`.

## Theo File

### `README.md`

Da viet lai thanh quickstart end-to-end:

- cau truc repo
- cai dat
- label convention
- tien xu ly 4 buoc
- huan luyen
- danh gia
- test
- lien ket tai lieu

### `summary_documentation.md`

File moi, dung lam muc luc va ban tom tat:

- pipeline tong the
- lenh chay chinh
- output moi stage
- cac rui ro/can chu y khi van hanh

### `architecture.md`

Da cap nhat:

- luong xu ly tu video den test report
- vai tro tung module
- thiet ke face detection hien tai
- temporal branch GRU
- checkpoint selection, early stopping va threshold evaluation

### `preprocessing-pipeline.md`

Da cap nhat:

- CLI argument thuc te cua cac stage preprocessing
- resume behavior cua frame extractor va face detection
- output theo split
- contract cua clip builder
- lenh mau day du cho train/val/test

### `training-workflow.md`

Da cap nhat:

- tat ca nhom argument training quan trong
- augmentation JPEG/blur
- label inversion
- temporal pooling `gru`
- checkpoint/history/figure output
- danh gia test va best-threshold workflow

### `data-contracts.md`

Da cap nhat:

- manifest video
- frame metadata va audit
- face frame shard
- clip shard
- dataloader batch
- `history.json`
- checkpoint `.pt`
- test output

## Ghi Chu Con Lai

Co mot diem code can luu y: `metadata_level.py` gan `original=1`, nhung mot so fallback/helper va truong text `label` trong pipeline van dung convention `real=0/fake=1`. Tai lieu hien tai xem `binary_label` trong manifest/shard la nguon su that va khuyen nghi dung `--invert-binary-labels` khi can dao quy uoc.
