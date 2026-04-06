# Đóng Góp

## Thiết Lập Môi Trường Phát Triển

Cài dự án ở chế độ editable:

```bash
pip install -e .
```

Kho mã sử dụng Python `>=3.12` và lưu các runtime setting trong [configs/settings.py](configs/settings.py).

## Nguyên Tắc Làm Việc

- Giữ các bước preprocessing có thể chạy được từ command line.
- Ưu tiên cách tổ chức theo class cho các stage lớn.
- Giữ tương thích ngược cho CLI flag và shard schema hiện có, trừ khi có migration rõ ràng.
- Không thay đổi ngầm ý nghĩa của các trường metadata.

## Kiểm Thử

Chạy các test trọng tâm trước khi mở một thay đổi:

```bash
python -m unittest tests.test_face_detection tests.test_face_detection_split tests.test_face_detection_crop
python -m unittest tests.test_dataset_loader tests.test_spatio_temporal_detector tests.test_training_train
python -m compileall preprocessing training dataloader tests
```

Nếu thay đổi ảnh hưởng đến stage nào, hãy thêm hoặc cập nhật test trong `tests/` cho stage đó trong cùng một change.

## Tài Liệu

Cập nhật tài liệu liên quan khi bạn thay đổi:

- CLI flag
- metadata schema
- nội dung shard
- thứ tự stage
- vị trí output mặc định

Những điểm vào tài liệu chính:

- [README.md](README.md)
- [docs/preprocessing-pipeline.md](docs/preprocessing-pipeline.md)
- [docs/training-workflow.md](docs/training-workflow.md)
- [docs/data-contracts.md](docs/data-contracts.md)
- [docs/architecture.md](docs/architecture.md)

## Hướng Dẫn Commit

Sử dụng commit message ngắn gọn, mô tả rõ phạm vi, ví dụ:

```text
feat(preprocessing): separate align canvas from final face crop size
fix(face-detection): isolate split-specific audit and shard output
docs: refresh preprocessing pipeline documentation
```
