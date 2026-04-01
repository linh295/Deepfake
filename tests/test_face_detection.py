from __future__ import annotations

import csv
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from preprocessing import face_detection as fd


def _face_result(score: float = 0.99) -> dict:
    return {
        "face_1": {
            "facial_area": [20, 20, 80, 80],
            "score": score,
            "landmarks": {
                "left_eye": [35.0, 40.0],
                "right_eye": [65.0, 40.0],
                "nose": [50.0, 55.0],
                "mouth_left": [38.0, 70.0],
                "mouth_right": [62.0, 70.0],
            },
        }
    }


class StubDetector:
    def __init__(self, responses: dict[int, object]) -> None:
        self.responses = responses
        self.calls: list[int] = []

    def detect_faces(self, image: np.ndarray, threshold: float) -> object:
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        response = self.responses[marker]
        if isinstance(response, Exception):
            raise response
        return response


class FakeShardWriter:
    instances: list["FakeShardWriter"] = []

    def __init__(self, pattern: str, maxcount: int, maxsize: int, start_shard: int) -> None:
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.start_shard = start_shard
        self.records: list[dict] = []
        type(self).instances.append(self)

    def __enter__(self) -> "FakeShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def write(self, sample: dict) -> None:
        self.records.append(sample)


class FaceDetectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.frame_root = self.root / "frames"
        self.output_dir = self.root / "crop_data"
        self.audit_csv = self.root / "audit" / "face_detection_audit.csv"
        self.metadata_csv = self.root / "frame_metadata.csv"
        self.frame_root.mkdir(parents=True, exist_ok=True)
        FakeShardWriter.instances.clear()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_frame(self, name: str, marker: int) -> str:
        image = np.full((96, 96, 3), marker, dtype=np.uint8)
        frame_path = self.frame_root / name
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(frame_path), image)
        self.assertTrue(ok)
        return str(frame_path.relative_to(self.frame_root))

    def _write_metadata_csv(self, rows: list[dict[str, object]]) -> None:
        fieldnames = ["frame_path", "video_id", "video_name", "category", "label", "binary_label", "frame_number"]
        with self.metadata_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _build_config(self, detect_every_k: int = 2) -> fd.FaceDetectionConfig:
        return fd.FaceDetectionConfig(
            metadata_csv=self.metadata_csv,
            frame_root=self.frame_root,
            output_dir=self.output_dir,
            category=None,
            threshold=0.9,
            max_side=640,
            aligned_size=(112, 112),
            crop_scale=1.3,
            image_format=".jpg",
            jpeg_quality=90,
            shard_maxcount=1000,
            shard_maxsize=1024 * 1024,
            limit=None,
            skip_no_face=True,
            audit_csv=self.audit_csv,
            detect_every_k=detect_every_k,
            retinaface_cache_dir=self.root / "retinaface_cache",
        )

    def test_helpers_group_sort_and_sample_key(self) -> None:
        rows = [
            {"category": "original", "video_name": "vid1", "frame_number": "10", "frame_path": "a/f10.jpg"},
            {"category": "original", "video_name": "vid1", "frame_number": "2", "frame_path": "a/f2.jpg"},
        ]
        grouped = fd._group_rows_by_video(rows, category_filter=None)
        self.assertEqual(list(grouped.keys()), ["original/vid1"])
        sorted_rows = fd._sort_video_rows(grouped["original/vid1"])
        self.assertEqual(sorted_rows[0][1]["frame_number"], "2")
        self.assertEqual(fd.build_sample_key(sorted_rows[0][1]), "original/vid1/f2")

    def test_helpers_resume_and_interpolation(self) -> None:
        output_dir = self.root / "shards"
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / "shard-000003.tar"
        with tarfile.open(shard_path, "w") as archive:
            tmp_file = self.root / "tmp.txt"
            tmp_file.write_text("x", encoding="utf-8")
            archive.add(tmp_file, arcname="original/vid1/frame_00001.jpg")
            archive.add(tmp_file, arcname="original/vid1/frame_00001.json")

        self.assertEqual(fd.infer_start_shard(output_dir), 4)
        self.assertEqual(fd.load_processed_keys_from_existing_shards(output_dir), {"original/vid1/frame_00001"})
        self.assertEqual(fd._interpolate_bbox([0, 0, 10, 10], [10, 10, 20, 20], 0.5), [5, 5, 15, 15])
        self.assertEqual(fd._infer_align_canvas_size((112, 112)), (320, 320))
        self.assertEqual(fd._infer_align_canvas_size((224, 160)), (336, 336))

        landmarks = {
            "left_eye": [0.0, 0.0],
            "right_eye": [10.0, 0.0],
            "nose": [5.0, 5.0],
            "mouth_left": [2.0, 10.0],
            "mouth_right": [8.0, 10.0],
        }
        interpolated = fd._interpolate_landmarks(landmarks, landmarks, 0.5)
        self.assertEqual(interpolated["nose"], [5.0, 5.0])

    def test_process_video_interpolates_and_streams_writes(self) -> None:
        rows = [
            {
                "frame_path": self._write_frame("original/video1/video1_frame_00000.jpg", 10),
                "video_id": "video1",
                "video_name": "video1",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
            },
            {
                "frame_path": self._write_frame("original/video1/video1_frame_00001.jpg", 20),
                "video_id": "video1",
                "video_name": "video1",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "1",
            },
            {
                "frame_path": self._write_frame("original/video1/video1_frame_00002.jpg", 30),
                "video_id": "video1",
                "video_name": "video1",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "2",
            },
            {
                "frame_path": self._write_frame("original/video1/video1_frame_00003.jpg", 40),
                "video_id": "video1",
                "video_name": "video1",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "3",
            },
        ]
        detector = StubDetector({10: _face_result(), 30: _face_result(), 40: {}})
        pipeline = fd.FaceDetectionPipeline(
            config=self._build_config(detect_every_k=2),
            detector_module=detector,
            shard_writer_cls=FakeShardWriter,
        )

        sink = FakeShardWriter("unused", 1000, 1000, 0)
        indexed_rows = list(enumerate(rows))
        stats = pipeline._process_video(sink=sink, video_id="original/video1", indexed_rows=indexed_rows)

        self.assertEqual(detector.calls, [10, 30, 40])
        self.assertEqual(stats.interpolated_frames, 1)
        self.assertEqual(stats.detected_frames_direct, 2)
        self.assertEqual(stats.written_samples, 3)
        self.assertEqual(stats.no_face, 1)
        self.assertEqual(len(sink.records), 3)
        self.assertEqual([sample["__key__"] for sample in sink.records], [
            "original/video1/video1_frame_00000",
            "original/video1/video1_frame_00001",
            "original/video1/video1_frame_00002",
        ])

    def test_run_skips_processed_keys_from_audit_and_preserves_cli_surface(self) -> None:
        rows = [
            {
                "frame_path": self._write_frame("original/video2/video2_frame_00000.jpg", 50),
                "video_id": "video2",
                "video_name": "video2",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
            },
            {
                "frame_path": self._write_frame("original/video2/video2_frame_00001.jpg", 60),
                "video_id": "video2",
                "video_name": "video2",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "1",
            },
        ]
        self._write_metadata_csv(rows)
        self.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["key"])
            writer.writeheader()
            writer.writerow({"key": "original/video2/video2_frame_00000"})

        detector = StubDetector({50: _face_result(), 60: _face_result()})
        pipeline = fd.FaceDetectionPipeline(
            config=self._build_config(detect_every_k=2),
            detector_module=detector,
            shard_writer_cls=FakeShardWriter,
        )
        pipeline.run()

        written_keys = [sample["__key__"] for sample in FakeShardWriter.instances[-1].records]
        self.assertEqual(written_keys, ["original/video2/video2_frame_00001"])

        argv = [
            "face_detection.py",
            "--metadata-csv",
            str(self.metadata_csv),
            "--frame-root",
            str(self.frame_root),
            "--output-dir",
            str(self.output_dir),
        ]
        with mock.patch.object(sys, "argv", argv):
            args = fd.parse_args()
        self.assertEqual(Path(args.metadata_csv), self.metadata_csv)
        self.assertEqual(Path(args.frame_root), self.frame_root)
        self.assertEqual(Path(args.output_dir), self.output_dir)


if __name__ == "__main__":
    unittest.main()
