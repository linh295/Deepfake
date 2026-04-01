from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from preprocessing import face_detection as fd


def _face_result() -> dict:
    return {
        "face_1": {
            "facial_area": [20, 20, 80, 80],
            "score": 0.99,
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


class FaceDetectionSplitTestCase(unittest.TestCase):
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
        fieldnames = [
            "frame_path",
            "video_id",
            "video_name",
            "category",
            "label",
            "binary_label",
            "frame_number",
            "split",
        ]
        with self.metadata_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _build_config(self, split: str | None) -> fd.FaceDetectionConfig:
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
            detect_every_k=1,
            retinaface_cache_dir=self.root / "retinaface_cache",
            split=split,
        )

    def test_split_filter_and_paths_are_isolated(self) -> None:
        rows = [
            {
                "frame_path": self._write_frame("original/video3/video3_frame_00000.jpg", 70),
                "video_id": "video3",
                "video_name": "video3",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
                "split": "train",
            },
            {
                "frame_path": self._write_frame("original/video4/video4_frame_00000.jpg", 80),
                "video_id": "video4",
                "video_name": "video4",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
                "split": "test",
            },
        ]
        self._write_metadata_csv(rows)

        pipeline = fd.FaceDetectionPipeline(
            config=self._build_config(split="train"),
            detector_module=StubDetector({70: _face_result(), 80: _face_result()}),
            shard_writer_cls=FakeShardWriter,
        )

        loaded_rows = pipeline._load_metadata_rows()
        self.assertEqual(len(loaded_rows), 1)
        self.assertEqual(loaded_rows[0]["video_id"], "video3")

        pipeline.run()

        self.assertEqual(pipeline._effective_output_dir(), self.output_dir / "train")
        self.assertEqual(
            pipeline._effective_audit_csv(),
            self.audit_csv.parent / "train" / self.audit_csv.name,
        )
        self.assertTrue(FakeShardWriter.instances[-1].pattern.endswith("crop_data\\train\\shard-%06d.tar"))
        self.assertEqual(
            [sample["__key__"] for sample in FakeShardWriter.instances[-1].records],
            ["original/video3/video3_frame_00000"],
        )
        payload = json.loads(FakeShardWriter.instances[-1].records[0]["json"].decode("utf-8"))
        self.assertEqual(payload["split"], "train")

    def test_split_resume_uses_split_specific_audit(self) -> None:
        rows = [
            {
                "frame_path": self._write_frame("original/video5/video5_frame_00000.jpg", 90),
                "video_id": "video5",
                "video_name": "video5",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
                "split": "train",
            },
            {
                "frame_path": self._write_frame("original/video6/video6_frame_00000.jpg", 100),
                "video_id": "video6",
                "video_name": "video6",
                "category": "original",
                "label": "real",
                "binary_label": "0",
                "frame_number": "0",
                "split": "val",
            },
        ]
        self._write_metadata_csv(rows)
        train_audit = self.audit_csv.parent / "train" / self.audit_csv.name
        train_audit.parent.mkdir(parents=True, exist_ok=True)
        with train_audit.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["key"])
            writer.writeheader()
            writer.writerow({"key": "original/video5/video5_frame_00000"})

        pipeline = fd.FaceDetectionPipeline(
            config=self._build_config(split="train"),
            detector_module=StubDetector({90: _face_result(), 100: _face_result()}),
            shard_writer_cls=FakeShardWriter,
        )
        pipeline.run()

        self.assertEqual(FakeShardWriter.instances[-1].records, [])

    def test_cli_parse_supports_split(self) -> None:
        argv = [
            "face_detection.py",
            "--metadata-csv",
            str(self.metadata_csv),
            "--frame-root",
            str(self.frame_root),
            "--output-dir",
            str(self.output_dir),
            "--split",
            "train",
        ]
        with mock.patch("sys.argv", argv):
            args = fd.parse_args()
        self.assertEqual(args.split, "train")


if __name__ == "__main__":
    unittest.main()
