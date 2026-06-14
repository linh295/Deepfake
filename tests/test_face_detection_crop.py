from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
    def __init__(self, pattern: str, maxcount: int, maxsize: int, start_shard: int) -> None:
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.start_shard = start_shard
        self.records: list[dict] = []

    def __enter__(self) -> "FakeShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def write(self, sample: dict) -> None:
        self.records.append(sample)


class FaceDetectionCropTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.frame_root = self.root / "frames"
        self.output_dir = self.root / "crop_data"
        self.audit_csv = self.root / "audit" / "face_detection_audit.csv"
        self.metadata_csv = self.root / "frame_metadata.csv"
        self.frame_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_frame(self, name: str, marker: int) -> str:
        image = np.full((96, 96, 3), marker, dtype=np.uint8)
        frame_path = self.frame_root / name
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(frame_path), image)
        self.assertTrue(ok)
        return str(frame_path.relative_to(self.frame_root))

    def _build_config(self) -> fd.FaceDetectionConfig:
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
            detect_every_k=2,
            retinaface_cache_dir=self.root / "retinaface_cache",
        )

    def test_bbox_square_crop_does_not_align_or_rotate(self) -> None:
        bbox = [10, 10, 30, 40]
        crop_box = fd._build_square_crop_box(bbox, image_size=(64, 64), crop_scale=1.3)
        self.assertIsNotNone(crop_box)
        assert crop_box is not None
        self.assertEqual(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :32] = 50
        image[:, 32:] = 200
        cropped, resolved_box, status = fd.crop_face_from_bbox(
            image=image,
            bbox=bbox,
            crop_scale=1.3,
            output_size=(112, 112),
        )
        self.assertEqual(status, "ok")
        self.assertEqual(cropped.shape[:2], (112, 112))
        self.assertEqual(crop_box, resolved_box)
        self.assertLess(float(cropped[:, :56].mean()), float(cropped[:, 56:].mean()))

    def test_process_video_writes_crop_metadata(self) -> None:
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

        config = self._build_config()

        detector = StubDetector({10: _face_result(), 30: _face_result(), 40: {}})
        pipeline = fd.FaceDetectionPipeline(
            config=config,
            detector_module=detector,
            shard_writer_cls=FakeShardWriter,
        )
        sink = FakeShardWriter("unused", 1000, 1000, 0)
        stats = pipeline._process_video(sink=sink, video_id="original/video1", indexed_rows=list(enumerate(rows)))

        self.assertEqual(stats.written_samples, 3)
        payload = json.loads(sink.records[0]["json"].decode("utf-8"))
        self.assertEqual(payload["alignment_mode"], "none_bbox_crop")
        self.assertEqual(payload["crop_source"], "original_bbox")
        self.assertEqual(payload["crop_status"], "ok")
        self.assertTrue(payload["crop_x2"] > payload["crop_x1"])
        self.assertTrue(payload["crop_y2"] > payload["crop_y1"])
        self.assertEqual(payload["aligned_size"], [112, 112])
        self.assertNotIn("affine_matrix", payload)
        self.assertNotIn("align_canvas_size", payload)
        self.assertLessEqual(payload["crop_x2"], 96)
        self.assertLessEqual(payload["crop_y2"], 96)
        self.assertEqual(detector.calls, [10, 30, 40])

    def test_process_video_does_not_require_landmarks(self) -> None:
        row = {
            "frame_path": self._write_frame("original/video1/video1_frame_00000.jpg", 10),
            "video_id": "video1",
            "video_name": "video1",
            "category": "original",
            "label": "real",
            "binary_label": "0",
            "frame_number": "0",
        }
        face = _face_result()
        face["face_1"]["landmarks"] = None
        pipeline = fd.FaceDetectionPipeline(
            config=self._build_config(),
            detector_module=StubDetector({10: face}),
            shard_writer_cls=FakeShardWriter,
        )
        sink = FakeShardWriter("unused", 1000, 1000, 0)

        stats = pipeline._process_video(sink=sink, video_id="original/video1", indexed_rows=[(0, row)])

        self.assertEqual(stats.written_samples, 1)
        payload = json.loads(sink.records[0]["json"].decode("utf-8"))
        self.assertIsNone(payload["landmarks"])
        self.assertEqual(payload["alignment_mode"], "none_bbox_crop")


if __name__ == "__main__":
    unittest.main()
