from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from preprocessing import frame_extractor as fe


class FrameExtractorResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = Path("d:/Deepfake/.tmp-tests")
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.root = tmp_root / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.root / "dataset"
        self.output_dir = self.root / "frame_data"
        self.manifest_path = self.root / "artifacts" / "videos_master.csv"
        self.video_name = "video1"
        self.category = "original"
        self.split = "train"
        self.frame_count = 5
        self.video_fps = 5.0
        self.target_fps = 5

        self.video_path = self.dataset_dir / self.category / f"{self.video_name}.avi"
        self._write_video(self.video_path, fps=self.video_fps, num_frames=self.frame_count)
        self._write_manifest()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_video(self, path: Path, fps: float, num_frames: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (24, 24))
        self.assertTrue(writer.isOpened(), "VideoWriter failed to open in test setup")
        for idx in range(num_frames):
            frame = np.full((24, 24, 3), idx * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def _write_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "video_id",
                    "video_path",
                    "category",
                    "binary_label",
                    "split",
                    "original_fps",
                    "num_frames",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "video_id": self.video_name,
                    "video_path": f"{self.category}/{self.video_name}.avi",
                    "category": self.category,
                    "binary_label": 0,
                    "split": self.split,
                    "original_fps": self.video_fps,
                    "num_frames": self.frame_count,
                }
            )

    def _extractor(self, *, resume: bool = True) -> fe.FrameExtractor:
        return fe.FrameExtractor(
            dataset_path=self.dataset_dir,
            output_path=self.output_dir,
            manifest_path=self.manifest_path,
            fps=self.target_fps,
            jpeg_quality=85,
            num_workers=1,
            resume=resume,
        )

    def _write_frame(self, frame_number: int, value: int = 120) -> Path:
        frame_path = self.output_dir / self.category / fe.build_frame_filename(self.video_name, frame_number)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(frame_path), np.full((24, 24, 3), value, dtype=np.uint8))
        self.assertTrue(ok)
        return frame_path

    def _metadata_row(self, frame_number: int) -> dict[str, object]:
        return {
            "frame_path": f"{self.category}/{fe.build_frame_filename(self.video_name, frame_number)}",
            "video_id": self.video_name,
            "video_path": f"{self.category}/{self.video_name}.avi",
            "video_name": self.video_name,
            "category": self.category,
            "label": "real",
            "binary_label": 0,
            "split": self.split,
            "frame_number": frame_number,
            "original_frame_index": frame_number,
            "timestamp": float(frame_number),
            "video_fps": self.video_fps,
            "extraction_fps": self.target_fps,
            "width": 24,
            "height": 24,
            "video_duration": self.frame_count / self.video_fps,
            "total_video_frames": self.frame_count,
            "extraction_date": "2026-01-01 00:00:00",
        }

    def _write_metadata_rows(self, frame_numbers: list[int]) -> None:
        csv_path = self.output_dir / "frame_extraction_metadata.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fe.CSV_FIELDNAMES)
            writer.writeheader()
            for frame_number in frame_numbers:
                writer.writerow(self._metadata_row(frame_number))

    def _write_audit_row(self) -> None:
        audit_path = self.output_dir / "frame_extraction_audit.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fe.AUDIT_FIELDNAMES)
            writer.writeheader()
            writer.writerow(
                {
                    "category": self.category,
                    "video_id": self.video_name,
                    "split": self.split,
                    "status": "complete",
                    "updated_at": "2026-01-01 00:00:00",
                }
            )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_resume_from_partial_frames_without_metadata_duplicates_rows(self) -> None:
        self._write_frame(0, 10)
        self._write_frame(1, 20)

        extractor = self._extractor()
        extractor.extract_all(only_category=self.category, only_split=self.split)

        metadata_rows = self._read_csv(self.output_dir / "frame_extraction_metadata.csv")
        self.assertEqual(len(metadata_rows), self.frame_count)
        self.assertEqual(sorted(int(row["frame_number"]) for row in metadata_rows), list(range(self.frame_count)))
        self.assertEqual(len(list((self.output_dir / self.category).glob("*.jpg"))), self.frame_count)

        audit_rows = self._read_csv(self.output_dir / "frame_extraction_audit.csv")
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["status"], "complete")

    def test_resume_skips_video_marked_complete_in_audit(self) -> None:
        for idx in range(self.frame_count):
            self._write_frame(idx, idx * 10)
        self._write_metadata_rows(list(range(self.frame_count)))
        self._write_audit_row()

        extractor = self._extractor()
        with mock.patch.object(fe, "process_video_standalone", side_effect=AssertionError("should skip")):
            extractor.extract_all(only_category=self.category, only_split=self.split)

        metadata_rows = self._read_csv(self.output_dir / "frame_extraction_metadata.csv")
        self.assertEqual(len(metadata_rows), self.frame_count)

    def test_resume_bootstraps_audit_from_complete_metadata(self) -> None:
        for idx in range(self.frame_count):
            self._write_frame(idx, idx * 10)
        self._write_metadata_rows(list(range(self.frame_count)))

        extractor = self._extractor()
        with mock.patch.object(fe, "process_video_standalone", side_effect=AssertionError("should skip")):
            extractor.extract_all(only_category=self.category, only_split=self.split)

        audit_rows = self._read_csv(self.output_dir / "frame_extraction_audit.csv")
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["video_id"], self.video_name)

    def test_resume_prunes_gapped_metadata_and_rebuilds_video_rows(self) -> None:
        self._write_frame(0, 10)
        self._write_frame(1, 20)
        self._write_frame(3, 40)
        self._write_metadata_rows([0, 1, 3])

        extractor = self._extractor()
        extractor.extract_all(only_category=self.category, only_split=self.split)

        metadata_rows = self._read_csv(self.output_dir / "frame_extraction_metadata.csv")
        self.assertEqual(len(metadata_rows), self.frame_count)
        self.assertEqual(sorted(int(row["frame_number"]) for row in metadata_rows), list(range(self.frame_count)))
        self.assertEqual(len({int(row["frame_number"]) for row in metadata_rows}), self.frame_count)

    def test_no_resume_ignores_existing_audit_and_restarts_clean(self) -> None:
        self._write_frame(0, 10)
        self._write_frame(1, 20)
        self._write_metadata_rows([0, 1])
        self._write_audit_row()

        extractor = self._extractor(resume=False)
        extractor.extract_all(only_category=self.category, only_split=self.split)

        metadata_rows = self._read_csv(self.output_dir / "frame_extraction_metadata.csv")
        self.assertEqual(len(metadata_rows), self.frame_count)
        self.assertEqual(sorted(int(row["frame_number"]) for row in metadata_rows), list(range(self.frame_count)))

        audit_rows = self._read_csv(self.output_dir / "frame_extraction_audit.csv")
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
