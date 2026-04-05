from __future__ import annotations

import io
import json
import shutil
import tarfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from preprocessing import build_clips as bc


class FakeShardWriter:
    instances: list["FakeShardWriter"] = []

    def __init__(self, pattern: str, maxcount: int, maxsize: int) -> None:
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.records: list[dict[str, object]] = []
        type(self).instances.append(self)

    def __enter__(self) -> "FakeShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def write(self, sample: dict[str, object]) -> None:
        self.records.append(sample)


class BuildClipsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = Path("d:/Deepfake/.tmp-tests")
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.root = tmp_root / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_dir = self.root / "crop_data" / "train"
        self.output_dir = self.root / "clip_data"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        FakeShardWriter.instances.clear()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _frame_image(self, marker: int) -> bytes:
        image = np.full((24, 24, 3), marker, dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        return buffer.tobytes()

    def _sample(self, *, key: str, video_id: str, video_name: str, frame_number: int, marker: int) -> dict[str, object]:
        category = video_id.split("/", 1)[0] if "/" in video_id else "original"
        meta = {
            "key": key,
            "split": "train",
            "category": category,
            "video_id": video_id,
            "video_name": video_name,
            "label": "real",
            "binary_label": 0,
            "frame_number": frame_number,
            "original_frame_index": frame_number,
            "timestamp": float(frame_number),
            "video_fps": 5.0,
            "extraction_fps": 5.0,
        }
        return {
            "__key__": key,
            "json": json.dumps(meta).encode("utf-8"),
            "jpg": self._frame_image(marker),
            "cls": b"0",
        }

    def _write_shard(self, name: str, samples: list[dict[str, object]]) -> None:
        shard_name = name % self.shard_index if "%" in name else name
        self.shard_index += 1
        shard_path = self.input_dir / shard_name

        with tarfile.open(shard_path, "w") as archive:
            for sample in samples:
                key = str(sample["__key__"])
                for suffix, value in sample.items():
                    if suffix == "__key__":
                        continue

                    data = bytes(value)
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))

    def test_build_clips_uses_canonical_video_id_for_group_and_key(self) -> None:
        samples = [
            self._sample(key=f"original/124/124_frame_{idx:05d}", video_id="original/124", video_name="124", frame_number=idx, marker=10 + idx)
            for idx in range(4)
        ]
        frames = list(bc.iter_frame_samples_from_shards(self.input_dir, fallback_split="train"))
        self.assertEqual(frames, [])
        self._write_shard("shard-%06d.tar", samples)

        frames = list(bc.iter_frame_samples_from_shards(self.input_dir, fallback_split="train"))
        clips = bc.build_clips_for_video(frames, clip_len=4, frame_stride=1, clip_stride=1)

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["__key__"], "original/124/clip_000000")
        meta = json.loads(clips[0]["json"].decode("utf-8"))
        self.assertEqual(meta["video_id"], "original/124")
        self.assertEqual(meta["video_name"], "124")

    def test_process_split_does_not_mix_same_video_name_with_different_video_ids(self) -> None:
        samples = [
            self._sample(key=f"original/124/124_frame_{idx:05d}", video_id="original/124", video_name="shared", frame_number=idx, marker=10 + idx)
            for idx in range(4)
        ]
        samples += [
            self._sample(key=f"original_alt/124/124_frame_{idx:05d}", video_id="original_alt/124", video_name="shared", frame_number=idx, marker=20 + idx)
            for idx in range(4)
        ]
        self._write_shard("shard-%06d.tar", samples)

        split_output_dir = self.output_dir / "train"
        with mock.patch.object(bc, "ShardWriter", FakeShardWriter):
            bc.process_split(
                split="train",
                split_input_dir=self.input_dir,
                split_output_dir=split_output_dir,
                shard_maxcount=1000,
                shard_maxsize=10_000_000,
                clip_len=4,
                frame_stride=1,
                clip_stride=1,
                overwrite=False,
            )

        written_keys = [record["__key__"] for record in FakeShardWriter.instances[-1].records]
        self.assertEqual(written_keys, ["original/124/clip_000000", "original_alt/124/clip_000000"])

    def test_process_split_merges_same_video_across_shard_boundary_when_contiguous(self) -> None:
        shard1 = [
            self._sample(key="original/124/124_frame_00000", video_id="original/124", video_name="124", frame_number=0, marker=10),
            self._sample(key="original/124/124_frame_00001", video_id="original/124", video_name="124", frame_number=1, marker=11),
        ]
        shard2 = [
            self._sample(key="original/124/124_frame_00002", video_id="original/124", video_name="124", frame_number=2, marker=12),
            self._sample(key="original/124/124_frame_00003", video_id="original/124", video_name="124", frame_number=3, marker=13),
        ]
        self._write_shard("shard-%06d.tar", shard1)
        self._write_shard("shard-%06d.tar", shard2)

        split_output_dir = self.output_dir / "train"
        with mock.patch.object(bc, "ShardWriter", FakeShardWriter):
            bc.process_split(
                split="train",
                split_input_dir=self.input_dir,
                split_output_dir=split_output_dir,
                shard_maxcount=1000,
                shard_maxsize=10_000_000,
                clip_len=4,
                frame_stride=1,
                clip_stride=1,
                overwrite=False,
            )

        self.assertEqual(len(FakeShardWriter.instances[-1].records), 1)
        meta = json.loads(FakeShardWriter.instances[-1].records[0]["json"].decode("utf-8"))
        self.assertEqual(meta["frame_numbers"], [0, 1, 2, 3])

    def test_process_split_raises_when_video_reappears_after_flush(self) -> None:
        samples = [
            self._sample(key="original/124/124_frame_00000", video_id="original/124", video_name="124", frame_number=0, marker=10),
            self._sample(key="original/999/999_frame_00000", video_id="original/999", video_name="999", frame_number=0, marker=20),
            self._sample(key="original/124/124_frame_00001", video_id="original/124", video_name="124", frame_number=1, marker=11),
            self._sample(key="original/124/124_frame_00002", video_id="original/124", video_name="124", frame_number=2, marker=12),
            self._sample(key="original/124/124_frame_00003", video_id="original/124", video_name="124", frame_number=3, marker=13),
        ]
        self._write_shard("shard-%06d.tar", samples)

        with mock.patch.object(bc, "ShardWriter", FakeShardWriter):
            with self.assertRaises(RuntimeError):
                bc.process_split(
                    split="train",
                    split_input_dir=self.input_dir,
                    split_output_dir=self.output_dir / "train",
                    shard_maxcount=1000,
                    shard_maxsize=10_000_000,
                    clip_len=4,
                    frame_stride=1,
                    clip_stride=1,
                    overwrite=False,
                )

    def test_continuity_filter_still_drops_invalid_gap(self) -> None:
        frames = [
            bc.FrameSample(
                key=f"original/124/124_frame_{frame_number:05d}",
                split="train",
                category="original",
                video_id="original/124",
                video_name="124",
                frame_number=frame_number,
                original_frame_index=frame_number,
                timestamp=float(frame_number),
                binary_label=0,
                extraction_fps=5.0,
                video_fps=5.0,
                image_rgb=np.zeros((24, 24, 3), dtype=np.uint8),
                metadata={},
            )
            for frame_number in (0, 1, 3, 4)
        ]
        clips = bc.build_clips_for_video(frames, clip_len=4, frame_stride=1, clip_stride=1)
        self.assertEqual(clips, [])

    def test_process_split_fails_if_output_has_existing_shards_without_overwrite(self) -> None:
        samples = [
            self._sample(key=f"original/124/124_frame_{idx:05d}", video_id="original/124", video_name="124", frame_number=idx, marker=10 + idx)
            for idx in range(4)
        ]
        self._write_shard("shard-%06d.tar", samples)
        split_output_dir = self.output_dir / "train"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        (split_output_dir / "shard-000000.tar").write_bytes(b"old")

        with self.assertRaises(RuntimeError):
            bc.process_split(
                split="train",
                split_input_dir=self.input_dir,
                split_output_dir=split_output_dir,
                shard_maxcount=1000,
                shard_maxsize=10_000_000,
                clip_len=4,
                frame_stride=1,
                clip_stride=1,
                overwrite=False,
            )

    def test_process_split_overwrite_rebuilds_clean(self) -> None:
        samples = [
            self._sample(key=f"original/124/124_frame_{idx:05d}", video_id="original/124", video_name="124", frame_number=idx, marker=10 + idx)
            for idx in range(4)
        ]
        self._write_shard("shard-%06d.tar", samples)
        split_output_dir = self.output_dir / "train"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        stale = split_output_dir / "shard-999999.tar"
        stale.write_bytes(b"old")

        with mock.patch.object(bc, "ShardWriter", FakeShardWriter):
            bc.process_split(
                split="train",
                split_input_dir=self.input_dir,
                split_output_dir=split_output_dir,
                shard_maxcount=1000,
                shard_maxsize=10_000_000,
                clip_len=4,
                frame_stride=1,
                clip_stride=1,
                overwrite=True,
            )

        self.assertFalse(stale.exists())
        self.assertEqual(len(FakeShardWriter.instances[-1].records), 1)


if __name__ == "__main__":
    unittest.main()
