from __future__ import annotations

import io
import json
import random
import unittest
import warnings
from unittest import mock

import numpy as np
from webdataset import filters, handlers

from dataloader.dataset import ClipDatasetConfig, ClipSampleDecodeError, ClipWebDataset


def _npy_bytes(array: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, array, allow_pickle=False)
    return bio.getvalue()


def _sample_bytes(
    *,
    clip_len: int,
    diff_len: int | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, bytes]:
    if diff_len is None:
        diff_len = clip_len - 1

    meta = {
        "key": "original/124/clip_000000",
        "clip_length": clip_len,
        "num_differences": diff_len,
        "binary_label": 1,
        "default_center_index": min(clip_len - 1, max(0, clip_len // 2)),
    }
    if metadata:
        meta.update(metadata)

    rgb = np.zeros((clip_len, 3, 16, 16), dtype=np.uint8)
    diff = np.zeros((diff_len, 3, 16, 16), dtype=np.uint8)

    return {
        "json": json.dumps(meta).encode("utf-8"),
        "rgb.npy": _npy_bytes(rgb),
        "diff.npy": _npy_bytes(diff),
        "cls": b"1",
    }


def torch_all_between_zero_and_one(array: np.ndarray) -> bool:
    return bool(np.all(array >= 0.0) and np.all(array <= 1.0))


class FakeWebDataset:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.shuffle_calls: list[tuple[int, object]] = []
        self.map_calls: list[tuple[object, object]] = []

    def shuffle(self, size: int, rng=None):
        self.shuffle_calls.append((size, rng))
        return self

    def map(self, fn, handler=None):
        self.map_calls.append((fn, handler))
        return self


class DatasetLoaderTestCase(unittest.TestCase):
    def test_config_derives_diff_len_from_clip_len(self) -> None:
        config = ClipDatasetConfig(shard_pattern="dummy", clip_len=4)
        self.assertEqual(config.diff_len, 3)

    def test_config_rejects_mismatched_diff_len(self) -> None:
        with self.assertRaises(ValueError):
            ClipDatasetConfig(shard_pattern="dummy", clip_len=4, diff_len=7)

    def test_process_sample_accepts_four_frame_clip_without_manual_diff_len(self) -> None:
        dataset = ClipWebDataset(ClipDatasetConfig(shard_pattern="dummy", clip_len=4))
        sample = _sample_bytes(
            clip_len=4,
            metadata={
                "center_candidate_indices": [1, 2],
                "default_center_index": 2,
            },
        )

        output = dataset._process_sample(sample)

        self.assertEqual(tuple(output["temporal"].shape), (3, 3, 16, 16))
        self.assertEqual(float(output["label"]), 1.0)
        self.assertTrue(torch_all_between_zero_and_one(output["temporal"].numpy()))

    def test_training_prefers_center_candidates_from_metadata(self) -> None:
        dataset = ClipWebDataset(
            ClipDatasetConfig(
                shard_pattern="dummy",
                training=True,
                spatial_candidate_indices=(0, 7),
            )
        )
        sample = _sample_bytes(
            clip_len=8,
            metadata={
                "center_candidate_indices": [1],
                "default_center_index": 4,
            },
        )

        with mock.patch("dataloader.dataset.random.choice", side_effect=random.choice) as choice_mock:
            output = dataset._process_sample(sample)

        choice_mock.assert_called_once_with([1])
        self.assertEqual(int(output["spatial_index"]), 1)

    def test_eval_uses_default_center_index_from_metadata(self) -> None:
        dataset = ClipWebDataset(
            ClipDatasetConfig(
                shard_pattern="dummy",
                training=False,
                spatial_candidate_indices=(0, 7),
            )
        )
        sample = _sample_bytes(
            clip_len=8,
            metadata={
                "center_candidate_indices": [1, 2],
                "default_center_index": 2,
            },
        )

        output = dataset._process_sample(sample)

        self.assertEqual(int(output["spatial_index"]), 2)

    def test_legacy_metadata_falls_back_to_config_candidates(self) -> None:
        dataset = ClipWebDataset(
            ClipDatasetConfig(
                shard_pattern="dummy",
                training=True,
                spatial_candidate_indices=(3,),
            )
        )
        sample = _sample_bytes(
            clip_len=8,
            metadata={
                "default_center_index": 4,
            },
        )
        metadata = json.loads(sample["json"].decode("utf-8"))
        metadata.pop("center_candidate_indices", None)
        sample["json"] = json.dumps(metadata).encode("utf-8")

        output = dataset._process_sample(sample)

        self.assertEqual(int(output["spatial_index"]), 3)

    def test_build_dataset_uses_explicit_shardshuffle_and_seed(self) -> None:
        fake_dataset = FakeWebDataset()
        config = ClipDatasetConfig(
            shard_pattern="clip_data/train/shard-*.tar",
            training=True,
            shuffle_buffer=256,
            seed=99,
        )
        dataset = ClipWebDataset(config)

        with mock.patch("dataloader.dataset.wds.WebDataset", return_value=fake_dataset) as webdataset_mock:
            built = dataset.build_dataset()

        self.assertIs(built, fake_dataset)
        webdataset_mock.assert_called_once_with(
            "clip_data/train/shard-*.tar",
            shardshuffle=100,
            seed=99,
        )
        self.assertEqual(len(fake_dataset.shuffle_calls), 1)
        self.assertEqual(fake_dataset.shuffle_calls[0][0], 256)
        self.assertEqual(len(fake_dataset.map_calls), 1)
        self.assertIs(fake_dataset.map_calls[0][1].__self__, dataset)
        self.assertIs(fake_dataset.map_calls[0][1].__func__, dataset._handle_sample_error.__func__)

    def test_process_sample_wraps_error_with_sample_key(self) -> None:
        dataset = ClipWebDataset(ClipDatasetConfig(shard_pattern="dummy"))
        sample = {
            "__key__": "broken/sample",
            "json": b"not-json",
            "rgb.npy": b"bad-rgb",
            "diff.npy": b"bad-diff",
        }

        with self.assertRaises(ClipSampleDecodeError) as ctx:
            dataset._process_sample(sample)

        self.assertIn("broken/sample", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_training_handler_skips_bad_sample_and_keeps_next_valid_sample(self) -> None:
        dataset = ClipWebDataset(ClipDatasetConfig(shard_pattern="dummy", training=True))
        bad_sample = _sample_bytes(clip_len=8)
        bad_sample["__key__"] = "broken/sample"
        bad_sample["diff.npy"] = _npy_bytes(np.zeros((5, 3, 16, 16), dtype=np.uint8))
        good_sample = _sample_bytes(
            clip_len=8,
            metadata={
                "key": "original/124/clip_000001",
                "center_candidate_indices": [2],
                "default_center_index": 2,
            },
        )
        good_sample["__key__"] = "original/124/clip_000001"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs = list(filters._map(iter([bad_sample, good_sample]), dataset._process_sample, handler=dataset._handle_sample_error))

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["__key__"], "original/124/clip_000001")
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("broken/sample", str(caught[0].message))

    def test_eval_raises_on_bad_sample(self) -> None:
        dataset = ClipWebDataset(ClipDatasetConfig(shard_pattern="dummy", training=False))
        bad_sample = _sample_bytes(clip_len=8)
        bad_sample["__key__"] = "broken/sample"
        bad_sample["diff.npy"] = _npy_bytes(np.zeros((5, 3, 16, 16), dtype=np.uint8))

        with self.assertRaises(ClipSampleDecodeError):
            list(filters._map(iter([bad_sample]), dataset._process_sample, handler=handlers.reraise_exception))

    def test_bad_sample_warning_limit_suppresses_extra_logs(self) -> None:
        dataset = ClipWebDataset(
            ClipDatasetConfig(
                shard_pattern="dummy",
                training=True,
                bad_sample_log_limit=1,
            )
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(dataset._handle_sample_error(ClipSampleDecodeError("a", "bad")))
            self.assertTrue(dataset._handle_sample_error(ClipSampleDecodeError("b", "bad")))
            self.assertTrue(dataset._handle_sample_error(ClipSampleDecodeError("c", "bad")))

        self.assertEqual(len(caught), 2)
        self.assertIn("sample_key=a", str(caught[0].message))
        self.assertIn("suppressing further", str(caught[1].message))


if __name__ == "__main__":
    unittest.main()
