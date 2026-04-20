from __future__ import annotations

import io
import importlib.util
import json
import random
import shutil
import tarfile
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.spatio_temporal_detector import ModelConfig
from training.train import TrainConfig, main
from training.utils.builders import build_dataloaders, build_loss, build_model, build_scheduler
from training.utils.checkpointing import load_checkpoint, save_checkpoint
from training.utils.class_balance import ClassBalanceInfo, build_class_balance_info
from training.utils.figures import render_training_figures, resolve_figure_output_dirs
from training.utils.loops import train_one_epoch, validate_one_epoch
from training.utils.metrics import ValidationDiagnostics, select_checkpoint_metric
from training.utils.progress import build_progress_totals, count_samples_in_shards, estimate_total_batches
from training.utils.runtime import restore_rng_state, set_seed


class _FakeTqdm:
    instances: list["_FakeTqdm"] = []

    def __init__(
        self,
        iterable,
        desc: str | None = None,
        total: int | None = None,
        dynamic_ncols: bool | None = None,
        bar_format: str | None = None,
    ) -> None:
        self._items = list(iterable)
        self.desc = desc
        self.total = total
        self.dynamic_ncols = dynamic_ncols
        self.bar_format = bar_format
        self.postfix_history: list[dict[str, object]] = []
        type(self).instances.append(self)

    def __iter__(self):
        return iter(self._items)

    def set_postfix(self, **kwargs) -> None:
        self.postfix_history.append(kwargs)
        return None


class _LoaderWithoutLen:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(spatial.shape[0])


def _batch(batch_size: int = 2, temporal_frames: int = 7) -> dict[str, object]:
    return {
        "spatial": torch.randn(batch_size, 3, 16, 16),
        "temporal": torch.randn(batch_size, temporal_frames, 3, 16, 16),
        "label": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "spatial_index": torch.zeros(batch_size, dtype=torch.long),
        "meta": [{"key": f"sample_{idx}"} for idx in range(batch_size)],
    }


class TrainModuleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="deepfake-tests-")
        self.root = Path(self._tmpdir.name) / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        _FakeTqdm.instances.clear()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._tmpdir.cleanup()

    def _write_progress_shard(self, shard_path: Path, sample_count: int) -> None:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(shard_path, "w") as archive:
            for idx in range(sample_count):
                key = f"video/sample_{idx:06d}"
                for suffix, payload in (("json", b"{}"), ("cls", b"0")):
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

    def _write_label_shard(self, shard_path: Path, labels: list[int]) -> None:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(shard_path, "w") as archive:
            for idx, label in enumerate(labels):
                key = f"video/sample_{idx:06d}"
                metadata = {
                    "key": key,
                    "binary_label": label,
                    "label": "fake" if label == 1 else "real",
                }
                json_payload = json.dumps(metadata).encode("utf-8")
                cls_payload = str(label).encode("utf-8")
                for suffix, payload in (("json", json_payload), ("cls", cls_payload)):
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

    def _write_clip_label_shard(self, shard_path: Path, labels: list[int]) -> None:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_payload = io.BytesIO()
        np.save(rgb_payload, np.zeros((8, 3, 4, 4), dtype=np.uint8), allow_pickle=False)
        diff_payload = io.BytesIO()
        np.save(diff_payload, np.zeros((7, 3, 4, 4), dtype=np.uint8), allow_pickle=False)

        with tarfile.open(shard_path, "w") as archive:
            for idx, label in enumerate(labels):
                key = f"video/sample_{idx:06d}"
                metadata = {
                    "key": key,
                    "binary_label": label,
                    "label": "fake" if label == 1 else "real",
                    "clip_length": 8,
                    "num_differences": 7,
                }
                parts = {
                    "json": json.dumps(metadata).encode("utf-8"),
                    "cls": str(label).encode("utf-8"),
                    "rgb.npy": rgb_payload.getvalue(),
                    "diff.npy": diff_payload.getvalue(),
                }
                for suffix, payload in parts.items():
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

    def test_build_model_derives_temporal_num_frames_from_clip_len(self) -> None:
        captured: dict[str, object] = {}

        class _FakeDetector(nn.Module):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__()
                captured["config"] = config

            def to(self, device: torch.device):
                captured["device"] = device
                return self

        cfg = TrainConfig(train_shards="train", val_shards="val", clip_len=10)
        with mock.patch("training.utils.builders.SpatioTemporalDeepfakeDetector", _FakeDetector):
            model, model_cfg = build_model(cfg, torch.device("cpu"))

        self.assertIsInstance(model_cfg, ModelConfig)
        self.assertEqual(model_cfg.temporal_num_frames, 9)
        self.assertEqual(captured["config"].temporal_num_frames, 9)
        self.assertEqual(captured["device"], torch.device("cpu"))

    def test_build_dataloaders_propagates_invert_binary_labels(self) -> None:
        cfg = TrainConfig(
            train_shards="train",
            val_shards="val",
            invert_binary_labels=True,
        )

        with mock.patch("training.utils.builders.build_clip_dataloader", side_effect=lambda config: config) as build_loader_mock:
            train_cfg, val_cfg = build_dataloaders(cfg)

        self.assertEqual(build_loader_mock.call_count, 2)
        self.assertTrue(train_cfg.invert_binary_labels)
        self.assertTrue(val_cfg.invert_binary_labels)

    def test_build_loss_uses_effective_pos_weight_on_device(self) -> None:
        cfg = TrainConfig(train_shards="train", val_shards="val")
        balance = ClassBalanceInfo(
            negative_count=30,
            positive_count=10,
            positive_class_name="fake",
            raw_pos_weight=3.0,
            effective_pos_weight=3.0,
        )

        criterion = build_loss(cfg, torch.device("cpu"), balance)

        self.assertIsInstance(criterion, nn.BCEWithLogitsLoss)
        self.assertIsNotNone(criterion.pos_weight)
        self.assertAlmostEqual(float(criterion.pos_weight.item()), 3.0)

    def test_build_loss_falls_back_to_unweighted_when_pos_weight_disabled(self) -> None:
        cfg = TrainConfig(train_shards="train", val_shards="val", use_pos_weight=False)
        balance = ClassBalanceInfo(
            negative_count=30,
            positive_count=10,
            positive_class_name="fake",
            raw_pos_weight=3.0,
            effective_pos_weight=3.0,
        )

        criterion = build_loss(cfg, torch.device("cpu"), balance)

        self.assertIsNone(criterion.pos_weight)

    def test_class_balance_counts_labels_and_inverts_effective_positive_class(self) -> None:
        train_dir = self.root / "clip_data" / "train"
        shard = train_dir / "shard-000000.tar"
        self._write_label_shard(shard, labels=[1, 0, 0, 0])
        pattern = str(train_dir / "shard-*.tar")

        fake_positive = build_class_balance_info(
            shard_pattern=pattern,
            invert_binary_labels=False,
        )
        real_positive = build_class_balance_info(
            shard_pattern=pattern,
            invert_binary_labels=True,
        )

        self.assertEqual(fake_positive.positive_class_name, "fake")
        self.assertEqual(fake_positive.positive_count, 1)
        self.assertEqual(fake_positive.negative_count, 3)
        self.assertAlmostEqual(float(fake_positive.effective_pos_weight), 3.0)

        self.assertEqual(real_positive.positive_class_name, "real")
        self.assertEqual(real_positive.positive_count, 3)
        self.assertEqual(real_positive.negative_count, 1)
        self.assertAlmostEqual(float(real_positive.effective_pos_weight), 1.0 / 3.0)

    def test_class_balance_clamps_pos_weight_and_handles_missing_positive(self) -> None:
        train_dir = self.root / "clip_data" / "train"
        shard = train_dir / "shard-000000.tar"
        self._write_label_shard(shard, labels=[1, 0, 0, 0, 0, 0])
        pattern = str(train_dir / "shard-*.tar")

        clamped = build_class_balance_info(
            shard_pattern=pattern,
            invert_binary_labels=False,
            max_pos_weight=2.0,
        )
        self.assertAlmostEqual(float(clamped.raw_pos_weight), 5.0)
        self.assertAlmostEqual(float(clamped.effective_pos_weight), 2.0)

        all_positive_shard = train_dir / "shard-000001.tar"
        self._write_label_shard(all_positive_shard, labels=[1, 1])
        all_positive_pattern = str(all_positive_shard)
        fallback = build_class_balance_info(
            shard_pattern=all_positive_pattern,
            invert_binary_labels=False,
        )
        self.assertEqual(fallback.negative_count, 0)
        self.assertEqual(fallback.positive_count, 2)
        self.assertIsNone(fallback.raw_pos_weight)
        self.assertIsNone(fallback.effective_pos_weight)

    def test_class_balance_supports_clip_shards_with_multi_part_npy_suffixes(self) -> None:
        train_dir = self.root / "clip_data" / "train"
        shard = train_dir / "shard-000000.tar"
        self._write_clip_label_shard(shard, labels=[1, 0, 1])

        balance = build_class_balance_info(
            shard_pattern=str(shard),
            invert_binary_labels=False,
        )

        self.assertEqual(balance.positive_count, 2)
        self.assertEqual(balance.negative_count, 1)
        self.assertAlmostEqual(float(balance.effective_pos_weight), 0.5)

    def test_train_one_epoch_works_without_loader_len(self) -> None:
        model = _TinyModel()
        loader = _LoaderWithoutLen([_batch(), _batch()])
        optimizer = AdamW(model.parameters(), lr=0.0)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=False)
        cfg = TrainConfig(train_shards="train", val_shards="val", log_every=100, use_amp=False)

        with mock.patch("training.utils.loops.tqdm", _FakeTqdm):
            metrics = train_one_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=torch.device("cpu"),
                epoch=1,
                cfg=cfg,
                total_batches=5,
                stage_label="Train 1/3",
            )

        self.assertIn("loss", metrics)
        self.assertAlmostEqual(metrics["loss"], 0.693147, places=4)
        self.assertEqual(_FakeTqdm.instances[0].total, 5)
        self.assertEqual(_FakeTqdm.instances[0].desc, "Train 1/3")
        self.assertEqual(_FakeTqdm.instances[0].postfix_history[0]["lr"], "0.00e+00")

    def test_validate_one_epoch_works_without_loader_len(self) -> None:
        model = _TinyModel()
        loader = _LoaderWithoutLen([_batch(), _batch()])
        criterion = nn.BCEWithLogitsLoss()
        cfg = TrainConfig(train_shards="train", val_shards="val", use_amp=False)

        with mock.patch("training.utils.loops.tqdm", _FakeTqdm):
            metrics, diagnostics = validate_one_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                device=torch.device("cpu"),
                epoch=1,
                cfg=cfg,
                total_batches=4,
                stage_label="Val 1/3",
            )

        self.assertIn("loss", metrics)
        self.assertAlmostEqual(metrics["loss"], 0.693147, places=4)
        self.assertIsInstance(diagnostics, ValidationDiagnostics)
        self.assertEqual(diagnostics.labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(diagnostics.preds.tolist(), [1, 1, 1, 1])
        self.assertEqual(_FakeTqdm.instances[0].total, 4)
        self.assertEqual(_FakeTqdm.instances[0].desc, "Val 1/3")

    def test_train_one_epoch_raises_on_empty_loader(self) -> None:
        model = _TinyModel()
        loader = _LoaderWithoutLen([])
        optimizer = AdamW(model.parameters(), lr=0.0)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=False)
        cfg = TrainConfig(train_shards="train", val_shards="val", use_amp=False)

        with mock.patch("training.utils.loops.tqdm", _FakeTqdm):
            with self.assertRaises(RuntimeError) as ctx:
                train_one_epoch(
                    model=model,
                    loader=loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=torch.device("cpu"),
                    epoch=1,
                    cfg=cfg,
                    total_batches=1,
                )

        self.assertIn("No valid training batches", str(ctx.exception))

    def test_validate_one_epoch_raises_on_empty_loader(self) -> None:
        model = _TinyModel()
        loader = _LoaderWithoutLen([])
        criterion = nn.BCEWithLogitsLoss()
        cfg = TrainConfig(train_shards="train", val_shards="val", use_amp=False)

        with mock.patch("training.utils.loops.tqdm", _FakeTqdm):
            with self.assertRaises(RuntimeError) as ctx:
                validate_one_epoch(
                    model=model,
                    loader=loader,
                    criterion=criterion,
                    device=torch.device("cpu"),
                    epoch=1,
                    cfg=cfg,
                    total_batches=1,
                )

        self.assertIn("No valid validation batches", str(ctx.exception))

    def test_select_checkpoint_metric_prefers_auc_and_falls_back_to_loss(self) -> None:
        auc_score, auc_name = select_checkpoint_metric({"auc": 0.81, "loss": 0.42})
        loss_score, loss_name = select_checkpoint_metric({"auc": float("nan"), "loss": 0.42})

        self.assertEqual(auc_name, "val_auc")
        self.assertEqual(auc_score, 0.81)
        self.assertEqual(loss_name, "neg_val_loss")
        self.assertEqual(loss_score, -0.42)

    def test_build_scheduler_uses_reduce_on_plateau_with_selection_metric_mode(self) -> None:
        model = nn.Linear(4, 1)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        cfg = TrainConfig(
            train_shards="train",
            val_shards="val",
            scheduler_factor=0.5,
            scheduler_patience=3,
            scheduler_threshold=1e-3,
            scheduler_min_lr=1e-6,
        )

        scheduler = build_scheduler(optimizer, cfg)

        self.assertIsInstance(scheduler, ReduceLROnPlateau)
        self.assertEqual(scheduler.mode, "max")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 3)
        self.assertEqual(scheduler.threshold, 1e-3)
        self.assertEqual(scheduler.min_lrs, [1e-6])

    def test_set_seed_and_restore_rng_state_keep_random_stream_reproducible(self) -> None:
        set_seed(123)
        original_state = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.random.get_rng_state(),
        }
        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = torch.rand(3)

        set_seed(999)
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(3)
        restore_rng_state(original_state)

        self.assertEqual(expected_python, random.random())
        self.assertEqual(expected_numpy, float(np.random.rand()))
        self.assertTrue(torch.allclose(expected_torch, torch.rand(3)))

    def test_progress_utils_count_samples_and_estimate_batches_from_glob_pattern(self) -> None:
        train_dir = self.root / "clip_data" / "train"
        shard_a = train_dir / "shard-000000.tar"
        shard_b = train_dir / "shard-000001.tar"
        self._write_progress_shard(shard_a, sample_count=3)
        self._write_progress_shard(shard_b, sample_count=2)

        pattern = str(train_dir / "shard-*.tar")

        self.assertEqual(count_samples_in_shards(pattern), 5)
        self.assertEqual(estimate_total_batches(shard_pattern=pattern, batch_size=2), 3)

        totals = build_progress_totals(
            train_shards=pattern,
            val_shards=pattern,
            batch_size=2,
        )
        self.assertEqual(totals["train"], 3)
        self.assertEqual(totals["val"], 3)

    def test_resolve_figure_output_dirs_uses_run_name_under_figure_root(self) -> None:
        figure_root = self.root / "figures"
        dirs = resolve_figure_output_dirs(
            output_dir=self.root / "artifacts" / "experiments" / "st_detector",
            figure_root=figure_root,
        )

        self.assertEqual(dirs.run_dir, figure_root / "st_detector")
        self.assertEqual(dirs.latest_dir, figure_root / "st_detector" / "latest")
        self.assertEqual(dirs.best_root_dir, figure_root / "st_detector" / "best")
        self.assertTrue(dirs.latest_dir.exists())
        self.assertTrue(dirs.best_root_dir.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
    def test_render_training_figures_writes_png_and_svg_outputs(self) -> None:
        history = {
            "run": {"class_balance": None, "use_pos_weight": True, "auto_pos_weight": True},
            "train": [
                {"epoch": 1, "loss": 0.6, "auc": 0.7, "accuracy": 0.65, "f1": 0.62},
                {"epoch": 2, "loss": 0.5, "auc": 0.76, "accuracy": 0.72, "f1": 0.7},
            ],
            "val": [
                {
                    "epoch": 1,
                    "loss": 0.55,
                    "auc": 0.74,
                    "accuracy": 0.68,
                    "f1": 0.64,
                    "selection_metric": 0.74,
                    "selection_metric_name": "val_auc",
                    "learning_rates": [1e-4, 5e-5, 5e-5],
                },
                {
                    "epoch": 2,
                    "loss": 0.48,
                    "auc": 0.8,
                    "accuracy": 0.77,
                    "f1": 0.74,
                    "selection_metric": 0.8,
                    "selection_metric_name": "val_auc",
                    "learning_rates": [8e-5, 4e-5, 4e-5],
                },
            ],
        }
        diagnostics = ValidationDiagnostics(
            labels=np.array([0, 0, 1, 1], dtype=np.int64),
            probs=np.array([0.1, 0.3, 0.78, 0.91], dtype=np.float32),
            preds=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        figure_dir = self.root / "figures" / "latest"

        render_training_figures(
            history=history,
            diagnostics=diagnostics,
            class_balance_info=None,
            current_epoch=2,
            best_epoch=2,
            selection_metric_name="val_auc",
            figure_dir=figure_dir,
            warmup_epochs=1,
            latest_bundle=True,
        )

        expected_stems = (
            "training_dashboard",
            "validation_curves_latest",
            "validation_confusion_latest",
            "validation_score_distribution_latest",
        )
        for stem in expected_stems:
            for suffix in (".png", ".svg"):
                path = figure_dir / f"{stem}{suffix}"
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
    def test_render_training_figures_handles_single_class_validation(self) -> None:
        history = {
            "run": {"class_balance": None, "use_pos_weight": True, "auto_pos_weight": True},
            "train": [{"epoch": 1, "loss": 0.7, "auc": float("nan"), "accuracy": 1.0, "f1": 0.0}],
            "val": [
                {
                    "epoch": 1,
                    "loss": 0.4,
                    "auc": float("nan"),
                    "accuracy": 1.0,
                    "f1": 0.0,
                    "selection_metric": -0.4,
                    "selection_metric_name": "neg_val_loss",
                    "learning_rates": [1e-4],
                }
            ],
        }
        diagnostics = ValidationDiagnostics(
            labels=np.array([0, 0, 0], dtype=np.int64),
            probs=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            preds=np.array([0, 0, 0], dtype=np.int64),
        )
        figure_dir = self.root / "figures" / "best" / "epoch_001"

        render_training_figures(
            history=history,
            diagnostics=diagnostics,
            class_balance_info={"positive_class_name": "fake", "positive_count": 0, "negative_count": 3, "effective_pos_weight": None},
            current_epoch=1,
            best_epoch=1,
            selection_metric_name="neg_val_loss",
            figure_dir=figure_dir,
            warmup_epochs=3,
            latest_bundle=False,
        )

        for stem in (
            "training_dashboard",
            "validation_curves",
            "validation_confusion",
            "validation_score_distribution",
        ):
            self.assertTrue((figure_dir / f"{stem}.png").exists())
            self.assertTrue((figure_dir / f"{stem}.svg").exists())

    def test_main_logs_class_balance_and_uses_warmup_schedule(self) -> None:
        model = nn.Linear(4, 1)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = mock.Mock()
        cfg = TrainConfig(
            train_shards="train",
            val_shards="val",
            output_dir=str(self.root),
            epochs=4,
            save_every=99,
            use_amp=False,
            early_stopping_patience=10,
            spatial_freeze_warmup_epochs=3,
            invert_binary_labels=True,
        )
        model_cfg = ModelConfig(temporal_num_frames=7)
        balance = ClassBalanceInfo(
            negative_count=7,
            positive_count=3,
            positive_class_name="real",
            raw_pos_weight=7.0 / 3.0,
            effective_pos_weight=2.0,
        )
        diagnostics = ValidationDiagnostics(
            labels=np.array([0, 1], dtype=np.int64),
            probs=np.array([0.2, 0.8], dtype=np.float32),
            preds=np.array([0, 1], dtype=np.int64),
        )

        with mock.patch("training.train.setup_logging"), \
             mock.patch("training.train.parse_args", return_value=cfg), \
             mock.patch("training.train.set_seed"), \
             mock.patch("training.train.resolve_device", return_value=torch.device("cpu")), \
             mock.patch("training.train.build_dataloaders", return_value=([], [])), \
             mock.patch("training.train.build_progress_totals", return_value={"train": 11, "val": 7}), \
             mock.patch("training.train.build_class_balance_info", return_value=balance), \
             mock.patch("training.train.build_model", return_value=(model, model_cfg)), \
             mock.patch("training.train.build_loss", return_value=mock.Mock()), \
             mock.patch("training.train.build_optimizer", return_value=optimizer), \
             mock.patch("training.train.build_scheduler", return_value=scheduler), \
             mock.patch("training.train.resolve_figure_output_dirs", return_value=mock.Mock(run_dir=self.root / "figures", latest_dir=self.root / "figures" / "latest", best_root_dir=self.root / "figures" / "best")) as resolve_figures_mock, \
             mock.patch("training.train.render_training_figures") as render_figures_mock, \
             mock.patch("training.train.set_spatial_branch_trainable") as freeze_mock, \
             mock.patch("training.train.train_one_epoch", return_value={"loss": 0.4, "auc": 0.7, "accuracy": 0.8, "f1": 0.75}) as train_epoch_mock, \
             mock.patch("training.train.validate_one_epoch", return_value=({"loss": 0.3, "auc": 0.8, "accuracy": 0.9, "f1": 0.85}, diagnostics)) as val_epoch_mock, \
             mock.patch("training.train.write_history") as write_history_mock, \
             mock.patch("training.train.save_checkpoint"), \
             mock.patch("training.train.logger") as logger_mock:
            main()

        self.assertEqual(resolve_figures_mock.call_count, 1)
        self.assertEqual(scheduler.step.call_count, 4)
        self.assertEqual(freeze_mock.call_count, 4)
        self.assertEqual(render_figures_mock.call_count, 5)
        self.assertTrue(render_figures_mock.call_args_list[0].kwargs["latest_bundle"])
        self.assertFalse(render_figures_mock.call_args_list[1].kwargs["latest_bundle"])
        self.assertEqual(
            [call.kwargs["trainable"] for call in freeze_mock.call_args_list],
            [False, False, False, True],
        )
        self.assertEqual(train_epoch_mock.call_args_list[0].kwargs["total_batches"], 11)
        self.assertEqual(train_epoch_mock.call_args_list[0].kwargs["stage_label"], "Train 1/4")
        self.assertEqual(val_epoch_mock.call_args_list[0].kwargs["total_batches"], 7)
        self.assertEqual(val_epoch_mock.call_args_list[0].kwargs["stage_label"], "Val 1/4")
        self.assertEqual(write_history_mock.call_args.args[1]["run"]["class_balance"]["positive_class_name"], "real")
        self.assertEqual(write_history_mock.call_args.args[1]["run"]["class_balance"]["effective_pos_weight"], 2.0)
        logged_messages = "\n".join(str(call.args[0]) for call in logger_mock.info.call_args_list)
        self.assertIn("Invert binary labels: {}", logged_messages)
        self.assertIn("Class balance | positive_class={}", logged_messages)
        self.assertIn("phase={}", logged_messages)

    def test_save_checkpoint_includes_train_model_and_class_balance_metadata(self) -> None:
        model = nn.Linear(4, 1)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="max")
        scaler = GradScaler("cuda", enabled=False)
        train_cfg = TrainConfig(
            train_shards="train",
            val_shards="val",
            clip_len=8,
            invert_binary_labels=True,
            max_pos_weight=2.0,
        )
        model_cfg = ModelConfig(temporal_num_frames=7)
        checkpoint_path = self.root / "checkpoint.pt"
        class_balance = {
            "negative_count": 7,
            "positive_count": 3,
            "positive_class_name": "real",
            "raw_pos_weight": 7.0 / 3.0,
            "effective_pos_weight": 2.0,
        }

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=3,
            best_val_auc=0.77,
            best_selection_metric=0.77,
            cfg=train_cfg,
            model_cfg=model_cfg,
            class_balance_info=class_balance,
        )

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["best_val_auc"], 0.77)
        self.assertEqual(payload["best_selection_metric"], 0.77)
        self.assertEqual(payload["train_config"]["clip_len"], 8)
        self.assertTrue(payload["train_config"]["invert_binary_labels"])
        self.assertTrue(payload["train_config"]["use_pos_weight"])
        self.assertTrue(payload["train_config"]["auto_pos_weight"])
        self.assertEqual(payload["train_config"]["max_pos_weight"], 2.0)
        self.assertEqual(payload["train_config"]["spatial_freeze_warmup_epochs"], 3)
        self.assertEqual(payload["model_config"]["temporal_num_frames"], 7)
        self.assertEqual(payload["class_balance"]["positive_class_name"], "real")
        self.assertEqual(payload["class_balance"]["effective_pos_weight"], 2.0)
        self.assertIn("rng_state", payload)


if __name__ == "__main__":
    unittest.main()
