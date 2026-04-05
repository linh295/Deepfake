from __future__ import annotations

import random
import shutil
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
from training.utils.builders import build_model, build_scheduler
from training.utils.checkpointing import load_checkpoint, save_checkpoint
from training.utils.loops import train_one_epoch, validate_one_epoch
from training.utils.metrics import select_checkpoint_metric
from training.utils.runtime import restore_rng_state, set_seed


class _FakeTqdm:
    def __init__(self, iterable, desc: str | None = None) -> None:
        self._items = list(iterable)
        self.desc = desc

    def __iter__(self):
        return iter(self._items)

    def set_postfix(self, **kwargs) -> None:
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
        tmp_root = Path("d:/Deepfake/.tmp-tests")
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.root = tmp_root / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

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
            )

        self.assertIn("loss", metrics)
        self.assertAlmostEqual(metrics["loss"], 0.693147, places=4)

    def test_validate_one_epoch_works_without_loader_len(self) -> None:
        model = _TinyModel()
        loader = _LoaderWithoutLen([_batch(), _batch()])
        criterion = nn.BCEWithLogitsLoss()
        cfg = TrainConfig(train_shards="train", val_shards="val", use_amp=False)

        with mock.patch("training.utils.loops.tqdm", _FakeTqdm):
            metrics = validate_one_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                device=torch.device("cpu"),
                epoch=1,
                cfg=cfg,
            )

        self.assertIn("loss", metrics)
        self.assertAlmostEqual(metrics["loss"], 0.693147, places=4)

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

    def test_main_steps_scheduler_with_selection_metric(self) -> None:
        model = nn.Linear(4, 1)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = mock.Mock()
        cfg = TrainConfig(
            train_shards="train",
            val_shards="val",
            output_dir=str(self.root),
            epochs=1,
            save_every=99,
            use_amp=False,
        )
        model_cfg = ModelConfig(temporal_num_frames=7)

        with mock.patch("training.train.setup_logging"), \
             mock.patch("training.train.parse_args", return_value=cfg), \
             mock.patch("training.train.set_seed"), \
             mock.patch("training.train.resolve_device", return_value=torch.device("cpu")), \
             mock.patch("training.train.build_dataloaders", return_value=([], [])), \
             mock.patch("training.train.build_model", return_value=(model, model_cfg)), \
             mock.patch("training.train.build_loss", return_value=mock.Mock()), \
             mock.patch("training.train.build_optimizer", return_value=optimizer), \
             mock.patch("training.train.build_scheduler", return_value=scheduler), \
             mock.patch("training.train.train_one_epoch", return_value={"loss": 0.4, "auc": 0.7, "accuracy": 0.8, "f1": 0.75}), \
             mock.patch("training.train.validate_one_epoch", return_value={"loss": 0.3, "auc": float("nan"), "accuracy": 0.9, "f1": 0.85}), \
             mock.patch("training.train.write_history"), \
             mock.patch("training.train.save_checkpoint"), \
             mock.patch("training.train.logger"):
            main()

        scheduler.step.assert_called_once_with(-0.3)

    def test_save_checkpoint_includes_train_and_model_config_metadata(self) -> None:
        model = nn.Linear(4, 1)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="max")
        scaler = GradScaler("cuda", enabled=False)
        train_cfg = TrainConfig(train_shards="train", val_shards="val", clip_len=8)
        model_cfg = ModelConfig(temporal_num_frames=7)
        checkpoint_path = self.root / "checkpoint.pt"

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
        )

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["best_val_auc"], 0.77)
        self.assertEqual(payload["best_selection_metric"], 0.77)
        self.assertEqual(payload["train_config"]["clip_len"], 8)
        self.assertEqual(payload["model_config"]["temporal_num_frames"], 7)
        self.assertIn("rng_state", payload)


if __name__ == "__main__":
    unittest.main()
