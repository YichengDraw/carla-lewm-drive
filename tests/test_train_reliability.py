import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from carla_lewm_drive.driving_lewm import train as train_module


def base_cfg(tmp_path):
    return {
        "run": {"name": "unit_run", "seed": 123, "output_dir": str(tmp_path / "out")},
        "data": {"dataset_path": "dummy.h5", "image_size": 32, "history_size": 1},
        "model": {},
        "optimizer": {"lr": 1.0, "weight_decay": 0.0},
        "trainer": {
            "max_epochs": 2,
            "batch_size": 1,
            "num_workers": 0,
            "precision": "bf16",
            "gradient_clip_norm": 100.0,
            "log_every_steps": 1,
            "batch_probe_order": [2],
        },
        "wandb": {"enabled": False},
    }


def test_wandb_uses_unique_id_by_default(monkeypatch, tmp_path):
    calls = []

    fake_wandb = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or SimpleNamespace())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = base_cfg(tmp_path)
    cfg["wandb"] = {"enabled": True, "project": "unit", "mode": "offline"}

    train_module.maybe_init_wandb(cfg, disabled=False)

    assert calls[0]["name"] == "unit_run"
    assert calls[0]["id"].startswith("unit_run-")
    assert calls[0]["id"] != "unit_run"
    assert calls[0]["resume"] == "never"


def test_wandb_resume_requires_explicit_id(monkeypatch, tmp_path):
    fake_wandb = SimpleNamespace(init=lambda **kwargs: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = base_cfg(tmp_path)
    cfg["wandb"] = {"enabled": True, "project": "unit", "resume": "allow"}

    with pytest.raises(ValueError, match="wandb.resume requires"):
        train_module.maybe_init_wandb(cfg, disabled=False)


def test_append_metrics_expands_namespaced_headers(tmp_path):
    metrics_path = tmp_path / "metrics.csv"

    train_module.append_metrics(metrics_path, {"stage": "train", "step": 1, "train/loss": 1.0})
    train_module.append_metrics(metrics_path, {"stage": "val", "step": 1, "val/loss": 0.5})
    train_module.append_metrics(metrics_path, {"stage": "test", "step": 1, "test/loss": 0.25})

    text = metrics_path.read_text(encoding="utf-8").splitlines()

    assert text[0] == "stage,step,train/loss,val/loss,test/loss"
    assert "train,1,1.0,," in text[1]
    assert "val,1,,0.5," in text[2]
    assert "test,1,,,0.25" in text[3]


def test_batch_probe_uses_autocast_and_cleans_each_tier(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)
    autocast_calls = []
    clear_calls = []

    class ProbeModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def loss(self, batch):
            loss = self.weight.square() + batch["action"].sum() * 0.0
            return {"loss": loss}

    @contextmanager
    def fake_autocast(device, call_cfg):
        autocast_calls.append((device.type, call_cfg["trainer"]["precision"], call_cfg["trainer"]["batch_size"]))
        yield

    batch = {"action": torch.zeros(1, 1, 2)}
    monkeypatch.setattr(train_module, "DrivingLeWM", ProbeModel)
    monkeypatch.setattr(train_module, "make_loader", lambda dataset, call_cfg, shuffle: [batch])
    monkeypatch.setattr(train_module, "autocast_context", fake_autocast)
    monkeypatch.setattr(train_module, "clear_cuda_probe_state", lambda device: clear_calls.append(device.type))

    results = train_module.run_batch_probe(cfg, train_set=object())

    assert results == [{"batch_size": 2, "status": "pass", "peak_vram_gb": 0.0}]
    assert len(autocast_calls) == 1
    assert autocast_calls[0][1:] == ("bf16", 2)
    assert clear_calls == [autocast_calls[0][0], autocast_calls[0][0]]


def test_batch_probe_handles_empty_loader(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)

    monkeypatch.setattr(train_module, "make_loader", lambda dataset, call_cfg, shuffle: [])
    monkeypatch.setattr(train_module, "DrivingLeWM", lambda _cfg: pytest.fail("model should not be built"))

    results = train_module.run_batch_probe(cfg, train_set=object())

    assert results == [{"batch_size": 2, "status": "empty", "peak_vram_gb": 0.0}]


def test_train_loads_best_checkpoint_for_test(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"]["max_epochs"] = 2
    test_weights = []

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def loss(self, _batch):
            zero = self.weight.detach() * 0.0
            return {
                "loss": self.weight,
                "pred_loss": zero,
                "sigreg_loss": zero,
                "aux_loss": zero,
            }

    def fake_build_splits(_dataset_path, _data_cfg):
        split = SimpleNamespace(train_episodes=[0], val_episodes=[1], test_episodes=[2])
        return "train", "val", "test", split

    def fake_make_loader(dataset, _cfg, shuffle):
        if dataset == "train":
            return [{"action": torch.zeros(1, 1, 2)}]
        return dataset

    val_losses = iter([0.1, 0.2])

    def fake_evaluate_loss(model, loader, _device, _cfg, _limit_batches=None):
        if loader == "val":
            value = next(val_losses)
            return {"loss": value, "pred_loss": value, "sigreg_loss": 0.0, "aux_loss": 0.0}
        if loader == "test":
            test_weights.append(float(model.weight.detach().cpu()))
            return {"loss": 0.05, "pred_loss": 0.05, "sigreg_loss": 0.0, "aux_loss": 0.0}
        raise AssertionError(f"unexpected loader {loader}")

    monkeypatch.setattr(train_module, "DrivingLeWM", TinyTrainModel)
    monkeypatch.setattr(train_module, "build_splits", fake_build_splits)
    monkeypatch.setattr(train_module, "make_loader", fake_make_loader)
    monkeypatch.setattr(train_module, "evaluate_loss", fake_evaluate_loss)

    best_path = train_module.train(cfg, no_wandb=True)
    report = json.loads((best_path.parent / "test_metrics.json").read_text(encoding="utf-8"))

    assert best_path.name == "best.pt"
    assert test_weights == pytest.approx([-1.0])
    assert report["checkpoint"]["kind"] == "best"
    assert report["checkpoint"]["epoch"] == 1
    assert report["test/loss"] == 0.05


def test_train_honors_eval_every_epochs(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"]["max_epochs"] = 3
    cfg["trainer"]["eval_every_epochs"] = 2

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def loss(self, _batch):
            zero = self.weight.detach() * 0.0
            return {"loss": self.weight.square(), "pred_loss": zero, "sigreg_loss": zero, "aux_loss": zero}

    def fake_build_splits(_dataset_path, _data_cfg):
        split = SimpleNamespace(train_episodes=[0], val_episodes=[1], test_episodes=[2])
        return "train", "val", "test", split

    def fake_make_loader(dataset, _cfg, shuffle):
        if dataset == "train":
            return [{"action": torch.zeros(1, 1, 2)}]
        return dataset

    def fake_evaluate_loss(model, loader, _device, call_cfg, _limit_batches=None):
        if loader == "val":
            return {"loss": 0.5, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        if loader == "test":
            return {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        raise AssertionError(f"unexpected loader {loader}")

    monkeypatch.setattr(train_module, "DrivingLeWM", TinyTrainModel)
    monkeypatch.setattr(train_module, "build_splits", fake_build_splits)
    monkeypatch.setattr(train_module, "make_loader", fake_make_loader)
    monkeypatch.setattr(train_module, "evaluate_loss", fake_evaluate_loss)

    train_module.train(cfg, no_wandb=True)

    rows = (tmp_path / "out" / "metrics.csv").read_text(encoding="utf-8").splitlines()
    val_rows = [row for row in rows if row.startswith("val,")]
    assert [row.split(",")[1] for row in val_rows] == ["2", "3"]


def test_train_honors_max_steps_before_max_epochs(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"]["max_epochs"] = 10
    cfg["trainer"]["max_steps"] = 3
    cfg["trainer"]["eval_every_epochs"] = 10

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def loss(self, _batch):
            zero = self.weight.detach() * 0.0
            return {"loss": self.weight.square(), "pred_loss": zero, "sigreg_loss": zero, "aux_loss": zero}

    def fake_build_splits(_dataset_path, _data_cfg):
        split = SimpleNamespace(train_episodes=[0], val_episodes=[1], test_episodes=[2])
        return "train", "val", "test", split

    def fake_make_loader(dataset, _cfg, shuffle):
        if dataset == "train":
            return [{"action": torch.zeros(1, 1, 2)}] * 5
        return dataset

    def fake_evaluate_loss(model, loader, _device, call_cfg, _limit_batches=None):
        if loader == "val":
            return {"loss": 0.5, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        if loader == "test":
            return {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        raise AssertionError(f"unexpected loader {loader}")

    monkeypatch.setattr(train_module, "DrivingLeWM", TinyTrainModel)
    monkeypatch.setattr(train_module, "build_splits", fake_build_splits)
    monkeypatch.setattr(train_module, "make_loader", fake_make_loader)
    monkeypatch.setattr(train_module, "evaluate_loss", fake_evaluate_loss)

    train_module.train(cfg, no_wandb=True)

    report = json.loads((tmp_path / "out" / "test_metrics.json").read_text(encoding="utf-8"))
    assert report["checkpoint"]["global_step"] == 3
    assert report["checkpoint"]["stop_reason"] == "max_steps"


def test_train_supports_validation_early_stop(monkeypatch, tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"]["max_epochs"] = 10
    cfg["trainer"]["eval_every_epochs"] = 1
    cfg["trainer"]["early_stop_patience_evals"] = 2
    cfg["trainer"]["early_stop_min_steps"] = 0

    class TinyTrainModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def loss(self, _batch):
            zero = self.weight.detach() * 0.0
            return {"loss": self.weight.square(), "pred_loss": zero, "sigreg_loss": zero, "aux_loss": zero}

    def fake_build_splits(_dataset_path, _data_cfg):
        split = SimpleNamespace(train_episodes=[0], val_episodes=[1], test_episodes=[2])
        return "train", "val", "test", split

    def fake_make_loader(dataset, _cfg, shuffle):
        if dataset == "train":
            return [{"action": torch.zeros(1, 1, 2)}]
        return dataset

    val_losses = iter([0.5, 0.6, 0.7])

    def fake_evaluate_loss(model, loader, _device, call_cfg, _limit_batches=None):
        if loader == "val":
            value = next(val_losses)
            return {"loss": value, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        if loader == "test":
            return {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
        raise AssertionError(f"unexpected loader {loader}")

    monkeypatch.setattr(train_module, "DrivingLeWM", TinyTrainModel)
    monkeypatch.setattr(train_module, "build_splits", fake_build_splits)
    monkeypatch.setattr(train_module, "make_loader", fake_make_loader)
    monkeypatch.setattr(train_module, "evaluate_loss", fake_evaluate_loss)

    train_module.train(cfg, no_wandb=True)

    report = json.loads((tmp_path / "out" / "test_metrics.json").read_text(encoding="utf-8"))
    assert report["checkpoint"]["global_step"] == 1
    assert report["checkpoint"]["stop_reason"] == "early_stop_val_loss"
