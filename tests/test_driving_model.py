from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig


def batch_with_progress(progress):
    progress_t = torch.tensor(progress, dtype=torch.float32).view(1, -1, 1)
    zeros = torch.zeros_like(progress_t)
    ones = torch.ones_like(progress_t)
    return {
        "speed_mps": ones,
        "route_progress_m": progress_t,
        "lane_offset_m": zeros,
        "heading_error_rad": zeros,
        "collision": zeros,
        "offroad": zeros,
        "red_light": zeros,
        "blocked": zeros,
    }


def test_progress_signal_supports_absolute_and_delta_modes():
    route_progress = torch.tensor([[[10.0], [12.5], [15.0]]])

    absolute = DrivingLeWM.progress_signal(route_progress, "absolute")
    delta = DrivingLeWM.progress_signal(route_progress, "delta")

    assert absolute.tolist() == [[[10.0], [12.5], [15.0]]]
    assert delta.tolist() == [[[0.0], [2.5], [2.5]]]


def test_aux_target_uses_delta_progress_when_configured():
    batch = batch_with_progress([1.0, 4.0, 9.5])

    target = DrivingLeWM.aux_target(batch, "delta")

    assert target.shape == (1, 3, 8)
    assert target[0, :, 1].tolist() == pytest.approx([0.0, 3.0, 5.5])


def test_progress_signal_rejects_unknown_mode():
    route_progress = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="Unknown progress_mode"):
        DrivingLeWM.progress_signal(route_progress, "bad-mode")


def test_action_conflict_loss_penalizes_positive_throttle_and_brake_overlap():
    actions = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0, 0.2, 0.0, 0.3],
                [0.0, 0.0, 0.4, -0.5, 0.0, 0.8],
            ]
        ],
        dtype=torch.float32,
    )

    loss = DrivingLeWM.action_conflict_loss(actions)

    assert float(loss) == pytest.approx((0.0 + 0.06 + 0.0 + 0.0) / 4)


def test_no_aux_config_disables_aux_head_and_aux_losses(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            values = torch.arange(batch * 8, dtype=pixels.dtype, device=pixels.device).reshape(batch, 2, 4)
            return SimpleNamespace(last_hidden_state=values)

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    cfg = DrivingLeWMConfig(
        image_size=8,
        patch_size=4,
        action_dim=3,
        embed_dim=4,
        history_size=2,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
        aux_weight=0.0,
        pred_aux_weight=0.0,
        action_weight=0.5,
        pred_action_weight=1.0,
    )
    model = DrivingLeWM(cfg)
    batch = {
        **batch_with_progress([0.0, 1.0, 2.0]),
        "pixels": torch.zeros(1, 3, 3, 8, 8),
        "action": torch.zeros(1, 3, 3),
    }

    out = model.forward(batch)
    losses = model.loss(batch)

    assert model.aux_head is None
    assert "aux" not in out
    assert "pred_aux" not in out
    assert float(losses["aux_loss"]) == 0.0
    assert float(losses["pred_aux_loss"]) == 0.0
    with pytest.raises(RuntimeError, match="use_aux_head=true"):
        model.rollout_aux(batch["pixels"], batch["action"])


def test_pred_weight_can_disable_latent_prediction_loss(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            values = torch.arange(batch * 8, dtype=pixels.dtype, device=pixels.device).reshape(batch, 2, 4)
            return SimpleNamespace(last_hidden_state=values)

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    cfg = DrivingLeWMConfig(
        image_size=8,
        patch_size=4,
        action_dim=3,
        embed_dim=4,
        history_size=2,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
        pred_weight=0.0,
        sigreg_weight=0.0,
        aux_weight=0.0,
        action_weight=1.0,
        pred_action_weight=0.0,
        action_conflict_weight=0.0,
    )
    model = DrivingLeWM(cfg)
    batch = {
        **batch_with_progress([0.0, 1.0, 2.0]),
        "pixels": torch.zeros(1, 3, 3, 8, 8),
        "action": torch.zeros(1, 3, 3),
    }

    losses = model.loss(batch)

    assert float(losses["pred_loss"]) > 0.0
    assert float(losses["loss"]) == pytest.approx(float(losses["action_loss"]))
