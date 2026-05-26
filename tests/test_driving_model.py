from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig, SpatialPatchLaneHead


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


def test_aux_regression_loss_can_weight_lane_feedback_terms():
    pred = torch.zeros(1, 1, 8)
    target = torch.zeros(1, 1, 8)
    target[..., 2] = 0.2
    target[..., 3] = -0.4

    loss = DrivingLeWM.aux_regression_loss(pred, target, [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0])

    expected = (4.0 * 0.5 * 0.2**2 + 4.0 * 0.5 * 0.4**2) / 8.0
    assert float(loss) == pytest.approx(expected)


def test_aux_regression_loss_can_upweight_lane_edge_samples():
    pred = torch.zeros(1, 2, 8)
    target = torch.zeros(1, 2, 8)
    target[..., 2] = torch.tensor([0.1, 0.5])

    base = DrivingLeWM.aux_regression_loss(pred, target, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    edge = DrivingLeWM.aux_regression_loss(
        pred,
        target,
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        lane_edge_threshold=0.25,
        lane_edge_weight=4.0,
    )

    assert float(base) == pytest.approx((0.5 * 0.1**2 + 0.5 * 0.5**2) / 2.0)
    assert float(edge) == pytest.approx((0.5 * 0.1**2 + 4.0 * 0.5 * 0.5**2) / 5.0)


def test_aux_control_loss_penalizes_wrong_steering_direction():
    target = torch.zeros(1, 1, 8)
    same_direction = torch.zeros(1, 1, 8)
    wrong_direction = torch.zeros(1, 1, 8)
    target[..., 2] = 1.0
    same_direction[..., 2] = 0.2
    wrong_direction[..., 2] = -0.2

    kwargs = dict(
        lane_gain=0.5,
        heading_gain=0.0,
        steer_limit=1.0,
        sign_weight=2.0,
        active_threshold=0.05,
    )
    same_loss = DrivingLeWM.aux_control_loss(same_direction, target, **kwargs)
    wrong_loss = DrivingLeWM.aux_control_loss(wrong_direction, target, **kwargs)

    assert float(wrong_loss) > float(same_loss)


def test_aux_control_loss_can_balance_steering_sign_groups():
    target = torch.zeros(1, 4, 8)
    pred = torch.zeros(1, 4, 8)
    target[..., 2] = torch.tensor([-1.0, 1.0, 1.0, 1.0])
    pred[..., 2] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    kwargs = dict(
        lane_gain=1.0,
        heading_gain=0.0,
        steer_limit=2.0,
        sign_weight=1.0,
        active_threshold=0.05,
    )
    unbalanced = DrivingLeWM.aux_control_loss(pred, target, balance_signs=False, **kwargs)
    balanced = DrivingLeWM.aux_control_loss(pred, target, balance_signs=True, **kwargs)

    assert float(balanced) > float(unbalanced)


def test_aux_lane_underamp_loss_penalizes_small_same_sign_prediction():
    target = torch.zeros(1, 2, 8)
    good = torch.zeros(1, 2, 8)
    weak = torch.zeros(1, 2, 8)
    target[..., 2] = torch.tensor([0.5, -0.5])
    good[..., 2] = torch.tensor([0.5, -0.5])
    weak[..., 2] = torch.tensor([0.05, -0.05])

    good_loss = DrivingLeWM.aux_lane_underamp_loss(
        good,
        target,
        active_threshold=0.25,
        margin=0.9,
    )
    weak_loss = DrivingLeWM.aux_lane_underamp_loss(
        weak,
        target,
        active_threshold=0.25,
        margin=0.9,
    )

    assert float(good_loss) == pytest.approx(0.0)
    assert float(weak_loss) == pytest.approx(0.4)


def test_scalar_sign_loss_penalizes_wrong_component_direction():
    target = torch.tensor([[-0.2, 0.2, 0.2]], dtype=torch.float32)
    same = torch.tensor([[-0.1, 0.1, 0.1]], dtype=torch.float32)
    wrong = torch.tensor([[0.1, -0.1, 0.1]], dtype=torch.float32)

    same_loss = DrivingLeWM.scalar_sign_loss(same, target, active_threshold=0.05, balance_signs=True)
    wrong_loss = DrivingLeWM.scalar_sign_loss(wrong, target, active_threshold=0.05, balance_signs=True)

    assert float(same_loss) == pytest.approx(0.0)
    assert float(wrong_loss) > float(same_loss)


def test_scalar_sign_loss_bce_penalizes_wrong_component_direction():
    target = torch.tensor([[-0.2, 0.2]], dtype=torch.float32)
    same = torch.tensor([[-0.3, 0.3]], dtype=torch.float32)
    wrong = torch.tensor([[0.3, -0.3]], dtype=torch.float32)

    same_loss = DrivingLeWM.scalar_sign_loss(
        same,
        target,
        active_threshold=0.05,
        balance_signs=True,
        mode="bce",
        logit_scale=10.0,
    )
    wrong_loss = DrivingLeWM.scalar_sign_loss(
        wrong,
        target,
        active_threshold=0.05,
        balance_signs=True,
        mode="bce",
        logit_scale=10.0,
    )

    assert float(same_loss) < 0.1
    assert float(wrong_loss) > float(same_loss) * 10.0


def test_scalar_sign_loss_rejects_unknown_mode():
    target = torch.tensor([[0.2]], dtype=torch.float32)
    pred = torch.tensor([[0.1]], dtype=torch.float32)

    with pytest.raises(ValueError, match="scalar sign loss mode"):
        DrivingLeWM.scalar_sign_loss(pred, target, active_threshold=0.05, mode="bad")


def test_aux_temporal_delta_loss_penalizes_unstable_lane_heading_changes():
    target = torch.zeros(1, 4, 8)
    good = torch.zeros(1, 4, 8)
    unstable = torch.zeros(1, 4, 8)
    target[..., 2] = torch.tensor([0.10, 0.20, 0.30, 0.40])
    target[..., 3] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    good[..., 2:4] = target[..., 2:4]
    unstable[..., 2] = torch.tensor([0.10, -0.20, 0.30, -0.40])
    unstable[..., 3] = torch.tensor([0.01, -0.02, 0.03, -0.04])

    good_loss = DrivingLeWM.aux_temporal_delta_loss(good, target, lane_weight=1.0, heading_weight=1.0)
    unstable_loss = DrivingLeWM.aux_temporal_delta_loss(unstable, target, lane_weight=1.0, heading_weight=1.0)

    assert float(good_loss) == pytest.approx(0.0)
    assert float(unstable_loss) > float(good_loss)


def test_progress_signal_rejects_unknown_mode():
    route_progress = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="Unknown progress_mode"):
        DrivingLeWM.progress_signal(route_progress, "bad-mode")


def test_encoder_pooling_can_use_patch_tokens(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            tokens = torch.tensor(
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=pixels.dtype,
                device=pixels.device,
            )
            return SimpleNamespace(last_hidden_state=tokens.unsqueeze(0).expand(batch, -1, -1))

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    base = dict(
        image_size=8,
        patch_size=4,
        action_dim=3,
        embed_dim=4,
        history_size=1,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
    )
    pixels = torch.zeros(1, 1, 3, 8, 8)

    mean_model = DrivingLeWM(DrivingLeWMConfig(**base, encoder_pooling="mean_patch"))
    mean_model.projector = nn.Identity()
    torch.testing.assert_close(mean_model.encode_pixels(pixels), torch.tensor([[[3.0, 4.0, 5.0, 6.0]]]))

    combined_model = DrivingLeWM(DrivingLeWMConfig(**base, encoder_pooling="cls_mean"))
    assert combined_model.projector[0].in_features == 8
    combined_model.projector = nn.Identity()
    torch.testing.assert_close(
        combined_model.encode_pixels(pixels),
        torch.tensor([[[10.0, 11.0, 12.0, 13.0, 3.0, 4.0, 5.0, 6.0]]]),
    )


def test_freeze_encoder_disables_only_encoder_gradients(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.proj = nn.Linear(1, 1)

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
        history_size=1,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
        freeze_encoder=True,
    )

    model = DrivingLeWM(cfg)

    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert any(param.requires_grad for param in model.projector.parameters())


def test_spatial_patch_lane_head_zero_init_starts_as_noop():
    head = SpatialPatchLaneHead(
        input_dim=4,
        hidden_dim=8,
        pooling="attention",
        dropout=0.0,
        zero_init=True,
    )
    tokens = torch.randn(2, 3, 5, 4)

    out = head(tokens)

    assert out.shape == (2, 3, 2)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_patch_lane_residual_affects_perception_aux_without_pred_aux(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            tokens = torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=pixels.dtype,
                device=pixels.device,
            )
            return SimpleNamespace(last_hidden_state=tokens.unsqueeze(0).expand(batch, -1, -1))

    class ConstantPatchLaneHead(nn.Module):
        def forward(self, patch_tokens):
            return patch_tokens.new_tensor([0.3, -0.2]).expand(*patch_tokens.shape[:-2], 2)

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    model = DrivingLeWM(
        DrivingLeWMConfig(
            image_size=8,
            patch_size=4,
            action_dim=3,
            embed_dim=4,
            history_size=2,
            predictor_depth=1,
            predictor_heads=1,
            predictor_mlp_dim=8,
            encoder_pooling="mean_patch",
            use_aux_head=True,
            aux_lane_patch_mode="residual",
        )
    )
    model.projector = nn.Identity()
    for param in model.aux_head.parameters():
        nn.init.zeros_(param)
    model.aux_lane_patch_head = ConstantPatchLaneHead()
    pixels = torch.zeros(1, 3, 3, 8, 8)
    actions = torch.zeros(1, 3, 3)

    perceived = model.perceive_aux(pixels)
    out = model({"pixels": pixels, "action": actions})
    rolled = model.rollout_aux(pixels[:, :2], actions[:, :2])

    assert perceived.shape == (1, 8)
    torch.testing.assert_close(perceived[:, 2:4], torch.tensor([[0.3, -0.2]]))
    torch.testing.assert_close(out["aux"][..., 2], torch.full((1, 3), 0.3))
    torch.testing.assert_close(out["aux"][..., 3], torch.full((1, 3), -0.2))
    torch.testing.assert_close(out["pred_aux"][..., 2:4], torch.zeros_like(out["pred_aux"][..., 2:4]))
    torch.testing.assert_close(rolled, torch.zeros_like(rolled))


def test_temporal_aux_residual_affects_perception_aux_without_pred_aux(monkeypatch):
    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            tokens = torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=pixels.dtype,
                device=pixels.device,
            )
            return SimpleNamespace(last_hidden_state=tokens.unsqueeze(0).expand(batch, -1, -1))

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    model = DrivingLeWM(
        DrivingLeWMConfig(
            image_size=8,
            patch_size=4,
            action_dim=3,
            embed_dim=4,
            history_size=3,
            predictor_depth=1,
            predictor_heads=1,
            predictor_mlp_dim=8,
            encoder_pooling="mean_patch",
            use_aux_head=True,
            use_temporal_aux_head=True,
            temporal_aux_zero_init=True,
        )
    )
    model.projector = nn.Identity()
    for param in model.aux_head.parameters():
        nn.init.zeros_(param)
    assert model.temporal_aux_head is not None
    for param in model.temporal_aux_head.parameters():
        nn.init.zeros_(param)
    model.temporal_aux_head[-1].bias.data = torch.tensor([0.4, -0.3])
    pixels = torch.zeros(1, 4, 3, 8, 8)
    actions = torch.zeros(1, 4, 3)

    perceived = model.perceive_aux(pixels)
    out = model({"pixels": pixels, "action": actions})
    rolled = model.rollout_aux(pixels[:, :3], actions[:, :3])

    torch.testing.assert_close(perceived[:, 2:4], torch.tensor([[0.4, -0.3]]))
    torch.testing.assert_close(out["aux"][..., 2], torch.full((1, 4), 0.4))
    torch.testing.assert_close(out["aux"][..., 3], torch.full((1, 4), -0.3))
    torch.testing.assert_close(out["pred_aux"][..., 2:4], torch.zeros_like(out["pred_aux"][..., 2:4]))
    torch.testing.assert_close(rolled, torch.zeros_like(rolled))


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


def test_action_regression_loss_can_weight_steering_component():
    pred = torch.zeros(1, 1, 3)
    target = torch.tensor([[[0.2, 0.2, 0.2]]], dtype=torch.float32)

    loss, parts = DrivingLeWM.action_regression_loss(pred, target, component_weights=[1.0, 8.0, 1.0])

    per_component = 0.5 * 0.2**2
    assert float(parts["throttle"]) == pytest.approx(per_component)
    assert float(parts["steer"]) == pytest.approx(per_component)
    assert float(parts["brake"]) == pytest.approx(per_component)
    assert float(loss) == pytest.approx(per_component)


def test_action_regression_loss_upweights_active_steer_samples():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[[0.0, 0.02, 0.0], [0.0, 0.20, 0.0]]], dtype=torch.float32)

    loss, _ = DrivingLeWM.action_regression_loss(
        pred,
        target,
        component_weights=[1.0, 1.0, 1.0],
        active_steer_weight=5.0,
        active_steer_threshold=0.1,
    )

    low = 0.5 * 0.02**2
    high = 0.5 * 0.20**2
    expected = (low + high * 5.0) / 10.0
    assert float(loss) == pytest.approx(expected)


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


def test_perceive_aux_reads_current_image_aux(monkeypatch):
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
        use_aux_head=True,
        aux_weight=1.0,
        pred_aux_weight=0.0,
    )
    model = DrivingLeWM(cfg)
    pixels = torch.zeros(1, 2, 3, 8, 8)

    aux = model.perceive_aux(pixels)

    assert aux.shape == (1, 8)


def test_action_loss_prefers_teacher_action_target_when_present(monkeypatch):
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
        pred_weight=0.0,
        sigreg_weight=0.0,
        use_aux_head=False,
        aux_weight=0.0,
        action_weight=1.0,
        pred_action_weight=0.0,
    )
    model = DrivingLeWM(cfg)
    for param in model.action_head.parameters():
        nn.init.zeros_(param)
    batch = {
        **batch_with_progress([0.0, 1.0]),
        "pixels": torch.zeros(1, 2, 3, 8, 8),
        "action": torch.zeros(1, 2, 3),
    }

    action_loss = model.loss(batch)["action_loss"]
    teacher_action_loss = model.loss({**batch, "teacher_action": torch.ones(1, 2, 3)})["action_loss"]

    assert float(action_loss) == pytest.approx(0.0)
    assert float(teacher_action_loss) > 0.0


def test_action_teacher_only_uses_teacher_mask(monkeypatch):
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
        pred_weight=0.0,
        sigreg_weight=0.0,
        use_aux_head=False,
        aux_weight=0.0,
        action_weight=1.0,
        pred_action_weight=0.0,
        action_teacher_only=True,
    )
    model = DrivingLeWM(cfg)
    for param in model.action_head.parameters():
        nn.init.zeros_(param)
    batch = {
        **batch_with_progress([0.0, 1.0]),
        "pixels": torch.zeros(1, 2, 3, 8, 8),
        "action": torch.zeros(1, 2, 3),
        "teacher_action": torch.ones(1, 2, 3),
    }

    masked_out = model.loss({**batch, "teacher_action_mask": torch.zeros(1, 2, 1)})["action_loss"]
    masked_in = model.loss({**batch, "teacher_action_mask": torch.ones(1, 2, 1)})["action_loss"]

    assert float(masked_out) == pytest.approx(0.0)
    assert float(masked_in) > 0.0


def test_route_conditioning_is_optional_and_requires_route_ids(monkeypatch):
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
        action_weight=1.0,
        pred_action_weight=0.0,
        route_vocab_size=16,
    )
    model = DrivingLeWM(cfg)
    pixels = torch.zeros(1, 2, 3, 8, 8)
    action = torch.zeros(1, 2, 3)
    batch = {
        **batch_with_progress([0.0, 1.0]),
        "pixels": pixels,
        "action": action,
        "route_id": torch.full((1, 2, 1), 10, dtype=torch.long),
    }

    out = model.forward(batch)
    policy = model.policy_action(pixels, route_id=10)

    assert out["action"].shape == (1, 2, 3)
    assert policy.shape == (1, 3)
    with pytest.raises(ValueError, match="route_id is required"):
        model.policy_action(pixels)


def test_temporal_action_head_uses_history_actions(monkeypatch):
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
        history_size=3,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
        pred_weight=0.0,
        sigreg_weight=0.0,
        aux_weight=0.0,
        action_weight=1.0,
        pred_action_weight=0.0,
        use_temporal_action_head=True,
        temporal_action_include_history_actions=True,
    )
    model = DrivingLeWM(cfg)
    pixels = torch.zeros(2, 4, 3, 8, 8)
    actions = torch.zeros(2, 4, 3)
    batch = {
        **batch_with_progress([0.0, 1.0, 2.0, 3.0]),
        "pixels": pixels,
        "action": actions,
    }

    out = model.forward(batch)
    policy = model.policy_action(pixels[:, -3:], action_history=actions[:, :3])
    losses = model.loss(batch)

    assert out["action"].shape == (2, 3)
    assert policy.shape == (2, 3)
    assert float(losses["pred_action_loss"]) == 0.0
    with pytest.raises(ValueError, match="action_history is required"):
        model.policy_action(pixels[:, -3:])


def test_temporal_action_head_can_use_image_history_only(monkeypatch):
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
        history_size=3,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=8,
        use_aux_head=False,
        pred_weight=0.0,
        sigreg_weight=0.0,
        aux_weight=0.0,
        action_weight=1.0,
        pred_action_weight=0.0,
        use_temporal_action_head=True,
        temporal_action_include_history_actions=False,
    )
    model = DrivingLeWM(cfg)
    pixels = torch.zeros(2, 4, 3, 8, 8)
    actions = torch.zeros(2, 4, 3)
    batch = {
        **batch_with_progress([0.0, 1.0, 2.0, 3.0]),
        "pixels": pixels,
        "action": actions,
    }

    out = model.forward(batch)
    policy = model.policy_action(pixels[:, -3:])
    losses = model.loss(batch)

    assert out["action"].shape == (2, 3)
    assert policy.shape == (2, 3)
    assert float(losses["pred_action_loss"]) == 0.0


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

    assert float(losses["pred_loss"].detach()) > 0.0
    assert float(losses["loss"].detach()) == pytest.approx(float(losses["action_loss"].detach()))
