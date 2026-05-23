import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carla_lewm_drive.closed_loop_eval.evaluate import (
    CEMPlanner,
    apply_cli_overrides,
    expand_action_to_model_dim,
    flatten_model_action_to_block,
    lane_keep_action,
    lane_keep_steer,
    load_model,
    maybe_govern_model_speed,
    normalize_policy,
    resolve_policy,
    run_dry_eval,
    write_action_trace,
    write_metrics,
)
from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics
from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig


def planner_cfg():
    return {
        "seed": 7,
        "horizon": 2,
        "num_samples": 2,
        "iterations": 1,
        "elite_frac": 0.5,
        "action_mean": [0.2, 0.0, 0.0],
        "action_std": [0.01, 0.01, 0.01],
        "target_speed_mps": 5.0,
        "cost": {
            "progress_reward": 1.0,
            "lane_offset_penalty": 1.0,
            "heading_penalty": 1.0,
            "speed_error_penalty": 1.0,
            "collision_penalty": 100.0,
            "offroad_penalty": 60.0,
            "red_light_penalty": 80.0,
            "blocked_penalty": 100.0,
            "action_smoothness_penalty": 0.1,
        },
    }


class DummyModel:
    def __init__(self):
        self.cfg = SimpleNamespace(action_dim=15)
        self.seen_shapes = []

    def rollout_aux(self, pixels, actions):
        self.seen_shapes.append(tuple(actions.shape))
        return torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=actions.device)


def test_expand_action_to_model_dim_repeats_raw_control_for_frameskip():
    out = expand_action_to_model_dim(np.array([0.1, -0.2, 0.3], dtype=np.float32), action_dim=15)

    assert out.shape == (15,)
    assert np.allclose(out.reshape(5, 3), np.array([[0.1, -0.2, 0.3]] * 5, dtype=np.float32))


def test_flatten_model_action_to_block_recovers_frameskip_controls():
    flat = np.array(
        [
            0.0,
            1.0,
            2.0,
            0.2,
            -0.5,
            -1.0,
            0.4,
            0.0,
            0.3,
            0.6,
            0.5,
            0.0,
            2.0,
            -2.0,
            0.8,
        ],
        dtype=np.float32,
    )

    block = flatten_model_action_to_block(flat, action_dim=15)

    assert block.shape == (5, 3)
    assert block[0].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert block[1].tolist() == pytest.approx([0.2, -0.5, 0.0])
    assert block[-1].tolist() == pytest.approx([1.0, -1.0, 0.8])


def test_cem_score_samples_keeps_history_action_dim_frameskip_times_three():
    model = DummyModel()
    planner = CEMPlanner(planner_cfg())
    pixels = torch.zeros(1, 3, 3, 8, 8)
    history_actions = torch.zeros(1, 3, 15)
    samples = np.zeros((2, 2, 3), dtype=np.float32)

    costs = planner.score_samples(model, pixels, history_actions, samples, torch.device("cpu"))

    assert costs.shape == (2,)
    assert model.seen_shapes == [(1, 3, 15), (1, 3, 15)]


def test_cem_score_samples_rejects_action_dim_mismatch():
    model = DummyModel()
    planner = CEMPlanner(planner_cfg())
    pixels = torch.zeros(1, 3, 3, 8, 8)
    history_actions = torch.zeros(1, 3, 3)
    samples = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="does not match model action_dim"):
        planner.score_samples(model, pixels, history_actions, samples, torch.device("cpu"))


def test_cem_planner_applies_configured_bounds_and_exclusive_throttle_brake():
    cfg = planner_cfg()
    cfg.update(
        {
            "action_low": [0.2, -0.1, 0.0],
            "action_high": [0.8, 0.1, 0.7],
            "exclusive_throttle_brake": True,
        }
    )
    planner = CEMPlanner(cfg)

    samples = np.array([[[1.0, 0.2, 0.5], [0.1, -0.2, 0.6]]], dtype=np.float32)
    clipped = np.clip(samples, planner.low.reshape(1, 1, 3), planner.high.reshape(1, 1, 3))
    sanitized = planner._sanitize_samples(clipped)

    assert sanitized[0, 0].tolist() == pytest.approx([0.8, 0.1, 0.0])
    assert sanitized[0, 1].tolist() == pytest.approx([0.0, -0.1, 0.6])


def test_lane_keep_steer_corrects_positive_lane_offset_left():
    eval_cfg = {
        "target_speed_kmh": 20.0,
        "lane_keep": {
            "lane_offset_gain": 0.2,
            "heading_error_gain": 1.0,
            "steer_limit": 0.1,
        },
    }

    assert lane_keep_steer(1.0, 0.0, eval_cfg) == pytest.approx(-0.1)
    assert lane_keep_steer(-1.0, 0.0, eval_cfg) == pytest.approx(0.1)


def test_lane_keep_action_uses_speed_control_and_brake_when_overspeeding():
    eval_cfg = {
        "target_speed_kmh": 18.0,
        "lane_keep": {
            "throttle": 0.4,
            "speed_kp": 0.05,
            "throttle_min": 0.1,
            "throttle_max": 0.7,
            "overspeed_margin_mps": 1.0,
            "brake_kp": 0.1,
            "steer_limit": 0.2,
        },
    }

    slow = lane_keep_action(0.0, 0.0, speed_mps=2.0, eval_cfg=eval_cfg)
    fast = lane_keep_action(0.0, 0.0, speed_mps=8.0, eval_cfg=eval_cfg)

    assert slow.tolist() == pytest.approx([0.55, 0.0, 0.0])
    assert fast[0] == pytest.approx(0.0)
    assert fast[2] > 0.0


def test_model_speed_governor_caps_throttle_and_raises_brake():
    eval_cfg = {
        "target_speed_kmh": 18.0,
        "lane_keep": {
            "govern_model_speed": True,
            "throttle": 0.4,
            "speed_kp": 0.05,
            "overspeed_margin_mps": 1.0,
            "brake_kp": 0.1,
        },
    }

    governed = maybe_govern_model_speed(
        np.array([0.7, 0.0, 0.0], dtype=np.float32),
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        speed_mps=8.0,
        eval_cfg=eval_cfg,
    )

    assert governed[0] == pytest.approx(0.0)
    assert governed[2] > 0.0


def test_write_metrics_outputs_episode_csv_and_summary(tmp_path):
    row = DrivingEpisodeMetrics(
        route_length_m=100.0,
        route_progress_m=80.0,
        collision_count=1,
        first_infraction_distance_m=30.0,
    )

    summary = write_metrics(tmp_path, [row])

    assert summary["mean_infraction_free_distance_m"] == 30.0
    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["route_completion_pct"] == "80.0"
    assert rows[0]["infraction_free_distance_m"] == "30.0"
    assert rows[0]["collision_count"] == "1"


def test_write_action_trace_outputs_stepwise_control_csv(tmp_path):
    path = write_action_trace(
        tmp_path,
        2,
        [
            {
                "step": 1,
                "route_progress_m": 0.5,
                "speed_mps": 1.0,
                "throttle": 0.4,
                "steer": -0.01,
                "brake": 0.0,
                "lane_offset_m": 0.12,
                "heading_error_rad": -0.03,
                "offroad": 0,
                "speed_limit_violation": 0,
                "blocked": 0,
                "collision_count": 0,
            }
        ],
    )

    assert path == tmp_path / "actions_episode_002.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["throttle"] == "0.4"
    assert rows[0]["steer"] == "-0.01"
    assert rows[0]["lane_offset_m"] == "0.12"
    assert rows[0]["heading_error_rad"] == "-0.03"
    assert rows[0]["speed_limit_violation"] == "0"


def test_autopilot_dry_run_does_not_require_checkpoint(tmp_path):
    cfg = {
        "eval": {
            "eval_episodes": 2,
            "route_cap_m": 100.0,
            "output_dir": str(tmp_path),
        }
    }

    out = run_dry_eval(cfg, checkpoint_path=None, policy="autopilot")

    assert out["status"] == "config_loaded"
    assert out["policy"] == "autopilot"
    assert out["episodes"] == 2


def test_constant_policy_dry_run_does_not_require_checkpoint(tmp_path):
    cfg = {
        "eval": {
            "eval_episodes": 1,
            "route_cap_m": 50.0,
            "output_dir": str(tmp_path),
        }
    }

    out = run_dry_eval(cfg, checkpoint_path=None, policy=normalize_policy("constant"))

    assert out["status"] == "config_loaded"
    assert out["policy"] == "constant"
    assert out["route_cap_m"] == 50.0


def test_lane_keep_policy_dry_run_does_not_require_checkpoint(tmp_path):
    cfg = {
        "eval": {
            "eval_episodes": 1,
            "route_cap_m": 80.0,
            "output_dir": str(tmp_path),
        }
    }

    out = run_dry_eval(cfg, checkpoint_path=None, policy=normalize_policy("lane_keep"))

    assert out["status"] == "config_loaded"
    assert out["policy"] == "lane_keep"
    assert out["route_cap_m"] == 80.0


def test_model_action_policy_normalizes_to_model_backed_action():
    assert normalize_policy("action") == "model_action"
    assert normalize_policy("model_action_lane_keep") == "model_action_lane_keep"


def test_checkpoint_argument_preserves_configured_action_policy():
    cfg = {"eval": {"policy": "model_action"}}
    args = SimpleNamespace(baseline=None, policy=None, checkpoint="best.pt")

    assert resolve_policy(cfg, args) == "model_action"


def test_checkpoint_argument_defaults_to_planning_policy_without_config_policy():
    cfg = {"eval": {}}
    args = SimpleNamespace(baseline=None, policy=None, checkpoint="best.pt")

    assert resolve_policy(cfg, args) == "model"


def test_load_model_preserves_temporal_action_head_config(tmp_path, monkeypatch):
    class DummyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, interpolate_pos_encoding=True):
            batch = pixels.shape[0]
            values = torch.arange(batch * 8, dtype=pixels.dtype, device=pixels.device).reshape(batch, 2, 4)
            return SimpleNamespace(last_hidden_state=values)

    monkeypatch.setattr(DrivingLeWM, "_make_vit", staticmethod(lambda _cfg: DummyEncoder()))
    payload_cfg = {
        "data": {"frameskip": 1, "history_size": 3, "image_size": 8},
        "model": {
            "encoder_scale": "tiny",
            "patch_size": 4,
            "embed_dim": 4,
            "predictor_depth": 1,
            "predictor_heads": 1,
            "predictor_mlp_dim": 8,
            "dropout": 0.1,
            "pred_weight": 0.0,
            "sigreg_weight": 0.0,
            "use_aux_head": False,
            "aux_weight": 0.0,
            "pred_aux_weight": 0.0,
            "action_weight": 1.0,
            "pred_action_weight": 0.0,
            "use_temporal_action_head": True,
            "temporal_action_include_history_actions": True,
        },
    }
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
            use_aux_head=False,
            pred_weight=0.0,
            sigreg_weight=0.0,
            aux_weight=0.0,
            action_weight=1.0,
            pred_action_weight=0.0,
            use_temporal_action_head=True,
            temporal_action_include_history_actions=True,
        )
    )
    checkpoint = tmp_path / "temporal.pt"
    torch.save({"cfg": payload_cfg, "model": model.state_dict()}, checkpoint)

    loaded = load_model(checkpoint)

    assert loaded.cfg.use_temporal_action_head is True
    assert loaded.action_head[0].normalized_shape == (21,)


def test_apply_cli_overrides_supports_short_eval_knobs(tmp_path):
    cfg = {
        "eval": {
            "eval_episodes": 5,
            "output_dir": "outputs/full",
            "route_cap_m": 500.0,
            "max_episode_seconds": 120.0,
            "save_frames": False,
        },
        "planner": {
            "horizon": 8,
            "num_samples": 512,
            "iterations": 4,
        },
    }
    args = SimpleNamespace(
        episodes=1,
        output_dir=tmp_path,
        max_episode_seconds=8.0,
        route_cap_m=50.0,
        planner_horizon=3,
        planner_samples=16,
        planner_iterations=1,
        save_frames=True,
    )

    apply_cli_overrides(cfg, args)

    assert cfg["eval"]["eval_episodes"] == 1
    assert cfg["eval"]["output_dir"] == str(tmp_path)
    assert cfg["eval"]["max_episode_seconds"] == 8.0
    assert cfg["eval"]["route_cap_m"] == 50.0
    assert cfg["eval"]["save_frames"] is True
    assert cfg["planner"]["horizon"] == 3
    assert cfg["planner"]["num_samples"] == 16
    assert cfg["planner"]["iterations"] == 1
