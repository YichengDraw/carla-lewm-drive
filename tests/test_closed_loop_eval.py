import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carla_lewm_drive.closed_loop_eval.evaluate import (
    CEMPlanner,
    apply_cli_overrides,
    expand_action_to_model_dim,
    run_dry_eval,
    write_metrics,
)
from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics


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
