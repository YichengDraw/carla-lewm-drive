from pathlib import Path

import h5py
import numpy as np
import pytest

from carla_lewm_drive.carla_collect.collect_dataset import (
    FrameRecord,
    RecoveryPerturbation,
    append_episode,
    lane_keep_action,
    select_recovery_perturbation,
    write_hdf5,
)


def empty_output() -> dict[str, list]:
    return {
        key: []
        for key in (
            "ep_len",
            "ep_offset",
            "pixels",
            "action",
            "state",
            "proprio",
            "ep_idx",
            "step_idx",
            "route_id",
            "weather_id",
            "teacher_policy_id",
            "initial_lateral_offset_m",
            "initial_yaw_offset_deg",
            "timestamp",
            "speed_mps",
            "route_progress_m",
            "lane_offset_m",
            "heading_error_rad",
            "collision",
            "offroad",
            "red_light",
            "blocked",
        )
    }


def test_select_recovery_perturbation_cycles_offsets_then_yaw():
    cfg = {
        "enabled": True,
        "initial_lateral_offsets_m": [0.0, -0.5, 0.5],
        "initial_yaw_offsets_deg": [0.0, 4.0],
    }

    assert select_recovery_perturbation(0, cfg) == RecoveryPerturbation(0.0, 0.0)
    assert select_recovery_perturbation(2, cfg) == RecoveryPerturbation(0.5, 0.0)
    assert select_recovery_perturbation(3, cfg) == RecoveryPerturbation(0.0, 4.0)


def test_lane_keep_action_corrects_offset_and_limits_speed():
    cfg = {
        "scenario": {"target_speed_kmh": 20.0},
        "lane_keep": {
            "target_speed_mps": 5.0,
            "throttle": 0.4,
            "speed_kp": 0.05,
            "throttle_min": 0.1,
            "throttle_max": 0.7,
            "overspeed_margin_mps": 1.0,
            "brake_kp": 0.1,
            "steer_limit": 0.2,
            "lane_offset_gain": 0.2,
            "heading_error_gain": 1.0,
        },
    }

    action = lane_keep_action(1.0, 0.0, speed_mps=2.0, cfg=cfg)
    fast = lane_keep_action(-1.0, 0.0, speed_mps=8.0, cfg=cfg)

    assert action.tolist() == pytest.approx([0.55, -0.2, 0.0])
    assert fast[0] == pytest.approx(0.0)
    assert fast[1] == pytest.approx(0.2)
    assert fast[2] > 0.0


def test_append_episode_records_teacher_and_perturbation_metadata():
    output = empty_output()
    perturbation = RecoveryPerturbation(lateral_offset_m=0.7, yaw_offset_deg=-4.0)

    next_step = append_episode(
        {},
        episode_idx=2,
        pixels=[np.zeros((4, 4, 3), dtype=np.uint8)],
        actions=[np.array([0.4, -0.1, 0.0], dtype=np.float32)],
        state=[np.zeros(6, dtype=np.float32)],
        proprio=[np.zeros(3, dtype=np.float32)],
        records=[
            FrameRecord(
                timestamp=1.0,
                speed_mps=2.0,
                route_progress_m=3.0,
                lane_offset_m=0.7,
                heading_error_rad=-0.1,
                collision=0.0,
                offroad=0.0,
                red_light=0.0,
                blocked=0.0,
            )
        ],
        route_id=10,
        weather_id=0,
        teacher_policy_id=1,
        perturbation=perturbation,
        output=output,
        global_step=5,
    )

    assert next_step == 6
    assert output["teacher_policy_id"] == [1]
    assert output["initial_lateral_offset_m"] == [0.7]
    assert output["initial_yaw_offset_deg"] == [-4.0]
    assert output["route_id"] == [10]


def test_write_hdf5_can_write_uncompressed_fast_pixels(tmp_path: Path):
    output = empty_output()
    append_episode(
        {},
        episode_idx=0,
        pixels=[np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)],
        actions=[np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
        state=[np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)],
        proprio=[np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
        records=[
            FrameRecord(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            FrameRecord(2.0, 1.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        route_id=3,
        weather_id=0,
        teacher_policy_id=1,
        perturbation=RecoveryPerturbation(0.35, 4.0),
        output=output,
        global_step=0,
    )

    path = tmp_path / "fast.h5"
    write_hdf5(path, output, {"pixel_compression": "none"})

    with h5py.File(path, "r") as f:
        assert f["pixels"].compression is None
        assert f["teacher_policy_id"][:].tolist() == [1, 1]
        assert f["initial_lateral_offset_m"][:].tolist() == pytest.approx([0.35, 0.35])
