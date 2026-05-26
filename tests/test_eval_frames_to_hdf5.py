import csv

import h5py
import numpy as np
import pytest
from PIL import Image

from scripts.eval_frames_to_hdf5 import convert, write_hdf5


def write_eval_trace(eval_dir, *, applied=(0.11, 0.22, 0.33), next_applied=(0.44, -0.55, 0.66)):
    frame_dir = eval_dir / "frames" / "episode_000"
    frame_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frame_dir / "step_00001.jpg")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frame_dir / "step_00002.jpg")
    with (eval_dir / "actions_episode_000.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "route_progress_m",
                "speed_mps",
                "throttle",
                "steer",
                "brake",
                "lane_offset_m",
                "heading_error_rad",
                "offroad",
                "collision_count",
                "red_light",
                "blocked",
            ],
        )
        writer.writeheader()
        for step, row_applied in [(1, applied), (2, next_applied)]:
            writer.writerow(
                {
                    "step": step,
                    "route_progress_m": 2.5 + step,
                    "speed_mps": 2.0,
                    "throttle": row_applied[0],
                    "steer": row_applied[1],
                    "brake": row_applied[2],
                    "lane_offset_m": 1.0,
                    "heading_error_rad": 0.0,
                    "offroad": 0,
                    "collision_count": 0,
                    "red_light": 0,
                    "blocked": 0,
                }
            )


def eval_cfg():
    return {
        "route_spawn_indices": [3],
        "target_speed_kmh": 18.0,
        "lane_keep": {
            "throttle": 0.4,
            "speed_kp": 0.05,
            "throttle_min": 0.1,
            "throttle_max": 0.7,
            "overspeed_margin_mps": 1.0,
            "brake_kp": 0.1,
            "lane_offset_gain": 0.2,
            "heading_error_gain": 1.0,
            "steer_limit": 0.3,
        },
    }


def test_convert_can_use_applied_trace_actions(tmp_path):
    write_eval_trace(tmp_path)

    data = convert(tmp_path, eval_cfg(), action_source="applied")

    assert data["action"].shape[0] == 1
    assert data["action"][0].tolist() == pytest.approx([0.44, -0.55, 0.66])


def test_convert_applied_can_store_teacher_action_target(tmp_path):
    write_eval_trace(tmp_path)

    data = convert(tmp_path, eval_cfg(), action_source="applied", store_teacher_action=True)

    assert data["action"].shape[0] == 1
    assert data["action"][0].tolist() == pytest.approx([0.44, -0.55, 0.66])
    assert data["teacher_action"][0, 1] == pytest.approx(-0.2)
    assert data["teacher_action"][0].tolist() != pytest.approx(data["action"][0].tolist())


def test_convert_teacher_action_keeps_lane_keep_label(tmp_path):
    write_eval_trace(tmp_path)

    data = convert(tmp_path, eval_cfg(), action_source="teacher")

    assert data["action"].shape[0] == 2
    assert data["action"][0, 1] == pytest.approx(-0.2)
    assert data["action"][0].tolist() != pytest.approx([0.11, 0.22, 0.33])


def test_write_hdf5_records_action_source(tmp_path):
    write_eval_trace(tmp_path)
    data = convert(tmp_path, eval_cfg(), action_source="applied", store_teacher_action=True)
    output = tmp_path / "out.h5"

    write_hdf5(data, output, overwrite=False, action_source="applied")

    with h5py.File(output, "r") as f:
        assert f.attrs["action_source"] == "applied"
        assert f.attrs["action_alignment"] == "next_row_applied"
        assert f.attrs["teacher_action_alignment"] == "same_frame_teacher"
        assert "teacher_action" in f
