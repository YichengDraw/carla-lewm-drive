from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from carla_lewm_drive.closed_loop_eval.evaluate import lane_keep_action, runtime_eval_config
from carla_lewm_drive.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert saved closed-loop eval frames into a DAgger-style HDF5 dataset.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action-source",
        choices=["teacher", "applied"],
        default="teacher",
        help=(
            "Use teacher lane-keep actions computed from each saved frame, or next-row applied trace actions "
            "aligned as current frame -> next simulator tick."
        ),
    )
    parser.add_argument(
        "--store-teacher-action",
        action="store_true",
        help="Also store same-frame lane-keep teacher actions in teacher_action without replacing the transition action.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_path(eval_dir: Path, episode_idx: int, step: int) -> Path:
    return eval_dir / "frames" / f"episode_{episode_idx:03d}" / f"step_{step:05d}.jpg"


def route_for_episode(eval_cfg: dict[str, Any], episode_idx: int) -> int:
    routes = [int(x) for x in eval_cfg["route_spawn_indices"]]
    return int(routes[episode_idx % len(routes)])


def row_applied_action(row: dict[str, str]) -> np.ndarray:
    try:
        return np.asarray([float(row["throttle"]), float(row["steer"]), float(row["brake"])], dtype=np.float32)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise KeyError(f"Trace row is missing {missing!r}; applied action source requires throttle, steer, and brake") from exc


def convert(
    eval_dir: Path,
    eval_cfg: dict[str, Any],
    *,
    action_source: str = "teacher",
    store_teacher_action: bool = False,
) -> dict[str, Any]:
    if action_source not in {"teacher", "applied"}:
        raise ValueError(f"Unknown action_source {action_source!r}; expected 'teacher' or 'applied'")

    pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    state: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    speed: list[float] = []
    progress: list[float] = []
    lane: list[float] = []
    heading: list[float] = []
    collision: list[float] = []
    offroad: list[float] = []
    red_light: list[float] = []
    blocked: list[float] = []
    ep_idx: list[int] = []
    step_idx: list[int] = []
    route_id: list[int] = []
    ep_len: list[int] = []

    for trace_path in sorted(eval_dir.glob("actions_episode_*.csv")):
        episode_idx = int(trace_path.stem.rsplit("_", 1)[-1])
        route = route_for_episode(eval_cfg, episode_idx)
        rows = read_trace(trace_path)
        episode_frames = 0
        for row_idx, row in enumerate(rows):
            step = int(float(row["step"]))
            image_path = frame_path(eval_dir, episode_idx, step)
            if not image_path.exists():
                continue
            rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            speed_mps = float(row["speed_mps"])
            lane_offset = float(row["lane_offset_m"])
            heading_error = float(row["heading_error_rad"])
            teacher_action = lane_keep_action(lane_offset, heading_error, speed_mps, eval_cfg)
            if action_source == "teacher":
                action = teacher_action
            else:
                if row_idx + 1 >= len(rows):
                    continue
                next_row = rows[row_idx + 1]
                next_step = int(float(next_row["step"]))
                if next_step != step + 1:
                    continue
                action = row_applied_action(next_row)

            pixels.append(rgb)
            actions.append(action.astype(np.float32))
            if store_teacher_action:
                teacher_actions.append(teacher_action.astype(np.float32))
            state.append(np.asarray([0.0, 0.0, 0.0, speed_mps, float(row["route_progress_m"]), lane_offset], dtype=np.float32))
            proprio.append(np.asarray([speed_mps, lane_offset, heading_error], dtype=np.float32))
            speed.append(speed_mps)
            progress.append(float(row["route_progress_m"]))
            lane.append(lane_offset)
            heading.append(heading_error)
            collision.append(float(row.get("collision_count", 0.0)))
            offroad.append(float(row.get("offroad", 0.0)))
            red_light.append(float(row.get("red_light", 0.0)))
            blocked.append(float(row.get("blocked", 0.0)))
            ep_idx.append(len(ep_len))
            step_idx.append(episode_frames)
            route_id.append(route)
            episode_frames += 1
        if episode_frames:
            ep_len.append(episode_frames)

    if not pixels:
        raise ValueError(f"No saved frames found under {eval_dir}")

    ep_offset = np.cumsum([0, *ep_len[:-1]], dtype=np.int64)
    data = {
        "pixels": np.asarray(pixels, dtype=np.uint8),
        "action": np.asarray(actions, dtype=np.float32),
        "state": np.asarray(state, dtype=np.float32),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "speed_mps": np.asarray(speed, dtype=np.float32),
        "route_progress_m": np.asarray(progress, dtype=np.float32),
        "lane_offset_m": np.asarray(lane, dtype=np.float32),
        "heading_error_rad": np.asarray(heading, dtype=np.float32),
        "collision": np.asarray(collision, dtype=np.float32),
        "offroad": np.asarray(offroad, dtype=np.float32),
        "red_light": np.asarray(red_light, dtype=np.float32),
        "blocked": np.asarray(blocked, dtype=np.float32),
        "ep_idx": np.asarray(ep_idx, dtype=np.int32),
        "step_idx": np.asarray(step_idx, dtype=np.int32),
        "route_id": np.asarray(route_id, dtype=np.int32),
        "weather_id": np.zeros(len(pixels), dtype=np.int32),
        "teacher_policy_id": np.ones(len(pixels), dtype=np.int32),
        "initial_lateral_offset_m": np.zeros(len(pixels), dtype=np.float32),
        "initial_yaw_offset_deg": np.zeros(len(pixels), dtype=np.float32),
        "ep_len": np.asarray(ep_len, dtype=np.int32),
        "ep_offset": ep_offset.astype(np.int32),
    }
    if store_teacher_action:
        data["teacher_action"] = np.asarray(teacher_actions, dtype=np.float32)
    return data


def action_alignment(action_source: str) -> str:
    return "same_frame_teacher" if action_source == "teacher" else "next_row_applied"


def write_hdf5(data: dict[str, np.ndarray], path: Path, *, overwrite: bool, action_source: str = "teacher") -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if key == "pixels":
                chunks = (min(256, value.shape[0]), *value.shape[1:])
                f.create_dataset(key, data=value, chunks=chunks)
            else:
                f.create_dataset(key, data=value)
        f.attrs["format"] = "carla_lewm_drive_eval_dagger_v1"
        f.attrs["action_source"] = action_source
        f.attrs["action_alignment"] = action_alignment(action_source)
        if "teacher_action" in data:
            f.attrs["teacher_action_alignment"] = "same_frame_teacher"


def main() -> None:
    args = parse_args()
    cfg = runtime_eval_config(load_yaml(args.eval_config))
    data = convert(
        args.eval_dir,
        cfg,
        action_source=args.action_source,
        store_teacher_action=args.store_teacher_action,
    )
    write_hdf5(data, args.output, overwrite=args.overwrite, action_source=args.action_source)
    report = {
        "output": str(args.output),
        "action_source": args.action_source,
        "action_alignment": action_alignment(args.action_source),
        "has_teacher_action": "teacher_action" in data,
        "frames": int(data["pixels"].shape[0]),
        "episodes": int(data["ep_len"].shape[0]),
        "lane_offset_abs_gt_0p5": float(np.mean(np.abs(data["lane_offset_m"]) > 0.5)),
        "lane_offset_abs_gt_1p0": float(np.mean(np.abs(data["lane_offset_m"]) > 1.0)),
        "offroad_fraction": float(np.mean(data["offroad"] > 0.0)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
