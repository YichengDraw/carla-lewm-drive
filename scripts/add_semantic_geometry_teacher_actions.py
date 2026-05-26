from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from carla_lewm_drive.closed_loop_eval.evaluate import filter_perception_lane_state
from carla_lewm_drive.semantic_geometry import load_semantic_geometry_lane_model, semantic_geometry_features_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add semantic-geometry teacher actions to existing CARLA HDF5 files.")
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--key", default="semgeom_rf_teacher_action_lg300_hg500_sl065")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-speed-mps", type=float, default=1.5)
    parser.add_argument("--throttle", type=float, default=0.13)
    parser.add_argument("--speed-kp", type=float, default=0.08)
    parser.add_argument("--throttle-min", type=float, default=0.02)
    parser.add_argument("--throttle-max", type=float, default=0.22)
    parser.add_argument("--overspeed-margin-mps", type=float, default=0.35)
    parser.add_argument("--brake-kp", type=float, default=0.14)
    parser.add_argument("--steer-limit", type=float, default=0.65)
    parser.add_argument("--lane-gain", type=float, default=3.0)
    parser.add_argument("--heading-gain", type=float, default=5.0)
    parser.add_argument("--filter-alpha", type=float, default=0.55)
    parser.add_argument("--filter-heading-alpha", type=float, default=0.65)
    return parser.parse_args()


def _lane_keep_action(lane: float, heading: float, speed: float, args: argparse.Namespace) -> np.ndarray:
    steer = -float(args.lane_gain) * float(lane) - float(args.heading_gain) * float(heading)
    steer = float(np.clip(steer, -abs(float(args.steer_limit)), abs(float(args.steer_limit))))
    speed_error = float(args.target_speed_mps) - float(speed)
    throttle = float(np.clip(float(args.throttle) + float(args.speed_kp) * speed_error, args.throttle_min, args.throttle_max))
    brake = 0.0
    if speed_error < -float(args.overspeed_margin_mps):
        throttle = 0.0
        brake = float(
            np.clip(float(args.brake_kp) * (-speed_error - float(args.overspeed_margin_mps)), 0.0, 0.35)
        )
    return np.asarray([throttle, steer, brake], dtype=np.float32)


def _filter_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "perception_lane_filter": {
            "enabled": True,
            "alpha": float(args.filter_alpha),
            "heading_alpha": float(args.filter_heading_alpha),
            "small_flip_abs": 0.06,
            "max_flip_abs": 0.18,
            "flip_decay": 0.98,
            "max_abs_lane": 2.0,
            "max_abs_heading": 0.7,
        }
    }


def add_teacher_actions(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = load_semantic_geometry_lane_model(args.model)
    with h5py.File(path, "r+") as h5:
        if args.key in h5:
            if not args.overwrite:
                return {"dataset": str(path), "status": "exists", "key": args.key, "frames": int(h5[args.key].shape[0])}
            del h5[args.key]
        pixels = h5["pixels"]
        speed = np.asarray(h5["speed_mps"], dtype=np.float32) if "speed_mps" in h5 else np.zeros((pixels.shape[0],), dtype=np.float32)
        out = h5.create_dataset(args.key, shape=(pixels.shape[0], 3), dtype=np.float32, chunks=True)
        pred = np.zeros((pixels.shape[0], 2), dtype=np.float32)
        for start in range(0, pixels.shape[0], int(args.batch_size)):
            stop = min(pixels.shape[0], start + int(args.batch_size))
            feats = semantic_geometry_features_batch(
                pixels[start:stop],
                rows=model.rows,
                band=model.band,
                image_size=model.image_size,
            )
            pred[start:stop] = model.predict_features(feats)
        filter_cfg = _filter_cfg(args)
        for offset, length in zip(np.asarray(h5["ep_offset"], dtype=np.int64), np.asarray(h5["ep_len"], dtype=np.int64)):
            state = None
            for idx in range(int(offset), int(offset + length)):
                lane, heading, state, _active = filter_perception_lane_state(
                    float(pred[idx, 0]),
                    float(pred[idx, 1]),
                    state,
                    filter_cfg,
                )
                out[idx] = _lane_keep_action(lane, heading, float(speed[idx]), args)
        h5.attrs[f"{args.key}_source"] = json.dumps(
            {
                "model": str(args.model),
                "lane_gain": float(args.lane_gain),
                "heading_gain": float(args.heading_gain),
                "steer_limit": float(args.steer_limit),
                "target_speed_mps": float(args.target_speed_mps),
            },
            sort_keys=True,
        )
        actions = np.asarray(out)
    return {
        "dataset": str(path),
        "status": "written",
        "key": args.key,
        "frames": int(actions.shape[0]),
        "steer_abs_p95": float(np.quantile(np.abs(actions[:, 1]), 0.95)),
        "steer_abs_max": float(np.max(np.abs(actions[:, 1]))),
        "throttle_mean": float(np.mean(actions[:, 0])),
        "brake_mean": float(np.mean(actions[:, 2])),
    }


def main() -> None:
    args = parse_args()
    reports = [add_teacher_actions(path, args) for path in args.dataset]
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
