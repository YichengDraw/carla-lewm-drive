from __future__ import annotations

import argparse
import json
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from carla_lewm_drive.config import load_yaml


@dataclass
class FrameRecord:
    timestamp: float
    speed_mps: float
    route_progress_m: float
    lane_offset_m: float
    heading_error_rad: float
    collision: float
    offroad: float
    red_light: float
    blocked: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a CARLA HDF5 dataset for Driving-LeWM.")
    parser.add_argument("--config", type=Path, default=Path("configs/d0_smoke.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    return parser.parse_args()


def require_carla():
    try:
        import carla
    except Exception as exc:
        raise RuntimeError(
            "CARLA Python API is not installed. Install the wheel matching the CARLA 0.9.16 server."
        ) from exc
    return carla


def camera_transform(carla, cfg: dict[str, Any]):
    t = cfg["camera"]["transform"]
    return carla.Transform(
        carla.Location(x=float(t["x"]), y=float(t["y"]), z=float(t["z"])),
        carla.Rotation(pitch=float(t["pitch"]), yaw=float(t["yaw"]), roll=float(t["roll"])),
    )


def make_camera(carla, world, ego, cfg: dict[str, Any]):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(cfg["camera"]["image_width"])))
    bp.set_attribute("image_size_y", str(int(cfg["camera"]["image_height"])))
    bp.set_attribute("fov", str(float(cfg["camera"]["fov"])))
    return world.spawn_actor(bp, camera_transform(carla, cfg), attach_to=ego)


def make_collision_sensor(carla, world, ego):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    return world.spawn_actor(bp, carla.Transform(), attach_to=ego)


def simplify_traffic_lights(carla, world, scenario: dict[str, Any]) -> None:
    if not bool(scenario.get("force_green_lights", False)):
        return
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        actor.set_state(carla.TrafficLightState.Green)
        actor.freeze(True)


def spawn_ego(carla, world, spawn_index: int):
    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    bp = blueprints[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points")
    transform = spawn_points[spawn_index % len(spawn_points)]
    ego = world.try_spawn_actor(bp, transform)
    if ego is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at spawn index {spawn_index}")
    return ego


def safe_stop_actor(actor: Any) -> None:
    try:
        if bool(getattr(actor, "is_alive", True)):
            actor.stop()
    except Exception:
        pass


def safe_destroy_actor(actor: Any) -> None:
    try:
        if bool(getattr(actor, "is_alive", True)):
            actor.destroy()
    except Exception:
        pass


def listen_queue(sensor):
    q: queue.Queue = queue.Queue()
    sensor.listen(q.put)
    return q


def read_rgb(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3][:, :, ::-1].copy()


def lane_metrics(carla, world_map, transform) -> tuple[float, float, float]:
    driving_waypoint = world_map.get_waypoint(
        transform.location,
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    projected_waypoint = driving_waypoint or world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if projected_waypoint is None:
        return 0.0, 0.0, 1.0
    dx = transform.location.x - projected_waypoint.transform.location.x
    dy = transform.location.y - projected_waypoint.transform.location.y
    yaw = np.deg2rad(projected_waypoint.transform.rotation.yaw)
    right_x = np.cos(yaw + np.pi / 2)
    right_y = np.sin(yaw + np.pi / 2)
    lane_offset = dx * right_x + dy * right_y
    heading_error = np.deg2rad(transform.rotation.yaw - projected_waypoint.transform.rotation.yaw)
    heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
    lane_half_width = max(float(projected_waypoint.lane_width) / 2.0, 0.1)
    offroad = float(driving_waypoint is None or abs(lane_offset) > lane_half_width + 0.4)
    return float(lane_offset), heading_error, offroad


def vehicle_speed(vehicle) -> float:
    v = vehicle.get_velocity()
    return float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)


def summarize_episode(records: list[FrameRecord]) -> dict[str, Any]:
    frames = len(records)
    if frames == 0:
        return {
            "frames": 0,
            "progress_m": 0.0,
            "collision_frames": 0,
            "offroad_frames": 0,
            "red_light_frames": 0,
            "blocked_frames": 0,
            "offroad_fraction": 1.0,
            "red_light_fraction": 1.0,
            "blocked_fraction": 1.0,
            "mean_speed_mps": 0.0,
            "max_abs_lane_offset_m": 0.0,
        }
    collision_frames = int(sum(record.collision > 0 for record in records))
    offroad_frames = int(sum(record.offroad > 0 for record in records))
    red_light_frames = int(sum(record.red_light > 0 for record in records))
    blocked_frames = int(sum(record.blocked > 0 for record in records))
    return {
        "frames": frames,
        "progress_m": float(records[-1].route_progress_m),
        "collision_frames": collision_frames,
        "offroad_frames": offroad_frames,
        "red_light_frames": red_light_frames,
        "blocked_frames": blocked_frames,
        "offroad_fraction": float(offroad_frames / frames),
        "red_light_fraction": float(red_light_frames / frames),
        "blocked_fraction": float(blocked_frames / frames),
        "mean_speed_mps": float(np.mean([record.speed_mps for record in records])),
        "max_abs_lane_offset_m": float(max(abs(record.lane_offset_m) for record in records)),
    }


def episode_quality_failures(stats: dict[str, Any], quality_cfg: dict[str, Any]) -> list[str]:
    if not bool(quality_cfg.get("enabled", True)):
        return []
    failures = []
    if stats["progress_m"] < float(quality_cfg.get("min_progress_m", 0.0)):
        failures.append(f"progress_m={stats['progress_m']:.2f}")
    if stats["collision_frames"] > int(quality_cfg.get("max_collision_frames", 0)):
        failures.append(f"collision_frames={stats['collision_frames']}")
    if stats["offroad_fraction"] > float(quality_cfg.get("max_offroad_fraction", 1.0)):
        failures.append(f"offroad_fraction={stats['offroad_fraction']:.4f}")
    if stats["red_light_fraction"] > float(quality_cfg.get("max_red_light_fraction", 1.0)):
        failures.append(f"red_light_fraction={stats['red_light_fraction']:.4f}")
    if stats["blocked_fraction"] > float(quality_cfg.get("max_blocked_fraction", 1.0)):
        failures.append(f"blocked_fraction={stats['blocked_fraction']:.4f}")
    return failures


def append_episode(
    cfg: dict[str, Any],
    *,
    episode_idx: int,
    pixels: list[np.ndarray],
    actions: list[np.ndarray],
    state: list[np.ndarray],
    proprio: list[np.ndarray],
    records: list[FrameRecord],
    route_id: int,
    weather_id: int,
    output: dict[str, list[Any]],
    global_step: int,
) -> int:
    output["ep_len"].append(len(pixels))
    output["ep_offset"].append(global_step)
    output["pixels"].extend(pixels)
    output["action"].extend(actions)
    output["state"].extend(state)
    output["proprio"].extend(proprio)
    output["ep_idx"].extend([episode_idx] * len(pixels))
    output["step_idx"].extend(list(range(len(pixels))))
    output["route_id"].extend([route_id] * len(pixels))
    output["weather_id"].extend([weather_id] * len(pixels))
    for record in records:
        for key, value in asdict(record).items():
            output[key].append(value)
    return global_step + len(pixels)


def run_collection(cfg: dict[str, Any]) -> Path:
    carla = require_carla()
    scenario = cfg["scenario"]
    out_cfg = cfg["output"]
    dataset_path = Path(out_cfg["dataset_path"])
    metadata_path = Path(out_cfg["metadata_path"])
    recorder_path = Path(out_cfg["recorder_path"])
    for path in (dataset_path, metadata_path, recorder_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    client = carla.Client(scenario["host"], int(scenario["port"]))
    client.set_timeout(float(scenario["timeout_s"]))
    world = client.load_world(scenario["town"])
    traffic_manager = client.get_trafficmanager(int(scenario["traffic_manager_port"]))

    original_settings = world.get_settings()
    actors = []
    output: dict[str, list[Any]] = {
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
    global_step = 0
    recorder_started = False
    collection_complete = False
    try:
        settings = world.get_settings()
        settings.synchronous_mode = bool(scenario["synchronous_mode"])
        settings.fixed_delta_seconds = float(scenario["fixed_delta_seconds"])
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(scenario["traffic_manager_seed"]))
        simplify_traffic_lights(carla, world, scenario)
        if bool(scenario.get("record_carla_log", True)):
            client.start_recorder(str(recorder_path), True)
            recorder_started = True

        spawn_indices = list(scenario["spawn_point_indices"])
        episodes = int(scenario["episodes"])
        episode_seconds = float(scenario["episode_seconds"])
        if cfg.get("_override_episodes") is not None:
            episodes = int(cfg["_override_episodes"])
        if cfg.get("_override_seconds") is not None:
            episode_seconds = float(cfg["_override_seconds"])
        steps_per_episode = int(round(episode_seconds / float(scenario["fixed_delta_seconds"])))
        quality_cfg = dict(scenario.get("quality_gate", {}))
        max_attempts = int(quality_cfg.get("max_attempts", max(episodes, episodes * max(1, len(spawn_indices)) * 2)))
        rejected_episodes: list[dict[str, Any]] = []

        accepted_episodes = 0
        attempted_episodes = 0
        while accepted_episodes < episodes and attempted_episodes < max_attempts:
            spawn_index = spawn_indices[attempted_episodes % len(spawn_indices)]
            ego = spawn_ego(carla, world, spawn_index)
            actors.append(ego)
            camera = make_camera(carla, world, ego, cfg)
            actors.append(camera)
            collision_sensor = make_collision_sensor(carla, world, ego)
            actors.append(collision_sensor)
            collision_frame = {"last": -1}
            collision_sensor.listen(lambda event: collision_frame.__setitem__("last", event.frame))
            q = listen_queue(camera)
            ego.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(ego, 100.0 - float(scenario["target_speed_kmh"]) / 30.0 * 100.0)

            ep_pixels: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_state: list[np.ndarray] = []
            ep_proprio: list[np.ndarray] = []
            ep_records: list[FrameRecord] = []
            start_loc = ego.get_location()
            previous_loc = start_loc
            travelled_distance = 0.0
            blocked_count = 0

            for _ in range(steps_per_episode):
                frame_id = world.tick()
                image = q.get(timeout=5.0)
                while image.frame < frame_id:
                    image = q.get(timeout=5.0)
                transform = ego.get_transform()
                loc = transform.location
                speed = vehicle_speed(ego)
                if speed < 0.1:
                    blocked_count += 1
                else:
                    blocked_count = 0
                travelled_distance += float(
                    ((loc.x - previous_loc.x) ** 2 + (loc.y - previous_loc.y) ** 2 + (loc.z - previous_loc.z) ** 2) ** 0.5
                )
                previous_loc = loc
                progress = travelled_distance
                lane_offset, heading_error, offroad = lane_metrics(carla, world.get_map(), transform)
                control = ego.get_control()
                at_red = float(ego.is_at_traffic_light() and ego.get_traffic_light_state() == carla.TrafficLightState.Red)
                collided = float(collision_frame["last"] == frame_id)
                action = np.array([control.throttle, control.steer, control.brake], dtype=np.float32)
                state_vec = np.array([loc.x, loc.y, transform.rotation.yaw, speed, progress, lane_offset], dtype=np.float32)
                proprio_vec = np.array([speed, lane_offset, heading_error], dtype=np.float32)

                ep_pixels.append(read_rgb(image))
                ep_actions.append(action)
                ep_state.append(state_vec)
                ep_proprio.append(proprio_vec)
                ep_records.append(
                    FrameRecord(
                        timestamp=time.time(),
                        speed_mps=speed,
                        route_progress_m=progress,
                        lane_offset_m=lane_offset,
                        heading_error_rad=heading_error,
                        collision=collided,
                        offroad=offroad,
                        red_light=at_red,
                        blocked=float(blocked_count > int(3.0 / float(scenario["fixed_delta_seconds"]))),
                    )
                )

            stats = summarize_episode(ep_records)
            failures = episode_quality_failures(stats, quality_cfg)
            accepted = not failures
            event = {
                "event": "episode_quality",
                "attempt": attempted_episodes,
                "accepted_episode": accepted_episodes if accepted else None,
                "accepted": accepted,
                "spawn_index": spawn_index,
                "failures": failures,
                **stats,
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            if accepted:
                global_step = append_episode(
                    cfg,
                    episode_idx=accepted_episodes,
                    pixels=ep_pixels,
                    actions=ep_actions,
                    state=ep_state,
                    proprio=ep_proprio,
                    records=ep_records,
                    route_id=spawn_index,
                    weather_id=0,
                    output=output,
                    global_step=global_step,
                )
                accepted_episodes += 1
                finished_after_episode = accepted_episodes >= episodes
            else:
                rejected_episodes.append(event)
                finished_after_episode = False
            if not finished_after_episode:
                safe_stop_actor(collision_sensor)
                safe_stop_actor(camera)
                safe_destroy_actor(collision_sensor)
                safe_destroy_actor(camera)
                safe_destroy_actor(ego)
            actors.clear()
            attempted_episodes += 1
        if accepted_episodes < episodes:
            raise RuntimeError(
                f"Accepted only {accepted_episodes}/{episodes} episodes after {attempted_episodes} attempts; "
                f"last rejections={rejected_episodes[-3:]}"
            )
        collection_complete = True
    finally:
        cleanup_after_success = bool(scenario.get("cleanup_after_success", True))
        if (not collection_complete) or cleanup_after_success:
            if recorder_started:
                try:
                    client.stop_recorder()
                except Exception:
                    pass
            for actor in reversed(actors):
                safe_destroy_actor(actor)
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

    write_hdf5(dataset_path, output)
    metadata = {
        "scenario": scenario,
        "camera": cfg["camera"],
        "num_episodes": len(output["ep_len"]),
        "num_frames": len(output["pixels"]),
        "attempted_episodes": attempted_episodes,
        "rejected_episodes": rejected_episodes,
        "created_at": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dataset_path


def write_hdf5(path: Path, output: dict[str, list[Any]]) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.asarray(output["ep_len"], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.asarray(output["ep_offset"], dtype=np.int32))
        f.create_dataset("pixels", data=np.asarray(output["pixels"], dtype=np.uint8), compression="gzip", compression_opts=4)
        f.create_dataset("action", data=np.asarray(output["action"], dtype=np.float32))
        f.create_dataset("state", data=np.asarray(output["state"], dtype=np.float32))
        f.create_dataset("proprio", data=np.asarray(output["proprio"], dtype=np.float32))
        f.create_dataset("ep_idx", data=np.asarray(output["ep_idx"], dtype=np.int32))
        f.create_dataset("step_idx", data=np.asarray(output["step_idx"], dtype=np.int32))
        for key in (
            "route_id",
            "weather_id",
            "timestamp",
            "speed_mps",
            "route_progress_m",
            "lane_offset_m",
            "heading_error_rad",
            "collision",
            "offroad",
            "red_light",
            "blocked",
        ):
            dtype = np.int32 if key in {"route_id", "weather_id"} else np.float32
            f.create_dataset(key, data=np.asarray(output[key], dtype=dtype))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.episodes is not None:
        cfg["_override_episodes"] = args.episodes
    if args.seconds is not None:
        cfg["_override_seconds"] = args.seconds
    print(run_collection(cfg))


if __name__ == "__main__":
    main()
