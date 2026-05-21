from __future__ import annotations

import argparse
import csv
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics, aggregate_metrics
from carla_lewm_drive.config import load_yaml
from carla_lewm_drive.driving_lewm.data import IMAGENET_MEAN, IMAGENET_STD
from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig
from carla_lewm_drive.schema import DrivingActionBounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop CARLA evaluation for Driving-LeWM or autopilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_d0.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy", choices=["checkpoint", "model", "expert", "autopilot", "constant"], default=None)
    parser.add_argument("--baseline", choices=["expert", "autopilot"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-episode-seconds", type=float, default=None)
    parser.add_argument("--route-cap-m", type=float, default=None)
    parser.add_argument("--planner-horizon", type=int, default=None)
    parser.add_argument("--planner-samples", type=int, default=None)
    parser.add_argument("--planner-iterations", type=int, default=None)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and checkpoint shape without launching CARLA.")
    return parser.parse_args()


def require_carla():
    try:
        import carla
    except Exception as exc:
        raise RuntimeError(
            "CARLA Python API is not installed. Install the wheel matching the CARLA 0.9.16 server."
        ) from exc
    return carla


def normalize_policy(policy: str | None) -> str:
    value = (policy or "checkpoint").lower()
    if value in {"checkpoint", "model"}:
        return "model"
    if value in {"expert", "autopilot"}:
        return "autopilot"
    if value == "constant":
        return "constant"
    raise ValueError(f"Unknown eval policy {policy!r}")


def resolve_policy(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> str:
    eval_cfg = cfg.get("eval", {})
    policy = eval_cfg.get("policy", "checkpoint")
    if args is not None:
        if args.baseline is not None:
            policy = args.baseline
        elif args.policy is not None:
            policy = args.policy
        elif args.checkpoint is not None:
            policy = "checkpoint"
    return normalize_policy(policy)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    eval_cfg = cfg.setdefault("eval", {})
    planner_cfg = cfg.setdefault("planner", {})
    if args.episodes is not None:
        eval_cfg["eval_episodes"] = int(args.episodes)
    if args.output_dir is not None:
        eval_cfg["output_dir"] = str(args.output_dir)
    if args.max_episode_seconds is not None:
        eval_cfg["max_episode_seconds"] = float(args.max_episode_seconds)
    if args.route_cap_m is not None:
        eval_cfg["route_cap_m"] = float(args.route_cap_m)
    if args.planner_horizon is not None:
        planner_cfg["horizon"] = int(args.planner_horizon)
    if args.planner_samples is not None:
        planner_cfg["num_samples"] = int(args.planner_samples)
    if args.planner_iterations is not None:
        planner_cfg["iterations"] = int(args.planner_iterations)
    if args.save_frames:
        eval_cfg["save_frames"] = True


def action_dim_to_frameskip(action_dim: int) -> int:
    action_dim = int(action_dim)
    if action_dim <= 0 or action_dim % 3 != 0:
        raise ValueError(f"DrivingLeWM action_dim must be frameskip * 3, got {action_dim}")
    return action_dim // 3


def expand_action_to_model_dim(action: np.ndarray, action_dim: int) -> np.ndarray:
    raw = np.asarray(action, dtype=np.float32).reshape(3)
    return np.tile(raw, action_dim_to_frameskip(action_dim)).astype(np.float32)


class CEMPlanner:
    def __init__(self, cfg: dict[str, Any], bounds: DrivingActionBounds | None = None) -> None:
        self.cfg = cfg
        self.bounds = bounds or DrivingActionBounds()
        self.low = np.asarray(cfg.get("action_low", self.bounds.low), dtype=np.float32)
        self.high = np.asarray(cfg.get("action_high", self.bounds.high), dtype=np.float32)
        self.low = np.maximum(self.low, np.asarray(self.bounds.low, dtype=np.float32))
        self.high = np.minimum(self.high, np.asarray(self.bounds.high, dtype=np.float32))
        if self.low.shape != (3,) or self.high.shape != (3,):
            raise ValueError("planner.action_low/action_high must each contain [throttle, steer, brake]")
        if np.any(self.low > self.high):
            raise ValueError(f"Invalid planner action bounds: low={self.low.tolist()} high={self.high.tolist()}")
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.target_speed_mps = float(cfg.get("target_speed_mps", 20.0 / 3.6))

    def propose(self, model: DrivingLeWM, pixels: torch.Tensor, history_actions: torch.Tensor, device: torch.device) -> np.ndarray:
        horizon = int(self.cfg["horizon"])
        num_samples = int(self.cfg["num_samples"])
        iterations = int(self.cfg["iterations"])
        elite = max(1, int(num_samples * float(self.cfg["elite_frac"])))
        mean = np.tile(np.asarray(self.cfg.get("action_mean", [0.35, 0.0, 0.0]), dtype=np.float32), (horizon, 1))
        std = np.tile(np.asarray(self.cfg["action_std"], dtype=np.float32), (horizon, 1))
        mean = np.clip(mean, self.low.reshape(1, 3), self.high.reshape(1, 3))
        min_std = float(self.cfg.get("min_action_std", 1e-4))
        best = mean[0].copy()
        for _ in range(iterations):
            samples = self.rng.normal(mean, std, size=(num_samples, horizon, 3)).astype(np.float32)
            samples = np.clip(samples, self.low.reshape(1, 1, 3), self.high.reshape(1, 1, 3))
            samples = self._sanitize_samples(samples)
            scores = self.score_samples(model, pixels, history_actions, samples, device)
            elite_idx = np.argsort(scores)[:elite]
            elite_samples = samples[elite_idx]
            mean = elite_samples.mean(axis=0)
            std = np.maximum(elite_samples.std(axis=0), min_std)
            best = samples[int(np.argmin(scores)), 0].copy()
        return self._sanitize_samples(np.clip(best, self.low, self.high).reshape(1, 1, 3))[0, 0].astype(np.float32)

    def _sanitize_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).copy()
        if bool(self.cfg.get("exclusive_throttle_brake", False)):
            throttle = samples[..., 0]
            brake = samples[..., 2]
            throttle_wins = throttle >= brake
            samples[..., 2] = np.where(throttle_wins, 0.0, brake)
            samples[..., 0] = np.where(throttle_wins, throttle, 0.0)
        return samples

    def score_samples(
        self,
        model: DrivingLeWM,
        pixels: torch.Tensor,
        history_actions: torch.Tensor,
        samples: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        if history_actions.ndim != 3:
            raise ValueError(f"Expected history_actions as [B,T,A], got shape {tuple(history_actions.shape)}")
        model_action_dim = int(model.cfg.action_dim)
        if int(history_actions.shape[-1]) != model_action_dim:
            raise ValueError(
                f"history action_dim {history_actions.shape[-1]} does not match model action_dim {model_action_dim}"
            )

        costs = []
        cost_cfg = self.cfg["cost"]
        pixels_device = pixels.to(device)
        history_device = history_actions.to(device)
        for actions in samples:
            candidate_history = history_device.clone()
            first_action = torch.from_numpy(expand_action_to_model_dim(actions[0], model_action_dim)).to(device)
            candidate_history[:, -1, :] = first_action
            aux = model.rollout_aux(pixels_device, candidate_history).detach().cpu().numpy()[0]
            speed, progress, lane_offset, heading, collision, offroad, red_light, blocked = aux.tolist()
            speed_error = abs(float(speed) - self.target_speed_mps)
            action_smooth = float(np.square(np.diff(actions, axis=0)).mean()) if len(actions) > 1 else 0.0
            first_raw = np.asarray(actions[0], dtype=np.float32)
            throttle_brake_conflict = float(first_raw[0] * first_raw[2])
            cost = (
                -float(cost_cfg["progress_reward"]) * float(progress)
                + float(cost_cfg["lane_offset_penalty"]) * abs(float(lane_offset))
                + float(cost_cfg["heading_penalty"]) * abs(float(heading))
                + float(cost_cfg["speed_error_penalty"]) * speed_error
                + float(cost_cfg["collision_penalty"]) * max(0.0, float(collision))
                + float(cost_cfg["offroad_penalty"]) * max(0.0, float(offroad))
                + float(cost_cfg["red_light_penalty"]) * max(0.0, float(red_light))
                + float(cost_cfg["blocked_penalty"]) * max(0.0, float(blocked))
                + float(cost_cfg["action_smoothness_penalty"]) * action_smooth
                + float(cost_cfg.get("brake_penalty", 0.0)) * float(first_raw[2])
                + float(cost_cfg.get("throttle_brake_conflict_penalty", 0.0)) * throttle_brake_conflict
            )
            costs.append(cost)
        return np.asarray(costs, dtype=np.float32)


@dataclass
class EpisodeAccumulator:
    route_length_m: float
    route_progress_m: float = 0.0
    first_infraction_distance_m: float | None = None
    collision_count: int = 0
    offroad_count: int = 0
    red_light_count: int = 0
    blocked_count: int = 0
    offroad_active: bool = False
    red_light_active: bool = False
    blocked_active: bool = False

    def _mark_first_infraction(self) -> None:
        if self.first_infraction_distance_m is None:
            self.first_infraction_distance_m = self.route_progress_m

    def add_collision_events(self, count: int) -> None:
        if count <= 0:
            return
        self.collision_count += int(count)
        self._mark_first_infraction()

    def update_flag(self, name: str, active: bool) -> None:
        previous = bool(getattr(self, f"{name}_active"))
        setattr(self, f"{name}_active", bool(active))
        if active and not previous:
            setattr(self, f"{name}_count", int(getattr(self, f"{name}_count")) + 1)
            self._mark_first_infraction()

    def to_metrics(self) -> DrivingEpisodeMetrics:
        return DrivingEpisodeMetrics(
            route_length_m=self.route_length_m,
            route_progress_m=min(self.route_progress_m, self.route_length_m),
            collision_count=self.collision_count,
            offroad_count=self.offroad_count,
            red_light_count=self.red_light_count,
            blocked_count=self.blocked_count,
            first_infraction_distance_m=self.first_infraction_distance_m,
        )


def load_model(checkpoint_path: Path) -> DrivingLeWM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload["cfg"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    action_dim = int(data_cfg["frameskip"]) * 3
    lewm_cfg = DrivingLeWMConfig(
        encoder_scale=model_cfg["encoder_scale"],
        patch_size=int(model_cfg["patch_size"]),
        image_size=int(data_cfg["image_size"]),
        action_dim=action_dim,
        embed_dim=int(model_cfg["embed_dim"]),
        history_size=int(data_cfg["history_size"]),
        predictor_depth=int(model_cfg["predictor_depth"]),
        predictor_heads=int(model_cfg["predictor_heads"]),
        predictor_mlp_dim=int(model_cfg["predictor_mlp_dim"]),
        dropout=float(model_cfg["dropout"]),
        sigreg_weight=float(model_cfg["sigreg_weight"]),
        aux_weight=float(model_cfg["aux_weight"]),
    )
    model = DrivingLeWM(lewm_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def run_dry_eval(cfg: dict[str, Any], checkpoint_path: Path | None, policy: str) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    out: dict[str, Any] = {
        "status": "config_loaded",
        "policy": policy,
        "episodes": int(eval_cfg["eval_episodes"]),
        "route_cap_m": float(eval_cfg["route_cap_m"]),
        "output_dir": str(eval_cfg["output_dir"]),
    }
    if policy == "model":
        if checkpoint_path is None:
            raise ValueError("A checkpoint is required for checkpoint/model dry-run validation")
        model = load_model(checkpoint_path)
        params = sum(p.numel() for p in model.parameters())
        out.update(
            {
                "status": "checkpoint_loaded",
                "checkpoint": str(checkpoint_path),
                "parameter_count": params,
                "model_action_dim": int(model.cfg.action_dim),
                "model_frameskip": action_dim_to_frameskip(int(model.cfg.action_dim)),
                "history_size": int(model.cfg.history_size),
            }
        )
    return out


def write_metrics(output_dir: Path, rows: list[DrivingEpisodeMetrics]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "route_length_m",
                "route_progress_m",
                "route_completion_pct",
                "infraction_free_distance_m",
                "mini_driving_score",
                "collision_count",
                "offroad_count",
                "red_light_count",
                "blocked_count",
            ],
        )
        writer.writeheader()
        for episode, row in enumerate(rows):
            writer.writerow(
                {
                    "episode": episode,
                    "route_length_m": row.route_length_m,
                    "route_progress_m": row.route_progress_m,
                    "route_completion_pct": row.route_completion_pct,
                    "infraction_free_distance_m": row.first_infraction_free_distance_m,
                    "mini_driving_score": row.mini_driving_score,
                    "collision_count": row.collision_count,
                    "offroad_count": row.offroad_count,
                    "red_light_count": row.red_light_count,
                    "blocked_count": row.blocked_count,
                }
            )
    summary = aggregate_metrics(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_action_trace(output_dir: Path, episode_idx: int, rows: list[dict[str, float | int]]) -> Path | None:
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"actions_episode_{episode_idx:03d}.csv"
    fieldnames = [
        "step",
        "carla_frame",
        "sim_time_s",
        "sim_delta_s",
        "wall_delta_s",
        "route_progress_m",
        "speed_mps",
        "throttle",
        "steer",
        "brake",
        "offroad",
        "blocked",
        "collision_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


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


def spawn_ego(carla, world, spawn_index: int):
    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")
    bp = blueprints[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points")
    transform = spawn_points[int(spawn_index) % len(spawn_points)]
    ego = world.try_spawn_actor(bp, transform)
    if ego is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at spawn index {spawn_index}")
    return ego


def listen_queue(sensor):
    image_queue: queue.Queue = queue.Queue()
    sensor.listen(image_queue.put)
    return image_queue


def wait_rgb(carla_world, image_queue, timeout_s: float) -> np.ndarray:
    return wait_rgb_packet(carla_world, image_queue, timeout_s)[0]


def wait_rgb_packet(carla_world, image_queue, timeout_s: float) -> tuple[np.ndarray, int, float]:
    frame_id = carla_world.tick()
    image = image_queue.get(timeout=timeout_s)
    while image.frame < frame_id:
        image = image_queue.get(timeout=timeout_s)
    return read_rgb(image), int(image.frame), float(image.timestamp)


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


def location_distance(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5)


def force_green_lights(carla, world, enabled: bool) -> None:
    if not enabled:
        return
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        actor.set_state(carla.TrafficLightState.Green)
        actor.freeze(True)


def set_weather(carla, world, weather_name: str) -> None:
    weather = getattr(carla.WeatherParameters, weather_name, None)
    if weather is None:
        raise ValueError(f"Unknown CARLA weather {weather_name!r}")
    world.set_weather(weather)


def to_vehicle_control(carla, action: np.ndarray):
    clipped = np.clip(np.asarray(action, dtype=np.float32), DrivingActionBounds().low, DrivingActionBounds().high)
    return carla.VehicleControl(throttle=float(clipped[0]), steer=float(clipped[1]), brake=float(clipped[2]))


def preprocess_pixels(frames: list[np.ndarray], image_size: int, device: torch.device) -> torch.Tensor:
    if not frames:
        raise ValueError("Need at least one RGB frame for model policy")
    pixels = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float().div_(255.0)
    if pixels.shape[-2:] != (image_size, image_size):
        pixels = F.interpolate(pixels, size=(image_size, image_size), mode="bilinear", align_corners=False)
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return pixels.unsqueeze(0).to(device)


def maybe_store_frame(
    output_dir: Path,
    frames: list[tuple[np.ndarray, str]],
    rgb: np.ndarray,
    *,
    episode: int,
    step: int,
    eval_cfg: dict[str, Any],
) -> None:
    every = max(1, int(eval_cfg.get("contact_sheet_every", 25)))
    max_frames = int(eval_cfg.get("max_contact_sheet_frames", 48))
    if (step == 1 or step % every == 0) and len(frames) < max_frames:
        frames.append((rgb.copy(), f"ep{episode} step{step}"))
    if bool(eval_cfg.get("save_frames", False)):
        frame_dir = output_dir / "frames" / f"episode_{episode:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(rgb).save(frame_dir / f"step_{step:05d}.jpg", quality=90)


def write_contact_sheet(output_dir: Path, frames: list[tuple[np.ndarray, str]]) -> Path | None:
    if not frames:
        return None
    from PIL import Image, ImageDraw

    thumb_w, thumb_h = 224, 224
    label_h = 20
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (rgb, label) in enumerate(frames):
        row, col = divmod(idx, cols)
        image = Image.fromarray(rgb).resize((thumb_w, thumb_h))
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_h + 3), label, fill=(0, 0, 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "contact_sheet.jpg"
    sheet.save(path, quality=92)
    return path


def run_episode(
    *,
    carla,
    world,
    traffic_manager,
    cfg: dict[str, Any],
    output_dir: Path,
    episode_idx: int,
    spawn_index: int,
    policy: str,
    model: DrivingLeWM | None,
    planner: CEMPlanner | None,
    device: torch.device,
) -> tuple[DrivingEpisodeMetrics, list[tuple[np.ndarray, str]]]:
    eval_cfg = cfg["eval"]
    timeout_s = float(eval_cfg.get("timeout_s", 10.0))
    fixed_delta = float(eval_cfg["fixed_delta_seconds"])
    max_steps = int(round(float(eval_cfg["max_episode_seconds"]) / fixed_delta))
    route_length_m = float(eval_cfg["route_cap_m"])
    acc = EpisodeAccumulator(route_length_m=route_length_m)
    actors = []
    frames: list[tuple[np.ndarray, str]] = []
    action_trace: list[dict[str, float | int]] = []
    collision_events = {"count": 0}
    last_collision_count = 0
    blocked_seconds = 0.0
    step = 0
    last_sim_time: float | None = None
    last_wall_time: float | None = None

    ego = spawn_ego(carla, world, spawn_index)
    actors.append(ego)
    camera = make_camera(carla, world, ego, cfg)
    actors.append(camera)
    collision_sensor = make_collision_sensor(carla, world, ego)
    actors.append(collision_sensor)
    collision_sensor.listen(lambda _event: collision_events.__setitem__("count", collision_events["count"] + 1))
    image_queue = listen_queue(camera)

    try:
        if policy == "autopilot":
            ego.set_autopilot(True, traffic_manager.get_port())
            target_speed = float(eval_cfg["target_speed_kmh"])
            traffic_manager.vehicle_percentage_speed_difference(ego, 100.0 - target_speed / 30.0 * 100.0)
        else:
            ego.set_autopilot(False)

        image_history: list[np.ndarray] = []
        action_history: list[np.ndarray] = []
        if policy == "model":
            if model is None or planner is None:
                raise ValueError("Model policy requires both model and planner")
            idle_action = np.asarray(eval_cfg.get("model_warmup_action", [0.0, 0.0, 1.0]), dtype=np.float32)
            for _ in range(int(model.cfg.history_size)):
                ego.apply_control(to_vehicle_control(carla, idle_action))
                rgb = wait_rgb(world, image_queue, timeout_s)
                image_history.append(rgb)
                action_history.append(expand_action_to_model_dim(idle_action, int(model.cfg.action_dim)))
            frameskip = action_dim_to_frameskip(int(model.cfg.action_dim))
        else:
            frameskip = 1
            wait_rgb(world, image_queue, timeout_s)

        previous_loc = ego.get_location()
        world_map = world.get_map()
        while step < max_steps and acc.route_progress_m < route_length_m:
            if policy == "model":
                assert model is not None and planner is not None
                pixels = preprocess_pixels(image_history[-int(model.cfg.history_size) :], int(model.cfg.image_size), device)
                hist = np.stack(action_history[-int(model.cfg.history_size) :], axis=0)
                history_actions = torch.from_numpy(hist).unsqueeze(0).float()
                action = planner.propose(model, pixels, history_actions, device)
                block_actions = [action for _ in range(frameskip)]
            elif policy == "constant":
                action = np.asarray(eval_cfg.get("constant_action", [0.45, 0.0, 0.0]), dtype=np.float32)
                block_actions = [action]
            else:
                block_actions = [None]

            block_raw_actions: list[np.ndarray] = []
            latest_rgb: np.ndarray | None = None
            for action in block_actions:
                if action is not None:
                    ego.apply_control(to_vehicle_control(carla, action))
                    block_raw_actions.append(np.asarray(action, dtype=np.float32))
                rgb, frame_id, sim_time = wait_rgb_packet(world, image_queue, timeout_s)
                wall_time = time.perf_counter()
                latest_rgb = rgb
                step += 1

                transform = ego.get_transform()
                loc = transform.location
                acc.route_progress_m += location_distance(previous_loc, loc)
                previous_loc = loc
                speed = vehicle_speed(ego)
                _, _, offroad = lane_metrics(carla, world_map, transform)
                control = ego.get_control()

                current_collision_count = int(collision_events["count"])
                acc.add_collision_events(current_collision_count - last_collision_count)
                last_collision_count = current_collision_count

                if step * fixed_delta >= float(eval_cfg.get("blocked_grace_seconds", 5.0)) and speed < float(
                    eval_cfg.get("blocked_speed_mps", 0.1)
                ):
                    blocked_seconds += fixed_delta
                else:
                    blocked_seconds = 0.0
                blocked = blocked_seconds >= float(eval_cfg.get("blocked_seconds", 3.0))
                red_light = bool(
                    ego.is_at_traffic_light()
                    and ego.get_traffic_light_state() == carla.TrafficLightState.Red
                    and speed > float(eval_cfg.get("red_light_speed_threshold_mps", 0.2))
                )

                acc.update_flag("offroad", bool(offroad))
                acc.update_flag("red_light", red_light)
                acc.update_flag("blocked", blocked)
                if bool(eval_cfg.get("save_action_trace", True)):
                    action_trace.append(
                        {
                            "step": int(step),
                            "carla_frame": int(frame_id),
                            "sim_time_s": float(sim_time),
                            "sim_delta_s": float(0.0 if last_sim_time is None else sim_time - last_sim_time),
                            "wall_delta_s": float(0.0 if last_wall_time is None else wall_time - last_wall_time),
                            "route_progress_m": float(acc.route_progress_m),
                            "speed_mps": float(speed),
                            "throttle": float(control.throttle),
                            "steer": float(control.steer),
                            "brake": float(control.brake),
                            "offroad": int(bool(offroad)),
                            "blocked": int(bool(blocked)),
                            "collision_count": int(acc.collision_count),
                        }
                    )
                last_sim_time = sim_time
                last_wall_time = wall_time
                maybe_store_frame(output_dir, frames, rgb, episode=episode_idx, step=step, eval_cfg=eval_cfg)

                if bool(eval_cfg.get("stop_on_collision", True)) and acc.collision_count > 0:
                    break
                if bool(eval_cfg.get("stop_on_blocked", True)) and blocked:
                    break
                if step >= max_steps or acc.route_progress_m >= route_length_m:
                    break

            if policy == "model" and latest_rgb is not None and block_raw_actions:
                image_history.append(latest_rgb)
                block = np.asarray(block_raw_actions, dtype=np.float32)
                if block.shape[0] < frameskip:
                    block = np.concatenate([block, np.repeat(block[-1:], frameskip - block.shape[0], axis=0)], axis=0)
                action_history.append(block[:frameskip].reshape(-1).astype(np.float32))

            if bool(eval_cfg.get("stop_on_collision", True)) and acc.collision_count > 0:
                break
            if bool(eval_cfg.get("stop_on_blocked", True)) and acc.blocked_count > 0:
                break
    finally:
        for actor in reversed(actors):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass

    if bool(eval_cfg.get("save_action_trace", True)):
        write_action_trace(output_dir, episode_idx, action_trace)
    return acc.to_metrics(), frames


def run_closed_loop_eval(cfg: dict[str, Any], checkpoint_path: Path | None, policy: str) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: DrivingLeWM | None = None
    planner: CEMPlanner | None = None
    if policy == "model":
        if checkpoint_path is None:
            raise ValueError("A checkpoint is required for checkpoint/model evaluation")
        model = load_model(checkpoint_path).to(device)
        planner_cfg = dict(cfg["planner"])
        planner_cfg["target_speed_mps"] = float(eval_cfg["target_speed_kmh"]) / 3.6
        planner = CEMPlanner(planner_cfg)

    carla = require_carla()
    client = carla.Client(str(eval_cfg["host"]), int(eval_cfg["port"]))
    client.set_timeout(float(eval_cfg.get("timeout_s", 10.0)))
    world = client.load_world(str(eval_cfg["town"])) if bool(eval_cfg.get("load_world", True)) else client.get_world()
    traffic_manager = client.get_trafficmanager(int(eval_cfg.get("traffic_manager_port", 8000)))
    original_settings = world.get_settings()
    rows: list[DrivingEpisodeMetrics] = []
    contact_frames: list[tuple[np.ndarray, str]] = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(eval_cfg["fixed_delta_seconds"])
        world.apply_settings(settings)
        applied_settings = world.get_settings()
        if not bool(applied_settings.synchronous_mode):
            raise RuntimeError("CARLA world did not enter synchronous_mode after apply_settings")
        if abs(float(applied_settings.fixed_delta_seconds or 0.0) - float(eval_cfg["fixed_delta_seconds"])) > 1e-6:
            raise RuntimeError(
                "CARLA fixed_delta_seconds mismatch: "
                f"expected {float(eval_cfg['fixed_delta_seconds'])}, got {applied_settings.fixed_delta_seconds}"
            )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(eval_cfg.get("traffic_manager_seed", 20260522)))
        set_weather(carla, world, str(eval_cfg.get("weather", "ClearNoon")))
        force_green_lights(carla, world, bool(eval_cfg.get("force_green_lights", True)))

        spawn_indices = list(eval_cfg["route_spawn_indices"])
        episodes = int(eval_cfg["eval_episodes"])
        for episode_idx in range(episodes):
            spawn_index = int(spawn_indices[episode_idx % len(spawn_indices)])
            row, frames = run_episode(
                carla=carla,
                world=world,
                traffic_manager=traffic_manager,
                cfg=cfg,
                output_dir=output_dir,
                episode_idx=episode_idx,
                spawn_index=spawn_index,
                policy=policy,
                model=model,
                planner=planner,
                device=device,
            )
            rows.append(row)
            remaining = int(eval_cfg.get("max_contact_sheet_frames", 48)) - len(contact_frames)
            if remaining > 0:
                contact_frames.extend(frames[:remaining])
            print(
                json.dumps(
                    {
                        "event": "eval_episode",
                        "episode": episode_idx,
                        "policy": policy,
                        "spawn_index": spawn_index,
                        "route_completion_pct": row.route_completion_pct,
                        "infraction_free_distance_m": row.first_infraction_free_distance_m,
                        "mini_driving_score": row.mini_driving_score,
                        "collision_count": row.collision_count,
                        "offroad_count": row.offroad_count,
                        "red_light_count": row.red_light_count,
                        "blocked_count": row.blocked_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    summary = write_metrics(output_dir, rows)
    if bool(eval_cfg.get("save_contact_sheet", True)):
        contact_sheet = write_contact_sheet(output_dir, contact_frames)
        if contact_sheet is not None:
            summary["contact_sheet"] = str(contact_sheet)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_cli_overrides(cfg, args)

    policy = resolve_policy(cfg, args)
    checkpoint = args.checkpoint or (Path(cfg["eval"]["checkpoint_path"]) if cfg["eval"].get("checkpoint_path") else None)
    if args.dry_run:
        print(json.dumps(run_dry_eval(cfg, checkpoint, policy), indent=2, sort_keys=True))
        return
    print(json.dumps(run_closed_loop_eval(cfg, checkpoint, policy), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
