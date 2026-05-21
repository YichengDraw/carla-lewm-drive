from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics, aggregate_metrics
from carla_lewm_drive.config import load_yaml
from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig
from carla_lewm_drive.schema import DrivingActionBounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop CARLA evaluation for a Driving-LeWM checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_d0.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and checkpoint shape without launching CARLA.")
    return parser.parse_args()


class CEMPlanner:
    def __init__(self, cfg: dict, bounds: DrivingActionBounds | None = None) -> None:
        self.cfg = cfg
        self.bounds = bounds or DrivingActionBounds()
        self.low = np.asarray(self.bounds.low, dtype=np.float32)
        self.high = np.asarray(self.bounds.high, dtype=np.float32)

    def propose(self, model: DrivingLeWM, pixels: torch.Tensor, history_actions: torch.Tensor, device: torch.device) -> np.ndarray:
        horizon = int(self.cfg["horizon"])
        num_samples = int(self.cfg["num_samples"])
        iterations = int(self.cfg["iterations"])
        elite = max(1, int(num_samples * float(self.cfg["elite_frac"])))
        mean = np.tile(np.array([0.35, 0.0, 0.0], dtype=np.float32), (horizon, 1))
        std = np.tile(np.asarray(self.cfg["action_std"], dtype=np.float32), (horizon, 1))
        best = mean[0]
        for _ in range(iterations):
            samples = np.random.normal(mean, std, size=(num_samples, horizon, 3)).astype(np.float32)
            samples = np.clip(samples, self.low, self.high)
            scores = self.score_samples(model, pixels, history_actions, samples, device)
            elite_idx = np.argsort(scores)[:elite]
            elite_samples = samples[elite_idx]
            mean = elite_samples.mean(axis=0)
            std = elite_samples.std(axis=0) + 1e-4
            best = samples[int(np.argmin(scores)), 0]
        return best

    def score_samples(
        self,
        model: DrivingLeWM,
        pixels: torch.Tensor,
        history_actions: torch.Tensor,
        samples: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        costs = []
        cost_cfg = self.cfg["cost"]
        for actions in samples:
            block = np.repeat(actions[:1], repeats=max(1, history_actions.shape[1]), axis=0)
            action_tensor = torch.from_numpy(block.reshape(1, history_actions.shape[1], -1)).to(device)
            aux = model.rollout_aux(pixels.to(device), action_tensor).detach().cpu().numpy()[0]
            speed, progress, lane_offset, heading, collision, offroad, red_light, blocked = aux.tolist()
            action_smooth = float(np.square(np.diff(actions, axis=0)).mean()) if len(actions) > 1 else 0.0
            cost = (
                -float(cost_cfg["progress_reward"]) * progress
                + float(cost_cfg["lane_offset_penalty"]) * abs(lane_offset)
                + float(cost_cfg["heading_penalty"]) * abs(heading)
                + float(cost_cfg["speed_error_penalty"]) * abs(speed)
                + float(cost_cfg["collision_penalty"]) * max(0.0, collision)
                + float(cost_cfg["offroad_penalty"]) * max(0.0, offroad)
                + float(cost_cfg["red_light_penalty"]) * max(0.0, red_light)
                + float(cost_cfg["action_smoothness_penalty"]) * action_smooth
                + 10.0 * max(0.0, blocked)
            )
            costs.append(cost)
        return np.asarray(costs, dtype=np.float32)


def load_model(checkpoint_path: Path) -> DrivingLeWM:
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


def run_dry_eval(cfg: dict, checkpoint_path: Path) -> dict:
    model = load_model(checkpoint_path)
    params = sum(p.numel() for p in model.parameters())
    return {"checkpoint": str(checkpoint_path), "parameter_count": params, "status": "checkpoint_loaded"}


def write_metrics(output_dir: Path, rows: list[DrivingEpisodeMetrics]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
        for row in rows:
            writer.writerow(
                {
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


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    checkpoint = args.checkpoint or Path(cfg["eval"]["checkpoint_path"])
    if args.dry_run:
        print(json.dumps(run_dry_eval(cfg, checkpoint), indent=2, sort_keys=True))
        return
    raise RuntimeError(
        "Full CARLA closed-loop evaluation requires a running CARLA 0.9.16 server. "
        "Use --dry-run for checkpoint validation, then run this command on the CARLA host."
    )


if __name__ == "__main__":
    main()
