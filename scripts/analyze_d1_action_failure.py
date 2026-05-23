from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from carla_lewm_drive.closed_loop_eval.evaluate import load_model
from carla_lewm_drive.config import load_yaml
from carla_lewm_drive.driving_lewm.data import CarlaSequenceDataset, build_splits


COMPONENTS = ("throttle", "steer", "brake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose D1 action-policy failures from data, checkpoints, and traces.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_d1_tiny_core_action_noaux_20k.yaml"))
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--eval-dir", action="append", type=Path, default=[])
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/d1_action_failure"))
    return parser.parse_args()


def finite_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    a = a[mask]
    b = b[mask]
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def quantiles(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = np.quantile(arr, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    keys = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")
    return {k: float(v) for k, v in zip(keys, qs, strict=True)}


def vector_stats(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 3)
    out: dict[str, Any] = {"count": int(actions.shape[0])}
    for i, name in enumerate(COMPONENTS):
        x = actions[:, i]
        out[name] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "mean_abs": float(np.mean(np.abs(x))),
            "q": quantiles(x),
        }
    out["throttle_brake_overlap_frac"] = float(np.mean((actions[:, 0] > 1e-4) & (actions[:, 2] > 1e-4)))
    return out


def action_feedback_stats(actions: np.ndarray, lane_offset: np.ndarray, heading_error: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 3)
    steer = actions[:, 1]
    lane = np.asarray(lane_offset, dtype=np.float64).reshape(-1)
    heading = np.asarray(heading_error, dtype=np.float64).reshape(-1)
    return {
        "corr_steer_lane_offset": finite_corr(steer, lane),
        "corr_steer_heading_error": finite_corr(steer, heading),
        "corr_steer_neg_lane_offset": finite_corr(steer, -lane),
        "corr_steer_neg_heading_error": finite_corr(steer, -heading),
        "mean_signed_lane_correction": float(np.mean(-steer * lane)) if steer.size and lane.size else math.nan,
        "mean_signed_heading_correction": float(np.mean(-steer * heading)) if steer.size and heading.size else math.nan,
    }


def summarize_dataset(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["action"][:], dtype=np.float32)
        lane = np.asarray(f["lane_offset_m"][:], dtype=np.float32)
        heading = np.asarray(f["heading_error_rad"][:], dtype=np.float32)
        route_id = np.asarray(f["route_id"][:], dtype=np.int32) if "route_id" in f else np.zeros(len(actions), dtype=np.int32)
        ep_idx = np.asarray(f["ep_idx"][:], dtype=np.int32)
        summary: dict[str, Any] = {
            "path": str(path),
            "frames": int(actions.shape[0]),
            "episodes": int(len(f["ep_len"])),
            "overall_action": vector_stats(actions),
            "overall_feedback": action_feedback_stats(actions, lane, heading),
            "lane_offset": quantiles(lane),
            "heading_error_rad": quantiles(heading),
            "routes": {},
        }
        for rid in sorted(int(x) for x in np.unique(route_id)):
            mask = route_id == rid
            summary["routes"][str(rid)] = {
                "frames": int(mask.sum()),
                "episodes": sorted(int(x) for x in np.unique(ep_idx[mask])),
                "action": vector_stats(actions[mask]),
                "feedback": action_feedback_stats(actions[mask], lane[mask], heading[mask]),
                "lane_offset": quantiles(lane[mask]),
                "heading_error_rad": quantiles(heading[mask]),
            }
    return summary


def make_prediction_dataset(cfg: dict[str, Any], dataset_path: Path, split: str):
    data_cfg = dict(cfg["data"])
    data_cfg["dataset_path"] = str(dataset_path)
    if split == "all":
        return CarlaSequenceDataset(
            dataset_path,
            frameskip=int(data_cfg["frameskip"]),
            history_size=int(data_cfg["history_size"]),
            num_preds=int(data_cfg["num_preds"]),
            image_size=int(data_cfg["image_size"]),
        )
    train, val, test, _ = build_splits(dataset_path, data_cfg)
    return {"train": train, "val": val, "test": test}[split]


@torch.no_grad()
def summarize_checkpoint(checkpoint: Path, cfg: dict[str, Any], dataset_path: Path, split: str, batch_size: int, max_batches: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint).to(device).eval()
    dataset = make_prediction_dataset(cfg, dataset_path, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    lane_values: list[np.ndarray] = []
    heading_values: list[np.ndarray] = []
    for i, batch in enumerate(loader, start=1):
        if max_batches is not None and i > max_batches:
            break
        pixels = batch["pixels"].to(device)
        pred = model.policy_action(pixels).detach().cpu().numpy().reshape(-1, model.cfg.action_dim // 3, 3)
        target = batch["action"][:, -1].numpy().reshape(-1, model.cfg.action_dim // 3, 3)
        pred_blocks.append(pred)
        target_blocks.append(target)
        lane_values.append(batch["lane_offset_m"][:, -1, 0].numpy())
        heading_values.append(batch["heading_error_rad"][:, -1, 0].numpy())
    pred_all = np.concatenate(pred_blocks, axis=0)
    target_all = np.concatenate(target_blocks, axis=0)
    lane_all = np.concatenate(lane_values, axis=0)
    heading_all = np.concatenate(heading_values, axis=0)
    pred_flat = pred_all.reshape(-1, 3)
    target_flat = target_all.reshape(-1, 3)
    error_flat = pred_flat - target_flat
    steer = {
        "corr_pred_target": finite_corr(pred_flat[:, 1], target_flat[:, 1]),
        "sign_accuracy_abs_gt_0p01": float(
            np.mean(np.sign(pred_flat[np.abs(target_flat[:, 1]) > 0.01, 1]) == np.sign(target_flat[np.abs(target_flat[:, 1]) > 0.01, 1]))
        )
        if np.any(np.abs(target_flat[:, 1]) > 0.01)
        else None,
    }
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "samples": int(pred_all.shape[0]),
        "prediction_action": vector_stats(pred_flat),
        "target_action": vector_stats(target_flat),
        "error": {
            "mae_by_component": {name: float(np.mean(np.abs(error_flat[:, i]))) for i, name in enumerate(COMPONENTS)},
            "bias_by_component": {name: float(np.mean(error_flat[:, i])) for i, name in enumerate(COMPONENTS)},
            "rmse_by_component": {name: float(np.sqrt(np.mean(np.square(error_flat[:, i])))) for i, name in enumerate(COMPONENTS)},
            "steer": steer,
        },
        "target_feedback": action_feedback_stats(target_all[:, -1, :], lane_all, heading_all),
        "prediction_feedback": action_feedback_stats(pred_all[:, -1, :], lane_all, heading_all),
        "lane_offset": quantiles(lane_all),
        "heading_error_rad": quantiles(heading_all),
    }


def read_trace(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
    return rows


def summarize_eval_dir(path: Path) -> dict[str, Any]:
    traces = sorted(path.glob("actions_episode_*.csv"))
    episodes = []
    for trace in traces:
        rows = read_trace(trace)
        if not rows:
            continue
        arr = {k: np.asarray([row.get(k, math.nan) for row in rows], dtype=np.float64) for k in rows[0]}
        actions = np.stack([arr["throttle"], arr["steer"], arr["brake"]], axis=1)
        episodes.append(
            {
                "trace": str(trace),
                "steps": int(len(rows)),
                "distance_m": float(arr["route_progress_m"][-1]),
                "final_lane_offset_m": float(arr["lane_offset_m"][-1]),
                "max_abs_lane_offset_m": float(np.nanmax(np.abs(arr["lane_offset_m"]))),
                "final_heading_error_rad": float(arr["heading_error_rad"][-1]),
                "action": vector_stats(actions),
                "feedback": action_feedback_stats(actions, arr["lane_offset_m"], arr["heading_error_rad"]),
                "offroad_count": int(np.nansum(arr.get("offroad", np.zeros(len(rows))))),
                "red_light_count": int(np.nansum(arr.get("red_light", np.zeros(len(rows))))),
                "blocked_count": int(np.nansum(arr.get("blocked", np.zeros(len(rows))))),
            }
        )
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"path": str(path), "summary": summary, "episodes": episodes}


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# D1 Action Failure Diagnostics",
        "",
        "## Verdict",
        "",
    ]
    ckpts = result.get("checkpoints", [])
    if ckpts:
        best = ckpts[0]
        pred_steer = best["prediction_action"]["steer"]
        target_steer = best["target_action"]["steer"]
        lines.append(
            f"- First checkpoint prediction steer mean/std: `{pred_steer['mean']:.5f}` / `{pred_steer['std']:.5f}`; "
            f"target steer mean/std: `{target_steer['mean']:.5f}` / `{target_steer['std']:.5f}`."
        )
        lines.append(
            f"- Steering corr(pred,target): `{best['error']['steer']['corr_pred_target']}`; "
            f"predicted feedback corr(steer, -lane_offset): `{best['prediction_feedback']['corr_steer_neg_lane_offset']}`."
        )
    for ev in result.get("eval_dirs", []):
        summ = ev.get("summary", {})
        if summ:
            lines.append(
                f"- Eval `{Path(ev['path']).name}` mean IFD `{summ.get('mean_infraction_free_distance_m')}` "
                f"with offroad `{summ.get('offroad_count')}` over `{summ.get('episodes')}` episodes."
            )
    lines.extend(
        [
            "",
            "## Dataset Action Summary",
            "",
            "```json",
            json.dumps(result["dataset"]["overall_action"], indent=2, sort_keys=True),
            "```",
            "",
            "## Dataset Feedback Summary",
            "",
            "```json",
            json.dumps(result["dataset"]["overall_feedback"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.train_config)
    result: dict[str, Any] = {
        "dataset": summarize_dataset(args.dataset_path),
        "checkpoints": [],
        "eval_dirs": [],
    }
    for checkpoint in args.checkpoint:
        result["checkpoints"].append(
            summarize_checkpoint(checkpoint, cfg, args.dataset_path, args.split, args.batch_size, args.max_batches)
        )
    for eval_dir in args.eval_dir:
        result["eval_dirs"].append(summarize_eval_dir(eval_dir))
    (args.output_dir / "d1_action_failure_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(result, args.output_dir / "d1_action_failure_diagnostics.md")
    print(args.output_dir / "d1_action_failure_diagnostics.json")


if __name__ == "__main__":
    main()
