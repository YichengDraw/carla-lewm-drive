from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from carla_lewm_drive.closed_loop_eval.evaluate import load_model, runtime_eval_config
from carla_lewm_drive.config import load_yaml
from carla_lewm_drive.driving_lewm.data import CarlaSequenceDataset, build_splits
from carla_lewm_drive.driving_lewm.model import AUX_COMPONENTS, DrivingLeWM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure aux lane perception quality for driving LeWM checkpoints.")
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_d1_tiny_aux_lane_perception_long_recovery_mixed_8k.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval_d1_city_free_drive_model_perception_lane_keep_long_recovery_mixed_1km_simple.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--mode", choices=("perceive", "rollout"), default="perceive")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/aux_perception"))
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


def component_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
        "corr": finite_corr(pred, target),
        "target": quantiles(target),
        "pred": quantiles(pred),
        "error": quantiles(err),
    }


def sign_accuracy(pred: np.ndarray, target: np.ndarray, *, active_threshold: float) -> float | None:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) > float(active_threshold))
    if not bool(np.any(mask)):
        return None
    return float(np.mean(np.sign(pred[mask]) == np.sign(target[mask])))


def masked_tail_summary(
    *,
    mask: np.ndarray,
    total_count: int,
    pred_all: np.ndarray,
    target_all: np.ndarray,
    pred_steer: np.ndarray,
    target_steer: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    count = int(mask.sum())
    out: dict[str, Any] = {"count": count, "fraction": float(count / max(1, int(total_count)))}
    if count == 0:
        return out
    pred_lane = pred_all[mask, 2]
    target_lane = target_all[mask, 2]
    pred_heading = pred_all[mask, 3]
    target_heading = target_all[mask, 3]
    ps = pred_steer[mask]
    ts = target_steer[mask]
    out.update(
        {
            "lane_target": quantiles(target_lane),
            "lane_pred": quantiles(pred_lane),
            "lane_error": quantiles(pred_lane - target_lane),
            "heading_target": quantiles(target_heading),
            "heading_pred": quantiles(pred_heading),
            "control_target": quantiles(ts),
            "control_pred": quantiles(ps),
            "control_error": quantiles(ps - ts),
            "lane_sign_accuracy_abs_target_gt_0p05": sign_accuracy(pred_lane, target_lane, active_threshold=0.05),
            "control_sign_accuracy_abs_target_gt_0p02": sign_accuracy(ps, ts, active_threshold=0.02),
            "pred_control_opposes_lane_fraction": float(np.mean(ps * target_lane < 0.0)),
            "mean_pred_corrective_against_lane": float(np.mean(-ps * target_lane)),
        }
    )
    return out


def make_dataset(cfg: dict[str, Any], split: str):
    data_cfg = cfg["data"]
    paths = data_cfg.get("dataset_paths") or data_cfg.get("dataset_path")
    if paths is None:
        raise ValueError("data.dataset_path or data.dataset_paths is required")
    if split == "all":
        if isinstance(paths, list) and len(paths) != 1:
            raise ValueError("--split all currently expects one dataset path")
        path = Path(paths[0] if isinstance(paths, list) else paths)
        return CarlaSequenceDataset(
            path,
            frameskip=int(data_cfg["frameskip"]),
            history_size=int(data_cfg["history_size"]),
            num_preds=int(data_cfg["num_preds"]),
            image_size=int(data_cfg["image_size"]),
        )
    train, val, test, _ = build_splits(paths, data_cfg)
    return {"train": train, "val": val, "test": test}[split]


def lane_keep_steer_array(lane_offset: np.ndarray, heading_error: np.ndarray, eval_cfg: dict[str, Any]) -> np.ndarray:
    lane_cfg = eval_cfg.get("lane_keep", {})
    steer_bias = float(lane_cfg.get("steer_bias", 0.0))
    lane_gain = float(lane_cfg.get("lane_offset_gain", 0.18))
    heading_gain = float(lane_cfg.get("heading_error_gain", 1.0))
    max_abs = float(lane_cfg.get("steer_limit", lane_cfg.get("max_abs_steer", 0.6)))
    steer = steer_bias - lane_gain * lane_offset.astype(np.float64) - heading_gain * heading_error.astype(np.float64)
    return np.clip(steer, -max_abs, max_abs)


@torch.no_grad()
def summarize(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.train_config)
    eval_cfg = runtime_eval_config(load_yaml(args.eval_config)) if args.eval_config else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device).eval()
    dataset = make_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for step, batch in enumerate(loader, start=1):
        if args.max_batches is not None and step > args.max_batches:
            break
        pixels = batch["pixels"].to(device)
        route_id = batch.get("route_id")
        if route_id is not None:
            route_id = route_id.to(device)
        aux_target = DrivingLeWM.aux_target(batch, model.cfg.progress_mode)
        if args.mode == "rollout":
            history_size = int(model.cfg.history_size)
            pred_pixels = pixels[:, :history_size]
            pred_actions = batch["action"][:, :history_size].to(device)
            pred_route_id = route_id[:, :history_size] if route_id is not None and route_id.ndim >= 2 else route_id
            pred = model.rollout_aux(pred_pixels, pred_actions, route_id=pred_route_id).detach().cpu()
            target = aux_target[:, history_size].detach().cpu()
        else:
            pred = model.perceive_aux(pixels, route_id=route_id).detach().cpu()
            target = aux_target[:, -1].detach().cpu()
        preds.append(pred.numpy())
        targets.append(target.numpy())

    pred_all = np.concatenate(preds, axis=0)
    target_all = np.concatenate(targets, axis=0)
    per_component = {
        name: component_metrics(pred_all[:, i], target_all[:, i])
        for i, name in enumerate(AUX_COMPONENTS)
    }
    pred_steer = lane_keep_steer_array(pred_all[:, 2], pred_all[:, 3], eval_cfg)
    target_steer = lane_keep_steer_array(target_all[:, 2], target_all[:, 3], eval_cfg)
    steer_err = pred_steer - target_steer
    active = np.abs(target_steer) > 0.01
    steer_summary = {
        "mae": float(np.mean(np.abs(steer_err))),
        "rmse": float(np.sqrt(np.mean(np.square(steer_err)))),
        "bias": float(np.mean(steer_err)),
        "corr": finite_corr(pred_steer, target_steer),
        "sign_accuracy_abs_target_gt_0p01": float(np.mean(np.sign(pred_steer[active]) == np.sign(target_steer[active]))) if bool(np.any(active)) else None,
        "target": quantiles(target_steer),
        "pred": quantiles(pred_steer),
        "error": quantiles(steer_err),
    }
    tail_masks = {
        "lane_offset_gt_0p15": target_all[:, 2] > 0.15,
        "lane_offset_gt_0p20": target_all[:, 2] > 0.20,
        "lane_offset_gt_0p25": target_all[:, 2] > 0.25,
        "lane_offset_gt_0p35": target_all[:, 2] > 0.35,
        "lane_offset_gt_0p5": target_all[:, 2] > 0.5,
        "lane_offset_gt_1p0": target_all[:, 2] > 1.0,
        "lane_offset_lt_minus_0p15": target_all[:, 2] < -0.15,
        "lane_offset_lt_minus_0p20": target_all[:, 2] < -0.20,
        "lane_offset_lt_minus_0p25": target_all[:, 2] < -0.25,
        "lane_offset_lt_minus_0p35": target_all[:, 2] < -0.35,
        "lane_offset_lt_minus_0p5": target_all[:, 2] < -0.5,
        "lane_offset_lt_minus_1p0": target_all[:, 2] < -1.0,
        "abs_lane_offset_gt_0p15": np.abs(target_all[:, 2]) > 0.15,
        "abs_lane_offset_gt_0p20": np.abs(target_all[:, 2]) > 0.20,
        "abs_lane_offset_gt_0p25": np.abs(target_all[:, 2]) > 0.25,
        "abs_lane_offset_gt_0p35": np.abs(target_all[:, 2]) > 0.35,
        "abs_lane_offset_gt_0p5": np.abs(target_all[:, 2]) > 0.5,
        "abs_lane_offset_gt_1p0": np.abs(target_all[:, 2]) > 1.0,
        "target_control_gt_0p05": target_steer > 0.05,
        "target_control_lt_minus_0p05": target_steer < -0.05,
        "abs_target_control_gt_0p05": np.abs(target_steer) > 0.05,
        "abs_target_control_gt_0p10": np.abs(target_steer) > 0.10,
    }
    tail_summaries = {
        name: masked_tail_summary(
            mask=mask,
            total_count=int(pred_all.shape[0]),
            pred_all=pred_all,
            target_all=target_all,
            pred_steer=pred_steer,
            target_steer=target_steer,
        )
        for name, mask in tail_masks.items()
    }
    return {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "eval_config": str(args.eval_config) if args.eval_config else None,
        "split": args.split,
        "mode": args.mode,
        "samples": int(pred_all.shape[0]),
        "device": str(device),
        "per_component": per_component,
        "lane_control": steer_summary,
        "tail_summaries": tail_summaries,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lane = result["per_component"]["lane_offset"]
    heading = result["per_component"]["heading_error"]
    control = result["lane_control"]
    lines = [
        "# Aux Perception Diagnostics",
        "",
        "## Verdict Inputs",
        "",
        f"- Mode: `{result['mode']}`.",
        f"- Samples: `{result['samples']}` from `{result['split']}`.",
        f"- Lane offset MAE/RMSE: `{lane['mae']:.5f}` / `{lane['rmse']:.5f}` m; corr `{lane['corr']}`.",
        f"- Heading error MAE/RMSE: `{heading['mae']:.5f}` / `{heading['rmse']:.5f}` rad; corr `{heading['corr']}`.",
        f"- Induced lane-keep steering MAE/RMSE: `{control['mae']:.5f}` / `{control['rmse']:.5f}`; corr `{control['corr']}`.",
        f"- Control sign accuracy (`|target steer| > 0.01`): `{control['sign_accuracy_abs_target_gt_0p01']}`.",
        "",
        "## Tail Checks",
        "",
        "| Slice | Count | Pred lane p50 | Target lane p50 | Control sign acc | Corrective steer mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, tail in result.get("tail_summaries", {}).items():
        pred_lane = tail.get("lane_pred", {}).get("p50")
        target_lane = tail.get("lane_target", {}).get("p50")
        sign_acc = tail.get("control_sign_accuracy_abs_target_gt_0p02")
        corrective = tail.get("mean_pred_corrective_against_lane")
        lines.append(
            f"| `{name}` | `{tail.get('count', 0)}` | `{pred_lane}` | `{target_lane}` | `{sign_acc}` | `{corrective}` |"
        )
    lines.extend(
        [
            "",
        "## Full JSON",
        "",
        "```json",
        json.dumps(result, indent=2, sort_keys=True),
        "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(args)
    json_path = args.output_dir / "aux_perception_diagnostics.json"
    md_path = args.output_dir / "aux_perception_diagnostics.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, md_path)
    print(json_path)


if __name__ == "__main__":
    main()
