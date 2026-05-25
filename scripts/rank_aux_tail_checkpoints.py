from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_aux_perception import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank driving LeWM checkpoints by failure-tail lane-control diagnostics.")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-glob", type=str, action="append", default=[])
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--mode", choices=("perceive", "rollout"), default="rollout")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/tail_checkpoint_rank"))
    parser.add_argument("--min-overall-sign", type=float, default=0.80)
    parser.add_argument("--min-control-tail-sign", type=float, default=0.80)
    parser.add_argument("--min-lane-tail-sign", type=float, default=0.80)
    parser.add_argument("--min-lane-corr", type=float, default=0.50)
    return parser.parse_args()


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def metric(result: dict[str, Any], path: tuple[str, ...]) -> float | None:
    return finite_float(nested_get(result, path))


def unit_metric(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def checkpoint_row(
    result: dict[str, Any],
    *,
    min_overall_sign: float,
    min_control_tail_sign: float,
    min_lane_tail_sign: float,
    min_lane_corr: float,
) -> dict[str, Any]:
    overall_sign = metric(result, ("lane_control", "sign_accuracy_abs_target_gt_0p01"))
    control_tail_sign = metric(
        result,
        ("tail_summaries", "abs_target_control_gt_0p05", "control_sign_accuracy_abs_target_gt_0p02"),
    )
    lane_tail_sign = metric(
        result,
        ("tail_summaries", "abs_lane_offset_gt_0p5", "control_sign_accuracy_abs_target_gt_0p02"),
    )
    lane_corr = metric(result, ("per_component", "lane_offset", "corr"))
    heading_corr = metric(result, ("per_component", "heading_error", "corr"))
    lane_mae = metric(result, ("per_component", "lane_offset", "mae"))
    corrective = metric(result, ("tail_summaries", "abs_lane_offset_gt_0p5", "mean_pred_corrective_against_lane"))
    corrective_score = 0.0 if corrective is None else max(0.0, min(1.0, corrective / 0.05))
    score = (
        0.30 * unit_metric(overall_sign)
        + 0.30 * unit_metric(control_tail_sign)
        + 0.25 * unit_metric(lane_tail_sign)
        + 0.075 * unit_metric(lane_corr)
        + 0.025 * unit_metric(heading_corr)
        + 0.05 * corrective_score
    )
    gate_pass = (
        overall_sign is not None
        and control_tail_sign is not None
        and lane_tail_sign is not None
        and lane_corr is not None
        and overall_sign >= float(min_overall_sign)
        and control_tail_sign >= float(min_control_tail_sign)
        and lane_tail_sign >= float(min_lane_tail_sign)
        and lane_corr >= float(min_lane_corr)
        and (corrective is None or corrective > 0.0)
    )
    return {
        "checkpoint": result["checkpoint"],
        "score": score,
        "gate_pass": gate_pass,
        "overall_control_sign": overall_sign,
        "control_tail_sign_abs_target_gt_0p05": control_tail_sign,
        "lane_tail_sign_abs_lane_gt_0p5": lane_tail_sign,
        "lane_corr": lane_corr,
        "heading_corr": heading_corr,
        "lane_mae": lane_mae,
        "mean_pred_corrective_against_lane_abs_lane_gt_0p5": corrective,
        "samples": result.get("samples"),
    }


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    found = [Path(p) for p in args.checkpoint]
    for pattern in args.checkpoint_glob:
        found.extend(sorted(Path().glob(pattern)))
    unique: dict[str, Path] = {}
    for path in found:
        unique[str(path)] = path
    checkpoints = list(unique.values())
    if not checkpoints:
        raise ValueError("No checkpoints found; pass --checkpoint or --checkpoint-glob")
    return checkpoints


def write_markdown(rows: list[dict[str, Any]], path: Path, thresholds: dict[str, float]) -> None:
    lines = [
        "# Tail Checkpoint Rank",
        "",
        "## Gate Thresholds",
        "",
        f"- Overall control sign: `{thresholds['min_overall_sign']}`.",
        f"- Control-tail sign: `{thresholds['min_control_tail_sign']}`.",
        f"- Lane-tail sign: `{thresholds['min_lane_tail_sign']}`.",
        f"- Lane correlation: `{thresholds['min_lane_corr']}`.",
        "",
        "## Ranking",
        "",
        "| Rank | Gate | Score | Checkpoint | Overall sign | Control-tail sign | Lane-tail sign | Lane corr | Corrective mean |",
        "|---:|:---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{'PASS' if row['gate_pass'] else 'hold'} | "
            f"{row['score']:.4f} | "
            f"`{row['checkpoint']}` | "
            f"{row['overall_control_sign']} | "
            f"{row['control_tail_sign_abs_target_gt_0p05']} | "
            f"{row['lane_tail_sign_abs_lane_gt_0p5']} | "
            f"{row['lane_corr']} | "
            f"{row['mean_pred_corrective_against_lane_abs_lane_gt_0p5']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    thresholds = {
        "min_overall_sign": args.min_overall_sign,
        "min_control_tail_sign": args.min_control_tail_sign,
        "min_lane_tail_sign": args.min_lane_tail_sign,
        "min_lane_corr": args.min_lane_corr,
    }
    for checkpoint in checkpoints:
        result = summarize(
            SimpleNamespace(
                train_config=args.train_config,
                checkpoint=checkpoint,
                eval_config=args.eval_config,
                split=args.split,
                mode=args.mode,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                output_dir=args.output_dir,
            )
        )
        row = checkpoint_row(result, **thresholds)
        rows.append(row)
        diagnostics[str(checkpoint)] = result
    rows.sort(key=lambda r: (bool(r["gate_pass"]), float(r["score"])), reverse=True)
    payload = {"thresholds": thresholds, "rows": rows, "diagnostics": diagnostics}
    (args.output_dir / "tail_checkpoint_rank.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(rows, args.output_dir / "tail_checkpoint_rank.md", thresholds)
    print(args.output_dir / "tail_checkpoint_rank.json")


if __name__ == "__main__":
    main()
