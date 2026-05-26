from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_aux_perception import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable aux perception failure-tail gates.")
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "TRAIN_CONFIG"), required=True)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--mode", choices=("perceive", "rollout"), default="perceive")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def unit(value: Any) -> float:
    out = finite_float(value)
    if out is None:
        return 0.0
    return max(0.0, min(1.0, out))


def corrective_score(value: Any) -> float:
    out = finite_float(value)
    if out is None:
        return 0.0
    return max(0.0, min(1.0, out / 0.08))


def checkpoint_row(name: str, checkpoint: str, result: dict[str, Any]) -> dict[str, Any]:
    lane = result["per_component"]["lane_offset"]
    heading = result["per_component"]["heading_error"]
    control = result["lane_control"]
    t25 = result["tail_summaries"]["abs_lane_offset_gt_0p25"]
    t35 = result["tail_summaries"]["abs_lane_offset_gt_0p35"]
    pos25 = result["tail_summaries"]["lane_offset_gt_0p25"]
    neg25 = result["tail_summaries"]["lane_offset_lt_minus_0p25"]
    score = (
        0.18 * unit(control.get("sign_accuracy_abs_target_gt_0p01"))
        + 0.20 * unit(control.get("corr"))
        + 0.20 * unit(lane.get("corr"))
        + 0.10 * unit(heading.get("corr"))
        + 0.14 * unit(t25.get("control_sign_accuracy_abs_target_gt_0p02"))
        + 0.10 * unit(t35.get("control_sign_accuracy_abs_target_gt_0p02"))
        + 0.08 * corrective_score(t25.get("mean_pred_corrective_against_lane"))
    )
    return {
        "name": name,
        "checkpoint": checkpoint,
        "score": score,
        "samples": result.get("samples"),
        "lane_mae": finite_float(lane.get("mae")),
        "lane_rmse": finite_float(lane.get("rmse")),
        "lane_corr": finite_float(lane.get("corr")),
        "heading_mae": finite_float(heading.get("mae")),
        "heading_corr": finite_float(heading.get("corr")),
        "control_mae": finite_float(control.get("mae")),
        "control_corr": finite_float(control.get("corr")),
        "control_sign": finite_float(control.get("sign_accuracy_abs_target_gt_0p01")),
        "abs25_count": t25.get("count"),
        "abs25_sign": finite_float(t25.get("control_sign_accuracy_abs_target_gt_0p02")),
        "abs25_corrective": finite_float(t25.get("mean_pred_corrective_against_lane")),
        "abs35_count": t35.get("count"),
        "abs35_sign": finite_float(t35.get("control_sign_accuracy_abs_target_gt_0p02")),
        "abs35_corrective": finite_float(t35.get("mean_pred_corrective_against_lane")),
        "pos25_count": pos25.get("count"),
        "pos25_sign": finite_float(pos25.get("control_sign_accuracy_abs_target_gt_0p02")),
        "pos25_corrective": finite_float(pos25.get("mean_pred_corrective_against_lane")),
        "neg25_count": neg25.get("count"),
        "neg25_sign": finite_float(neg25.get("control_sign_accuracy_abs_target_gt_0p02")),
        "neg25_corrective": finite_float(neg25.get("mean_pred_corrective_against_lane")),
    }


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for data in payload.get("datasets", {}).values():
        rows = data.get("rows") or []
        rows.sort(key=lambda r: r.get("score", -1.0), reverse=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_existing(args.output)
    payload["eval_config"] = str(args.eval_config)
    payload.setdefault("datasets", {})
    for dataset_name, train_config in args.dataset:
        dataset = payload["datasets"].setdefault(
            dataset_name,
            {"train_config": train_config, "rows": [], "diagnostics": {}},
        )
        dataset["train_config"] = train_config
        rows_by_name = {row.get("name"): row for row in dataset.get("rows", [])}
        for checkpoint_name, checkpoint_path in args.checkpoint:
            if checkpoint_name in rows_by_name and not args.overwrite:
                print(f"skip {dataset_name} {checkpoint_name}", flush=True)
                continue
            if not Path(checkpoint_path).exists():
                row = {"name": checkpoint_name, "checkpoint": checkpoint_path, "missing": True}
                rows_by_name[checkpoint_name] = row
                dataset["rows"] = list(rows_by_name.values())
                write_output(args.output, payload)
                print(f"missing {dataset_name} {checkpoint_name}", flush=True)
                continue
            print(f"analyze {dataset_name} {checkpoint_name}", flush=True)
            result = summarize(
                SimpleNamespace(
                    train_config=Path(train_config),
                    checkpoint=Path(checkpoint_path),
                    eval_config=args.eval_config,
                    split=args.split,
                    mode=args.mode,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    output_dir=args.output.parent,
                )
            )
            dataset.setdefault("diagnostics", {})[checkpoint_name] = result
            rows_by_name[checkpoint_name] = checkpoint_row(checkpoint_name, checkpoint_path, result)
            dataset["rows"] = list(rows_by_name.values())
            write_output(args.output, payload)
    write_output(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
