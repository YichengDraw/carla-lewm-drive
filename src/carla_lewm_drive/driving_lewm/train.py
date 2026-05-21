from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_lewm_drive.config import load_yaml
from carla_lewm_drive.driving_lewm.data import build_splits
from carla_lewm_drive.driving_lewm.model import DrivingLeWM, DrivingLeWMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact LeWM on CARLA HDF5 data.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_tiny.yaml"))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-probe", action="store_true", help="Probe configured batch sizes and exit.")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-id", type=str, default=None, help="Explicit W&B run id for intentional resume.")
    parser.add_argument(
        "--wandb-resume",
        type=str,
        choices=("allow", "must", "never", "auto"),
        default=None,
        help="W&B resume policy. Requires --wandb-id unless set to never.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def autocast_dtype(cfg: dict[str, Any]) -> torch.dtype:
    precision = str(cfg["trainer"].get("precision", "fp32")).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, cfg: dict[str, Any]) -> AbstractContextManager:
    dtype = autocast_dtype(cfg)
    enabled = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "memory" in message)


def clear_cuda_probe_state(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def make_unique_run_id(run_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip(".-") or "run"
    safe_name = safe_name[:80]
    return f"{safe_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def maybe_init_wandb(cfg: dict[str, Any], disabled: bool):
    wandb_cfg = cfg.get("wandb", {})
    if disabled or not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B is required for serious runs; install wandb or pass --no-wandb for smoke only") from exc

    run_id = wandb_cfg.get("id") or cfg.get("run", {}).get("id")
    resume = wandb_cfg.get("resume")
    if run_id is None:
        if resume not in (None, False, "never"):
            raise ValueError("wandb.resume requires wandb.id or run.id so an intentional resume cannot mix runs")
        run_id = make_unique_run_id(str(cfg["run"]["name"]))
        resume = "never"
    elif resume is None:
        resume = "allow"

    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project", "carla-lewm-drive"),
        group=wandb_cfg.get("group"),
        name=cfg["run"]["name"],
        id=run_id,
        resume=resume,
        mode=wandb_cfg.get("mode", "online"),
        config=cfg,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    new_fields = [key for key in row if key not in fieldnames]
    if new_fields:
        fieldnames.extend(new_fields)
        rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def loss_values(losses: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        else:
            out[key] = float(value)
    return out


def namespaced_losses(prefix: str, losses: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in loss_values(losses).items()}


def make_model_cfg(cfg: dict[str, Any], action_dim: int) -> DrivingLeWMConfig:
    model = dict(cfg["model"])
    data = cfg["data"]
    model["image_size"] = int(data["image_size"])
    model["history_size"] = int(data["history_size"])
    model["action_dim"] = int(action_dim)
    return DrivingLeWMConfig(**model)


def make_loader(dataset, cfg: dict[str, Any], *, shuffle: bool) -> DataLoader:
    trainer = cfg["trainer"]
    return DataLoader(
        dataset,
        batch_size=int(trainer["batch_size"]),
        shuffle=shuffle,
        num_workers=int(trainer.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=bool(trainer.get("num_workers", 0)),
    )


def run_batch_probe(cfg: dict[str, Any], train_set) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for batch_size in cfg["trainer"].get("batch_probe_order", [cfg["trainer"]["batch_size"]]):
        probe_cfg = dict(cfg)
        probe_cfg["trainer"] = dict(cfg["trainer"])
        probe_cfg["trainer"]["batch_size"] = int(batch_size)
        loader = make_loader(train_set, probe_cfg, shuffle=True)
        status = "pass"
        peak_gb = 0.0
        batch = model = opt = scaler = loss = None
        clear_cuda_probe_state(device)
        try:
            try:
                batch = move_batch(next(iter(loader)), device)
            except StopIteration:
                status = "empty"
                results.append({"batch_size": int(batch_size), "status": status, "peak_vram_gb": peak_gb})
                break
            model = DrivingLeWM(make_model_cfg(probe_cfg, batch["action"].shape[-1])).to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(probe_cfg["optimizer"]["lr"]),
                weight_decay=float(probe_cfg["optimizer"].get("weight_decay", 0.0)),
            )
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, probe_cfg):
                loss = model.loss(batch)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(probe_cfg["trainer"]["gradient_clip_norm"]))
            scaler.step(opt)
            scaler.update()
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        except RuntimeError as exc:
            status = "oom" if is_cuda_oom(exc) else f"fail:{type(exc).__name__}"
        finally:
            del loss, scaler, opt, model, batch, loader
            clear_cuda_probe_state(device)
        results.append({"batch_size": int(batch_size), "status": status, "peak_vram_gb": round(peak_gb, 3)})
        if status == "pass":
            break
    return results


def train(cfg: dict[str, Any], *, no_wandb: bool = False) -> Path:
    if cfg["data"].get("dataset_path") is None:
        raise ValueError("data.dataset_path is required")
    seed_everything(int(cfg["run"]["seed"]))
    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, split = build_splits(cfg["data"]["dataset_path"], cfg["data"])
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    train_loader = make_loader(train_set, cfg, shuffle=True)
    val_loader = make_loader(val_set, cfg, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_batch = next(iter(train_loader))
    model = DrivingLeWM(make_model_cfg(cfg, first_batch["action"].shape[-1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    wandb_run = maybe_init_wandb(cfg, no_wandb)

    metrics_csv = output_dir / "metrics.csv"
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    max_epochs = int(cfg["trainer"]["max_epochs"])
    log_every = int(cfg["trainer"].get("log_every_steps", 50))
    eval_every = max(1, int(cfg["trainer"].get("eval_every_epochs", 1)))
    limit_train = cfg.get("limit_train_batches") or cfg["trainer"].get("limit_train_batches")
    limit_val = cfg.get("limit_val_batches") or cfg["trainer"].get("limit_val_batches")
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            if limit_train is not None and step > int(limit_train):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg):
                losses = model.loss(batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"]["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            loss_value = float(losses["loss"].detach().cpu())
            epoch_losses.append(loss_value)
            if global_step % log_every == 0:
                payload = {
                    "stage": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "time": time.time(),
                    **namespaced_losses("train", losses),
                }
                append_metrics(metrics_csv, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=global_step)

        should_eval = epoch == max_epochs or epoch % eval_every == 0
        if should_eval:
            val_losses = evaluate_loss(model, val_loader, device, cfg, limit_val)
            val_row = {
                "stage": "val",
                "epoch": epoch,
                "step": global_step,
                "time": time.time(),
                **namespaced_losses("val", val_losses),
            }
            append_metrics(metrics_csv, val_row)
            if wandb_run is not None:
                wandb_run.log(val_row, step=global_step)
            if val_losses["loss"] < best_val:
                best_val = val_losses["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "best_val": best_val,
                        "best": {"val/loss": best_val, "epoch": epoch},
                        "epoch": epoch,
                    },
                    best_path,
                )
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "last.pt")

    test_loader = make_loader(test_set, cfg, shuffle=False)
    test_checkpoint = best_path if best_path.exists() else output_dir / "last.pt"
    checkpoint_kind = "best" if test_checkpoint == best_path else "last_no_best"
    checkpoint = torch.load(test_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_losses = evaluate_loss(model, test_loader, device, cfg, limit_val)
    test_row = {
        "stage": "test",
        "epoch": checkpoint.get("epoch"),
        "step": global_step,
        "time": time.time(),
        "checkpoint/kind": checkpoint_kind,
        "checkpoint/path": str(test_checkpoint),
        **namespaced_losses("test", test_losses),
    }
    append_metrics(metrics_csv, test_row)
    test_report = {
        "checkpoint": {
            "kind": checkpoint_kind,
            "path": str(test_checkpoint),
            "epoch": checkpoint.get("epoch"),
            "best_val/loss": checkpoint.get("best_val"),
        },
        "test": loss_values(test_losses),
        **namespaced_losses("test", test_losses),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2, sort_keys=True), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log(test_row, step=global_step)
        wandb_run.finish()
    return best_path


@torch.no_grad()
def evaluate_loss(
    model: DrivingLeWM,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for step, batch in enumerate(loader, start=1):
        if limit_batches is not None and step > int(limit_batches):
            break
        batch = move_batch(batch, device)
        with autocast_context(device, cfg):
            losses = model.loss(batch)
        for key in losses:
            totals.setdefault(key, 0.0)
            totals[key] += float(losses[key].detach().cpu())
        count += 1
    if count == 0:
        raise ValueError("Validation loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.dataset_path is not None:
        cfg["data"]["dataset_path"] = str(args.dataset_path)
    if args.batch_size is not None:
        cfg["trainer"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["trainer"]["num_workers"] = int(args.num_workers)
    if args.max_epochs is not None:
        cfg["trainer"]["max_epochs"] = int(args.max_epochs)
    if args.output_dir is not None:
        cfg["run"]["output_dir"] = str(args.output_dir)
    if args.run_name is not None:
        cfg["run"]["name"] = args.run_name
    if args.limit_train_batches is not None:
        cfg["limit_train_batches"] = int(args.limit_train_batches)
    if args.limit_val_batches is not None:
        cfg["limit_val_batches"] = int(args.limit_val_batches)
    if args.wandb_id is not None:
        cfg.setdefault("wandb", {})["id"] = args.wandb_id
    if args.wandb_resume is not None:
        cfg.setdefault("wandb", {})["resume"] = args.wandb_resume

    train_set, _, _, _ = build_splits(cfg["data"]["dataset_path"], cfg["data"])
    if args.batch_probe:
        out = Path(cfg["run"]["output_dir"]) / "batch_probe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        results = run_batch_probe(cfg, train_set)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(out)
        return
    print(train(cfg, no_wandb=args.no_wandb))


if __name__ == "__main__":
    main()
