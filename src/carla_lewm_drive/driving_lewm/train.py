from __future__ import annotations

import argparse
import csv
import json
import random
import time
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
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--batch-probe", action="store_true", help="Probe configured batch sizes and exit.")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def maybe_init_wandb(cfg: dict[str, Any], disabled: bool):
    wandb_cfg = cfg.get("wandb", {})
    if disabled or not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B is required for serious runs; install wandb or pass --no-wandb for smoke only") from exc
    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project", "carla-lewm-drive"),
        group=wandb_cfg.get("group"),
        name=cfg["run"]["name"],
        id=cfg["run"]["name"],
        resume="allow",
        mode=wandb_cfg.get("mode", "online"),
        config=cfg,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
        cfg["trainer"]["batch_size"] = int(batch_size)
        loader = make_loader(train_set, cfg, shuffle=True)
        status = "pass"
        peak_gb = 0.0
        try:
            batch = move_batch(next(iter(loader)), device)
            model = DrivingLeWM(make_model_cfg(cfg, batch["action"].shape[-1])).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]))
            loss = model.loss(batch)["loss"]
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            status = "oom" if "out of memory" in str(exc).lower() else f"fail:{type(exc).__name__}"
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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
                    "loss": loss_value,
                    "pred_loss": float(losses["pred_loss"].cpu()),
                    "sigreg_loss": float(losses["sigreg_loss"].cpu()),
                    "aux_loss": float(losses["aux_loss"].cpu()),
                    "time": time.time(),
                }
                append_metrics(metrics_csv, payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=global_step)

        val_losses = evaluate_loss(model, val_loader, device, limit_val)
        val_row = {
            "stage": "val",
            "epoch": epoch,
            "step": global_step,
            "loss": val_losses["loss"],
            "pred_loss": val_losses["pred_loss"],
            "sigreg_loss": val_losses["sigreg_loss"],
            "aux_loss": val_losses["aux_loss"],
            "time": time.time(),
        }
        append_metrics(metrics_csv, val_row)
        if wandb_run is not None:
            wandb_run.log(val_row, step=global_step)
        if val_losses["loss"] < best_val:
            best_val = val_losses["loss"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "best_val": best_val, "epoch": epoch}, best_path)
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "last.pt")

    test_loader = make_loader(test_set, cfg, shuffle=False)
    test_losses = evaluate_loss(model, test_loader, device, limit_val)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_losses, indent=2, sort_keys=True), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in test_losses.items()}, step=global_step)
        wandb_run.finish()
    return best_path


@torch.no_grad()
def evaluate_loss(model: DrivingLeWM, loader: DataLoader, device: torch.device, limit_batches: int | None = None) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "aux_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        if limit_batches is not None and step > int(limit_batches):
            break
        batch = move_batch(batch, device)
        losses = model.loss(batch)
        for key in totals:
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
    if args.max_epochs is not None:
        cfg["trainer"]["max_epochs"] = int(args.max_epochs)
    if args.limit_train_batches is not None:
        cfg["limit_train_batches"] = int(args.limit_train_batches)
    if args.limit_val_batches is not None:
        cfg["limit_val_batches"] = int(args.limit_val_batches)

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
