# Execution Runbook

## Phase 0

```powershell
$env:PYTHONPATH="src"
python -m py_compile (Get-ChildItem -Recurse -Path src,tests -Filter *.py | ForEach-Object { $_.FullName })
python -m pytest
carla-lewm-build-html --output interactive_plan.html
```

Record CARLA, CUDA, W&B, disk, and repo status before real collection.

## Phase 1

```bash
carla-lewm-collect --config configs/d0_smoke.yaml --episodes 2 --seconds 5
carla-lewm-qc --dataset data/d0_smoke/carla_d0_smoke.h5 --out-dir outputs/qc_d0_smoke --strict
```

Gate: zero hard QC issues and visually acceptable sampled frames.

## Phase 2

```bash
carla-lewm-train --config configs/train_tiny.yaml --batch-probe
```

Gate: use the largest passing batch from `batch_probe.json`.

## Phase 3

```bash
carla-lewm-train --config configs/train_tiny.yaml
```

Gate: W&B run exists, local CSV exists, validation loss is finite, and best checkpoint is selected by validation.

## Phase 4

Run `configs/train_small.yaml` only after tiny fails for capacity reasons.

## Phase 5

Add D1 low-density traffic only after D0 held-out closed-loop driving produces credible infraction-free distance.
