# Current Execution Status

Last checked: 2026-05-22 Asia/Shanghai.

## Verdict

The first strict D0 pass is complete end to end: clean CARLA data was collected, QC passed, tiny and small LeWM variants were trained with W&B, and short closed-loop sanity evaluation ran in CARLA.

The current result is not yet a successful driving model. The strongest evidence is that `small` improved offline loss substantially but failed the closed-loop sanity test by getting blocked. The next gate is planner/objective/action calibration, not another blind ViT size increase.

## Repository And Environment

- GitHub repository: `https://github.com/YichengDraw/carla-lewm-drive`
- Local repo: this repository checkout.
- Remote run root: `/home/ubuntu/carla_lewm_drive`
- Remote GPU: NVIDIA GeForce RTX 5090, 32GB class VRAM
- CARLA server: 0.9.16, `127.0.0.1:2000`
- Serious training used W&B from process start and preserved local CSV/JSON/checkpoints.

## Data Evidence

| Dataset | Episodes | Frames | QC Gate | Collision | Off-road | Red light | Blocked |
|---|---:|---:|---|---:|---:|---:|---:|
| `D0-smoke` | 20 | 12,000 | pass | 0 | 0 | 0 | 0 |
| `D0-train` | 80 | 48,000 | pass | 0 | 0 | 0 | 0 |

Notes:

- The earlier 40s D0-train route was rejected because long-tail episodes started hitting off-road/collision cases.
- The accepted first-stage dataset uses 30s episodes on spawn point `3`, forced green lights, and strict collection-time quality gates.
- `data/d0_train/carla_d0_train_fast.h5` was exported for training because compressed HDF5 random reads were too slow.

## Training Evidence

| Run | Model | Batch | W&B | Best Val Loss | Test Loss | Test Pred Loss | Test Aux Loss |
|---|---|---:|---|---:|---:|---:|---:|
| `d0_tiny_h3_fs5_fast_e5` | ViT tiny | 192 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_fast_e5-20260522-014214-db1cb3e1) | 2.9568 | 2.9620 | 0.7015 | 10.9257 |
| `d0_small_h3_fs5_fast_e5` | ViT small | 128 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_fast_e5-20260522-024715-2f674d5e) | 1.1281 | 1.1324 | 0.5269 | 2.6708 |

Batch/resource notes:

- Tiny batch probe selected `batch_size=192`; `batch_size=256` OOMed.
- Small batch probe passed `batch_size=128` with peak allocated VRAM about `24.06GB`; live run used about `31GB` total GPU memory including overhead.
- `num_workers=2` was selected after the fast HDF5 export improved loader behavior.

## Closed-Loop Evidence

| Policy | Episodes | Route Cap | Planner | Mean IFD | Mini Score | Result |
|---|---:|---:|---|---:|---:|---|
| Autopilot baseline | 2 | 100m | CARLA autopilot | 100.00m | 100.0 | pass |
| Tiny checkpoint | 1 | 50m | CEM64, 2 iter, horizon 4 | 14.10m | 49.0 | failed off-road |
| Small checkpoint | 1 | 50m | CEM64, 2 iter, horizon 4 | 0.007m | 0.011 | failed blocked |

Interpretation:

- The autopilot baseline verifies that the route, CARLA world, and metric implementation are usable.
- The tiny checkpoint can move but leaves the lane in the short sanity test.
- The small checkpoint learns lower offline loss but selects near-stationary actions under the current planner and fails by blocked/stuck.

## Completed Code Work

- Added collection-time episode quality gates and cumulative-distance progress.
- Added strict HDF5 QC thresholds and `--allow-infractions` escape hatch.
- Added fast HDF5 export for random-access training.
- Hardened training around batch probing, W&B run IDs/resume policy, local CSV fields, checkpoint selection, and CLI overrides.
- Added CARLA closed-loop evaluator for autopilot and checkpoint policies, including short-eval CLI overrides.
- Added tests for QC, training reliability, fast export, metrics, and closed-loop evaluator.

## Current Next Gate

Before spending more GPU time on larger ViTs, fix the closed-loop action pathway:

1. Add direct behavior-cloning action loss or a small action head sanity policy.
2. Calibrate CEM cost terms against expert trajectory rollouts.
3. Log chosen throttle/steer/brake distributions during closed-loop eval.
4. Re-run the same short sanity test and require at least `50m` infraction-free distance before longer 500m evaluation.
