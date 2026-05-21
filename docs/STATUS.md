# Current Execution Status

Last checked: 2026-05-22 Asia/Shanghai.

## Verdict

The first strict D0 pass is complete end to end: clean CARLA data was collected, QC passed, tiny and small LeWM variants were trained with W&B, and closed-loop CARLA evaluation now has timing guards against external simulator ticks.

The current valid success claim is intentionally narrow: on an isolated CARLA server and a throttle-only D0 target, both tiny and small checkpoints reach 150m with no hard infractions. At 200m, both fail near 191m by off-road. The next gate is steering/action-objective calibration, not another blind ViT size increase.

## Repository And Environment

- GitHub repository: `https://github.com/YichengDraw/carla-lewm-drive`
- Local repo: this repository checkout.
- Remote run root: remote Linux GPU checkout
- Remote GPU: NVIDIA GeForce RTX 5090, 32GB class VRAM
- CARLA server: 0.9.16. Valid model eval used isolated `127.0.0.1:2100`; port 2000 had an external HIL client ticking the world.
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
| Tiny checkpoint | 1 | 150m | isolated port 2100, throttle-only CEM | 150.00m | 100.0 | pass |
| Small checkpoint | 1 | 150m | isolated port 2100, throttle-only CEM | 150.00m | 100.0 | pass |
| Tiny checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.11m | 70.0 | failed off-road |
| Small checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.87m | 70.0 | failed off-road |
| Tiny checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |
| Small checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |

Interpretation:

- The autopilot baseline verifies that the route, CARLA world, and metric implementation are usable.
- The valid model-policy results require an isolated CARLA server. The evaluator now records `sim_delta_s` and raises on external tick jumps.
- The throttle-only target shows the model can sustain a simplified long-ish control loop. The 200m failures show that route-following still needs steering calibration.

## Completed Code Work

- Added collection-time episode quality gates and cumulative-distance progress.
- Added strict HDF5 QC thresholds and `--allow-infractions` escape hatch.
- Added fast HDF5 export for random-access training.
- Hardened training around batch probing, W&B run IDs/resume policy, local CSV fields, checkpoint selection, and CLI overrides.
- Added CARLA closed-loop evaluator for autopilot, constant-action, and checkpoint policies, including short-eval CLI overrides, action traces, and CARLA timing guards.
- Added tests for QC, training reliability, fast export, metrics, and closed-loop evaluator.

## Current Next Gate

Before spending more GPU time on larger ViTs, fix the closed-loop steering/action pathway:

1. Replace absolute progress reward with a delta-progress or speed-tracking objective.
2. Train or calibrate a predicted-aux head directly on predicted latents.
3. Add a steering-safe action prior or behavior-cloning action head before re-enabling steering.
4. Re-run 150m and 200m on isolated CARLA, then only attempt 500m after 200m passes without off-road.
