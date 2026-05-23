# Current Execution Status

Last checked: 2026-05-23 Asia/Shanghai.

## Verdict

The first strict D0 pass is complete end to end: clean CARLA data was collected, QC passed, tiny and small historical LeWM variants were trained with W&B, a tiny delta-progress + predicted-aux variant was trained, and closed-loop CARLA evaluation now has timing guards plus speed-limit-aware scoring.

The current valid model-only success claim is intentionally narrow: on an isolated CARLA server and a throttle-only D0 target, the latest tiny and small delta/pred-aux checkpoints reach 190m with no hard infractions. At 200m they fail by off-road at 191.75m and 191.67m, and steering-safe 150m still fails at 107.63m and 103.42m.

The first 200m pass is a governed hybrid sanity result: tiny LeWM stays in the loop for throttle/brake planning, while lane keeping and speed limiting are guarded by deterministic feedback. Without the speed governor, the same hybrid fails the fair speed-gated metric at 24.17m from speeding.

The action-prior branch has now run for 10,000 optimizer steps. It improved offline action prediction, but closed-loop control still needs action decoding constraints: raw action policy was blocked at about 0.003m, throttle/brake exclusivity lifted pure action to 24.84m before speed violation, and action+lane-keep reached 157.69m before blocked. With throttle/brake exclusivity plus speed-governor brake release, action+lane-keep passed the 200m speed-gated route. This is a governed hybrid success, and the next gate is now a clearer D1 city free-drive task where road safety distance is primary and speed is tertiary.

D1 is now active. The first D1 dataset has `48,000` frames across `60` episodes and six Town03 spawn routes, strict QC passed with zero collision/off-road/red-light/blocked frames, and the sampled contact sheet was inspected before training. The active main training run is `d1_tiny_h3_fs5_core_action_noaux_20k` with `use_aux_head: false`; at the latest check it had passed step `4700`, kept `aux_loss=0` and `pred_aux_loss=0`, and the total-validation best remained `0.15595` at epoch `27` / step `3942`. Because D1-A uses `policy_action(...)`, a `best_action.pt` checkpoint is also maintained; current action-best is epoch `31` / step `4526` with `val/action_loss=0.003202`. Remote `d1_best_action_watch` and `d1_auto_eval_noaux` tmux watchers are armed; after training exits they will run D1-A on both total-best and action-best checkpoints.

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

| Run | Model | Batch | W&B | Objective | Best Val Loss | Test Loss | Test Pred Loss | Test Aux Loss | Test Pred-Aux Loss |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `d0_tiny_h3_fs5_fast_e5` | ViT tiny | 192 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_fast_e5-20260522-014214-db1cb3e1) | historical absolute progress | 2.9568 | 2.9620 | 0.7015 | 10.9257 | n/a |
| `d0_small_h3_fs5_fast_e5` | ViT small | 128 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_fast_e5-20260522-024715-2f674d5e) | historical absolute progress | 1.1281 | 1.1324 | 0.5269 | 2.6708 | n/a |
| `d0_tiny_h3_fs5_delta_predaux_e5` | ViT tiny | 192 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_delta_predaux_e5-20260522-042925-552fb819) | delta progress + predicted aux | 0.2156 | 0.2147 | 0.0625 | 0.1606 | 0.1025 |
| `d0_small_h3_fs5_delta_predaux_e5` | ViT small | 96 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_delta_predaux_e5-20260522-054348-da4e9fd5) | delta progress + predicted aux | 0.3311 | 0.3300 | 0.1243 | 0.1600 | 0.1225 |
| `d0_tiny_h3_fs5_delta_predaux_action_10k` | ViT tiny | 192 | [run](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_delta_predaux_action_10k-20260522-103256-1a514e1e) | delta + pred-aux + action-prior | 0.1425 | 0.1400 | 0.0240 | 0.0730 | 0.0186 |

Batch/resource notes:

- Tiny batch probe selected `batch_size=192`; `batch_size=256` OOMed.
- The latest small delta/pred-aux batch probe OOMed at `128` and passed `96` with `18.139GB` peak allocated VRAM, while the live run used about `26GB` total GPU memory including overhead and other active processes.
- The small delta/pred-aux best checkpoint came from epoch 1; later validation losses were worse: epoch 2 `0.3721`, epoch 3 `0.7614`, epoch 4 `0.3587`, epoch 5 `0.3413`.
- The action-prior 10k run stopped by `max_steps`; best checkpoint was step `9843`, with test action loss `0.0201` and test pred-action loss `0.0116`.
- `num_workers=2` was selected after the fast HDF5 export improved loader behavior.

## Closed-Loop Evidence

| Policy | Episodes | Route Cap | Planner | Mean IFD | Mini Score | Result |
|---|---:|---:|---|---:|---:|---|
| Autopilot baseline | 2 | 100m | CARLA autopilot | 100.00m | 100.0 | pass |
| Historical tiny checkpoint | 1 | 150m | isolated port 2100, throttle-only CEM | 150.00m | 100.0 | pass |
| Historical small checkpoint | 1 | 150m | isolated port 2100, throttle-only CEM | 150.00m | 100.0 | pass |
| Historical tiny checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.11m | 70.0 | failed off-road |
| Historical small checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.87m | 70.0 | failed off-road |
| Tiny delta/pred-aux checkpoint | 1 | 150m | isolated port 2100, throttle-only CEM | 150.00m | 100.0 | pass |
| Tiny delta/pred-aux checkpoint | 1 | 190m | isolated port 2100, throttle-only CEM | 190.00m | 100.0 | pass |
| Tiny delta/pred-aux checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.75m | 70.0 | failed off-road |
| Tiny delta/pred-aux checkpoint | 1 | 150m | isolated port 2100, steering-safe CEM | 107.63m | 70.0 | failed off-road |
| Small delta/pred-aux checkpoint | 1 | 190m | isolated port 2100, throttle-only CEM | 190.00m | 100.0 | pass |
| Small delta/pred-aux checkpoint | 1 | 200m | isolated port 2100, throttle-only CEM | 191.67m | 70.0 | failed off-road |
| Small delta/pred-aux checkpoint | 1 | 150m | isolated port 2100, steering-safe CEM | 103.42m | 70.0 | failed off-road |
| Lane-keep controller | 1 | 200m | isolated port 2100, speed gate 35km/h | 200.00m | 100.0 | pass |
| Tiny delta/pred-aux checkpoint + lane-keep steer | 1 | 200m | isolated port 2100, speed gate 35km/h | 24.17m | 10.27 | failed speed-limit |
| Tiny delta/pred-aux checkpoint + lane/speed governor | 1 | 200m | isolated port 2100, speed gate 35km/h | 200.00m | 100.0 | governed hybrid pass |
| Action-prior raw `model_action` | 1 | 200m | isolated port 2100, no throttle/brake sanitization | 0.003m | 0.001 | failed blocked |
| Action-prior `model_action` | 1 | 200m | isolated port 2100, throttle/brake exclusive | 24.84m | 10.56 | failed speed-limit |
| Action-prior `model_action_lane_keep` | 1 | 200m | isolated port 2100, throttle/brake exclusive | 157.69m | 63.08 | failed blocked |
| Action-prior `model_action_lane_keep` | 1 | 200m | isolated port 2100, exclusive + brake-release speed governor | 200.00m | 100.0 | governed hybrid pass |
| Tiny checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |
| Small checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |

Interpretation:

- The autopilot baseline verifies that the route, CARLA world, and metric implementation are usable.
- The valid model-policy results require an isolated CARLA server. The evaluator now records `sim_delta_s` and raises on external tick jumps.
- The throttle-only target shows the model can sustain a simplified long-ish control loop up to 190m. The small ViT did not move the 200m boundary, and the steering-safe failures show that route-following still needs steering/lane-keeping calibration.
- The speed-gated trace shows the fair metric matters: ungoverned model-lane-keep reaches only 24.17m before speed-limit violation, while governed hybrid reaches 200m with max speed `6.57m/s` and max lane offset `0.094m`.
- The action-prior trace shows an action representation problem: the dataset has mutually exclusive throttle/brake, while the raw action head often predicts both. Evaluation-side exclusivity and speed-governor brake release can make the hybrid pass 200m, but pure action still fails speed compliance.

## Completed Code Work

- Added collection-time episode quality gates and cumulative-distance progress.
- Added strict HDF5 QC thresholds and `--allow-infractions` escape hatch.
- Added fast HDF5 export for random-access training.
- Hardened training around batch probing, W&B run IDs/resume policy, local CSV fields, checkpoint selection, and CLI overrides.
- Added CARLA closed-loop evaluator for autopilot, constant-action, checkpoint, lane-keep, and model-lane-keep policies, including short-eval CLI overrides, action traces, lane/heading trace columns, speed-limit infractions, and CARLA timing guards.
- Added optional action-prior supervision on latent states plus `model_action` and `model_action_lane_keep` policies.
- Added throttle/brake exclusivity and speed-governor brake release options for action-policy closed-loop evaluation.
- Added tests for QC, training reliability, fast export, metrics, closed-loop evaluator, and delta-progress/predicted-aux targets.

## Current Next Gate

The active next gate is D1 no-traffic city free-drive:

1. Finish or stop-on-plateau `D1-city-free-drive-no-traffic` no-aux training with W&B and local CSV/checkpoints intact.
2. Evaluate by distance before primary road-safety failures: collision, off-road / roadside departure, or blocked.
3. Add real traffic lights as D1-B after D1-A is stable; keep speed-limit violations as logged soft penalties during the first D1 attempt.
4. Add commanded lane changes only after route-command labels exist; D1-A already covers natural left/right road geometry through turns and intersections.
5. Run the `aux + pred_aux` version only as an ablation and report it as an engineering adapter if it wins closed-loop.
