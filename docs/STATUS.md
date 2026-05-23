# Current Execution Status

Last checked: 2026-05-23 19:26 Asia/Shanghai.

## Verdict

The first strict D0 pass is complete end to end: clean CARLA data was collected, QC passed, tiny and small historical LeWM variants were trained with W&B, a tiny delta-progress + predicted-aux variant was trained, and closed-loop CARLA evaluation now has timing guards plus speed-limit-aware scoring.

The current valid model-only success claim is intentionally narrow: on an isolated CARLA server and a throttle-only D0 target, the latest tiny and small delta/pred-aux checkpoints reach 190m with no hard infractions. At 200m they fail by off-road at 191.75m and 191.67m, and steering-safe 150m still fails at 107.63m and 103.42m.

The first 200m pass is a governed hybrid sanity result: tiny LeWM stays in the loop for throttle/brake planning, while lane keeping and speed limiting are guarded by deterministic feedback. Without the speed governor, the same hybrid fails the fair speed-gated metric at 24.17m from speeding.

The action-prior branch has now run for 10,000 optimizer steps. It improved offline action prediction, but closed-loop control still needs action decoding constraints: raw action policy was blocked at about 0.003m, throttle/brake exclusivity lifted pure action to 24.84m before speed violation, and action+lane-keep reached 157.69m before blocked. With throttle/brake exclusivity plus speed-governor brake release, action+lane-keep passed the 200m speed-gated route. This is a governed hybrid success, and the next gate is now a clearer D1 city free-drive task where road safety distance is primary and speed is tertiary.

D1-A no-aux tiny has now completed and did not pass the small-scope gate. The dataset quality was good: `48,000` frames across `60` episodes and six Town03 spawn routes, strict QC passed with zero collision/off-road/red-light/blocked frames, and sampled contact sheets were inspected before training. The run `d1_tiny_h3_fs5_core_action_noaux_20k` used `use_aux_head: false` with `aux_loss=0` and `pred_aux_loss=0`; training stopped by early stop after epoch `60` / step `8760`. Total-validation best was epoch `40` / step `5840` with `val/loss=0.13641`; action-best was epoch `52` / step `7592` with `val/action_loss=0.002903`.

Closed-loop D1-A failed on both checkpoints after fixing the evaluator policy override. Total-best reached mean Safety IFD `23.19m`; action-best reached `29.77m`; both had `0/6` primary-safety-success and `6/6` off-road failures. Contact sheets were copied to `outputs_d1_eval_total_policyfix_contact_sheet.jpg` and `outputs_d1_eval_best_action_policyfix_contact_sheet.jpg`. Contact sheets and action traces show a consistent learned-control issue: the model outputs near-constant throttle with a small left steering bias, lane offset drifts toward about `+-1.8m`, and the vehicle leaves the road. This is not yet a weather-robustness or model-size question, so D1-W mixed weather and ViT small are paused.

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
| D1 no-aux total-best `model_action` | 6 | 500m | isolated port 2100, no traffic, green lights | 23.19m | 3.25 | failed off-road on 6/6 routes |
| D1 no-aux action-best `model_action` | 6 | 500m | isolated port 2100, no traffic, green lights | 29.77m | 4.17 | failed off-road on 6/6 routes |
| Tiny checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |
| Small checkpoint | 1 | 50m | port 2000, wider CEM | invalid | invalid | external HIL client ticked CARLA |

Interpretation:

- The autopilot baseline verifies that the route, CARLA world, and metric implementation are usable.
- The valid model-policy results require an isolated CARLA server. The evaluator now records `sim_delta_s` and raises on external tick jumps.
- The throttle-only target shows the model can sustain a simplified long-ish control loop up to 190m. The small ViT did not move the 200m boundary, and the steering-safe failures show that route-following still needs steering/lane-keeping calibration.
- The speed-gated trace shows the fair metric matters: ungoverned model-lane-keep reaches only 24.17m before speed-limit violation, while governed hybrid reaches 200m with max speed `6.57m/s` and max lane offset `0.094m`.
- The action-prior trace shows an action representation problem: the dataset has mutually exclusive throttle/brake, while the raw action head often predicts both. Evaluation-side exclusivity and speed-governor brake release can make the hybrid pass 200m, but pure action still fails speed compliance.
- The D1 no-aux trace shows the stronger city free-drive task is not solved: action-best improves mean IFD by only `6.58m` over total-best, and all six routes still end in off-road. Mean steering is slightly negative on most routes, which produces slow lateral drift rather than closed-loop lane correction.

## Completed Code Work

- Added collection-time episode quality gates and cumulative-distance progress.
- Added strict HDF5 QC thresholds and `--allow-infractions` escape hatch.
- Added fast HDF5 export for random-access training.
- Hardened training around batch probing, W&B run IDs/resume policy, local CSV fields, checkpoint selection, and CLI overrides.
- Added CARLA closed-loop evaluator for autopilot, constant-action, checkpoint, lane-keep, and model-lane-keep policies, including short-eval CLI overrides, action traces, lane/heading trace columns, speed-limit infractions, and CARLA timing guards.
- Added optional action-prior supervision on latent states plus `model_action` and `model_action_lane_keep` policies.
- Added throttle/brake exclusivity and speed-governor brake release options for action-policy closed-loop evaluation.
- Fixed closed-loop policy resolution so `--checkpoint` no longer overrides a YAML `policy: model_action`; added regression tests and verified the remote target suite with `17 passed`.
- Added tests for QC, training reliability, fast export, metrics, closed-loop evaluator, and delta-progress/predicted-aux targets.

## Current Next Gate

The active next gate is D1 action-policy diagnosis, not weather or scale:

1. Compare expert action distribution against predicted action traces route by route, especially steering sign, mean, and variance.
2. Train or evaluate a simple supervised behavior-cloning policy on the same D1 dataset as a control baseline; if BC fails similarly, the dataset/task interface is the bottleneck.
3. Add recovery/perturbation data or route-command labels before another long LeWM run; current pure no-aux action decoding lacks lane-correction behavior.
4. Keep D1-W mixed weather paused until D1-A has a real single-weather closed-loop signal.
5. Keep ViT small paused until the failure looks like capacity or visual robustness rather than action decoding / task conditioning.
6. Run `aux + pred_aux` only as an ablation after the action-policy baseline is diagnosed; if it wins, report it as an engineering adapter, not the clean LeWM story.
