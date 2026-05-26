# Current Execution Status

Last checked: 2026-05-26 18:20 Asia/Shanghai.

## 2026-05-26 Route6 Safety Snapshot

The active stop condition is unchanged: continue exploration until a learned model can drive at least `1km` on the simplified CARLA city task without lane invasion, roadside/off-road failure, collision, blocked failure, or traffic-light violation. This condition is not met.

Current route6 evidence is now more specific than the older D1 action-policy branch:

- The simplified Town03/no-traffic/green-light route6 task is feasible with the deterministic oracle `lane_keep` controller.
- The best learned route6 result remains below the target: `ft33 tail190 retry` reached `265.03m` before `lane_invasion`, with no collision/offroad/red-light/blocked failure.
- `ft34 tail220 laneonly` regressed to `248.81m` before `lane_invasion`.
- `ft35 temporal_aux + lane/heading sign` trained cleanly with W&B online. Best checkpoint was epoch `7`, step `1260`, with `best val/loss=0.919829`; final validation rose to `0.968075` at step `1800`, so `best.pt` is the main candidate.
- `ft35 retry` did not solve closed-loop route6: best checkpoint reached `249.80m`, last checkpoint reached `249.90m`, both terminating by `lane_invasion` with collision/offroad/red-light/blocked counts at `0`.
- The next implementation branch is `ft37 BCE-sign`: keep the same tiny patch-lane/perception architecture and aux vector, but change lane/heading sign supervision from margin loss to BCE on scalar lane/heading outputs. Before training it, run a fair robust-filter eval on `ft35 best` to check whether the remaining error is mostly output sign jitter.

Prepared artifacts for the next remote resume:

- Remote recovery script: `scripts/remote_route6_ft37_recovery.ps1`.
- Resumable gate script: `scripts/run_aux_tail_gate.py`.
- Robust-filter eval config: `configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml`.
- Training config: `configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml`.
- Hard-boundary fallback configs remain available if BCE-sign fails:
  - `configs/eval_d1_route6_perception_lane_keep_ft35_fs1_temporal_aux_tail220_sign_lg080_hg060_failure_frames_x6_port2110.yaml`
  - `configs/eval_d1_route6_perception_lane_keep_ft35_fs1_temporal_aux_tail220_sign_lg120_h0_failure_frames_x6_port2110.yaml`
  - `configs/train_d1_tiny_route6_perception_lane_ft36_fs1_temporal_delta_hardboundary_2200.yaml`

Verification so far: local model/eval tests pass after adding BCE sign loss, temporal delta loss, and perception-lane filtering tests; remote 5090 is temporarily unreachable over SSH despite port `22` accepting TCP.

## Latest 1km Goal Update

The active stop condition is unchanged: continue exploration until a learned model can drive at least `1km` on the simplified CARLA city route without lane/roadside departure, collision, blocked failure, or red-light violation.

The current mainline is `cls_mean + rollout aux + control-sign + aligned applied-action DAgger`. The previous `cls_mean` rollout model improved offline lane perception but failed the 100m closed-loop gate. After fixing runtime evaluation so top-level `lane_keep` parameters are actually merged into `eval`, the model reached mean Safety IFD `11.73m` over routes `[3, 4, 6, 10, 12]`, with `5/5` off-road failures and no collision/red-light/blocked failures. Action traces show the core failure: true lane offset reaches about `+1.7m` to `+1.8m`, while predicted `aux_lane_offset_m` remains negative, so the lane-keep controller steers in the wrong direction.

The valid DAgger dataset for the active branch is `data/d1_city_free_drive_dagger/carla_d1_city_free_drive_clsmean_fail_step1818_100m_appliednext_fast.h5`. It uses model-applied closed-loop actions aligned from the next trace row, because row `i` in an eval trace is the action that produced saved state `i`, while a training sample at frame/state `i` needs the action for transition `i -> i+1`. The dataset contains `329` frames across five failed episodes; `|lane_offset| > 0.5m` accounts for `41.0%`, `|lane_offset| > 1.0m` accounts for `22.5%`, and offroad frames are excluded.

The same-row applied-action DAgger run is invalid and should not be used for conclusions. The corrected aligned-appliednext 8k run improved failure-tail lane correlation to about `0.66`, but its control-sign stability plateaued around `0.6-0.7`, below the gate for CARLA evaluation.

The active serious run is now `d1_tiny_h3_fs5_auxpred_lane_perception_clsmean_wide_noroute_rollout_dagger_controlsign_appliednext_tailgate_6k`, W&B run id `d1_tiny_h3_fs5_auxpred_lane_perception_clsmean_wide_noroute_rollout_dagger_contr-20260525-111653-42200fc9`. It uses batch size `256`, W&B online, local CSV metrics, and per-validation checkpoints. Epoch 1 saved `epoch001_step000470.pt`; failure-tail diagnostics show high corrective steering sign (`overall sign=0.974`, `control-tail sign=0.977`, `lane-tail sign=0.956`) but poor lane calibration (`lane_corr=-0.114`, `lane MAE=0.564m`). The epoch-1 checkpoint is therefore not promoted to CARLA 100m.

Evaluation has been tightened: the 1km simple gate now records CARLA `sensor.other.lane_invasion`, writes `lane_invasion_count`, `spawn_index`, and `termination_reason`, and can stop on lane invasion. The headline 1km target now means no collision, offroad, lane invasion, blocked failure, or red-light violation. A low-threshold fallback config is ready as `configs/train_d1_tiny_auxpred_lane_perception_clsmean_wide_noroute_rollout_dagger_controlsign_appliednext_tailgate_thr002_6k.yaml`; it keeps the same architecture/data and lowers `aux_control_active_threshold` from `0.08` to `0.02` if the active run remains a near-constant corrective-steer model.

## Verdict

The first strict D0 pass is complete end to end: clean CARLA data was collected, QC passed, tiny and small historical LeWM variants were trained with W&B, a tiny delta-progress + predicted-aux variant was trained, and closed-loop CARLA evaluation now has timing guards plus speed-limit-aware scoring.

The active long-horizon goal is now stricter: a learned model must drive at least `1km` on city streets without lane/roadside departure, collision, or blocked failure, and then pass a real-traffic-light variant with no red-light violations. Exploration should continue until this target is met; weather mixing and larger ViT are subordinate to the single-weather safety gate.

The current valid model-only success claim is intentionally narrow: on an isolated CARLA server and a throttle-only D0 target, the latest tiny and small delta/pred-aux checkpoints reach 190m with no hard infractions. At 200m they fail by off-road at 191.75m and 191.67m, and steering-safe 150m still fails at 107.63m and 103.42m.

The first 200m pass is a governed hybrid sanity result: tiny LeWM stays in the loop for throttle/brake planning, while lane keeping and speed limiting are guarded by deterministic feedback. Without the speed governor, the same hybrid fails the fair speed-gated metric at 24.17m from speeding.

The action-prior branch has now run for 10,000 optimizer steps. It improved offline action prediction, but closed-loop control still needs action decoding constraints: raw action policy was blocked at about 0.003m, throttle/brake exclusivity lifted pure action to 24.84m before speed violation, and action+lane-keep reached 157.69m before blocked. With throttle/brake exclusivity plus speed-governor brake release, action+lane-keep passed the 200m speed-gated route. This is a governed hybrid success, and the next gate is now a clearer D1 city free-drive task where road safety distance is primary and speed is tertiary.

D1-A no-aux tiny has now completed and did not pass the small-scope gate. The dataset quality was good: `48,000` frames across `60` episodes and six Town03 spawn routes, strict QC passed with zero collision/off-road/red-light/blocked frames, and sampled contact sheets were inspected before training. The run `d1_tiny_h3_fs5_core_action_noaux_20k` used `use_aux_head: false` with `aux_loss=0` and `pred_aux_loss=0`; training stopped by early stop after epoch `60` / step `8760`. Total-validation best was epoch `40` / step `5840` with `val/loss=0.13641`; action-best was epoch `52` / step `7592` with `val/action_loss=0.002903`.

Closed-loop D1-A failed on both checkpoints after fixing the evaluator policy override. Total-best reached mean Safety IFD `23.19m`; action-best reached `29.77m`; both had `0/6` primary-safety-success and `6/6` off-road failures. Contact sheets were copied to `outputs_d1_eval_total_policyfix_contact_sheet.jpg` and `outputs_d1_eval_best_action_policyfix_contact_sheet.jpg`. Contact sheets and action traces show a consistent learned-control issue: the model outputs near-constant throttle with a small left steering bias, lane offset drifts toward about `+-1.8m`, and the vehicle leaves the road. This is not yet a weather-robustness or model-size question, so D1-W mixed weather and ViT small are paused.

The first action-failure diagnostic confirms steering collapse. On a light test split sample, the expert target steering has mean/std `-0.01106 / 0.11660`, while total-best predicts `-0.02599 / 0.01747` with steering correlation `0.0055`; action-best predicts `-0.02355 / 0.03072` with steering correlation `-0.0452`. In other words, the offline action loss looked small while the steering channel lost most of its variance and feedback structure. The next required baseline is action-only BC plus a deterministic 1km control gate.

The deterministic 1km control gate is partly passed. On the full six-route set `[3, 4, 6, 8, 10, 12]`, lane-keep reached `1000m` cleanly on five routes and failed off-road on spawn `8` at `231.62m`, for mean Safety IFD `871.94m` and primary-safety success `5/6`. This confirms that a simple 1km city route is feasible, while spawn `8` should be treated as a separate route-conditioning/recovery case. The immediate learned-policy gate is now `D1-simple-1km`: spawn routes `[3, 4, 6, 10, 12]`, no traffic, forced green lights, `1000m` cap, and no collision/off-road/blocked/red-light. Action-only BC 10k is running on the 5090 with W&B enabled: `d1_tiny_h3_fs5_action_only_bc_10k-20260523-222519-3bb58927`.

An early epoch-1 offline diagnostic of the action-only BC checkpoint still shows steering collapse: predicted steer std `0.00497` versus target std `0.10760`, with near-zero steer correlation. This is not enough to stop the run, but the next fallback is already implemented as `configs/train_d1_tiny_action_weighted_bc_10k.yaml`: weighted action BC with steer component weight `8.0` and active-steer weight `4.0` for targets above `0.03`.

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
| D1 lane-keep controller full-6 | 6 | 1000m | isolated port 2100, no traffic, green lights | 871.94m | 86.04 | 5/6 clean; spawn 8 off-road at 231.62m |
| D1 lane-keep controller simple-5 | 5 | 1000m | subset of full-6 routes: [3, 4, 6, 10, 12] | 1000.00m | 100.0 | derived pass from full-6 episode table |
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
2. Use `configs/eval_d1_city_free_drive_model_action_1km_simple.yaml` as the first 1km learned-policy gate because the deterministic controller already proves those five routes are feasible.
3. Keep `configs/eval_d1_city_free_drive_model_action_1km.yaml` as the full-6 stress gate; spawn `8` is now a route-conditioning/recovery diagnostic, not the first success criterion.
4. Train or evaluate `configs/train_d1_tiny_action_only_bc_10k.yaml` on the same D1 dataset as a control baseline; if BC fails similarly, the dataset/task interface is the bottleneck.
5. If action-only BC still has low steering variance, launch `configs/train_d1_tiny_action_weighted_bc_10k.yaml` before returning to LeWM latent objectives.
6. Add recovery/perturbation data or route-command labels before another long LeWM run; current pure no-aux action decoding lacks lane-correction behavior.
7. Keep D1-W mixed weather paused until D1-A has a real single-weather closed-loop signal.
8. Keep ViT small paused until the failure looks like capacity or visual robustness rather than action decoding / task conditioning.
9. Run `aux + pred_aux` only as an ablation after the action-policy baseline is diagnosed; if it wins, report it as an engineering adapter, not the clean LeWM story.
