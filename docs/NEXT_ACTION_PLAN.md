# Next Execution Plan: D1 1km City Driving

Last updated: 2026-05-26 18:20 Asia/Shanghai.

## Current 2026-05-26 Status

The active goal remains a learned closed-loop model that drives at least `1km` in a city street without lane-line invasion, roadside/off-road contact, collision, blocked failure, or traffic-light violation. The goal is not complete.

The task definition has been narrowed to the useful first gate:

- `D1-simple-route6`: Town03, no vehicles, no walkers, ClearNoon, forced green lights, strict lane/offroad/collision/blocked stopping, `1000m` cap.
- Primary score: distance before the first primary safety infraction.
- Speed is logged but is not the main optimization target.
- Real red lights and mixed weather stay behind the single-weather safety gate.

Current evidence:

- Oracle `lane_keep` can finish the 1km simplified route, so the simulator task is feasible.
- Tiny learned variants improved from roughly `170-190m` to `204.38m` with `ft31 patchlane tail150`, `217.69m` with `ft32 patchlane tail180`, and `265.03m` with `ft33 tail190 retry` plus `lg120_h0`.
- `ft34 tail220 laneonly` completed but regressed to `248.81m`; `ft35 temporal_aux + lane/heading sign` completed and reached `249.80m` best / `249.90m` last. Both failed by `lane_invasion` with collision/offroad/red-light/blocked counts at `0`.
- `ft35 temporal_aux + lane/heading sign` trained cleanly with W&B online. Best checkpoint was epoch `7` / step `1260`, with `best val/loss=0.919829`. The final epoch `10` / step `1800` rose to `val/loss=0.968075`, so `best.pt` is the checkpoint to use for robust-filter eval.
- The `ft35` training run did not prove the goal: it reproduced the roughly `250m` lane-invasion boundary rather than breaking it.
- All current best closed-loop failures remain `lane_invasion`; collision/offroad/red-light/blocked counts are `0` in the simplified route6 runs.
- The failure mechanism is still lane/heading perception sign/calibration failure near the boundary. In `ft35 retry`, the final trace has true `lane_offset_m=+0.325m` while model aux reports about `-0.057m`; last-60-frame lane sign accuracy is only about `0.417`.

Active next branch:

- `d1_tiny_h1_fs1_route6_perception_lane_ft37_temporal_aux_tail220_bce_sign_2400`
- Config: `configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml`
- Purpose: continue from `ft33 retry` best, keep the same aux vector and checkpoint compatibility, and change lane/heading sign supervision from margin loss to BCE on the scalar lane/heading outputs.
- New code: BCE mode in `scalar_sign_loss`; local tests cover BCE sign loss and robust perception-lane filtering. `aux_temporal_delta_loss` and `ft36 hardboundary` remain as fallback tooling if BCE-sign fails.
- Monitoring: W&B required from launch, local `metrics.csv`, validation checkpoints every epoch, and failure-tail gates before CARLA promotion.
- Current blocker: remote 5090 SSH port is reachable, but the server closes connections before SSH key exchange. Work can continue locally; remote sync/eval/train resumes when SSH recovers.

Prepared remote-resume tooling:

- Resumable gate script: `scripts/run_aux_tail_gate.py`. It writes JSON after every checkpoint, so another SSH drop will not erase two hours of diagnostics.
- Fair robust-filter eval config:
  - `configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml`
- ft37 training config:
  - `configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml`
- Hard-boundary fallback configs:
  - `configs/eval_d1_route6_perception_lane_keep_ft35_fs1_temporal_aux_tail220_sign_lg080_hg060_failure_frames_x6_port2110.yaml`
  - `configs/eval_d1_route6_perception_lane_keep_ft35_fs1_temporal_aux_tail220_sign_lg120_h0_failure_frames_x6_port2110.yaml`
  - `configs/train_d1_tiny_route6_perception_lane_ft36_fs1_temporal_delta_hardboundary_2200.yaml`

Immediate gate sequence:

1. When SSH recovers, sync BCE-sign/filter code and run the fair robust-filter eval on `ft35 retry best` using `configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml` with the `ft35` checkpoint override.
2. If the robust-filter eval still fails around `250m`, train `ft37` with W&B online from process start.
3. Promote a `ft37` checkpoint to CARLA only after validation improves and failure-tail sign metrics beat the `ft35` tail profile.
4. If route6 exceeds `300m`, run repeated route6 episodes before the full `1km` gate.
5. If `ft37` still fails around `250m`, collect new hard-boundary failure frames, convert them with `scripts/eval_frames_to_hdf5.py`, and run the `ft36 hardboundary` fallback.
6. Only after single-weather no-traffic safety works, run real traffic-light and mixed-weather gates.

## Immediate 2026-05-25 Plan

The current branch is `cls_mean + rollout aux + control-sign + aligned applied-action DAgger`. It uses the model-applied actions from the next eval trace row, so the training action label matches the transition from the saved frame/state to the next state.

Current execution sequence:

1. Continue monitoring `d1_tiny_h3_fs5_auxpred_lane_perception_clsmean_wide_noroute_rollout_dagger_controlsign_appliednext_tailgate_6k`.
2. After each validation checkpoint, run `scripts/rank_aux_tail_checkpoints.py` on the aligned appliednext failure dataset.
3. Promote a checkpoint to CARLA only if failure-tail diagnostics show both correct steering direction and calibrated lane/heading values.
4. Run a 100m gate on routes `[3, 4, 6, 10, 12]` with collision/offroad/lane-invasion/red-light/blocked stopping enabled.
5. Run the 1km gate only if the 100m gate is `5/5` clean.
6. If the active run keeps producing near-constant corrective steering, stop it and launch `configs/train_d1_tiny_auxpred_lane_perception_clsmean_wide_noroute_rollout_dagger_controlsign_appliednext_tailgate_thr002_6k.yaml`.
7. If 100m fails, inspect `spawn_index`, `termination_reason`, action traces, and frame contact sheets; then convert another aligned applied-action DAgger slice.

Current No-Go rule: do not spend 1km CARLA evaluation time after a branch fails 100m. The previous corrected 100m gate reached only `11.73m` mean Safety IFD with `5/5` off-road failures.

Current Go rule: promote a branch to 100m only when offline failure-tail control sign is high and the lane/heading prediction is not a collapsed constant; promote to 1km only when the 100m closed-loop gate is clean.

## Long-Horizon Goal

Continue exploration until a learned model can drive at least `1km` on a CARLA city street without lane/roadside departure, collision, or blocked failure, and then pass a real-traffic-light variant without red-light violations. This is the stop condition for the research loop. Weather mixing and larger ViT are useful only after the single-weather safety/control interface starts working.

## Current Interpretation

The previous D0 speed-gated action experiments answered a narrower question than the project should ultimately care about. They showed that:

- the data/QC/training/eval pipeline is usable;
- tiny LeWM can drive a simplified D0 target to about `190m`;
- the 200m pass is still a governed hybrid, not a pure learned policy;
- the latest pure action runs fail by speed-limit count while staying collision-free and off-road-free over the first ~20-25m.

That means the next task should not be "optimize speed compliance first." The next fair task is:

```text
Drive as far as possible on a simple city street with no other vehicles.
Primary failures: collision, off-road / lane departure into roadside, lane invasion, blocked.
Secondary failure: red-light violation.
Tertiary penalty: speed-limit violation.
```

Speed remains logged and penalized, but the main score is safety-distance before the first primary road-safety failure.

The first D1-A no-aux tiny run has now answered the smallest-scope question negatively. After fixing evaluator policy resolution so `--checkpoint` no longer overwrites `policy: model_action`, the total-best checkpoint reached mean Safety IFD `23.19m`, and the action-best checkpoint reached `29.77m`. Both failed off-road on all 6 routes. Contact sheets and action traces show near-constant throttle plus a small steering bias, not active lane correction. This is below the threshold for D1-W weather mixing or ViT small.

The action failure is now quantified: expert steering on a held-out sample has std `0.11660`, while total-best predicts std `0.01747` and action-best predicts std `0.03072`; steering correlation with the target is near zero. The next step is therefore not a larger world model. It is a control-interface diagnosis: prove the 1km route with deterministic lane keeping, then train action-only BC on the same dataset.

The deterministic 1km check returned a useful split instead of a clean full pass: lane-keep finished `1000m` on spawn routes `3`, `4`, `6`, `10`, and `12`, but failed off-road on spawn `8` at `231.62m`. The current first learned-policy target is therefore `D1-simple-1km`: the five controller-validated routes, no traffic, forced green lights, `1000m` cap, and primary safety success on every route. The original six-route config remains as `D1-full-6` stress evaluation and as evidence that spawn `8` needs route conditioning or recovery data.

The first epoch of action-only BC is not yet enough, and its current `best.pt` still predicts near-constant steering. A fallback is ready: `configs/train_d1_tiny_action_weighted_bc_10k.yaml` keeps the same architecture and data but weights the steering component and high-steer samples so MSE cannot hide behind mostly straight driving frames.

## Task Definition

### D1-simple-1km: City Free-Drive, No Traffic, Green Lights

Purpose: isolate vision, steering, road-boundary awareness, and long-horizon drift.

- CARLA town: `Town03`.
- Traffic: `vehicles: 0`, `walkers: 0`.
- Weather: `ClearNoon`.
- Lights: `force_green_lights: true`.
- Routes: five controller-validated fixed spawn indices `[3, 4, 6, 10, 12]`, cycled deterministically.
- Maneuvers: include natural straight, curve, and intersection driving from fixed routes. This covers left/right road geometry when the route naturally turns.
- Commanded lane changes are intentionally deferred until route-command labels exist. In a no-traffic setting, "change lane now" is not a well-defined target unless the dataset includes an explicit command or navigation objective.
- Episode target: `1000m` closed-loop cap.
- Primary metric: `Safety IFD`, meters before collision, off-road, lane invasion, or blocked.
- Secondary logs: lane offset, heading error, action trace, contact sheet.
- Speed: soft penalty only; do not stop the episode on normal speed-limit violations.

### D1-full-6: Stress Route Set

Purpose: keep the original six-route definition visible and fair after the first 1km learned-policy pass.

- Routes: `[3, 4, 6, 8, 10, 12]`.
- Current deterministic lane-keep result: `5/6` clean `1000m`; spawn `8` failed off-road at `231.62m`.
- Interpretation: spawn `8` is not removed from the research problem. It is promoted to a diagnostic route for route commands, recovery data, or a stronger controller/planner interface after `D1-simple-1km` is solved.

### D1-B: City Free-Drive, No Traffic, Real Lights

Purpose: add traffic-light rule following after D1-A is stable.

- Same as D1-A, but `force_green_lights: false`.
- Red-light violation stops the episode and is reported separately.
- Speed remains tertiary.

### D1-W: City Free-Drive, No Traffic, Mixed Weather

Purpose: test whether the learned road-following behavior is robust to visual shifts rather than only memorizing `ClearNoon`.

Run this only after D1-A has a credible closed-loop result. The first weather-mix pilot should keep the same task and routes, and vary only weather:

- Candidate weather presets: `ClearNoon`, `CloudyNoon`, `WetNoon`, `SoftRainNoon`.
- Traffic: still `vehicles: 0`, `walkers: 0`.
- Lights: keep `force_green_lights: true` for the first weather pass.
- Scale: start with the same total frame budget as D1-A by splitting episodes across weather presets; expand only if QC and closed-loop results justify it.
- Evaluation: report per-weather Safety IFD, not only aggregate mean. A model that succeeds only in one weather is not robust.

Current gate status: paused. If D1-W succeeds, then add D1-B real lights on the successful weather distribution. If D1-W fails while D1-A succeeds, prioritize data diversity and augmentation before increasing model size.

### Later D2

Add other vehicles only after D1-A and D1-B are reproducible. The first vehicle setting should use sparse traffic and fixed seeds, not dense urban traffic.
Lane-change episodes enter here or in a D1-C command-following phase, after the policy has a route command such as `follow_lane`, `left`, `right`, `straight`, or `change_lane_left/right`.

## Metric Contract

Report these metrics together:

| Priority | Metric | Meaning |
|---|---|---|
| 1 | `mean_infraction_free_distance_m` with `speed_limit_as_infraction: false` | distance before collision/off-road/lane-invasion/red-light/blocked, depending on config |
| 1 | `success_rate_no_primary_safety_infraction` | no collision, no off-road, no lane invasion, no blocked |
| 2 | `success_rate_no_primary_or_red_infraction` | no collision, off-road, lane invasion, red-light, or blocked |
| 3 | `speed_limit_count` | soft speed-rule violations |
| 3 | `mean_mini_driving_score` | route completion with all penalties, including speed |

For D1-A, red lights are forced green, so `success_rate_no_primary_safety_infraction` is the headline. For D1-B, `success_rate_no_primary_or_red_infraction` becomes the headline.

## Aux-Loss Policy

The auxiliary heads are no longer the default research story.

The clean baseline for the next attempt is:

```text
pred_loss + sigreg + action BC
```

That is not a pure original LeWM objective because the action head is needed for direct closed-loop control, but it removes the disputed `aux` and `pred_aux` supervision from the representation objective.
The main no-aux config uses `use_aux_head: false`, so the auxiliary heads are absent rather than merely zero-weighted.

We keep the auxiliary version only as an ablation:

```text
pred_loss + sigreg + action BC + aux + pred_aux
```

Interpretation rule:

- If no-aux wins or ties, the project claim should prefer the simpler objective.
- If aux wins, present it as an empirical driving adapter, not as original LeWM.
- If both fail, the next fix should target task conditioning / route commands / data coverage before scaling ViT. The current no-aux result has already triggered this rule for the main branch.

This keeps the LeWM argument clean: auxiliary losses are allowed to be useful engineering, but they must earn their place.

## Dataset Plan

Use `configs/d1_city_free_drive.yaml`. The pilot scale is `60` episodes x `40s`, about `48k` frames, matching the accepted D0-train size. Scale to `120` episodes x `50s` only after the pilot passes QC and remote disk has enough headroom.

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.carla_collect.collect_dataset \
  --config configs/d1_city_free_drive.yaml

PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.dataset_qc.validate_hdf5 \
  --dataset data/d1_city_free_drive/carla_d1_city_free_drive.h5 \
  --out-dir outputs/qc_d1_city_free_drive \
  --strict

PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.dataset_qc.export_fast_hdf5 \
  --src data/d1_city_free_drive/carla_d1_city_free_drive.h5 \
  --dst data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5 \
  --chunk-frames 256 \
  --overwrite
```

Frame-level acceptance:

- `0` collision frames.
- off-road fraction `<= 0.005`.
- red-light fraction `0` for D1-A.
- blocked fraction `<= 0.05`.
- minimum episode progress `170m`.
- sampled contact sheet inspected before training.
- six route IDs represented; if a spawn index repeatedly fails QC, drop it and document the replacement.

## Training Plan

### Main Run: Tiny Core Action, No Aux

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_core_action_noaux_20k.yaml \
  --dataset-path data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5 \
  --batch-size 256 \
  --num-workers 4 \
  --max-steps 20000 \
  --output-dir outputs/d1_tiny_h3_fs5_core_action_noaux_20k \
  --run-name d1_tiny_h3_fs5_core_action_noaux_20k
```

### Ablation: Tiny Core Action + Aux

Run only after the no-aux baseline has a closed-loop result and the pure-action failure mode has been diagnosed.

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_core_action_aux_ablation_20k.yaml \
  --dataset-path data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 20000 \
  --output-dir outputs/d1_tiny_h3_fs5_core_action_aux_ablation_20k \
  --run-name d1_tiny_h3_fs5_core_action_aux_ablation_20k
```

Training requirements:

- W&B online from process start.
- Local CSV/JSON/checkpoints preserved.
- Batch probe first if the D1 dataset shape or GPU memory changes.
- Monitor at least every 10 minutes.
- Do not scale to small ViT until tiny succeeds on D1-A or shows a capacity-looking failure rather than a task/metric/control-interface failure.

### Model-Size Escalation

Run ViT small only after the task has earned it:

- If tiny D1-A fails before `50m` by primary safety failure, do not scale first; fix task conditioning, data, or action decoding.
- If tiny D1-A reaches a credible distance but is inconsistent across routes, collect/weight missing route and recovery data before scaling.
- If tiny D1-A is stable and D1-W exposes visual robustness failures, try weather-mix data first, then ViT small.
- If tiny D1-A and D1-W are both stable but the control remains visibly underfit or offline action loss plateaus high, run ViT small with the same no-aux objective and the same D1-A/D1-W eval suite.

Current gate status: paused. The observed D1-A failure is before `50m` and looks like action decoding / closed-loop correction failure, so increasing ViT size is not the next experiment.

## Immediate Diagnostic Loop

### Gate 1: Deterministic 1km Control

Run:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_city_free_drive_lane_keep_1km.yaml
```

Observed result: the full six-route gate reached `1000m` cleanly on five routes and failed off-road on spawn `8` at `231.62m`, with mean Safety IFD `871.94m` and primary-safety success `5/6`.

Decision: first use the five passing routes as `D1-simple-1km`, then keep spawn `8` as the route-conditioning/recovery stress case.

### Gate 2: Action-Only BC Baseline

Run:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_action_only_bc_10k.yaml \
  --dataset-path data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5
```

Then evaluate:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_city_free_drive_model_action_1km_simple.yaml \
  --checkpoint outputs/d1_tiny_h3_fs5_action_only_bc_10k/best.pt
```

Decision rule:

- If BC also fails near 30m on `D1-simple-1km`, prioritize action representation, route commands, and recovery data before another LeWM run.
- If BC reaches hundreds of meters or 1km on `D1-simple-1km`, the dataset/action interface is viable and the LeWM objective/action coupling is the bottleneck.
- If BC wins, it becomes the control baseline that future LeWM variants must beat.
- After any `D1-simple-1km` pass, run `configs/eval_d1_city_free_drive_model_action_1km.yaml` on the full six-route set and report spawn `8` separately.

### Gate 3: Weighted Steering BC Fallback

Run only if action-only BC keeps predicted steer variance far below target variance or fails closed-loop near the previous 30m boundary:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_action_weighted_bc_10k.yaml \
  --dataset-path data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5
```

This is still no-aux and action-only. It changes the loss geometry, not the model architecture: throttle/steer/brake are logged separately, steer gets component weight `8.0`, and active steering targets above `0.03` get an extra factor `4.0`.

## Evaluation Plan

Run D1-A first:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --policy model_action \
  --config configs/eval_d1_city_free_drive_model_action.yaml \
  --output-dir outputs/d1_eval_tiny_core_action_noaux_model_action_500m
```

For the action-policy D1-A gate, evaluate both the total-validation best checkpoint and a best-action checkpoint when available. The closed-loop policy calls `policy_action(...)`, so `val/action_loss` is a useful secondary checkpoint selector even when total validation loss is dominated by latent/SIGReg terms.

Latest D1-A evidence:

| Checkpoint | Output dir | Mean Safety IFD | Primary success | Failure |
|---|---|---:|---:|---|
| total-best epoch 40 / step 5840 | `outputs/d1_eval_tiny_core_action_noaux_model_action_total_policyfix_500m` | `23.19m` | `0/6` | off-road on 6/6 |
| action-best epoch 52 / step 7592 | `outputs/d1_eval_tiny_core_action_noaux_model_action_best_action_policyfix_500m` | `29.77m` | `0/6` | off-road on 6/6 |

Then run D1-B only if D1-A has no primary safety failures:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_city_free_drive_model_action_lights.yaml \
  --output-dir outputs/d1_eval_tiny_core_action_noaux_model_action_500m_lights
```

For aux ablation, reuse the same eval configs with `--checkpoint` and `--output-dir` overrides.

## Go / No-Go

Continue from D1-A to D1-B if:

- six-route D1 dataset passes strict QC;
- lane-keep proves the simple route cap and the learned model reaches `1000m` on `D1-simple-1km`;
- no-aux tiny has finite W&B curves and improves validation action loss;
- closed-loop D1-A mean Safety IFD is meaningfully above the old 25m speed-failure boundary;
- contact sheets show road-following rather than accidental straight-line survival.

Continue from D1-A to D1-W if:

- D1-A Safety IFD is above the D0 speed-failure boundary by a clear margin;
- at least several routes survive long enough to show turns and lane keeping, not only straight-road behavior;
- total-best and action-best checkpoints agree qualitatively, or the action-best clearly wins closed-loop.

Current decision: do not continue to D1-W. Action-best is only modestly better than total-best and both fail off-road on every route.

Pause if:

- route selection produces many QC rejections;
- model action succeeds only because throttle/brake sanitization or lane-keep overrides dominate;
- closed-loop distance improves but lane offset steadily grows toward the road edge;
- aux ablation wins only offline while closed-loop stays worse.

Stop this branch if:

- no-aux and aux both fail before `50m` by primary safety infractions;
- W&B or local metrics are missing;
- CARLA timing guard reports external ticking;
- the model cannot move beyond the D0 boundary after the task and metric are simplified.

Current decision: pause the no-aux tiny branch before weather/scale. The aux branch has not been run, but the clean main baseline is below `50m`, so the next work item is diagnosis rather than another larger training job.

## If D1 Still Fails

The current D1-A result has entered this section. The next likely fixes are task-conditioning and data, not model size:

1. Compare expert actions and predicted actions route by route, especially steering mean, sign, variance, and lane-offset correlation.
2. Train or evaluate a simple behavior-cloning policy on the same images/actions. If it also drifts off-road, the dataset or task interface is insufficient for closed-loop driving.
3. Add recovery data for small lane-offset and heading-error perturbations.
4. Add route-command labels such as `follow_lane`, `left`, `right`, `straight` so turns and lane choices are explicit.
5. Split lateral and longitudinal heads so steering can be judged independently from throttle.
6. Add commanded lane-change episodes after route commands work on fixed turns.
7. Add sparse red-light examples only after D1-A is stable.
8. Add multi-weather data only after the single-weather task has a real closed-loop signal.
9. Try ViT small only after the above checks show tiny is under-capacity.
