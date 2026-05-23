# Next Execution Plan: D1 No-Traffic City Free-Drive

Last updated: 2026-05-23 Asia/Shanghai.

## Current Interpretation

The previous D0 speed-gated action experiments answered a narrower question than the project should ultimately care about. They showed that:

- the data/QC/training/eval pipeline is usable;
- tiny LeWM can drive a simplified D0 target to about `190m`;
- the 200m pass is still a governed hybrid, not a pure learned policy;
- the latest pure action runs fail by speed-limit count while staying collision-free and off-road-free over the first ~20-25m.

That means the next task should not be "optimize speed compliance first." The next fair task is:

```text
Drive as far as possible on a simple city street with no other vehicles.
Primary failures: collision, off-road / lane departure into roadside, blocked.
Secondary failure: red-light violation.
Tertiary penalty: speed-limit violation.
```

Speed remains logged and penalized, but the main score is safety-distance before the first primary road-safety failure.

## Task Definition

### D1-A: City Free-Drive, No Traffic, Green Lights

Purpose: isolate vision, steering, road-boundary awareness, and long-horizon drift.

- CARLA town: `Town03`.
- Traffic: `vehicles: 0`, `walkers: 0`.
- Weather: `ClearNoon`.
- Lights: `force_green_lights: true`.
- Routes: six fixed spawn indices, cycled deterministically.
- Maneuvers: include natural straight/curve/intersection driving from fixed routes; defer commanded lane changes until route-command labels exist.
- Episode target: `500m` closed-loop cap.
- Primary metric: `Safety IFD`, meters before collision, off-road, or blocked.
- Secondary logs: lane offset, heading error, action trace, contact sheet.
- Speed: soft penalty only; do not stop the episode on normal speed-limit violations.

### D1-B: City Free-Drive, No Traffic, Real Lights

Purpose: add traffic-light rule following after D1-A is stable.

- Same as D1-A, but `force_green_lights: false`.
- Red-light violation stops the episode and is reported separately.
- Speed remains tertiary.

### Later D2

Add other vehicles only after D1-A and D1-B are reproducible. The first vehicle setting should use sparse traffic and fixed seeds, not dense urban traffic.

## Metric Contract

Report these metrics together:

| Priority | Metric | Meaning |
|---|---|---|
| 1 | `mean_infraction_free_distance_m` with `speed_limit_as_infraction: false` | distance before collision/off-road/red-light/blocked, depending on config |
| 1 | `success_rate_no_primary_safety_infraction` | no collision, no off-road, no blocked |
| 2 | `success_rate_no_primary_or_red_infraction` | no collision, off-road, red-light, or blocked |
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

We keep the auxiliary version only as an ablation:

```text
pred_loss + sigreg + action BC + aux + pred_aux
```

Interpretation rule:

- If no-aux wins or ties, the project claim should prefer the simpler objective.
- If aux wins, present it as an empirical driving adapter, not as original LeWM.
- If both fail, the next fix should target task conditioning / route commands / data coverage before scaling ViT.

This keeps the LeWM argument clean: auxiliary losses are allowed to be useful engineering, but they must earn their place.

## Dataset Plan

Use `configs/d1_city_free_drive.yaml`.

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
- sampled contact sheet inspected before training.
- six route IDs represented; if a spawn index repeatedly fails QC, drop it and document the replacement.

## Training Plan

### Main Run: Tiny Core Action, No Aux

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_core_action_noaux_20k.yaml \
  --dataset-path data/d1_city_free_drive/carla_d1_city_free_drive_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 20000 \
  --output-dir outputs/d1_tiny_h3_fs5_core_action_noaux_20k \
  --run-name d1_tiny_h3_fs5_core_action_noaux_20k
```

### Ablation: Tiny Core Action + Aux

Run only after the no-aux baseline has a closed-loop result.

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
- Do not scale to small ViT until tiny shows a capacity-looking failure rather than a task/metric/control-interface failure.

## Evaluation Plan

Run D1-A first:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_city_free_drive_model_action.yaml \
  --output-dir outputs/d1_eval_tiny_core_action_noaux_model_action_500m
```

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
- autopilot/lane-keep baseline passes the same route cap;
- no-aux tiny has finite W&B curves and improves validation action loss;
- closed-loop D1-A mean Safety IFD is meaningfully above the old 25m speed-failure boundary;
- contact sheets show road-following rather than accidental straight-line survival.

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

## If D1 Still Fails

The next likely fixes are task-conditioning and data, not model size:

1. Add route-command labels such as `follow_lane`, `left`, `right`, `straight`.
2. Add recovery data for small lane-offset and heading-error perturbations.
3. Split lateral and longitudinal heads so steering can be judged independently from throttle.
4. Add commanded lane-change episodes after route commands work on fixed turns.
5. Add sparse red-light examples only after D1-A is stable.
6. Try ViT small only after the above checks show tiny is under-capacity.
