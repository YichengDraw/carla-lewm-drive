# Next Execution Plan: Constraint-Aware Action LeWM

Last updated: 2026-05-22 Asia/Shanghai.

## Current Gate Result

The action-prior 10k gate is complete.

- W&B run: `d0_tiny_h3_fs5_delta_predaux_action_10k`.
- Best validation loss: `0.1425` at step `9843`.
- Test loss: `0.1400`.
- Test action loss: `0.0201`; test pred-action loss: `0.0116`.
- Raw action-policy closed loop failed immediately because the action head predicted throttle and brake together.
- Throttle/brake exclusivity lifted pure `model_action` to `24.84m`, then it failed by speed-limit violation.
- Throttle/brake exclusivity lifted `model_action_lane_keep` to `157.69m`, then it failed by blocked.
- Exclusivity plus speed-governor brake release passed `200.00m` with 100.0 mini score, max speed `6.38m/s`, and max absolute lane offset `0.072m`.

Interpretation: the latent/action head contains useful control information, but the action representation and objective are still mismatched to CARLA control. The current 200m result is a governed hybrid success, not a pure action-policy success.

## Next Hypothesis

The next useful experiment is not a larger ViT. It is to train the action output so it obeys the same physical constraints already present in the expert data:

```text
expert data: throttle and brake are mutually exclusive
current action head: often predicts both
needed objective: make illegal throttle/brake conflict expensive during training
```

Pure action also needs explicit speed compliance. Offline L1 action loss alone can look good while closed-loop speed drifts into violations.

## Implementation Scope

1. Keep the current tiny LeWM backbone, delta-progress objective, pred-aux head, and action-prior head.
2. Add an action-conflict penalty on predicted throttle/brake products.
3. Log `action_conflict_loss` and `pred_action_conflict_loss` to W&B and local CSV.
4. Keep evaluation-side throttle/brake exclusivity as a safety guard, but report pure action results separately from governed hybrid results.
5. Re-run the same 200m speed-gated D0 route before any 500m or D1 expansion.

## Main Run

Create a new config from `configs/train_tiny_action_prior_10k.yaml`:

```bash
configs/train_tiny_action_conflict_10k.yaml
```

Use the same dataset and GPU settings:

```bash
cd /home/ubuntu/carla_lewm_drive
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny_action_conflict_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_conflict_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_conflict_10k
```

## Evaluation

Run pure action first:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_conflict_speedgate_200m
```

Then run the guarded hybrid:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_lane_keep_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_conflict_lane_keep_speedgate_200m
```

## Go / No-Go

Continue if:

- W&B logs from process start.
- Validation loss and action losses are finite.
- Conflict losses decrease from the first validation window.
- Pure action improves beyond the `24.84m` speed-limit failure.
- Guarded hybrid still passes 200m without collision, off-road, red light, speed-limit, or blocked events.

Pause if:

- Pure action still fails by speed before 50m.
- Hybrid passes only because the governor erases most model brake outputs.
- Validation improves while closed-loop IFD does not improve.

Stop this branch if:

- Action-conflict training does not reduce throttle/brake conflicts.
- Pure action remains below 50m after a clean 10k run.
- Disk, W&B, or CARLA timing evidence becomes unreliable.

## Next Branches

If conflict-aware action improves speed but not steering:

- Add centerline/waypoint conditioning or collect lateral recovery data.

If it improves neither speed nor steering:

- Keep the honest claim as governed latent/action hybrid on simplified D0.
- Avoid larger ViTs until the action representation is fixed.

If pure action passes 200m speed-gated:

- Repeat on 500m D0.
- Add held-out spawn indices before D1 traffic.
