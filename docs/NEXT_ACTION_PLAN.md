# Next Execution Plan: Action-Prior LeWM Driving

Last updated: 2026-05-22 Asia/Shanghai.

## Why This Is The Next Gate

The current D0 evidence separates three facts:

- Tiny/small delta-progress + predicted-aux LeWM can run the simplified throttle-only target to 190m.
- Pure model-lane-keep without a speed governor fails the fair speed-gated metric at 24.17m by speeding.
- A deterministic lane/speed governor can make the route pass 200m, so the simulator route and metric are not the blocker.

The next useful experiment is therefore not a larger ViT. It is to make the model learn the expert action distribution directly enough that speed and steering become plausible before the safety governor intervenes.

## Primary Hypothesis

Adding a behavior-cloning action prior to the LeWM latent state should reduce unsafe CEM/action outputs:

```text
image history + action history -> latent world model
latent -> auxiliary driving state
latent -> expert action block
```

The action head is not a replacement for LeWM. It is a policy prior/readout trained on the same latent so we can test whether the learned latent is control-useful.

## Implementation Scope

1. Keep the existing latent prediction, SIGReg, aux, and pred-aux losses.
2. Add an `action_head` from latent to the flattened 5-frame action block.
3. Add `action_loss` on encoded latents and `pred_action_loss` on predicted latents.
4. Add step-based training with `trainer.max_steps`, because optimizer steps are the comparable unit for this stage.
5. Add `model_action` and `model_action_lane_keep` closed-loop policies.
6. Keep speed-gated scoring as the fair metric.

## Main Run

Config:

```bash
configs/train_tiny_action_prior_10k.yaml
```

Remote launch target:

```bash
cd /home/ubuntu/carla_lewm_drive
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny_action_prior_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_10k
```

Expected scale:

- Dataset: 80 episodes, 48,000 raw frames.
- Train split: 64 episodes.
- Tiny batch 192: about 193 optimizer steps per epoch.
- 10,000 optimizer steps: about 52 epochs.
- Early stop: patience 20 validation checks after 3,000 optimizer steps.

## Evaluation Commands

Automatic remote watcher:

```bash
cd /home/ubuntu/carla_lewm_drive
TRAIN_PID=<train-pid> tmux new-session -d -s carla_action10k_watch \
  "scripts/watch_action10k_and_eval.sh"
```

The watcher writes `outputs/d0_tiny_h3_fs5_delta_predaux_action_10k/watch_eval.log`, records 5-minute health snapshots, waits for training to finish, starts CARLA on port 2100 if needed, and runs the two 200m closed-loop evaluations below.

Pure action head:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_speedgate_200m
```

Action head plus deterministic lane/speed guard:

```bash
PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_lane_keep_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_lane_keep_speedgate_200m
```

## Go / No-Go Gates

Continue if:

- W&B starts online and logs train/val curves from process start.
- `best.pt`, `last.pt`, `metrics.csv`, `split_manifest.json`, and `test_metrics.json` exist.
- Validation loss improves before 10k steps or plateaus without divergence.
- `action_loss` and `pred_action_loss` decrease enough to change closed-loop behavior.

Pause if:

- Validation loss worsens for multiple epochs while train loss falls.
- Pure `model_action` fails before 50m by speed-limit violation.
- `model_action_lane_keep` only passes by hard clamping while raw model throttle stays unstable.

Stop this branch if:

- After 10k steps, speed-gated IFD does not improve over the 24.17m ungoverned model-lane-keep baseline.
- The action head learns throttle but steering remains worse than the existing governed hybrid.
- Disk/W&B/CARLA infrastructure prevents trustworthy curve and closed-loop evidence.

## Next Branches After This Gate

If action-prior improves speed but not steering:

- Collect recovery data with lateral/steering perturbations and expert correction.
- Add waypoint or centerline target conditioning.

If action-prior improves neither speed nor steering:

- Keep this as evidence against pure small-model action control on D0.
- Switch to governed latent planning as the honest project claim.

If action-prior passes 200m speed-gated pure policy:

- Repeat on 500m D0.
- Add a held-out spawn index before attempting D1 traffic.
