# Execution Runbook

This runbook keeps the cheap, repeatable gates before expensive GPU or CARLA runs.

## Phase 0: Local Code Gate

```powershell
$env:PYTHONPATH="src"
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m carla_lewm_drive.html_report.build_interactive_plan --output interactive_plan.html
```

Gate: all tests pass and the HTML dashboard rebuilds.

## Phase 1: D0 Collection And QC

Run on the CARLA 0.9.16 host with the simulator already listening on `127.0.0.1:2000`.

```bash
export PYTHONPATH=src
python -m carla_lewm_drive.carla_collect.collect_dataset --config configs/d0_smoke.yaml
python -m carla_lewm_drive.dataset_qc.validate_hdf5 \
  --dataset data/d0_smoke/carla_d0_smoke.h5 \
  --out-dir outputs/qc_d0_smoke \
  --strict

python -m carla_lewm_drive.carla_collect.collect_dataset --config configs/d0_train.yaml
python -m carla_lewm_drive.dataset_qc.validate_hdf5 \
  --dataset data/d0_train/carla_d0_train.h5 \
  --out-dir outputs/qc_d0_train \
  --strict
```

Gate:

- Zero hard QC issues.
- Zero collision/off-road/red-light frames for D0.
- Contact sheet visually shows lane-centered driving, no sidewalk/building-heavy samples.

Observed first-pass result:

- `D0-smoke`: 20 episodes, 12,000 frames, strict pass.
- `D0-train`: 80 episodes, 48,000 frames, strict pass.

## Phase 2: Fast Dataset Export

Use this before training. Compressed HDF5 was too slow for random reads on this workload.

```bash
python -m carla_lewm_drive.dataset_qc.export_fast_hdf5 \
  --src data/d0_train/carla_d0_train.h5 \
  --dst data/d0_train/carla_d0_train_fast.h5 \
  --chunk-frames 256 \
  --overwrite
```

Gate: loader first batches should be seconds, not minutes. The first-pass run selected `num_workers=2`.

## Phase 3: Batch Probe

Tiny:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --num-workers 2 \
  --batch-probe \
  --output-dir outputs/d0_tiny_h3_fs5_probe
```

Small:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_small.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --num-workers 2 \
  --batch-probe \
  --output-dir outputs/d0_small_h3_fs5_probe
```

Observed first-pass result:

- Tiny: `batch_size=192` was viable; `256` OOMed.
- Small: `batch_size=128` passed with about `24.06GB` peak allocated VRAM.
- Latest small delta/pred-aux probe: `batch_size=128` OOMed, `96` passed with `18.139GB` peak allocated VRAM under the current remote load.

## Phase 4: W&B Training

Tiny 5-epoch comparison run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-epochs 5 \
  --output-dir outputs/d0_tiny_h3_fs5_fast_e5 \
  --run-name d0_tiny_h3_fs5_fast_e5
```

Tiny delta-progress + predicted-aux run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-epochs 5 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_e5 \
  --run-name d0_tiny_h3_fs5_delta_predaux_e5
```

Small delta-progress + predicted-aux run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_small.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 96 \
  --num-workers 2 \
  --max-epochs 5 \
  --output-dir outputs/d0_small_h3_fs5_delta_predaux_e5 \
  --run-name d0_small_h3_fs5_delta_predaux_e5
```

Completed tiny action-prior 10k-step run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny_action_prior_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_10k
```

Conflict-aware follow-up run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny_action_conflict_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_conflict_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_conflict_10k
```

Attach the automatic watcher and evaluation handoff:

```bash
cd /home/ubuntu/carla_lewm_drive
TRAIN_PID=<train-pid> tmux new-session -d -s carla_action10k_watch \
  "scripts/watch_action10k_and_eval.sh"
```

The watcher logs to `outputs/d0_tiny_h3_fs5_delta_predaux_action_10k/watch_eval.log`, records health snapshots every five minutes, then runs the pure action-policy and lane-guarded action-policy closed-loop checks after training exits.
For follow-up runs, override `RUN_DIR`, `RUN_NAME`, `CHECKPOINT_PATH`, `PURE_EVAL_DIR`, and `LANE_EVAL_DIR` so the watcher evaluates the new checkpoint.

Gate:

- W&B URL appears at launch.
- `metrics.csv`, `split_manifest.json`, `best.pt`, and `test_metrics.json` exist.
- Validation/test loss is finite.
- The stop reason is explicit.
- For action-prior runs, `action_loss` and `pred_action_loss` must be logged.

## Phase 5: Closed-Loop Evaluation

First validate the environment with autopilot. Use an isolated CARLA server for model-policy evaluation. If another client is ticking the same CARLA world, the evaluator raises on large `sim_delta_s` jumps.

Example isolated server launch:

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -world-port=2100
```

```bash
python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0.yaml \
  --baseline autopilot \
  --episodes 2 \
  --route-cap-m 100 \
  --max-episode-seconds 30 \
  --output-dir outputs/d0_eval_autopilot_e2_short
```

Then run the validated simplified model-policy target:

```bash
python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_throttle_only.yaml \
  --checkpoint outputs/d0_tiny_h3_fs5_delta_predaux_e5/best.pt \
  --route-cap-m 190 \
  --output-dir outputs/d0_eval_tiny_delta_predaux_throttle_only_190m
```

Then run the fair speed-gated hybrid checks:

```bash
python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_lane_keep.yaml \
  --output-dir outputs/d0_eval_tiny_model_lane_keep_200m_speedgate_v1

python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_lane_keep_governed.yaml \
  --output-dir outputs/d0_eval_tiny_model_lane_keep_governed_200m_v1
```

After the action-prior checkpoint exists, run the action-policy checks. Keep pure action and governed hybrid output directories separate:

```bash
python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_speedgate_200m_exclusive_v1

python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_model_action_lane_keep_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_lane_keep_speedgate_200m_brake_release_v2
```

Gate:

- Autopilot must score 100m IFD on the short route.
- A model checkpoint must reach at least 190m IFD with no hard infraction on the throttle-only D0 target before longer runs are attempted.
- A 200m speed-gated model-policy run must pass before longer 500m evaluation.
- A steering-safe 150m run must pass before claiming steering recovery.

Observed valid result:

- Autopilot passed.
- Tiny and small both passed 150m throttle-only on isolated CARLA port 2100.
- Tiny failed the 200m throttle-only run at 191.11m IFD by off-road.
- Small failed the 200m throttle-only run at 191.87m IFD by off-road.
- Latest tiny delta/pred-aux passed 150m and 190m throttle-only, then failed the 200m throttle-only run at 191.75m IFD by off-road.
- Latest tiny delta/pred-aux failed the steering-safe 150m run at 107.63m IFD by off-road.
- Latest small delta/pred-aux passed 190m throttle-only, then failed the 200m throttle-only run at 191.67m IFD by off-road.
- Latest small delta/pred-aux failed the steering-safe 150m run at 103.42m IFD by off-road.
- Lane-keep controller passed 200m with the 35km/h speed gate, max lane offset 0.097m, max speed 7.10m/s.
- Tiny delta/pred-aux plus lane-keep steering failed the speed-gated 200m run at 24.17m by speed-limit violation.
- Tiny delta/pred-aux plus lane/speed governor passed 200m with the 35km/h speed gate, max lane offset 0.094m, max speed 6.57m/s.
- Action-prior raw policies without throttle/brake exclusivity were blocked at about 0.003m because throttle and brake were predicted together.
- Action-prior pure `model_action` with throttle/brake exclusivity reached 24.84m, then failed by speed-limit violation.
- Action-prior `model_action_lane_keep` with throttle/brake exclusivity reached 157.69m, then failed by blocked.
- Action-prior `model_action_lane_keep` with exclusivity plus speed-governor brake release passed 200m with 100.0 mini score, max speed 6.38m/s, and max absolute lane offset 0.072m.
- Earlier port-2000 model-policy outputs were invalidated because an external HIL client was ticking CARLA during model inference.

## Stop Rule

Do not spend more time on larger ViTs until model-controlled speed and steering/lane keeping are fixed. Delta-progress and predicted-aux are now tested on tiny and small; action-prior 10k is complete. The current useful work is conflict-aware/speed-aware action training before any 500m run.
