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

Small 5-epoch comparison run:

```bash
python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_small.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 128 \
  --num-workers 2 \
  --max-epochs 5 \
  --output-dir outputs/d0_small_h3_fs5_fast_e5 \
  --run-name d0_small_h3_fs5_fast_e5
```

Gate:

- W&B URL appears at launch.
- `metrics.csv`, `split_manifest.json`, `best.pt`, and `test_metrics.json` exist.
- Validation/test loss is finite.
- The stop reason is explicit.

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
  --checkpoint outputs/d0_tiny_h3_fs5_fast_e5/best.pt \
  --output-dir outputs/d0_eval_tiny_throttle_only_150m
```

Gate:

- Autopilot must score 100m IFD on the short route.
- A model checkpoint must reach at least 150m IFD with no hard infraction on the throttle-only D0 target before steering is re-enabled.
- A 200m throttle-only run must pass before longer 500m evaluation.

Observed valid result:

- Autopilot passed.
- Tiny and small both passed 150m throttle-only on isolated CARLA port 2100.
- Tiny failed the 200m throttle-only run at 191.11m IFD by off-road.
- Small failed the 200m throttle-only run at 191.87m IFD by off-road.
- Earlier port-2000 model-policy outputs were invalidated because an external HIL client was ticking CARLA during model inference.

## Stop Rule

Do not spend more time on larger ViTs until steering/action-objective calibration is fixed. The next useful experiment is a steering-safe policy or delta-progress planner objective, not `base` model training.
