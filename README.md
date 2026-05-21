# CARLA-LeWM Drive

Compact CARLA data collection, frame-level dataset quality control, and small LeWM training for simple long-horizon driving.

The project is intentionally narrow: start with an easy single-ego CARLA setting, prove that the data is clean frame by frame, train the smallest useful LeWM, then judge it with closed-loop distance before scaling the model or scenario.

## Highlights

- CARLA 0.9.16 collection config for `D0-smoke` and `D0-train`.
- Collection-time quality gates reject episodes that fail progress, collision, off-road, red-light, or blocked thresholds.
- StableWorldModel-style HDF5 layout: `pixels`, `action`, `state`, `proprio`, `ep_idx`, `step_idx`, `ep_len`, `ep_offset`.
- Frame-level QC with strict gates for missing keys, row mismatch, blank frames, non-finite actions, inconsistent episode indices, and driving infractions.
- Fast uncompressed HDF5 export (`carla-lewm-export-fast-hdf5`) for random-access training on the remote GPU host.
- Compact Driving-LeWM with ViT `tiny` and `small` configs, latent prediction, SIGReg-style regularization, and auxiliary driving heads.
- W&B logging for serious runs plus local `metrics.csv`, `split_manifest.json`, `best.pt`, and `test_metrics.json`.
- Closed-loop evaluator for CARLA autopilot baselines and checkpoint-driven model control, using `Infraction-Free Distance` and `Mini Driving Score`.
- Interactive Chinese project dashboard at `interactive_plan.html`.

## Current Evidence

Status after the first D0 execution pass:

- `D0-smoke`: 20 accepted episodes, 12,000 frames, strict QC pass, 0 collision/off-road/red-light/blocked frames.
- `D0-train`: 80 accepted episodes, 48,000 frames, strict QC pass, 0 collision/off-road/red-light/blocked frames.
- Fast training dataset: `data/d0_train/carla_d0_train_fast.h5`, used because the compressed HDF5 path was too slow for random reads.
- Tiny 5-epoch W&B run: [`d0_tiny_h3_fs5_fast_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_fast_e5-20260522-014214-db1cb3e1), test loss `2.9620`, test pred loss `0.7015`.
- Small 5-epoch W&B run: [`d0_small_h3_fs5_fast_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_fast_e5-20260522-024715-2f674d5e), test loss `1.1324`, test pred loss `0.5269`.
- Autopilot short baseline: 2/2 episodes, 100m mean infraction-free distance, 100.0 mean mini driving score.
- The first model sanity runs on port 2000 were invalid for model-policy conclusions because an external HIL client was also ticking CARLA. The evaluator now writes action/timing traces and fails on external tick jumps.
- Isolated CARLA port 2100, throttle-only D0 target: tiny and small checkpoints both reached 150m IFD with 100.0 mini driving score. Tiny reached 191.11m IFD on a 200m cap before off-road; small reached 191.87m in the same 200m setting.
- Wider steering CEM remains unreliable: tiny/small select negative steering at the planner boundary and leave the lane. The simplified throttle-only target is the current fair success claim.

Interpretation: the data and training pipeline are usable, and the tiny LeWM can satisfy a very simplified long-ish closed-loop D0 target. Increasing ViT size improved offline losses but did not extend the throttle-only 200m boundary, so the next bottleneck is still planner/objective/action calibration rather than raw encoder capacity.

## Installation

Use Python 3.10 or newer. Install the CARLA Python API wheel that matches the CARLA server version before collection or closed-loop evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

CARLA collection requires a running CARLA 0.9.16 server. Training can run on any CUDA host with the dataset available; the intended serious-training target is the RTX 5090 32GB host.

## Data Preparation

Start with the smoke scenario:

```bash
carla-lewm-collect --config configs/d0_smoke.yaml --episodes 2 --seconds 5
carla-lewm-qc --dataset data/d0_smoke/carla_d0_smoke.h5 --out-dir outputs/qc_d0_smoke --strict
```

Only collect `D0-train` after the smoke dataset passes QC:

```bash
carla-lewm-collect --config configs/d0_train.yaml
carla-lewm-qc --dataset data/d0_train/carla_d0_train.h5 --out-dir outputs/qc_d0_train --strict
```

The QC output contains:

- `qc_report.json`
- `sampled_frames.csv`
- `contact_sheet.jpg`

## Training

Probe batch size first:

```bash
carla-lewm-train --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --num-workers 2 \
  --batch-probe
```

Run the main tiny model:

```bash
carla-lewm-train --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-epochs 5 \
  --output-dir outputs/d0_tiny_h3_fs5_fast_e5 \
  --run-name d0_tiny_h3_fs5_fast_e5
```

Use `configs/train_small.yaml` only after tiny has failed a controlled gate. In the first valid isolated evaluation, `small` improved offline loss but did not extend the throttle-only 200m boundary, so further model scaling is not the next step.

## Evaluation

The metric layer is implemented in `carla_lewm_drive.closed_loop_eval.metrics`. Full closed-loop evaluation requires a running CARLA server and a trained checkpoint:

```bash
carla-lewm-eval --config configs/eval_d0.yaml --baseline autopilot --episodes 2 --route-cap-m 100 --max-episode-seconds 30
carla-lewm-eval --config configs/eval_d0.yaml --checkpoint outputs/d0_tiny_h3_fs5_fast_e5/best.pt \
  --episodes 1 --route-cap-m 50 --max-episode-seconds 8 \
  --planner-samples 64 --planner-iterations 2 --planner-horizon 4
```

For the current validated simplified target, use an isolated CARLA server on port 2100 and the throttle-only config:

```bash
carla-lewm-eval --config configs/eval_d0_throttle_only.yaml \
  --checkpoint outputs/d0_tiny_h3_fs5_fast_e5/best.pt \
  --output-dir outputs/d0_eval_tiny_throttle_only_150m
```

Primary metrics:

- `Infraction-Free Distance`: meters before first collision, off-road event, red-light violation, or blocked/stuck condition.
- `Mini Driving Score`: route completion percentage multiplied by infraction penalties.

## Interactive Plan

Build or refresh the HTML project dashboard:

```bash
carla-lewm-build-html --output interactive_plan.html
```

Open `interactive_plan.html` in a browser. It includes phase gates, model-size policy, data QC gates, W&B expectations, metrics, and run-name decoding.

## Current Limitations

- This repository does not vendor the CARLA simulator or Python wheel.
- Full closed-loop control must be run on a machine where the CARLA server is active.
- The first credible claim should be limited to the simplified D0/D1 settings until held-out closed-loop videos and metrics exist.
- Large datasets, checkpoints, W&B folders, videos, and CARLA logs are ignored by git.

## License

MIT.
