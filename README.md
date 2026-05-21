# CARLA-LeWM Drive

Compact CARLA data collection, frame-level dataset quality control, and small LeWM training for simple long-horizon driving.

The project is intentionally narrow: start with an easy single-ego CARLA setting, prove that the data is clean frame by frame, train the smallest useful LeWM, then judge it with closed-loop distance before scaling the model or scenario.

## Highlights

- CARLA 0.9.16 collection config for `D0-smoke` and `D0-train`.
- StableWorldModel-style HDF5 layout: `pixels`, `action`, `state`, `proprio`, `ep_idx`, `step_idx`, `ep_len`, `ep_offset`.
- Frame-level QC with hard gates for missing keys, row mismatch, blank frames, non-finite actions, and inconsistent episode indices.
- Compact Driving-LeWM with ViT `tiny` and `small` configs, latent prediction, SIGReg-style regularization, and auxiliary driving heads.
- W&B logging for serious runs plus local `metrics.csv`, `split_manifest.json`, `best.pt`, and `test_metrics.json`.
- Closed-loop metric definitions for `Infraction-Free Distance` and `Mini Driving Score`.
- Interactive Chinese project dashboard at `interactive_plan.html`.

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
carla-lewm-train --config configs/train_tiny.yaml --batch-probe
```

Run the main tiny model:

```bash
carla-lewm-train --config configs/train_tiny.yaml
```

Use `configs/train_small.yaml` only when tiny passes data and planner sanity checks but appears capacity-limited.

## Evaluation

The metric layer is implemented in `carla_lewm_drive.closed_loop_eval.metrics`. Full closed-loop evaluation requires a running CARLA server and a trained checkpoint:

```bash
carla-lewm-eval --config configs/eval_d0.yaml --checkpoint outputs/d0_tiny_h3_fs5/best.pt
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
