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
- Closed-loop evaluator for CARLA autopilot baselines, checkpoint-driven model control, lane-keep/hybrid sanity checks, and speed-limit-aware `Infraction-Free Distance` / `Mini Driving Score`.
- Action-prior extension: a latent action head, 10k-step W&B run, action trace diagnostics, and throttle/brake sanitization for CARLA control.
- Interactive Chinese project dashboard at `interactive_plan.html`.

## Current Evidence

Status after the first D0 execution pass:

- `D0-smoke`: 20 accepted episodes, 12,000 frames, strict QC pass, 0 collision/off-road/red-light/blocked frames.
- `D0-train`: 80 accepted episodes, 48,000 frames, strict QC pass, 0 collision/off-road/red-light/blocked frames.
- Fast training dataset: `data/d0_train/carla_d0_train_fast.h5`, used because the compressed HDF5 path was too slow for random reads.
- Historical absolute-progress tiny run: [`d0_tiny_h3_fs5_fast_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_fast_e5-20260522-014214-db1cb3e1), test loss `2.9620`, test pred loss `0.7015`.
- Historical absolute-progress small run: [`d0_small_h3_fs5_fast_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_fast_e5-20260522-024715-2f674d5e), test loss `1.1324`, test pred loss `0.5269`.
- Current tiny delta-progress + predicted-aux run: [`d0_tiny_h3_fs5_delta_predaux_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_delta_predaux_e5-20260522-042925-552fb819), test loss `0.2147`, test pred loss `0.0625`, test aux loss `0.1606`, test pred-aux loss `0.1025`.
- Current small delta-progress + predicted-aux run: [`d0_small_h3_fs5_delta_predaux_e5`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_delta_predaux_e5-20260522-054348-da4e9fd5), best checkpoint from epoch 1, test loss `0.3300`, test pred loss `0.1243`, test pred-aux loss `0.1225`.
- Autopilot short baseline: 2/2 episodes, 100m mean infraction-free distance, 100.0 mean mini driving score.
- The first model sanity runs on port 2000 were invalid for model-policy conclusions because an external HIL client was also ticking CARLA. The evaluator now writes action/timing traces and fails on external tick jumps.
- Isolated CARLA port 2100, latest tiny delta/pred-aux checkpoint, throttle-only D0 target: 150m pass, 190m pass, and 200m failed by off-road with first infraction at `191.75m`.
- Isolated CARLA port 2100, small delta/pred-aux checkpoint, throttle-only D0 target: 190m pass and 200m failed by off-road with first infraction at `191.67m`.
- Steering-safe D0 target remains failed: tiny first went off-road at `107.63m`, and small first went off-road at `103.42m` on the 150m cap.
- Speed-gated lane-keep controller sanity baseline: 200m pass, 100.0 mini score, max lane offset `0.097m`, max speed `7.10m/s`.
- Speed-gated tiny model + lane-keep steering without speed governor: failed at `24.17m` from speed-limit violation, proving the fair metric must include speed rules.
- Speed-gated tiny model + lane/speed governor: 200m pass, 100.0 mini score, max lane offset `0.094m`, max speed `6.57m/s`.
- Action-prior 10k tiny run: [`d0_tiny_h3_fs5_delta_predaux_action_10k`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_delta_predaux_action_10k-20260522-103256-1a514e1e), best val loss `0.1425` at step `9843`, test loss `0.1400`, test action loss `0.0201`, test pred-action loss `0.0116`.
- Raw action-policy evaluation exposed a control-interface failure: without throttle/brake exclusivity, both `model_action` and `model_action_lane_keep` were blocked at about `0.003m` because the action head often predicted throttle and brake together.
- With throttle/brake exclusivity, pure `model_action` reached `24.84m` before speed-limit violation; `model_action_lane_keep` reached `157.69m` before blocked.
- With throttle/brake exclusivity plus speed-governor brake release, `model_action_lane_keep` reached `200.00m`, 100.0 mini score, success rate `1.0`, max speed `6.38m/s`, max absolute lane offset `0.072m`.
- Conflict-aware action run: [`d0_tiny_h3_fs5_delta_predaux_action_conflict_10k`](https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_delta_predaux_action_conflict_10k-20260522-205206-3b40dc67), early-stopped at step `5800` with best total validation loss `0.1685` at step `2900`. Best-checkpoint test loss was `0.1698`, test action loss `0.0370`, and test pred-action loss `0.0354`.
- The conflict-aware run required adding the same CEM planner block to `eval_d0_model_action_speedgate.yaml` and `eval_d0_model_action_lane_keep_speedgate.yaml`; without that, the action-policy evaluator failed before closed-loop scoring.
- Closed-loop candidate comparison on the conflict-aware run: total-loss `best.pt` reached `22.14m` pure and `22.94m` lane-keep; action-best step `4350` reached `25.06m` pure and `25.00m` lane-keep; pred-action-best step `5800` reached `20.67m` pure and `20.87m` lane-keep. All six had zero collision/off-road/red-light/blocked events but failed the 200m target because each triggered one speed-limit violation.

Interpretation: the data and training pipeline are usable, and the tiny/small LeWM variants can satisfy a very simplified model-driven D0 target up to 190m. Delta-progress and predicted-aux training improved offline learning, but scaling from tiny to small did not solve steering or speed compliance. The action-prior head learned useful controls only after the evaluator enforced the same throttle/brake exclusivity that exists in the dataset. Conflict-aware action training improved the best pure action distance slightly, but it did not solve speed compliance. The current 200m pass remains a governed hybrid result, not a pure action-policy success.

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
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_e5 \
  --run-name d0_tiny_h3_fs5_delta_predaux_e5
```

Use `configs/train_small.yaml` only after tiny has failed a controlled gate. In this D0 run, small delta-progress + predicted-aux did not improve the throttle-only 200m boundary or the steering-safe 150m gate, so the next useful work is control/objective calibration rather than a larger ViT.

Completed control-focused tiny run:

```bash
carla-lewm-train --config configs/train_tiny_action_prior_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_10k
```

Next conflict-aware action run:

```bash
carla-lewm-train --config configs/train_tiny_action_conflict_10k.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 \
  --num-workers 2 \
  --max-steps 10000 \
  --output-dir outputs/d0_tiny_h3_fs5_delta_predaux_action_conflict_10k \
  --run-name d0_tiny_h3_fs5_delta_predaux_action_conflict_10k
```

## Evaluation

The metric layer is implemented in `carla_lewm_drive.closed_loop_eval.metrics`. Full closed-loop evaluation requires a running CARLA server and a trained checkpoint:

```bash
carla-lewm-eval --config configs/eval_d0.yaml --baseline autopilot --episodes 2 --route-cap-m 100 --max-episode-seconds 30
carla-lewm-eval --config configs/eval_d0.yaml --checkpoint outputs/d0_tiny_h3_fs5_delta_predaux_e5/best.pt \
  --episodes 1 --route-cap-m 50 --max-episode-seconds 8 \
  --planner-samples 64 --planner-iterations 2 --planner-horizon 4
```

For the current validated simplified target, use an isolated CARLA server on port 2100 and the throttle-only config:

```bash
carla-lewm-eval --config configs/eval_d0_throttle_only.yaml \
  --checkpoint outputs/d0_tiny_h3_fs5_delta_predaux_e5/best.pt \
  --route-cap-m 190 \
  --output-dir outputs/d0_eval_tiny_delta_predaux_throttle_only_190m
```

Primary metrics:

- `Infraction-Free Distance`: meters before first collision, off-road event, red-light violation, or blocked/stuck condition.
- `Mini Driving Score`: route completion percentage multiplied by infraction penalties, including speed-limit violations when `speed_limit_kmh` is set.

Speed-gated hybrid sanity checks:

```bash
carla-lewm-eval --config configs/eval_d0_model_lane_keep.yaml \
  --output-dir outputs/d0_eval_tiny_model_lane_keep_200m_speedgate_v1
carla-lewm-eval --config configs/eval_d0_model_lane_keep_governed.yaml \
  --output-dir outputs/d0_eval_tiny_model_lane_keep_governed_200m_v1
```

After the action-prior checkpoint exists:

```bash
carla-lewm-eval --config configs/eval_d0_model_action_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_speedgate_200m_exclusive_v1
carla-lewm-eval --config configs/eval_d0_model_action_lane_keep_speedgate.yaml \
  --output-dir outputs/d0_eval_tiny_model_action_lane_keep_speedgate_200m_brake_release_v2
```

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
