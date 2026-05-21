# Current Execution Status

Last checked: 2026-05-21 Asia/Shanghai.

## Completed

- Project isolated under `carla_lewm_drive/` as its own git repository.
- Interactive dashboard generated at `interactive_plan.html`.
- CARLA collection configs added for `D0-smoke` and `D0-train`.
- HDF5 QC, compact Driving-LeWM training, and metric code added.
- Lightweight tests passed in the existing workspace virtual environment: `6 passed`.
- SSH 5090 host reachable: `NVIDIA GeForce RTX 5090`, `32607 MiB`, driver `590.44.01`.

## Local Environment

- Existing workspace venv: parent workspace `.venv`.
- In that venv: `wandb`, CUDA `torch`, `h5py`, and `transformers` are installed.
- CARLA Python API is not installed locally yet. Collection and full closed-loop evaluation require the CARLA 0.9.16 wheel matching the simulator server.
- Default system Python lacks `h5py` and `pytest`; use the project venv or run `pip install -r requirements.txt`.

## Next Hard Gate

Run a CARLA 0.9.16 server, install the matching Python API wheel, collect `D0-smoke`, and pass:

```powershell
$env:PYTHONPATH="src"
..\.venv\Scripts\python.exe -m carla_lewm_drive.carla_collect.collect_dataset --config configs\d0_smoke.yaml --episodes 2 --seconds 5
..\.venv\Scripts\python.exe -m carla_lewm_drive.dataset_qc.validate_hdf5 --dataset data\d0_smoke\carla_d0_smoke.h5 --out-dir outputs\qc_d0_smoke --strict
```
