from pathlib import Path

import h5py
import numpy as np

from carla_lewm_drive.dataset_qc.validate_hdf5 import validate_hdf5


def make_dataset(path: Path, frames: int = 12) -> None:
    pixels = np.full((frames, 16, 16, 3), 127, dtype=np.uint8)
    pixels[:, 4:12, 4:12, :] = 200
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.array([6, 6], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array([0, 6], dtype=np.int32))
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("action", data=np.zeros((frames, 3), dtype=np.float32))
        f.create_dataset("state", data=np.zeros((frames, 6), dtype=np.float32))
        f.create_dataset("proprio", data=np.zeros((frames, 3), dtype=np.float32))
        f.create_dataset("ep_idx", data=np.repeat(np.arange(2), 6).astype(np.int32))
        f.create_dataset("step_idx", data=np.tile(np.arange(6), 2).astype(np.int32))
        f.create_dataset("route_progress_m", data=np.concatenate([np.arange(6), np.arange(6)]).astype(np.float32))
        f.create_dataset("speed_mps", data=np.ones(frames, dtype=np.float32))
        f.create_dataset("collision", data=np.zeros(frames, dtype=np.float32))
        f.create_dataset("offroad", data=np.zeros(frames, dtype=np.float32))
        f.create_dataset("red_light", data=np.zeros(frames, dtype=np.float32))


def test_validate_hdf5_passes_valid_dataset(tmp_path):
    dataset = tmp_path / "valid.h5"
    make_dataset(dataset)
    report = validate_hdf5(dataset, tmp_path / "qc", sample_frames=4)
    assert report["gate"] == "pass"
    assert (tmp_path / "qc" / "qc_report.json").exists()
    assert (tmp_path / "qc" / "contact_sheet.jpg").exists()


def test_validate_hdf5_fails_blank_frame(tmp_path):
    dataset = tmp_path / "blank.h5"
    make_dataset(dataset)
    with h5py.File(dataset, "a") as f:
        f["pixels"][0] = 0
    report = validate_hdf5(dataset, tmp_path / "qc_blank", sample_frames=12)
    assert report["gate"] == "fail"
