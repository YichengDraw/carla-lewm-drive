from pathlib import Path

import h5py
import numpy as np

from carla_lewm_drive.dataset_qc.export_fast_hdf5 import export_fast_hdf5


def test_export_fast_hdf5_copies_pixels_without_compression(tmp_path: Path):
    src = tmp_path / "src.h5"
    dst = tmp_path / "dst.h5"
    pixels = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    with h5py.File(src, "w") as f:
        f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=4)
        f.create_dataset("ep_len", data=np.array([4], dtype=np.int32))

    summary = export_fast_hdf5(src, dst, chunk_frames=2)

    assert summary["frames"] == 4
    with h5py.File(dst, "r") as f:
        assert f["pixels"].compression is None
        assert f["pixels"].chunks == (2, 8, 8, 3)
        assert np.array_equal(f["pixels"][:], pixels)
        assert np.array_equal(f["ep_len"][:], np.array([4], dtype=np.int32))
