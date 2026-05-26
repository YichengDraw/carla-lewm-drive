import h5py
import numpy as np

from scripts.filter_hdf5_by_progress import filter_by_progress


def write_dataset(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=np.zeros((8, 4, 4, 3), dtype=np.uint8))
        f.create_dataset("route_progress_m", data=np.asarray([0, 10, 20, 30, 0, 12, 24, 36], dtype=np.float32))
        f.create_dataset("lane_offset_m", data=np.asarray([0, 0.1, 0.2, 0.3, 0, -0.1, -0.2, -0.3], dtype=np.float32))
        f.create_dataset("route_id", data=np.full(8, 6, dtype=np.int32))
        f.create_dataset("action", data=np.zeros((8, 3), dtype=np.float32))
        f.create_dataset("ep_len", data=np.asarray([4, 4], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.asarray([0, 4], dtype=np.int32))
        f.create_dataset("ep_idx", data=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32))
        f.create_dataset("step_idx", data=np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32))


def test_filter_by_progress_keeps_contiguous_episode_segments(tmp_path):
    src = tmp_path / "in.h5"
    dst = tmp_path / "out.h5"
    write_dataset(src)

    report = filter_by_progress(src, dst, min_progress_m=15.0, overwrite=True)

    assert report["episodes"] == 2
    assert report["frames"] == 4
    with h5py.File(dst, "r") as f:
        assert f["ep_len"][:].tolist() == [2, 2]
        assert f["ep_offset"][:].tolist() == [0, 2]
        assert f["ep_idx"][:].tolist() == [0, 0, 1, 1]
        assert f["step_idx"][:].tolist() == [0, 1, 0, 1]
        assert f["route_progress_m"][:].tolist() == [20.0, 30.0, 24.0, 36.0]
        assert f.attrs["min_progress_m"] == 15.0
