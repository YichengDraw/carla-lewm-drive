import h5py
import numpy as np
import torch

from carla_lewm_drive.driving_lewm.data import CarlaSequenceDataset, split_episodes


def test_split_episodes_is_episode_disjoint():
    split = split_episodes(10, train_split=0.8, val_split=0.1, seed=1)
    train = set(split.train_episodes)
    val = set(split.val_episodes)
    test = set(split.test_episodes)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 10


def test_carla_sequence_dataset_shapes(tmp_path):
    path = tmp_path / "tiny.h5"
    frames = 16
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.array([frames], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array([0], dtype=np.int32))
        f.create_dataset("pixels", data=np.full((frames, 32, 32, 3), 128, dtype=np.uint8))
        f.create_dataset("action", data=np.zeros((frames, 3), dtype=np.float32))
        f.create_dataset("state", data=np.zeros((frames, 6), dtype=np.float32))
        f.create_dataset("proprio", data=np.zeros((frames, 3), dtype=np.float32))
        f.create_dataset("ep_idx", data=np.zeros(frames, dtype=np.int32))
        f.create_dataset("step_idx", data=np.arange(frames, dtype=np.int32))
        f.create_dataset("route_id", data=np.full(frames, 10, dtype=np.int32))
    ds = CarlaSequenceDataset(path, frameskip=2, history_size=3, num_preds=1, image_size=32)
    item = ds[0]
    assert item["pixels"].shape == (4, 3, 32, 32)
    assert item["action"].shape == (4, 6)
    assert item["route_id"].shape == (4, 1)
    assert item["route_id"].dtype == torch.long
    assert item["route_id"].squeeze(-1).tolist() == [10, 10, 10, 10]
