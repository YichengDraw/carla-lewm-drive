import h5py
import numpy as np
import torch

from carla_lewm_drive.driving_lewm.data import CarlaSequenceDataset, MultiDatasetSplit, build_splits, split_episodes


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


def write_split_test_h5(path, *, episodes=4, frames_per_episode=16, route_id=3):
    frames = episodes * frames_per_episode
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.full(episodes, frames_per_episode, dtype=np.int32))
        f.create_dataset("ep_offset", data=np.arange(episodes, dtype=np.int32) * frames_per_episode)
        f.create_dataset("pixels", data=np.full((frames, 32, 32, 3), 128, dtype=np.uint8))
        f.create_dataset("action", data=np.zeros((frames, 3), dtype=np.float32))
        f.create_dataset("ep_idx", data=np.repeat(np.arange(episodes, dtype=np.int32), frames_per_episode))
        f.create_dataset("step_idx", data=np.tile(np.arange(frames_per_episode, dtype=np.int32), episodes))
        f.create_dataset("route_id", data=np.full(frames, route_id, dtype=np.int32))


def test_build_splits_supports_repeated_multi_dataset_training(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    write_split_test_h5(a, route_id=3)
    write_split_test_h5(b, route_id=10)
    cfg = {
        "frameskip": 2,
        "history_size": 3,
        "num_preds": 1,
        "image_size": 32,
        "train_split": 0.5,
        "val_split": 0.25,
        "split_seed": 7,
        "dataset_repeat_factors": [1, 3],
    }

    train, val, test, split = build_splits([a, b], cfg)

    assert isinstance(split, MultiDatasetSplit)
    assert split.datasets[1]["train_repeat_factor"] == 3
    assert len(train.datasets) == 4
    assert len(val.datasets) == 2
    assert len(test.datasets) == 2
