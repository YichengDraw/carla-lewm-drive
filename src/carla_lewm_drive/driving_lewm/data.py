from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class EpisodeSplit:
    train_episodes: list[int]
    val_episodes: list[int]
    test_episodes: list[int]


def split_episodes(
    num_episodes: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> EpisodeSplit:
    if num_episodes < 3:
        raise ValueError("Need at least 3 episodes for train/val/test splitting")
    if train_split <= 0 or val_split <= 0 or train_split + val_split >= 1:
        raise ValueError("Expected positive train/val splits with non-empty test split")

    rng = np.random.default_rng(seed)
    episodes = np.arange(num_episodes)
    rng.shuffle(episodes)
    n_train = max(1, int(math.floor(num_episodes * train_split)))
    n_val = max(1, int(math.floor(num_episodes * val_split)))
    n_train = min(n_train, num_episodes - 2)
    n_val = min(n_val, num_episodes - n_train - 1)
    train = sorted(int(x) for x in episodes[:n_train])
    val = sorted(int(x) for x in episodes[n_train : n_train + n_val])
    test = sorted(int(x) for x in episodes[n_train + n_val :])
    return EpisodeSplit(train, val, test)


class CarlaSequenceDataset(Dataset):
    """StableWorldModel-style CARLA HDF5 sequence dataset.

    Pixels are sampled every `frameskip` raw frames. Actions keep every raw
    control inside the block and are flattened to match the LeWM convention.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        frameskip: int,
        history_size: int,
        num_preds: int,
        image_size: int = 224,
        episodes: Iterable[int] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.frameskip = int(frameskip)
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.num_steps = self.history_size + self.num_preds
        self.span = self.num_steps * self.frameskip
        self.image_size = int(image_size)
        self._h5: h5py.File | None = None

        with h5py.File(self.dataset_path, "r") as f:
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            self.action_dim = int(f["action"].shape[1])
            selected = set(range(len(self.ep_len))) if episodes is None else {int(e) for e in episodes}
            self.clip_indices = [
                (ep, start)
                for ep, length in enumerate(self.ep_len)
                if ep in selected and length >= self.span
                for start in range(int(length) - self.span + 1)
            ]
        if not self.clip_indices:
            raise ValueError(
                f"No valid clips in {self.dataset_path}; need episodes with at least {self.span} raw frames"
            )

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r", swmr=True)
        return self._h5

    def _load_optional(self, h5: h5py.File, key: str, indices: np.ndarray, dim: int = 1) -> torch.Tensor:
        if key in h5:
            data = np.asarray(h5[key][indices], dtype=np.float32)
            if data.ndim == 1 and dim == 1:
                data = data[:, None]
            return torch.from_numpy(data)
        return torch.zeros((len(indices), dim), dtype=torch.float32)

    def _load_optional_int(self, h5: h5py.File, key: str, indices: np.ndarray, dim: int = 1) -> torch.Tensor:
        if key in h5:
            data = np.asarray(h5[key][indices], dtype=np.int64)
            if data.ndim == 1 and dim == 1:
                data = data[:, None]
            return torch.from_numpy(data)
        return torch.zeros((len(indices), dim), dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5 = self._open()
        ep, start = self.clip_indices[idx]
        offset = int(self.ep_offset[ep])
        raw_indices = np.arange(offset + start, offset + start + self.span, dtype=np.int64)
        frame_indices = raw_indices[:: self.frameskip]

        pixels_np = np.asarray(h5["pixels"][frame_indices], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).float().div_(255.0)
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            pixels = torch.stack([TF.resize(frame, [self.image_size, self.image_size], antialias=True) for frame in pixels])
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD

        actions = np.asarray(h5["action"][raw_indices], dtype=np.float32)
        action = torch.from_numpy(actions.reshape(self.num_steps, self.frameskip * self.action_dim))

        aux = {
            "speed_mps": self._load_optional(h5, "speed_mps", frame_indices),
            "route_progress_m": self._load_optional(h5, "route_progress_m", frame_indices),
            "lane_offset_m": self._load_optional(h5, "lane_offset_m", frame_indices),
            "heading_error_rad": self._load_optional(h5, "heading_error_rad", frame_indices),
            "collision": self._load_optional(h5, "collision", frame_indices),
            "offroad": self._load_optional(h5, "offroad", frame_indices),
            "red_light": self._load_optional(h5, "red_light", frame_indices),
            "blocked": self._load_optional(h5, "blocked", frame_indices),
            "route_id": self._load_optional_int(h5, "route_id", frame_indices),
        }
        return {
            "pixels": pixels,
            "action": action,
            "episode": torch.tensor(ep, dtype=torch.long),
            "start_step": torch.tensor(start, dtype=torch.long),
            **aux,
        }


def build_splits(dataset_path: str | Path, cfg: dict) -> tuple[Subset, Subset, Subset, EpisodeSplit]:
    with h5py.File(dataset_path, "r") as f:
        num_episodes = int(len(f["ep_len"]))
    split = split_episodes(
        num_episodes,
        float(cfg["train_split"]),
        float(cfg["val_split"]),
        int(cfg["split_seed"]),
    )

    common = {
        "dataset_path": dataset_path,
        "frameskip": int(cfg["frameskip"]),
        "history_size": int(cfg["history_size"]),
        "num_preds": int(cfg["num_preds"]),
        "image_size": int(cfg["image_size"]),
    }
    train = CarlaSequenceDataset(**common, episodes=split.train_episodes)
    val = CarlaSequenceDataset(**common, episodes=split.val_episodes)
    test = CarlaSequenceDataset(**common, episodes=split.test_episodes)
    return train, val, test, split
