from __future__ import annotations

from typing import Any

import numpy as np


_SEMANTIC_CAMERA_TYPES = {"semantic", "semantic_segmentation"}
# CARLA 0.9.16 CityObjectLabel ids from the Python API on the target host:
# Roads=1, Sidewalks=2, RoadLines=24.
_ROAD_TAGS = np.asarray([1], dtype=np.uint8)
_ROAD_LINE_TAGS = np.asarray([24], dtype=np.uint8)
_OBSTACLE_TAGS = np.asarray(
    [
        2,   # Sidewalks
        4,   # Walls
        5,   # Fences
        6,   # Poles
        7,   # TrafficLight
        8,   # TrafficSigns
        10,  # Terrain
        12,  # Pedestrians
        13,  # Rider
        14,  # Car
        15,  # Truck
        16,  # Bus
        17,  # Train
        18,  # Motorcycle
        19,  # Bicycle
        20,  # Static
        21,  # Dynamic
        22,  # Other
        25,  # Ground
        28,  # GuardRail
    ],
    dtype=np.uint8,
)


def camera_type(cfg: dict[str, Any] | None) -> str:
    camera_cfg = (cfg or {}).get("camera", {})
    return str(camera_cfg.get("type", "rgb")).strip().lower().replace("-", "_")


def camera_sensor_id(cfg: dict[str, Any] | None) -> str:
    kind = camera_type(cfg)
    if kind == "rgb":
        return "sensor.camera.rgb"
    if kind in _SEMANTIC_CAMERA_TYPES:
        return "sensor.camera.semantic_segmentation"
    raise ValueError(f"Unsupported camera.type {kind!r}; expected 'rgb' or 'semantic'")


def is_semantic_camera(cfg: dict[str, Any] | None) -> bool:
    return camera_type(cfg) in _SEMANTIC_CAMERA_TYPES


def semantic_tags_to_rgb(tags: np.ndarray) -> np.ndarray:
    tags = np.asarray(tags, dtype=np.uint8)
    out = np.zeros((*tags.shape, 3), dtype=np.uint8)
    out[..., 0] = np.isin(tags, _ROAD_TAGS).astype(np.uint8) * 255
    out[..., 1] = np.isin(tags, _ROAD_LINE_TAGS).astype(np.uint8) * 255
    out[..., 2] = np.isin(tags, _OBSTACLE_TAGS).astype(np.uint8) * 255
    return out


def read_camera_image(image: Any, cfg: dict[str, Any] | None = None) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    if is_semantic_camera(cfg):
        # CARLA semantic camera stores the object class id in the red byte of BGRA raw data.
        return semantic_tags_to_rgb(arr[:, :, 2])
    return arr[:, :, :3][:, :, ::-1].copy()
