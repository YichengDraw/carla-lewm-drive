from __future__ import annotations

from dataclasses import dataclass


REQUIRED_H5_KEYS = (
    "ep_len",
    "ep_offset",
    "pixels",
    "action",
    "state",
    "proprio",
    "ep_idx",
    "step_idx",
)

OPTIONAL_H5_KEYS = (
    "route_id",
    "weather_id",
    "timestamp",
    "speed_mps",
    "route_progress_m",
    "lane_offset_m",
    "heading_error_rad",
    "collision",
    "offroad",
    "red_light",
    "blocked",
)


@dataclass(frozen=True)
class DrivingActionBounds:
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    steer_min: float = -1.0
    steer_max: float = 1.0
    brake_min: float = 0.0
    brake_max: float = 1.0

    @property
    def low(self) -> tuple[float, float, float]:
        return (self.throttle_min, self.steer_min, self.brake_min)

    @property
    def high(self) -> tuple[float, float, float]:
        return (self.throttle_max, self.steer_max, self.brake_max)
