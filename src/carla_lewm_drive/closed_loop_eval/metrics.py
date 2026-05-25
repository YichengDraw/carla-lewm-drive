from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrivingEpisodeMetrics:
    route_length_m: float
    route_progress_m: float
    collision_count: int = 0
    offroad_count: int = 0
    lane_invasion_count: int = 0
    red_light_count: int = 0
    blocked_count: int = 0
    speed_limit_count: int = 0
    first_infraction_distance_m: float | None = None

    @property
    def route_completion_pct(self) -> float:
        if self.route_length_m <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * self.route_progress_m / self.route_length_m))

    @property
    def first_infraction_free_distance_m(self) -> float:
        distance = self.route_progress_m if self.first_infraction_distance_m is None else self.first_infraction_distance_m
        return max(0.0, min(distance, self.route_length_m))

    @property
    def infraction_penalty(self) -> float:
        penalty = 1.0
        penalty *= 0.50 ** self.collision_count
        penalty *= 0.70 ** self.offroad_count
        penalty *= 0.75 ** self.lane_invasion_count
        penalty *= 0.70 ** self.red_light_count
        penalty *= 0.80 ** self.blocked_count
        penalty *= 0.85 ** self.speed_limit_count
        return penalty

    @property
    def mini_driving_score(self) -> float:
        return self.route_completion_pct * self.infraction_penalty

    @property
    def hard_failed(self) -> bool:
        return (
            self.collision_count
            + self.offroad_count
            + self.lane_invasion_count
            + self.red_light_count
            + self.blocked_count
            + self.speed_limit_count
        ) > 0

    @property
    def primary_safety_failed(self) -> bool:
        return (self.collision_count + self.offroad_count + self.lane_invasion_count + self.blocked_count) > 0

    @property
    def primary_or_red_failed(self) -> bool:
        return (
            self.collision_count
            + self.offroad_count
            + self.lane_invasion_count
            + self.red_light_count
            + self.blocked_count
        ) > 0


def aggregate_metrics(rows: list[DrivingEpisodeMetrics]) -> dict[str, float]:
    if not rows:
        raise ValueError("No episode metrics to aggregate")
    n = float(len(rows))
    return {
        "episodes": len(rows),
        "mean_infraction_free_distance_m": sum(r.first_infraction_free_distance_m for r in rows) / n,
        "mean_route_completion_pct": sum(r.route_completion_pct for r in rows) / n,
        "mean_mini_driving_score": sum(r.mini_driving_score for r in rows) / n,
        "success_rate_no_hard_infraction": sum(0 if r.hard_failed else 1 for r in rows) / n,
        "success_rate_no_primary_safety_infraction": sum(0 if r.primary_safety_failed else 1 for r in rows) / n,
        "success_rate_no_primary_or_red_infraction": sum(0 if r.primary_or_red_failed else 1 for r in rows) / n,
        "collision_count": sum(r.collision_count for r in rows),
        "offroad_count": sum(r.offroad_count for r in rows),
        "lane_invasion_count": sum(r.lane_invasion_count for r in rows),
        "red_light_count": sum(r.red_light_count for r in rows),
        "blocked_count": sum(r.blocked_count for r in rows),
        "speed_limit_count": sum(r.speed_limit_count for r in rows),
    }
