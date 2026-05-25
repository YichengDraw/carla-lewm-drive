from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics, aggregate_metrics


def test_mini_driving_score_penalizes_infractions():
    clean = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0)
    crash = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0, collision_count=1)
    lane_cross = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0, lane_invasion_count=1)
    speeding = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0, speed_limit_count=1)

    assert clean.route_completion_pct == 50.0
    assert clean.mini_driving_score == 50.0
    assert crash.mini_driving_score < clean.mini_driving_score
    assert lane_cross.mini_driving_score < clean.mini_driving_score
    assert lane_cross.primary_safety_failed
    assert lane_cross.primary_or_red_failed
    assert speeding.mini_driving_score < clean.mini_driving_score
    assert speeding.hard_failed
    assert not speeding.primary_safety_failed
    assert not speeding.primary_or_red_failed


def test_infraction_free_distance_uses_first_infraction_distance():
    row = DrivingEpisodeMetrics(
        route_length_m=500.0,
        route_progress_m=250.0,
        collision_count=1,
        first_infraction_distance_m=40.0,
    )

    assert row.route_completion_pct == 50.0
    assert row.first_infraction_free_distance_m == 40.0


def test_aggregate_metrics():
    rows = [
        DrivingEpisodeMetrics(route_length_m=100.0, route_progress_m=100.0),
        DrivingEpisodeMetrics(route_length_m=100.0, route_progress_m=50.0, lane_invasion_count=1),
    ]
    out = aggregate_metrics(rows)
    assert out["episodes"] == 2
    assert out["mean_route_completion_pct"] == 75.0
    assert out["success_rate_no_hard_infraction"] == 0.5
    assert out["success_rate_no_primary_safety_infraction"] == 0.5
    assert out["success_rate_no_primary_or_red_infraction"] == 0.5
    assert out["lane_invasion_count"] == 1
    assert out["speed_limit_count"] == 0
