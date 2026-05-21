from carla_lewm_drive.closed_loop_eval.metrics import DrivingEpisodeMetrics, aggregate_metrics


def test_mini_driving_score_penalizes_infractions():
    clean = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0)
    crash = DrivingEpisodeMetrics(route_length_m=500.0, route_progress_m=250.0, collision_count=1)

    assert clean.route_completion_pct == 50.0
    assert clean.mini_driving_score == 50.0
    assert crash.mini_driving_score < clean.mini_driving_score


def test_aggregate_metrics():
    rows = [
        DrivingEpisodeMetrics(route_length_m=100.0, route_progress_m=100.0),
        DrivingEpisodeMetrics(route_length_m=100.0, route_progress_m=50.0, offroad_count=1),
    ]
    out = aggregate_metrics(rows)
    assert out["episodes"] == 2
    assert out["mean_route_completion_pct"] == 75.0
    assert out["success_rate_no_hard_infraction"] == 0.5
