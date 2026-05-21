import pytest
import torch

from carla_lewm_drive.driving_lewm.model import DrivingLeWM


def batch_with_progress(progress):
    progress_t = torch.tensor(progress, dtype=torch.float32).view(1, -1, 1)
    zeros = torch.zeros_like(progress_t)
    ones = torch.ones_like(progress_t)
    return {
        "speed_mps": ones,
        "route_progress_m": progress_t,
        "lane_offset_m": zeros,
        "heading_error_rad": zeros,
        "collision": zeros,
        "offroad": zeros,
        "red_light": zeros,
        "blocked": zeros,
    }


def test_progress_signal_supports_absolute_and_delta_modes():
    route_progress = torch.tensor([[[10.0], [12.5], [15.0]]])

    absolute = DrivingLeWM.progress_signal(route_progress, "absolute")
    delta = DrivingLeWM.progress_signal(route_progress, "delta")

    assert absolute.tolist() == [[[10.0], [12.5], [15.0]]]
    assert delta.tolist() == [[[0.0], [2.5], [2.5]]]


def test_aux_target_uses_delta_progress_when_configured():
    batch = batch_with_progress([1.0, 4.0, 9.5])

    target = DrivingLeWM.aux_target(batch, "delta")

    assert target.shape == (1, 3, 8)
    assert target[0, :, 1].tolist() == pytest.approx([0.0, 3.0, 5.5])


def test_progress_signal_rejects_unknown_mode():
    route_progress = torch.zeros(1, 2, 1)

    with pytest.raises(ValueError, match="Unknown progress_mode"):
        DrivingLeWM.progress_signal(route_progress, "bad-mode")
