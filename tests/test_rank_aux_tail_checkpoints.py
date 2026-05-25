import pytest

from scripts.rank_aux_tail_checkpoints import checkpoint_row


def result_with_metrics(
    *,
    overall=0.85,
    control_tail=0.84,
    lane_tail=0.83,
    lane_corr=0.62,
    heading_corr=0.50,
    corrective=0.02,
):
    return {
        "checkpoint": "epoch001.pt",
        "samples": 128,
        "lane_control": {"sign_accuracy_abs_target_gt_0p01": overall},
        "per_component": {
            "lane_offset": {"corr": lane_corr, "mae": 0.2},
            "heading_error": {"corr": heading_corr},
        },
        "tail_summaries": {
            "abs_target_control_gt_0p05": {"control_sign_accuracy_abs_target_gt_0p02": control_tail},
            "abs_lane_offset_gt_0p5": {
                "control_sign_accuracy_abs_target_gt_0p02": lane_tail,
                "mean_pred_corrective_against_lane": corrective,
            },
        },
    }


def row(result):
    return checkpoint_row(
        result,
        min_overall_sign=0.80,
        min_control_tail_sign=0.80,
        min_lane_tail_sign=0.80,
        min_lane_corr=0.50,
    )


def test_checkpoint_row_passes_when_tail_metrics_clear_gate():
    ranked = row(result_with_metrics())

    assert ranked["gate_pass"] is True
    assert ranked["score"] == pytest.approx(0.7935)


def test_checkpoint_row_holds_when_lane_tail_sign_is_weak():
    ranked = row(result_with_metrics(lane_tail=0.61))

    assert ranked["gate_pass"] is False
    assert ranked["lane_tail_sign_abs_lane_gt_0p5"] == pytest.approx(0.61)


def test_checkpoint_row_holds_when_corrective_direction_is_wrong():
    ranked = row(result_with_metrics(corrective=-0.01))

    assert ranked["gate_pass"] is False
    assert ranked["mean_pred_corrective_against_lane_abs_lane_gt_0p5"] == pytest.approx(-0.01)
