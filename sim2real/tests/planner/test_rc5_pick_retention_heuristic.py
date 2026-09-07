from openreal2sim.simulation.maniskill.rc5_pick_retention_heuristic import (
    RC5_PICK_RETENTION_MAX_OBJECT_TCP_DIST_M,
    evaluate_rc5_pick_lift_success,
)


def test_pick_lift_success_prefers_direct_grasp_after_lift():
    decision = evaluate_rc5_pick_lift_success(
        grasp_flag_after_lift=True,
        lift_dz=0.05,
        lift_success_threshold=0.025,
        grasp_flag_after_close=False,
        object_tcp_dist_after_lift=None,
    )

    assert decision.success is True
    assert decision.reason == "grasp_after_lift_and_lift_height"


def test_pick_lift_success_accepts_retention_when_object_stays_near_tcp():
    decision = evaluate_rc5_pick_lift_success(
        grasp_flag_after_lift=False,
        lift_dz=0.0832,
        lift_success_threshold=0.025,
        grasp_flag_after_close=True,
        object_tcp_dist_after_lift=0.094,
    )

    assert decision.success is True
    assert decision.reason == "close_grasp_plus_lift_with_object_near_tcp"


def test_pick_lift_success_rejects_far_object_even_if_lifted():
    decision = evaluate_rc5_pick_lift_success(
        grasp_flag_after_lift=False,
        lift_dz=0.0832,
        lift_success_threshold=0.025,
        grasp_flag_after_close=True,
        object_tcp_dist_after_lift=RC5_PICK_RETENTION_MAX_OBJECT_TCP_DIST_M + 0.01,
    )

    assert decision.success is False
    assert decision.reason == "object_too_far_from_tcp_after_lift"
