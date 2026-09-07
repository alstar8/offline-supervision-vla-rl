from __future__ import annotations

from dataclasses import dataclass


RC5_PICK_RETENTION_MAX_OBJECT_TCP_DIST_M = 0.12


@dataclass(frozen=True)
class PickLiftSuccessDecision:
    success: bool
    reason: str


def evaluate_rc5_pick_lift_success(
    *,
    grasp_flag_after_lift: bool,
    lift_dz: float,
    lift_success_threshold: float,
    grasp_flag_after_close: bool | None,
    object_tcp_dist_after_lift: float | None,
    max_object_tcp_dist_m: float = RC5_PICK_RETENTION_MAX_OBJECT_TCP_DIST_M,
) -> PickLiftSuccessDecision:
    if bool(grasp_flag_after_lift) and float(lift_dz) >= float(lift_success_threshold):
        return PickLiftSuccessDecision(
            success=True,
            reason="grasp_after_lift_and_lift_height",
        )

    if float(lift_dz) < float(lift_success_threshold):
        return PickLiftSuccessDecision(
            success=False,
            reason="insufficient_lift_height",
        )

    if grasp_flag_after_close is not True:
        return PickLiftSuccessDecision(
            success=False,
            reason="no_close_stage_grasp",
        )

    if object_tcp_dist_after_lift is None:
        return PickLiftSuccessDecision(
            success=False,
            reason="missing_object_tcp_distance",
        )

    if float(object_tcp_dist_after_lift) > float(max_object_tcp_dist_m):
        return PickLiftSuccessDecision(
            success=False,
            reason="object_too_far_from_tcp_after_lift",
        )

    return PickLiftSuccessDecision(
        success=True,
        reason="close_grasp_plus_lift_with_object_near_tcp",
    )
