"""Validation helpers for planner stage contracts."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pose_semantics import SemanticPose
from .stage_result import StageResult


def compute_joint_tracking_linf(
    joint_target: Optional[np.ndarray],
    joint_realized: Optional[np.ndarray],
) -> Optional[float]:
    """Return max absolute joint-tracking error, or None when unavailable."""

    if joint_target is None or joint_realized is None:
        return None
    target = np.asarray(joint_target, dtype=np.float32).reshape(-1)
    realized = np.asarray(joint_realized, dtype=np.float32).reshape(-1)
    if target.shape != realized.shape:
        raise ValueError(
            "joint_target and joint_realized must have the same flattened shape: "
            "{target_shape} vs {realized_shape}".format(
                target_shape=target.shape,
                realized_shape=realized.shape,
            )
        )
    return float(np.max(np.abs(target - realized))) if target.size > 0 else 0.0


def validate_stage_result(
    stage_name: str,
    target_pose: Optional[SemanticPose] = None,
    achieved_pose: Optional[SemanticPose] = None,
    pos_err_m: Optional[float] = None,
    rot_err_deg: Optional[float] = None,
    joint_target: Optional[np.ndarray] = None,
    joint_realized: Optional[np.ndarray] = None,
    planner_status: str = "unknown",
    execution_status: str = "unknown",
    max_pos_err_m: float = 0.02,
    max_rot_err_deg: Optional[float] = None,
    max_joint_tracking_linf: Optional[float] = None,
) -> StageResult:
    """Build a validated StageResult from common planner metrics."""

    joint_tracking_linf = compute_joint_tracking_linf(joint_target, joint_realized)

    success = True
    notes = []

    if pos_err_m is not None and pos_err_m > float(max_pos_err_m):
        success = False
        notes.append(
            "pos_err_m={pos_err:.4f} exceeds threshold {threshold:.4f}".format(
                pos_err=float(pos_err_m),
                threshold=float(max_pos_err_m),
            )
        )

    if (
        rot_err_deg is not None
        and max_rot_err_deg is not None
        and rot_err_deg > float(max_rot_err_deg)
    ):
        success = False
        notes.append(
            "rot_err_deg={rot_err:.2f} exceeds threshold {threshold:.2f}".format(
                rot_err=float(rot_err_deg),
                threshold=float(max_rot_err_deg),
            )
        )

    if (
        joint_tracking_linf is not None
        and max_joint_tracking_linf is not None
        and joint_tracking_linf > float(max_joint_tracking_linf)
    ):
        success = False
        notes.append(
            "joint_tracking_linf={err:.4f} exceeds threshold {threshold:.4f}".format(
                err=float(joint_tracking_linf),
                threshold=float(max_joint_tracking_linf),
            )
        )

    return StageResult(
        stage_name=stage_name,
        success=success,
        planner_status=planner_status,
        execution_status=execution_status,
        target_pose=target_pose,
        achieved_pose=achieved_pose,
        pos_err_m=pos_err_m,
        rot_err_deg=rot_err_deg,
        joint_target=joint_target,
        joint_realized=joint_realized,
        joint_tracking_linf=joint_tracking_linf,
        notes=notes,
    )
