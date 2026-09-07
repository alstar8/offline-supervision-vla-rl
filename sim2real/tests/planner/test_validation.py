import numpy as np
import pytest

from openreal2sim.simulation.maniskill.planner_core.pose_semantics import (
    TASK_WORLD,
    SemanticPose,
)
from openreal2sim.simulation.maniskill.planner_core.validation import (
    compute_joint_tracking_linf,
    validate_stage_result,
)


def test_compute_joint_tracking_linf_returns_max_abs_error():
    joint_target = np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
    joint_realized = np.asarray([0.1, 0.8, -1.05], dtype=np.float32)

    result = compute_joint_tracking_linf(joint_target, joint_realized)

    assert result == pytest.approx(0.2)


def test_validate_stage_result_rejects_large_pose_error():
    target_pose = SemanticPose(pose_world=object(), semantics=TASK_WORLD, source_stage="lift")
    achieved_pose = SemanticPose(pose_world=object(), semantics=TASK_WORLD, source_stage="lift")

    result = validate_stage_result(
        stage_name="lift",
        target_pose=target_pose,
        achieved_pose=achieved_pose,
        pos_err_m=0.08,
        max_pos_err_m=0.02,
        planner_status="ok",
        execution_status="ok",
    )

    assert result.success is False
    assert any("pos_err_m" in note for note in result.notes)


def test_validate_stage_result_accepts_small_pose_and_tracking_error():
    target_pose = SemanticPose(pose_world=object(), semantics=TASK_WORLD, source_stage="reach")
    achieved_pose = SemanticPose(pose_world=object(), semantics=TASK_WORLD, source_stage="reach")

    result = validate_stage_result(
        stage_name="reach",
        target_pose=target_pose,
        achieved_pose=achieved_pose,
        pos_err_m=0.005,
        joint_target=np.asarray([0.1, -0.2], dtype=np.float32),
        joint_realized=np.asarray([0.1005, -0.199], dtype=np.float32),
        max_pos_err_m=0.02,
        max_joint_tracking_linf=0.01,
        planner_status="ok",
        execution_status="ok",
    )

    assert result.success is True
    assert result.joint_tracking_linf is not None
    assert result.joint_tracking_linf < 0.01
