import pytest

from openreal2sim.simulation.maniskill.planner_core.pose_semantics import (
    RUNTIME_EE_WORLD,
    TASK_WORLD,
    RuntimePoseReuseError,
    SemanticPose,
    ensure_not_runtime_pose_for_new_task,
)


def test_runtime_pose_cannot_be_reused_as_task_pose_without_normalization():
    runtime_pose = SemanticPose(
        pose_world=object(),
        semantics=RUNTIME_EE_WORLD,
        source_stage="close",
        ee_link_name="ee_gripper_link",
        move_group="ee_gripper_link",
        already_robot_adapted=True,
    )

    with pytest.raises(RuntimePoseReuseError):
        ensure_not_runtime_pose_for_new_task(runtime_pose, next_stage="lift")


def test_task_pose_passes_runtime_reuse_guard():
    task_pose = SemanticPose(
        pose_world=object(),
        semantics=TASK_WORLD,
        source_stage="grasp",
    )

    result = ensure_not_runtime_pose_for_new_task(task_pose, next_stage="lift")

    assert result is task_pose
