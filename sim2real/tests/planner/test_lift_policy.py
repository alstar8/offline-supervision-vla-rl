from dataclasses import dataclass

from openreal2sim.simulation.maniskill.planner_core.lift_policy import (
    build_vertical_lift_pose,
    select_lift_reference_pose,
)
from openreal2sim.simulation.maniskill.planner_core.pose_semantics import (
    RUNTIME_EE_WORLD,
    TASK_WORLD,
    SemanticPose,
)


@dataclass(frozen=True)
class FakePose:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


def test_select_lift_reference_pose_keeps_task_pose_for_widowx():
    task_pose = SemanticPose(
        pose_world=FakePose((0.05, -1.22, 0.166), (0.0, 0.6605, 0.7508, 0.0)),
        semantics=TASK_WORLD,
        source_stage="grasp",
    )
    runtime_pose = SemanticPose(
        pose_world=FakePose((0.048, -1.223, 0.174), (0.5341, 0.4693, 0.5263, -0.4664)),
        semantics=RUNTIME_EE_WORLD,
        source_stage="close",
        ee_link_name="ee_gripper_link",
        move_group="ee_gripper_link",
        already_robot_adapted=True,
    )

    selected = select_lift_reference_pose(
        "widowx250s_bridgedataset_flat_table_openr2s",
        task_pose=task_pose,
        runtime_pose=runtime_pose,
    )

    assert selected is task_pose


def test_select_lift_reference_pose_uses_runtime_pose_for_rc5():
    task_pose = SemanticPose(
        pose_world=FakePose((-0.247, -0.425, 0.3833), (-0.6785, 0.5161, -0.2630, 0.4517)),
        semantics=TASK_WORLD,
        source_stage="grasp",
    )
    runtime_pose = SemanticPose(
        pose_world=FakePose((-0.2476, -0.4248, 0.3878), (0.6785, -0.5163, 0.2640, -0.4510)),
        semantics=RUNTIME_EE_WORLD,
        source_stage="close",
        ee_link_name="prehand",
        move_group="prehand",
        already_robot_adapted=True,
    )

    selected = select_lift_reference_pose(
        "rc5_aero_hand_openr2s",
        task_pose=task_pose,
        runtime_pose=runtime_pose,
    )

    assert selected is runtime_pose


def test_build_vertical_lift_pose_preserves_orientation_and_shifts_z():
    reference_pose = SemanticPose(
        pose_world=FakePose((0.05, -1.22, 0.166), (0.0, 0.6605, 0.7508, 0.0)),
        semantics=TASK_WORLD,
        source_stage="grasp",
    )

    lifted = build_vertical_lift_pose(reference_pose, lift_delta_z=0.05)

    assert lifted.pose_world.p == (0.05, -1.22, 0.21600000000000003)
    assert lifted.pose_world.q == (0.0, 0.6605, 0.7508, 0.0)
    assert lifted.source_stage == "lift"
