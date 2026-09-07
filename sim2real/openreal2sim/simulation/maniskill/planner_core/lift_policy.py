"""Shared policy for selecting lift-pose provenance.

The key contract is:

- RC5 may intentionally build lift from the runtime EE pose reached after close.
- Generic two-finger robots such as WidowX and Panda should keep the task/grasp
  pose semantics unless a robot adapter explicitly normalizes runtime pose reuse.
"""

from __future__ import annotations

from typing import Any

from .pose_semantics import (
    RUNTIME_EE_WORLD,
    TASK_WORLD,
    RuntimePoseReuseError,
    SemanticPose,
    ensure_not_runtime_pose_for_new_task,
)


def is_rc5_robot(agent_uid: str) -> bool:
    uid = str(agent_uid or "").lower()
    return "rc5_aero_hand" in uid


def select_lift_reference_pose(
    agent_uid: str,
    task_pose: SemanticPose,
    runtime_pose: SemanticPose,
) -> SemanticPose:
    """Return the semantic pose that should seed the lift stage.

    For RC5 the current runtime EE pose may be the correct reference because the
    actual post-close pose can deviate meaningfully from the planned grasp pose.
    For other robots we preserve the pre-existing generic contract and keep lift
    anchored to the task/grasp pose unless a later adapter explicitly says
    otherwise.
    """

    if runtime_pose.semantics != RUNTIME_EE_WORLD:
        raise RuntimePoseReuseError(
            "runtime_pose must have semantics='runtime_ee_world', got {semantics!r}".format(
                semantics=runtime_pose.semantics
            )
        )
    if task_pose.semantics != TASK_WORLD:
        raise RuntimePoseReuseError(
            "task_pose must have semantics='task_world', got {semantics!r}".format(
                semantics=task_pose.semantics
            )
        )

    if is_rc5_robot(agent_uid):
        return runtime_pose

    return ensure_not_runtime_pose_for_new_task(task_pose, next_stage="lift")


def build_vertical_lift_pose(
    reference_pose: SemanticPose,
    lift_delta_z: float,
) -> SemanticPose:
    """Build a simple vertical-lift pose from a semantic reference pose.

    The helper assumes `pose_world` exposes `.p` and `.q` attributes compatible
    with ManiSkill/SAPIEN-like pose objects. This keeps the policy layer light
    and testable without importing the full simulator runtime.
    """

    pose_world = reference_pose.pose_world
    current_p = list(pose_world.p)
    current_p[2] = float(current_p[2]) + float(lift_delta_z)
    pose_type = type(pose_world)
    position_type = type(pose_world.p)
    try:
        lifted_p = position_type(current_p)
    except TypeError:
        lifted_p = current_p
    lifted_pose = pose_type(p=lifted_p, q=pose_world.q)
    return reference_pose.with_updates(
        pose_world=lifted_pose,
        source_stage="lift",
    )
