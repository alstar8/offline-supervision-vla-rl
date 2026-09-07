"""Semantic provenance for planner poses.

This module provides a lightweight contract that distinguishes between:

- task-space poses produced by stage builders;
- runtime EE poses observed from the environment;
- planner-adapted poses after robot-specific transforms.

The main goal is to prevent accidentally reusing a runtime EE pose as a new
task pose without going through robot-specific normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Optional


TASK_WORLD = "task_world"
RUNTIME_EE_WORLD = "runtime_ee_world"
PLANNER_TARGET_WORLD = "planner_target_world"

PoseSemantics = Literal[
    "task_world",
    "runtime_ee_world",
    "planner_target_world",
]


class RuntimePoseReuseError(ValueError):
    """Raised when a runtime EE pose is reused as a task pose without normalization."""


@dataclass(frozen=True)
class SemanticPose:
    """World-frame pose with explicit semantic provenance."""

    pose_world: Any
    semantics: PoseSemantics
    source_stage: Optional[str] = None
    ee_link_name: Optional[str] = None
    move_group: Optional[str] = None
    already_robot_adapted: bool = False

    def with_updates(self, **changes: Any) -> "SemanticPose":
        return replace(self, **changes)

    @property
    def is_runtime_pose(self) -> bool:
        return self.semantics == RUNTIME_EE_WORLD

    @property
    def is_task_pose(self) -> bool:
        return self.semantics == TASK_WORLD


def ensure_not_runtime_pose_for_new_task(
    pose: SemanticPose,
    next_stage: str,
) -> SemanticPose:
    """Guard against silently using a runtime EE pose as a new task-stage pose."""

    if pose.semantics == RUNTIME_EE_WORLD:
        raise RuntimePoseReuseError(
            "Cannot build stage '{stage}' directly from a runtime EE pose. "
            "Normalize it through the robot adapter first.".format(stage=next_stage)
        )
    return pose
