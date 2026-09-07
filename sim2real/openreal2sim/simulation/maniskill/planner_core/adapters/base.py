"""Base contracts for robot-specific planner adapters."""

from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np

from ..pose_semantics import SemanticPose


class RobotAdapter(Protocol):
    """Minimum contract for manipulator-specific planner semantics."""

    robot_type: str

    def get_ee_link_name(self) -> str:
        ...

    def get_move_group(self) -> str:
        ...

    def supports_local_ik(self) -> bool:
        ...

    def get_runtime_ee_pose(self, env: Any) -> SemanticPose:
        ...

    def adapt_task_pose_for_planner(self, pose: SemanticPose) -> SemanticPose:
        ...

    def normalize_runtime_pose_for_stage(self, pose: SemanticPose) -> SemanticPose:
        ...

    def build_open_command(self, env: Any) -> np.ndarray:
        ...

    def build_close_command(self, env: Any) -> np.ndarray:
        ...

    def evaluate_grasp(self, env: Any, target_object: Any) -> Dict[str, Any]:
        ...
