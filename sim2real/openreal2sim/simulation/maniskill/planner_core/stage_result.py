"""Shared stage execution result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .pose_semantics import SemanticPose


@dataclass
class StageResult:
    """Unified stage result for main/debug/debug_planner paths."""

    stage_name: str
    success: bool
    planner_status: str = "unknown"
    execution_status: str = "unknown"
    target_pose: Optional[SemanticPose] = None
    achieved_pose: Optional[SemanticPose] = None
    pos_err_m: Optional[float] = None
    rot_err_deg: Optional[float] = None
    joint_target: Optional[np.ndarray] = None
    joint_realized: Optional[np.ndarray] = None
    joint_tracking_linf: Optional[float] = None
    grasp_diag: Dict[str, Any] = field(default_factory=dict)
    controller_diag: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        self.notes.append(note)
