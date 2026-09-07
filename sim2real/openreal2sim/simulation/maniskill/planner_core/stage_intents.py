"""Manipulator-agnostic stage intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .pose_semantics import SemanticPose


@dataclass(frozen=True)
class StageIntent:
    """Robot-independent description of a planner stage."""

    stage_name: str
    target_pose: Optional[SemanticPose] = None
    motion_method: Optional[str] = None
    gripper_intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
