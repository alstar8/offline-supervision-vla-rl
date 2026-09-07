"""Shared planner grasp-state helpers.

This module centralizes the hand/gripper state that must survive between
planner stages, especially when a debug flow recreates solver instances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np


PLANNER_GRASP_STATE_ATTR = "_planner_grasp_state"

_UNSET = object()


def _normalize_optional_qpos(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1).copy()


def _normalize_optional_flag_rows(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=bool).reshape(-1).copy()


@dataclass(frozen=True)
class GraspState:
    """Minimal hand/gripper state that should survive planner stage boundaries."""

    target_hand_qpos: Optional[np.ndarray] = None
    realized_hand_qpos: Optional[np.ndarray] = None
    gripper_state: Optional[Any] = None
    grasp_flag: Optional[bool] = None
    grasp_flag_rows: Optional[np.ndarray] = None
    object_id: Optional[str] = None
    source_stage: Optional[str] = None

    def with_updates(self, **changes: Any) -> "GraspState":
        normalized = dict(changes)
        if "target_hand_qpos" in normalized:
            normalized["target_hand_qpos"] = _normalize_optional_qpos(normalized["target_hand_qpos"])
        if "realized_hand_qpos" in normalized:
            normalized["realized_hand_qpos"] = _normalize_optional_qpos(normalized["realized_hand_qpos"])
        if "grasp_flag_rows" in normalized:
            normalized["grasp_flag_rows"] = _normalize_optional_flag_rows(normalized["grasp_flag_rows"])
        return replace(self, **normalized)


def infer_gripper_state_from_hand_target(
    target_hand_qpos: Any,
    *,
    open_hand_qpos: Any,
    closed_hand_qpos: Any,
    open_state: Any,
    closed_state: Any,
) -> Any:
    """Infer the discrete gripper state from the nearest hand target reference."""

    target = _normalize_optional_qpos(target_hand_qpos)
    if target is None:
        return open_state
    open_q = _normalize_optional_qpos(open_hand_qpos)
    closed_q = _normalize_optional_qpos(closed_hand_qpos)
    open_dist = float(np.linalg.norm(target - open_q))
    closed_dist = float(np.linalg.norm(target - closed_q))
    return closed_state if closed_dist < open_dist else open_state


def get_planner_grasp_state(env_unwrapped: Any) -> Optional[GraspState]:
    state = getattr(env_unwrapped, PLANNER_GRASP_STATE_ATTR, None)
    if isinstance(state, GraspState):
        return state
    return None


def save_planner_grasp_state(
    env_unwrapped: Any,
    *,
    target_hand_qpos: Any = _UNSET,
    realized_hand_qpos: Any = _UNSET,
    gripper_state: Any = _UNSET,
    grasp_flag: Any = _UNSET,
    grasp_flag_rows: Any = _UNSET,
    object_id: Any = _UNSET,
    source_stage: Any = _UNSET,
) -> GraspState:
    """Persist planner grasp-state while preserving unspecified fields."""

    state = get_planner_grasp_state(env_unwrapped) or GraspState()
    updates = {}

    if target_hand_qpos is not _UNSET:
        updates["target_hand_qpos"] = target_hand_qpos
    if realized_hand_qpos is not _UNSET:
        updates["realized_hand_qpos"] = realized_hand_qpos
    if gripper_state is not _UNSET:
        updates["gripper_state"] = gripper_state
    if grasp_flag is not _UNSET:
        updates["grasp_flag"] = None if grasp_flag is None else bool(grasp_flag)
    if grasp_flag_rows is not _UNSET:
        updates["grasp_flag_rows"] = grasp_flag_rows
    if object_id is not _UNSET:
        updates["object_id"] = object_id
    if source_stage is not _UNSET:
        updates["source_stage"] = source_stage

    state = state.with_updates(**updates)
    setattr(env_unwrapped, PLANNER_GRASP_STATE_ATTR, state)
    return state


def restore_planner_grasp_state(
    env_unwrapped: Any,
    *,
    default_target_hand_qpos: Any = None,
    default_gripper_state: Any = None,
    source_stage: Optional[str] = None,
) -> GraspState:
    """Return a grasp-state object, seeding it from defaults if needed."""

    state = get_planner_grasp_state(env_unwrapped)
    if state is None:
        state = GraspState(
            target_hand_qpos=_normalize_optional_qpos(default_target_hand_qpos),
            gripper_state=default_gripper_state,
            source_stage=source_stage,
        )
        setattr(env_unwrapped, PLANNER_GRASP_STATE_ATTR, state)
        return state

    updates = {}
    if state.target_hand_qpos is None and default_target_hand_qpos is not None:
        updates["target_hand_qpos"] = default_target_hand_qpos
    if state.gripper_state is None and default_gripper_state is not None:
        updates["gripper_state"] = default_gripper_state
    if state.source_stage is None and source_stage is not None:
        updates["source_stage"] = source_stage
    if updates:
        state = save_planner_grasp_state(env_unwrapped, **updates)
    return state
