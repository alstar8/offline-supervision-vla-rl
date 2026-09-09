"""Reachable, non-overlapping XY spawn sampling for OpenReal2Sim RL."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Conservative RC5 workspace around the current AIRI-table cube layout.
DEFAULT_REACHABLE_BOUNDS_MIN_XY = np.array([-0.38, -0.90], dtype=np.float64)
DEFAULT_REACHABLE_BOUNDS_MAX_XY = np.array([0.05, -0.52], dtype=np.float64)
DEFAULT_ROBOT_BASE_XY = np.array([0.45, -0.75], dtype=np.float64)
DEFAULT_MIN_ROBOT_CLEARANCE = 0.28
DEFAULT_PAIR_GAP = 0.015
DEFAULT_MAX_ATTEMPTS = 80


def xy_half_extent_from_bbox(bbox) -> float:
    if bbox is None:
        return 0.03
    bounds_min, bounds_max = bbox
    size = np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64)
    return float(0.5 * max(float(size[0]), float(size[1]), 0.02))


def _xy_in_bounds(xy: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> bool:
    return bool(np.all(xy >= bounds_min) and np.all(xy <= bounds_max))


def sample_nonoverlapping_xy(
    rng: np.random.RandomState,
    *,
    radius: float,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    occupied: Iterable[tuple[np.ndarray, float]] = (),
    robot_base_xy: Sequence[float] = DEFAULT_ROBOT_BASE_XY,
    min_robot_clearance: float = DEFAULT_MIN_ROBOT_CLEARANCE,
    pair_gap: float = DEFAULT_PAIR_GAP,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> np.ndarray | None:
    lo = np.asarray(bounds_min, dtype=np.float64)[:2]
    hi = np.asarray(bounds_max, dtype=np.float64)[:2]
    base = np.asarray(robot_base_xy, dtype=np.float64)[:2]
    occupied_list = [(np.asarray(center, dtype=np.float64)[:2], float(ext)) for center, ext in occupied]

    for _ in range(int(max_attempts)):
        xy = rng.uniform(lo, hi)
        if not _xy_in_bounds(xy, lo, hi):
            continue
        if float(np.linalg.norm(xy - base)) < float(min_robot_clearance) + float(radius):
            continue
        if any(
            float(np.linalg.norm(xy - center)) < radius + other_radius + float(pair_gap)
            for center, other_radius in occupied_list
        ):
            continue
        return xy.astype(np.float64)
    return None


def instruction_for_manip_object(
    *,
    task_description: str | None,
    manip_object_id: str | None,
    object_placements: dict | None,
    scene_task_desc: str | None,
) -> str:
    if task_description:
        return str(task_description)
    if str(manip_object_id or "") == "orange_cube_ext":
        return "Pick red cube"
    placements = object_placements or {}
    if manip_object_id:
        obj_cfg = placements.get(str(manip_object_id), {}) or {}
        semantic = obj_cfg.get("task_semantic_name") or obj_cfg.get("name")
        if semantic:
            return f"Pick up the {semantic}."
    if scene_task_desc:
        return str(scene_task_desc)
    return "Pick red cube"
