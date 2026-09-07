from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

TASK_PICK_UP = "pick_up"
TASK_PICK_AND_PLACE = "pick_and_place"
SUPPORTED_TASK_TYPES = (TASK_PICK_UP, TASK_PICK_AND_PLACE)
CURRENTLY_EXECUTABLE_TASK_TYPES = {TASK_PICK_UP}


@dataclass(frozen=True)
class EpisodeIntent:
    task_type: str
    object_id: str | None = None
    destination_id: str | None = None
    destination_pose: Sequence[float] | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    intent: EpisodeIntent
    stages: List[StageSpec]


def resolve_episode_intent(
    *,
    task_type: str,
    object_id: str | None = None,
    destination_id: str | None = None,
    destination_pose: Sequence[float] | None = None,
    prompt: str | None = None,
) -> EpisodeIntent:
    if task_type not in SUPPORTED_TASK_TYPES:
        valid = ", ".join(SUPPORTED_TASK_TYPES)
        raise ValueError(f"Unsupported task_type '{task_type}'. Expected one of: {valid}")

    if task_type == TASK_PICK_AND_PLACE and destination_id is None and destination_pose is None:
        raise ValueError(
            "pick_and_place requires either --task_destination_id or a future destination pose binding."
        )

    return EpisodeIntent(
        task_type=task_type,
        object_id=object_id,
        destination_id=destination_id,
        destination_pose=destination_pose,
        prompt=prompt,
    )


def build_task_plan(intent: EpisodeIntent) -> TaskPlan:
    if intent.task_type == TASK_PICK_UP:
        stages = [
            StageSpec(name="pregrasp", kind="move_to_pregrasp"),
            StageSpec(name="descend", kind="move_to_descend"),
            StageSpec(name="close", kind="close_gripper"),
            StageSpec(name="lift", kind="lift_object"),
            StageSpec(name="retention_check", kind="retention_check"),
        ]
        return TaskPlan(intent=intent, stages=stages)

    if intent.task_type == TASK_PICK_AND_PLACE:
        stages = [
            StageSpec(name="pregrasp", kind="move_to_pregrasp"),
            StageSpec(name="descend", kind="move_to_descend"),
            StageSpec(name="close", kind="close_gripper"),
            StageSpec(name="lift", kind="lift_object"),
            StageSpec(name="move_to_place_prepose", kind="move_to_place_prepose"),
            StageSpec(name="place_descend", kind="move_to_place_descend"),
            StageSpec(name="open", kind="open_gripper"),
            StageSpec(name="retreat", kind="retreat"),
        ]
        return TaskPlan(intent=intent, stages=stages)

    valid = ", ".join(SUPPORTED_TASK_TYPES)
    raise ValueError(f"Unsupported task_type '{intent.task_type}'. Expected one of: {valid}")


def validate_task_supported_for_runtime(intent: EpisodeIntent) -> None:
    if intent.task_type not in CURRENTLY_EXECUTABLE_TASK_TYPES:
        valid = ", ".join(sorted(CURRENTLY_EXECUTABLE_TASK_TYPES))
        raise ValueError(
            f"Task '{intent.task_type}' is modeled in the unified task layer but is not yet executable. "
            f"Currently executable task types: {valid}"
        )


def summarize_task_plan(task_plan: TaskPlan) -> str:
    stage_names = ",".join(stage.name for stage in task_plan.stages)
    return (
        f"task_type={task_plan.intent.task_type} "
        f"object_id={task_plan.intent.object_id or '<auto>'} "
        f"destination_id={task_plan.intent.destination_id or '<none>'} "
        f"stages={stage_names}"
    )
