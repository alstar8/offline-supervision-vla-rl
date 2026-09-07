from __future__ import annotations

import pytest

import openreal2sim.simulation.maniskill.scripts.rc5_unified_tasks as uut


def test_resolve_episode_intent_supports_pick_up_without_explicit_object():
    intent = uut.resolve_episode_intent(task_type=uut.TASK_PICK_UP)

    assert intent.task_type == uut.TASK_PICK_UP
    assert intent.object_id is None
    assert intent.destination_id is None


def test_resolve_episode_intent_requires_destination_for_pick_and_place():
    with pytest.raises(ValueError, match="pick_and_place requires either --task_destination_id"):
        uut.resolve_episode_intent(
            task_type=uut.TASK_PICK_AND_PLACE,
            object_id="coke_can",
        )


def test_build_task_plan_for_pick_up_matches_current_macro_shape():
    intent = uut.resolve_episode_intent(
        task_type=uut.TASK_PICK_UP,
        object_id="orange_cube_ext",
    )

    task_plan = uut.build_task_plan(intent)

    assert [stage.name for stage in task_plan.stages] == [
        "pregrasp",
        "descend",
        "close",
        "lift",
        "retention_check",
    ]
    assert [stage.kind for stage in task_plan.stages] == [
        "move_to_pregrasp",
        "move_to_descend",
        "close_gripper",
        "lift_object",
        "retention_check",
    ]


def test_build_task_plan_for_pick_and_place_models_future_stage_sequence():
    intent = uut.resolve_episode_intent(
        task_type=uut.TASK_PICK_AND_PLACE,
        object_id="coke_can",
        destination_id="yellow_plate",
    )

    task_plan = uut.build_task_plan(intent)

    assert [stage.name for stage in task_plan.stages] == [
        "pregrasp",
        "descend",
        "close",
        "lift",
        "move_to_place_prepose",
        "place_descend",
        "open",
        "retreat",
    ]


def test_validate_task_supported_for_runtime_fails_fast_for_non_executable_task():
    intent = uut.resolve_episode_intent(
        task_type=uut.TASK_PICK_AND_PLACE,
        object_id="coke_can",
        destination_id="yellow_plate",
    )

    with pytest.raises(ValueError, match="is modeled in the unified task layer but is not yet executable"):
        uut.validate_task_supported_for_runtime(intent)


def test_summarize_task_plan_is_human_readable():
    intent = uut.resolve_episode_intent(
        task_type=uut.TASK_PICK_UP,
        object_id="orange_cube_ext",
    )
    task_plan = uut.build_task_plan(intent)

    summary = uut.summarize_task_plan(task_plan)

    assert "task_type=pick_up" in summary
    assert "object_id=orange_cube_ext" in summary
    assert "stages=pregrasp,descend,close,lift,retention_check" in summary
